import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from active_adaptation.assets import get_robot_cfg, get_tennis_ball_cfg, get_tennis_court_cfg
from active_adaptation.utils.math import quat_apply


def _read_scalar(npz, key, default):
    if key not in npz:
        return default
    value = npz[key]
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    return value


def _read_str(npz, key, default):
    if key not in npz:
        return default
    value = npz[key]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return str(value.reshape(-1)[0])
    return str(value)


def _find_latest_train_record(project_root: Path) -> Path:
    outputs_dir = project_root / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs directory not found: {outputs_dir}")

    candidates: list[Path] = []
    for pattern in ("*/*/train_records/*.npz", "*/train_records/*.npz", "train_records/*.npz"):
        candidates.extend(outputs_dir.glob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No train record npz found under: {outputs_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_scene(
    *,
    device,
    robot_name,
    physics_dt,
    env_spacing,
):
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.sim.sim import Simulation

    scene_cfg = SceneCfg(num_envs=1, env_spacing=env_spacing)
    scene_cfg.entities["robot"] = get_robot_cfg(robot_name)
    scene_cfg.entities["tennis_court"] = get_tennis_court_cfg()
    scene_cfg.entities["tennis_ball"] = get_tennis_ball_cfg()

    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(
            nconmax=50,
            njmax=500,
            mujoco=MujocoCfg(
                timestep=physics_dt,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        model=scene.compile(),
        device=device,
    )
    scene.initialize(
        mj_model=sim.mj_model,
        model=sim.model,
        data=sim.data,
    )
    if not hasattr(scene, "env_origins") and hasattr(scene, "env_offsets"):
        scene.env_origins = scene.env_offsets
    _align_tennis_court_to_env_origins(scene, device=device)
    return scene, sim


def _align_tennis_court_to_env_origins(scene, *, device: str) -> None:
    _align_tennis_court_to_env_origins_with_origins(scene, device=device, env_origins=None)


def _align_tennis_court_to_env_origins_with_origins(scene, *, device: str, env_origins: torch.Tensor | None) -> None:
    if "tennis_court" not in scene.entities:
        return
    court = scene["tennis_court"]
    if not getattr(court, "is_mocap", False):
        return
    if env_origins is None:
        if not hasattr(scene, "env_origins"):
            return
        env_origins = scene.env_origins
    env_ids = torch.arange(env_origins.shape[0], device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    pose = torch.zeros((env_ids.numel(), 7), device=device, dtype=torch.float32)
    pose[:, :3] = env_origins[env_ids]
    pose[:, 3] = 1.0
    court.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def _env_origins_from_env_ids(
    *,
    env_ids: np.ndarray,
    env_spacing: float,
    device: str,
    train_num_envs: int | None = None,
) -> torch.Tensor:
    if env_ids.ndim != 1 or env_ids.size == 0:
        raise ValueError(f"Invalid env_ids shape: {env_ids.shape}.")
    env_ids_i64 = env_ids.astype(np.int64, copy=False)
    if np.any(env_ids_i64 < 0):
        raise ValueError("env_ids must be non-negative.")

    # TrainStateRecorder stores sampled global env_ids. If train_num_envs is
    # available, use it to recover the exact grid side used in training.
    if train_num_envs is not None and int(train_num_envs) > 0:
        side = int(np.ceil(np.sqrt(float(int(train_num_envs)))))
    else:
        side = int(np.ceil(np.sqrt(float(int(env_ids_i64.max()) + 1))))
    row = env_ids_i64 // side
    col = env_ids_i64 % side
    origins = torch.zeros((env_ids_i64.shape[0], 3), device=device, dtype=torch.float32)
    origins[:, 0] = torch.as_tensor(
        (col.astype(np.float32) - (side - 1) * 0.5) * float(env_spacing),
        device=device,
        dtype=torch.float32,
    )
    origins[:, 1] = torch.as_tensor(
        (row.astype(np.float32) - (side - 1) * 0.5) * float(env_spacing),
        device=device,
        dtype=torch.float32,
    )
    return origins


def _extract_first_root_xy(*, qpos: np.ndarray, root_state: np.ndarray | None) -> np.ndarray:
    if root_state is not None and root_state.ndim == 3 and root_state.shape[0] > 0 and root_state.shape[-1] >= 2:
        return root_state[0, :, :2].astype(np.float32, copy=False)
    if qpos.ndim == 3 and qpos.shape[0] > 0 and qpos.shape[-1] >= 2:
        return qpos[0, :, :2].astype(np.float32, copy=False)
    raise ValueError("Cannot extract first-frame root XY from record.")


def _create_viewer(sim):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    viewer = viser.ViserServer(label="gmt-train-record")
    viser_scene = ViserMujocoScene.create(
        server=viewer,
        mj_model=sim.mj_model,
        num_envs=1,
    )
    viser_scene.create_visualization_gui()
    viser_scene.debug_visualization_enabled = False
    viser_scene.camera_tracking_enabled = True
    return viewer, viser_scene


def _hide_court_overlay_geoms(sim) -> None:
    """Hide tennis-court collision overlay geoms in viewer.

    These geoms are needed for physics but can visually occlude court lines
    in replay when rendered as regular meshes.
    """
    import mujoco

    model = sim.mj_model
    hidden = 0
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        if not name.startswith("tennis_court/"):
            continue
        # Hide all court-side helper collision layers, keep visual court/net geoms.
        if name.endswith("_collision"):
            model.geom_group[geom_id] = 3  # default hidden in Viser (groups >= 3)
            hidden += 1
    if hidden > 0:
        print(f"[INFO] Hidden {hidden} tennis-court collision overlay geoms for clearer replay visualization.")

def _to_cpu_wp_data(wp_data):
    """Build a lightweight CPU view for Viser when sim tensors are on CUDA."""
    try:
        device = getattr(wp_data.xpos, "device", None)
        if device is None or device.type == "cpu":
            return wp_data
        return SimpleNamespace(
            xpos=wp_data.xpos.detach().cpu(),
            xmat=wp_data.xmat.detach().cpu(),
            mocap_pos=wp_data.mocap_pos.detach().cpu(),
            mocap_quat=wp_data.mocap_quat.detach().cpu(),
            qpos=wp_data.qpos.detach().cpu(),
            qvel=wp_data.qvel.detach().cpu(),
        )
    except Exception:
        return wp_data


def _compute_racket_center_w(
    *,
    robot,
    racket_body_id: int,
    racket_center_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute racket body origin and reward-consistent racket center in world frame."""
    body_pos_w = robot.data.body_link_pos_w[:, racket_body_id]
    body_quat_w = robot.data.body_link_quat_w[:, racket_body_id]
    offset_local = racket_center_offset.unsqueeze(0).expand(body_pos_w.shape[0], -1)
    center_pos_w = body_pos_w + quat_apply(body_quat_w, offset_local)
    return body_pos_w, center_pos_w


def _axis_vector(axis_name: str, *, device: str) -> torch.Tensor:
    key = axis_name.lower()
    if key == "x":
        vec = (1.0, 0.0, 0.0)
    elif key == "y":
        vec = (0.0, 1.0, 0.0)
    elif key == "z":
        vec = (0.0, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported axis '{axis_name}', expected one of: x, y, z.")
    return torch.tensor(vec, device=device, dtype=torch.float32)


def _infer_racket_face_normal_local_from_collision_geom(*, sim, device: str) -> tuple[torch.Tensor, str]:
    """Infer racket face normal in racket-body local frame from thin axis of collision geom."""
    fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    try:
        import mujoco

        model = sim.mj_model
        geom_id = -1
        geom_name = ""
        for name in ("robot/tennis_racket_collision", "tennis_racket_collision"):
            try:
                gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
            except Exception:
                gid = -1
            if gid >= 0:
                geom_id = gid
                geom_name = name
                break
        if geom_id < 0:
            return fallback, "fallback:z (collision geom not found)"

        geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float32)[:3]
        thin_axis = int(np.argmin(np.abs(geom_size)))
        axis_geom = np.zeros((3,), dtype=np.float32)
        axis_geom[thin_axis] = 1.0
        geom_quat = np.asarray(model.geom_quat[geom_id], dtype=np.float32)[:4]

        q = torch.tensor(geom_quat, device=device, dtype=torch.float32).unsqueeze(0)
        v = torch.tensor(axis_geom, device=device, dtype=torch.float32).unsqueeze(0)
        axis_local = quat_apply(q, v).squeeze(0)
        axis_local = axis_local / axis_local.norm().clamp_min(1.0e-6)
        desc = (
            f"auto:{geom_name}, thin_axis={thin_axis}, size=({geom_size[0]:.4f},{geom_size[1]:.4f},{geom_size[2]:.4f})"
        )
        return axis_local, desc
    except Exception:
        return fallback, "fallback:z (auto infer failed)"


def _compute_racket_face_dirs_w(
    *,
    robot,
    racket_body_id: int,
    racket_center_offset: torch.Tensor,
    face_axis_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return racket center, +face dir, -face dir in world frame."""
    body_pos_w = robot.data.body_link_pos_w[:, racket_body_id]
    body_quat_w = robot.data.body_link_quat_w[:, racket_body_id]
    offset_local = racket_center_offset.unsqueeze(0).expand(body_pos_w.shape[0], -1)
    center_pos_w = body_pos_w + quat_apply(body_quat_w, offset_local)
    face_axis = face_axis_local.unsqueeze(0).expand(body_pos_w.shape[0], -1)
    face_plus_dir_w = quat_apply(body_quat_w, face_axis)
    face_plus_dir_w = face_plus_dir_w / face_plus_dir_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    face_minus_dir_w = -face_plus_dir_w
    return center_pos_w, face_plus_dir_w, face_minus_dir_w




def _find_launch_events(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    robot_nq: int,
    robot_nv: int,
    env_idx: int,
    local_y_min: float,
    vel_y_max: float,
    z_min: float,
    teleport_pos_jump_min: float,
) -> tuple[np.ndarray, dict]:
    steps = int(qpos.shape[0])
    if steps == 0:
        return np.zeros((0,), dtype=np.int64), {}

    ball_pos = qpos[:, env_idx, robot_nq : robot_nq + 3]
    ball_vel = qvel[:, env_idx, robot_nv : robot_nv + 3]
    root_xy = qpos[:, env_idx, :2]
    local_y = ball_pos[:, 1] - root_xy[:, 1]
    vel_y = ball_vel[:, 1]
    z = ball_pos[:, 2]
    pos_jump = np.zeros((steps,), dtype=np.float32)
    if steps > 1:
        pos_jump[1:] = np.linalg.norm(ball_pos[1:] - ball_pos[:-1], axis=-1)

    launch_mask = (local_y > float(local_y_min)) & (vel_y < float(vel_y_max)) & (z > float(z_min))
    rising = launch_mask & np.concatenate((np.ones((1,), dtype=bool), ~launch_mask[:-1]))
    teleport = pos_jump >= float(teleport_pos_jump_min)

    # Prefer teleport-based launch detection (ball relaunch writes root state and causes
    # a large position jump between control steps). This avoids many false positives from
    # pure kinematic thresholding during regular rally motion.
    launch_starts = np.flatnonzero(teleport).astype(np.int64)
    if launch_mask[0]:
        launch_starts = np.unique(np.concatenate((np.asarray([0], dtype=np.int64), launch_starts))).astype(np.int64)
    if launch_starts.size == 0:
        launch_starts = np.flatnonzero(rising).astype(np.int64)

    diag = {
        "local_y": local_y,
        "vel_y": vel_y,
        "z": z,
        "launch_mask": launch_mask,
        "pos_jump": pos_jump,
        "teleport": teleport,
    }
    return launch_starts, diag


def _summarize_launch_events(
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    robot_nq: int,
    robot_nv: int,
    env_ids: np.ndarray | None,
    local_y_min: float,
    vel_y_max: float,
    z_min: float,
    teleport_pos_jump_min: float,
    verbose: bool,
) -> list[np.ndarray]:
    launch_steps_all: list[np.ndarray] = []
    num_envs = int(qpos.shape[1])
    for env_idx in range(num_envs):
        launch_steps, _ = _find_launch_events(
            qpos=qpos,
            qvel=qvel,
            robot_nq=robot_nq,
            robot_nv=robot_nv,
            env_idx=env_idx,
            local_y_min=local_y_min,
            vel_y_max=vel_y_max,
            z_min=z_min,
            teleport_pos_jump_min=teleport_pos_jump_min,
        )
        launch_steps_all.append(launch_steps)
    if verbose:
        print("[INFO] Launch summary by slot:")
        for env_idx, launch_steps in enumerate(launch_steps_all):
            env_id_str = str(int(env_ids[env_idx])) if env_ids is not None else str(env_idx)
            preview = ", ".join(str(int(v)) for v in launch_steps[:6])
            if launch_steps.size > 6:
                preview = f"{preview}, ..."
            if not preview:
                preview = "-"
            print(
                f"  slot={env_idx:2d} env_id={env_id_str:>5s} launches={int(launch_steps.size):2d} first_steps=[{preview}]"
            )
    return launch_steps_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=str, nargs="?", help="Path to train record npz file.")
    parser.add_argument(
        "--new",
        action="store_true",
        default=False,
        help=(
            "Auto-select latest train_records/*.npz under outputs/, and enable convenient defaults: "
            "--env-index 0 --list-launches --auto-select-env-with-launch --cycle-launches."
        ),
    )
    parser.add_argument("--robot", type=str, default=None, help="Override robot name.")
    parser.add_argument("--device", type=str, default=None, help="Simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--step-dt", type=float, default=None, help="Override playback frame dt.")
    parser.add_argument("--physics-dt", type=float, default=None, help="Override mujoco physics dt.")
    parser.add_argument("--env-spacing", type=float, default=None, help="Override scene env spacing.")
    parser.add_argument("--start-step", type=int, default=None, help="Inclusive start frame index.")
    parser.add_argument("--end-step", type=int, default=None, help="Exclusive end frame index.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--viewer-max-fps",
        type=float,
        default=45.0,
        help="Throttle Viser updates to at most this FPS. <=0 disables throttling.",
    )
    parser.add_argument("--env-index", type=int, default=0, help="Initial env slot index to replay.")
    parser.add_argument(
        "--from-launch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-start playback from detected launch frame (tennis mode only).",
    )
    parser.add_argument(
        "--launch-index",
        type=int,
        default=0,
        help="Which detected launch event to use in selected env (0-based).",
    )
    parser.add_argument(
        "--cycle-launches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --from-launch is enabled, automatically advance to next launch segment "
            "after current segment ends. Always wraps to first launch after the last one."
        ),
    )
    parser.add_argument(
        "--tail-after-launch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using --from-launch, keep playing to record end instead of stopping at next launch.",
    )
    parser.add_argument(
        "--list-launches",
        action="store_true",
        default=False,
        help="Print launch counts/steps for every env slot before playback.",
    )
    parser.add_argument(
        "--auto-select-env-with-launch",
        action="store_true",
        default=False,
        help=(
            "If selected --env-index has no launch events, auto-switch to the slot "
            "with the most launch events."
        ),
    )
    parser.add_argument(
        "--launch-local-y-min",
        type=float,
        default=1.0,
        help="Launch detection threshold: ball local y > this value.",
    )
    parser.add_argument(
        "--launch-vel-y-max",
        type=float,
        default=-2.0,
        help="Launch detection threshold: ball vy < this value.",
    )
    parser.add_argument(
        "--launch-z-min",
        type=float,
        default=0.8,
        help="Launch detection threshold: ball z > this value.",
    )
    parser.add_argument(
        "--launch-teleport-jump-min",
        type=float,
        default=2.5,
        help=(
            "Launch detection threshold: ||ball_pos[t]-ball_pos[t-1]|| >= this value (meters). "
            "Used to robustly detect relaunch teleports."
        ),
    )
    parser.add_argument(
        "--racket-body-name",
        type=str,
        default="tennis_racket_mount",
        help="Racket body name used by reward center computation.",
    )
    parser.add_argument(
        "--racket-center-offset",
        type=float,
        nargs=3,
        default=(0.1025, -0.004, 0.4),
        metavar=("OX", "OY", "OZ"),
        help="Local XYZ offset from racket body origin to reward racket center.",
    )
    parser.add_argument(
        "--racket-center-radius",
        type=float,
        default=0.045,
        help="Visualization sphere radius for racket center debug overlay.",
    )
    parser.add_argument(
        "--racket-face-axis",
        type=str,
        default="auto",
        choices=("auto", "x", "y", "z"),
        help="Local racket-body axis for +face debug direction; 'auto' infers from tennis_racket_collision thin axis.",
    )
    parser.add_argument(
        "--racket-face-length",
        type=float,
        default=0.22,
        help="Debug line length for racket +face/-face directions.",
    )
    args = parser.parse_args()
    argv = set(sys.argv[1:])

    if args.new:
        if args.record is None:
            project_root = Path(__file__).resolve().parents[2]
            latest_record = _find_latest_train_record(project_root)
            args.record = str(latest_record)
            print(f"[INFO] --new selected latest record: {latest_record}")
        if "--env-index" not in argv:
            args.env_index = 0
        if "--list-launches" not in argv:
            args.list_launches = True
        if "--auto-select-env-with-launch" not in argv:
            args.auto_select_env_with_launch = True
        if ("--cycle-launches" not in argv) and ("--no-cycle-launches" not in argv):
            args.cycle_launches = True

    if args.record is None:
        parser.error("record is required unless --new is specified.")

    record_path = Path(args.record).expanduser()
    if not record_path.is_absolute():
        record_path = (Path.cwd() / record_path).resolve()
    if not record_path.exists():
        raise FileNotFoundError(f"Record file not found: {record_path}")
    print(f"[INFO] Replay record: {record_path}")
    print(
        "[INFO] Replay mode writes recorded qpos/qvel directly; "
        "training-time termination/reset logic is not executed in this script."
    )

    if args.speed <= 0:
        raise ValueError("--speed must be > 0.")
    if args.cycle_launches and not args.from_launch:
        print("[WARN] --cycle-launches requires --from-launch; disabling cycle mode.")
        args.cycle_launches = False
    if args.cycle_launches and args.tail_after_launch:
        print("[WARN] --cycle-launches is incompatible with --tail-after-launch; forcing --tail-after-launch=False.")
        args.tail_after_launch = False

    with np.load(str(record_path), allow_pickle=False) as npz:
        qpos = npz["qpos"]
        qvel = npz["qvel"]
        root_state_full = npz["root_state"] if "root_state" in npz else None
        env_origins_full = npz["env_origins"].copy() if "env_origins" in npz else None
        train_num_envs = int(_read_scalar(npz, "train_num_envs", -1))
        if "env_ids" not in npz:
            raise ValueError("Record npz missing required key: env_ids.")
        env_ids_full = npz["env_ids"].copy()
        if qpos.ndim != 3 or qvel.ndim != 3:
            raise ValueError("Expected qpos/qvel with shape [steps, envs, dim].")
        if qpos.shape[:2] != qvel.shape[:2]:
            raise ValueError("qpos and qvel shape mismatch.")

        steps, num_env_slots = qpos.shape[:2]
        robot_name = args.robot or _read_str(npz, "robot_name", "g1_col_full_self")
        step_dt = args.step_dt or float(_read_scalar(npz, "step_dt", 0.02))
        physics_dt = args.physics_dt or float(_read_scalar(npz, "physics_dt", 0.0025))
        env_spacing = args.env_spacing or float(_read_scalar(npz, "env_spacing", 2.5))

    selected_env_index = int(args.env_index)
    if selected_env_index < 0 or selected_env_index >= int(num_env_slots):
        raise ValueError(f"--env-index must be in [0, {int(num_env_slots) - 1}]")
    if args.start_step is not None and (args.start_step < 0 or args.start_step >= steps):
        raise ValueError(f"--start-step must be in [0, {steps - 1}]")
    if args.end_step is not None and (args.end_step <= 0 or args.end_step > steps):
        raise ValueError(f"--end-step must be in [1, {steps}]")

    base_start_step = int(args.start_step) if args.start_step is not None else 0
    base_end_step = int(args.end_step) if args.end_step is not None else int(steps)
    if base_start_step >= base_end_step:
        raise ValueError(
            f"Invalid frame range: start={base_start_step}, end={base_end_step}. "
            "Adjust --start-step/--end-step."
        )

    # Infer robot DoF split from record tensor layout for launch diagnostics.
    robot_nq_record = int(qpos.shape[-1]) - 7
    robot_nv_record = int(qvel.shape[-1]) - 6
    launch_steps_all = _summarize_launch_events(
        qpos=qpos,
        qvel=qvel,
        robot_nq=robot_nq_record,
        robot_nv=robot_nv_record,
        env_ids=env_ids_full,
        local_y_min=float(args.launch_local_y_min),
        vel_y_max=float(args.launch_vel_y_max),
        z_min=float(args.launch_z_min),
        teleport_pos_jump_min=float(args.launch_teleport_jump_min),
        verbose=bool(args.list_launches or args.auto_select_env_with_launch),
    )
    effective_env_index = selected_env_index
    if args.auto_select_env_with_launch and args.from_launch:
        selected_launches = launch_steps_all[selected_env_index]
        if selected_launches.size == 0:
            candidates = [idx for idx, launches in enumerate(launch_steps_all) if launches.size > 0]
            if candidates:
                best_idx = max(candidates, key=lambda idx: int(launch_steps_all[idx].size))
                effective_env_index = int(best_idx)
                print(
                    "[WARN] Selected env slot has no launch events. "
                    f"Auto-switched slot {selected_env_index} -> {effective_env_index}."
                )
            else:
                print("[WARN] No launch events found in any env slot.")

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Replay script is tennis-only and always simulates a single environment.
    scene, sim = _build_scene(
        device=device,
        robot_name=robot_name,
        physics_dt=physics_dt,
        env_spacing=env_spacing,
    )
    robot_nq = int(sim.data.qpos.shape[-1]) - 7
    robot_nv = int(sim.data.qvel.shape[-1]) - 6
    if qpos.shape[-1] != int(sim.data.qpos.shape[-1]) or qvel.shape[-1] != int(sim.data.qvel.shape[-1]):
        raise ValueError(
            f"Recorded qpos/qvel dim ({qpos.shape[-1]}/{qvel.shape[-1]}) does not match "
            f"tennis scene dim ({int(sim.data.qpos.shape[-1])}/{int(sim.data.qvel.shape[-1])})."
        )

    if env_origins_full is not None:
        all_env_origins = torch.as_tensor(env_origins_full, device=device, dtype=torch.float32)
        if (
            all_env_origins.ndim != 2
            or all_env_origins.shape[0] != int(num_env_slots)
            or all_env_origins.shape[1] < 3
        ):
            raise ValueError(
                f"Invalid env_origins shape in record: {tuple(all_env_origins.shape)}, "
                f"expected [{int(num_env_slots)}, >=3]."
            )
        all_env_origins = all_env_origins[:, :3]
        print("[INFO] Using env_origins directly from npz.")
    else:
        all_env_origins = _env_origins_from_env_ids(
            env_ids=env_ids_full,
            env_spacing=env_spacing,
            device=device,
            train_num_envs=(train_num_envs if train_num_envs > 0 else None),
        )
        print("[INFO] env_origins missing in npz; recovered from env_ids/env_spacing.")
    _hide_court_overlay_geoms(sim)

    viewer, viser_scene = _create_viewer(sim)
    robot = scene["robot"]
    racket_body_ids, racket_body_names = robot.find_bodies(args.racket_body_name)
    if len(racket_body_ids) != 1:
        raise ValueError(
            f"Expected exactly one racket body from '{args.racket_body_name}', got {list(racket_body_names)}."
        )
    racket_body_id = int(racket_body_ids[0])
    racket_center_offset = torch.tensor(args.racket_center_offset, device=device, dtype=torch.float32)
    if racket_center_offset.numel() != 3:
        raise ValueError(f"--racket-center-offset must provide exactly 3 floats, got {args.racket_center_offset}.")
    if str(args.racket_face_axis).lower() == "auto":
        racket_face_axis_local, face_axis_desc = _infer_racket_face_normal_local_from_collision_geom(
            sim=sim,
            device=device,
        )
    else:
        racket_face_axis_local = _axis_vector(args.racket_face_axis, device=device)
        face_axis_desc = f"manual:+{str(args.racket_face_axis).upper()}"
    racket_face_length = max(float(args.racket_face_length), 1.0e-3)

    slot_options = [str(i) for i in range(int(num_env_slots))]
    env_dropdown = viewer.gui.add_dropdown(
        "Env Slot",
        options=slot_options,
        initial_value=str(effective_env_index),
        hint="Recorded env slot index (not global env_id).",
    )
    racket_center_debug_checkbox = viewer.gui.add_checkbox(
        "Show Racket Center Debug",
        initial_value=False,
        hint="Draw racket mount origin + reward racket center (body + local offset).",
    )

    state: dict[str, object] = {
        "slot": -1,
        "qpos_slot": None,
        "qvel_slot": None,
        "segment_idx": 0,
        "segment_steps": 0,
        "start_step": 0,
        "end_step": 0,
    }

    def _switch_slot(slot_idx: int, *, reason: str, launch_index_override: int | None = None) -> None:
        if slot_idx < 0 or slot_idx >= int(num_env_slots):
            print(f"[WARN] Ignore invalid slot index: {slot_idx}.")
            return

        slot_origin = all_env_origins[slot_idx : slot_idx + 1]
        if hasattr(scene, "env_origins"):
            scene.env_origins[:1] = slot_origin
        else:
            scene.env_origins = slot_origin.clone()
        _align_tennis_court_to_env_origins_with_origins(scene, device=device, env_origins=slot_origin)

        qpos_slot = qpos[:, slot_idx : slot_idx + 1, :]
        qvel_slot = qvel[:, slot_idx : slot_idx + 1, :]
        root_slot = root_state_full[:, slot_idx : slot_idx + 1, :] if root_state_full is not None else None

        start_step = int(base_start_step)
        end_step = int(base_end_step)
        launch_steps = launch_steps_all[slot_idx]
        used_launch_index: int | None = None
        if args.from_launch:
            if launch_steps.size == 0:
                print(f"[WARN] slot={slot_idx}: no launch event found, keep base frame range.")
            else:
                selected_launch_index = int(args.launch_index if launch_index_override is None else launch_index_override)
                if selected_launch_index < 0 or selected_launch_index >= int(launch_steps.size):
                    raise ValueError(
                        f"--launch-index must be in [0, {int(launch_steps.size) - 1}] "
                        f"for env slot {slot_idx}."
                    )
                launch_idx = int(selected_launch_index)
                used_launch_index = launch_idx
                launch_step = int(launch_steps[launch_idx])
                start_step = max(start_step, launch_step)
                if not args.tail_after_launch and launch_idx + 1 < int(launch_steps.size):
                    end_step = min(end_step, int(launch_steps[launch_idx + 1]))

                launch_diag = _find_launch_events(
                    qpos=qpos,
                    qvel=qvel,
                    robot_nq=robot_nq,
                    robot_nv=robot_nv,
                    env_idx=slot_idx,
                    local_y_min=float(args.launch_local_y_min),
                    vel_y_max=float(args.launch_vel_y_max),
                    z_min=float(args.launch_z_min),
                    teleport_pos_jump_min=float(args.launch_teleport_jump_min),
                )[1]
                local_y = float(launch_diag["local_y"][launch_step])
                vel_y = float(launch_diag["vel_y"][launch_step])
                z = float(launch_diag["z"][launch_step])
                print(
                    "[INFO] Launch-selected segment: "
                    f"slot={slot_idx}, launch_idx={launch_idx}, step={launch_step}, "
                    f"ball_local_y={local_y:+.3f}, ball_vy={vel_y:+.3f}, ball_z={z:+.3f}"
                )

        if start_step >= end_step:
            print(
                f"[WARN] slot={slot_idx}: launch-selected range invalid ({start_step}, {end_step}), "
                f"fallback to base range ({base_start_step}, {base_end_step})."
            )
            start_step = int(base_start_step)
            end_step = int(base_end_step)
        if start_step >= end_step:
            raise ValueError(
                f"Invalid frame range after slot switch: start={start_step}, end={end_step}. "
                "Adjust --start-step/--end-step/--launch-index."
            )

        root_xy_first = _extract_first_root_xy(qpos=qpos_slot, root_state=root_slot)
        residual = root_xy_first - slot_origin.detach().cpu().numpy()[:, :2]
        residual_std = float(np.linalg.norm(np.std(residual, axis=0)))
        env_id = int(env_ids_full[slot_idx]) if env_ids_full is not None else slot_idx

        state["slot"] = int(slot_idx)
        state["qpos_slot"] = qpos_slot
        state["qvel_slot"] = qvel_slot
        state["segment_idx"] = 0
        state["start_step"] = int(start_step)
        state["end_step"] = int(end_step)
        state["segment_steps"] = int(end_step - start_step)
        state["launch_steps"] = launch_steps
        state["launch_count"] = int(launch_steps.size)
        state["launch_index"] = used_launch_index

        print(
            "[INFO] Switched replay slot: "
            f"slot={slot_idx}, env_id={env_id}, reason={reason}, "
            f"range=[{start_step}, {end_step}), residual_std_vs_root0={residual_std:.4f}"
        )

    _switch_slot(int(effective_env_index), reason="init")

    print(
        f"Replay loaded: steps={steps}, env_slots={int(num_env_slots)}, robot={robot_name}, "
        f"device={device}, step_dt={step_dt:.4f}, physics_dt={physics_dt:.4f}"
    )
    print(
        "[INFO] Racket center debug params: "
        f"body='{args.racket_body_name}' (id={racket_body_id}), "
        f"offset=({float(racket_center_offset[0]):+.4f}, {float(racket_center_offset[1]):+.4f}, {float(racket_center_offset[2]):+.4f})"
    )
    print(
        "[INFO] Racket face debug colors: +axis(red), -axis(blue), "
        f"axis={face_axis_desc}, vec=({float(racket_face_axis_local[0]):+.3f}, {float(racket_face_axis_local[1]):+.3f}, {float(racket_face_axis_local[2]):+.3f}), length={racket_face_length:.3f}m."
    )
    print(f"[INFO] Base playback frame range: [{base_start_step}, {base_end_step}) ({base_end_step - base_start_step} frames).")
    print("[INFO] Change `Env Slot` in GUI to switch replay target without restarting.")
    print("[INFO] Toggle `Show Racket Center Debug` in GUI to visualize reward racket-center geometry.")
    print("Press Ctrl+C to exit.")

    frame_dt = float(step_dt) / float(args.speed)
    viewer_step_interval = 1
    if float(args.viewer_max_fps) > 0.0:
        viewer_step_interval = max(1, int(round(1.0 / (float(args.viewer_max_fps) * max(frame_dt, 1.0e-6)))))
    print(
        "[INFO] viewer throttle:",
        f"max_fps={float(args.viewer_max_fps):.1f}",
        f"step_interval={viewer_step_interval}",
    )
    start_time = time.perf_counter()
    racket_debug_prev_on = False
    try:
        while True:
            desired_slot = int(env_dropdown.value)
            if desired_slot != int(state["slot"]):
                _switch_slot(desired_slot, reason="gui")
                start_time = time.perf_counter()

            frame_idx = int(state["start_step"]) + int(state["segment_idx"])
            should_render = ((int(state["segment_idx"]) % viewer_step_interval) == 0) or (int(state["segment_idx"]) == 0)
            if should_render:
                qpos_t = torch.as_tensor(state["qpos_slot"][frame_idx], device=device, dtype=torch.float32)
                qvel_t = torch.as_tensor(state["qvel_slot"][frame_idx], device=device, dtype=torch.float32)

                sim.data.qpos[:] = qpos_t
                sim.data.qvel[:] = qvel_t
                sim.forward()
                scene.update(physics_dt)

                show_racket_debug = bool(racket_center_debug_checkbox.value)
                if show_racket_debug:
                    if not viser_scene.debug_visualization_enabled:
                        viser_scene.debug_visualization_enabled = True
                    viser_scene.clear()
                    racket_mount_w, racket_center_w = _compute_racket_center_w(
                        robot=robot,
                        racket_body_id=racket_body_id,
                        racket_center_offset=racket_center_offset,
                    )
                    _, face_plus_dir_w, face_minus_dir_w = _compute_racket_face_dirs_w(
                        robot=robot,
                        racket_body_id=racket_body_id,
                        racket_center_offset=racket_center_offset,
                        face_axis_local=racket_face_axis_local,
                    )
                    ball_pos_w = scene["tennis_ball"].data.root_link_pos_w
                    center_radius = max(float(args.racket_center_radius), 1.0e-3)
                    face_len = float(racket_face_length)
                    face_plus_end_w = racket_center_w + face_plus_dir_w * face_len
                    face_minus_end_w = racket_center_w + face_minus_dir_w * face_len
                    viser_scene.add_sphere(
                        racket_mount_w[0],
                        radius=center_radius * 0.55,
                        color=(1.0, 0.6, 0.0, 0.95),
                    )
                    viser_scene.add_sphere(
                        racket_center_w[0],
                        radius=center_radius,
                        color=(0.1, 1.0, 0.1, 0.95),
                    )
                    viser_scene.add_cylinder(
                        racket_mount_w[0],
                        racket_center_w[0],
                        radius=max(center_radius * 0.20, 0.003),
                        color=(0.1, 1.0, 0.1, 0.80),
                    )
                    viser_scene.add_cylinder(
                        racket_center_w[0],
                        face_plus_end_w[0],
                        radius=max(center_radius * 0.16, 0.0028),
                        color=(1.0, 0.15, 0.15, 0.95),
                    )
                    viser_scene.add_cylinder(
                        racket_center_w[0],
                        face_minus_end_w[0],
                        radius=max(center_radius * 0.16, 0.0028),
                        color=(0.15, 0.35, 1.0, 0.95),
                    )
                    viser_scene.add_sphere(
                        face_plus_end_w[0],
                        radius=max(center_radius * 0.42, 0.010),
                        color=(1.0, 0.15, 0.15, 0.95),
                    )
                    viser_scene.add_sphere(
                        face_minus_end_w[0],
                        radius=max(center_radius * 0.42, 0.010),
                        color=(0.15, 0.35, 1.0, 0.95),
                    )
                    viser_scene.add_sphere(
                        ball_pos_w[0],
                        radius=max(center_radius * 0.45, 0.015),
                        color=(0.15, 0.55, 1.0, 0.90),
                    )
                elif racket_debug_prev_on:
                    viser_scene.clear()
                racket_debug_prev_on = show_racket_debug

                viser_scene.update(_to_cpu_wp_data(sim.data))

            state["segment_idx"] = int(state["segment_idx"]) + 1
            if int(state["segment_idx"]) >= int(state["segment_steps"]):
                if args.cycle_launches and args.from_launch:
                    launch_count = int(state.get("launch_count", 0) or 0)
                    launch_index = state.get("launch_index", None)
                    if launch_count > 0 and launch_index is not None:
                        next_launch_index = int(launch_index) + 1
                        if next_launch_index < launch_count:
                            _switch_slot(int(state["slot"]), reason="cycle-next-launch", launch_index_override=next_launch_index)
                            start_time = time.perf_counter()
                            continue
                        _switch_slot(int(state["slot"]), reason="cycle-wrap-launch", launch_index_override=0)
                        start_time = time.perf_counter()
                        continue
                    state["segment_idx"] = 0
                    start_time = time.perf_counter()
                else:
                    state["segment_idx"] = 0
                    start_time = time.perf_counter()

            target_time = start_time + int(state["segment_idx"]) * frame_dt
            delay = target_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
