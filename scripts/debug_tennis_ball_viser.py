import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from omegaconf import DictConfig, OmegaConf

# Add project root to path for local imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.assets.tennis import get_tennis_ball_cfg, get_tennis_court_cfg

VALID_MODES = ("easy", "medium", "hard")
MODE_COLORS = {
    "easy": np.array([0.20, 0.85, 0.20, 1.0], dtype=np.float32),
    "medium": np.array([1.00, 0.65, 0.15, 1.0], dtype=np.float32),
    "hard": np.array([0.95, 0.20, 0.20, 1.0], dtype=np.float32),
}


@dataclass
class LaunchBank:
    path: str
    pos_local: np.ndarray
    vel: np.ndarray
    ang: np.ndarray
    sim_physics_dt: float | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path


def _cfg_get(cfg: DictConfig | dict, key: str, default):
    node = cfg
    for part in key.split("."):
        if isinstance(node, DictConfig):
            if part not in node:
                return default
            node = node[part]
        elif isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        else:
            return default
    return node


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode == "mid":
        mode = "medium"
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {VALID_MODES} (or 'mid').")
    return mode


def _parse_modes(modes_csv: str) -> list[str]:
    modes = []
    for raw in str(modes_csv).split(","):
        token = raw.strip()
        if not token:
            continue
        modes.append(_normalize_mode(token))
    if not modes:
        raise ValueError("At least one mode must be provided in --modes.")
    return modes


def _load_launch_bank(path: Path) -> LaunchBank:
    if not path.exists():
        raise FileNotFoundError(f"Launch bank file not found: {path}")
    with np.load(path) as data:
        keys = set(data.keys())

        def _read(*cands: str) -> np.ndarray | None:
            for cand in cands:
                if cand in keys:
                    return np.asarray(data[cand], dtype=np.float32)
            return None

        pos_local = _read("launch_pos_local", "local_pos")
        vel = _read("launch_vel", "vel")
        ang = _read("launch_ang", "ang")
        sim_physics_dt = None
        if "sim_physics_dt" in keys:
            sim_physics_dt = float(np.asarray(data["sim_physics_dt"], dtype=np.float64).reshape(-1)[0])

    if pos_local is None or vel is None or ang is None:
        raise ValueError(
            f"Invalid launch bank file {path}. "
            "Expected keys: launch_pos_local/local_pos, launch_vel/vel, launch_ang/ang."
        )
    if (
        pos_local.ndim != 2
        or vel.ndim != 2
        or ang.ndim != 2
        or pos_local.shape[1] != 3
        or vel.shape[1] != 3
        or ang.shape[1] != 3
    ):
        raise ValueError(
            f"Invalid launch bank tensor shapes from {path}: "
            f"pos={pos_local.shape}, vel={vel.shape}, ang={ang.shape}"
        )
    n = int(pos_local.shape[0])
    if n <= 0 or int(vel.shape[0]) != n or int(ang.shape[0]) != n:
        raise ValueError(
            f"Inconsistent launch bank lengths from {path}: "
            f"pos={pos_local.shape[0]}, vel={vel.shape[0]}, ang={ang.shape[0]}"
        )
    finite = np.isfinite(pos_local).all(axis=1) & np.isfinite(vel).all(axis=1) & np.isfinite(ang).all(axis=1)
    if not finite.all():
        keep = int(finite.sum())
        if keep <= 0:
            raise ValueError(f"Launch bank {path} contains no finite rows.")
        pos_local = pos_local[finite]
        vel = vel[finite]
        ang = ang[finite]
        print(f"[WARN] Dropped {n - keep} non-finite launch rows from {path}.")

    return LaunchBank(
        path=str(path),
        pos_local=pos_local,
        vel=vel,
        ang=ang,
        sim_physics_dt=sim_physics_dt,
    )


def _load_launch_bank_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Launch bank manifest not found: {manifest_path}")
    out: dict[str, str] = {}
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _align_tennis_court_to_env_origins(scene: Scene, *, device: str) -> None:
    if "tennis_court" not in scene.entities:
        return
    court = scene["tennis_court"]
    if not getattr(court, "is_mocap", False):
        return
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    pose = torch.zeros((1, 7), device=device, dtype=torch.float32)
    pose[:, :3] = scene.env_origins[env_ids]
    pose[:, 3] = 1.0
    court.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def _to_cpu_wp_data(wp_data):
    xpos = getattr(wp_data, "xpos", None)
    if xpos is None:
        return wp_data
    # Viser rendering only needs transform/mocap arrays; avoid copying qpos/qvel every frame.
    return SimpleNamespace(
        xpos=wp_data.xpos.detach().cpu().clone(),
        xmat=wp_data.xmat.detach().cpu().clone(),
        mocap_pos=wp_data.mocap_pos.detach().cpu().clone() if hasattr(wp_data, "mocap_pos") else None,
        mocap_quat=wp_data.mocap_quat.detach().cpu().clone() if hasattr(wp_data, "mocap_quat") else None,
    )


def _create_viser_viewer(sim: Simulation):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    viewer = viser.ViserServer(label="tennis-multiball-debug")
    viser_scene = ViserMujocoScene.create(server=viewer, mj_model=sim.mj_model, num_envs=1)
    viser_scene.create_visualization_gui()
    viser_scene.debug_visualization_enabled = False
    viser_scene.camera_tracking_enabled = False
    return viewer, viser_scene


def _hide_court_overlay_geoms(sim: Simulation) -> None:
    import mujoco

    hidden = 0
    for geom_id in range(sim.mj_model.ngeom):
        name = mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        if name.startswith("tennis_court/") and name.endswith("_collision"):
            sim.mj_model.geom_group[geom_id] = 3
            hidden += 1
    if hidden > 0:
        print(f"[INFO] Hidden {hidden} court collision overlay geoms for clearer texture rendering.")


def _ball_entity_name(ball_id: int) -> str:
    return f"tennis_ball_{ball_id:03d}"


def _resolve_sim_params(args, task_cfg: DictConfig) -> dict:
    physics_dt_cfg = float(_cfg_get(task_cfg, "sim.mujoco_physics_dt", 0.0005))
    nconmax_cfg = int(_cfg_get(task_cfg, "sim.nconmax", 192))
    njmax_cfg = int(_cfg_get(task_cfg, "sim.njmax", 900))
    iterations_cfg = int(_cfg_get(task_cfg, "sim.mujoco_iterations", 32))
    ls_iterations_cfg = int(_cfg_get(task_cfg, "sim.mujoco_ls_iterations", 64))
    ccd_iterations_cfg = int(_cfg_get(task_cfg, "sim.mujoco_ccd_iterations", 96))
    multiccd_cfg = bool(_cfg_get(task_cfg, "sim.mujoco_multiccd", False))

    preset = str(args.sim_preset).strip().lower()
    if preset not in {"train", "fast"}:
        raise ValueError(f"Unsupported --sim-preset={args.sim_preset}. Expected one of: train, fast.")

    if preset == "train":
        nconmax = nconmax_cfg
        njmax = njmax_cfg
        iterations = iterations_cfg
        ls_iterations = ls_iterations_cfg
        ccd_iterations = ccd_iterations_cfg
        multiccd = multiccd_cfg
    else:
        # Faster visualization defaults: keep physics_dt unchanged, only relax solver cost.
        nconmax = min(nconmax_cfg, 128)
        njmax = min(njmax_cfg, 400)
        iterations = min(iterations_cfg, 10)
        ls_iterations = min(ls_iterations_cfg, 20)
        ccd_iterations = min(ccd_iterations_cfg, 16)
        multiccd = False

    return {
        "preset": preset,
        "physics_dt": float(physics_dt_cfg),
        "nconmax": int(nconmax),
        "njmax": int(njmax),
        "iterations": int(iterations),
        "ls_iterations": int(ls_iterations),
        "ccd_iterations": int(ccd_iterations),
        "multiccd": bool(multiccd),
    }


def _build_scene(args, task_cfg: DictConfig):
    sim_params = _resolve_sim_params(args, task_cfg)
    sim_dt = float(sim_params["physics_dt"])

    env_spacing = float(_cfg_get(task_cfg, "viewer.env_spacing", 30.0))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=env_spacing)
    scene_cfg.terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing, num_envs=1)
    scene_cfg.entities["tennis_court"] = get_tennis_court_cfg(
        texture=str(_cfg_get(task_cfg, "tennis.court_texture", "green")),
        net_height=float(_cfg_get(task_cfg, "tennis.net_height", 0.914)),
        net_collision_half_thickness=float(_cfg_get(task_cfg, "tennis.net_collision_half_thickness", 0.06)),
        enable_racket_court_collision=bool(_cfg_get(task_cfg, "tennis.racket_court_collision", False)),
    )

    for bi in range(int(args.num_balls)):
        scene_cfg.entities[_ball_entity_name(bi)] = get_tennis_ball_cfg()

    scene = Scene(scene_cfg, device=args.device)
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(
            nconmax=int(sim_params["nconmax"]),
            njmax=int(sim_params["njmax"]),
            mujoco=MujocoCfg(
                timestep=sim_dt,
                iterations=int(sim_params["iterations"]),
                ls_iterations=int(sim_params["ls_iterations"]),
                ccd_iterations=int(sim_params["ccd_iterations"]),
                multiccd=bool(sim_params["multiccd"]),
            ),
        ),
        model=scene.compile(),
        device=args.device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    if not hasattr(scene, "env_origins") and hasattr(scene, "env_offsets"):
        scene.env_origins = scene.env_offsets
    _align_tennis_court_to_env_origins(scene, device=args.device)
    scene.write_data_to_sim()
    sim.forward()
    sim.sense()
    scene.update(sim_dt)
    balls = [scene[_ball_entity_name(bi)] for bi in range(int(args.num_balls))]
    return scene, sim, balls, sim_dt, sim_params


def _set_ball_colors(sim: Simulation, ball_modes: list[str]) -> None:
    for bi, mode in enumerate(ball_modes):
        geom_name = f"{_ball_entity_name(bi)}/tennis_ball_geom"
        geom_id = sim.mj_model.geom(geom_name).id
        sim.mj_model.geom_rgba[geom_id, :4] = MODE_COLORS[mode]


def _sample_launch(bank: LaunchBank, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    idx = int(rng.integers(0, bank.pos_local.shape[0]))
    return idx, bank.pos_local[idx], bank.vel[idx], bank.ang[idx]


def _spawn_ball(
    *,
    ball,
    mode: str,
    bank: LaunchBank,
    rng: np.random.Generator,
    env_origin: torch.Tensor,
    env_ids: torch.Tensor,
    device: str,
) -> int:
    launch_idx, launch_pos_local, launch_vel, launch_ang = _sample_launch(bank, rng)
    state = torch.zeros((1, 13), device=device, dtype=torch.float32)
    state[:, :3] = torch.as_tensor(launch_pos_local, device=device, dtype=torch.float32) + env_origin
    state[:, 3] = 1.0
    state[:, 7:10] = torch.as_tensor(launch_vel, device=device, dtype=torch.float32)
    state[:, 10:13] = torch.as_tensor(launch_ang, device=device, dtype=torch.float32)
    ball.write_root_state_to_sim(state, env_ids=env_ids)
    return launch_idx


def run(args):
    if args.num_balls <= 0:
        raise ValueError("--num-balls must be > 0.")
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    modes = _parse_modes(args.modes)
    task_cfg_path = _resolve_path(args.task_cfg)
    task_cfg = OmegaConf.load(task_cfg_path)
    OmegaConf.resolve(task_cfg)

    launch_root = _resolve_path(args.launch_bank_root)
    manifest_data: Optional[dict[str, str]] = None
    manifest_path: Optional[Path] = None
    if str(args.launch_bank_manifest).strip():
        manifest_path = _resolve_path(args.launch_bank_manifest)
        manifest_data = _load_launch_bank_manifest(manifest_path)
    elif bool(args.use_manifest):
        auto_manifest = launch_root / "launch_bank_manifest.txt"
        if auto_manifest.exists():
            manifest_path = auto_manifest
            manifest_data = _load_launch_bank_manifest(manifest_path)

    launch_banks = {}
    for mode in sorted(set(modes), key=modes.index):
        bank_path: Path
        if manifest_data is not None:
            key = f"{mode}"
            if key not in manifest_data:
                raise KeyError(
                    f"Manifest {manifest_path} missing key '{key}'. "
                    f"Available keys: {sorted(manifest_data.keys())}"
                )
            bank_path = _resolve_path(manifest_data[key])
        else:
            bank_path = launch_root / f"launch_bank_{mode}.npz"
        launch_banks[mode] = _load_launch_bank(bank_path)

    expected_dt = float(_cfg_get(task_cfg, "sim.mujoco_physics_dt", 0.0005))
    dt_tol = max(1.0e-6, abs(expected_dt) * 1.0e-3)
    for mode, bank in launch_banks.items():
        if bank.sim_physics_dt is None:
            print(
                f"[WARN] Launch bank {mode} has no sim_physics_dt metadata: {bank.path}. "
                "Regenerate with current generate_traj.sh for strict consistency checks."
            )
            continue
        if abs(float(bank.sim_physics_dt) - expected_dt) > dt_tol:
            raise ValueError(
                f"Launch bank physics_dt mismatch for mode={mode}: bank={bank.sim_physics_dt}, "
                f"task_cfg={expected_dt}. Regenerate with matching config."
            )

    scene, sim, balls, physics_dt, sim_params = _build_scene(args, task_cfg)
    if not args.show_collision_overlays:
        _hide_court_overlay_geoms(sim)
    env_origin = scene.env_origins[0]
    env_ids = torch.tensor([0], device=args.device, dtype=torch.long)

    ball_modes = [modes[i % len(modes)] for i in range(args.num_balls)]
    if not args.disable_mode_colors:
        _set_ball_colors(sim, ball_modes)

    rng = np.random.default_rng(args.seed)
    launch_ids = np.full((args.num_balls,), -1, dtype=np.int64)
    age_steps = torch.zeros((args.num_balls,), device=args.device, dtype=torch.int64)

    def respawn(ball_ids: list[int]) -> None:
        for bi in ball_ids:
            launch_ids[bi] = _spawn_ball(
                ball=balls[bi],
                mode=ball_modes[bi],
                bank=launch_banks[ball_modes[bi]],
                rng=rng,
                env_origin=env_origin,
                env_ids=env_ids,
                device=args.device,
            )
            age_steps[bi] = 0
        scene.write_data_to_sim()
        sim.forward()
        sim.sense()
        scene.update(physics_dt)

    respawn(list(range(args.num_balls)))

    control_dt = float(_cfg_get(task_cfg, "sim.step_dt", 0.02))
    if int(args.sim_substeps_per_loop) > 0:
        sim_substeps_per_loop = int(args.sim_substeps_per_loop)
    else:
        sim_substeps_per_loop = max(1, int(round(control_dt / physics_dt)))

    sleep_dt = max(float(args.realtime_scale), 0.0) * physics_dt * float(sim_substeps_per_loop)
    min_loop_dt = 0.0 if args.viewer_max_fps <= 0 else 1.0 / float(args.viewer_max_fps)

    out_margin_z = float(_cfg_get(task_cfg, "command.out_margin_z", -0.25))
    out_max_z = float(_cfg_get(task_cfg, "command.out_max_z", 6.0))
    court_x_limit = float(_cfg_get(task_cfg, "command.court_x_limit", 4.2))
    court_y_limit = float(_cfg_get(task_cfg, "command.court_y_limit", 12.2))
    max_flight_steps = max(1, int(round(float(args.max_flight_time_s) / physics_dt)))

    print(
        f"[INFO] task_cfg={task_cfg_path} | device={args.device} | num_envs=1 | num_balls={args.num_balls} | dt={physics_dt:.6f}"
    )
    print(
        "[INFO] sim params: "
        f"preset={sim_params['preset']} nconmax={sim_params['nconmax']} njmax={sim_params['njmax']} "
        f"iters={sim_params['iterations']}/{sim_params['ls_iterations']} ccd={sim_params['ccd_iterations']} "
        f"multiccd={sim_params['multiccd']}"
    )
    if manifest_path is not None:
        print(f"[INFO] launch manifest: {manifest_path}")
    for mode in sorted(launch_banks.keys()):
        bank = launch_banks[mode]
        print(
            f"[INFO] bank[{mode}] path={bank.path} samples={bank.pos_local.shape[0]} "
            f"sim_physics_dt={bank.sim_physics_dt}"
        )

    print(f"[INFO] Generating {args.offline_frames} frames offline...")
    frames = []
    sim_step = 0
    
    # Generate offline trajectory
    while sim_step < args.offline_frames:
        ball_pos_l = torch.stack([ball.data.root_link_pos_w[0] - env_origin for ball in balls], dim=0)
        ball_vel = torch.stack([ball.data.root_link_lin_vel_w[0] for ball in balls], dim=0)
        finite_mask = torch.isfinite(ball_pos_l).all(dim=-1) & torch.isfinite(ball_vel).all(dim=-1)
        in_range_mask = (
            (ball_pos_l[:, 2] >= out_margin_z)
            & (ball_pos_l[:, 2] <= out_max_z)
            & (ball_pos_l[:, 0].abs() <= court_x_limit + 2.0)
            & (ball_pos_l[:, 1].abs() <= court_y_limit + 4.0)
            & (age_steps < max_flight_steps)
        )
        respawn_mask = ~(finite_mask & in_range_mask)
        if respawn_mask.any():
            respawn_ids = respawn_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            respawn(respawn_ids)

        substeps = sim_substeps_per_loop
        for _ in range(max(0, int(substeps))):
            sim.step()
        if substeps > 0:
            scene.update(physics_dt * float(substeps))
            age_steps.add_(int(substeps))
            sim_step += 1 # We treat sim_step as frame index here
            
        frames.append(_to_cpu_wp_data(sim.data))
        
        if sim_step % 200 == 0:
            print(f"[INFO] Generated {sim_step}/{args.offline_frames} frames...")

    print(f"[INFO] Offline generation complete. Starting Viser to replay {len(frames)} frames in loop.")
    _viewer, viser_scene = _create_viser_viewer(sim)

    try:
        frame_idx = 0
        while True:
            t0 = time.perf_counter()
            viser_scene.update(frames[frame_idx])
            frame_idx = (frame_idx + 1) % len(frames)
            
            target_loop_dt = max(float(sleep_dt), float(min_loop_dt))
            if target_loop_dt > 0.0:
                delay = target_loop_dt - (time.perf_counter() - t0)
                if delay > 0.0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-court, multi-ball tennis visualization in Viser. "
            "Uses one env (no robot) and continuously launches easy/medium/hard bank trajectories."
        )
    )
    parser.add_argument("--task-cfg", type=str, default="cfg/task/G1/G1_tennis_highlevel.yaml")
    parser.add_argument(
        "--launch-bank-root",
        type=str,
        default="data/tennis_launch_bank/highlevel_subsets",
        help="Directory containing launch_bank_easy.npz / launch_bank_medium.npz / launch_bank_hard.npz",
    )
    parser.add_argument(
        "--launch-bank-manifest",
        type=str,
        default="",
        help="Optional manifest path written by generate_traj.sh. If set, bank paths are read from this file.",
    )
    parser.add_argument(
        "--use-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-load launch_bank_manifest.txt under --launch-bank-root when present (default: true).",
    )
    parser.add_argument("--num-balls", type=int, default=48, help="Concurrent balls in the same court.")
    parser.add_argument(
        "--modes",
        type=str,
        default="easy,medium,hard",
        help="Comma-separated launch subsets to mix, e.g. 'easy,mid,hard'.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--sim-preset",
        type=str,
        default="train",
        choices=("fast", "train"),
        help="Physics solver preset. Both presets keep physics_dt identical to task cfg; 'fast' only lowers solver workload.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline-frames", type=int, default=1500, help="Number of frames to generate offline before replaying.")
    parser.add_argument("--max-flight-time-s", type=float, default=2.8, help="Respawn a ball if its flight exceeds this time.")
    parser.add_argument(
        "--sim-substeps-per-loop",
        type=int,
        default=0,
        help="Physics steps per render loop. <=0 uses auto ratio round(task.sim.step_dt / physics_dt).",
    )
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--realtime-scale", type=float, default=0.0, help="sleep = physics_dt * scale; 0 means fastest.")
    parser.add_argument("--viewer-max-fps", type=float, default=45.0, help="Throttle main loop FPS. <=0 disables.")
    parser.add_argument("--show-collision-overlays", action="store_true", help="Show court collision helper geoms.")
    parser.add_argument("--disable-mode-colors", action="store_true", help="Do not color balls by easy/medium/hard mode.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
