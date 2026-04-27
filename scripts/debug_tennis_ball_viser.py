import argparse
import math
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

from active_adaptation.assets.tennis import (
    TERRAIN_BALL_BOUNCE_FRICTION,
    TERRAIN_BALL_BOUNCE_SOLREF,
    get_tennis_ball_cfg,
    get_tennis_court_cfg,
)
from active_adaptation.envs.mdp.commands.highlevel_tennis import HighLevelTennisConfig

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
    air_drag_k: float | None = None
    drag_coef: float | None = None
    lift_spin_scale: float | None = None
    spin_damping_coef: float | None = None
    air_density: float | None = None
    ball_radius: float | None = None


@dataclass
class BallAeroParams:
    ball_radius: float
    air_density: float
    air_drag_k: float
    drag_coef: float
    lift_spin_scale: float
    spin_damping_coef: float

    @property
    def aero_force_k(self) -> float:
        return 0.5 * self.air_density * math.pi * (self.ball_radius ** 2) * self.air_drag_k


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
        air_drag_k = float(np.asarray(data["air_drag_k"], dtype=np.float64).reshape(-1)[0]) if "air_drag_k" in keys else None
        drag_coef = float(np.asarray(data["drag_coef"], dtype=np.float64).reshape(-1)[0]) if "drag_coef" in keys else None
        lift_spin_scale = (
            float(np.asarray(data["lift_spin_scale"], dtype=np.float64).reshape(-1)[0]) if "lift_spin_scale" in keys else None
        )
        spin_damping_coef = (
            float(np.asarray(data["spin_damping_coef"], dtype=np.float64).reshape(-1)[0]) if "spin_damping_coef" in keys else None
        )
        air_density = float(np.asarray(data["air_density"], dtype=np.float64).reshape(-1)[0]) if "air_density" in keys else None
        ball_radius = float(np.asarray(data["ball_radius"], dtype=np.float64).reshape(-1)[0]) if "ball_radius" in keys else None

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
        air_drag_k=air_drag_k,
        drag_coef=drag_coef,
        lift_spin_scale=lift_spin_scale,
        spin_damping_coef=spin_damping_coef,
        air_density=air_density,
        ball_radius=ball_radius,
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


def _resolve_command_cfg(task_cfg: DictConfig) -> HighLevelTennisConfig:
    raw = _cfg_get(task_cfg, "command.config", {})
    return HighLevelTennisConfig.from_any(raw)


def _resolve_ball_aero_params(task_cfg: DictConfig) -> BallAeroParams:
    cmd_cfg = _resolve_command_cfg(task_cfg)
    return BallAeroParams(
        ball_radius=float(cmd_cfg.ball_radius),
        air_density=float(cmd_cfg.air_density),
        air_drag_k=float(cmd_cfg.air_drag_k),
        drag_coef=float(cmd_cfg.drag_coef),
        lift_spin_scale=float(cmd_cfg.lift_spin_scale),
        spin_damping_coef=float(cmd_cfg.spin_damping_coef),
    )


def _assert_bank_matches_physics(mode: str, bank: LaunchBank, aero: BallAeroParams) -> None:
    checks = (
        ("air_drag_k", bank.air_drag_k, aero.air_drag_k),
        ("drag_coef", bank.drag_coef, aero.drag_coef),
        ("lift_spin_scale", bank.lift_spin_scale, aero.lift_spin_scale),
        ("spin_damping_coef", bank.spin_damping_coef, aero.spin_damping_coef),
        ("air_density", bank.air_density, aero.air_density),
        ("ball_radius", bank.ball_radius, aero.ball_radius),
    )
    tol = 1.0e-6
    for name, got, expected in checks:
        if got is None:
            print(f"[WARN] Launch bank {mode} missing '{name}' metadata: {bank.path}")
            continue
        if abs(float(got) - float(expected)) > tol:
            raise ValueError(
                f"Launch bank physics mismatch for mode={mode}, field={name}: "
                f"bank={float(got):.8f}, task_cfg={float(expected):.8f}. "
                "Regenerate bank with current config to keep debug equivalent to training."
            )


class BallAeroApplier:
    def __init__(self, *, params: BallAeroParams):
        self.params = params
        self.aero_force_k = float(params.aero_force_k)

    def _compute_wrench(self, *, vel_w: torch.Tensor, ang_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        speed = vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        spin_mag = ang_w.norm(dim=-1, keepdim=True)
        spin_scaled = spin_mag / (2.0 * math.pi) * float(self.params.lift_spin_scale)
        vel_dir = vel_w / speed
        spin_axis = ang_w / spin_mag.clamp_min(1e-6)
        cl = 1.0 / (2.0 + torch.abs(speed / (spin_scaled + 1e-6)))
        drag_force = -self.aero_force_k * float(self.params.drag_coef) * speed * vel_w
        lift_dir = torch.cross(spin_axis, vel_dir, dim=-1)
        lift_force = self.aero_force_k * cl * speed.square() * lift_dir
        total_force = drag_force + lift_force
        spin_damping_torque = -float(self.params.spin_damping_coef) * ang_w
        return total_force, spin_damping_torque

    def apply_to_sim(self, *, sim: Simulation, ball_model_body_ids: torch.Tensor) -> None:
        # Read body spatial velocity directly from sim data for all balls in one batch:
        # cvel[..., 0:3] = angular velocity, cvel[..., 3:6] = linear velocity.
        cvel = sim.data.cvel._tensor[0, ball_model_body_ids]
        ang_w = cvel[:, 0:3]
        vel_w = cvel[:, 3:6]
        total_force, spin_damping_torque = self._compute_wrench(vel_w=vel_w, ang_w=ang_w)
        xfrc = sim.data.xfrc_applied._tensor
        xfrc[0, ball_model_body_ids, 0:3] = total_force
        xfrc[0, ball_model_body_ids, 3:6] = spin_damping_torque


def _build_scene(args, task_cfg: DictConfig):
    from mjlab.utils import spec_config as spec_cfg

    sim_params = _resolve_sim_params(args, task_cfg)
    sim_dt = float(sim_params["physics_dt"])

    env_spacing = float(_cfg_get(task_cfg, "viewer.env_spacing", 30.0))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=env_spacing)
    scene_cfg.terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing, num_envs=1)
    scene_cfg.terrain.collisions = (
        spec_cfg.CollisionCfg(
            geom_names_expr=(".*",),
            contype=1,
            conaffinity=17,
            condim=3,
            disable_other_geoms=False,
            friction=TERRAIN_BALL_BOUNCE_FRICTION,
            solref=TERRAIN_BALL_BOUNCE_SOLREF,
        ),
    )
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
    aero_params = _resolve_ball_aero_params(task_cfg)

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
        elif abs(float(bank.sim_physics_dt) - expected_dt) > dt_tol:
            raise ValueError(
                f"Launch bank physics_dt mismatch for mode={mode}: bank={bank.sim_physics_dt}, "
                f"task_cfg={expected_dt}. Regenerate with matching config."
            )
        _assert_bank_matches_physics(mode, bank, aero_params)

    scene, sim, balls, physics_dt, sim_params = _build_scene(args, task_cfg)
    aero_applier = BallAeroApplier(params=aero_params)
    ball_model_body_ids = torch.tensor(
        [int(sim.mj_model.body(f"{_ball_entity_name(bi)}/tennis_ball").id) for bi in range(args.num_balls)],
        device=args.device,
        dtype=torch.long,
    )
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
    prev_ground_contact = torch.zeros((args.num_balls,), device=args.device, dtype=torch.bool)
    active_bounce_markers: list[tuple[np.ndarray, int]] = []
    bounce_count_total = 0

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
            prev_ground_contact[bi] = False
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

    out_margin_z = float(
        _cfg_get(task_cfg, "command.config.court.out_margin_z", _cfg_get(task_cfg, "command.out_margin_z", -0.25))
    )
    out_max_z = float(_cfg_get(task_cfg, "command.out_max_z", 6.0))
    court_x_limit = float(
        _cfg_get(task_cfg, "command.config.court.x_limit", _cfg_get(task_cfg, "command.court_x_limit", 4.2))
    )
    court_y_limit = float(
        _cfg_get(task_cfg, "command.config.court.y_limit", _cfg_get(task_cfg, "command.court_y_limit", 12.2))
    )
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
    print(
        "[INFO] aero params: "
        f"ball_radius={aero_params.ball_radius:.6f} air_density={aero_params.air_density:.6f} "
        f"air_drag_k={aero_params.air_drag_k:.6f} drag_coef={aero_params.drag_coef:.6f} "
        f"lift_spin_scale={aero_params.lift_spin_scale:.6f} spin_damping_coef={aero_params.spin_damping_coef:.6f}"
    )
    if manifest_path is not None:
        print(f"[INFO] launch manifest: {manifest_path}")
    for mode in sorted(launch_banks.keys()):
        bank = launch_banks[mode]
        print(
            f"[INFO] bank[{mode}] path={bank.path} samples={bank.pos_local.shape[0]} "
            f"sim_physics_dt={bank.sim_physics_dt}"
        )

    def _draw_bounce_markers(viser_scene, markers: list[np.ndarray]) -> None:
        if not bool(args.show_bounce_markers):
            return
        viser_scene.clear()
        if len(markers) == 0:
            return
        if not viser_scene.debug_visualization_enabled:
            viser_scene.debug_visualization_enabled = True
        radius = float(args.bounce_marker_radius)
        for p in markers:
            viser_scene.add_sphere(p, radius=radius, color=(0.1, 0.9, 1.0, 0.95))

    def _advance_live_bounce_markers(new_markers: list[np.ndarray]) -> list[np.ndarray]:
        nonlocal active_bounce_markers
        ttl_init = max(1, int(args.bounce_marker_ttl))
        max_markers = max(1, int(args.bounce_marker_max))
        for p in new_markers:
            active_bounce_markers.append((p, ttl_init))
        if len(active_bounce_markers) > max_markers:
            active_bounce_markers = active_bounce_markers[-max_markers:]
        next_markers: list[tuple[np.ndarray, int]] = []
        draw_points: list[np.ndarray] = []
        for p, ttl in active_bounce_markers:
            if ttl > 0:
                draw_points.append(p)
                next_markers.append((p, ttl - 1))
        active_bounce_markers = next_markers
        return draw_points

    def _simulate_one_loop() -> list[np.ndarray]:
        nonlocal bounce_count_total
        loop_bounce_points: list[np.ndarray] = []
        ball_pos_w = sim.data.xpos._tensor[0, ball_model_body_ids]
        ball_vel = sim.data.cvel._tensor[0, ball_model_body_ids, 3:6]
        ball_pos_l = ball_pos_w - env_origin.unsqueeze(0)
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
            aero_applier.apply_to_sim(sim=sim, ball_model_body_ids=ball_model_body_ids)
            sim.step()
            if bool(args.show_bounce_markers):
                pos_w = sim.data.xpos._tensor[0, ball_model_body_ids]
                vz = sim.data.cvel._tensor[0, ball_model_body_ids, 5]
                ground_contact = pos_w[:, 2] <= (
                    float(aero_params.ball_radius) + float(args.bounce_contact_eps)
                )
                new_bounce = (~prev_ground_contact) & ground_contact & (vz < 0.0)
                if new_bounce.any():
                    pts = pos_w[new_bounce].detach().clone()
                    pts[:, 2] = float(aero_params.ball_radius) + 0.005
                    points_np = list(pts.cpu().numpy())
                    loop_bounce_points.extend(points_np)
                    bounce_count_total += len(points_np)
                prev_ground_contact[:] = ground_contact
        if substeps > 0:
            scene.update(physics_dt * float(substeps))
            age_steps.add_(int(substeps))
        else:
            scene.update(0.0)
        return loop_bounce_points

    if int(args.offline_frames) <= 0:
        print("[INFO] Live mode enabled (offline_frames<=0): start rendering immediately.")
        _viewer, viser_scene = _create_viser_viewer(sim)
        live_step = 0
        live_t0 = time.perf_counter()
        try:
            while True:
                loop_t0 = time.perf_counter()
                new_bounces = _simulate_one_loop()
                live_step += 1
                draw_points = _advance_live_bounce_markers(new_bounces)
                _draw_bounce_markers(viser_scene, draw_points)
                viser_scene.update(sim.data)
                if int(args.print_every) > 0 and (live_step % int(args.print_every) == 0):
                    elapsed = max(time.perf_counter() - live_t0, 1.0e-6)
                    fps = float(live_step) / elapsed
                    print(
                        f"[INFO] Live simulated {live_step} loops | live_fps={fps:.1f} "
                        f"| bounces={bounce_count_total}"
                    )
                if int(args.max_live_steps) > 0 and live_step >= int(args.max_live_steps):
                    print(f"[INFO] Reached max_live_steps={args.max_live_steps}, exiting live mode.")
                    break
                target_loop_dt = max(float(sleep_dt), float(min_loop_dt))
                if target_loop_dt > 0.0:
                    delay = target_loop_dt - (time.perf_counter() - loop_t0)
                    if delay > 0.0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
        return

    print(f"[INFO] Generating {args.offline_frames} frames offline...")
    frames = []
    frame_bounce_events: list[list[np.ndarray]] = []
    sim_step = 0
    gen_t0 = time.perf_counter()
    while sim_step < int(args.offline_frames):
        frame_bounce_events.append(_simulate_one_loop())
        sim_step += 1  # frame index
        frames.append(_to_cpu_wp_data(sim.data))
        if int(args.print_every) > 0 and (sim_step % int(args.print_every) == 0):
            elapsed = max(time.perf_counter() - gen_t0, 1.0e-6)
            fps = float(sim_step) / elapsed
            print(
                f"[INFO] Generated {sim_step}/{args.offline_frames} frames | offline_fps={fps:.1f} "
                f"| bounces={bounce_count_total}"
            )

    print(f"[INFO] Offline generation complete. Starting Viser to replay {len(frames)} frames in loop.")
    _viewer, viser_scene = _create_viser_viewer(sim)
    try:
        frame_idx = 0
        while True:
            t0 = time.perf_counter()
            if bool(args.show_bounce_markers):
                _draw_bounce_markers(viser_scene, frame_bounce_events[frame_idx])
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
    parser.add_argument("--num-balls", type=int, default=12, help="Concurrent balls in the same court.")
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
    parser.add_argument(
        "--offline-frames",
        type=int,
        default=1200,
        help="Number of frames to pre-generate offline before replaying. Set <=0 to run in live mode.",
    )
    parser.add_argument(
        "--max-live-steps",
        type=int,
        default=-1,
        help="Only for live mode (offline_frames<=0). >0 means auto-exit after this many loops.",
    )
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
    parser.add_argument(
        "--show-bounce-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show detected ball-ground bounce markers in Viser (default: true).",
    )
    parser.add_argument(
        "--bounce-marker-radius",
        type=float,
        default=0.05,
        help="Radius of bounce marker spheres.",
    )
    parser.add_argument(
        "--bounce-marker-max",
        type=int,
        default=1200,
        help="Maximum number of active bounce markers to keep.",
    )
    parser.add_argument(
        "--bounce-marker-ttl",
        type=int,
        default=8,
        help="Marker lifetime in simulation loops (live mode only).",
    )
    parser.add_argument(
        "--bounce-contact-eps",
        type=float,
        default=0.012,
        help="Ground-contact epsilon above ball radius used for bounce detection.",
    )
    parser.add_argument("--show-collision-overlays", action="store_true", help="Show court collision helper geoms.")
    parser.add_argument("--disable-mode-colors", action="store_true", help="Do not color balls by easy/medium/hard mode.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
