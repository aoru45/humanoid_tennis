#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg

from active_adaptation.assets import get_tennis_ball_cfg, get_tennis_court_cfg

BALL_RADIUS_M = 0.0335
MODE_PRESETS = {
    "single": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "target_bounce_x_range": [-4.0, 4.0],
        "target_bounce_y_range": [-12.0, -7.0],
        "flight_time_range": [0.70, 1.00],
    },
    "easy": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "target_bounce_x_range": [-1.2, 1.6],
        "target_bounce_y_range": [-10.2, -7.4],
        "flight_time_range": [0.70, 1.00],
    },
    "medium": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "target_bounce_x_range": [-2.8, 2.8],
        "target_bounce_y_range": [-11.2, -5.8],
        "flight_time_range": [0.70, 1.00],
    },
    "hard": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "target_bounce_x_range": [-3.8, 3.8],
        "target_bounce_y_range": [-11.5, -3.5],
        "flight_time_range": [0.70, 1.00],
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _apply_training_sim_defaults(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    cfg_path = _repo_root() / "cfg/task/G1/G1_tennis_highlevel.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    sim = cfg.get("sim", {})

    args.physics_dt = float(sim.get("mujoco_physics_dt", args.physics_dt))
    args.iterations = int(sim.get("mujoco_iterations", args.iterations))
    args.ls_iterations = int(sim.get("mujoco_ls_iterations", args.ls_iterations))
    args.ccd_iterations = int(sim.get("mujoco_ccd_iterations", args.ccd_iterations))
    args.nconmax = int(sim.get("nconmax", args.nconmax))
    args.njmax = int(sim.get("njmax", args.njmax))


def _load_training_tennis_defaults() -> dict[str, object]:
    from omegaconf import OmegaConf

    cfg_path = _repo_root() / "cfg/task/G1/G1_tennis_highlevel.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    tennis = cfg.get("tennis", {})
    return {
        "court_texture": str(tennis.get("court_texture", "green")),
        "net_height": float(tennis.get("net_height", 0.914)),
        "net_collision_half_thickness": float(tennis.get("net_collision_half_thickness", 0.06)),
        "enable_racket_court_collision": bool(tennis.get("racket_court_collision", False)),
    }


@dataclass
class LaunchCase:
    launch_pos: np.ndarray
    launch_vel: np.ndarray
    launch_ang: np.ndarray
    speed: float
    azimuth_deg: float
    elevation_deg: float


@dataclass
class BounceResult:
    bounce_count: int
    first_rebound_height_m: float | None


@dataclass
class RunningStats:
    samples: int = 0
    bounce_sum: float = 0.0
    bounce_max: int = 0
    rebound_n: int = 0
    rebound_sum: float = 0.0
    rebound_sq_sum: float = 0.0
    rebound_min: float = float("inf")
    rebound_max: float = float("-inf")

    def update(self, results: list[BounceResult]) -> None:
        if not results:
            return
        self.samples += len(results)
        for r in results:
            self.bounce_sum += float(r.bounce_count)
            self.bounce_max = max(self.bounce_max, int(r.bounce_count))
            if r.first_rebound_height_m is not None:
                x = float(r.first_rebound_height_m)
                self.rebound_n += 1
                self.rebound_sum += x
                self.rebound_sq_sum += x * x
                self.rebound_min = min(self.rebound_min, x)
                self.rebound_max = max(self.rebound_max, x)

    def print_summary(self) -> None:
        if self.samples <= 0:
            print("[SUMMARY] samples=0")
            return
        bounce_mean = self.bounce_sum / float(self.samples)
        print(
            "[SUMMARY]",
            f"samples={self.samples}",
            f"bounce_count_mean={bounce_mean:.2f}",
            f"bounce_count_max={self.bounce_max}",
        )
        if self.rebound_n > 0:
            mean = self.rebound_sum / float(self.rebound_n)
            var = max(self.rebound_sq_sum / float(self.rebound_n) - mean * mean, 0.0)
            std = math.sqrt(var)
            print(
                "[SUMMARY] first_rebound_height_m:",
                f"mean={mean:.3f}",
                f"std={std:.3f}",
                f"min={self.rebound_min:.3f}",
                f"max={self.rebound_max:.3f}",
                f"n={self.rebound_n}",
            )
        else:
            print("[SUMMARY] No valid rebound height extracted.")


def _desired_y_sign() -> float:
    # Keep launch direction fixed (+Y -> -Y) for consistent stress tests.
    return -1.0


def _to_cpu_wp_data(wp_data):
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


def _align_tennis_court_to_env_origins(scene: Scene, *, device: str, env_ids: torch.Tensor | None = None) -> None:
    if "tennis_court" not in scene.entities:
        return
    court = scene["tennis_court"]
    if not getattr(court, "is_mocap", False):
        return
    if env_ids is None:
        env_ids = torch.arange(scene.num_envs, device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    pose = torch.zeros((env_ids.numel(), 7), device=device, dtype=torch.float32)
    pose[:, :3] = scene.env_origins[env_ids]
    pose[:, 3] = 1.0
    court.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def _build_mjlab_scene(
    *,
    device: str,
    num_envs: int,
    env_spacing: float,
    physics_dt: float,
    iterations: int,
    ls_iterations: int,
    ccd_iterations: int,
    nconmax: int,
    njmax: int,
    court_texture: str,
    net_height: float,
    net_collision_half_thickness: float,
    enable_racket_court_collision: bool,
):
    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=float(env_spacing))
    scene_cfg.terrain = TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=float(env_spacing),
        num_envs=num_envs,
    )
    scene_cfg.entities["tennis_ball"] = get_tennis_ball_cfg()
    scene_cfg.entities["tennis_court"] = get_tennis_court_cfg(
        texture=str(court_texture),
        net_height=float(net_height),
        net_collision_half_thickness=float(net_collision_half_thickness),
        enable_racket_court_collision=bool(enable_racket_court_collision),
    )

    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=num_envs,
        cfg=SimulationCfg(
            nconmax=int(nconmax),
            njmax=int(njmax),
            mujoco=MujocoCfg(
                timestep=float(physics_dt),
                iterations=int(iterations),
                ls_iterations=int(ls_iterations),
                ccd_iterations=int(ccd_iterations),
                multiccd=False,
            ),
        ),
        model=scene.compile(),
        device=device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    if not hasattr(scene, "env_origins") and hasattr(scene, "env_offsets"):
        scene.env_origins = scene.env_offsets

    _align_tennis_court_to_env_origins(scene, device=device)
    scene.write_data_to_sim()
    sim.forward()
    sim.sense()
    scene.update(float(physics_dt))

    ball = scene["tennis_ball"]
    court_gid = sim.mj_model.geom("tennis_court/tennis_court_ball_collision").id
    ball_gid = sim.mj_model.geom("tennis_ball/tennis_ball_geom").id
    court_top_z = float(sim.mj_model.geom_pos[court_gid, 2] + sim.mj_model.geom_size[court_gid, 2])
    ball_radius = float(sim.mj_model.geom_size[ball_gid, 0])
    contact_center_z_local = court_top_z + ball_radius

    return scene, sim, ball, contact_center_z_local


def _set_ball_states(
    *,
    scene: Scene,
    sim: Simulation,
    ball,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    launch_cases: list[LaunchCase],
    device: str,
    physics_dt: float,
) -> None:
    n = env_ids.numel()
    state = torch.zeros((n, 13), device=device, dtype=torch.float32)
    launch_pos_local = np.stack([c.launch_pos for c in launch_cases], axis=0)
    launch_vel = np.stack([c.launch_vel for c in launch_cases], axis=0)
    launch_ang = np.stack([c.launch_ang for c in launch_cases], axis=0)

    state[:, :3] = torch.as_tensor(launch_pos_local, device=device, dtype=torch.float32) + env_origins[env_ids]
    state[:, 3] = 1.0
    state[:, 7:10] = torch.as_tensor(launch_vel, device=device, dtype=torch.float32)
    state[:, 10:13] = torch.as_tensor(launch_ang, device=device, dtype=torch.float32)

    ball.write_root_state_to_sim(state, env_ids=env_ids)
    scene.write_data_to_sim()
    sim.forward()
    sim.sense()
    scene.update(float(physics_dt))


def _sample_case_angle(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> LaunchCase:
    desired_sign = _desired_y_sign()
    for _ in range(int(args.sample_attempts)):
        x = rng.uniform(args.launch_x_range[0], args.launch_x_range[1])
        y = rng.uniform(args.launch_y_range[0], args.launch_y_range[1])
        z = rng.uniform(args.launch_z_range[0], args.launch_z_range[1])
        speed = float(rng.uniform(args.speed_range[0], args.speed_range[1]))
        az_deg = float(rng.uniform(args.azimuth_range_deg[0], args.azimuth_range_deg[1]))
        el_deg = float(rng.uniform(args.elevation_range_deg[0], args.elevation_range_deg[1]))
        az = math.radians(az_deg)
        el = math.radians(el_deg)

        horiz = speed * math.cos(el)
        vx = horiz * math.cos(az)
        vy = horiz * math.sin(az)
        vz = speed * math.sin(el)
        if vy * desired_sign <= 0.0:
            continue

        spin_rps = rng.uniform(args.spin_rps_range[0], args.spin_rps_range[1], size=3)
        spin_rad_s = spin_rps * (2.0 * math.pi)
        return LaunchCase(
            launch_pos=np.array([x, y, z], dtype=np.float64),
            launch_vel=np.array([vx, vy, vz], dtype=np.float64),
            launch_ang=spin_rad_s.astype(np.float64),
            speed=speed,
            azimuth_deg=az_deg,
            elevation_deg=el_deg,
        )

    raise RuntimeError("Failed to sample angle-launch with requested Y direction.")


def _sample_case_target(
    rng: np.random.Generator,
    args: argparse.Namespace,
    *,
    contact_center_z: float,
    gravity_z: float,
) -> LaunchCase:
    desired_sign = _desired_y_sign()
    for _ in range(int(args.sample_attempts)):
        x = rng.uniform(args.launch_x_range[0], args.launch_x_range[1])
        y = rng.uniform(args.launch_y_range[0], args.launch_y_range[1])
        z = rng.uniform(args.launch_z_range[0], args.launch_z_range[1])
        tx = rng.uniform(args.target_bounce_x_range[0], args.target_bounce_x_range[1])
        ty = rng.uniform(args.target_bounce_y_range[0], args.target_bounce_y_range[1])
        t = rng.uniform(args.flight_time_range[0], args.flight_time_range[1])
        if t <= 1.0e-4:
            continue

        vx = (tx - x) / t
        vy = (ty - y) / t
        vz = (contact_center_z - z - 0.5 * gravity_z * (t * t)) / t
        if vy * desired_sign <= 0.0:
            continue

        speed = float(np.linalg.norm(np.array([vx, vy, vz], dtype=np.float64)))
        if speed < float(args.speed_range[0]) or speed > float(args.speed_range[1]):
            continue

        az_deg = math.degrees(math.atan2(vy, vx))
        el_deg = math.degrees(math.atan2(vz, max(math.hypot(vx, vy), 1.0e-8)))
        spin_rps = rng.uniform(args.spin_rps_range[0], args.spin_rps_range[1], size=3)
        spin_rad_s = spin_rps * (2.0 * math.pi)
        return LaunchCase(
            launch_pos=np.array([x, y, z], dtype=np.float64),
            launch_vel=np.array([vx, vy, vz], dtype=np.float64),
            launch_ang=spin_rad_s.astype(np.float64),
            speed=speed,
            azimuth_deg=float(az_deg),
            elevation_deg=float(el_deg),
        )

    raise RuntimeError(
        "Failed to sample a valid launch in sample_attempts. "
        "Relax speed/target/time ranges."
    )


def _create_viser_viewer(sim: Simulation, *, num_envs: int):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    viewer = viser.ViserServer(label="tennis-court-bounce")
    viser_scene = ViserMujocoScene.create(server=viewer, mj_model=sim.mj_model, num_envs=int(num_envs))
    viser_scene.create_visualization_gui()
    viser_scene.debug_visualization_enabled = False
    viser_scene.camera_tracking_enabled = False
    return viewer, viser_scene


def _hide_court_overlay_geoms(sim: Simulation) -> None:
    """Hide tennis-court collision overlay geoms for clear textured rendering."""
    import mujoco

    model = sim.mj_model
    hidden = 0
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        if not name.startswith("tennis_court/"):
            continue
        # Keep visual court/net geoms; hide collision helper layers that overlay the texture.
        if name.endswith("_collision"):
            model.geom_group[geom_id] = 3
            hidden += 1
    if hidden > 0:
        print(f"[INFO] Hidden {hidden} tennis-court collision overlay geoms for clearer visualization.")


def _run_wave(
    *,
    scene: Scene,
    sim: Simulation,
    ball,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    contact_center_z_local: float,
    launch_cases: list[LaunchCase],
    physics_dt: float,
    max_time_s: float,
    device: str,
    viser_scene=None,
    viewer_step_interval: int = 1,
    realtime_scale: float = 1.0,
) -> list[BounceResult]:
    _set_ball_states(
        scene=scene,
        sim=sim,
        ball=ball,
        env_ids=env_ids,
        env_origins=env_origins,
        launch_cases=launch_cases,
        device=device,
        physics_dt=physics_dt,
    )

    dt = float(physics_dt)
    max_steps = max(1, int(math.ceil(float(max_time_s) / dt)))
    num_envs = int(env_ids.numel())

    z_hist = np.zeros((max_steps, num_envs), dtype=np.float64)
    prev_contact = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    bounce_steps: list[list[int]] = [[] for _ in range(num_envs)]

    z_contact_eps = float(contact_center_z_local + 1.0e-3)
    for step in range(max_steps):
        z_local = ball.data.root_link_pos_w[env_ids, 2] - env_origins[env_ids, 2]
        z_hist[step] = z_local.detach().cpu().numpy()

        contact_now = z_local <= z_contact_eps
        rising = torch.logical_and(contact_now, torch.logical_not(prev_contact))
        hit_ids = torch.nonzero(rising, as_tuple=False).squeeze(-1).tolist()
        for bi in hit_ids:
            bounce_steps[int(bi)].append(step)
        prev_contact = contact_now

        if viser_scene is not None and (
            (step % int(max(1, viewer_step_interval)) == 0) or (step == max_steps - 1)
        ):
            viser_scene.update(_to_cpu_wp_data(sim.data))

        sim.step()
        scene.update(dt)

        if realtime_scale > 0.0:
            time.sleep(dt * realtime_scale)

    results: list[BounceResult] = []
    for bi in range(num_envs):
        events = bounce_steps[bi]
        first_rebound_h = None
        if len(events) >= 1:
            start = min(events[0] + 1, max_steps - 1)
            end = (events[1] - 1) if len(events) >= 2 else (max_steps - 1)
            if end >= start:
                peak_z = float(np.max(z_hist[start : end + 1, bi]))
                first_rebound_h = peak_z - float(contact_center_z_local)
        results.append(BounceResult(bounce_count=len(events), first_rebound_height_m=first_rebound_h))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MJLab tennis-court bounce visualization with concurrent random launches "
            "(same runtime chain as training: mjlab + mujoco_warp)."
        )
    )
    parser.add_argument("--num-cases", type=int, default=64, help="Concurrent balls (mapped to num_envs).")
    parser.add_argument(
        "--num-waves",
        type=int,
        default=0,
        help="How many random launch waves to run. <=0 means infinite until Ctrl+C.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--use-training-cfg",
        action="store_true",
        help="Force sim params from cfg/task/G1/G1_tennis_highlevel.yaml.",
    )
    parser.add_argument("--physics-dt", type=float, default=0.0005)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--ls-iterations", type=int, default=48)
    parser.add_argument("--ccd-iterations", type=int, default=50)
    parser.add_argument("--nconmax", type=int, default=600)
    parser.add_argument("--njmax", type=int, default=2000)
    parser.add_argument("--env-spacing", type=float, default=30.0)
    parser.add_argument("--max-time-s", type=float, default=2.5)
    parser.add_argument("--realtime-scale", type=float, default=1.0)
    parser.add_argument(
        "--viewer-max-fps",
        type=float,
        default=45.0,
        help="Throttle Viser updates to at most this FPS. <=0 disables throttling.",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Exit immediately after summary (only for finite num-waves).",
    )
    parser.add_argument("--pause-between-waves-s", type=float, default=0.4)
    parser.add_argument("--sampling", type=str, choices=["target", "angle"], default="target")
    parser.add_argument("--mode", type=str, choices=["single", "easy", "medium", "hard"], default="single")
    parser.add_argument("--sample-attempts", type=int, default=200)

    # Ranges align to generate_traj.sh presets by mode; explicit CLI ranges can override.
    parser.add_argument("--speed-range", type=float, nargs=2, default=None)
    parser.add_argument("--azimuth-range-deg", type=float, nargs=2, default=[-125.0, -55.0])
    parser.add_argument("--elevation-range-deg", type=float, nargs=2, default=[-8.0, 18.0])
    parser.add_argument("--spin-rps-range", type=float, nargs=2, default=[-8.0, 8.0])
    parser.add_argument("--launch-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--launch-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--launch-z-range", type=float, nargs=2, default=[1.2, 2.8])
    parser.add_argument("--target-bounce-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--target-bounce-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--flight-time-range", type=float, nargs=2, default=None)
    args = parser.parse_args()
    tennis_defaults = _load_training_tennis_defaults()

    if bool(args.use_training_cfg):
        _apply_training_sim_defaults(args)

    preset = MODE_PRESETS[str(args.mode)]
    args.speed_range = list(args.speed_range) if args.speed_range is not None else list(preset["speed_range"])
    args.launch_x_range = (
        list(args.launch_x_range) if args.launch_x_range is not None else list(preset["launch_x_range"])
    )
    args.launch_y_range = (
        list(args.launch_y_range) if args.launch_y_range is not None else list(preset["launch_y_range"])
    )
    args.target_bounce_x_range = (
        list(args.target_bounce_x_range)
        if args.target_bounce_x_range is not None
        else list(preset["target_bounce_x_range"])
    )
    args.target_bounce_y_range = (
        list(args.target_bounce_y_range)
        if args.target_bounce_y_range is not None
        else list(preset["target_bounce_y_range"])
    )
    args.flight_time_range = (
        list(args.flight_time_range) if args.flight_time_range is not None else list(preset["flight_time_range"])
    )

    num_envs = max(1, int(args.num_cases))
    num_waves = int(args.num_waves)
    rng = np.random.default_rng(int(args.seed))

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    scene, sim, ball, contact_center_z_local = _build_mjlab_scene(
        device=device,
        num_envs=num_envs,
        env_spacing=float(args.env_spacing),
        physics_dt=float(args.physics_dt),
        iterations=int(args.iterations),
        ls_iterations=int(args.ls_iterations),
        ccd_iterations=int(args.ccd_iterations),
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        court_texture=str(tennis_defaults["court_texture"]),
        net_height=float(tennis_defaults["net_height"]),
        net_collision_half_thickness=float(tennis_defaults["net_collision_half_thickness"]),
        enable_racket_court_collision=bool(tennis_defaults["enable_racket_court_collision"]),
    )
    _hide_court_overlay_geoms(sim)
    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)

    print(
        "[INFO] runtime:",
        "chain=mjlab+mujoco_warp",
        f"device={device}",
        f"dt={float(sim.mj_model.opt.timestep):.4f}s",
        f"iterations={int(sim.mj_model.opt.iterations)}",
        f"ls_iterations={int(sim.mj_model.opt.ls_iterations)}",
        f"ccd_iterations={int(args.ccd_iterations)}",
        f"nconmax={int(args.nconmax)}",
        f"njmax={int(args.njmax)}",
        f"court_texture={str(tennis_defaults['court_texture'])}",
    )
    print(
        "[INFO] launch:",
        f"mode={args.mode}",
        f"sampling={args.sampling}",
        "y_direction=neg",
        f"concurrent_balls={num_envs}",
        f"waves={'inf' if num_waves <= 0 else num_waves}",
        f"speed={tuple(float(x) for x in args.speed_range)} m/s",
    )

    use_viewer = not bool(args.no_viewer)
    viewer = None
    viser_scene = None
    if use_viewer:
        viewer, viser_scene = _create_viser_viewer(sim, num_envs=num_envs)
        print("[INFO] Viser launched. Camera tracking disabled; you can move camera freely.")
    viewer_step_interval = 1
    if use_viewer:
        viewer_max_fps = float(args.viewer_max_fps)
        if viewer_max_fps > 0.0:
            viewer_step_interval = max(1, int(round(1.0 / (viewer_max_fps * float(args.physics_dt)))))
        print(
            "[INFO] viewer throttle:",
            f"max_fps={float(args.viewer_max_fps):.1f}",
            f"step_interval={viewer_step_interval}",
        )

    stats = RunningStats()
    wave_iter = range(num_waves) if num_waves > 0 else itertools.count(0)

    gravity_z = float(sim.mj_model.opt.gravity[2])
    env_origins = scene.env_origins

    try:
        for wave in wave_iter:
            launch_cases: list[LaunchCase] = []
            for _ in range(num_envs):
                if args.sampling == "target":
                    case = _sample_case_target(
                        rng,
                        args,
                        contact_center_z=float(contact_center_z_local),
                        gravity_z=gravity_z,
                    )
                else:
                    case = _sample_case_angle(rng, args)
                launch_cases.append(case)

            results = _run_wave(
                scene=scene,
                sim=sim,
                ball=ball,
                env_ids=env_ids,
                env_origins=env_origins,
                contact_center_z_local=float(contact_center_z_local),
                launch_cases=launch_cases,
                physics_dt=float(args.physics_dt),
                max_time_s=float(args.max_time_s),
                device=device,
                viser_scene=viser_scene,
                viewer_step_interval=viewer_step_interval,
                realtime_scale=float(args.realtime_scale),
            )
            stats.update(results)

            print(f"[wave {wave:02d}] finished {num_envs} concurrent balls")
            for i, (case, result) in enumerate(zip(launch_cases, results)):
                h1 = "NA" if result.first_rebound_height_m is None else f"{result.first_rebound_height_m:.3f}m"
                print(
                    f"  [ball {i:03d}] "
                    f"pos=({case.launch_pos[0]:+.2f},{case.launch_pos[1]:+.2f},{case.launch_pos[2]:+.2f}) "
                    f"vel=({case.launch_vel[0]:+.2f},{case.launch_vel[1]:+.2f},{case.launch_vel[2]:+.2f}) "
                    f"speed={case.speed:.2f} bounces={result.bounce_count} h1={h1}"
                )

            if use_viewer and float(args.pause_between_waves_s) > 0.0:
                time.sleep(float(args.pause_between_waves_s))

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    stats.print_summary()

    if use_viewer and (num_waves > 0) and (not bool(args.auto_close)):
        print("[INFO] Viewer is kept alive. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")


if __name__ == "__main__":
    main()
