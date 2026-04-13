import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.assets import get_robot_cfg


def _read_scalar(npz, key: str, default: float) -> float:
    if key not in npz:
        return default
    value = np.asarray(npz[key])
    if value.size == 0:
        return default
    return float(value.reshape(-1)[0])


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., 0:1]], axis=-1)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] <= 1:
        return np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x, dtype=np.float32)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v.astype(np.float32)


def _angvel_from_quat_wxyz(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    q_xyzw = _wxyz_to_xyzw(np.asarray(quat_wxyz, dtype=np.float32))
    rot = sRot.from_quat(q_xyzw)
    dt = 1.0 / max(float(fps), 1.0e-6)
    n = q_xyzw.shape[0]
    if n <= 1:
        return np.zeros((n, 3), dtype=np.float32)
    if n == 2:
        rel = rot[1] * rot[0].inv()
        w = (rel.as_rotvec() / dt).astype(np.float32)
        return np.stack([w, w], axis=0)
    rel_mid = rot[2:] * rot[:-2].inv()
    w_mid = (rel_mid.as_rotvec() / (2.0 * dt)).astype(np.float32)
    w = np.zeros((n, 3), dtype=np.float32)
    w[1:-1] = w_mid
    w[0] = ((rot[1] * rot[0].inv()).as_rotvec() / dt).astype(np.float32)
    w[-1] = ((rot[-1] * rot[-2].inv()).as_rotvec() / dt).astype(np.float32)
    return w


def _load_motion(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.files)

        # Format A: mjlab-exported tennis npz.
        if {"body_pos_w", "body_quat_w", "joint_pos"}.issubset(keys):
            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            fps = _read_scalar(data, "fps", 50.0)

            root_pos = body_pos_w[:, 0, :]
            root_quat_w = body_quat_w[:, 0, :]
            joint_vel = (
                np.asarray(data["joint_vel"], dtype=np.float32)
                if "joint_vel" in keys
                else _finite_diff(joint_pos, dt=1.0 / max(fps, 1.0e-6))
            )

        # Format B: motion_tracking training-format npz.
        elif {"root_pos", "root_rot", "dof_pos"}.issubset(keys):
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)
            root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
            joint_pos = np.asarray(data["dof_pos"], dtype=np.float32)
            fps = _read_scalar(data, "fps", 50.0)

            root_quat_w = _xyzw_to_wxyz(root_rot_xyzw).astype(np.float32)
            joint_vel = _finite_diff(joint_pos, dt=1.0 / max(fps, 1.0e-6))

        else:
            raise ValueError(
                f"Unsupported npz format for {npz_path}. Keys={sorted(keys)}"
            )

    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"Unexpected root_pos shape {root_pos.shape}")
    if root_quat_w.ndim != 2 or root_quat_w.shape[-1] != 4:
        raise ValueError(f"Unexpected root_quat shape {root_quat_w.shape}")
    if joint_pos.ndim != 2:
        raise ValueError(f"Unexpected joint_pos shape {joint_pos.shape}")
    if root_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError(
            f"Frame mismatch root={root_pos.shape[0]} joints={joint_pos.shape[0]}"
        )

    fps = max(float(fps), 1.0e-6)
    root_lin_vel = _finite_diff(root_pos, dt=1.0 / fps)
    root_ang_vel = _angvel_from_quat_wxyz(root_quat_w, fps=fps)

    qpos = np.concatenate([root_pos, root_quat_w, joint_pos], axis=-1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1).astype(np.float32)
    return qpos, qvel, fps


def _build_scene(*, device: str, robot_name: str, physics_dt: float, add_plane: bool):
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.sim.sim import Simulation
    from mjlab.terrains import TerrainEntityCfg

    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.5)
    if add_plane:
        scene_cfg.terrain = TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=2.5,
            num_envs=1,
        )
    scene_cfg.entities["robot"] = get_robot_cfg(robot_name)

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
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    return scene, sim


def _create_viewer(sim):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    viewer = viser.ViserServer(label="motion-tracking-motion-replay")
    viser_scene = ViserMujocoScene.create(server=viewer, mj_model=sim.mj_model, num_envs=1)
    viser_scene.create_visualization_gui()
    viser_scene.debug_visualization_enabled = False
    return viewer, viser_scene


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


def main():
    parser = argparse.ArgumentParser(description="Replay motion npz with Viser.")
    parser.add_argument("motion", type=str, help="Path to motion npz file.")
    parser.add_argument("--robot", type=str, default="g1_col_full_self", help="Robot config name.")
    parser.add_argument("--device", type=str, default=None, help="Simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop playback.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument("--physics-dt", type=float, default=0.0025, help="MuJoCo physics dt.")
    parser.add_argument("--no-plane", action="store_true", default=False, help="Disable ground plane.")
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0.")

    qpos, qvel, fps = _load_motion(args.motion)
    total = qpos.shape[0]
    start = max(0, int(args.start))
    end = total if args.end is None else min(total, int(args.end))
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total}")

    qpos = qpos[start:end]
    qvel = qvel[start:end]
    steps = qpos.shape[0]
    frame_dt = (1.0 / fps) / float(args.speed)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    scene, sim = _build_scene(
        device=device,
        robot_name=args.robot,
        physics_dt=float(args.physics_dt),
        add_plane=not args.no_plane,
    )
    sim_nq = int(sim.data.qpos.shape[-1])
    sim_nv = int(sim.data.qvel.shape[-1])
    if qpos.shape[-1] != sim_nq or qvel.shape[-1] != sim_nv:
        raise ValueError(
            f"Motion qpos/qvel dim ({qpos.shape[-1]}/{qvel.shape[-1]}) does not match robot sim dim ({sim_nq}/{sim_nv})."
        )

    viewer, viser_scene = _create_viewer(sim)

    print(
        f"Replay loaded: frames={steps}, fps={fps:.3f}, range=[{start}, {end}), "
        f"robot={args.robot}, device={device}, speed={args.speed}"
    )
    print("Press Ctrl+C to exit.")

    frame_idx = 0
    start_time = time.perf_counter()
    try:
        while True:
            qpos_t = torch.as_tensor(qpos[frame_idx : frame_idx + 1], device=device, dtype=torch.float32)
            qvel_t = torch.as_tensor(qvel[frame_idx : frame_idx + 1], device=device, dtype=torch.float32)

            sim.data.qpos[:] = qpos_t
            sim.data.qvel[:] = qvel_t
            sim.forward()
            scene.update(float(args.physics_dt))
            viser_scene.update(_to_cpu_wp_data(sim.data))

            frame_idx += 1
            if frame_idx >= steps:
                if args.loop:
                    frame_idx = 0
                    start_time = time.perf_counter()
                else:
                    break

            target_time = start_time + frame_idx * frame_dt
            delay = target_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop()


if __name__ == "__main__":
    main()
