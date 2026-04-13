import argparse
import time
from types import SimpleNamespace

import numpy as np
import torch

from active_adaptation.assets import get_robot_cfg


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


def _build_scene(*, num_envs, device, robot_name, physics_dt, env_spacing, add_plane):
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.sim.sim import Simulation
    from mjlab.terrains import TerrainEntityCfg

    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    if add_plane:
        scene_cfg.terrain = TerrainEntityCfg(
            terrain_type="plane",
            env_spacing=env_spacing,
            num_envs=num_envs,
        )
    scene_cfg.entities["robot"] = get_robot_cfg(robot_name)

    scene = Scene(scene_cfg, device=device)
    sim = Simulation(
        num_envs=scene.num_envs,
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
    return scene, sim


def _create_viewer(sim, num_envs):
    import viser
    from mjlab.viewer.viser.scene import ViserMujocoScene

    viewer = viser.ViserServer(label="gmt-train-record")
    viser_scene = ViserMujocoScene.create(
        server=viewer,
        mj_model=sim.mj_model,
        num_envs=num_envs,
    )
    viser_scene.create_visualization_gui()
    viser_scene.debug_visualization_enabled = False
    return viewer, viser_scene


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=str, help="Path to train record npz file.")
    parser.add_argument("--robot", type=str, default=None, help="Override robot name.")
    parser.add_argument("--device", type=str, default=None, help="Simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--step-dt", type=float, default=None, help="Override playback frame dt.")
    parser.add_argument("--physics-dt", type=float, default=None, help="Override mujoco physics dt.")
    parser.add_argument("--env-spacing", type=float, default=None, help="Override scene env spacing.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop playback.")
    parser.add_argument("--no-plane", action="store_true", default=False, help="Disable ground plane.")
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0.")

    with np.load(args.record, allow_pickle=False) as npz:
        qpos = npz["qpos"]
        qvel = npz["qvel"]
        if qpos.ndim != 3 or qvel.ndim != 3:
            raise ValueError("Expected qpos/qvel with shape [steps, envs, dim].")
        if qpos.shape[:2] != qvel.shape[:2]:
            raise ValueError("qpos and qvel shape mismatch.")

        steps, num_envs = qpos.shape[:2]
        robot_name = args.robot or _read_str(npz, "robot_name", "g1_col_full_self")
        step_dt = args.step_dt or float(_read_scalar(npz, "step_dt", 0.02))
        physics_dt = args.physics_dt or float(_read_scalar(npz, "physics_dt", 0.0025))
        env_spacing = args.env_spacing or float(_read_scalar(npz, "env_spacing", 2.5))

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    scene, sim = _build_scene(
        num_envs=num_envs,
        device=device,
        robot_name=robot_name,
        physics_dt=physics_dt,
        env_spacing=env_spacing,
        add_plane=not args.no_plane,
    )
    sim_nq = sim.data.qpos.shape[-1]
    sim_nv = sim.data.qvel.shape[-1]
    if qpos.shape[-1] != sim_nq or qvel.shape[-1] != sim_nv:
        raise ValueError(
            f"Recorded qpos/qvel dim ({qpos.shape[-1]}/{qvel.shape[-1]}) "
            f"does not match scene dim ({sim_nq}/{sim_nv})."
        )
    viewer, viser_scene = _create_viewer(sim, num_envs)

    print(
        f"Replay loaded: steps={steps}, envs={num_envs}, robot={robot_name}, "
        f"device={device}, step_dt={step_dt:.4f}, physics_dt={physics_dt:.4f}"
    )
    print("Press Ctrl+C to exit.")

    frame_dt = float(step_dt) / float(args.speed)
    frame_idx = 0
    start_time = time.perf_counter()
    try:
        while True:
            qpos_t = torch.as_tensor(qpos[frame_idx], device=device, dtype=torch.float32)
            qvel_t = torch.as_tensor(qvel[frame_idx], device=device, dtype=torch.float32)

            sim.data.qpos[:] = qpos_t
            sim.data.qvel[:] = qvel_t
            sim.forward()
            scene.update(physics_dt)
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
