import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from humanoid_tennis.assets import get_robot_cfg
from humanoid_tennis.utils.motion import MotionDataset


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


@dataclass(frozen=True)
class _ReplaySource:
    kind: str  # "npz" | "memmap"
    path: Path
    segment_idx: int | None = None
    label: str = ""


def _load_motion_npz(npz_path: str):
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


def _is_memmap_motion_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta_motion.json").is_file() and (path / "_tensordict").is_dir()


def _load_memmap_index(mem_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    meta_path = mem_path / "meta_motion.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    starts = np.asarray(meta.get("starts", []), dtype=np.int64)
    ends = np.asarray(meta.get("ends", []), dtype=np.int64)
    if starts.size == 0 or ends.size == 0 or starts.shape != ends.shape:
        raise ValueError(f"Invalid starts/ends in {meta_path}")
    id_labels: list[dict] = []
    label_path = mem_path / "id_label.json"
    if label_path.is_file():
        try:
            with label_path.open("r", encoding="utf-8") as f:
                id_labels = list(json.load(f))
        except Exception:
            id_labels = []
    return starts, ends, id_labels


def _build_memmap_source_label(mem_path: Path, segment_idx: int, id_labels: list[dict]) -> str:
    if 0 <= segment_idx < len(id_labels):
        item = id_labels[segment_idx]
        src = str(item.get("source_path", ""))
        seg_s = item.get("segment_start", None)
        seg_e = item.get("segment_end", None)
        stem = Path(src).stem if src else f"segment_{segment_idx:05d}"
        if seg_s is not None and seg_e is not None:
            return f"{stem}[{int(seg_s)}:{int(seg_e)}]#{segment_idx:05d}"
        return f"{stem}#{segment_idx:05d}"
    return f"{mem_path.name}/segment_{segment_idx:05d}"


def _resolve_motion_sources(motion_arg: str, recursive: bool) -> tuple[list[_ReplaySource], Path]:
    motion_path = Path(motion_arg).expanduser()
    if not motion_path.is_absolute():
        motion_path = (Path.cwd() / motion_path).resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion path not found: {motion_path}")

    if motion_path.is_file():
        if motion_path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got: {motion_path}")
        return [
            _ReplaySource(kind="npz", path=motion_path, segment_idx=None, label=motion_path.name)
        ], motion_path.parent

    if not motion_path.is_dir():
        raise ValueError(f"Unsupported motion path type: {motion_path}")

    if _is_memmap_motion_dir(motion_path):
        starts, ends, id_labels = _load_memmap_index(motion_path)
        sources: list[_ReplaySource] = []
        for seg_idx in range(int(starts.shape[0])):
            label = _build_memmap_source_label(motion_path, seg_idx, id_labels)
            sources.append(
                _ReplaySource(
                    kind="memmap",
                    path=motion_path,
                    segment_idx=int(seg_idx),
                    label=label,
                )
            )
        if not sources:
            raise RuntimeError(f"No segments found in memmap dataset: {motion_path}")
        return sources, motion_path

    pattern = "**/*.npz" if recursive else "*.npz"
    files = sorted(p for p in motion_path.glob(pattern) if p.is_file())
    if not files:
        raise RuntimeError(
            f"No .npz files found in directory: {motion_path}. "
            "If this is a memmap dataset, ensure it contains meta_motion.json and _tensordict/."
        )
    return [
        _ReplaySource(kind="npz", path=p, segment_idx=None, label=str(p.relative_to(motion_path)))
        for p in files
    ], motion_path


def _load_motion_memmap_segment(
    mem_path: Path,
    *,
    segment_idx: int,
    mem_fps: float,
    mem_cache: dict[str, object],
):
    cache_key = str(mem_path)
    cache = mem_cache.get(cache_key, None)
    if cache is None:
        ds = MotionDataset.create_from_path_lazy(mem_path=str(mem_path), device=torch.device("cpu"))
        starts, ends, _ = _load_memmap_index(mem_path)
        cache = {"ds": ds, "starts": starts, "ends": ends}
        mem_cache[cache_key] = cache

    starts = cache["starts"]
    ends = cache["ends"]
    if segment_idx < 0 or segment_idx >= int(starts.shape[0]):
        raise IndexError(f"segment_idx out of range: {segment_idx} (num_segments={int(starts.shape[0])})")
    s = int(starts[segment_idx])
    e = int(ends[segment_idx])
    if s >= e:
        raise ValueError(f"Empty segment for idx={segment_idx}: start={s}, end={e}")

    ds = cache["ds"]
    root_pos = ds.data.root_pos_w[s:e].to(dtype=torch.float32).cpu().numpy()
    root_quat_w = ds.data.root_quat_w[s:e].to(dtype=torch.float32).cpu().numpy()
    joint_pos = ds.data.joint_pos[s:e].to(dtype=torch.float32).cpu().numpy()

    fps = max(float(mem_fps), 1.0e-6)
    root_lin_vel = _finite_diff(root_pos, dt=1.0 / fps)
    root_ang_vel = _angvel_from_quat_wxyz(root_quat_w, fps=fps)
    joint_vel = _finite_diff(joint_pos, dt=1.0 / fps)

    qpos = np.concatenate([root_pos, root_quat_w, joint_pos], axis=-1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=-1).astype(np.float32)
    return qpos, qvel, fps


def _prepare_clip(
    source: _ReplaySource,
    *,
    start: int,
    end: int | None,
    speed: float,
    mem_fps: float,
    mem_cache: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, float, float, int, int, int]:
    if source.kind == "npz":
        qpos, qvel, fps = _load_motion_npz(str(source.path))
    elif source.kind == "memmap":
        if source.segment_idx is None:
            raise ValueError(f"memmap source missing segment_idx: {source}")
        qpos, qvel, fps = _load_motion_memmap_segment(
            source.path,
            segment_idx=int(source.segment_idx),
            mem_fps=float(mem_fps),
            mem_cache=mem_cache,
        )
    else:
        raise ValueError(f"Unsupported source kind: {source.kind}")

    total = qpos.shape[0]
    start_idx = max(0, int(start))
    end_idx = total if end is None else min(total, int(end))
    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid frame range for {source.label or source.path.name}: "
            f"start={start_idx}, end={end_idx}, total={total}"
        )

    qpos_clip = qpos[start_idx:end_idx]
    qvel_clip = qvel[start_idx:end_idx]
    frame_dt = (1.0 / max(float(fps), 1.0e-6)) / float(speed)
    return qpos_clip, qvel_clip, float(fps), float(frame_dt), start_idx, end_idx, total


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
    parser = argparse.ArgumentParser(description="Replay motion (npz or memmap dataset) with Viser.")
    parser.add_argument("motion", type=str, help="Path to a motion npz file, npz directory, or memmap dataset directory.")
    parser.add_argument("--robot", type=str, default="g1_col_full_self", help="Robot config name.")
    parser.add_argument("--device", type=str, default=None, help="Simulation device, e.g. cuda:0 or cpu.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop playback.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument("--physics-dt", type=float, default=0.0005, help="MuJoCo physics dt.")
    parser.add_argument(
        "--viewer-max-fps",
        type=float,
        default=45.0,
        help="Throttle Viser updates to at most this FPS. <=0 disables throttling.",
    )
    parser.add_argument("--no-plane", action="store_true", default=False, help="Disable ground plane.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="When motion is a directory, recursively scan subdirectories for *.npz.",
    )
    parser.add_argument(
        "--memmap-fps",
        type=float,
        default=50.0,
        help="FPS to use when replaying memmap dataset segments (meta has no fps).",
    )
    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0.")

    motion_sources, motion_root = _resolve_motion_sources(args.motion, recursive=bool(args.recursive))
    if len(motion_sources) > 1:
        print(f"[INFO] Directory mode: found {len(motion_sources)} clips under {motion_root}")

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
    viewer, viser_scene = _create_viewer(sim)

    mem_cache: dict[str, object] = {}
    motion_labels = []
    label_to_index: dict[str, int] = {}
    for idx, src in enumerate(motion_sources):
        label = src.label
        if len(label) == 0:
            label = src.path.name
        if label in label_to_index:
            label = f"{label}#{idx}"
        motion_labels.append(label)
        label_to_index[label] = idx

    current_idx = -1
    qpos = None
    qvel = None
    fps = 0.0
    frame_dt = 0.0
    steps = 0
    start_idx = 0
    end_idx = 0
    hold_last_frame = False
    frame_idx = 0
    start_time = time.perf_counter()
    viewer_step_interval = 1

    def _load_motion_by_index(target_idx: int, reason: str) -> bool:
        nonlocal current_idx, qpos, qvel, fps, frame_dt, steps, start_idx, end_idx, frame_idx, start_time, hold_last_frame, viewer_step_interval
        src = motion_sources[target_idx]
        try:
            qpos_new, qvel_new, fps_new, frame_dt_new, start_new, end_new, _total_new = _prepare_clip(
                src,
                start=args.start,
                end=args.end,
                speed=float(args.speed),
                mem_fps=float(args.memmap_fps),
                mem_cache=mem_cache,
            )
        except Exception as exc:
            print(f"[WARN] Failed to load {src.path}: {exc}")
            return False

        if qpos_new.shape[-1] != sim_nq or qvel_new.shape[-1] != sim_nv:
            print(
                f"[WARN] Skip {src.label or src.path.name}: qpos/qvel dim "
                f"({qpos_new.shape[-1]}/{qvel_new.shape[-1]}) != sim ({sim_nq}/{sim_nv})."
            )
            return False

        qpos = qpos_new
        qvel = qvel_new
        fps = fps_new
        frame_dt = frame_dt_new
        if float(args.viewer_max_fps) > 0.0:
            viewer_step_interval = max(
                1, int(round(1.0 / (float(args.viewer_max_fps) * max(frame_dt, 1.0e-6))))
            )
        else:
            viewer_step_interval = 1
        steps = qpos.shape[0]
        start_idx = start_new
        end_idx = end_new
        frame_idx = 0
        start_time = time.perf_counter()
        hold_last_frame = False
        current_idx = target_idx
        print(
            f"[INFO] Loaded ({reason}): {src.label or src.path} | frames={steps}, fps={fps:.3f}, "
            f"range=[{start_idx}, {end_idx}), robot={args.robot}, device={device}, speed={args.speed}"
        )
        print(
            f"[INFO] viewer throttle: max_fps={float(args.viewer_max_fps):.1f}, "
            f"frame_dt={frame_dt:.6f}, step_interval={viewer_step_interval}"
        )
        return True

    loaded = False
    for idx in range(len(motion_sources)):
        if _load_motion_by_index(idx, reason="init"):
            loaded = True
            break
    if not loaded:
        raise RuntimeError("No playable motion found (all files/clips failed to load or dimension mismatch).")

    motion_dropdown = None
    if len(motion_labels) > 1:
        motion_dropdown = viewer.gui.add_dropdown(
            "Motion File",
            options=motion_labels,
            initial_value=motion_labels[current_idx],
            hint="Select clip to replay.",
        )

    print("Press Ctrl+C to exit.")

    try:
        while True:
            if motion_dropdown is not None:
                selected = str(motion_dropdown.value)
                target_idx = label_to_index.get(selected, current_idx)
                if target_idx != current_idx:
                    if _load_motion_by_index(target_idx, reason="gui"):
                        pass
                    else:
                        motion_dropdown.value = motion_labels[current_idx]

            should_render = hold_last_frame or ((frame_idx % viewer_step_interval) == 0)
            if should_render:
                qpos_t = torch.as_tensor(qpos[frame_idx : frame_idx + 1], device=device, dtype=torch.float32)
                qvel_t = torch.as_tensor(qvel[frame_idx : frame_idx + 1], device=device, dtype=torch.float32)

                sim.data.qpos[:] = qpos_t
                sim.data.qvel[:] = qvel_t
                sim.forward()
                scene.update(float(args.physics_dt))
                viser_scene.update(_to_cpu_wp_data(sim.data))

            if hold_last_frame:
                time.sleep(0.03)
                continue

            frame_idx += 1
            if frame_idx >= steps:
                if args.loop:
                    frame_idx = 0
                    start_time = time.perf_counter()
                elif motion_dropdown is not None:
                    frame_idx = max(steps - 1, 0)
                    hold_last_frame = True
                    print(
                        f"[INFO] Reached end of {motion_sources[current_idx].label or motion_sources[current_idx].path.name}. "
                        "Use 'Motion File' dropdown to switch."
                    )
                else:
                    break

            if not hold_last_frame:
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
