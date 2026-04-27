import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
from pathlib import Path

import numpy as np
import torch

from humanoid_tennis.utils.motion import MotionDataset

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def preprocess_motion(motion, foot_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]
    offset_xy = root_pos[0, :2].copy()
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    z_l = motion["xpos"][:, foot_idx[0], 2]
    z_r = motion["xpos"][:, foot_idx[1], 2]

    if not always_on_ground:
        z_min = float(min(z_l.min(), z_r.min()))
        dz = 0.0 - z_min
        motion["qpos"][:, 2] += dz
        motion["xpos"][:, :, 2] += dz
    else:
        z_min = np.min(
            np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
            axis=-1,
            keepdims=True,
        )
        dz = 0.0 - z_min
        motion["qpos"][:, 2] += dz.reshape(-1)
        motion["xpos"][:, :, 2] += dz
    return motion


def none_callback(_ctx, motion):
    motion["metadata"] = None


def check_motion(motion, _foot_idx, _path, _start_idx, _end_idx) -> bool:
    qvel = motion["qvel"]
    qpos = motion["qpos"]
    xpos = motion["xpos"]

    if np.any(np.abs(qvel[:, :6]) > 10):
        print("Invalid motion due to high velocity spike")
        return False
    if qpos.shape[0] < 250:
        print("Invalid motion due to short length")
        return False

    min_body_z = np.min(xpos[:, :, 2], axis=1)
    all_off = min_body_z > 0.2
    fps = int(motion.get("fps", 50))
    if fps <= 0:
        fps = 50
    if np.any(all_off):
        padded = np.concatenate(([0], all_off.astype(np.int8), [0]))
        edges = np.diff(padded)
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        max_run = (run_ends - run_starts).max() if run_starts.size else 0
        if max_run > fps:
            print("Invalid motion due to all bodies off ground > 1s")
            return False
    max_body_z = float(np.max(xpos[:, :, 2]))
    if max_body_z <= 0.2:
        print("Invalid motion due to low max body height")
        return False
    return True


def _convert_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_wxyz[..., 1:], quat_wxyz[..., 0:1]], axis=-1)


def _normalize_quat_wxyz(quat_wxyz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    return quat_wxyz / np.clip(norm, eps, None)


def _quat_conjugate_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    out = quat_wxyz.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]
    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_apply_inverse_wxyz(root_quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # root_quat_wxyz: [T, 4], vec: [T, B, 3]
    qv = root_quat_wxyz[:, None, 1:]  # [T,1,3]
    qw = root_quat_wxyz[:, None, 0:1]  # [T,1,1]
    # Apply inverse rotation by using conjugate quaternion in vector-rotation form.
    t = 2.0 * np.cross(-qv, vec)
    return vec + qw * t + np.cross(-qv, t)


def _extract_scalar_fps(value, default: float = 50.0) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _extract_common_arrays(data, src: Path, output_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = set(data.files)

    # Format A: mjlab-exported format (already supported before).
    if {"body_pos_w", "body_quat_w", "joint_pos"}.issubset(keys):
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        dof_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        if "fps" in keys:
            _ = _extract_scalar_fps(data["fps"], default=output_fps)
        return body_pos_w, body_quat_w, dof_pos

    # Format B: LA/legacy motion dump with qpos/xpos/xquat.
    if {"qpos", "xpos", "xquat"}.issubset(keys):
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        xpos = np.asarray(data["xpos"], dtype=np.float32)
        xquat = np.asarray(data["xquat"], dtype=np.float32)
        if "frequency" in keys:
            _ = _extract_scalar_fps(data["frequency"], default=output_fps)
        elif "fps" in keys:
            _ = _extract_scalar_fps(data["fps"], default=output_fps)

        if qpos.ndim != 2:
            raise ValueError(f"{src}: unexpected qpos shape {qpos.shape}")
        if xpos.ndim != 3 or xpos.shape[-1] != 3:
            raise ValueError(f"{src}: unexpected xpos shape {xpos.shape}")
        if xquat.ndim != 3 or xquat.shape[-1] != 4:
            raise ValueError(f"{src}: unexpected xquat shape {xquat.shape}")
        if xpos.shape[:2] != xquat.shape[:2]:
            raise ValueError(f"{src}: mismatched xpos/xquat shape {xpos.shape} vs {xquat.shape}")
        if xpos.shape[0] != qpos.shape[0]:
            raise ValueError(f"{src}: frame mismatch xpos={xpos.shape[0]} qpos={qpos.shape[0]}")

        if "body_names" in keys:
            body_names = [str(x) for x in np.asarray(data["body_names"]).tolist()]
            body_index = {name: i for i, name in enumerate(body_names)}
            missing_body = [name for name in BODY_NAMES if name not in body_index]
            if missing_body:
                raise KeyError(f"{src}: missing body names: {missing_body}")
            body_cols = [body_index[name] for name in BODY_NAMES]
            body_pos_w = xpos[:, body_cols, :]
            body_quat_w = xquat[:, body_cols, :]
        else:
            # Fallback: world body at index 0, followed by BODY_NAMES in order.
            if xpos.shape[1] >= (len(BODY_NAMES) + 1):
                body_pos_w = xpos[:, 1 : 1 + len(BODY_NAMES), :]
                body_quat_w = xquat[:, 1 : 1 + len(BODY_NAMES), :]
            elif xpos.shape[1] == len(BODY_NAMES):
                body_pos_w = xpos
                body_quat_w = xquat
            else:
                raise ValueError(
                    f"{src}: cannot infer body mapping from xpos shape {xpos.shape}, expected >= {len(BODY_NAMES)} bodies."
                )

        # qpos = [root_pos(3), root_quat(4), dof_pos...]
        if qpos.shape[1] < (7 + len(JOINT_NAMES)):
            raise ValueError(
                f"{src}: unexpected qpos width {qpos.shape[1]}, expected at least {7 + len(JOINT_NAMES)}."
            )
        if "joint_names" in keys:
            joint_names = [str(x) for x in np.asarray(data["joint_names"]).tolist()]
            if len(joint_names) >= 1 and joint_names[0] == "root":
                joint_index = {name: i for i, name in enumerate(joint_names[1:])}
                missing_joint = [name for name in JOINT_NAMES if name not in joint_index]
                if missing_joint:
                    raise KeyError(f"{src}: missing joint names: {missing_joint}")
                qpos_cols = [7 + joint_index[name] for name in JOINT_NAMES]
                dof_pos = qpos[:, qpos_cols]
            else:
                dof_pos = qpos[:, 7 : 7 + len(JOINT_NAMES)]
        else:
            dof_pos = qpos[:, 7 : 7 + len(JOINT_NAMES)]

        return body_pos_w.astype(np.float32), body_quat_w.astype(np.float32), dof_pos.astype(np.float32)

    raise KeyError(
        f"{src}: unsupported npz format. Need either "
        "{body_pos_w,body_quat_w,joint_pos} or {qpos,xpos,xquat}. keys={sorted(keys)}"
    )


def _convert_single_npz(src: Path, dst: Path, output_fps: float) -> None:
    with np.load(src, allow_pickle=True) as data:
        body_pos_w, body_quat_w, dof_pos = _extract_common_arrays(data, src=src, output_fps=output_fps)

    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"{src}: unexpected body_pos_w shape {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"{src}: unexpected body_quat_w shape {body_quat_w.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"{src}: unexpected joint_pos shape {dof_pos.shape}")
    if body_pos_w.shape[:2] != body_quat_w.shape[:2]:
        raise ValueError(
            f"{src}: mismatched body_pos_w/body_quat_w shape {body_pos_w.shape} vs {body_quat_w.shape}"
        )
    if body_pos_w.shape[0] != dof_pos.shape[0]:
        raise ValueError(f"{src}: frame mismatch body={body_pos_w.shape[0]} joint={dof_pos.shape[0]}")
    if body_pos_w.shape[1] != len(BODY_NAMES):
        raise ValueError(
            f"{src}: expected {len(BODY_NAMES)} bodies, got {body_pos_w.shape[1]}"
        )
    if dof_pos.shape[1] != len(JOINT_NAMES):
        raise ValueError(
            f"{src}: expected {len(JOINT_NAMES)} joints, got {dof_pos.shape[1]}"
        )

    root_pos = body_pos_w[:, 0, :]
    root_quat_wxyz = _normalize_quat_wxyz(np.asarray(body_quat_w[:, 0, :], dtype=np.float32))
    body_quat_wxyz = _normalize_quat_wxyz(np.asarray(body_quat_w, dtype=np.float32))
    root_rot_xyzw = _convert_wxyz_to_xyzw(root_quat_wxyz).astype(np.float32)

    rel_world_pos = body_pos_w - root_pos[:, None, :]
    local_body_pos = _quat_apply_inverse_wxyz(root_quat_wxyz, rel_world_pos).astype(np.float32)

    local_rot_wxyz = _quat_mul_wxyz(_quat_conjugate_wxyz(root_quat_wxyz)[:, None, :], body_quat_wxyz)
    local_rot_wxyz = _normalize_quat_wxyz(local_rot_wxyz.astype(np.float32, copy=False))
    local_rot_xyzw = _convert_wxyz_to_xyzw(local_rot_wxyz).astype(np.float32)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        fps=np.array(output_fps, dtype=np.float32),
        root_pos=root_pos.astype(np.float32),
        root_rot=root_rot_xyzw,
        dof_pos=dof_pos.astype(np.float32),
        local_body_pos=local_body_pos,
        local_body_rot=local_rot_xyzw,
        body_names=np.asarray(BODY_NAMES),
        joint_names=np.asarray(JOINT_NAMES),
    )


def _convert_single_npz_task(args: tuple[str, str, float]) -> tuple[str, str, str]:
    src_s, dst_s, output_fps = args
    src = Path(src_s)
    dst = Path(dst_s)
    try:
        _convert_single_npz(src, dst, output_fps=float(output_fps))
        return "OK", src.name, ""
    except Exception as exc:
        return "FAIL", src.name, str(exc)


def _build_mem_dataset(converted_root: Path, mem_path: Path, segment_len: int, disable_filter: bool) -> None:
    motion_filter = None if disable_filter else check_motion
    MotionDataset.create_from_path(
        str(converted_root),
        target_fps=50,
        mem_path=str(mem_path),
        callback=none_callback,
        motion_processer=partial(preprocess_motion, always_on_ground=False),
        motion_filter=motion_filter,
        segment_len=segment_len,
        storage_float_dtype=torch.float16,
        storage_int_dtype=torch.int32,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert mjlab-exported tennis npz files into motion_tracking training npz format."
    )
    parser.add_argument("--input-dir", type=str, default="data/tennis", help="Directory of source tennis npz files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tennis_tracking_npz",
        help="Directory to write converted training-format npz files.",
    )
    parser.add_argument("--output-fps", type=float, default=50.0, help="Force output fps. Must match training target fps.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing converted files.")
    parser.add_argument("--max-files", type=int, default=None, help="Convert only first N files (for quick checks).")
    parser.add_argument(
        "--input-list",
        type=str,
        default=None,
        help="Optional text file listing npz filenames or absolute paths to convert (one per line).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for npz conversion.")
    parser.add_argument("--progress-sec", type=float, default=5.0, help="Progress heartbeat interval in seconds.")
    parser.add_argument(
        "--build-mem-path",
        type=str,
        default=None,
        help="If set, build motion mem-dataset at this path (e.g. dataset/tennis).",
    )
    parser.add_argument("--segment-len", type=int, default=1000, help="Segment length for mem-dataset building.")
    parser.add_argument(
        "--disable-filter",
        action="store_true",
        default=False,
        help="Disable quality filters when building mem-dataset.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.input_list:
        list_path = Path(args.input_list)
        if not list_path.exists():
            raise FileNotFoundError(f"--input-list not found: {list_path}")
        files = []
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            item = raw.strip()
            if not item:
                continue
            p = Path(item)
            if not p.is_absolute():
                p = input_dir / p
            files.append(p)
        files = sorted(files)
    else:
        files = sorted(input_dir.glob("*.npz"))
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise RuntimeError(f"No npz files found in {input_dir}")

    workers = max(1, int(args.workers))
    progress_sec = max(0.2, float(args.progress_sec))

    converted = 0
    skipped = 0
    failed = []

    tasks: list[tuple[str, str, float]] = []
    scan_last_log = 0.0
    for idx, src in enumerate(files, start=1):
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
        else:
            tasks.append((str(src), str(dst), float(args.output_fps)))

        now = time.time()
        if (now - scan_last_log) >= progress_sec:
            scan_last_log = now
            queued = len(tasks)
            print(
                f"[convert] scan progress scanned={idx}/{len(files)} queued={queued} "
                f"skipped={skipped} overwrite={bool(args.overwrite)}",
                flush=True,
            )

    print(
        f"[convert] scan done total={len(files)} queued={len(tasks)} skipped={skipped}",
        flush=True,
    )

    start_ts = time.time()
    if workers <= 1 or len(tasks) <= 1:
        last_log = 0.0
        for idx, (src_s, dst_s, out_fps) in enumerate(tasks, start=1):
            status, name, err = _convert_single_npz_task((src_s, dst_s, out_fps))
            if status == "OK":
                converted += 1
            else:
                failed.append((name, err))

            now = time.time()
            if (now - last_log) >= progress_sec:
                last_log = now
                elapsed = int(now - start_ts)
                done = converted + len(failed)
                print(
                    f"[convert] progress done={done}/{len(tasks)} converted={converted} "
                    f"failed={len(failed)} skipped={skipped} elapsed={elapsed}s"
                ,
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            total_tasks = len(tasks)
            max_in_flight = max(64, workers * 4)
            pending = set()
            submit_idx = 0

            def _submit_until_full() -> None:
                nonlocal submit_idx
                while submit_idx < total_tasks and len(pending) < max_in_flight:
                    fut = ex.submit(_convert_single_npz_task, tasks[submit_idx])
                    pending.add(fut)
                    submit_idx += 1

            _submit_until_full()
            last_log = 0.0
            last_done = -1
            while pending:
                done_set, pending = wait(pending, timeout=progress_sec, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    status, name, err = fut.result()
                    if status == "OK":
                        converted += 1
                    else:
                        failed.append((name, err))
                _submit_until_full()
                now = time.time()
                done = converted + len(failed)
                should_log = (now - last_log) >= progress_sec or (done == len(tasks) and done != last_done)
                if should_log:
                    last_log = now
                    elapsed = int(now - start_ts)
                    last_done = done
                    rate = (done / max(now - start_ts, 1e-6))
                    remain = max(total_tasks - done, 0)
                    eta_str = "NA" if rate <= 1e-6 else f"{int(remain / rate)}s"
                    print(
                        f"[convert] progress done={done}/{len(tasks)} converted={converted} "
                        f"failed={len(failed)} skipped={skipped} workers={workers} "
                        f"submitted={submit_idx}/{total_tasks} inflight={len(pending)} "
                        f"rate={rate:.1f}/s eta={eta_str} elapsed={elapsed}s"
                    ,
                        flush=True,
                    )

    print(
        f"[convert] total={len(files)} converted={converted} skipped={skipped} failed={len(failed)} output={output_dir}"
    ,
        flush=True,
    )
    if failed:
        print("[convert] failed examples:", flush=True)
        for name, err in failed[:10]:
            print(f"  - {name}: {err}", flush=True)
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more", flush=True)

    if args.build_mem_path:
        mem_path = Path(args.build_mem_path)
        _build_mem_dataset(
            converted_root=output_dir,
            mem_path=mem_path,
            segment_len=int(args.segment_len),
            disable_filter=bool(args.disable_filter),
        )
        meta_path = mem_path / "meta_motion.json"
        id_label_path = mem_path / "id_label.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            print(
                f"[mem] path={mem_path} segments={len(meta.get('starts', []))} joints={len(meta.get('joint_names', []))}"
                ,
                flush=True,
            )
        if id_label_path.exists():
            with id_label_path.open("r", encoding="utf-8") as f:
                labels = json.load(f)
            unique_files = len({Path(x["source_path"]).name for x in labels})
            print(f"[mem] accepted_unique_files={unique_files}", flush=True)


if __name__ == "__main__":
    main()
