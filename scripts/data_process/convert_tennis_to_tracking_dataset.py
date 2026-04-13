import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.utils.motion import MotionDataset

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


def _extract_scalar_fps(value, default: float = 50.0) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _convert_single_npz(src: Path, dst: Path, output_fps: float) -> None:
    with np.load(src, allow_pickle=True) as data:
        required = {"body_pos_w", "body_quat_w", "joint_pos", "fps"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{src} is missing keys: {sorted(missing)}")

        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        dof_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        _ = _extract_scalar_fps(data["fps"], default=output_fps)

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
    root_quat_wxyz = body_quat_w[:, 0, :]
    root_rot_xyzw = _convert_wxyz_to_xyzw(root_quat_wxyz).astype(np.float32)

    root_rot_m = sRot.from_quat(root_rot_xyzw).as_matrix().astype(np.float32)
    rel_world_pos = body_pos_w - root_pos[:, None, :]
    local_body_pos = np.einsum("tji,tbj->tbi", root_rot_m, rel_world_pos).astype(np.float32)

    body_rot_xyzw = _convert_wxyz_to_xyzw(body_quat_w).astype(np.float32)
    body_rot = sRot.from_quat(body_rot_xyzw.reshape(-1, 4))
    root_rot_rep = sRot.from_quat(np.repeat(root_rot_xyzw, body_quat_w.shape[1], axis=0))
    local_rot_xyzw = (root_rot_rep.inv() * body_rot).as_quat().reshape(body_quat_w.shape).astype(np.float32)

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

    files = sorted(input_dir.glob("*.npz"))
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise RuntimeError(f"No npz files found in {input_dir}")

    converted = 0
    skipped = 0
    failed = []
    for src in files:
        dst = output_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            _convert_single_npz(src, dst, output_fps=float(args.output_fps))
            converted += 1
        except Exception as exc:
            failed.append((src.name, str(exc)))

    print(
        f"[convert] total={len(files)} converted={converted} skipped={skipped} failed={len(failed)} output={output_dir}"
    )
    if failed:
        print("[convert] failed examples:")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

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
            )
        if id_label_path.exists():
            with id_label_path.open("r", encoding="utf-8") as f:
                labels = json.load(f)
            unique_files = len({Path(x["source_path"]).name for x in labels})
            print(f"[mem] accepted_unique_files={unique_files}")


if __name__ == "__main__":
    main()
