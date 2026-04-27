#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from humanoid_tennis.utils.fk_helper import (  # noqa: E402
    MotionFKHelper,
    angvel_from_quat_wxyz_torch,
    finite_diff_torch,
)


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


def _append_status(status_file: str | None, tag: str, path: str, message: str | None = None) -> None:
    line = f"{tag}\t{path}"
    if message:
        msg = " ".join(str(message).split())
        line = f"{line}\t{msg}"
    line = f"{line}\n"
    if status_file:
        fd = os.open(status_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8", errors="ignore"))
        finally:
            os.close(fd)
    else:
        sys.stdout.write(line)
        sys.stdout.flush()


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    return quat_xyzw / np.clip(norm, 1e-8, None)


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0n = _normalize_quat_xyzw(q0.astype(np.float64, copy=False))
    q1n = _normalize_quat_xyzw(q1.astype(np.float64, copy=False))

    dot = np.sum(q0n * q1n, axis=1, keepdims=True)
    q1n = np.where(dot < 0.0, -q1n, q1n)
    dot = np.sum(q0n * q1n, axis=1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)

    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * blend
    sin_theta = np.sin(theta)

    small = np.abs(sin_theta0) < 1e-6
    s0 = np.sin(theta0 - theta) / np.where(small, 1.0, sin_theta0)
    s1 = sin_theta / np.where(small, 1.0, sin_theta0)

    out = s0 * q0n + s1 * q1n
    out_lin = (1.0 - blend) * q0n + blend * q1n
    out = np.where(small, out_lin, out)
    return _normalize_quat_xyzw(out).astype(np.float32, copy=False)


def _load_seed_csv_as_root_quat_joint(csv_path: Path, position_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    has_header = any(ch.isalpha() for ch in first_line)

    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1 if has_header else 0, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] == 36:
        quat_candidate = arr[:, 3:7]
        quat_norm = np.linalg.norm(quat_candidate, axis=1)
        is_quat_like = (
            np.isfinite(quat_norm).all()
            and np.max(np.abs(quat_candidate)) <= 1.5
            and np.median(np.abs(quat_norm - 1.0)) < 0.2
        )
        if is_quat_like:
            packed = arr.astype(np.float32, copy=False)
        else:
            root = arr[:, 1:4] * position_scale
            euler_deg = arr[:, 4:7]
            quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
            joints_rad = np.deg2rad(arr[:, 7:])
            packed = np.concatenate([root, quat_xyzw, joints_rad], axis=1).astype(np.float32)
    elif arr.shape[1] == 35:
        root = arr[:, 0:3] * position_scale
        euler_deg = arr[:, 3:6]
        quat_xyzw = R.from_euler("xyz", euler_deg, degrees=True).as_quat()
        joints_rad = np.deg2rad(arr[:, 6:])
        packed = np.concatenate([root, quat_xyzw, joints_rad], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported csv width for {csv_path}: got {arr.shape[1]}, expected 35 or 36")

    if packed.shape[1] != 36:
        raise ValueError(f"Converted csv width mismatch for {csv_path}: got {packed.shape[1]}, expected 36")

    root_pos = packed[:, 0:3].astype(np.float32, copy=False)
    root_quat_xyzw = _normalize_quat_xyzw(packed[:, 3:7].astype(np.float32, copy=False))
    joint_pos = packed[:, 7:].astype(np.float32, copy=False)
    return root_pos, root_quat_xyzw, joint_pos


def _resample_motion(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError(f"input_fps/output_fps must be > 0, got {input_fps}/{output_fps}")

    num_frames = int(root_pos.shape[0])
    if num_frames <= 0:
        raise ValueError("Empty motion")
    if num_frames == 1:
        return root_pos, root_quat_xyzw, joint_pos

    duration = (num_frames - 1) / float(input_fps)
    times = np.arange(0.0, duration, 1.0 / float(output_fps), dtype=np.float64)
    if times.size == 0:
        times = np.array([0.0], dtype=np.float64)

    phase = times * float(input_fps)
    index_0 = np.floor(phase).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, num_frames - 1)
    blend = (phase - index_0).astype(np.float32).reshape(-1, 1)

    root_out = root_pos[index_0] * (1.0 - blend) + root_pos[index_1] * blend
    joint_out = joint_pos[index_0] * (1.0 - blend) + joint_pos[index_1] * blend
    quat_out = _slerp_xyzw(root_quat_xyzw[index_0], root_quat_xyzw[index_1], blend)

    return (
        root_out.astype(np.float32, copy=False),
        quat_out.astype(np.float32, copy=False),
        joint_out.astype(np.float32, copy=False),
    )


def _build_fk_helper(device: torch.device, mjlab_repo: Path) -> MotionFKHelper:
    sys.path.insert(0, str(mjlab_repo))
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

    spec = get_spec()
    model = spec.compile()

    body_name_to_id: dict[str, int] = {"world": 0}
    for body_id in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            body_name_to_id[name.split("/")[-1]] = int(body_id)

    joint_id_to_name: dict[int, str] = {}
    for joint_id in range(int(model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not name:
            raise ValueError(f"Joint id={joint_id} has no name in model")
        joint_id_to_name[int(joint_id)] = name.split("/")[-1]

    return MotionFKHelper._build(
        model=model,
        body_name_to_id=body_name_to_id,
        joint_id_to_name=joint_id_to_name,
        dataset_joint_names=JOINT_NAMES,
        output_body_names=BODY_NAMES,
        base_body_name="pelvis",
        device=device,
    )


@torch.no_grad()
def _convert_one(
    fk_helper: MotionFKHelper,
    src_csv: Path,
    dst_npz: Path,
    *,
    input_fps: float,
    output_fps: float,
    position_scale: float,
) -> None:
    root_pos, root_quat_xyzw, joint_pos = _load_seed_csv_as_root_quat_joint(src_csv, position_scale=position_scale)
    root_pos, root_quat_xyzw, joint_pos = _resample_motion(
        root_pos, root_quat_xyzw, joint_pos, input_fps=input_fps, output_fps=output_fps
    )
    root_quat_wxyz = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, 0:3]], axis=1).astype(np.float32)

    device = fk_helper.device
    root_pos_t = torch.from_numpy(root_pos).to(device=device, dtype=torch.float32)
    root_quat_t = torch.from_numpy(root_quat_wxyz).to(device=device, dtype=torch.float32)
    joint_pos_t = torch.from_numpy(joint_pos).to(device=device, dtype=torch.float32)

    _, _, body_pos_w_t, body_quat_w_t = fk_helper.body_pose(root_pos_t, root_quat_t, joint_pos_t)
    joint_vel_t = finite_diff_torch(joint_pos_t, fps=output_fps, dim=0)
    body_lin_vel_t = finite_diff_torch(body_pos_w_t, fps=output_fps, dim=0)
    body_ang_vel_t = angvel_from_quat_wxyz_torch(body_quat_w_t, fps=output_fps, dim=0)

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = dst_npz.with_name(f".{dst_npz.stem}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp_npz,
        fps=np.asarray([output_fps], dtype=np.float32),
        joint_pos=joint_pos_t.detach().cpu().numpy().astype(np.float32),
        joint_vel=joint_vel_t.detach().cpu().numpy().astype(np.float32),
        body_pos_w=body_pos_w_t.detach().cpu().numpy().astype(np.float32),
        body_quat_w=body_quat_w_t.detach().cpu().numpy().astype(np.float32),
        body_lin_vel_w=body_lin_vel_t.detach().cpu().numpy().astype(np.float32),
        body_ang_vel_w=body_ang_vel_t.detach().cpu().numpy().astype(np.float32),
    )
    os.replace(tmp_npz, dst_npz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert seed CSV files into stage-1 raw npz using fast FK.")
    parser.add_argument("--csv-list", type=str, required=True, help="Text file containing CSV paths, one per line.")
    parser.add_argument("--raw-npz-dir", type=str, required=True, help="Output directory for raw npz files.")
    parser.add_argument("--mjlab-repo", type=str, required=True, help="Path to unitree_rl_mjlab repo root.")
    parser.add_argument("--input-fps", type=float, default=120.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--position-scale", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20, help="Print worker progress every N processed files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_list_path = Path(args.csv_list)
    raw_npz_dir = Path(args.raw_npz_dir)
    mjlab_repo = Path(args.mjlab_repo)

    if not csv_list_path.is_file():
        raise FileNotFoundError(f"CSV list file not found: {csv_list_path}")
    if not mjlab_repo.is_dir():
        raise FileNotFoundError(f"mjlab repo path not found: {mjlab_repo}")

    csv_files = [Path(line.strip()) for line in csv_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(csv_files) == 0:
        print(f"[worker] no files in list: {csv_list_path}")
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[worker] CUDA unavailable, fallback to CPU", flush=True)
        device = torch.device("cpu")

    fk_helper = _build_fk_helper(device=device, mjlab_repo=mjlab_repo)
    print(f"[worker] init done: files={len(csv_files)} device={device}", flush=True)

    converted = 0
    skipped = 0
    failed = 0
    for idx, csv_path in enumerate(csv_files):
        stem = csv_path.stem
        dst = raw_npz_dir / f"{stem}.npz"
        if dst.exists() and not args.overwrite:
            skipped += 1
            _append_status(args.status_file, "SKIP", str(csv_path))
            continue

        try:
            _convert_one(
                fk_helper=fk_helper,
                src_csv=csv_path,
                dst_npz=dst,
                input_fps=float(args.input_fps),
                output_fps=float(args.output_fps),
                position_scale=float(args.position_scale),
            )
            converted += 1
            _append_status(args.status_file, "OK", str(csv_path))
        except Exception as exc:
            failed += 1
            _append_status(args.status_file, "FAIL", str(csv_path), message=str(exc))
            print(f"[worker] FAIL: {csv_path}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

        if args.log_every > 0 and ((idx + 1) % int(args.log_every) == 0):
            print(
                f"[worker] progress {idx + 1}/{len(csv_files)} converted={converted} skipped={skipped} failed={failed}",
                flush=True,
            )

    print(
        f"[worker] done total={len(csv_files)} converted={converted} skipped={skipped} failed={failed}",
        flush=True,
    )
    return 1 if (failed > 0 and args.fail_fast) else 0


if __name__ == "__main__":
    raise SystemExit(main())
