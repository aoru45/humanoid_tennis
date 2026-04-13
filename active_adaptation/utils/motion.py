import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from tensordict import MemoryMappedTensor, TensorClass
from tqdm import tqdm

from active_adaptation.utils.motion_utils import angvel_from_rot, finite_diff_vel


class MotionMinimalData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor


class MotionData(TensorClass):
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_pos_b: torch.Tensor
    body_vel_w: torch.Tensor
    body_vel_b: torch.Tensor
    body_quat_w: torch.Tensor
    body_quat_b: torch.Tensor
    body_angvel_w: torch.Tensor
    body_angvel_b: torch.Tensor


class MotionOriginalData(TensorClass):
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor


FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
PRESERVED_KEYS = ("fps", "qpos", "qvel", "xpos")


@dataclass(frozen=True)
class _MotionSchema:
    joint_names: list[str]
    body_names: list[str]
    joint_idx: np.ndarray
    body_idx: np.ndarray
    foot_idx: list[int]


@dataclass(frozen=True)
class _AcceptedSegment:
    path: Path
    start_idx: int
    end_idx: int


def _resolve_motion_paths(root_path: str) -> list[Path]:
    root = Path(root_path)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No motions found in {root_path}")
    return paths


def _extract_motion_fps(data) -> int:
    return int(data.get("mocap_framerate", data.get("frequency", data.get("fps", 0))))


def _build_motion_schema(data) -> _MotionSchema:
    joint_names = data["joint_names"].tolist()
    body_names = data["body_names"].tolist()
    foot_idx = [body_names.index(name) for name in FOOT_NAMES]
    return _MotionSchema(
        joint_names=list(joint_names),
        body_names=list(body_names),
        joint_idx=np.arange(len(joint_names), dtype=np.int64),
        body_idx=np.arange(len(body_names), dtype=np.int64),
        foot_idx=foot_idx,
    )


def _convert_quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _prepare_motion_arrays(path: Path, schema: _MotionSchema | None, target_fps: int) -> tuple[dict, _MotionSchema]:
    with np.load(path, allow_pickle=True) as data:
        if schema is None:
            schema = _build_motion_schema(data)

        fps = _extract_motion_fps(data)
        if fps != target_fps:
            raise ValueError(f"Expected fps={target_fps}, got {fps} for {path}")

        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_quat_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
        joint_pos = np.asarray(data["dof_pos"][:, schema.joint_idx], dtype=np.float32)
        local_body_pos = np.asarray(data["local_body_pos"][:, schema.body_idx], dtype=np.float32)

    root_rot = sRot.from_quat(root_quat_xyzw)
    root_rot_m = root_rot.as_matrix().astype(np.float32)
    body_pos_w = np.einsum("tij,tbj->tbi", root_rot_m, local_body_pos) + root_pos[:, None, :]

    root_lin_vel_w = finite_diff_vel(root_pos, fps).astype(np.float32)
    joint_vel = finite_diff_vel(joint_pos, fps).astype(np.float32)
    root_ang_vel_w = angvel_from_rot(root_rot, fps=fps)

    root_quat_wxyz = _convert_quat_xyzw_to_wxyz(root_quat_xyzw).astype(np.float32)
    qpos = np.concatenate([root_pos, root_quat_wxyz, joint_pos], axis=-1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel_w, root_ang_vel_w, joint_vel], axis=-1).astype(np.float32)

    motion = {
        "fps": fps,
        "qpos": qpos,
        "qvel": qvel,
        "xpos": body_pos_w.astype(np.float32),
        "joint_names": schema.joint_names,
        "body_names": schema.body_names,
    }
    return motion, schema


def _iter_motion_segments(motion: dict, segment_len: int) -> Iterator[tuple[int, int, dict]]:
    total_steps = int(motion["qpos"].shape[0])
    for start_idx in range(0, total_steps, segment_len):
        end_idx = min(start_idx + segment_len, total_steps)
        segment = {key: motion[key] if key == "fps" else motion[key][start_idx:end_idx] for key in PRESERVED_KEYS}
        segment["joint_names"] = motion["joint_names"]
        segment["body_names"] = motion["body_names"]
        yield start_idx, end_idx, segment


def _slice_motion_segment(motion: dict, start_idx: int, end_idx: int) -> dict:
    segment = {key: motion[key] if key == "fps" else motion[key][start_idx:end_idx] for key in PRESERVED_KEYS}
    segment["joint_names"] = motion["joint_names"]
    segment["body_names"] = motion["body_names"]
    return segment


def _apply_motion_processer(motion_processer: Callable | None, segment: dict, foot_idx: list[int], path: Path, start_idx: int, end_idx: int) -> dict:
    if motion_processer is None:
        return segment
    try:
        return motion_processer(segment, foot_idx, path, start_idx, end_idx)
    except TypeError:
        return motion_processer(segment, foot_idx)


def _apply_motion_filter(motion_filter: Callable | None, segment: dict, foot_idx: list[int], path: Path, start_idx: int, end_idx: int) -> bool:
    if motion_filter is None:
        return True
    try:
        return bool(motion_filter(segment, foot_idx, path, start_idx, end_idx))
    except TypeError:
        return bool(motion_filter(segment, foot_idx, path))


def _run_callback(callback: Callable | None, segment: dict, foot_idx: list[int], path: Path, start_idx: int, end_idx: int) -> None:
    if callback is None:
        return
    callback(
        {
            "path": path,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "foot_idx": foot_idx,
        },
        segment,
    )


def _stack_metadata_rows(rows: list[dict]) -> dict:
    if len(rows) == 0:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        arr = np.asarray(values)
        out[key] = arr.reshape(len(rows), -1).tolist()
    return out


def _allocate_minimal_storage(total: int, joint_count: int, storage_float_dtype: torch.dtype) -> MotionMinimalData:
    mm = {
        "root_pos_w": MemoryMappedTensor.empty(total, 3, dtype=storage_float_dtype),
        "root_quat_w": MemoryMappedTensor.empty(total, 4, dtype=storage_float_dtype),
        "joint_pos": MemoryMappedTensor.empty(total, joint_count, dtype=storage_float_dtype),
    }
    return MotionMinimalData(**mm, batch_size=[total])


def _write_minimal_segment(data: MotionMinimalData, cursor: int, segment: dict, storage_float_dtype: torch.dtype) -> int:
    length = int(segment["qpos"].shape[0])
    joint_count = len(segment["joint_names"])
    span = slice(cursor, cursor + length)
    data.root_pos_w[span] = torch.as_tensor(segment["qpos"][:, :3], dtype=storage_float_dtype)
    data.root_quat_w[span] = torch.as_tensor(segment["qpos"][:, 3:7], dtype=storage_float_dtype)
    data.joint_pos[span] = torch.as_tensor(segment["qpos"][:, 7:7 + joint_count], dtype=storage_float_dtype)
    return length


class MotionDataset:
    def __init__(
        self,
        joint_names: list[str],
        starts: list[int],
        ends: list[int],
        data: MotionMinimalData,
        info: dict,
        device: torch.device = torch.device("cpu"),
    ):
        self.joint_names = joint_names
        self.starts = torch.as_tensor(starts, dtype=torch.int32, device=device)
        self.ends = torch.as_tensor(ends, dtype=torch.int32, device=device)
        self.lengths = self.ends - self.starts
        self.data = data
        self.info = info

    @classmethod
    def create_from_path_lazy(
        cls,
        mem_path: str,
        dataset_extra_keys: list[dict] = [],
        device: torch.device = torch.device("cpu"),
    ):
        path_root = os.environ.get("MEMPATH")
        if path_root is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../..")
            path_root = os.path.join(project_root, "dataset")
        mem_path = os.path.join(path_root, mem_path)

        data = MotionMinimalData.load_memmap(os.path.join(mem_path, "_tensordict"))
        data = data.to(device)
        with open(mem_path + "/meta_motion.json", "r") as f:
            meta = json.load(f)

        infos = {}
        for k in dataset_extra_keys:
            name = k["name"]
            shape = k["shape"]
            if name not in meta["info"]:
                infos[name] = torch.zeros((len(meta["starts"]), shape), dtype=k["dtype"], device=device)
            else:
                infos[name] = torch.tensor(meta["info"][name], dtype=k["dtype"], device=device)
            if infos[name].shape != (len(meta["starts"]), shape):
                raise ValueError(f"Shape of {name} does not match: {infos[name].shape} != {(len(meta['starts']), shape)}")

        return cls(
            joint_names=meta["joint_names"],
            starts=meta["starts"],
            ends=meta["ends"],
            data=data,
            info=infos,
            device=device,
        )

    @classmethod
    def create_from_path(
        cls,
        root_path: str,
        target_fps: int = 50,
        mem_path: str | None = None,
        motion_processer: Callable | None = None,
        motion_filter: Callable | None = None,
        callback: Callable | None = None,
        segment_len: int = 1000,
        *,
        build_dataset: bool = True,
        storage_float_dtype: torch.dtype = torch.float16,
        storage_int_dtype: torch.dtype = torch.int32,
    ):
        paths = _resolve_motion_paths(root_path)
        schema = None
        total = 0
        accepted_segments: list[_AcceptedSegment] = []
        metadata_rows = []
        id_labels = []

        scan_bar = tqdm(paths, desc="scan")
        for path in scan_bar:
            motion, schema = _prepare_motion_arrays(path, schema, target_fps)
            for start_idx, end_idx, segment in _iter_motion_segments(motion, segment_len):
                segment = _apply_motion_processer(motion_processer, segment, schema.foot_idx, path, start_idx, end_idx)
                if not _apply_motion_filter(motion_filter, segment, schema.foot_idx, path, start_idx, end_idx):
                    continue
                _run_callback(callback, segment, schema.foot_idx, path, start_idx, end_idx)
                metadata = segment.get("metadata") or {}
                total += int(segment["qpos"].shape[0])
                accepted_segments.append(_AcceptedSegment(path=path, start_idx=int(start_idx), end_idx=int(end_idx)))
                metadata_rows.append(metadata)
                id_labels.append({
                    "source_path": str(path),
                    "segment_start": int(start_idx),
                    "segment_end": int(end_idx),
                })
            scan_bar.set_postfix(total=total, motions=len(id_labels))

        if schema is None:
            raise RuntimeError(f"No valid motions found in {root_path}")

        meta = {
            "joint_names": schema.joint_names,
            "info": _stack_metadata_rows(metadata_rows),
        }

        if not build_dataset:
            return None, meta

        data = _allocate_minimal_storage(total, len(schema.joint_names), storage_float_dtype)
        starts = []
        ends = []
        cursor = 0
        motion_id = 0

        write_bar = tqdm(accepted_segments, desc="write")
        motion_cache: dict[Path, dict] = {}
        for accepted in write_bar:
            motion = motion_cache.get(accepted.path)
            if motion is None:
                motion, _ = _prepare_motion_arrays(accepted.path, schema, target_fps)
                motion_cache = {accepted.path: motion}
            segment = _slice_motion_segment(motion, accepted.start_idx, accepted.end_idx)
            segment = _apply_motion_processer(
                motion_processer,
                segment,
                schema.foot_idx,
                accepted.path,
                accepted.start_idx,
                accepted.end_idx,
            )
            _run_callback(
                callback,
                segment,
                schema.foot_idx,
                accepted.path,
                accepted.start_idx,
                accepted.end_idx,
            )
            starts.append(cursor)
            cursor += _write_minimal_segment(data, cursor, segment, storage_float_dtype)
            ends.append(cursor)
            motion_id += 1
            write_bar.set_postfix(cursor=cursor, motions=motion_id)

        if cursor != total or motion_id != len(id_labels):
            raise RuntimeError(
                f"Streaming write mismatch: cursor={cursor}, total={total}, motion_id={motion_id}, labels={len(id_labels)}"
            )

        if mem_path is not None:
            path = Path(mem_path)
            path.mkdir(parents=True, exist_ok=True)
            data.memmap(str(path / "_tensordict"))
            dump_data = {
                "joint_names": meta["joint_names"],
                "starts": starts,
                "ends": ends,
                "info": meta["info"],
            }
            with (path / "meta_motion.json").open("w") as f:
                json.dump(dump_data, f)
            with (path / "id_label.json").open("w") as f:
                json.dump(id_labels, f, ensure_ascii=True)

        return data, meta

    @property
    def num_motions(self):
        return len(self.starts)

    @property
    def num_steps(self):
        return len(self.data)

    def get_slice(self, motion_ids: torch.Tensor, starts: torch.Tensor, steps: int = 1) -> MotionMinimalData:
        motion_ids = motion_ids.to(dtype=torch.long, device=self.starts.device)
        starts = starts.to(dtype=torch.long, device=self.starts.device)
        starts_per_motion = self.starts[motion_ids].to(torch.long).unsqueeze(1)
        if isinstance(steps, int):
            idx = (starts_per_motion.squeeze(1) + starts).unsqueeze(1) + torch.arange(steps, device=self.starts.device, dtype=torch.long)
        else:
            idx = (starts_per_motion.squeeze(1) + starts).unsqueeze(1) + steps.to(dtype=torch.long, device=self.starts.device)
        ends_per_motion = (self.ends[motion_ids].to(torch.long) - 1).unsqueeze(1)
        idx.clamp_min_(starts_per_motion).clamp_max_(ends_per_motion)
        return self.data[idx]
