import re
from typing import Any, List, Sequence

import torch

from active_adaptation.utils.fk_helper import MotionFKHelper
from active_adaptation.utils.joint_modifier import apply_joint_abc_modification_
from active_adaptation.utils.math import wrap_to_pi
from active_adaptation.utils.motion import MotionData, MotionDataset, MotionOriginalData


def _select_required_body_names(asset_body_names: Sequence[str], required_motion_body_patterns: Sequence[str]) -> list[str]:
    if len(required_motion_body_patterns) == 0:
        raise ValueError("required_motion_body_patterns must be provided and non-empty.")
    body_names = [
        name for name in asset_body_names
        if any(re.match(pattern, name) for pattern in required_motion_body_patterns)
    ]
    if len(body_names) == 0:
        raise ValueError(
            "required_motion_body_patterns did not match any asset bodies: "
            f"patterns={list(required_motion_body_patterns)}, asset_body_names={list(asset_body_names)}"
        )
    return body_names


class ProgressiveMultiMotionDataset:
    def __init__(
        self,
        mem_paths: List[str],
        path_weights: List[float],
        env_size: int,
        max_step_size: int,
        required_motion_body_patterns: List[str],
        fk_asset: Any,
        dataset_extra_keys: List[dict] = [],
        device: torch.device = torch.device("cpu"),
        ds_device: torch.device = torch.device("cpu"),
        fix_ds: int = None,
        fix_motion_id: int = None,
        motion_fps: float = 50.0,
    ):
        self.device = device
        self.ds_device = ds_device
        self.env_size = env_size
        self.max_step_size = max_step_size
        self.dataset_extra_keys = dataset_extra_keys
        self.required_motion_body_patterns = list(required_motion_body_patterns)
        self.motion_fps = float(motion_fps)

        self.fix_ds = fix_ds
        self.fix_motion_id = fix_motion_id
        self.enable_modify_joint = False

        self.datasets = [
            MotionDataset.create_from_path_lazy(
                p,
                dataset_extra_keys,
                device=ds_device,
            )
            for p in mem_paths
        ]
        if len(self.datasets) != len(path_weights):
            raise ValueError("mem_paths and path_weights must have the same length")

        joint0 = self.datasets[0].joint_names
        for ds in self.datasets[1:]:
            if ds.joint_names != joint0:
                raise ValueError("All datasets must resolve to the same joint_names")
        self.joint_names = joint0
        self.body_names = _select_required_body_names(fk_asset.body_names, self.required_motion_body_patterns)
        self._fk_helper = MotionFKHelper.from_mjlab_asset(
            asset=fk_asset,
            dataset_joint_names=self.joint_names,
            output_body_names=self.body_names,
        )

        weights = torch.tensor(path_weights, dtype=torch.float32)
        self.probs = (weights / weights.sum()).float().to(device)
        self.counts = [ds.num_motions for ds in self.datasets]

        self._buf_A = self._allocate_empty_buffer()
        self._len_A = torch.zeros(env_size, dtype=torch.int32, device=device)
        self._info_A = self._allocate_info_buffer()
        self._modified_mask_A = torch.zeros((env_size, max_step_size), dtype=torch.bool, device=device)

        self._populate_buffer_full()

        self.joint_pos_limit: torch.Tensor | None = None
        self.joint_vel_limit: torch.Tensor | None = None

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._len_A[env_ids]

    def get_slice(self, env_ids: torch.Tensor | None, starts: torch.Tensor, steps: int | torch.Tensor = 1) -> MotionData:
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        starts = starts.to(self.device)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx.clamp_min_(0).clamp_max_((self._len_A[env_ids] - 1).unsqueeze(1))
            sub = self._buf_A[env_ids.unsqueeze(-1), idx]
        else:
            idx.clamp_min_(0).clamp_max_((self._len_A[:] - 1).unsqueeze(1))
            sub = self._buf_A.gather(1, idx)
        return self._post_process(self._to_float(sub, dtype=torch.float32))

    def get_slice_original(self, env_ids: torch.Tensor | None, starts: torch.Tensor, steps: int | torch.Tensor = 1) -> MotionOriginalData:
        if not self.enable_modify_joint or self.original_joint_pos is None or self.original_joint_vel is None:
            raise RuntimeError("get_slice_original requires enabled joint modification and initialized backups")
        if self.original_body_pos_w is None:
            raise RuntimeError("original_body_pos_w backup is not initialized")

        if env_ids is not None:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        starts = starts.to(self.device, dtype=torch.long)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device, dtype=torch.long)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx.clamp_min_(0).clamp_max_((self._len_A[env_ids] - 1).unsqueeze(1))
            joint_pos = self.original_joint_pos[env_ids.unsqueeze(-1), idx]
            joint_vel = self.original_joint_vel[env_ids.unsqueeze(-1), idx]
            body_pos_w = self.original_body_pos_w[env_ids.unsqueeze(-1), idx]
        else:
            idx.clamp_min_(0).clamp_max_((self._len_A[:] - 1).unsqueeze(1))
            jp_idx = idx.unsqueeze(-1).expand(-1, -1, self.original_joint_pos.shape[-1])
            jv_idx = idx.unsqueeze(-1).expand(-1, -1, self.original_joint_vel.shape[-1])
            bp_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.original_body_pos_w.shape[-2], self.original_body_pos_w.shape[-1])
            joint_pos = torch.gather(self.original_joint_pos, dim=1, index=jp_idx)
            joint_vel = torch.gather(self.original_joint_vel, dim=1, index=jv_idx)
            body_pos_w = torch.gather(self.original_body_pos_w, dim=1, index=bp_idx)
        body_pos_w[..., 2] += 0.035
        return MotionOriginalData(
            joint_pos=joint_pos.to(dtype=torch.float32),
            joint_vel=joint_vel.to(dtype=torch.float32),
            body_pos_w=body_pos_w.to(dtype=torch.float32),
            batch_size=[joint_pos.shape[0], joint_pos.shape[1]],
            device=self.device,
        )

    def get_slice_info(self, env_ids: torch.Tensor):
        return {k["name"]: self._info_A[k["name"]][env_ids] for k in self.dataset_extra_keys}

    def get_slice_modified_mask(self, env_ids: torch.Tensor | None, starts: torch.Tensor, steps: int | torch.Tensor = 1) -> torch.Tensor:
        if env_ids is not None:
            env_ids = env_ids.to(self.device, dtype=torch.long)
        starts = starts.to(self.device, dtype=torch.long)

        if isinstance(steps, int):
            idx = starts.unsqueeze(1) + torch.arange(steps, device=self.device, dtype=torch.long)
        else:
            idx = starts.unsqueeze(1) + steps.to(device=self.device, dtype=torch.long)

        if env_ids is not None:
            idx.clamp_min_(0).clamp_max_((self._len_A[env_ids] - 1).unsqueeze(1))
            return self._modified_mask_A[env_ids.unsqueeze(-1), idx]
        idx.clamp_min_(0).clamp_max_((self._len_A[:] - 1).unsqueeze(1))
        return self._modified_mask_A.gather(1, idx)

    def set_limit(self, joint_pos_limit: torch.Tensor, joint_vel_limit: torch.Tensor, joint_names: List[str]):
        self.joint_pos_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_vel_limit = torch.zeros(1, len(self.joint_names), 2, device=self.device)
        self.joint_pos_limit[:, :, 0] = -3.14
        self.joint_pos_limit[:, :, 1] = 3.14
        self.joint_vel_limit[:, :, 0] = -10.0
        self.joint_vel_limit[:, :, 1] = 10.0
        for asset_idx, name in enumerate(joint_names):
            if name in self.joint_names:
                motion_idx = self.joint_names.index(name)
                self.joint_pos_limit[:, motion_idx] = joint_pos_limit[0, asset_idx]
                self.joint_vel_limit[:, motion_idx] = joint_vel_limit[0, asset_idx]

    def _allocate_empty_buffer(self) -> MotionData:
        float_dtype = torch.float16
        body_count = len(self.body_names)
        joint_count = len(self.joint_names)
        mm = {
            "root_pos_w": torch.zeros((self.env_size, self.max_step_size, 3), dtype=float_dtype, device=self.device),
            "root_quat_w": torch.zeros((self.env_size, self.max_step_size, 4), dtype=float_dtype, device=self.device),
            "root_lin_vel_w": torch.zeros((self.env_size, self.max_step_size, 3), dtype=float_dtype, device=self.device),
            "root_ang_vel_w": torch.zeros((self.env_size, self.max_step_size, 3), dtype=float_dtype, device=self.device),
            "joint_pos": torch.zeros((self.env_size, self.max_step_size, joint_count), dtype=float_dtype, device=self.device),
            "joint_vel": torch.zeros((self.env_size, self.max_step_size, joint_count), dtype=float_dtype, device=self.device),
            "body_pos_w": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
            "body_pos_b": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
            "body_vel_w": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
            "body_vel_b": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
            "body_quat_w": torch.zeros((self.env_size, self.max_step_size, body_count, 4), dtype=float_dtype, device=self.device),
            "body_quat_b": torch.zeros((self.env_size, self.max_step_size, body_count, 4), dtype=float_dtype, device=self.device),
            "body_angvel_w": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
            "body_angvel_b": torch.zeros((self.env_size, self.max_step_size, body_count, 3), dtype=float_dtype, device=self.device),
        }
        mm["root_quat_w"][..., 0] = 1.0
        mm["body_quat_w"][..., 0] = 1.0
        mm["body_quat_b"][..., 0] = 1.0
        return MotionData(**mm, batch_size=[self.env_size, self.max_step_size], device=self.device)

    def _allocate_info_buffer(self):
        return {k["name"]: torch.zeros((self.env_size, k["shape"]), dtype=k["dtype"], device=self.device) for k in self.dataset_extra_keys}

    @torch.no_grad()
    def _populate_buffer_full(self):
        path_samples = torch.multinomial(self.probs, self.env_size, replacement=True).to(torch.int32)
        if self.fix_ds is not None:
            path_samples[:] = self.fix_ds

        for dataset_idx, ds in enumerate(self.datasets):
            env_mask = path_samples == dataset_idx
            if not env_mask.any():
                continue
            motion_count = self.counts[dataset_idx]
            motion_ids = (torch.rand(env_mask.sum(), device=self.ds_device) * motion_count).floor().to(torch.int32)
            if self.fix_motion_id is not None:
                motion_ids[:] = self.fix_motion_id

            motion_ids_long = motion_ids.to(torch.long)
            local_starts = ds.starts[motion_ids_long]
            local_ends = ds.ends[motion_ids_long] - 1
            steps = torch.arange(self.max_step_size, device=self.ds_device, dtype=torch.long)
            local_idx = (local_starts.unsqueeze(1) + steps).clamp(max=local_ends.unsqueeze(1))

            minimal = ds.data[local_idx].to(self.device)
            full = self._fk_helper.expand_minimal_motion(minimal, fps=self.motion_fps)
            self._buf_A[env_mask] = self._to_float(full, dtype=self._buf_A.root_pos_w.dtype)
            self._len_A[env_mask] = ds.lengths[motion_ids_long].clamp_max(self.max_step_size).to(self.device)

            for extra in self.dataset_extra_keys:
                name = extra["name"]
                self._info_A[name][env_mask] = ds.info[name][motion_ids_long].to(self.device)

    def _post_process(self, data: MotionData) -> MotionData:
        data = self._clamp_joint_pos_vel(data)
        data = self._offset_pos_z(data)
        return data

    def _offset_pos_z(self, data: MotionData, z_offset: float = 0.035) -> MotionData:
        data.root_pos_w[..., 2] += z_offset
        data.body_pos_w[..., 2] += z_offset
        return data

    def _clamp_joint_pos_vel(self, data: MotionData) -> MotionData:
        if self.joint_pos_limit is None:
            return data
        joint_pos = wrap_to_pi(data.joint_pos)
        data.joint_pos[:] = torch.clamp(joint_pos, self.joint_pos_limit[:, :, 0], self.joint_pos_limit[:, :, 1])
        data.joint_vel[:] = torch.clamp(data.joint_vel, self.joint_vel_limit[:, :, 0], self.joint_vel_limit[:, :, 1])
        return data

    @staticmethod
    def _to_float(data, dtype=torch.float32):
        for field in data.__dataclass_fields__:
            value = getattr(data, field)
            if torch.is_floating_point(value):
                setattr(data, field, value.to(dtype=dtype))
        return data

    def _sample_joint_pos_bank_online(self, num_frames: int) -> torch.Tensor | None:
        num_frames = int(num_frames)
        if num_frames <= 0:
            return None
        ds_ids = torch.multinomial(self.probs, num_frames, replacement=True)
        counts = torch.bincount(ds_ids, minlength=len(self.datasets))
        chunks: list[torch.Tensor] = []
        for dataset_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            ds = self.datasets[dataset_idx]
            lengths = ds.lengths.to(dtype=torch.long)
            starts = ds.starts.to(dtype=torch.long)
            total_frames = int(lengths.sum().item())
            if total_frames <= 0:
                continue
            flat_ids = torch.randint(0, total_frames, (count,), device=self.ds_device)
            cdf = lengths.cumsum(dim=0)
            motion_ids = torch.searchsorted(cdf, flat_ids, right=True)
            prev_cdf = torch.zeros_like(flat_ids)
            valid_motion = motion_ids > 0
            prev_cdf[valid_motion] = cdf[motion_ids[valid_motion] - 1]
            local_offsets = flat_ids - prev_cdf
            frame_ids = starts[motion_ids] + local_offsets
            chunks.append(ds.data.joint_pos[frame_ids].to(device=self.device, dtype=self._buf_A.joint_pos.dtype))
        if len(chunks) == 0:
            return None
        bank = torch.cat(chunks, dim=0)
        return bank[torch.randperm(bank.shape[0], device=bank.device)].contiguous()

    def setup_joint_modification(
        self,
        *,
        ac_len_range: Sequence[int],
        b_ratio_range: Sequence[float],
        fps: float,
        modify_b_dataset_prob: float,
        modify_joint_pos_bank_size: int = 20000,
        modify_joint_left_patterns: List[str],
        modify_joint_right_patterns: List[str],
        fk_asset: Any,
    ):
        self.enable_modify_joint = True
        self.modify_joint_left_prob = 0.7
        self.modify_joint_right_prob = 0.7
        self.modify_ac_len_range = tuple(int(x) for x in ac_len_range)
        self.modify_b_ratio_range = tuple(float(x) for x in b_ratio_range)
        self.modify_fps = float(fps)
        self.modify_b_dataset_prob = float(modify_b_dataset_prob)
        self.modify_joint_left_patterns = list(modify_joint_left_patterns)
        self.modify_joint_right_patterns = list(modify_joint_right_patterns)
        self.modify_joint_left_ids = torch.tensor(
            [i for i, name in enumerate(self.joint_names) if any(re.match(pattern, name) for pattern in self.modify_joint_left_patterns)],
            device=self.device,
            dtype=torch.long,
        )
        self.modify_joint_right_ids = torch.tensor(
            [i for i, name in enumerate(self.joint_names) if any(re.match(pattern, name) for pattern in self.modify_joint_right_patterns)],
            device=self.device,
            dtype=torch.long,
        )
        if self.modify_joint_left_ids.numel() == 0 and self.modify_joint_right_ids.numel() == 0:
            raise ValueError("No joints matched modify_joint_left_patterns/modify_joint_right_patterns")

        self.modify_joint_pos_bank = self._sample_joint_pos_bank_online(modify_joint_pos_bank_size)
        if self.modify_joint_pos_bank is None or self.modify_joint_pos_bank.numel() == 0:
            raise RuntimeError(f"Failed to sample non-empty modify_joint_pos_bank (size={modify_joint_pos_bank_size}).")

        self._fk_helper = MotionFKHelper.from_mjlab_asset(
            asset=fk_asset,
            dataset_joint_names=self.joint_names,
            output_body_names=self.body_names,
        )
        self.original_joint_pos = self._buf_A.joint_pos.clone()
        self.original_joint_vel = self._buf_A.joint_vel.clone()
        self.original_body_pos_w = self._buf_A.body_pos_w.clone()

    @torch.no_grad()
    def modify_joint(self, env_ids_restore: torch.Tensor, env_ids_modify: torch.Tensor):
        if not self.enable_modify_joint or env_ids_restore.numel() == 0:
            return
        if self._fk_helper is None or self.modify_joint_pos_bank is None:
            raise RuntimeError("Joint modification is not initialized")

        self._buf_A.joint_pos[env_ids_restore] = self.original_joint_pos[env_ids_restore]
        self._buf_A.joint_vel[env_ids_restore] = self.original_joint_vel[env_ids_restore]
        self._modified_mask_A[env_ids_restore] = False

        if env_ids_modify.numel() > 0:
            sub_joint_pos = self._buf_A.joint_pos[env_ids_modify].clone()
            sub_joint_vel = self._buf_A.joint_vel[env_ids_modify].clone()
            sub_lengths = self._len_A[env_ids_modify].to(dtype=torch.long)
            sub_modified_mask = apply_joint_abc_modification_(
                sub_joint_pos,
                sub_joint_vel,
                sub_lengths,
                left_joint_ids=self.modify_joint_left_ids,
                right_joint_ids=self.modify_joint_right_ids,
                left_prob=self.modify_joint_left_prob,
                right_prob=self.modify_joint_right_prob,
                b_dataset_prob=self.modify_b_dataset_prob,
                joint_pos_bank=self.modify_joint_pos_bank,
                ac_len_range=self.modify_ac_len_range,
                b_ratio_range=self.modify_b_ratio_range,
                fps=self.modify_fps,
            )
            self._buf_A.joint_pos[env_ids_modify] = sub_joint_pos
            self._buf_A.joint_vel[env_ids_modify] = sub_joint_vel
            self._modified_mask_A[env_ids_modify] = sub_modified_mask

        sub_motion = self._to_float(self._buf_A[env_ids_restore].clone(), dtype=torch.float32)
        self._fk_helper.rewrite_motion_data_(sub_motion, fps=self.motion_fps)
        self._buf_A.body_pos_w[env_ids_restore] = sub_motion.body_pos_w.to(dtype=self._buf_A.body_pos_w.dtype)
        self._buf_A.body_pos_b[env_ids_restore] = sub_motion.body_pos_b.to(dtype=self._buf_A.body_pos_b.dtype)
        self._buf_A.body_vel_w[env_ids_restore] = sub_motion.body_vel_w.to(dtype=self._buf_A.body_vel_w.dtype)
        self._buf_A.body_vel_b[env_ids_restore] = sub_motion.body_vel_b.to(dtype=self._buf_A.body_vel_b.dtype)
        self._buf_A.body_quat_w[env_ids_restore] = sub_motion.body_quat_w.to(dtype=self._buf_A.body_quat_w.dtype)
        self._buf_A.body_quat_b[env_ids_restore] = sub_motion.body_quat_b.to(dtype=self._buf_A.body_quat_b.dtype)
        self._buf_A.body_angvel_w[env_ids_restore] = sub_motion.body_angvel_w.to(dtype=self._buf_A.body_angvel_w.dtype)
        self._buf_A.body_angvel_b[env_ids_restore] = sub_motion.body_angvel_b.to(dtype=self._buf_A.body_angvel_b.dtype)
