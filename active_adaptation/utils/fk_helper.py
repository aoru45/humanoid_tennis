from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from active_adaptation.utils.math import normalize, quat_apply, quat_apply_inverse, quat_conjugate, quat_from_angle_axis, quat_mul
from active_adaptation.utils.motion import MotionData, MotionMinimalData


def _basename(name: str) -> str:
    return name.split("/")[-1]


def _as_torch(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if hasattr(value, "_tensor"):
        return value._tensor.to(device=device)
    import warp as wp  # type: ignore

    if isinstance(value, wp.array):  # type: ignore[arg-type]
        return wp.to_torch(value).to(device=device)
    return torch.as_tensor(value, device=device)


def _as_scalar_1d(value: Any, *, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 2:
        out = out[0]
    if out.ndim != 1:
        raise ValueError(f"Expected scalar field [N] or [E,N], got {tuple(out.shape)}")
    return out


def _as_vec_field(value: Any, *, dim: int, device: torch.device) -> torch.Tensor:
    out = _as_torch(value, device=device)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2 or out.shape[-1] != dim:
        raise ValueError(f"Expected vector field [N,{dim}] or [E,N,{dim}], got {tuple(out.shape)}")
    return out


def finite_diff_torch(x: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    x_t = x.movedim(dim, 0)
    vel = torch.zeros_like(x_t)
    if fps <= 0 or x_t.shape[0] < 2:
        return vel.movedim(0, dim)
    vel[1:-1] = (x_t[2:] - x_t[:-2]) * (fps / 2.0)
    vel[0] = (x_t[1] - x_t[0]) * fps
    vel[-1] = (x_t[-1] - x_t[-2]) * fps
    return vel.movedim(0, dim)


def angvel_from_quat_wxyz_torch(quat_wxyz: torch.Tensor, fps: float, dim: int) -> torch.Tensor:
    quat_t = normalize(quat_wxyz.movedim(dim, 0))
    if fps <= 0 or quat_t.shape[0] < 2:
        return torch.zeros(quat_t.shape[:-1] + (3,), dtype=quat_t.dtype, device=quat_t.device).movedim(0, dim)

    flat = quat_t.reshape(quat_t.shape[0], -1, 4)
    dots = (flat[1:] * flat[:-1]).sum(dim=-1)
    signs = torch.where(dots < 0.0, -torch.ones_like(dots), torch.ones_like(dots))
    signs = torch.cat([torch.ones_like(signs[:1]), signs], dim=0)
    flat = flat * torch.cumprod(signs, dim=0).unsqueeze(-1)

    qdot = torch.zeros_like(flat)
    qdot[1:-1] = (flat[2:] - flat[:-2]) * (fps / 2.0)
    qdot[0] = (flat[1] - flat[0]) * fps
    qdot[-1] = (flat[-1] - flat[-2]) * fps

    omega = 2.0 * quat_mul(qdot, quat_conjugate(flat))[..., 1:]
    return omega.reshape(quat_t.shape[:-1] + (3,)).movedim(0, dim)


@dataclass(frozen=True)
class FKTreeInfo:
    body_names: list[str]
    body_model_ids: torch.Tensor
    joint_dataset_idx: torch.Tensor


class MotionFKHelper:
    def __init__(
        self,
        *,
        device: torch.device,
        base_body_id: int,
        tree_body_ids: torch.Tensor,
        parent_local_idx: torch.Tensor,
        body_pos0: torch.Tensor,
        body_quat0: torch.Tensor,
        joint_types: torch.Tensor,
        joint_pos_local: torch.Tensor,
        joint_axis_local: torch.Tensor,
        joint_dataset_idx: torch.Tensor,
        output_local_idx: torch.Tensor,
        output_body_names: list[str],
        output_body_ids: torch.Tensor,
        world_output_idx: int,
    ):
        self.device = device
        self.base_body_id = int(base_body_id)
        self.tree_body_ids = tree_body_ids
        self.parent_local_idx = parent_local_idx
        self.body_pos0 = body_pos0
        self.body_quat0 = body_quat0
        self.joint_types = joint_types
        self.joint_pos_local = joint_pos_local
        self.joint_axis_local = joint_axis_local
        self.joint_dataset_idx = joint_dataset_idx
        self.output_local_idx = output_local_idx
        self.output_body_names = output_body_names
        self.output_body_ids = output_body_ids
        self.world_output_idx = int(world_output_idx)
        self.base_local_idx = int((self.tree_body_ids == self.base_body_id).nonzero(as_tuple=False)[0].item())
        self._dtype_cache: dict[torch.dtype, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def tree_info(self) -> FKTreeInfo:
        return FKTreeInfo(
            body_names=self.output_body_names,
            body_model_ids=self.output_body_ids,
            joint_dataset_idx=self.joint_dataset_idx,
        )

    @classmethod
    def from_mjlab_asset(
        cls,
        *,
        asset: Any,
        dataset_joint_names: Sequence[str],
        output_body_names: Sequence[str],
    ) -> "MotionFKHelper":
        device = torch.device(asset.data.device)
        model = asset.data.model

        body_name_to_id = {"world": 0}
        for i, name in enumerate(asset.body_names):
            gid = int(asset.indexing.body_ids[i].item())
            body_name_to_id[_basename(name)] = gid

        joint_id_to_name = {}
        for i, name in enumerate(asset.joint_names):
            gid = int(asset.indexing.joint_ids[i].item())
            joint_id_to_name[gid] = _basename(name)

        base_body_name = _basename(asset.body_names[0])

        return cls._build(
            model=model,
            body_name_to_id=body_name_to_id,
            joint_id_to_name=joint_id_to_name,
            dataset_joint_names=dataset_joint_names,
            output_body_names=list(output_body_names),
            base_body_name=base_body_name,
            device=device,
        )

    @classmethod
    def _build(
        cls,
        *,
        model: Any,
        body_name_to_id: dict[str, int],
        joint_id_to_name: dict[int, str],
        dataset_joint_names: Sequence[str],
        output_body_names: list[str],
        base_body_name: str,
        device: torch.device,
    ) -> "MotionFKHelper":
        body_parentid = _as_scalar_1d(getattr(model, "body_parentid"), device=device).to(torch.long)
        body_jntnum = _as_scalar_1d(getattr(model, "body_jntnum"), device=device).to(torch.long)
        body_jntadr = _as_scalar_1d(getattr(model, "body_jntadr"), device=device).to(torch.long)
        body_pos_all = _as_vec_field(getattr(model, "body_pos"), dim=3, device=device)
        body_quat_all = _as_vec_field(getattr(model, "body_quat"), dim=4, device=device)
        jnt_type = _as_scalar_1d(getattr(model, "jnt_type"), device=device).to(torch.long)
        jnt_pos = _as_vec_field(getattr(model, "jnt_pos"), dim=3, device=device)
        jnt_axis = _as_vec_field(getattr(model, "jnt_axis"), dim=3, device=device)

        if base_body_name not in body_name_to_id:
            raise ValueError(f"Base body '{base_body_name}' not found in model")
        base_body_id = int(body_name_to_id[base_body_name])

        requested_ids: list[int] = []
        world_output_idx = -1
        for out_idx, body_name in enumerate(output_body_names):
            if body_name == "world":
                world_output_idx = out_idx
                continue
            if body_name not in body_name_to_id:
                raise ValueError(f"Output body '{body_name}' not found in model")
            requested_ids.append(int(body_name_to_id[body_name]))

        selected: set[int] = {base_body_id}
        for body_id in requested_ids:
            cur = body_id
            while True:
                selected.add(cur)
                if cur == base_body_id:
                    break
                cur = int(body_parentid[cur].item())
                if cur < 0:
                    raise RuntimeError(f"Cannot trace body id={body_id} back to base '{base_body_name}'")

        children_by_gid: dict[int, list[int]] = {body_id: [] for body_id in selected}
        for body_id in selected:
            if body_id == base_body_id:
                continue
            parent_id = int(body_parentid[body_id].item())
            if parent_id in selected:
                children_by_gid[parent_id].append(body_id)
        for children in children_by_gid.values():
            children.sort()

        order_gid: list[int] = []

        def _dfs(body_id: int):
            order_gid.append(body_id)
            for child_id in children_by_gid.get(body_id, []):
                _dfs(child_id)

        _dfs(base_body_id)
        gid_to_local = {gid: idx for idx, gid in enumerate(order_gid)}
        joint_name_to_dataset_idx = {name: idx for idx, name in enumerate(dataset_joint_names)}
        output_name_to_index = {name: idx for idx, name in enumerate(output_body_names)}
        id_to_body_name = {value: key for key, value in body_name_to_id.items()}

        tree_body_ids = torch.tensor(order_gid, device=device, dtype=torch.long)
        parent_local_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        body_pos0 = torch.empty((len(order_gid), 3), device=device, dtype=body_pos_all.dtype)
        body_quat0 = torch.empty((len(order_gid), 4), device=device, dtype=body_quat_all.dtype)
        joint_types = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        joint_pos_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_pos.dtype)
        joint_axis_local = torch.zeros((len(order_gid), 3), device=device, dtype=jnt_axis.dtype)
        joint_dataset_idx = torch.full((len(order_gid),), -1, device=device, dtype=torch.long)
        output_local_idx = torch.full((len(output_body_names),), -1, device=device, dtype=torch.long)
        output_body_ids = torch.full((len(output_body_names),), -1, device=device, dtype=torch.long)

        for local_idx, body_id in enumerate(order_gid):
            parent_id = int(body_parentid[body_id].item())
            parent_local_idx[local_idx] = gid_to_local[parent_id] if parent_id in gid_to_local else -1
            body_pos0[local_idx] = body_pos_all[body_id]
            body_quat0[local_idx] = body_quat_all[body_id]

            body_name = id_to_body_name[body_id]
            if body_name in output_name_to_index:
                out_idx = output_name_to_index[body_name]
                output_local_idx[out_idx] = local_idx
                output_body_ids[out_idx] = body_id

            joint_count = int(body_jntnum[body_id].item())
            if joint_count > 1:
                raise NotImplementedError(f"Body '{body_name}' has {joint_count} joints; only <=1 joint/body is supported.")
            if joint_count == 0:
                continue

            joint_id = int(body_jntadr[body_id].item())
            joint_type = int(jnt_type[joint_id].item())
            if body_id == base_body_id and joint_type == 0:
                continue
            if joint_type not in (2, 3):
                raise NotImplementedError(f"Joint '{joint_id_to_name[joint_id]}' type={joint_type} unsupported.")

            joint_name = joint_id_to_name[joint_id]
            if joint_name not in joint_name_to_dataset_idx:
                raise ValueError(f"Joint '{joint_name}' missing from dataset_joint_names")
            joint_types[local_idx] = joint_type
            joint_pos_local[local_idx] = jnt_pos[joint_id]
            joint_axis_local[local_idx] = jnt_axis[joint_id]
            joint_dataset_idx[local_idx] = joint_name_to_dataset_idx[joint_name]

        missing_mask = output_local_idx < 0
        if world_output_idx >= 0:
            missing_mask[world_output_idx] = False
        if missing_mask.any():
            missing = [output_body_names[i] for i in missing_mask.nonzero(as_tuple=False).squeeze(-1).tolist()]
            raise ValueError(f"Failed to resolve requested output bodies: {missing}")

        return cls(
            device=device,
            base_body_id=base_body_id,
            tree_body_ids=tree_body_ids,
            parent_local_idx=parent_local_idx,
            body_pos0=body_pos0,
            body_quat0=body_quat0,
            joint_types=joint_types,
            joint_pos_local=joint_pos_local,
            joint_axis_local=joint_axis_local,
            joint_dataset_idx=joint_dataset_idx,
            output_local_idx=output_local_idx,
            output_body_names=output_body_names,
            output_body_ids=output_body_ids,
            world_output_idx=world_output_idx,
        )

    def _typed_constants(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._dtype_cache.get(dtype)
        if cached is None:
            cached = (
                self.body_pos0.to(dtype=dtype),
                normalize(self.body_quat0.to(dtype=dtype)),
                self.joint_pos_local.to(dtype=dtype),
                normalize(self.joint_axis_local.to(dtype=dtype)),
            )
            self._dtype_cache[dtype] = cached
        return cached

    def body_pose(
        self,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = root_pos_w.dtype
        if root_pos_w.device != self.device:
            raise RuntimeError(f"FK helper device mismatch: helper={self.device}, root={root_pos_w.device}")

        prefix = root_pos_w.shape[:-1]
        flat_count = int(torch.tensor(prefix).prod().item()) if len(prefix) > 0 else 1
        joint_pos_f = joint_pos.reshape(flat_count, joint_pos.shape[-1])
        root_pos_f = root_pos_w.reshape(flat_count, 3)
        root_quat_f = normalize(root_quat_w.reshape(flat_count, 4))

        body_pos0, body_quat0, joint_pos_local, joint_axis_local = self._typed_constants(dtype)
        tree_pos_b = torch.zeros((flat_count, len(self.tree_body_ids), 3), device=self.device, dtype=dtype)
        tree_quat_b = torch.zeros((flat_count, len(self.tree_body_ids), 4), device=self.device, dtype=dtype)
        tree_quat_b[:, self.base_local_idx, 0] = 1.0

        for local_idx in range(len(self.tree_body_ids)):
            if local_idx == self.base_local_idx:
                continue
            parent_idx = int(self.parent_local_idx[local_idx].item())
            if parent_idx < 0:
                continue

            parent_pos_b = tree_pos_b[:, parent_idx]
            parent_quat_b = tree_quat_b[:, parent_idx]
            pos0 = body_pos0[local_idx].unsqueeze(0)
            quat0 = body_quat0[local_idx].unsqueeze(0)
            joint_type = int(self.joint_types[local_idx].item())

            if joint_type < 0:
                rel_quat = quat0
                rel_pos = pos0
            else:
                joint_idx = int(self.joint_dataset_idx[local_idx].item())
                joint_value = joint_pos_f[:, joint_idx]
                axis_local = joint_axis_local[local_idx].unsqueeze(0)
                anchor_local = joint_pos_local[local_idx].unsqueeze(0)

                if joint_type == 3:
                    joint_quat = quat_from_angle_axis(joint_value, axis_local)
                    rel_quat = quat_mul(quat0, joint_quat)
                    rel_pos = pos0 + quat_apply(quat0, anchor_local - quat_apply(joint_quat, anchor_local))
                else:
                    axis_parent = quat_apply(quat0, axis_local)
                    rel_quat = quat0
                    rel_pos = pos0 + axis_parent * joint_value.unsqueeze(-1)

            tree_quat_b[:, local_idx] = normalize(quat_mul(parent_quat_b, rel_quat))
            tree_pos_b[:, local_idx] = parent_pos_b + quat_apply(parent_quat_b, rel_pos)

        body_count = len(self.output_body_names)
        body_pos_b = torch.zeros((flat_count, body_count, 3), device=self.device, dtype=dtype)
        body_quat_b = torch.zeros((flat_count, body_count, 4), device=self.device, dtype=dtype)
        body_quat_b[..., 0] = 1.0

        valid_output = (self.output_local_idx >= 0).nonzero(as_tuple=False).squeeze(-1)
        if valid_output.numel() > 0:
            local_ids = self.output_local_idx[valid_output]
            body_pos_b[:, valid_output] = tree_pos_b[:, local_ids]
            body_quat_b[:, valid_output] = tree_quat_b[:, local_ids]

        body_pos_w = quat_apply(root_quat_f.unsqueeze(1), body_pos_b) + root_pos_f.unsqueeze(1)
        body_quat_w = normalize(quat_mul(root_quat_f.unsqueeze(1), body_quat_b))

        if self.world_output_idx >= 0:
            world_idx = self.world_output_idx
            body_pos_w[:, world_idx] = 0.0
            body_quat_w[:, world_idx] = 0.0
            body_quat_w[:, world_idx, 0] = 1.0
            body_pos_b[:, world_idx] = quat_apply_inverse(root_quat_f, -root_pos_f)
            body_quat_b[:, world_idx] = quat_conjugate(root_quat_f)

        body_pos_b = body_pos_b.reshape(prefix + (body_count, 3))
        body_quat_b = body_quat_b.reshape(prefix + (body_count, 4))
        body_pos_w = body_pos_w.reshape(prefix + (body_count, 3))
        body_quat_w = body_quat_w.reshape(prefix + (body_count, 4))
        return body_pos_b, body_quat_b, body_pos_w, body_quat_w

    def expand_minimal_motion(self, motion: MotionMinimalData, fps: float) -> MotionData:
        root_pos_w = motion.root_pos_w.to(dtype=torch.float32, device=self.device)
        root_quat_w = normalize(motion.root_quat_w.to(dtype=torch.float32, device=self.device))
        joint_pos = motion.joint_pos.to(dtype=torch.float32, device=self.device)

        root_lin_vel_w = finite_diff_torch(root_pos_w, fps, dim=1)
        root_ang_vel_w = angvel_from_quat_wxyz_torch(root_quat_w, fps, dim=1)
        joint_vel = finite_diff_torch(joint_pos, fps, dim=1)
        body_pos_b, body_quat_b, body_pos_w, body_quat_w = self.body_pose(root_pos_w, root_quat_w, joint_pos)
        body_vel_w = finite_diff_torch(body_pos_w, fps, dim=1)
        body_vel_b = quat_apply_inverse(root_quat_w.unsqueeze(2), body_vel_w - root_lin_vel_w.unsqueeze(2))
        body_angvel_b = angvel_from_quat_wxyz_torch(body_quat_b, fps, dim=1)
        body_angvel_w = quat_apply(root_quat_w.unsqueeze(2), body_angvel_b) + root_ang_vel_w.unsqueeze(2)

        return MotionData(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_pos_b=body_pos_b,
            body_vel_w=body_vel_w,
            body_vel_b=body_vel_b,
            body_quat_w=body_quat_w,
            body_quat_b=body_quat_b,
            body_angvel_w=body_angvel_w,
            body_angvel_b=body_angvel_b,
            batch_size=list(motion.batch_size),
            device=self.device,
        )

    def rewrite_motion_data_(self, motion: MotionData, fps: float) -> MotionData:
        root_pos_w = motion.root_pos_w.to(dtype=torch.float32, device=self.device)
        root_quat_w = normalize(motion.root_quat_w.to(dtype=torch.float32, device=self.device))
        body_pos_b, body_quat_b, body_pos_w, body_quat_w = self.body_pose(root_pos_w, root_quat_w, motion.joint_pos.to(dtype=torch.float32, device=self.device))
        body_vel_w = finite_diff_torch(body_pos_w, fps, dim=1)
        root_lin_vel_w = motion.root_lin_vel_w.to(dtype=torch.float32, device=self.device)
        root_ang_vel_w = motion.root_ang_vel_w.to(dtype=torch.float32, device=self.device)
        body_vel_b = quat_apply_inverse(root_quat_w.unsqueeze(2), body_vel_w - root_lin_vel_w.unsqueeze(2))
        body_angvel_b = angvel_from_quat_wxyz_torch(body_quat_b, fps, dim=1)
        body_angvel_w = quat_apply(root_quat_w.unsqueeze(2), body_angvel_b) + root_ang_vel_w.unsqueeze(2)

        motion.body_pos_b = body_pos_b.to(dtype=motion.body_pos_b.dtype)
        motion.body_quat_b = body_quat_b.to(dtype=motion.body_quat_b.dtype)
        motion.body_pos_w = body_pos_w.to(dtype=motion.body_pos_w.dtype)
        motion.body_quat_w = body_quat_w.to(dtype=motion.body_quat_w.dtype)
        motion.body_vel_w = body_vel_w.to(dtype=motion.body_vel_w.dtype)
        motion.body_vel_b = body_vel_b.to(dtype=motion.body_vel_b.dtype)
        motion.body_angvel_b = body_angvel_b.to(dtype=motion.body_angvel_b.dtype)
        motion.body_angvel_w = body_angvel_w.to(dtype=motion.body_angvel_w.dtype)
        return motion
