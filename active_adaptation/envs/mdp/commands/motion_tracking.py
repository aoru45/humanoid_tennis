import torch
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity as RigidObject
    from mjlab.sensor import ContactSensor

from active_adaptation.envs.mdp import reward, termination, observation, random_noise
from active_adaptation.utils.multimotion import (
    ProgressiveMultiMotionDataset,
)
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils import joint_order as joint_order_utils
from active_adaptation.utils.math import (
    quat_apply_inverse,
    quat_apply,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    yaw_quat,
    matrix_from_quat
)
from .utils import add_spherical_noise, clamp_norm, perturb_quaternion
from .base import Command
import re
import math
import gc
import os
import tempfile
from typing import Sequence


def _match_indices(motion_names, asset_names, patterns, name_map=None, device=None, debug=False):
    asset_idx, motion_idx = [], []
    for i, a in enumerate(asset_names):
        if any(re.match(p, a) for p in patterns):
            m = name_map.get(a, a) if name_map else a
            if m in motion_names:
                asset_idx.append(i)
                motion_idx.append(motion_names.index(m))
                if debug:
                    print(f"Matched asset '{a}' (idx {i}) to motion '{m}' (idx {motion_names.index(m)})")
    return torch.tensor(motion_idx, device=device), torch.tensor(asset_idx, device=device)

def _calc_exp_sigma(error : torch.Tensor, sigma_list : list[float], reduce_last_dim : bool = False):
    if sigma_list is None or len(sigma_list) == 0:
        raise ValueError("sigma must be provided and non-empty.")
    count = len(sigma_list)
    if reduce_last_dim:
        rewards = [torch.exp(- error / sigma).mean(dim=-1, keepdim=True) for sigma in sigma_list]
    else:
        rewards = [torch.exp(- error / sigma) for sigma in sigma_list]
    return sum(rewards) / count

def get_items_by_index(list, indexes):
    if isinstance(indexes, torch.Tensor):
        indexes = indexes.tolist()
    return [list[i] for i in indexes]

def convert_dtype(dtype_str):
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'bool': bool,
        'long': torch.long
    }
    if isinstance(dtype_str, str):
        if dtype_str not in dtype_map:
            raise ValueError(f"Unsupported dtype string: {dtype_str}")
        return dtype_map[dtype_str]
    return dtype_str


def _resolve_joint_indices(
    motion_joint_names: Sequence[str],
    asset_joint_names: Sequence[str],
    ordered_joint_names: Sequence[str],
    *,
    ignore_patterns: Sequence[str] | None = None,
    strict: bool = False,
    require_non_empty: bool = True,
    device=None,
    context: str = "joint mapping",
):
    motion_name_to_idx = {n: i for i, n in enumerate(motion_joint_names)}
    asset_name_to_idx = {n: i for i, n in enumerate(asset_joint_names)}

    selected_joint_names = []
    joint_idx_motion = []
    joint_idx_asset = []
    ignore_patterns = ignore_patterns or []

    for name in ordered_joint_names:
        if any(re.match(p, name) for p in ignore_patterns):
            continue

        in_motion = name in motion_name_to_idx
        in_asset = name in asset_name_to_idx
        if not (in_motion and in_asset):
            if strict:
                raise ValueError(
                    f"Joint '{name}' in {context} is not found in motion dataset or asset."
                )
            continue

        selected_joint_names.append(name)
        joint_idx_motion.append(motion_name_to_idx[name])
        joint_idx_asset.append(asset_name_to_idx[name])

    if require_non_empty and len(selected_joint_names) == 0:
        raise RuntimeError(f"No joints resolved for {context}.")

    return (
        selected_joint_names,
        torch.tensor(joint_idx_motion, device=device, dtype=torch.long),
        torch.tensor(joint_idx_asset, device=device, dtype=torch.long),
    )

class MotionTrackingCommand(Command):
    def __init__(self, env, dataset: dict,
                dataset_extra_keys: list[dict] = [],
                required_motion_body_patterns: list[str] = [],
                keypoint_map: dict = {},
                keypoint_patterns: list[str] = [],
                lower_keypoint_patterns: list[str] = [],
                upper_keypoint_patterns: list[str] = [],
                joint_patterns: list[str] = [],
                ignore_joint_patterns: list[str] = [],
                feet_patterns: list[str] = [],
                feet_standing_z_enter: float = 0.12,
                feet_standing_z_exit: float = 0.15,
                feet_standing_vxy_enter: float = 0.30,
                feet_standing_vxy_exit: float = 0.50,
                feet_standing_vz_enter: float = 0.20,
                feet_standing_vz_exit: float = 0.35,
                init_noise: dict[str, float] = {},
                future_steps: list[int] = [],
                student_future_steps: list[int] | None = None,
                cum_root_pos_scale: float = 0.0,
                cum_keypoint_scale: float = 0.0,
                cum_orientation_scale: float = 0.0,
                boot_indicator_max: int = 0,
                body_z_terminate_thres: float = 0.0,
                body_z_terminate_patterns: list[str] = [],
                reinit_prob: float = 0.0,
                reinit_min_steps: int = 0,
                reinit_max_steps: int = 0,
                gravity_terminate_thres: float = 0.0,
                show_tracking_debug: bool = True,
                show_reference_motion: bool = True,
                reference_alpha: float = 0.55,
                debug_mode: bool = False,):
        super().__init__(env)
        
        self.future_steps = torch.tensor(future_steps, device=self.device, dtype=torch.long)
        if student_future_steps is None:
            student_future_steps = future_steps
        self.student_future_steps = torch.tensor(
            student_future_steps, device=self.device, dtype=torch.long
        )
        if self.future_steps.numel() == 0:
            raise ValueError("future_steps must contain at least one element, and should include 0.")
        if self.student_future_steps.numel() == 0:
            raise ValueError("student_future_steps must contain at least one element, and should include 0.")
        if self.future_steps[0].item() != 0:
            raise ValueError(f"future_steps[0] must be 0, got {self.future_steps.tolist()}")
        if self.student_future_steps[0].item() != 0:
            raise ValueError(
                f"student_future_steps[0] must be 0, got {self.student_future_steps.tolist()}"
            )
        # Used for init-time upper bound on t. Negative steps do not constrain the future horizon.
        self.max_future_step = torch.clamp_min(
            torch.cat([self.future_steps, self.student_future_steps]), 0
        ).max()

        self.zero_init_prob = 0.0

        dataset_extra_keys = [
            {**k, 'dtype': convert_dtype(k['dtype'])} 
            for k in dataset_extra_keys
        ]

        self.debug_mode = debug_mode
        self.dataset = ProgressiveMultiMotionDataset(
            **dataset,
            env_size=self.num_envs,
            max_step_size=1000,
            required_motion_body_patterns=required_motion_body_patterns,
            fk_asset=self.asset,
            dataset_extra_keys=dataset_extra_keys,
            device=self.device,
            ds_device=torch.device("cpu"), # dataset will consume a lot of memory, keep it on CPU and move slices to GPU as needed
        )
        joint_vel_limits = getattr(self.asset.data, "soft_joint_vel_limits", None)
        if joint_vel_limits is None:
            joint_vel_limits = torch.zeros_like(self.asset.data.soft_joint_pos_limits)
            joint_vel_limits[..., 0] = -10.0
            joint_vel_limits[..., 1] = 10.0
        self.dataset.set_limit(
            self.asset.data.soft_joint_pos_limits,
            joint_vel_limits,
            self.asset.joint_names,
        )
        self._gravity_vec_w = self.asset.data.gravity_vec_w
        self.show_tracking_debug = bool(show_tracking_debug)
        self.show_reference_motion = bool(show_reference_motion)
        self.reference_alpha = float(min(1.0, max(0.0, reference_alpha)))
        self._reference_ghost_logged = False
        self._reference_ghost_model = self._build_reference_ghost_model()

        # bodies for full‑body keypoint tracking
        self.keypoint_patterns = keypoint_patterns
        self.lower_keypoint_patterns = lower_keypoint_patterns
        self.upper_keypoint_patterns = upper_keypoint_patterns
        self.keypoint_map = keypoint_map
        self.keypoint_idx_motion, self.keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.lower_keypoint_idx_motion, self.lower_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.lower_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )
        self.upper_keypoint_idx_motion, self.upper_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.upper_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        # joints for full‑body joint tracking
        self.joint_patterns = joint_patterns
        self.joint_idx_motion, self.joint_idx_asset = _match_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            self.joint_patterns,
            device=self.device
        )
        
        self.feet_patterns = feet_patterns
        self.feet_idx_motion, self.feet_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.feet_patterns,
            device=self.device
        )
        self.feet_standing_z_enter = float(feet_standing_z_enter)
        self.feet_standing_z_exit = float(feet_standing_z_exit)
        self.feet_standing_vxy_enter = float(feet_standing_vxy_enter)
        self.feet_standing_vxy_exit = float(feet_standing_vxy_exit)
        self.feet_standing_vz_enter = float(feet_standing_vz_enter)
        self.feet_standing_vz_exit = float(feet_standing_vz_exit)

        # all joints except ankles
        self.ignore_joint_patterns = ignore_joint_patterns
        self.all_joint_names, self.all_joint_idx_dataset, self.all_joint_idx_asset = _resolve_joint_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            self.asset.joint_names,
            ignore_patterns=self.ignore_joint_patterns,
            strict=False,
            device=self.device,
            context="all_joint_idx",
        )

        # joint indices for target_joint_pos_obs: follow asset-configured canonical order.
        self.target_joint_names, self.target_joint_idx_motion, self.target_joint_idx_asset = _resolve_joint_indices(
            self.dataset.joint_names,
            self.asset.joint_names,
            joint_order_utils.get_joint_name_order(self.asset),
            strict=True,
            device=self.device,
            context="asset canonical order",
        )
        self._target_joint_qpos_idx = 7 + self.target_joint_idx_asset
        joint_dtype = self.asset.data.joint_pos.dtype
        body_dtype = self.asset.data.body_link_pos_w.dtype
        self._target_joint_pos_bias = torch.zeros(
            self.num_envs,
            len(self.target_joint_names),
            device=self.device,
            dtype=joint_dtype,
        )
        self._root_drift_vel_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=body_dtype)
        self._root_z_offset = torch.zeros(self.num_envs, device=self.device, dtype=body_dtype)

        self.last_reset_env_ids = None

        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._cum_root_pos_scale = cum_root_pos_scale
        self._cum_keypoint_scale = cum_keypoint_scale
        self._cum_orientation_scale = cum_orientation_scale

        self.body_z_terminate_thres = body_z_terminate_thres
        self.body_z_terminate_patterns = body_z_terminate_patterns
        self.reinit_prob = reinit_prob
        self.reinit_min_steps = reinit_min_steps
        self.reinit_max_steps = reinit_max_steps
        self.gravity_terminate_thres = gravity_terminate_thres
        self.body_z_idx_motion, self.body_z_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.body_z_terminate_patterns,
            name_map=self.keypoint_map,
            device=self.device
        )

        self.feet_standing = torch.zeros(
            self.num_envs, int(self.feet_idx_motion.numel()), dtype=torch.bool, device=self.device
        )

        self.lengths = torch.full((self.num_envs,), 1, dtype=torch.int32, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boot_indicator = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.boot_indicator_max = boot_indicator_max

        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()
        self.next_init_t = torch.full((self.num_envs,), -1, dtype=torch.int32, device=self.device)
        self._reinit_requested = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        ## init noise
        self.init_noise_params = init_noise

    def _build_reference_ghost_model(self):
        """Create a tinted ghost model so reference is visually distinct."""
        try:
            import mujoco

            tmp_path = None
            with tempfile.NamedTemporaryFile(suffix=".mjb", delete=False) as f:
                tmp_path = f.name
            try:
                mujoco.mj_saveModel(self.env.sim.mj_model, tmp_path, None)
                ghost_model = mujoco.MjModel.from_binary_path(tmp_path)
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            # Tint all visual geoms toward cyan-blue to separate from policy robot colors.
            rgba = ghost_model.geom_rgba.copy()
            tint_rgb = np.array([0.10, 0.85, 1.00], dtype=rgba.dtype)
            rgba[:, :3] = 0.15 * rgba[:, :3] + 0.85 * tint_rgb
            rgba[:, 3] = 1.0
            ghost_model.geom_rgba[:] = rgba
            print("[INFO] Reference ghost color tint enabled (cyan-blue).")
            return ghost_model
        except Exception as exc:
            if self.debug_mode:
                print(f"[WARN] Failed to build tinted ghost model: {exc}")
            return self.env.sim.mj_model

    def _steps_for_horizon(self, horizon: str = "teacher") -> torch.Tensor:
        horizon_key = str(horizon).lower()
        if horizon_key == "teacher":
            return self.future_steps
        if horizon_key == "student":
            return self.student_future_steps
        raise ValueError(f"Invalid horizon '{horizon}', expected 'teacher' or 'student'.")

    def _motion_for_horizon(self, horizon: str = "teacher"):
        horizon_key = str(horizon).lower()
        if horizon_key == "teacher":
            return self._motion
        if horizon_key == "student":
            return self._motion_student
        raise ValueError(f"Invalid horizon '{horizon}', expected 'teacher' or 'student'.")

    def _motion_original_for_horizon(self, horizon: str = "teacher"):
        horizon_key = str(horizon).lower()
        if horizon_key == "teacher":
            if not hasattr(self, "_motion_original"):
                raise RuntimeError(
                    "source='original' requires original motion cache, but '_motion_original' is missing."
                )
            return self._motion_original
        if horizon_key == "student":
            if not hasattr(self, "_motion_original_student"):
                raise RuntimeError(
                    "source='original' with horizon='student' requires '_motion_original_student', but it is missing."
                )
            return self._motion_original_student
        raise ValueError(f"Invalid horizon '{horizon}', expected 'teacher' or 'student'.")

    def _apply_root_drift(self, root_pos_w: torch.Tensor, horizon: str = "teacher"):
        if str(horizon).lower() != "student":
            return root_pos_w
        steps = self._steps_for_horizon(horizon)
        offsets = steps.view(1, -1, 1) * float(self.env.step_dt)
        root_pos_w = root_pos_w + self._root_drift_vel_w.unsqueeze(1) * offsets
        root_pos_w[..., 2] = root_pos_w[..., 2] + self._root_z_offset.unsqueeze(1)
        return root_pos_w

    def sample_init(self, env_ids: torch.Tensor):
        t = self.t[env_ids]
        self.last_reset_env_ids = env_ids
        # resample motion
        lengths = self.dataset.reset(env_ids)

        n = env_ids.shape[0]

        if self.debug_mode:
            t[:] = 0
        else:
            # --- 1) reinit: rewind from previous t when requested ---
            use_reinit = torch.zeros(n, dtype=torch.bool, device=self.device)
            if self.reinit_prob > 0.0:
                use_reinit = self._reinit_requested[env_ids] & (torch.rand(n, device=self.device) < self.reinit_prob)
                rewind = torch.randint(self.reinit_min_steps, self.reinit_max_steps + 1, (n,), device=self.device, dtype=self.t.dtype)
                reinit_t = (t - rewind).clamp(min=0)
                self._reinit_requested[env_ids] = False

            # --- 2) random sample: uniform in [0, sample_interval), with zero_init_prob chance of t=0 ---
            max_start = (lengths - self.max_future_step.to(lengths.dtype) - 1).clamp_min(0)
            sample_interval = torch.minimum(max_start * 3 // 4, max_start - 100).clamp_min(0)
            t_rand = (torch.rand(n, device=self.device) * sample_interval.to(torch.float32)).floor().to(self.t.dtype)
            zero_init = torch.rand(n, device=self.device) < self.zero_init_prob
            t_rand = torch.where(zero_init, torch.zeros_like(t_rand), t_rand)

            # --- merge: reinit takes priority over random ---
            t[:] = torch.where(use_reinit, reinit_t, t_rand)

        self.lengths[env_ids] = lengths
        self.t[env_ids] = t

        motion = self.dataset.get_slice(env_ids, self.t[env_ids], 1)

        # set robot state
        self.sample_init_robot(env_ids, motion)
        self.next_init_t[env_ids] = -1
        return None

    def sample_init_robot(self, env_ids: Sequence[int], motion, lift_height: float = 0.04):
        # Get subsets for the current envs
        init_root_state = self.init_root_state[env_ids].clone()
        init_joint_pos = self.init_joint_pos[env_ids].clone()
        init_joint_vel = self.init_joint_vel[env_ids].clone()
        env_origins = self.env.scene.env_origins[env_ids]

        # Extract motion data
        motion_root_pos = motion.root_pos_w[:, 0]
        motion_root_quat = motion.root_quat_w[:, 0]
        motion_root_lin_vel = motion.root_lin_vel_w[:, 0]
        motion_root_ang_vel = motion.root_ang_vel_w[:, 0]
        motion_joint_pos = motion.joint_pos[:, 0]
        motion_joint_vel = motion.joint_vel[:, 0]

        # -------- root state ----------------------------------------------------
        init_root_state[:, :3] = env_origins + motion_root_pos
        init_root_state[:, 2] += lift_height
        root_pos_noise = torch.randn_like(init_root_state[:, :3]).clamp(-1, 1) * self.init_noise_params["root_pos"]
        root_pos_noise[:, 2].clamp_min_(0.0)
        init_root_state[:, :3] += root_pos_noise

        init_root_state[:, 3:7] = motion_root_quat
        init_root_state[:, 3:7] = perturb_quaternion(
            init_root_state[:, 3:7], self.init_noise_params["root_ori"]
        )

        init_root_state[:, 7:10] = motion_root_lin_vel
        lin_vel_noise = torch.randn_like(init_root_state[:, 7:10]).clamp(-1, 1) * self.init_noise_params["root_lin_vel"]
        init_root_state[:, 7:10] += lin_vel_noise
        
        init_root_state[:, 10:13] = motion_root_ang_vel
        ang_vel_noise = torch.randn_like(init_root_state[:, 10:13]).clamp(-1, 1) * self.init_noise_params["root_ang_vel"]
        init_root_state[:, 10:13] += ang_vel_noise

        # -------- joint state ----------------------------------------------------
        init_joint_pos[:, self.all_joint_idx_asset] = motion_joint_pos[:, self.all_joint_idx_dataset]
        self.joint_pos_boot_protect[env_ids] = init_joint_pos

        init_joint_vel[:, self.all_joint_idx_asset] = motion_joint_vel[:, self.all_joint_idx_dataset]
        joint_pos_noise = torch.randn_like(init_joint_pos).clamp(-1, 1) * self.init_noise_params["joint_pos"]
        joint_vel_noise = torch.randn_like(init_joint_vel).clamp(-1, 1) * self.init_noise_params["joint_vel"]
        init_joint_pos += joint_pos_noise
        init_joint_vel += joint_vel_noise

        # Apply the calculated states to the simulation
        self.asset.write_root_state_to_sim(init_root_state, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(init_joint_pos, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(init_joint_vel, env_ids=env_ids)
        self.asset.set_joint_position_target(init_joint_pos, env_ids=env_ids)
    
    def reset(self, env_ids):
        self.finished[env_ids] = False
        self.boot_indicator[env_ids] = self.boot_indicator_max
        self._cum_error[env_ids] = 0.0
        self.feet_standing[env_ids] = False

    @termination
    def body_z_termination(self):
        if self.body_z_terminate_thres <= 0 or self.body_z_idx_asset.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        target_z = self._motion.body_pos_w[:, 0, self.body_z_idx_motion, 2]
        target_z_min_thres = target_z - self.body_z_terminate_thres # [N, B]
        target_z_max_thres = target_z + self.body_z_terminate_thres # [N, B]
        target_z_min = target_z.amin(dim=1, keepdim=True)
        # Relax lower bound for airborne motions:
        # target_z_min <= 0.1: no relax
        # target_z_min >= 0.3: max relax = 0.2
        # in-between: linear interpolation
        lower_relax = ((target_z_min - 0.1) / 0.2).clamp(0.0, 1.0) * 0.2
        target_z_min_thres = target_z_min_thres - lower_relax

        current_z = self.asset.data.body_link_pos_w[:, self.body_z_idx_asset, 2] # [N, B]
        exceed = ((current_z < target_z_min_thres) | (current_z > target_z_max_thres)).any(dim=1, keepdim=True)
        self._reinit_requested.logical_or_(exceed.view(-1))
        return exceed

    @termination
    def gravity_dir_termination(self):
        if self.gravity_terminate_thres <= 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        motion_quat = self._motion.root_quat_w[:, 0]
        motion_g_b = quat_apply_inverse(motion_quat, self._gravity_vec_w)
        current_quat = self.asset.data.root_link_quat_w
        robot_g_b = quat_apply_inverse(current_quat, self._gravity_vec_w)
        exceed = (motion_g_b[:, 2:] - robot_g_b[:, 2:]).abs() > self.gravity_terminate_thres
        self._reinit_requested.logical_or_(exceed.view(-1))
        return exceed

    @observation
    def command_obs(self, noise_std: float = 0.0, horizon: str = "teacher", target_quat_noise_std: float = 0.0, target_pos_noise_std: float = 0.0):
        motion = self._motion_for_horizon(horizon)
        root_quat = self.asset.data.root_link_quat_w
        if noise_std > 0.0:
            root_quat = perturb_quaternion(root_quat, noise_std)
        root_quat = root_quat.unsqueeze(1)
        root_quat_future = motion.root_quat_w[:, 0:, :].clone()
        if target_quat_noise_std > 0.0:
            root_quat_future = perturb_quaternion(root_quat_future, target_quat_noise_std)
        root_quat_future0 = root_quat_future[:, 0, :].unsqueeze(1)

        root_pos_w = self._apply_root_drift(motion.root_pos_w.clone(), horizon)
        if target_pos_noise_std > 0.0:
            root_pos_w = add_spherical_noise(root_pos_w, target_pos_noise_std)
        root_pos_future = root_pos_w[:, 1:, :]
        root_pos_future0 = root_pos_w[:, 0, :].unsqueeze(1)

        # pos diff is applied in expected root frame
        pos_diff_b = quat_apply_inverse(
            root_quat_future0,
            root_pos_future - root_pos_future0
        )
        
        # quat diff is applied in current root frame
        # because we can get reliable quat from real robot IMU
        root_quat = root_quat.expand(-1, root_quat_future.shape[1], -1)
        quat_diff = quat_mul(quat_conjugate(root_quat), root_quat_future)
        rotmat_diff = matrix_from_quat(quat_diff)
        rot6d_diff = rotmat_diff[..., :, :2].transpose(-2, -1)

        return torch.cat([
            pos_diff_b.reshape(self.num_envs, -1),
            rot6d_diff.reshape(self.num_envs, -1),
        ], dim=-1)

    def command_obs_sym(self, horizon: str = "teacher"):
        steps = self._steps_for_horizon(horizon)
        return sym_utils.SymmetryTransform.cat([
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]).repeat(len(steps) - 1),
            sym_utils.SymmetryTransform(
                perm=torch.arange(6),
                signs=[1, -1, 1, -1, 1, -1]
            ).repeat(len(steps)),
        ])

    @observation
    def target_root_z_obs(self, horizon: str = "teacher", target_noise_std: float = 0.0):
        motion = self._motion_for_horizon(horizon)
        target_root_z = self._apply_root_drift(motion.root_pos_w.clone(), horizon)[:, :, 2]
        if target_noise_std > 0.0:
            target_root_z = random_noise(target_root_z, target_noise_std)
        return target_root_z.reshape(self.num_envs, -1)
    def target_root_z_obs_sym(self, horizon: str = "teacher"):
        steps = self._steps_for_horizon(horizon)
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]).repeat(len(steps))


    @observation
    def target_pos_b_obs(self):
        current_pos = self.asset.data.root_link_pos_w.unsqueeze(1) - self.env.scene.env_origins.unsqueeze(1)
        current_quat = self.asset.data.root_link_quat_w.unsqueeze(1)
        target_pos_b = quat_apply_inverse(
            current_quat,
            (self._motion.root_pos_w - current_pos)
        )
        return target_pos_b.reshape(self.num_envs, -1)
    def target_pos_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_rot_b_obs(self):
        relative_quat = quat_mul(
            quat_conjugate(self.asset.data.root_link_quat_w.unsqueeze(1)),
            self._motion.root_quat_w
        )
        rotmat = matrix_from_quat(relative_quat)
        rot6d = rotmat[..., :, :2].transpose(-2, -1)
        return rot6d.reshape(self.num_envs, -1)
    def target_rot_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(6),
            signs=[1, -1, 1, -1, 1, -1]
        ).repeat(len(self.future_steps))
    
    @observation
    def target_linvel_b_obs(self):
        target_linvel_b = quat_apply_inverse(self.asset.data.root_link_quat_w.unsqueeze(1), self._motion.root_lin_vel_w)
        return target_linvel_b.reshape(self.num_envs, -1)
    def target_linvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_angvel_b_obs(self):
        target_angvel_b = quat_apply_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            self._motion.root_ang_vel_w,
        )
        return target_angvel_b.reshape(self.num_envs, -1)
    def target_angvel_b_obs_sym(self):
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[-1., 1., -1.]
        ).repeat(len(self.future_steps))

    @observation
    def target_projected_gravity_b_obs(self, horizon: str = "teacher", target_noise_std: float = 0.0):
        motion = self._motion_for_horizon(horizon)
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        target_root_quat = motion.root_quat_w
        if target_noise_std > 0.0:
            target_root_quat = perturb_quaternion(target_root_quat, target_noise_std)
        g_b = quat_apply_inverse(target_root_quat, gravity)  # [N, S, 3]
        return g_b.reshape(self.num_envs, -1)

    def target_projected_gravity_b_obs_sym(self, horizon: str = "teacher"):
        steps = self._steps_for_horizon(horizon)
        return sym_utils.SymmetryTransform(
            perm=torch.arange(3),
            signs=[1., -1., 1.]
        ).repeat(len(steps))

    @observation
    def target_keypoints_pos_b_obs(self):
        target_keypoints_b = self._motion.body_pos_b[:, :, self.keypoint_idx_motion]

        actual_w = self.asset.data.body_link_pos_w[:, self.keypoint_idx_asset] - self.asset.data.root_link_pos_w.unsqueeze(1) # N, B, 3
        actual_b = quat_apply_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            actual_w
        ) # N, B, 3
        target_w = self._motion.body_pos_w[:, :, self.keypoint_idx_motion] - self._motion.root_pos_w[:, 0:1, :].unsqueeze(2) # [N, S, B, 3] - [N, 1, 1, 3] = [N, S, B, 3]
        target_b = quat_apply_inverse(
            self._motion.root_quat_w[:, 0:1, :].unsqueeze(2), # [N, 1, 1, 4]
            target_w
        ) # [N, S, B, 3]
        diff_b = target_b - actual_b.unsqueeze(1) # [N, S, B, 3] - [N, 1, B, 3] = [N, S, B, 3]
        return torch.cat(
            [
                target_keypoints_b.reshape(self.num_envs, -1),
                diff_b.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_keypoints_pos_b_obs_sym(self):
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1],
        ).repeat(len(self.future_steps))
        return sym_utils.SymmetryTransform.cat([transform, transform])

    @observation
    def target_keypoints_rot_b_obs(self):
        target_quat_b = self._motion.body_quat_b[:, :, self.keypoint_idx_motion]
        target_rot6d_b = matrix_from_quat(target_quat_b)[..., :, :2].transpose(-2, -1)

        actual_quat_b = quat_mul(
            quat_conjugate(self.asset.data.root_link_quat_w).unsqueeze(1),
            self.asset.data.body_link_quat_w[:, self.keypoint_idx_asset],
        )  # [N, B, 4]
        target_quat_ref = quat_mul(
            quat_conjugate(self._motion.root_quat_w[:, 0:1, :]).unsqueeze(2),  # [N, 1, 1, 4]
            self._motion.body_quat_w[:, :, self.keypoint_idx_motion],          # [N, S, B, 4]
        )  # [N, S, B, 4]
        diff_quat_b = quat_mul(
            target_quat_ref,
            quat_conjugate(actual_quat_b.unsqueeze(1)),
        )  # [N, S, B, 4]
        diff_rot6d_b = matrix_from_quat(diff_quat_b)[..., :, :2].transpose(-2, -1)

        return torch.cat(
            [
                target_rot6d_b.reshape(self.num_envs, -1),
                diff_rot6d_b.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_keypoints_rot_b_obs_sym(self):
        transform = sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1, -1, 1, -1],
        ).repeat(len(self.future_steps))
        return sym_utils.SymmetryTransform.cat([transform, transform])

    @observation
    def target_joint_pos_obs(self, noise_std: float = 0.0, horizon: str = "teacher", source: str = "modified", target_noise_std: float = 0.0, use_bias: bool = False):
        if source == "modified":
            target_joint_pos = self._motion_for_horizon(horizon).joint_pos[
                :, :, self.target_joint_idx_motion
            ]
        elif source == "original":
            target_joint_pos = self._motion_original_for_horizon(horizon).joint_pos[
                :, :, self.target_joint_idx_motion
            ]
        else:
            raise ValueError(
                f"Invalid source '{source}', expected 'modified' or 'original'."
            )
        if use_bias:
            target_joint_pos = target_joint_pos + self._target_joint_pos_bias.unsqueeze(1)
        if target_noise_std > 0.0:
            target_joint_pos = random_noise(target_joint_pos, target_noise_std)
        current_joint_pos = self.asset.data.joint_pos[:, self.target_joint_idx_asset] - self.env.action_manager.offset[:, self.target_joint_idx_asset]
        if noise_std > 0.0:
            current_joint_pos = random_noise(current_joint_pos, noise_std)
        current_joint_pos = current_joint_pos.unsqueeze(1) # N, 1, J
        target_minus_current = target_joint_pos - current_joint_pos # N, T, J
        return torch.cat(
            [
                target_joint_pos.reshape(self.num_envs, -1),
                target_minus_current.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )
    def target_joint_pos_obs_sym(self, horizon: str = "teacher", source: str = "modified"):
        steps = self._steps_for_horizon(horizon)
        transform = sym_utils.joint_space_symmetry(self.asset, self.target_joint_names).repeat(
            len(steps)
        )
        return sym_utils.SymmetryTransform.cat([transform, transform])


    @observation
    def current_keypoint_pos_b_obs(self):
        actual_w = self.asset.data.body_link_pos_w[:, self.keypoint_idx_asset]
        actual_b = quat_apply_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            actual_w - self.asset.data.root_link_pos_w.unsqueeze(1)
        )
        return actual_b.reshape(self.num_envs, -1)
    def current_keypoint_pos_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation
    def current_keypoint_rot_b_obs(self):
        root_quat_w = self.asset.data.root_link_quat_w
        actual_quat_b = quat_mul(
            quat_conjugate(root_quat_w).unsqueeze(1),
            self.asset.data.body_link_quat_w[:, self.keypoint_idx_asset],
        )
        rotmat = matrix_from_quat(actual_quat_b)
        rot6d = rotmat[..., :, :2].transpose(-2, -1)
        return rot6d.reshape(self.num_envs, -1)
    def current_keypoint_rot_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[1, -1, 1, -1, 1, -1],
        )

    @observation
    def current_keypoint_linvel_b_obs(self):
        actual_vel_w = self.asset.data.body_link_lin_vel_w[:, self.keypoint_idx_asset]
        root_vel_w = self.asset.data.root_link_lin_vel_w.unsqueeze(1)
        actual_vel_b = quat_apply_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            actual_vel_w - root_vel_w
        )
        return actual_vel_b.reshape(self.num_envs, -1)
    def current_keypoint_linvel_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(self.asset, get_items_by_index(self.asset.body_names, self.keypoint_idx_asset), sign=[1, -1, 1])

    @observation
    def current_keypoint_angvel_b_obs(self):
        actual_angvel_w = self.asset.data.body_link_ang_vel_w[:, self.keypoint_idx_asset]
        root_angvel_w = self.asset.data.root_link_ang_vel_w.unsqueeze(1)
        actual_angvel_b = quat_apply_inverse(
            self.asset.data.root_link_quat_w.unsqueeze(1),
            actual_angvel_w - root_angvel_w,
        )
        return actual_angvel_b.reshape(self.num_envs, -1)
    def current_keypoint_angvel_b_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.keypoint_idx_asset),
            sign=[-1, 1, -1],
        )
    
    @observation
    def boot_indicator_state_obs(self):
        return self.boot_indicator / self.boot_indicator_max
    def boot_indicator_state_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1.])

    @observation
    def target_feet_contact_state_obs(self):
        return self.feet_standing.float()
    def target_feet_contact_state_obs_sym(self):
        return sym_utils.cartesian_space_symmetry(
            self.asset,
            get_items_by_index(self.asset.body_names, self.feet_idx_asset),
            sign=(1,),
        )

    @reward
    def root_pos_tracking(self, sigma: list[float] | None = None):
        current_pos = self.asset.data.root_link_pos_w
        target_pos = self.reward_root_pos_w
        diff = target_pos - current_pos
        error = diff.norm(dim=-1, keepdim=True)
        self._cum_error[:, 0:1] = error / self._cum_root_pos_scale
        return _calc_exp_sigma(error, sigma)

    @reward
    def root_vel_tracking(self, sigma: list[float] | None = None):
        current_linvel_w = self.asset.data.root_link_lin_vel_w
        current_quat = self.asset.data.root_link_quat_w
        ref_linvel_w = self._motion.root_lin_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_linvel_b = quat_apply_inverse(current_quat, current_linvel_w)
        ref_linvel_b = quat_apply_inverse(ref_quat, ref_linvel_w)
        diff = ref_linvel_b - current_linvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward
    def root_rot_tracking(self, sigma: list[float] | None = None):
        current_quat = self.asset.data.root_link_quat_w
        target_quat = self.reward_root_quat_w
        diff = axis_angle_from_quat(quat_mul(
            target_quat,
            quat_conjugate(current_quat)
        ))
        error = torch.norm(diff, dim=-1, keepdim=True)
        self._cum_error[:, 1:2] = error / self._cum_orientation_scale
        return _calc_exp_sigma(error, sigma)
    
    @reward
    def root_ang_vel_tracking(self, sigma: list[float] | None = None):
        current_angvel_w = self.asset.data.root_link_ang_vel_w
        current_quat = self.asset.data.root_link_quat_w
        ref_angvel_w = self._motion.root_ang_vel_w[:, 0]
        ref_quat = self._motion.root_quat_w[:, 0, :]

        current_angvel_b = quat_apply_inverse(current_quat, current_angvel_w)
        ref_angvel_b = quat_apply_inverse(ref_quat, ref_angvel_w)
        diff = ref_angvel_b - current_angvel_b

        error = diff.norm(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward
    def keypoint_tracking(self, sigma: list[float] | None = None):
        return self._keypoint_tracking(
            self.keypoint_idx_asset,
            self.keypoint_idx_motion,
            update_cum_error=True,
            sigma=sigma,
        )
    
    @reward
    def lower_keypoint_tracking(self, sigma: list[float] | None = None):
        return self._keypoint_tracking(
            self.lower_keypoint_idx_asset,
            self.lower_keypoint_idx_motion,
            sigma=sigma,
        )

    @reward
    def upper_keypoint_tracking(self, sigma: list[float] | None = None):
        return self._keypoint_tracking(
            self.upper_keypoint_idx_asset,
            self.upper_keypoint_idx_motion,
            sigma=sigma,
        )

    @reward
    def keypoint_vel_tracking(self, sigma: list[float] | None = None):
        current_root_quat = self.asset.data.root_link_quat_w
        actual_vel_w = self.asset.data.body_link_lin_vel_w[:, self.keypoint_idx_asset]
        actual_vel_b = quat_apply_inverse(
            current_root_quat.unsqueeze(1),
            actual_vel_w - self.asset.data.root_link_lin_vel_w.unsqueeze(1),
        )

        target_vel_b = self._motion.body_vel_b[:, 0, self.keypoint_idx_motion]
        error = (target_vel_b - actual_vel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward
    def keypoint_rot_tracking(self, sigma: list[float] | None = None):
        current_root_quat = self.asset.data.root_link_quat_w
        actual_quat_b = quat_mul(
            quat_conjugate(current_root_quat).unsqueeze(1),
            self.asset.data.body_link_quat_w[:, self.keypoint_idx_asset],
        )
        target_quat_b = self._motion.body_quat_b[:, 0, self.keypoint_idx_motion]
        diff = axis_angle_from_quat(quat_mul(target_quat_b, quat_conjugate(actual_quat_b)))
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward
    def keypoint_angvel_tracking(self, sigma: list[float] | None = None):
        current_root_quat = self.asset.data.root_link_quat_w
        actual_angvel_w = self.asset.data.body_link_ang_vel_w[:, self.keypoint_idx_asset]
        root_angvel_w = self.asset.data.root_link_ang_vel_w.unsqueeze(1)
        actual_angvel_b = quat_apply_inverse(
            current_root_quat.unsqueeze(1),
            actual_angvel_w - root_angvel_w,
        )
        target_angvel_b = self._motion.body_angvel_b[:, 0, self.keypoint_idx_motion]
        error = (target_angvel_b - actual_angvel_b).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    def _keypoint_tracking(self, keypoint_idx_asset: torch.Tensor, keypoint_idx_motion: torch.Tensor, update_cum_error: bool = False, sigma: list[float] | None = None):
        actual = self.asset.data.body_link_pos_w[:, keypoint_idx_asset]
        target = self.reward_keypoints_w[:, keypoint_idx_motion]
        diff = target - actual
        error = diff.norm(dim=-1).mean(dim=-1, keepdim=True)
        if update_cum_error:
            self._cum_error[:, 2:3] = error / self._cum_keypoint_scale
        return _calc_exp_sigma(error, sigma)

    @reward
    def joint_pos_tracking(self, sigma: list[float] | None = None):
        actual = self.asset.data.joint_pos[:, self.joint_idx_asset]
        target = self._motion.joint_pos[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    @reward
    def joint_vel_tracking(self, sigma: list[float] | None = None):
        actual = self.asset.data.joint_vel[:, self.joint_idx_asset]
        target = self._motion.joint_vel[:, 0, self.joint_idx_motion]
        error = (target - actual).abs().mean(dim=-1, keepdim=True)
        return _calc_exp_sigma(error, sigma)

    def update_reward_target_raw(self):
        delta_quat = quat_mul(
            self.asset.data.root_link_quat_w,
            quat_conjugate(self._motion.root_quat_w[:, 0])
        )
        tgt_rel = self._motion.body_pos_w[:, 0] - self._motion.root_pos_w[:, 0].unsqueeze(1)
        self.reward_keypoints_w = quat_apply(delta_quat.unsqueeze(1), tgt_rel) + self.asset.data.root_link_pos_w.unsqueeze(1)

        if not self.env.student_train:
            self.reward_root_pos_w = self._motion.root_pos_w[:, 0] + self.env.scene.env_origins
            self.reward_root_quat_w = self._motion.root_quat_w[:, 0]
        else:
            steps = 50  # calc t+50 target root pos/rot from current root pos/rot
            # prepare future root pos/rot cache
            if hasattr(self, 'ts_root_pos_w') is False:
                self.ts_root_pos_w = torch.zeros(self.num_envs, steps, 3, device=self.device, dtype=torch.float32)
            # update only for reset envs
            if self.last_reset_env_ids is not None:
                future_motion = self.dataset.get_slice(self.last_reset_env_ids, self.t[self.last_reset_env_ids], steps=steps)
                self.ts_root_pos_w[self.last_reset_env_ids] = future_motion.root_pos_w + self.env.scene.env_origins[self.last_reset_env_ids].unsqueeze(1)
            # get current root pos/rot from cache
            reward_pos = self.ts_root_pos_w[:, 0].clone()
            # roll forward the cache
            self.ts_root_pos_w[:, :-1] = self.ts_root_pos_w[:, 1:].clone()
            # compute target root pos/rot at t+steps
            current_pos_t = self.asset.data.root_link_pos_w
            current_quat_t = self.asset.data.root_link_quat_w

            ref_motion_plus = self.dataset.get_slice(None, self.t, steps=torch.tensor([steps], device=self.device, dtype=torch.int64))
            ref_pos_t = self._motion.root_pos_w[:, 0]
            ref_pos_t_plus = ref_motion_plus.root_pos_w[:, 0]
            ref_quat_t = self._motion.root_quat_w[:, 0]

            delta_quat = quat_mul(current_quat_t, quat_conjugate(ref_quat_t))
            self.ts_root_pos_w[:, -1] = quat_apply(delta_quat, (ref_pos_t_plus - ref_pos_t)) + current_pos_t

            self.reward_root_pos_w = reward_pos
            self.reward_root_quat_w = ref_quat_t

    def before_update(self):
        self.t = torch.clamp_max(self.t + 1, self.lengths - 1)
        self.finished[:] = self.t >= self.lengths - 1
        self.boot_indicator[:] = torch.clamp_min(self.boot_indicator - 1, 0)

        self._motion = self.dataset.get_slice(None, self.t, steps=self.future_steps)
        if torch.equal(self.student_future_steps, self.future_steps):
            self._motion_student = self._motion
        else:
            self._motion_student = self.dataset.get_slice(
                None, self.t, steps=self.student_future_steps
            )

        feet_vel_w = self._motion.body_vel_w[:, 0, self.feet_idx_motion, :]
        feet_pos_w = self._motion.body_pos_w[:, 0, self.feet_idx_motion, :]
        root_vxy = self._motion.root_lin_vel_w[:, 0, :2].norm(dim=-1, keepdim=True).clamp_min(1.0)

        feet_vxy = feet_vel_w[..., :2].norm(dim=-1)
        feet_vz_abs = feet_vel_w[..., 2].abs()
        feet_z = feet_pos_w[..., 2]

        enter_contact = (
            (feet_z < self.feet_standing_z_enter)
            & (feet_vxy < self.feet_standing_vxy_enter * root_vxy)
            & (feet_vz_abs < self.feet_standing_vz_enter * root_vxy)
        )
        exit_contact = (
            (feet_z > self.feet_standing_z_exit)
            | (feet_vxy > self.feet_standing_vxy_exit * root_vxy)
            | (feet_vz_abs > self.feet_standing_vz_exit * root_vxy)
        )

        self.feet_standing = (self.feet_standing & (~exit_contact)) | enter_contact

        self.update_reward_target_raw()
        # self.sample_init_robot(torch.arange(self.num_envs, device=self.device), self._motion, lift_height=0.0)

    def update(self):
        self.dataset.update()
        if self.last_reset_env_ids is not None:
            self.last_reset_env_ids = None

    def debug_draw(self):
        if self.show_tracking_debug:
            root_pos = self.asset.data.root_link_pos_w    # [N,1,3]
            root_quat = self.asset.data.root_link_quat_w  # [N,1,4]
            target_root_quat = self.reward_root_quat_w  # [N,1,4]
            heading_rel = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
            heading_world = quat_apply(root_quat, heading_rel)
            heading_world_target = quat_apply(target_root_quat, heading_rel)

            # —— original world‐frame drawing —— 
            target_keypoints_w = self.reward_keypoints_w[:, self.keypoint_idx_motion]
            target_keypoints_w_raw = self._motion.body_pos_w[:, 0, self.keypoint_idx_motion] + self.env.scene.env_origins.unsqueeze(1)
            robot_keypoints_w = self.asset.data.body_link_pos_w[:, self.keypoint_idx_asset]

            # draw points and error vectors
            self.env.debug_draw.point(
                target_keypoints_w.reshape(-1, 3), color=(1, 0, 0, 1)
            )
            self.env.debug_draw.point(
                robot_keypoints_w.reshape(-1, 3), color=(0, 1, 0, 1)
            )
            self.env.debug_draw.point(
                target_keypoints_w_raw.reshape(-1, 3), color=(1, 0.5, 0, 1), size=40.0
            )
            self.env.debug_draw.vector(
                robot_keypoints_w.reshape(-1, 3),
                (target_keypoints_w - robot_keypoints_w).reshape(-1, 3),
                color=(0, 0, 1, 1)
            )
            
            self.env.debug_draw.vector(
                root_pos.reshape(-1, 3),
                heading_world.reshape(-1, 3),
                color=(0, 0, 1, 2)
            )
            
            self.env.debug_draw.vector(
                self.reward_root_pos_w.reshape(-1, 3),
                heading_world_target.reshape(-1, 3),
                color=(1, 0, 0, 2)
            )
            if self.feet_idx_motion.numel() > 0 and self.feet_standing.any():
                target_feet_w = self.reward_keypoints_w[:, self.feet_idx_motion]
                standing_points = target_feet_w[self.feet_standing]
                if standing_points.numel() > 0:
                    self.env.debug_draw.point(
                        standing_points,
                        color=(1, 1, 0, 1),
                        size=20.0,
                    )
        self._draw_reference_ghost()

    def _draw_reference_ghost(self):
        if not self.show_reference_motion:
            return
        if not hasattr(self.env, "sim") or not hasattr(self.env.sim, "mj_model"):
            return
        sim_qpos = getattr(self.env.sim.data, "qpos", None)
        if sim_qpos is None or sim_qpos.ndim != 2 or sim_qpos.shape[-1] < 7:
            return
        if not hasattr(self, "reward_root_pos_w") or not hasattr(self, "reward_root_quat_w"):
            return
        if not hasattr(self, "_motion"):
            return

        ref_qpos = sim_qpos.clone()
        ref_qpos[:, 0:3] = self.reward_root_pos_w
        ref_qpos[:, 3:7] = self.reward_root_quat_w

        target_joint_pos = self._motion.joint_pos[:, 0, self.target_joint_idx_motion]
        target_joint_qpos_idx = self._target_joint_qpos_idx
        valid_mask = target_joint_qpos_idx < ref_qpos.shape[-1]
        if valid_mask.any():
            ref_qpos[:, target_joint_qpos_idx[valid_mask]] = target_joint_pos[:, valid_mask]

        ghost_model = self._reference_ghost_model
        if getattr(ghost_model, "nq", -1) != ref_qpos.shape[-1]:
            ghost_model = self.env.sim.mj_model

        try:
            self.env.debug_draw.ghost(
                qpos=ref_qpos,
                model=ghost_model,
                alpha=self.reference_alpha,
            )
            if not self._reference_ghost_logged:
                print(
                    f"[INFO] Reference ghost enabled (alpha={self.reference_alpha:.2f}, "
                    f"envs={ref_qpos.shape[0]}, nq={ref_qpos.shape[1]})."
                )
                self._reference_ghost_logged = True
        except Exception as exc:
            if self.debug_mode:
                print(f"[WARN] Failed to draw reference ghost: {exc}")

class MotionTrackingComplianceCommand(MotionTrackingCommand):
    def __init__(
        self,
        env,
        dataset: dict,
        upper_force_keypoint_patterns: list[str] = [".*_hand_mimic", ".*wrist_roll_link.*"],
        modify_ac_len_range: Sequence[int] = (100, 800),
        modify_b_ratio_range: Sequence[float] = (0.3, 0.7),
        modify_fps: float = 50.0,
        modify_b_dataset_prob: float = 0.8,
        modify_joint_left_patterns: list[str] = (".*left_.*shoulder.*", ".*left_.*elbow.*"),
        modify_joint_right_patterns: list[str] = (".*right_.*shoulder.*", ".*right_.*elbow.*"),
        modify_resample_countdown_steps: int = 5000,
        modify_resample_prob: float = 0.75,
        force_threshold_range: Sequence[float] = (10.0, 20.0),
        *args,
        **kwargs,
    ):
        super().__init__(env, dataset, *args, **kwargs)
        self.modify_resample_countdown_steps = int(modify_resample_countdown_steps)
        self.modify_resample_prob = float(modify_resample_prob)
        self.modify_countdown = torch.zeros(
            (self.num_envs,),
            dtype=torch.int32,
            device=self.device,
        )
        self.modify_countdown[:] = torch.randint(0, self.modify_resample_countdown_steps, (self.num_envs,), device=self.device)
        self.modify_suitable_flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.compliance_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_threshold_min = float(force_threshold_range[0])
        self.force_threshold_max = float(force_threshold_range[1])
        self.force_threshold = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.force_kp = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._motion_modified_mask = torch.zeros(
            (self.num_envs, 1), dtype=torch.bool, device=self.device
        )

        self.upper_force_keypoint_patterns = upper_force_keypoint_patterns
        self.upper_force_keypoint_idx_motion, self.upper_force_keypoint_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            self.upper_force_keypoint_patterns,
            name_map=self.keypoint_map,
            device=self.device,
        )
        self.torso_idx_motion, self.torso_idx_asset = _match_indices(
            self.dataset.body_names,
            self.asset.body_names,
            [".*torso.*"],
            name_map=self.keypoint_map,
            device=self.device,
        )
        if self.torso_idx_motion.numel() != 1 or self.torso_idx_asset.numel() != 1:
            raise ValueError(
                "Torso index matching must resolve exactly one body in both motion and asset. "
                f"got motion={self.torso_idx_motion.tolist()}, asset={self.torso_idx_asset.tolist()}"
            )
        self.upper_force_applied = torch.zeros((self.num_envs, len(self.upper_force_keypoint_idx_asset), 3), dtype=torch.float32, device=self.device)
        self.torso_force_applied = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.torso_torque_applied = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.net_force_limit_max = torch.as_tensor(20.0, dtype=torch.float32, device=self.device)
        self.net_torque_limit_max = torch.as_tensor(10.0, dtype=torch.float32, device=self.device)
        self.net_limit_full_progress = 0.75
        self.net_force_limit = torch.zeros((), dtype=torch.float32, device=self.device)
        self.net_torque_limit = torch.zeros((), dtype=torch.float32, device=self.device)

        self.dataset.setup_joint_modification(
            ac_len_range=modify_ac_len_range,
            b_ratio_range=modify_b_ratio_range,
            fps=modify_fps,
            modify_b_dataset_prob=modify_b_dataset_prob,
            modify_joint_pos_bank_size=20000,
            modify_joint_left_patterns=modify_joint_left_patterns,
            modify_joint_right_patterns=modify_joint_right_patterns,
            fk_asset=self.asset,
        )
        self.modify_suitable_flags = self._compute_modify_suitable_flags()
        self.step_schedule(0.0, None)

    def step_schedule(self, progress: float, iters: int | None = None):
        if self.env.student_train:
            ramp = 1.0
        else:
            p = float(min(max(progress, 0.0), 1.0))
            ramp = min(p / self.net_limit_full_progress, 1.0)
        self.net_force_limit.copy_(self.net_force_limit_max * ramp)
        self.net_torque_limit.copy_(self.net_torque_limit_max * ramp)

    def _compute_modify_suitable_flags(self) -> torch.Tensor:
        if not isinstance(self.dataset, ProgressiveMultiMotionDataset):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.feet_idx_motion.numel() == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        buf = self.dataset._buf_A
        lengths = self.dataset._len_A.to(torch.long)
        E, T = lengths.shape[0], buf.root_pos_w.shape[1]
        t = torch.arange(T, device=self.device).unsqueeze(0).expand(E, T)
        valid = t < lengths.unsqueeze(1)

        root_z = buf.root_pos_w[:, :, 2]
        root_speed = buf.root_lin_vel_w.norm(dim=-1)
        root_angvel = buf.root_ang_vel_w.norm(dim=-1)
        feet_z = buf.body_pos_w[:, :, self.feet_idx_motion, 2]

        root_z_ok = (
            ((root_z >= 0.6) & (root_z <= 0.85)) | (~valid)
        ).all(dim=1)
        root_speed_ok = ((root_speed < 2.0) | (~valid)).all(dim=1)
        root_angvel_ok = ((root_angvel < 3.0) | (~valid)).all(dim=1)
        feet_ok = ((feet_z < 0.35) | (~valid.unsqueeze(-1))).all(dim=(1, 2))

        return root_z_ok & root_speed_ok & root_angvel_ok & feet_ok

    def _maybe_resample_joint_modify(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        due_mask = self.modify_countdown[env_ids] <= 0
        if not due_mask.any():
            return

        due_env_ids = env_ids[due_mask]
        self.compliance_flag[due_env_ids] = False
        self.modify_countdown[due_env_ids] = self.modify_resample_countdown_steps

        suitable_mask = self.modify_suitable_flags[due_env_ids]
        candidate_env_ids = due_env_ids[suitable_mask]

        if candidate_env_ids.numel() > 0:
            prob_mask = torch.rand(candidate_env_ids.numel(), device=self.device) < self.modify_resample_prob
            modify_env_ids = candidate_env_ids[prob_mask]
            self.compliance_flag[modify_env_ids] = True
        else:
            modify_env_ids = candidate_env_ids

        self.dataset.modify_joint(due_env_ids, modify_env_ids)

    @observation
    def compliance_flag_obs(self):
        flag = self.compliance_flag.float().unsqueeze(-1)
        return torch.cat(
            [
                flag,
                self.force_threshold * flag,
                self.force_kp * flag,
            ],
            dim=-1,
        )
    def compliance_flag_obs_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., 1., 1.])

    def sample_init(self, env_ids: torch.Tensor):
        self._maybe_resample_joint_modify(env_ids)
        return super().sample_init(env_ids)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        sampled_force_threshold = torch.rand(
            (env_ids.numel(), 1), device=self.device, dtype=torch.float32
        ) * (self.force_threshold_max - self.force_threshold_min) + self.force_threshold_min
        self.force_threshold[env_ids] = sampled_force_threshold
        self.force_kp[env_ids] = sampled_force_threshold / 0.05
        self.upper_force_applied[env_ids] = 0.0
        self.torso_force_applied[env_ids] = 0.0
        self.torso_torque_applied[env_ids] = 0.0

    def before_update(self):
        super().before_update()
        self.modify_countdown -= 1
        self._motion_original = self.dataset.get_slice_original(
            None, self.t, steps=self.future_steps
        )
        if torch.equal(self.student_future_steps, self.future_steps):
            self._motion_original_student = self._motion_original
        else:
            self._motion_original_student = self.dataset.get_slice_original(
                None, self.t, steps=self.student_future_steps
            )
        self._motion_modified_mask = self.dataset.get_slice_modified_mask(None, self.t, steps=1)
    
    def step(self, substep):
        super().step(substep)
        raw_target_pos_torso = quat_apply_inverse(
            self._motion.body_quat_w[:, 0, self.torso_idx_motion],
            self._motion_original.body_pos_w[:, 0, self.upper_force_keypoint_idx_motion]
            - self._motion.body_pos_w[:, 0, self.torso_idx_motion],
        )
        modified_target_pos_torso = quat_apply_inverse(self._motion.body_quat_w[:, 0, self.torso_idx_motion], self._motion.body_pos_w[:, 0, self.upper_force_keypoint_idx_motion] - self._motion.body_pos_w[:, 0, self.torso_idx_motion])
        current_target_pos_torso = quat_apply_inverse(
            self.asset.data.body_link_quat_w[:, self.torso_idx_asset],
            self.asset.data.body_link_pos_w[:, self.upper_force_keypoint_idx_asset] - self.asset.data.body_link_pos_w[:, self.torso_idx_asset]
        )

        upper_force_applied_torso = clamp_norm(
            (modified_target_pos_torso - raw_target_pos_torso) * self.force_kp.unsqueeze(-1), self.force_threshold.unsqueeze(-1)
        ) + clamp_norm((modified_target_pos_torso - current_target_pos_torso) * 100, 5.0)
        active_force_mask = self.compliance_flag & self._motion_modified_mask[:, 0]
        upper_force_applied_torso[~active_force_mask] = 0.0
        self.upper_force_applied[:] = quat_apply(
            self.asset.data.body_link_quat_w[:, self.torso_idx_asset],
            upper_force_applied_torso
        )
        dist = add_spherical_noise(torch.zeros_like(current_target_pos_torso), 0.02)
        torque_pull = self.upper_force_applied.cross(dist, dim=-1)

        # Limit net wrench about torso and compensate exceeded part on torso.
        upper_pos_w = self.asset.data.body_link_pos_w[:, self.upper_force_keypoint_idx_asset]
        torso_pos_w = self.asset.data.body_link_pos_w[:, self.torso_idx_asset]
        r_w = upper_pos_w - torso_pos_w
        force_net_w = self.upper_force_applied.sum(dim=1)
        torque_net_w = torch.cross(r_w, self.upper_force_applied, dim=-1).sum(dim=1) + torque_pull.sum(dim=1)
        force_allow_w = clamp_norm(force_net_w, self.net_force_limit)
        torque_allow_w = clamp_norm(torque_net_w, self.net_torque_limit)
        self.torso_force_applied[:] = force_allow_w - force_net_w
        self.torso_torque_applied[:] = torque_allow_w - torque_net_w

        force_all = torch.cat([self.upper_force_applied, self.torso_force_applied.unsqueeze(1)], dim=1)
        torque_all = torch.cat([torque_pull, self.torso_torque_applied.unsqueeze(1)], dim=1)
        body_ids_all = torch.cat([self.upper_force_keypoint_idx_asset, self.torso_idx_asset], dim=0)
        self.asset.write_external_wrench_to_sim(forces=force_all, torques=torque_all, body_ids=body_ids_all)
        
    
    def debug_draw(self):
        super().debug_draw()
        if self.compliance_flag.any():
            active_mask = self.compliance_flag
            marker_pos = self.asset.data.root_link_pos_w[active_mask].clone()
            marker_pos[:, 2] += 0.5
            self.env.debug_draw.point(
                marker_pos,
                color=(0, 0, 1, 1),
                size=20.0,
            )
            force_vis_scale = 0.02
            upper_pos_w = self.asset.data.body_link_pos_w[active_mask][:, self.upper_force_keypoint_idx_asset]
            upper_force_w = self.upper_force_applied[active_mask] * force_vis_scale
            self.env.debug_draw.vector(
                upper_pos_w.reshape(-1, 3),
                upper_force_w.reshape(-1, 3),
                color=(0, 0.8, 1, 1),
            )
            torso_pos_w = self.asset.data.body_link_pos_w[active_mask][:, self.torso_idx_asset]
            torso_force_w = self.torso_force_applied[active_mask].unsqueeze(1) * force_vis_scale
            self.env.debug_draw.vector(
                torso_pos_w.reshape(-1, 3),
                torso_force_w.reshape(-1, 3),
                color=(1, 0.4, 0, 1),
            )
        if not hasattr(self, "_motion_original"):
            return
        origin_upper_pos_w = (
            self._motion_original.body_pos_w[:, 0, self.upper_force_keypoint_idx_motion]
            + self.env.scene.env_origins.unsqueeze(1)
        )
        self.env.debug_draw.point(
            origin_upper_pos_w.reshape(-1, 3),
            color=(0, 1, 0, 1),
            size=20.0,
        )
