import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple, cast
from collections.abc import Mapping
import warp as wp

import mjlab.utils.lab_api.string as string_utils
from mjlab.managers.event_manager import RecomputeLevel
from active_adaptation.envs.mdp.commands.utils import add_spherical_noise, rand_points_disk


if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from active_adaptation.envs.base import _Env
    from active_adaptation.envs.mdp.commands.motion_tracking import MotionTrackingCommand


class Randomization:
    _RECOMPUTE_DERIVED_FIELDS = {
        RecomputeLevel.none: (),
        RecomputeLevel.set_const_fixed: ("body_subtreemass",),
        RecomputeLevel.set_const_0: (
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
        RecomputeLevel.set_const: (
            "body_subtreemass",
            "dof_invweight0",
            "body_invweight0",
            "tendon_length0",
            "tendon_invweight0",
        ),
    }

    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device
    
    def startup(self):
        pass
    
    def reset(self, env_ids: torch.Tensor):
        pass
    
    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass

    def ensure_model_fields_expanded(self, *fields: str):
        missing = tuple(f for f in fields if f not in self.env.sim.expanded_fields)
        if missing:
            self.env.sim.expand_model_fields(missing)

    def ensure_recompute_fields_expanded(self, level: RecomputeLevel):
        self.ensure_model_fields_expanded(*self._RECOMPUTE_DERIVED_FIELDS[level])


RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]

class motor_params_implicit(Randomization):
    def __init__(
        self,
        env,
        stiffness_range,
        damping_range,
        armature_range,
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded(
            "actuator_gainprm",
            "actuator_biasprm",
            "dof_armature",
        )
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const_0)
        self.asset: Articulation = self.env.scene["robot"]

        # 存下区间字典
        self.stiffness_range = dict(stiffness_range)
        self.damping_range   = dict(damping_range)
        self.armature_range  = dict(armature_range)
        self.model = self.env.sim.model
        # ------- stiffness / damping via actuator gains (mjlab randomize_pd_gains style) -------
        kp_ids, _, kp_ranges = string_utils.resolve_matching_names_values(
            self.stiffness_range, self.asset.actuator_names
        )
        kd_ids, _, kd_ranges = string_utils.resolve_matching_names_values(
            self.damping_range, self.asset.actuator_names
        )

        self.kp_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.tensor(kp_ids, device=self.device, dtype=torch.long)
        ]
        self.kd_ctrl_ids = self.asset.indexing.ctrl_ids[
            torch.tensor(kd_ids, device=self.device, dtype=torch.long)
        ]

        default_gainprm = self.env.sim.get_default_field("actuator_gainprm")
        default_biasprm = self.env.sim.get_default_field("actuator_biasprm")

        self.kp_gain_def = default_gainprm[self.kp_ctrl_ids, 0]
        self.kp_bias_def = default_biasprm[self.kp_ctrl_ids, 1]
        self.kd_bias_def = default_biasprm[self.kd_ctrl_ids, 2]

        kp_low, kp_high = torch.tensor(kp_ranges, device=self.device).unbind(1)
        kd_low, kd_high = torch.tensor(kd_ranges, device=self.device).unbind(1)
        self._validate_log_uniform_range("stiffness_range", kp_low, kp_high)
        self._validate_log_uniform_range("damping_range", kd_low, kd_high)

        self.kp_low = kp_low
        self.kp_high = kp_high
        self.kd_low = kd_low
        self.kd_high = kd_high

        # ------- armature (改为相对值) -------
        arm_ids, _, arm_ranges = string_utils.resolve_matching_names_values(
            self.armature_range, self.asset.joint_names
        )
        self.arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self.arm_dof_ids = self.asset.indexing.joint_v_adr[self.arm_ids]
        default_armature = self.env.sim.get_default_field("dof_armature")
        self.arm_def = default_armature[self.arm_dof_ids]

        arm_low, arm_high = torch.tensor(arm_ranges, device=self.device).unbind(1)
        self._validate_log_uniform_range("armature_range", arm_low, arm_high)
        self.arm_low = arm_low
        self.arm_high = arm_high

    def _validate_log_uniform_range(self, range_name: str, low: torch.Tensor, high: torch.Tensor):
        if torch.any(low <= 0.0) or torch.any(high <= 0.0):
            raise ValueError(f"{range_name} must be strictly positive for log-uniform sampling, got low={low.tolist()}, high={high.tolist()}")
        if torch.any(high < low):
            raise ValueError(f"{range_name} must satisfy low <= high, got low={low.tolist()}, high={high.tolist()}")

    def _rand_log_uniform(self, n_env: int, low: torch.Tensor, high: torch.Tensor):
        low_expand = low.unsqueeze(0).expand(n_env, -1)
        high_expand = high.unsqueeze(0).expand(n_env, -1)
        return log_uniform(low_expand, high_expand)

    def _randomize_pd_gain(self, env_ids: torch.Tensor):
        n_env = env_ids.numel()
        if n_env == 0:
            return

        if self.kp_ctrl_ids.numel() > 0:
            kp_samples = self._rand_log_uniform(n_env, self.kp_low, self.kp_high)
            kp_gain = self.kp_gain_def.unsqueeze(0) * kp_samples
            kp_bias = self.kp_bias_def.unsqueeze(0) * kp_samples
            self.model.actuator_gainprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 0] = kp_gain
            self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kp_ctrl_ids, 1] = kp_bias

        if self.kd_ctrl_ids.numel() > 0:
            kd_samples = self._rand_log_uniform(n_env, self.kd_low, self.kd_high)
            kd_bias = self.kd_bias_def.unsqueeze(0) * kd_samples
            self.model.actuator_biasprm[env_ids.unsqueeze(1), self.kd_ctrl_ids, 2] = kd_bias

    def startup(self):
        n_env = self.num_envs

        # armature
        if self.arm_dof_ids.numel() > 0:
            arma = self._rand_log_uniform(n_env, self.arm_low, self.arm_high)
            self.model.dof_armature[:, self.arm_dof_ids] = (
                self.arm_def.unsqueeze(0) * arma
            )
            self.env.sim.recompute_constants(RecomputeLevel.set_const_0)

        if self.arm_dof_ids.numel() > 0:
            assert torch.allclose(self.model.dof_armature[:, self.arm_dof_ids], self.arm_def.unsqueeze(0) * arma)

    # ----------------------------------------------------------
    def reset(self, env_ids):
        self._randomize_pd_gain(env_ids)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range=(0.6, 1.0),
        solref_time_constant_range=(0.02, 0.02),
        solref_dampratio_range=(1.0, 1.0),
        homogeneous: bool = False,
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded("geom_friction", "geom_solref")
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)

        self.static_friction_range = static_friction_range
        self.solref_time_constant_range = solref_time_constant_range
        self.solref_dampratio_range = solref_dampratio_range
        self.homogeneous = homogeneous

        if len(self.body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided names for material perturbation."
            )

        local_body_ids = torch.as_tensor(
            self.body_ids, device=self.device, dtype=torch.long
        )
        self.global_body_ids = self.asset.indexing.body_ids[local_body_ids]
        selected_body_set = set(self.global_body_ids.cpu().tolist())

        geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
        geom_names = self.asset.geom_names
        selected_geom_local: list[int] = []
        selected_geom_global: list[int] = []
        selected_geom_names: list[str] = []

        cpu_model = self.env.sim.mj_model
        for local_idx, global_idx in enumerate(geom_global_ids):
            body_id = int(cpu_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                selected_geom_local.append(local_idx)
                selected_geom_global.append(global_idx)
                selected_geom_names.append(geom_names[local_idx])

        if not selected_geom_global:
            raise ValueError(
                "No geoms found for the specified bodies when configuring material perturbation."
            )

        self.geom_local_ids = torch.as_tensor(
            selected_geom_local, device=self.device, dtype=torch.long
        )
        self.geom_global_ids = torch.as_tensor(
            selected_geom_global, device=self.device, dtype=torch.long
        )
        self.geom_names = selected_geom_names

    def startup(self):
        logging.info(f"Randomize body materials of {self.geom_names} upon startup.")

        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        shape = (self.num_envs, sample_cols)

        sf = sample_uniform(shape, *self.static_friction_range, device=self.device)
        tc = sample_uniform(shape, *self.solref_time_constant_range, device=self.device)
        dr = sample_log_uniform(shape, *self.solref_dampratio_range, device=self.device)

        if sample_cols == 1:
            sf = sf.expand(-1, num_geoms)
            tc = tc.expand(-1, num_geoms)
            dr = dr.expand(-1, num_geoms)

        model = self.env.sim.model
        model.geom_friction[:, self.geom_global_ids, 0] = sf
        model.geom_solref[:, self.geom_global_ids, 0] = tc
        model.geom_solref[:, self.geom_global_ids, 1] = dr

        assert torch.allclose(model.geom_friction[:, self.geom_global_ids, 0], sf)
        assert torch.allclose(model.geom_solref[:, self.geom_global_ids, 0], tc)
        assert torch.allclose(model.geom_solref[:, self.geom_global_ids, 1], dr)

class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.ensure_model_fields_expanded("body_mass", "body_inertia")
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const)
        self.asset: Articulation = self.env.scene["robot"]
        if not perturb_ranges:
            raise ValueError("perturb_body_mass requires at least one body range entry.")

        body_ids, body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided patterns for mass perturbation."
            )

        self.body_names = body_names
        self.local_body_ids = torch.as_tensor(body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self.mass_ranges = torch.as_tensor(values, device=self.device, dtype=torch.float32)

        model = self.env.sim.model
        self._default_mass = model.body_mass[:, self.global_body_ids].clone()
        self._default_inertia = model.body_inertia[:, self.global_body_ids].clone()


    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")
        low = self.mass_ranges[:, 0].unsqueeze(0)
        high = self.mass_ranges[:, 1].unsqueeze(0)
        scale = uniform_like(self._default_mass, low, high)

        model = self.env.sim.model
        new_mass = self._default_mass * scale
        new_inertia = self._default_inertia * scale.unsqueeze(-1)
        model.body_mass[:, self.global_body_ids] = new_mass
        model.body_inertia[:, self.global_body_ids] = new_inertia
        self.env.sim.recompute_constants(RecomputeLevel.set_const)

        assert torch.allclose(model.body_mass[:, self.global_body_ids], new_mass)
        assert torch.allclose(model.body_inertia[:, self.global_body_ids], new_inertia)


class perturb_body_com(Randomization):
    def __init__(self, env, body_names = ".*", com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.ensure_model_fields_expanded("body_ipos")
        self.ensure_recompute_fields_expanded(RecomputeLevel.set_const)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError("No bodies matched the provided names for COM perturbation.")

        self.com_range = tuple(com_range)
        self.local_body_ids = torch.as_tensor(self.body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]

        model = self.env.sim.model
        self._default_body_ipos = model.body_ipos[:, self.global_body_ids].clone()
    
    def startup(self):
        num_bodies = self.global_body_ids.numel()
        low, high = self.com_range
        offsets = sample_uniform((self.num_envs, num_bodies, 3), low, high, device=self.device)

        model = self.env.sim.model
        new_ipos = self._default_body_ipos + offsets
        model.body_ipos[:, self.global_body_ids] = new_ipos
        self.env.sim.recompute_constants(RecomputeLevel.set_const)

        assert torch.allclose(model.body_ipos[:, self.global_body_ids], new_ipos)


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        low = self.offset_range[:, 0].unsqueeze(0)
        high = self.offset_range[:, 1].unsqueeze(0)
        offset = uniform(
            low.expand(env_ids.numel(), -1),
            high.expand(env_ids.numel(), -1),
        )
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset


def _resolve_named_std(spec: Union[float, Dict[str, float], Mapping[str, float], None], names: list[str], device, dtype, context: str):
    if spec is None:
        return torch.zeros(len(names), device=device, dtype=dtype)
    if isinstance(spec, (int, float)):
        value = float(spec)
        if value < 0.0:
            raise ValueError(f"{context} must be non-negative, got {value}")
        return torch.full((len(names),), value, device=device, dtype=dtype)
    if not isinstance(spec, Mapping):
        raise TypeError(f"{context} must be float/dict/None, got {type(spec).__name__}")
    idx, _, vals = string_utils.resolve_matching_names_values(dict(spec), names)
    vals = torch.as_tensor(vals, device=device, dtype=dtype)
    if (vals < 0.0).any():
        raise ValueError(f"{context} contains negative values: {vals.tolist()}")
    out = torch.zeros(len(names), device=device, dtype=dtype)
    out[torch.as_tensor(idx, device=device, dtype=torch.long)] = vals
    return out


class motion_tracking_target_joint_pos_bias(Randomization):
    def __init__(self, env, target_joint_pos_bias_noise_std: Union[float, Dict[str, float], None] = None):
        super().__init__(env)
        self.command: "MotionTrackingCommand" = cast("MotionTrackingCommand", self.env.command_manager)
        self.target_joint_pos_bias_std = _resolve_named_std(
            target_joint_pos_bias_noise_std,
            self.command.target_joint_names,
            self.device,
            self.command._target_joint_pos_bias.dtype,
            "target_joint_pos_bias_noise_std",
        )

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0 or self.command._target_joint_pos_bias.shape[1] == 0:
            return
        n = env_ids.numel()
        joint_noise = sample_uniform((n, self.command._target_joint_pos_bias.shape[1]), -1.0, 1.0, device=self.device)
        self.command._target_joint_pos_bias[env_ids] = joint_noise * self.target_joint_pos_bias_std.unsqueeze(0)


class motion_tracking_root_drift_vel(Randomization):
    def __init__(self, env, root_drift_vel_xy_max: float = 0.0, root_drift_vel_z_max: float = 0.0):
        super().__init__(env)
        self.command: "MotionTrackingCommand" = cast("MotionTrackingCommand", self.env.command_manager)
        self.root_drift_vel_xy_max = max(float(root_drift_vel_xy_max), 0.0)
        self.root_drift_vel_z_max = max(float(root_drift_vel_z_max), 0.0)

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        self.command._root_drift_vel_w[env_ids] = 0.0
        if self.root_drift_vel_xy_max > 0.0:
            self.command._root_drift_vel_w[env_ids, :2] = rand_points_disk(
                n,
                1,
                r_max=self.root_drift_vel_xy_max,
                device=self.device,
                dtype=self.command._root_drift_vel_w.dtype,
            ).squeeze(1)
        if self.root_drift_vel_z_max > 0.0:
            self.command._root_drift_vel_w[env_ids, 2] = sample_uniform(
                (n,),
                -self.root_drift_vel_z_max,
                self.root_drift_vel_z_max,
                device=self.device,
            )


class motion_tracking_root_z_offset(Randomization):
    def __init__(self, env, z_offset_range: Tuple[float, float] = (-0.03, 0.03)):
        super().__init__(env)
        self.command: "MotionTrackingCommand" = cast("MotionTrackingCommand", self.env.command_manager)
        if len(z_offset_range) != 2:
            raise ValueError(f"z_offset_range must have exactly 2 values, got {z_offset_range}")
        self.z_offset_low = float(z_offset_range[0])
        self.z_offset_high = float(z_offset_range[1])
        if self.z_offset_high < self.z_offset_low:
            raise ValueError(
                f"z_offset_range must satisfy low <= high, got {z_offset_range}"
            )

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.command._root_z_offset[env_ids] = sample_uniform(
            (env_ids.numel(),),
            self.z_offset_low,
            self.z_offset_high,
            device=self.device,
        )


class perturb_root_vel(Randomization):
    def __init__(
        self,
        env,
        min_s: float,
        max_s: float,
        x: Tuple[float, float] = (0.0, 0.0),
        y: Tuple[float, float] = (0.0, 0.0),
        z: Tuple[float, float] = (0.0, 0.0),
        roll: Tuple[float, float] = (0.0, 0.0),
        pitch: Tuple[float, float] = (0.0, 0.0),
        yaw: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        self.min_s = float(min_s)
        self.max_s = float(max_s)
        if self.min_s < 0.0:
            raise ValueError(f"min_s must be >= 0, got {self.min_s}")
        if self.max_s < self.min_s:
            raise ValueError(f"max_s must be >= min_s, got max_s={self.max_s}, min_s={self.min_s}")

        self._range_names = ("x", "y", "z", "roll", "pitch", "yaw")
        ranges = (x, y, z, roll, pitch, yaw)
        lows = []
        highs = []
        for name, axis_range in zip(self._range_names, ranges, strict=True):
            if len(axis_range) != 2:
                raise ValueError(f"{name} range must have exactly 2 values, got {axis_range}")
            low = float(axis_range[0])
            high = float(axis_range[1])
            if high < low:
                raise ValueError(f"{name} range must satisfy low <= high, got {axis_range}")
            lows.append(low)
            highs.append(high)
        self.low = torch.tensor(lows, dtype=torch.float32, device=self.device)
        self.high = torch.tensor(highs, dtype=torch.float32, device=self.device)

        self.time_left_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _sample_interval(self, n: int):
        if n <= 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        return sample_uniform((n,), self.min_s, self.max_s, device=self.device)

    def _sample_delta_vel(self, n: int):
        if n <= 0:
            return torch.empty((0, 6), dtype=torch.float32, device=self.device)
        return uniform(
            self.low.unsqueeze(0).expand(n, -1),
            self.high.unsqueeze(0).expand(n, -1),
        )

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.time_left_s[env_ids] = self._sample_interval(len(env_ids))

    def update(self):
        self.time_left_s.sub_(self.env.step_dt)
        trigger_ids = torch.nonzero(self.time_left_s <= 1e-6, as_tuple=False).squeeze(-1)
        if trigger_ids.numel() == 0:
            return

        delta_vel = self._sample_delta_vel(trigger_ids.numel())
        lin_vel = self.asset.data.root_link_lin_vel_w[trigger_ids] + delta_vel[:, :3]
        ang_vel = self.asset.data.root_link_ang_vel_w[trigger_ids] + delta_vel[:, 3:]
        root_vel = torch.cat((lin_vel, ang_vel), dim=-1)

        self.asset.write_root_link_velocity_to_sim(root_vel, env_ids=trigger_ids)

        self.time_left_s[trigger_ids] = self._sample_interval(trigger_ids.numel())

class perturb_gravity(Randomization):
    def __init__(self, env, mean: Tuple[float, float, float] = (0.0, 0.0, -9.81), std: float = 0.0):
        super().__init__(env)
        if len(mean) != 3:
            raise ValueError(f"mean must have 3 elements (x, y, z), got {mean}")
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.std = float(std)
        if self.std < 0.0:
            raise ValueError(f"std must be >= 0, got {self.std}")
        self.asset: Articulation = self.env.scene["robot"]

    def _ensure_per_env_gravity_storage(self):
        gravity = self.env.sim.model.opt.gravity
        has_per_env_storage = gravity.shape[0] == self.num_envs and gravity.stride(0) != 0
        if has_per_env_storage:
            return

        init_gravity = self.mean.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        with wp.ScopedDevice(self.env.sim.wp_device):
            self.env.sim.wp_model.opt.gravity = wp.from_torch(init_gravity, dtype=wp.vec3)
        self.env.sim.model.clear_cache()
        self.env.sim.create_graph()

    def _sample_gravity(self, n_envs: int):
        gravity = self.mean.unsqueeze(0).expand(n_envs, -1).clone()
        if self.std > 0.0:
            gravity = add_spherical_noise(gravity, self.std)
        return gravity

    def startup(self):
        self._ensure_per_env_gravity_storage()
        gravity = self._sample_gravity(self.num_envs)
        self.env.sim.model.opt.gravity[:] = gravity

        assert torch.allclose(self.env.sim.model.opt.gravity, gravity)

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        if not self.env._has_gui():
            return

        gravity = self.env.sim.model.opt.gravity
        if gravity.shape[0] == 1:
            gravity = gravity.expand(self.num_envs, -1)
        gravity_dir = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        starts = self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.35], device=self.device)
        vectors = gravity_dir * 0.25
        self.env.debug_draw.vector(starts, vectors, color=(0.2, 0.9, 0.2, 1.0))


def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def sample_log_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    low_t = torch.tensor(low, device=device, dtype=torch.float32)
    high_t = torch.tensor(high, device=device, dtype=torch.float32)
    return log_uniform(low_t, high_t).expand(size) if size == () else log_uniform(
        low_t.expand(size), high_t.expand(size)
    )

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight
