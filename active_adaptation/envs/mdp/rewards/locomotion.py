from math import inf
import inspect
import torch
import abc
from typing import TYPE_CHECKING, Callable, List, Tuple

import mjlab.utils.lab_api.string as string_utils
from mjlab.utils.lab_api.string import resolve_matching_names
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat, normalize
from ..commands import *
from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices

if TYPE_CHECKING:
    from mjlab.sensor import ContactSensor
    from mjlab.entity import Entity as Articulation
    from active_adaptation.envs.base import _Env


class Reward:
    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
    ):
        self.env: _Env = env
        self.weight = weight
        self.enabled = enabled

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count 

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass


def reward_func(func):
    class RewFunc(Reward):
        def compute(self):
            return func(self.env)

    return RewFunc


def reward_wrapper(func: Callable[[], torch.Tensor]):
    def _select_kwargs(fn: Callable, params: dict):
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(params), True
        valid_keys = {
            name for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {k: v for k, v in params.items() if k in valid_keys}, False

    class RewardWrapper(Reward):
        def __init__(self, env, weight: float, enabled: bool = True, **params):
            super().__init__(env, weight=weight, enabled=enabled)
            self.params = params
            self._func_kwargs, func_accepts_all = _select_kwargs(func, params)
            if not func_accepts_all:
                unknown = set(params.keys()) - set(self._func_kwargs.keys())
                if len(unknown) > 0:
                    raise ValueError(
                        f"Unknown YAML params for wrapped reward '{func.__name__}': {sorted(unknown)}"
                    )

        def compute(self):
            return func(**self._func_kwargs)
    return RewardWrapper


class _ContactMajorityCache:
    """Shared per-env-step contact state using substep majority voting."""

    def __init__(self, env, contact_sensor):
        self.env = env
        self.contact_sensor = contact_sensor
        found = self.contact_sensor.data.found
        if found is None:
            raise RuntimeError(
                "Contact sensor must include 'found' field to use majority contact cache."
            )
        self.decimation = max(int(self.env.decimation), 1)
        self.num_bodies = int(found.shape[1])
        self.substep_found = torch.zeros(
            (self.env.num_envs, self.num_bodies, self.decimation),
            dtype=torch.bool,
            device=self.env.device,
        )
        self.current_contact = torch.zeros(
            (self.env.num_envs, self.num_bodies), dtype=torch.bool, device=self.env.device
        )
        self.first_contact = torch.zeros_like(self.current_contact)
        self.first_air = torch.zeros_like(self.current_contact)
        self._last_post_stamp = (-1, -1)
        self._last_update_stamp = -1

    def reset(self, env_ids: torch.Tensor):
        self.substep_found[env_ids] = False
        self.current_contact[env_ids] = False
        self.first_contact[env_ids] = False
        self.first_air[env_ids] = False

    def post_step(self, substep: int):
        stamp = (int(self.env.timestamp), int(substep))
        if stamp == self._last_post_stamp:
            return
        found = self.contact_sensor.data.found
        if found is None:
            raise RuntimeError(
                "Contact sensor must include 'found' field to use majority contact cache."
            )
        self.substep_found[:, :, substep] = found > 0
        self._last_post_stamp = stamp

    def update(self):
        stamp = int(self.env.timestamp)
        if stamp == self._last_update_stamp:
            return
        votes = self.substep_found.sum(dim=-1)
        contact_majority = votes >= (self.decimation // 2)
        prev_contact = self.current_contact
        self.first_contact[:] = (~prev_contact) & contact_majority
        self.first_air[:] = prev_contact & (~contact_majority)
        self.current_contact[:] = contact_majority
        self.substep_found.zero_()
        self._last_update_stamp = stamp

    def current_for(self, body_ids: torch.Tensor):
        return self.current_contact[:, body_ids]

    def first_contact_for(self, body_ids: torch.Tensor):
        return self.first_contact[:, body_ids]

    def first_air_for(self, body_ids: torch.Tensor):
        return self.first_air[:, body_ids]


def _get_contact_majority_cache(env, contact_sensor):
    cache = getattr(env, "_contact_majority_cache", None)
    if cache is None:
        cache = _ContactMajorityCache(env, contact_sensor)
        env._contact_majority_cache = cache
    elif cache.contact_sensor is not contact_sensor:
        raise RuntimeError("Multiple contact sensors are not supported by shared contact cache.")
    return cache


@reward_func
def survival(self):
    return torch.ones(self.num_envs, 1, device=self.device)


class joint_torques_l2(Reward):
    def __init__(
        self, env, weight: float, enabled: bool = True, joint_names: str = ".*"
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)

    def compute(self) -> torch.Tensor:
        return (
            -self.asset.data.actuator_force[:, self.act_idx]
            .square()
            .sum(1, keepdim=True)
        )


class impact_force_l2(Reward):
    def __init__(
        self,
        env,
        body_names,
        body2_names=None,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.articulation_body_ids = self.asset.find_bodies(self.body_names)[0]
        if body2_names is None:
            self.articulation_body2_ids = self.articulation_body_ids
        else:
            self.articulation_body2_ids = self.asset.find_bodies(body2_names)[0]
            if len(self.articulation_body2_ids) != len(self.articulation_body_ids):
                raise ValueError(
                    "body2_names must match body_names length for impact_force_l2."
                )
        self.last_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device, dtype=bool
        )
        self.prev_down_vel = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.down_vel = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )

        print(f"Penalizing impact forces on {self.body_names}.")

    def reset(self, env_ids: torch.Tensor):
        self.last_contact[env_ids] = False
        self.prev_down_vel[env_ids] = 0.0
        self.down_vel[env_ids] = 0.0

    def update(self):
        vel_z = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, 2]
        vel_z2 = self.asset.data.body_com_lin_vel_w[:, self.articulation_body2_ids, 2]
        down_vel = torch.maximum((-vel_z).clamp_min(0.0), (-vel_z2).clamp_min(0.0))
        self.prev_down_vel.copy_(self.down_vel)
        self.down_vel[:] = torch.where(
            self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.0,
            self.down_vel,
            down_vel,
        )

    def compute(self) -> torch.Tensor:
        current_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.physics_dt
        first_contact = (~self.last_contact) & current_contact
        self.last_contact[:] = current_contact
        impact = self.prev_down_vel.square() * first_contact
        return -impact.clamp_max(10.0).sum(1, True)

class feet_slip(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_ids] > self.env.physics_dt
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        slip = (in_contact * feet_vel.norm(dim=-1).square()).sum(dim=1, keepdim=True)
        return -slip

class feet_upright(Reward):
    def __init__(
        self, env, body_names: str, xy_sigma: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.body_ids_asset, _ = self.asset.find_bodies(body_names)

        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device)
        self.down = down.expand(self.num_envs, len(self.body_ids_asset), -1)
        self.xy_sigma = xy_sigma
        
    def compute(self):
        feet_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids_asset]
        feet_projected_down = quat_apply(feet_quat_w, self.down)
        feet_projected_down_xy = feet_projected_down[:, :, :2].norm(dim=-1)
        rew = (torch.exp(-feet_projected_down_xy / self.xy_sigma) - 1.0)
        return rew.float().mean(dim=1, keepdim=True)

class feet_air_time_ref(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

        self.reward_time = torch.zeros(self.num_envs, len(self.body_ids), device=self.env.device)

    def reset(self, env_ids):
        self.reward_time[env_ids] = 0.0
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        first_contact = self.contact_cache.first_contact_for(self.body_ids)

        if hasattr(self.env.command_manager, "skip_ref") and self.env.command_manager.skip_ref:
            self.reward_time = self.reward_time + self.env.step_dt
        else:
            contact_diff = self.env.command_manager.feet_standing ^ current_contact
            self.reward_time = self.reward_time + torch.where(
                contact_diff, -self.env.step_dt, self.env.step_dt
            )
        
        self.reward = torch.sum(
            (self.reward_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        
        self.reward_time = self.reward_time * (~current_contact)
        return self.reward

class feet_air_time_ref_dense(Reward):
    def __init__(
        self,
        env: "LocomotionEnv",
        body_names: str,
        body2_names: str | None = None,
        air_h_low: float = 0.035,
        air_h_high: float = 0.155,
        contact_h_low: float = 0.035,
        contact_h_high: float = 0.125,
        weight: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        if body2_names is None:
            self.articulation_body2_ids = self.articulation_body_ids
        else:
            self.articulation_body2_ids = self.asset.find_bodies(body2_names)[0]
            if len(self.articulation_body2_ids) != len(self.articulation_body_ids):
                raise ValueError(
                    "body2_names must match body_names length for feet_air_time_ref_dense."
                )

        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

        self.air_h_low = float(air_h_low)
        self.air_h_high = float(air_h_high)
        self.air_h_span = max(self.air_h_high - self.air_h_low, 1e-6)
        self.contact_h_low = float(contact_h_low)
        self.contact_h_high = float(contact_h_high)
        self.contact_h_span = max(self.contact_h_high - self.contact_h_low, 1e-6)

    def reset(self, env_ids):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        target_contact = self.env.command_manager.feet_standing

        # mismatch -> -1; both contact/both air -> height-based penalty in [-1, 0]
        mismatch = current_contact ^ target_contact
        both_air = (~current_contact) & (~target_contact)
        both_contact = current_contact & target_contact

        penalty = torch.zeros_like(current_contact, dtype=torch.float32)
        penalty[mismatch] = -1.0

        feet_height_air = torch.minimum(
            self.asset.data.body_link_pos_w[:, self.articulation_body_ids, 2],
            self.asset.data.body_link_pos_w[:, self.articulation_body2_ids, 2],
        )
        air_ratio = ((feet_height_air - self.air_h_low) / self.air_h_span).clamp(0.0, 1.0)
        air_penalty = -(1.0 - air_ratio)
        penalty = torch.where(both_air, air_penalty, penalty)

        feet_height_contact = torch.maximum(
            self.asset.data.body_link_pos_w[:, self.articulation_body_ids, 2],
            self.asset.data.body_link_pos_w[:, self.articulation_body2_ids, 2],
        )
        t_contact = (
            (feet_height_contact - self.contact_h_low) / self.contact_h_span
        ).clamp(0.0, 1.0)
        # low -> 0, high -> -1
        contact_penalty = -t_contact
        penalty = torch.where(both_contact, contact_penalty, penalty)

        return penalty.mean(dim=1, keepdim=True)

class feet_contact_count(Reward):
    def __init__(
        self, env: "LocomotionEnv", body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.contact_cache = _get_contact_majority_cache(self.env, self.contact_sensor)

    def reset(self, env_ids: torch.Tensor):
        self.contact_cache.reset(env_ids)

    def post_step(self, substep: int):
        self.contact_cache.post_step(substep)

    def update(self):
        self.contact_cache.update()

    def compute(self):
        first_contact = self.contact_cache.first_contact_for(self.body_ids)
        return first_contact.sum(1, keepdim=True)

    def debug_draw(self):
        current_contact = self.contact_cache.current_for(self.body_ids)
        if not current_contact.any():
            return
        feet_pos = self.asset.data.body_link_pos_w[:, self.articulation_body_ids].clone()
        feet_pos[..., 2] -= 0.1
        points = feet_pos[current_contact]
        if points.numel() > 0:
            self.env.debug_draw.point(points, color=(1.0, 0.0, 0.0, 1.0), size=20.0)


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().sum(1, True)

class joint_acc_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        # print(self.asset.data.joint_acc[:, self.joint_ids].max())
        r = - self.asset.data.joint_acc[:, self.joint_ids].clamp_max(100.0).square().sum(1, True)
        return r

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_factor = soft_factor
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True) / (1-self.soft_factor)

class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)

        # MJLab: derive limits from actuator forcerange (if available)
        model = self.env.sim.model
        if not hasattr(model, "actuator_forcerange"):
            raise RuntimeError("Actuator force limits are not available in MJLab model.")
        ctrl_ids = self.asset.indexing.ctrl_ids
        force_range = model.actuator_forcerange[:, ctrl_ids]
        limits = torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs())

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            else:
                act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)
        self.soft_limits = limits[:, self.act_idx] * soft_factor
    
    def compute(self) -> torch.Tensor:
        applied_torque = self.asset.data.actuator_force[:, self.act_idx]
        violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        return - (violation_high + violation_low).sum(dim=1, keepdim=True)

@reward_func
def action_rate_l2(self):
    action_buf = self.action_manager.action_buf
    action_diff = action_buf[:, 0, :] - action_buf[:, 1, :]
    return - action_diff.square().sum(dim=-1, keepdim=True)


@reward_func
def action_rate2_l2(self):
    action_buf = self.action_manager.action_buf
    action_diff = (
        action_buf[:, 0, :] - 2 * action_buf[:, 1, :] + action_buf[:, 2, :]
    )
    return - action_diff.square().sum(dim=-1, keepdim=True)
