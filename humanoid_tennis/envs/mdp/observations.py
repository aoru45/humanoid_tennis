import torch
import numpy as np
import abc
import einops
import inspect
import logging
import os
from typing import Tuple, TYPE_CHECKING, Callable

import humanoid_tennis
from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate
import humanoid_tennis.utils.symmetry as sym_utils
import humanoid_tennis.utils.joint_order as joint_order_utils
from humanoid_tennis.envs.mdp.commands.utils import add_spherical_noise, perturb_quaternion

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor, BuiltinSensor
    from humanoid_tennis.envs.base import _Env

from mjlab.utils.lab_api.string import resolve_matching_names
from humanoid_tennis.envs.mdp.contact_utils import resolve_contact_indices


class Observation:
    """
    Base class for all observations.
    """

    def __init__(self, env):
        self.env: _Env = env
        self.command_manager = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor
    
    def startup(self):
        """Called once upon initialization of the environment"""
        pass
    
    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass

    def symmetry_transforms(self):
        breakpoint()
        raise NotImplementedError(
            "This observation does not support symmetry transforms. "
            "Please implement the symmetry_transforms method if needed."
        )


def observation_func(func):

    class ObsFunc(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params

        def compute(self):
            return func(self.env, **self.params)
    
    return ObsFunc

def observation_wrapper(func: Callable[[], torch.Tensor], func_sym: Callable):
    def _select_kwargs(fn: Callable, params: dict):
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return dict(params), True
        valid_keys = {
            name for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return {k: v for k, v in params.items() if k in valid_keys}, False

    class ObservationWrapper(Observation):
        def __init__(self, env, **params):
            super().__init__(env)
            self.params = params
            self._func_kwargs, func_accepts_all = _select_kwargs(func, params)
            self._func_sym_kwargs, _ = _select_kwargs(func_sym, params)
            if not func_accepts_all:
                unknown = set(params.keys()) - set(self._func_kwargs.keys())
                if len(unknown) > 0:
                    raise ValueError(
                        f"Unknown YAML params for wrapped observation '{func.__name__}': {sorted(unknown)}"
                    )

        def compute(self):
            return func(**self._func_kwargs)

        def symmetry_transforms(self):
            return func_sym(**self._func_sym_kwargs)

    return ObservationWrapper

class root_angvel_b_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.imu_ang_vel_sensor: BuiltinSensor = self.env.scene["robot/imu_ang_vel"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        self.buffer[env_ids] = 0

    def update(self):
        root_ang_vel_b = self.imu_ang_vel_sensor.data
        if self.noise_std > 0:
            root_ang_vel_b = add_spherical_noise(root_ang_vel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_ang_vel_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)
    
    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[-1., 1., -1.])
        return transform.repeat(len(self.history_steps))

class root_linacc_b_history(Observation):
    def __init__(self, env, noise_std: float=0., bias_noise_std: float=0., history_steps: list[int]=[0]):
        super().__init__(env)
        self.imu_lin_acc_sensor: BuiltinSensor = self.env.scene["robot/imu_lin_acc"]
        self.noise_std = max(noise_std, 0.)
        self.bias_noise_std = max(bias_noise_std, 0.)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device, dtype=self.imu_lin_acc_sensor.data.dtype)
        self.bias = torch.zeros((self.num_envs, 3), device=self.device, dtype=self.imu_lin_acc_sensor.data.dtype)
        self.update()

    def reset(self, env_ids):
        if self.bias_noise_std > 0:
            bias = torch.zeros((len(env_ids), 3), device=self.device, dtype=self.bias.dtype)
            self.bias[env_ids] = add_spherical_noise(bias, self.bias_noise_std)
        else:
            self.bias[env_ids] = 0
        self.buffer[env_ids] = 0

    def update(self):
        lin_acc_b = self.imu_lin_acc_sensor.data + self.bias
        if self.noise_std > 0:
            lin_acc_b = add_spherical_noise(lin_acc_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = lin_acc_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1., -1., 1.])
        return transform.repeat(len(self.history_steps))

class projected_gravity_history(Observation):
    def __init__(
        self,
        env,
        noise_std: float=0.,
        history_steps: list[int]=[1],
        bias_noise_std: float=0.,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.bias_noise_std = max(bias_noise_std, 0.)
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.bias_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.bias_quat[:, 0] = 1.0
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)
        self.update()
    
    def reset(self, env_ids):
        base_quat = torch.zeros((len(env_ids), 4), device=self.device, dtype=self.asset.data.root_link_quat_w.dtype)
        base_quat[:, 0] = 1.0
        if self.bias_noise_std > 0:
            base_quat = perturb_quaternion(base_quat, self.bias_noise_std)
        self.bias_quat[env_ids] = base_quat

        self.buffer[env_ids] = 0
    
    def update(self):
        root_quat = quat_mul(self.bias_quat, self.asset.data.root_link_quat_w)
        if self.noise_std > 0:
            root_quat = perturb_quaternion(root_quat, self.noise_std)
        projected_gravity_b = quat_apply_inverse(root_quat, self._gravity_vec_w.expand(self.num_envs, -1))
        projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = projected_gravity_b
    
    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(len(self.history_steps))

class root_linvel_b_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[0]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()

    def reset(self, env_ids):
        self.buffer[env_ids] = 0

    def update(self):
        root_linvel_b = self.asset.data.root_link_lin_vel_b
        if self.noise_std > 0:
            root_linvel_b = random_noise(root_linvel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_linvel_b
    
    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])
        return transform.repeat(len(self.history_steps))

    def debug_draw(self):
        if self.env._has_gui():
            linvel = self.asset.data.root_link_lin_vel_w
            self.env.debug_draw.vector(
                self.asset.data.root_link_pos_w + torch.tensor([0.0, 0.0, 0.2], device=self.device),
                linvel,
                color=(0.8, 0.1, 0.1, 1.)
            )
    
class JointObs(Observation):
    def __init__(
        self, 
        env,
        joint_names: str=".*",
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        joint_ids, self.joint_names = joint_order_utils.resolve_joint_order(
            self.asset, joint_names
        )
        self.joint_ids = torch.tensor(joint_ids, device=self.device)
        self.num_joints = len(joint_ids)

class joint_params(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
    ):
        super().__init__(env, joint_names)
        raise NotImplementedError("Not implemented for MJLab backend.")
        self.dof_ids = self.asset.indexing.joint_v_adr[self.joint_ids]

    def compute(self) -> torch.Tensor:
        model = self.env.sim.model
        arm = model.dof_armature[:, self.dof_ids]
        fric = model.dof_frictionloss[:, self.dof_ids]
        breakpoint()
        if hasattr(model, "jnt_stiffness"):
            stiff = model.jnt_stiffness[:, self.joint_ids]
        else:
            stiff = torch.zeros_like(arm)
        damp = model.dof_damping[:, self.dof_ids]
        return torch.cat([
            arm,
            fric,
            stiff,
            damp
        ], dim=-1)
    
    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names).repeat(4)
        return transform

class joint_pos_history(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1], 
        noise_std: float=0.,
    ):
        super().__init__(env, joint_names)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.)
        from humanoid_tennis.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager
        self.joint_pos_offset = action_manager.offset

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def reset(self, env_ids):
        self.buffer[env_ids] = 0
    
    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_pos = self.joint_pos.mean(1)
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer[:, 0] = joint_pos
    
    def compute(self):
        joint_pos = self.buffer - self.joint_pos_offset[:, self.joint_ids].unsqueeze(1)
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(len(self.history_steps))

class joint_vel_history(JointObs):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1],
        noise_std: float=0.,
    ):
        super().__init__(env, joint_names)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.)

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.joint_vel = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = torch.zeros(shape, device=self.device)

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def reset(self, env_ids):
        self.buffer[env_ids] = 0

    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_vel = self.joint_vel.mean(1)
        if self.noise_std > 0:
            joint_vel = random_noise(joint_vel, self.noise_std)
        self.buffer[:, 0] = joint_vel

    def compute(self):
        joint_vel_selected = self.buffer[:, self.history_steps]
        return joint_vel_selected.reshape(self.num_envs, -1)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform.repeat(len(self.history_steps))

class applied_torque(JointObs):
    def __init__(self, env, joint_names: str=".*", noise_std: float=0.):
        super().__init__(env, joint_names)
        self.noise_std = max(noise_std, 0.)

        actuator_names = list(self.asset.actuator_names)
        name_to_act = {n: i for i, n in enumerate(actuator_names)}
        act_idx = []
        for name in self.joint_names:
            if name not in name_to_act:
                raise RuntimeError(f"Actuator for joint '{name}' not found.")
            else:
                act_idx.append(name_to_act[name])
        self.act_idx = torch.tensor(act_idx, device=self.device, dtype=torch.long)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.actuator_force
        return random_noise(applied_efforts[:, self.act_idx], self.noise_std)

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

class feet_contact_state(Observation):
    def __init__(self, env, body_names, divide_by_mass: bool=True):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        robot_cfg = getattr(self.env.cfg, "robot", None)
        mass_total = getattr(robot_cfg, "mass", None) if robot_cfg is not None else None
        if mass_total is None:
            breakpoint()
        self.default_mass_total = mass_total * 9.81
        self.denom = self.default_mass_total if divide_by_mass else 1.
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names
        )

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.force_history[:, self.body_ids, :, :].mean(2)
        force = (contact_forces / self.denom).clamp(-10.0, 10.0)
        contact_time = self.contact_sensor.data.current_contact_time[:, self.body_ids]
        air_time = self.contact_sensor.data.current_air_time[:, self.body_ids]
        in_contact = (contact_time > self.env.physics_dt).float()
        return torch.cat(
            [
                force.view(self.num_envs, -1),
                in_contact,
                contact_time,
                air_time,
            ],
            dim=-1,
        )
    
    def symmetry_transforms(self):
        force_transform = sym_utils.cartesian_space_symmetry(
            self.asset, self.body_names, sign=[1, -1, 1]
        )
        scalar_transform = sym_utils.cartesian_space_symmetry(
            self.asset, self.body_names, sign=(1,)
        )
        return sym_utils.SymmetryTransform.cat(
            [force_transform, scalar_transform, scalar_transform, scalar_transform]
        )

class body_height(Observation):
    def __init__(self, env, body_names=".*_foot"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self._nonfinite_debug_emitted = False
        self._nonfinite_dump_dir = os.environ.get("HT_NONFINITE_DUMP_DIR", "").strip()

    def _safe_tensor_sample(self, tensor: torch.Tensor, env_id: int, max_items: int = 16):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim == 0:
            return float(tensor.detach().cpu().item())
        if tensor.shape[0] <= env_id:
            return None
        value = tensor[env_id].detach().float().cpu().reshape(-1)
        if value.numel() > max_items:
            value = value[:max_items]
        return value.tolist()

    def _emit_nonfinite_debug(self, heights: torch.Tensor, bad_env: torch.Tensor):
        if self._nonfinite_debug_emitted:
            return
        self._nonfinite_debug_emitted = True

        sample_env_ids = bad_env[:4].detach().cpu().tolist()
        lines = []
        lines.append(
            "[DEBUG][body_height] non-finite detected: "
            f"num_bad_envs={int(bad_env.numel())}, sample_env_ids={sample_env_ids}, "
            f"timestamp={int(getattr(self.env, 'timestamp', -1))}"
        )
        if hasattr(self.env, "episode_length_buf"):
            episode_len = self.env.episode_length_buf[bad_env[:16]].detach().cpu().tolist()
            lines.append(f"[DEBUG][body_height] episode_length(sample)={episode_len}")

        body_names = [str(n) for n in self.body_names]
        for env_id in sample_env_ids:
            env_i = int(env_id)
            row = heights[env_i].detach()
            bad_cols = (~torch.isfinite(row)).nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            bad_body_names = [body_names[int(c)] for c in bad_cols if int(c) < len(body_names)]
            lines.append(
                f"[DEBUG][body_height][env={env_i}] bad_cols={bad_cols} bad_body_names={bad_body_names}"
            )

            # Root states.
            root_pos = self._safe_tensor_sample(getattr(self.asset.data, "root_link_pos_w", None), env_i, 3)
            root_quat = self._safe_tensor_sample(getattr(self.asset.data, "root_link_quat_w", None), env_i, 4)
            root_lin = self._safe_tensor_sample(getattr(self.asset.data, "root_link_lin_vel_w", None), env_i, 3)
            root_ang = self._safe_tensor_sample(getattr(self.asset.data, "root_link_ang_vel_w", None), env_i, 3)
            lines.append(
                f"[DEBUG][body_height][env={env_i}] "
                f"root_pos={root_pos} root_quat={root_quat} root_lin_vel={root_lin} root_ang_vel={root_ang}"
            )

            # Joint states.
            joint_pos = getattr(self.asset.data, "joint_pos", None)
            joint_vel = getattr(self.asset.data, "joint_vel", None)
            if isinstance(joint_pos, torch.Tensor) and joint_pos.shape[0] > env_i:
                jp = joint_pos[env_i]
                jv = joint_vel[env_i] if isinstance(joint_vel, torch.Tensor) and joint_vel.shape[0] > env_i else None
                lines.append(
                    f"[DEBUG][body_height][env={env_i}] "
                    f"joint_pos_finite={bool(torch.isfinite(jp).all().item())} "
                    f"joint_vel_finite={bool(torch.isfinite(jv).all().item()) if jv is not None else None} "
                    f"joint_pos_absmax={float(jp.abs().max().detach().cpu().item()):.6f} "
                    f"joint_vel_absmax="
                    f"{(float(jv.abs().max().detach().cpu().item()) if jv is not None else float('nan')):.6f}"
                )

            # Selected body z values and finite mask.
            z_row = row.float().cpu()
            z_preview = z_row[: min(12, z_row.numel())].tolist()
            z_finite_preview = torch.isfinite(z_row[: min(12, z_row.numel())]).tolist()
            lines.append(
                f"[DEBUG][body_height][env={env_i}] z_preview={z_preview} z_isfinite_preview={z_finite_preview}"
            )

            # Simulation buffers if available.
            sim_data = getattr(self.env, "sim", None)
            sim_data = getattr(sim_data, "data", None)
            if sim_data is not None:
                nacon = getattr(sim_data, "nacon", None)
                nacon_i = 0
                if isinstance(nacon, torch.Tensor):
                    flat_nacon = nacon.reshape(-1)
                    if flat_nacon.numel() > env_i:
                        nacon_i = int(flat_nacon[env_i].detach().cpu().item())
                        lines.append(f"[DEBUG][body_height][env={env_i}] nacon={nacon_i}")
                qpos = getattr(sim_data, "qpos", None)
                qvel = getattr(sim_data, "qvel", None)
                if isinstance(qpos, torch.Tensor) and qpos.ndim == 2 and qpos.shape[0] > env_i:
                    qpos_i = qpos[env_i]
                    lines.append(
                        f"[DEBUG][body_height][env={env_i}] qpos_finite={bool(torch.isfinite(qpos_i).all().item())} "
                        f"qpos_absmax={float(qpos_i.abs().max().detach().cpu().item()):.6f}"
                    )
                if isinstance(qvel, torch.Tensor) and qvel.ndim == 2 and qvel.shape[0] > env_i:
                    qvel_i = qvel[env_i]
                    lines.append(
                        f"[DEBUG][body_height][env={env_i}] qvel_finite={bool(torch.isfinite(qvel_i).all().item())} "
                        f"qvel_absmax={float(qvel_i.abs().max().detach().cpu().item()):.6f}"
                    )

                # Contact-pair diagnostics for this env.
                contact = getattr(sim_data, "contact", None)
                if contact is not None and nacon_i > 0:
                    geom = getattr(contact, "geom", None)
                    worldid = getattr(contact, "worldid", None)
                    dist = getattr(contact, "dist", None)
                    if isinstance(geom, torch.Tensor) and isinstance(worldid, torch.Tensor):
                        geom_i = geom[:nacon_i]
                        world_i = worldid[:nacon_i].to(torch.long)
                        env_mask = world_i == env_i
                        if env_mask.any():
                            pair_rows = geom_i[env_mask]
                            max_pairs = min(12, int(pair_rows.shape[0]))
                            pair_rows = pair_rows[:max_pairs]
                            pair_ids = [[int(a), int(b)] for a, b in pair_rows.detach().cpu().tolist()]
                            lines.append(
                                f"[DEBUG][body_height][env={env_i}] contact_pairs(count={int(env_mask.sum().item())}, "
                                f"show={max_pairs})={pair_ids}"
                            )

                            # Highlight whether racket/body collision geoms are involved.
                            cmd = getattr(self.env, "command_manager", None)
                            racket_ids = None
                            body_ids = None
                            if cmd is not None:
                                racket_ids = getattr(cmd, "racket_contact_geom_ids", None)
                                body_ids = getattr(cmd, "racket_body_contact_geom_ids", None)
                            if isinstance(racket_ids, torch.Tensor) and racket_ids.numel() > 0:
                                p0 = pair_rows[:, 0]
                                p1 = pair_rows[:, 1]
                                racket_hit = torch.isin(p0, racket_ids) | torch.isin(p1, racket_ids)
                                lines.append(
                                    f"[DEBUG][body_height][env={env_i}] contact_involves_racket="
                                    f"{bool(racket_hit.any().item())}"
                                )
                                if isinstance(body_ids, torch.Tensor) and body_ids.numel() > 0:
                                    b0 = torch.isin(p0, body_ids)
                                    b1 = torch.isin(p1, body_ids)
                                    rb_hit = (torch.isin(p0, racket_ids) & b1) | (torch.isin(p1, racket_ids) & b0)
                                    lines.append(
                                        f"[DEBUG][body_height][env={env_i}] contact_is_racket_body_pair="
                                        f"{bool(rb_hit.any().item())}"
                                    )

                            if isinstance(dist, torch.Tensor):
                                dist_i = dist[:nacon_i][env_mask][:max_pairs].detach().cpu().tolist()
                                lines.append(
                                    f"[DEBUG][body_height][env={env_i}] contact_dist_preview={dist_i}"
                                )

            # High-level command state hints (if available).
            cmd = getattr(self.env, "command_manager", None)
            if cmd is not None:
                for state_key in (
                    "racket_body_contact",
                    "racket_body_contact_event",
                    "fail_racket_body",
                    "has_hit",
                    "success_done",
                ):
                    v = getattr(cmd, state_key, None)
                    if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] > env_i:
                        val = bool(v[env_i].detach().cpu().item())
                        lines.append(f"[DEBUG][body_height][env={env_i}] {state_key}={val}")

        logging.error("\n".join(lines))

        if self._nonfinite_dump_dir:
            try:
                os.makedirs(self._nonfinite_dump_dir, exist_ok=True)
                dump = {
                    "bad_env": bad_env.detach().cpu(),
                    "sample_env_ids": sample_env_ids,
                    "timestamp": int(getattr(self.env, "timestamp", -1)),
                    "heights": heights[bad_env[:16]].detach().cpu(),
                }
                env_sel = bad_env[:16]
                for k in (
                    "root_link_pos_w",
                    "root_link_quat_w",
                    "root_link_lin_vel_w",
                    "root_link_ang_vel_w",
                    "joint_pos",
                    "joint_vel",
                ):
                    v = getattr(self.asset.data, k, None)
                    if isinstance(v, torch.Tensor) and v.shape[0] > 0:
                        dump[k] = v[env_sel].detach().cpu()
                dump_path = os.path.join(
                    self._nonfinite_dump_dir,
                    f"body_height_nonfinite_rank{humanoid_tennis.get_local_rank()}_t{int(getattr(self.env, 'timestamp', -1))}.pt",
                )
                torch.save(dump, dump_path)
                logging.error(f"[DEBUG][body_height] dumped non-finite snapshot to: {dump_path}")
            except Exception as exc:
                logging.error(f"[DEBUG][body_height] failed to dump snapshot: {exc}")
    
    def compute(self) -> torch.Tensor:
        heights = self.asset.data.body_link_pos_w[:, self.body_ids, 2].reshape(
            self.num_envs, -1
        )
        skip_mask = getattr(self.env, "_obs_skip_mask", None)
        if isinstance(skip_mask, torch.Tensor) and skip_mask.shape[0] == self.num_envs:
            skip_mask = skip_mask.to(device=heights.device, dtype=torch.bool)
        else:
            skip_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=heights.device)

        nonfinite = ~torch.isfinite(heights)
        active_nonfinite = nonfinite & (~skip_mask).unsqueeze(-1)
        if active_nonfinite.any():
            bad_env = active_nonfinite.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            self._emit_nonfinite_debug(heights, bad_env)
            raise FloatingPointError(
                f"Non-finite body_height observation: num_envs={int(bad_env.numel())}, "
                f"sample_env_ids={bad_env[:16].detach().cpu().tolist()}"
            )
        if skip_mask.any():
            heights = heights.clone()
            heights[skip_mask] = 0.0
        return heights

    def symmetry_transforms(self):
        return sym_utils.cartesian_space_symmetry(self.asset, self.body_names, sign=(1,))

class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :self.steps, :]
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms().repeat(self.steps)
        return transform

class applied_action(JointObs):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.asset.data.joint_pos_target[:, self.joint_ids]

    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

class cum_error(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.command_manager = self.env.command_manager
    
    def compute(self) -> torch.Tensor:
        return self.command_manager._cum_error

    def symmetry_transforms(self):
        transform = sym_utils.SymmetryTransform(
            perm=torch.arange(self.command_manager._cum_error.shape[-1]),
            signs=[1.] * self.command_manager._cum_error.shape[-1]
        )
        return transform
    
def symlog(x: torch.Tensor, a: float=1.):
    return x.sign() * torch.log(x.abs() * a + 1.) / a

def random_noise(x: torch.Tensor, std: float):
    if std <= 0.0:
        return x
    return x + (torch.rand_like(x) * 2.0 - 1.0) * std
