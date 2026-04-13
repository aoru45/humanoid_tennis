import torch
import numpy as np
import abc
import einops
import inspect
from typing import Tuple, TYPE_CHECKING, Callable

import active_adaptation
from active_adaptation.utils.math import quat_apply, quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate
import active_adaptation.utils.symmetry as sym_utils
import active_adaptation.utils.joint_order as joint_order_utils
from active_adaptation.envs.mdp.commands.utils import add_spherical_noise, perturb_quaternion

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor, BuiltinSensor
    from active_adaptation.envs.base import _Env

from mjlab.utils.lab_api.string import resolve_matching_names
from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices


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
        from active_adaptation.envs.mdp.action import JointPosition
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
    
    def compute(self) -> torch.Tensor:
        return self.asset.data.body_link_pos_w[:, self.body_ids, 2].reshape(
            self.num_envs, -1
        )

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
