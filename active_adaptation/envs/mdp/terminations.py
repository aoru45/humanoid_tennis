import torch
import abc

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor

from active_adaptation.envs.mdp.contact_utils import resolve_contact_indices


class Termination:
    def __init__(self, env):
        self.env = env
    
    def update(self):
        pass

    def reset(self, env_ids):
        pass
    
    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs


def termination_func(func):
    class TermFunc(Termination):
        def __call__(self):
            return func(self.env)
    return TermFunc


def termination_wrapper(func):
    class TerminationWrapper(Termination):
        def __call__(self):
            return func()
    return TerminationWrapper


class crash(Termination):
    def __init__(
        self, 
        env, 
        body_names_expr: str,
        t_thres: float = 0.,
        min_time: float = 0.,
        **kwargs
    ):
        super().__init__(env)
        raise NotImplementedError("Not implemented for MJLab backend.")
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = resolve_contact_indices(
            self.contact_sensor, self.asset, body_names_expr
        )
        self.t_thres = t_thres
        self._decay = 0.98
        self._thres = (self.t_thres / self.env.physics_dt) * 0.9
        self.count = torch.zeros(self.num_envs, len(self.body_indices), device=self.env.device)
        self.min_steps = int(min_time / self.env.step_dt)
        print(f"Terminate upon contact on {self.body_names}")
    
    def reset(self, env_ids):
        self.count[env_ids] = 0.
    
    def update(self):
        data = self.contact_sensor.data
        if getattr(data, "force", None) is None:
            raise RuntimeError("Contact sensor does not provide force data.")
        in_contact = data.force[:, self.body_indices].norm(dim=-1) > 1.0
        self.count.add_(in_contact.float()).mul_(self._decay)
        
    def __call__(self):
        valid = (self.env.episode_length_buf > self.min_steps)
        undesired_contact = (self.count > self._thres).any(-1)
        return (undesired_contact & valid).reshape(self.num_envs, 1)
    
class fall_over(Termination):
    def __init__(
        self, 
        env, 
        xy_thres: float=0.8,
        z_thres: float=0.5
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres
        self.z_thres = z_thres
    
    def __call__(self):
        fall_over = (self.asset.data.projected_gravity_b[:, :2].norm(dim=1, keepdim=True) >= self.xy_thres) | (-self.asset.data.projected_gravity_b[:, 2:] < self.z_thres)
        return fall_over

class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85, min_steps: int = 50):
        super().__init__(env)
        self.thres = torch.tensor(thres, device=self.env.device)
        self.min_steps = min_steps # tolerate the first few steps
        self.error_exceeded_count = torch.zeros(self.env.num_envs, 1, device=self.env.device, dtype=torch.int32)
        self.command_manager = self.env.command_manager
    
    def reset(self, env_ids):
        self.error_exceeded_count[env_ids] = 0

    def update(self):
        error_exceeded = (self.command_manager._cum_error > self.thres).any(-1, True)
        self.error_exceeded_count[error_exceeded] += 1
        self.error_exceeded_count[~error_exceeded] = 0
    
    def __call__(self) -> torch.Tensor:
        return (self.error_exceeded_count > self.min_steps).reshape(-1, 1)
