import torch
import einops
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING
from tensordict import TensorDictBase
import mjlab.utils.lab_api.string as string_utils
import active_adaptation.utils.symmetry as symmetry_utils
import active_adaptation.utils.joint_order as joint_order_utils

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from active_adaptation.envs.base import _Env


class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
    
    def symmetry_transforms(self):
        raise NotImplementedError(
            "ActionManager subclasses must implement symmetry_transforms method."
            "This method should return a SymmetryTransform object that applies to the action space."
        )


class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] | float = 0.5,
        max_delay: int | None = None,
        delay_full_progress: float = 1.0,
        boot_delay_steps: int = 0,
        alpha: Tuple[float, float] = (0.9, 0.9),
        boot_protect: bool = False,
        **kwargs,
    ):
        super().__init__(env)

        # ------------------------------------------------------------------ cfg
        self.joint_ids, self.joint_names, self.action_scaling = (
            joint_order_utils.resolve_joint_order_with_values(
                self.asset, dict(action_scaling)
            )
        )
        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.action_dim = len(self.joint_ids)

        self.max_delay = max_delay or 0  # physics steps
        self.delay_full_progress = float(delay_full_progress)
        if not (0.0 < self.delay_full_progress <= 1.0):
            raise ValueError(
                f"delay_full_progress must be in (0, 1], got {self.delay_full_progress}."
            )
        self.boot_delay_steps = int(max(boot_delay_steps, 0))

        self.alpha_range = alpha

        # Boot‑protection ----------------------------------------------------
        self.boot_protect_enabled = boot_protect
        if self.boot_protect_enabled:
            self.boot_delay = torch.zeros(self.num_envs, 1, dtype=int, device=self.device)

        # Persistent tensors -------------------------------------------------
        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)

        with torch.device(self.device):
            # Max delay index used below is ceil(max_delay / decimation), so
            # history length must be at least that index + 1.
            max_hist_idx = (self.max_delay + self.env.decimation - 1) // self.env.decimation
            hist = max(max_hist_idx + 1, 8)
            self.action_buf = torch.zeros(self.num_envs, hist, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)
            self.delay_probs = torch.zeros(int(self.max_delay) + 1, dtype=torch.float32)

        # Initialize scheduled delay cap at progress=0.
        self.step_schedule(0.0, None)

    # --------------------------------------------------------------------- util
    def resolve(self, spec, names=None):
        """Convenience helper for user APIs."""
        target_names = (
            joint_order_utils.get_joint_name_order(self.asset)
            if names is None
            else names
        )
        return string_utils.resolve_matching_names_values(dict(spec), target_names)
    
    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

    def step_schedule(self, progress: float, iters: int | None = None):
        D = int(self.max_delay)
        if D <= 0:
            self.delay_probs.fill_(1.0)
            return

        if self.env.student_train:
            self.delay_probs.fill_(1.0 / float(D + 1))
            return

        p = float(min(max(progress, 0.0), 1.0))
        p = min(p / self.delay_full_progress, 1.0)
        # Smooth transition:
        # [0, 1/(D+1)] -> all delay=0,
        # then linearly blend uniform(0..k) -> uniform(0..k+1) per segment.
        q = max(0.0, min(float(D), p * float(D + 1) - 1.0))
        k = int(q)
        a = q - float(k)
        kp1 = min(k + 1, D)

        self.delay_probs.zero_()

        self.delay_probs[: k + 1] += (1.0 - a) / float(k + 1)
        if kp1 > k:
            self.delay_probs[: kp1 + 1] += a / float(kp1 + 1)

        # Keep probabilities normalized against numerical drift.
        self.delay_probs /= self.delay_probs.sum().clamp_min(1e-8)

    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor):
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        # Delay selection ---------------------------------------------------
        sampled_delay = torch.multinomial(self.delay_probs, len(env_ids), replacement=True)
        self.delay[env_ids] = sampled_delay.unsqueeze(-1).to(self.delay.dtype)
        if self.boot_protect_enabled:
            self.boot_delay[env_ids] = self.boot_delay_steps

        # α per environment --------------------------------------------------
        alpha = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.alpha_range)
        self.alpha[env_ids] = alpha

    # ---------------------------------------------------------------- forward
    def __call__(self, tensordict: TensorDictBase, substep: int):
        if substep == 0:
            raw_action = tensordict["action"].clamp(-10, 10)

            ### debug symmetry
            # raw_action = self.symmetry_transforms().to(raw_action.device).forward(raw_action)

            self.action_buf = torch.roll(self.action_buf, shifts=1, dims=1)
            self.action_buf[:, 0, :] = raw_action

        # Communication delay ----------------------------------------------
        idx = (self.delay - substep + self.env.decimation - 1) // self.env.decimation
        idx = idx.clamp(0, self.action_buf.shape[1] - 1)
        delayed_action = self.action_buf.take_along_dim(idx.unsqueeze(1), dim=1).squeeze(1)
        self.applied_action.lerp_(delayed_action, self.alpha)

        # Joint targets -----------------------------------------------------
        pos_tgt = self.default_joint_pos + self.offset
        pos_tgt[:, self.joint_ids] += self.applied_action * self.action_scaling

        # Optional boot‑protection -----------------------------------------
        if self.boot_protect_enabled:
            pos_tgt = torch.where(
                self.boot_delay > 0,
                self.env.command_manager.joint_pos_boot_protect,
                pos_tgt,
            )
            self.boot_delay.sub_(1).clamp_min_(0)

        # Write to simulator -----------------------------------------------
        self.asset.set_joint_position_target(pos_tgt)
