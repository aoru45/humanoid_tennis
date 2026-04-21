import warnings
import copy
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchrl.data import Composite, TensorSpec
from torchrl.envs.transforms import TensorDictPrimer, ExcludeTransform
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    CudaGraphModule
)

from hydra.core.config_store import ConfigStore

# ---- utils ------------------------------------------------------------------------------------ #
from ..modules.distributions import IndependentNormal
from ..utils.valuenorm import ValueNorm1, ValueNormFake
from .common import *
import active_adaptation as aa
import functools

__all__ = ["PPOPolicy", "PPOConfig"]

PULSE_PRIOR_OBS_KEY = "pulse_policy"


class Split(nn.Module):
    def __init__(self, split_size: int):
        super().__init__()
        self.split_size = split_size

    def forward(self, x):
        return x[..., :self.split_size], x[..., self.split_size:]


class GaussianSampler(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar) * self.temp
        return mu + torch.randn_like(std) * std


class IdentityAction(nn.Module):
    def forward(self, x):
        return x

class LatentActionBarrier(nn.Module):
    def __init__(self, latent_scale: float = 1.0, logvar_min: float = -5.0, logvar_max: float = 2.0):
        super().__init__()
        self.latent_scale = float(latent_scale)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(self, prior_mu: torch.Tensor, prior_logvar: torch.Tensor, delta_z: torch.Tensor):
        prior_logvar = prior_logvar.clamp(self.logvar_min, self.logvar_max)
        prior_std = torch.exp(0.5 * prior_logvar)
        return prior_mu + self.latent_scale * prior_std * torch.tanh(delta_z)
        # return prior_mu + torch.randn_like(prior_std) * prior_std


class ReplaceJointAction(nn.Module):
    def __init__(self, joint_ids: List[int], scale: float = 1.0):
        super().__init__()
        self.register_buffer("joint_ids", torch.tensor(joint_ids, dtype=torch.long))
        self.scale = float(scale)

    def forward(self, base_action: torch.Tensor, wrist_action: torch.Tensor):
        if self.joint_ids.numel() == 0:
            return base_action
        if wrist_action.shape[-1] != self.joint_ids.numel():
            raise ValueError(
                f"wrist action dim mismatch: got {wrist_action.shape[-1]}, "
                f"expected {self.joint_ids.numel()}."
            )
        out = base_action.clone()
        out[..., self.joint_ids] = wrist_action * self.scale
        return out

# ------------------------------------------------------------------------------------------------ #
# 1. Config
# ------------------------------------------------------------------------------------------------ #


@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo.PPOPolicy"
    name: str = "ppo"

    # PPO hyper‑params
    train_every: int = 32
    ppo_epochs: int = 5
    num_minibatches: int = 8

    lr: float = 5e-4
    desired_kl: float = 0.01 # kl schedule

    clip_param: float = 0.2

    entropy_coef_start: float = 0.005
    entropy_coef_end: float = 0.002

    init_noise_scale: float = 1.0  # initial std for actor
    init_noise_scale_overrides: Dict[str, float] = field(default_factory=dict)  # regex map overrides
    load_noise_scale: float | None = None  # multiplier on std loaded from checkpoint

    latent_dim: int = 256

    # distillation
    reg_lambda: float = 0.2  # weight of priv-feature alignment
    ############################## PULSE ##########################
    teacher_checkpoint_path: Union[str, None] = None
    pulse_epochs: int = 2
    pulse_latent_dim: int = 32
    pulse_kl_coef_start: float = 1.0e-2
    pulse_kl_coef_end: float = 1.0e-3
    pulse_regu_coef: float = 0.05
    pulse_kl_anneal_start: float = 0.5
    pulse_kl_anneal_end: float = 1.0
    pulse_use_temporal_reg: bool = True
    pulse_logvar_min: float = -5.0
    pulse_logvar_max: float = 2.0
    pulse_prior_temp: float = 1.0
    pulse_posterior_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 512])
    pulse_prior_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 512])
    pulse_decoder_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 512])
    # If not None, clamp rollout reward sum from below before GAE.
    # Set to null for phases/tasks that require learning from negative rewards.
    adv_reward_clamp_min: float | None = 0.0
    highlevel_lab_lambda: float = 1.0
    highlevel_wrist_residual_enabled: bool = True
    highlevel_wrist_joint_patterns: List[str] = field(
        default_factory=lambda: [
            "right_wrist_.*_joint",
        ]
    )
    highlevel_wrist_action_scale: float = 1.0
    ############################## PULSE ##########################
    # misc
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False

    # phase switch
    phase: str = "train"  # train | finetune | adapt | pulse | highlevel
    vecnorm: Union[str, None] = None
    symmetry_enabled: bool = True

    # I/O keys
    in_keys: List[str] = field(
        default_factory=lambda: [
            OBS_KEY,
            OBS_PRIV_KEY,
            CRITIC_PRIV_KEY
        ]
    )

    command_modes: Union[List[int], None] = None
    checkpoint_path: Union[str, None] = None


cs = ConfigStore.instance()
cs.store("ppo_train", node=PPOConfig(phase="train", vecnorm="train", entropy_coef_start=0.01, entropy_coef_end=0.0025), group="algo")
cs.store("ppo_adapt", node=PPOConfig(phase="adapt", vecnorm="eval", train_every=16), group="algo")
cs.store("ppo_finetune", node=PPOConfig(phase="finetune", vecnorm="eval", lr=1e-4, entropy_coef_start=0.0025, entropy_coef_end=0.0005), group="algo")
cs.store("ppo_pulse", node=PPOConfig(phase="pulse", vecnorm="eval", train_every=16), group="algo")
cs.store("ppo_highlevel", node=PPOConfig(phase="highlevel", vecnorm="eval", train_every=16, lr=2e-4, entropy_coef_start=0.005, entropy_coef_end=0.001), group="algo")

class PPOPolicy(TensorDictModuleBase):
    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device: str = "cuda:0",
        env = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.observation_spec = observation_spec
        assert cfg.phase in {"train", "finetune", "adapt", "pulse", "highlevel"}

        self.entropy_coef = cfg.entropy_coef_start
        self.clip_param = cfg.clip_param
        self.action_dim = action_spec.shape[-1]
        self.action_manager = env.action_manager
        self.joint_names = env.action_manager.joint_names
        self.highlevel_action_key = "highlevel_action"
        (
            self.highlevel_wrist_joint_ids,
            self.highlevel_wrist_joint_names,
        ) = self._resolve_highlevel_wrist_joints()
        self.highlevel_wrist_dim = len(self.highlevel_wrist_joint_ids)
        self.highlevel_action_dim = int(self.cfg.pulse_latent_dim) + int(self.highlevel_wrist_dim)
        init_noise_scale = self._resolve_init_noise_scale()
        self._init_noise_scale_max = torch.tensor(init_noise_scale, device=device, dtype=torch.float32)
        self.gae = GAE(0.99, 0.95)
        self.reg_lambda = 0.0  # will be annealed
        self.pulse_kl_weight = float(self.cfg.pulse_kl_coef_start)
        self.num_minibatches = cfg.num_minibatches
        self.progress = 0.0
        self.current_lr = cfg.lr

        self.reward_groups = list(env.cfg.reward.keys())

        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_td = observation_spec.zero().to(device)

        # ---------------------------------------------------------------------------- private encoder
        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([512]), nn.LazyLinear(self.cfg.latent_dim)), [OBS_PRIV_KEY], ["priv_feature"]),
        ).to(device)

        # ---------------------------------------------------------------------------- state estimator (student)
        self.adapt_module = Mod(
            nn.Sequential(
                make_mlp([512, 512]),
                nn.LazyLinear(self.cfg.latent_dim),
            ),
            [OBS_KEY],
            ["priv_pred"],
        ).to(device)
        # ---------------------------------------------------------------------------- actor(s)
        actor_in_keys_train = [OBS_KEY, "priv_feature"]
        actor_in_keys_adapt = [OBS_KEY, "priv_pred"]

        def build_actor(in_keys, *, output_dim: int | None = None, out_key: str = ACTION_KEY, init_scale=None):
            if output_dim is None:
                output_dim = self.action_dim
            if init_scale is None:
                init_scale = init_noise_scale
            return ProbabilisticActor(
                module=Seq(
                    CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                    Mod(make_mlp([1024, 512, 512]), ["_actor_inp"], ["_actor_feature"]),
                    Mod(
                        Actor(
                            output_dim,
                            init_noise_scale=init_scale,
                            load_noise_scale=self.cfg.load_noise_scale if out_key == ACTION_KEY else None,
                        ),
                        ["_actor_feature"],
                        ["loc", "scale"],
                    ),
                ),
                in_keys=["loc", "scale"],
                out_keys=[out_key],
                distribution_class=IndependentNormal,
                return_log_prob=True,
            ).to(device)

        self.actor_teacher = build_actor(actor_in_keys_train)
        self.actor_student = build_actor(actor_in_keys_adapt)
        self.actor_highlevel = build_actor(
            [OBS_KEY],
            output_dim=self.highlevel_action_dim,
            out_key=self.highlevel_action_key,
            init_scale=float(self.cfg.init_noise_scale),
        )

        # --------------------------------------------------------------------- pulse modules
        self.pulse_posterior = Seq(
            CatTensors([OBS_KEY, "priv_feature"], "_pulse_post_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp(self.cfg.pulse_posterior_hidden_dims),
                    nn.LazyLinear(self.cfg.pulse_latent_dim * 2),
                ),
                ["_pulse_post_inp"],
                ["_pulse_post_stats"],
            ),
            Mod(
                Split(self.cfg.pulse_latent_dim),
                ["_pulse_post_stats"],
                ["pulse_post_mu", "pulse_post_logvar"],
            ),
        ).to(device)
        self.pulse_prior = Seq(
            CatTensors([PULSE_PRIOR_OBS_KEY], "_pulse_prior_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp(self.cfg.pulse_prior_hidden_dims),
                    nn.LazyLinear(self.cfg.pulse_latent_dim * 2),
                ),
                ["_pulse_prior_inp"],
                ["_pulse_prior_stats"],
            ),
            Mod(
                Split(self.cfg.pulse_latent_dim),
                ["_pulse_prior_stats"],
                ["pulse_prior_mu", "pulse_prior_logvar"],
            ),
        ).to(device)
        self.pulse_decoder = Seq(
            CatTensors([PULSE_PRIOR_OBS_KEY, "pulse_z"], "_pulse_decoder_inp", del_keys=False, sort=False),
            Mod(
                nn.Sequential(
                    make_mlp(self.cfg.pulse_decoder_hidden_dims),
                    nn.LazyLinear(self.action_dim),
                ),
                ["_pulse_decoder_inp"],
                ["pulse_action"],
            ),
        ).to(device)
        self.pulse_prior_sampler = Mod(
            GaussianSampler(self.cfg.pulse_prior_temp),
            ["pulse_prior_mu", "pulse_prior_logvar"],
            ["pulse_z"],
        ).to(device)
        self.pulse_post_sampler = Mod(
            GaussianSampler(1.0),
            ["pulse_post_mu", "pulse_post_logvar"],
            ["pulse_z"],
        ).to(device)
        self.pulse_action_head = Mod(
            IdentityAction(),
            ["pulse_action"],
            [ACTION_KEY],
        ).to(device)
        self.highlevel_action_split = Mod(
            Split(self.cfg.pulse_latent_dim),
            [self.highlevel_action_key],
            ["delta_z", "wrist_action_replace"],
        ).to(device)
        self.highlevel_latent_barrier = Mod(
            LatentActionBarrier(
                latent_scale=float(getattr(self.cfg, "highlevel_lab_lambda", 1.0)),
                logvar_min=float(getattr(self.cfg, "pulse_logvar_min", -5.0)),
                logvar_max=float(getattr(self.cfg, "pulse_logvar_max", 2.0)),
            ),
            ["pulse_prior_mu", "pulse_prior_logvar", "delta_z"],
            ["pulse_z"],
        ).to(device)
        self.highlevel_wrist_residual = Mod(
            ReplaceJointAction(
                self.highlevel_wrist_joint_ids,
                scale=float(getattr(self.cfg, "highlevel_wrist_action_scale", 1.0)),
            ),
            ["pulse_action", "wrist_action_replace"],
            ["pulse_action"],
        ).to(device)

        # ---------------------------------------------------------------------------- critic (shared)
        self.critic = Seq(
            CatTensors([OBS_KEY, OBS_PRIV_KEY, CRITIC_PRIV_KEY], "_critic_inp", del_keys=False),
            Mod(nn.Sequential(make_mlp([1024, 512, 512]), nn.LazyLinear(1)), ["_critic_inp"], ["state_value"]),
        ).to(device)

        # ---------------------------------------------------------------------------- lazy init pass
        with torch.device(device):
            fake_td["is_init"] = torch.ones(fake_td.shape[0], 1, dtype=torch.bool)
        self.encoder_priv(fake_td)
        self.adapt_module(fake_td)
        self.actor_teacher(fake_td)
        self.actor_student(fake_td)
        fake_td["pulse_z"] = torch.zeros(*fake_td[OBS_KEY].shape[:-1], self.cfg.pulse_latent_dim, device=device)
        self.pulse_posterior(fake_td)
        self.pulse_prior(fake_td)
        self.pulse_prior_sampler(fake_td)
        self.pulse_post_sampler(fake_td)
        self.actor_highlevel(fake_td)
        self.highlevel_action_split(fake_td)
        self.highlevel_latent_barrier(fake_td)
        self.pulse_decoder(fake_td)
        self.highlevel_wrist_residual(fake_td)
        self.pulse_action_head(fake_td)
        self.critic(fake_td)

        # init weights (orthogonal for MLPS/linear)
        def ortho_(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

        self.apply(ortho_)

        self.world_size = 1
        self.num_updates = 0
        if aa.is_distributed():
            self.world_size = aa.get_world_size()
            self._wrap_ddp(local_rank=aa.get_local_rank())

        # ---------------------------------------------------------------------------- optimisers
        self.opt_teacher = torch.optim.Adam(
            list(self.actor_teacher.parameters()) + list(self.encoder_priv.parameters()),
            lr=cfg.lr,
        )
        self.opt_student = torch.optim.Adam(
            self.actor_student.parameters(),
            lr=cfg.lr,
        )
        self.opt_highlevel = torch.optim.Adam(
            self.actor_highlevel.parameters(),
            lr=cfg.lr,
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.opt_estimator = torch.optim.Adam(self.adapt_module.parameters(), lr=cfg.lr)
        self.opt_pulse = torch.optim.Adam(
            list(self.pulse_posterior.parameters())
            + list(self.pulse_prior.parameters())
            + list(self.pulse_decoder.parameters()),
            lr=cfg.lr,
        )

        self.update_teacher = functools.partial(
            self._update,
            actor=self.actor_teacher,
            encoder=self.encoder_priv,
            critic=self.critic,
            opt_actor=self.opt_teacher,
            opt_critic=self.opt_critic,
        )
        self.update_student = functools.partial(
            self._update,
            actor=self.actor_student,
            encoder=self.adapt_module,
            critic=self.critic,
            opt_actor=self.opt_student,
            opt_critic=self.opt_critic,
            update_encoder=False,
            update_actor=True,
        )
        self.update_student_critic = functools.partial(
            self._update,
            actor=self.actor_student,
            encoder=self.adapt_module,
            critic=self.critic,
            opt_actor=self.opt_student,
            opt_critic=self.opt_critic,
            update_encoder=False,
            update_actor=False,
        )
        self.update_highlevel = functools.partial(
            self._update,
            actor=self.actor_highlevel,
            encoder=None,
            critic=self.critic,
            opt_actor=self.opt_highlevel,
            opt_critic=self.opt_critic,
            update_encoder=False,
            update_actor=True,
        )
        self.update2 = functools.partial(self._update2, adapt_module=self.adapt_module, opt_estimator=self.opt_estimator)

        self.use_symmetry_ppo = bool(getattr(self.cfg, "symmetry_enabled", True))
        aa.print(f"use_symmetry_ppo={self.use_symmetry_ppo}")
        if self.use_symmetry_ppo:
            self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms().to(self.device)
            self.obs_priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transforms().to(self.device)
            self.critic_priv_transform = env.observation_funcs[CRITIC_PRIV_KEY].symmetry_transforms().to(self.device)
            self.act_transform = env.action_manager.symmetry_transforms().to(self.device)
        else:
            self.obs_transform = None
            self.obs_priv_transform = None
            self.critic_priv_transform = None
            self.act_transform = None

    def _wrap_ddp(self, local_rank: int):
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank,
                        broadcast_buffers=True, find_unused_parameters=False)

        self.actor_teacher = DDP(self.actor_teacher, **ddp_kwargs)
        self.actor_student = DDP(self.actor_student, **ddp_kwargs)
        self.actor_highlevel = DDP(self.actor_highlevel, **ddp_kwargs)
        self.encoder_priv  = DDP(self.encoder_priv,  **ddp_kwargs)
        self.critic        = DDP(self.critic,        **ddp_kwargs)
        self.adapt_module  = DDP(self.adapt_module,  **ddp_kwargs)
        self.pulse_posterior = DDP(self.pulse_posterior, **ddp_kwargs)
        self.pulse_prior = DDP(self.pulse_prior, **ddp_kwargs)
        self.pulse_decoder = DDP(self.pulse_decoder, **ddp_kwargs)

    def broadcast_parameters(self, extra_modules=[]):
        if self.num_updates % 32 == 0:
            update_list = [self.value_norm] + extra_modules
            if aa.is_distributed():
                for m in update_list:
                    for p in m.parameters():
                        dist.broadcast(p, src=0)
                    for p in m.buffers():
                        dist.broadcast(p, src=0)

    def _resolve_init_noise_scale(self):
        base_scale = float(self.cfg.init_noise_scale)
        overrides = getattr(self.cfg, "init_noise_scale_overrides", None) or {}
        overrides = dict(overrides)
        if not overrides:
            return base_scale

        scales = [base_scale] * self.action_dim
        joint_ids, _, joint_scales = self.action_manager.resolve(
            overrides, names=self.joint_names
        )
        for idx, scale in zip(joint_ids, joint_scales):
            scales[idx] = float(scale)
        return scales

    def _resolve_highlevel_wrist_joints(self):
        if not bool(getattr(self.cfg, "highlevel_wrist_residual_enabled", False)):
            return [], []
        patterns = list(getattr(self.cfg, "highlevel_wrist_joint_patterns", []) or [])
        if len(patterns) == 0:
            warnings.warn(
                "highlevel_wrist_residual_enabled=True but highlevel_wrist_joint_patterns is empty. "
                "Disable wrist replace branch."
            )
            return [], []
        ids = []
        names = []
        for i, name in enumerate(self.joint_names):
            if any(re.match(pattern, name) for pattern in patterns):
                ids.append(i)
                names.append(name)
        if len(ids) == 0:
            warnings.warn(
                f"No joints matched highlevel_wrist_joint_patterns={patterns}. "
                "Disable wrist replace branch."
            )
            return [], []
        aa.print(
            "highlevel wrist replace joints: "
            + ", ".join(f"{i}:{n}" for i, n in zip(ids, names))
        )
        return ids, names

    def do_lr_schedule(self, kl):
        if not hasattr(self, "current_lr"):
            self.current_lr = self.cfg.lr
        
        if self.progress < 0.1:
            return

        if aa.is_distributed():
            kl_tensor = torch.tensor(kl, device=self.device)
            dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
            kl = (kl_tensor / self.world_size).item()

        new_lr = self.current_lr
        if kl > self.cfg.desired_kl * 2.0:
            new_lr = max(1e-5, new_lr / 1.1)
        elif 0.0 < kl < self.cfg.desired_kl / 2.0:
            new_lr = min(5e-3, new_lr * 1.1)

        self.current_lr = new_lr

        for opt in (self.opt_teacher, self.opt_student, self.opt_highlevel):
            for param_group in opt.param_groups:
                param_group["lr"] = self.current_lr

    def make_tensordict_primer(self):
        return None

    def get_rollout_policy(self, mode: str = "train"):
        modules = []
        if mode == "pulse_random":
            modules += [self.pulse_prior, self.pulse_prior_sampler, self.pulse_decoder, self.pulse_action_head]
            return Seq(*modules)
        if self.cfg.phase == "train":
            modules += [self.encoder_priv, self.actor_teacher]
        elif self.cfg.phase == "finetune":
            modules += [self.adapt_module]
            modules += [self.actor_student]
        elif self.cfg.phase == "adapt":
            modules += [self.adapt_module]
            modules += [self.actor_student]
        elif self.cfg.phase == "highlevel":
            modules += [self.actor_highlevel, self.highlevel_action_split, self.pulse_prior]
            modules += [
                self.highlevel_latent_barrier,
                self.pulse_decoder,
                self.highlevel_wrist_residual,
                self.pulse_action_head,
            ]
        elif self.cfg.phase == "pulse":
            modules += [self.encoder_priv, self.pulse_posterior, self.pulse_post_sampler, self.pulse_decoder, self.pulse_action_head]
        policy = Seq(*modules)
        return policy

    def step_schedule(self, progress: float, iter: int):
        self.reg_lambda = progress * self.cfg.reg_lambda
        start = self.cfg.entropy_coef_start
        end = self.cfg.entropy_coef_end
        # exponential decay from start to end based on progress in [0,1]
        self.entropy_coef = start * (end / start) ** progress
        if self.cfg.phase == "pulse":
            anneal_start = float(self.cfg.pulse_kl_anneal_start)
            anneal_end = float(self.cfg.pulse_kl_anneal_end)
            beta_start = float(self.cfg.pulse_kl_coef_start)
            beta_end = float(self.cfg.pulse_kl_coef_end)
            if anneal_end <= anneal_start:
                self.pulse_kl_weight = beta_end
            elif progress <= anneal_start:
                self.pulse_kl_weight = beta_start
            elif progress >= anneal_end:
                self.pulse_kl_weight = beta_end
            else:
                ratio = (progress - anneal_start) / (anneal_end - anneal_start)
                if beta_start > 0.0 and beta_end > 0.0:
                    log_beta = torch.lerp(
                        torch.tensor(beta_start, device=self.device).log(),
                        torch.tensor(beta_end, device=self.device).log(),
                        torch.tensor(ratio, device=self.device),
                    )
                    self.pulse_kl_weight = float(log_beta.exp().item())
                else:
                    self.pulse_kl_weight = float(beta_start + ratio * (beta_end - beta_start))
        self.progress = progress

    def train_op(self, td: TensorDict, vecnorm):
        """One optimisation step on a batched rollout tensor-dict."""
        if self.cfg.phase == "train":
            info = {}
            info.update(self._ppo_update(td, self.update_teacher))
            info.update(self.train_estimator(td))
        elif self.cfg.phase == "finetune":
            info = {}
            if self.progress > 0.025:
                info.update(self._ppo_update(td, self.update_student))
            else:
                info.update(self._ppo_update(td, self.update_student_critic))
        elif self.cfg.phase == "pulse":
            info = self.train_pulse(td)
        elif self.cfg.phase == "highlevel":
            info = {}
            info.update(self._ppo_update(td, self.update_highlevel))
        else:  # adapt
            info = self.train_estimator(td)
        self.num_updates += 1
        self.broadcast_parameters(extra_modules=[vecnorm])
        return info

    def _ppo_update(self, td, update_func: callable = None):
        infos = []
        reward_clamp_min = getattr(self.cfg, "adv_reward_clamp_min", 0.0)
        self._compute_advantage(td, self.critic, self.gae, self.value_norm, 
                               REWARD_KEY=REWARD_KEY, TERM_KEY=TERM_KEY, DONE_KEY=DONE_KEY,
                               reward_clamp_min=reward_clamp_min)
        self._modewise_adv_norm(td)

        for _ in range(self.cfg.ppo_epochs):
            for mb in make_batch(td, self.num_minibatches):
                infos.append(TensorDict(update_func(mb), []))
        info = {k: v.mean().item() for k, v in torch.stack(infos).items()}

        with torch.no_grad():
            if self.cfg.phase == "train":
                actor = self.actor_teacher
            elif self.cfg.phase == "highlevel":
                actor = self.actor_highlevel
            else:
                actor = self.actor_student
            base = actor.module if isinstance(actor, DDP) else actor
            action_std = base.module[0][2].module.actor_std.detach()
            if self.cfg.phase != "highlevel":
                for joint_name, std in zip(self.joint_names, action_std):
                    info[f"actor_std/{joint_name}"] = std
            else:
                latent_dim = int(self.cfg.pulse_latent_dim)
                latent_std = action_std[:latent_dim]
                info["actor_std/latent_min"] = latent_std.min()
                info["actor_std/latent_max"] = latent_std.max()
                info["actor_std/latent_mean"] = latent_std.mean()
                if self.highlevel_wrist_dim > 0:
                    wrist_std = action_std[latent_dim: latent_dim + self.highlevel_wrist_dim]
                    info["actor_std/wrist_min"] = wrist_std.min()
                    info["actor_std/wrist_max"] = wrist_std.max()
                    info["actor_std/wrist_mean"] = wrist_std.mean()
                    for name, std in zip(self.highlevel_wrist_joint_names, wrist_std):
                        info[f"actor_std/wrist/{name}"] = std
            info["actor_std/mean"] = action_std.mean()

        kl = info["actor/kl"]
        self.do_lr_schedule(kl)
        info["lr"] = self.current_lr

        neg_reward_ratio = (td[REWARD_KEY] <= 0.0).float().mean().item()
        info["critic/neg_reward_ratio"] = neg_reward_ratio

        return info

    def _update(
        self,
        mb,
        actor=None,
        encoder=None,
        critic=None,
        opt_actor=None,
        opt_critic=None,
        update_actor: bool = True,
        update_encoder: bool = True,
    ):
        def _module_grads_finite(module: nn.Module | DDP | None) -> bool:
            if module is None:
                return True
            base = module.module if isinstance(module, DDP) else module
            for p in base.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return False
            return True

        def _all_finite(*tensors: torch.Tensor) -> bool:
            for t in tensors:
                if t is None:
                    continue
                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                    return False
            return True

        def _skip_update_info() -> dict[str, torch.Tensor]:
            z = torch.tensor(0.0, device=self.device)
            return {
                "actor/policy_loss": z,
                "actor/entropy": z,
                "adapt/reg_loss": z,
                "actor/actor_grad_norm": z,
                "action/encoder_grad_norm": z,
                "actor/clamp_ratio": z,
                "critic/critic_grad_norm": z,
                "actor/kl": z,
                "actor/symmetry_loss_loc": z,
                "actor/symmetry_loss_std": z,
                "critic/explained_var": z,
                "critic/value_loss": z,
                "train/nonfinite_skip": torch.tensor(1.0, device=self.device),
            }

        bsize = mb.shape[0]
        loc_old, scale_old = mb["loc"].clone(), mb["scale"].clone()
        if self.cfg.phase == "highlevel":
            action_key = self.highlevel_action_key
            logp_key = f"{action_key}_log_prob"
            action_old = mb[action_key].clone()
            logp_old = mb[logp_key].clone()
            exclude_keys = [
                logp_key,
                action_key,
                "delta_z",
                "wrist_action_replace",
                "action_log_prob",
                "action",
            ]
        else:
            action_old = mb["action"].clone()
            logp_old = mb["action_log_prob"].clone()
            exclude_keys = ["action_log_prob", "action"]

        if self.use_symmetry_ppo:
            mb_sym = mb.clone()
            mb_sym[OBS_KEY] = self.obs_transform(mb_sym[OBS_KEY])
            mb_sym[OBS_PRIV_KEY] = self.obs_priv_transform(mb_sym[OBS_PRIV_KEY])
            mb_sym[CRITIC_PRIV_KEY] = self.critic_priv_transform(mb_sym[CRITIC_PRIV_KEY])
            mb_sym["adv"] = mb["adv"]
            mb_sym["ret"] = mb["ret"]
            mb_sym["is_init"] = mb["is_init"]

            mb_sym = mb_sym.exclude("next")
            mb = mb.exclude("next")
            mb = torch.cat([mb, mb_sym], dim=0)
        else:
            mb = mb.exclude("next")
        valid = ~mb["is_init"]
        keys_all = mb.keys(True, True)
        exclude_keys = [k for k in exclude_keys if k in keys_all]
        mb = mb.exclude(*exclude_keys)

        if not _all_finite(
            loc_old,
            scale_old,
            action_old,
            logp_old,
            mb.get("adv", None),
            mb.get("ret", None),
        ):
            return _skip_update_info()

        if encoder is not None:
            if update_encoder:
                encoder(mb)
            else:
                with torch.no_grad():
                    encoder(mb)
        
        if update_actor:
            actor(mb)
        else:
            with torch.no_grad():
                actor(mb)

        loc = mb["loc"][:bsize]
        scale = mb["scale"][:bsize]
        if not _all_finite(loc, scale):
            return _skip_update_info()

        dist = IndependentNormal(loc, scale)
        logp = dist.log_prob(action_old)
        entropy = dist.entropy().mean()
        if not _all_finite(logp, entropy):
            return _skip_update_info()

        ratio = torch.exp(logp - logp_old).unsqueeze(-1)
        if not _all_finite(ratio):
            return _skip_update_info()
        surr1 = mb["adv"][:bsize] * ratio
        surr2 = mb["adv"][:bsize] * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2) * valid[:bsize])
        entropy_loss = - self.entropy_coef * entropy

        values = critic(mb)["state_value"]
        value_loss = F.mse_loss(mb["ret"], values, reduction="none")
        value_loss = (value_loss * valid).mean(dim=0)
        if not _all_finite(policy_loss, entropy_loss, value_loss):
            return _skip_update_info()

        if self.cfg.phase == "train":
            if "priv_pred" not in mb.keys():
                with torch.no_grad():
                    self.adapt_module(mb)
            reg_loss = F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none")
            reg_loss = self.reg_lambda * torch.mean(reg_loss * valid)
        else:
            reg_loss = 0.0
        
        if self.use_symmetry_ppo:
            symmetry_loss_loc = F.mse_loss(mb["loc"][:bsize], self.act_transform(mb["loc"][bsize:])) * 0.2
            symmetry_loss_std = F.mse_loss(
                mb["scale"][:bsize],
                self.act_transform(mb["scale"][bsize:], sign=False),
            ) * 10
        else:
            symmetry_loss_loc = torch.zeros((), device=self.device)
            symmetry_loss_std = torch.zeros((), device=self.device)

        loss = (
            policy_loss
            + entropy_loss
            + value_loss.mean()
            + reg_loss
            + symmetry_loss_loc
            + symmetry_loss_std
        )
        if not _all_finite(loss):
            return _skip_update_info()

        # do optimisation step
        opt_actor.zero_grad()
        opt_critic.zero_grad()

        loss.backward()

        if update_encoder and update_actor and encoder is not None:
            encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        else:
            encoder_grad_norm = torch.tensor(0.0, device=self.device)

        actor_step_ok = True
        if update_actor:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_step_ok = _all_finite(actor_grad_norm) and _module_grads_finite(actor)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)

        critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_step_ok = _all_finite(critic_grad_norm) and _module_grads_finite(critic)

        if not (actor_step_ok and critic_step_ok):
            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            return _skip_update_info()

        if update_actor:
            opt_actor.step()
            self._clamp_actor_std(actor)
        opt_critic.step()

        with torch.no_grad():
            explained_var = 1 - value_loss / (mb["ret"] * valid).var(dim=0).clamp_min(1.0e-6)
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = mb["loc"][:bsize], mb["scale"][:bsize]
            scale_safe = scale.clamp_min(1.0e-6)
            scale_old_safe = scale_old.clamp_min(1.0e-6)
            kl = torch.sum(
                torch.log(scale_safe) - torch.log(scale_old_safe)
                + (torch.square(scale_old_safe) + torch.square(loc_old - loc))
                / (2.0 * torch.square(scale_safe))
                - 0.5,
                axis=-1,
            ).mean()

        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "adapt/reg_loss": reg_loss if isinstance(reg_loss, torch.Tensor) else torch.tensor(0.0),
            "actor/actor_grad_norm": actor_grad_norm,
            "action/encoder_grad_norm": encoder_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "critic/critic_grad_norm": critic_grad_norm,
            "actor/kl": kl.detach(),
            "actor/symmetry_loss_loc": symmetry_loss_loc.detach(),
            "actor/symmetry_loss_std": symmetry_loss_std.detach(),
            "train/nonfinite_skip": torch.tensor(0.0, device=self.device),
        }

        info["critic/explained_var"] = explained_var.mean().detach()
        info["critic/value_loss"] = value_loss.mean().detach()
    
        return info

    def _clamp_actor_std(self, actor):
        base = actor.module if isinstance(actor, DDP) else actor
        actor_std = None
        for module in base.modules():
            if isinstance(module, Actor) and hasattr(module, "actor_std"):
                actor_std = module.actor_std
                break
        if actor_std is None:
            return
        max_scale = self._init_noise_scale_max
        if isinstance(max_scale, torch.Tensor):
            if max_scale.numel() > 1 and max_scale.numel() != actor_std.numel():
                max_scale = torch.full_like(actor_std.data, float(max_scale.mean().item()))
        reset_scale = float(max_scale.mean().item()) if isinstance(max_scale, torch.Tensor) else float(max_scale)
        actor_std.data = torch.nan_to_num(
            actor_std.data,
            nan=reset_scale,
            posinf=reset_scale,
            neginf=1.0e-4,
        )
        actor_std.data.clamp_(min=1.0e-4)
        actor_std.data = torch.minimum(actor_std.data, max_scale)

    def train_estimator(self, td):
        infos = []
        
        for _ in range(2):
            for mb in make_batch(td, self.num_minibatches, self.cfg.train_every):
                infos.append(TensorDict(self.update2(mb), []))

        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    def train_pulse(self, td):
        infos = []

        for _ in range(self.cfg.pulse_epochs):
            for mb in make_batch(td, self.num_minibatches, self.cfg.train_every):
                infos.append(TensorDict(self._update_pulse(mb), []))

        return {k: v.mean().item() for k, v in torch.stack(infos).items()}

    def _update2(self, mb, adapt_module, opt_estimator):
        if self.use_symmetry_ppo:
            mb_sym = mb.clone()
            mb_sym[OBS_KEY] = self.obs_transform(mb_sym[OBS_KEY])
            mb_sym[OBS_PRIV_KEY] = self.obs_priv_transform(mb_sym[OBS_PRIV_KEY])
            mb_sym[CRITIC_PRIV_KEY] = self.critic_priv_transform(mb_sym[CRITIC_PRIV_KEY])
            mb_sym["is_init"] = mb["is_init"]

            mb_sym = mb_sym.exclude("next")
            mb = mb.exclude("next")
            mb = torch.cat([mb, mb_sym], dim=0)
        else:
            mb = mb.exclude("next")

        with torch.no_grad():
            self.encoder_priv(mb)
        adapt_module(mb)

        valid = ~mb["is_init"]
        loss = torch.mean(F.mse_loss(mb["priv_pred"], mb["priv_feature"], reduction="none") * (valid))

        opt_estimator.zero_grad()
        loss.backward()
        opt_estimator.step()

        return {"adapt/estimator_loss": loss.detach()}

    def _update_pulse(self, mb):
        done = mb["next", "done"].clone() if ("next", "done") in mb.keys(True, True) else None
        mb = mb.exclude("next")

        with torch.no_grad():
            self.encoder_priv(mb)
            self.adapt_module(mb)
            self.actor_teacher(mb)

        teacher_action = mb["loc"].detach().clone()

        self.pulse_posterior(mb)
        self.pulse_prior(mb)

        post_mu = mb["pulse_post_mu"]
        post_logvar = mb["pulse_post_logvar"].clamp(self.cfg.pulse_logvar_min, self.cfg.pulse_logvar_max)
        prior_mu = mb["pulse_prior_mu"]
        prior_logvar = mb["pulse_prior_logvar"].clamp(self.cfg.pulse_logvar_min, self.cfg.pulse_logvar_max)

        mb["pulse_post_logvar"] = post_logvar
        mb["pulse_prior_logvar"] = prior_logvar

        pulse_z = self._sample_gaussian(post_mu, post_logvar)
        mb["pulse_z"] = pulse_z
        self.pulse_decoder(mb)

        valid = (~mb["is_init"]).float()
        action_loss = torch.mean(
            F.mse_loss(mb["pulse_action"], teacher_action, reduction="none") * valid
        )

        kl = self._diag_gaussian_kl(post_mu, post_logvar, prior_mu, prior_logvar)
        kl_loss = torch.mean(kl * valid)

        regu_loss = torch.zeros((), device=self.device)
        if self.cfg.pulse_use_temporal_reg and post_mu.ndim >= 3:
            regu_valid = (~mb["is_init"][:, 1:]) & (~mb["is_init"][:, :-1])
            if done is not None:
                regu_valid = regu_valid & (~done[:, :-1])
            if regu_valid.any():
                regu_valid = regu_valid.float()
                regu_loss = torch.mean(
                    torch.mean((post_mu[:, 1:] - post_mu[:, :-1]).square(), dim=-1, keepdim=True) * regu_valid
                )

        loss = action_loss + self.pulse_kl_weight * kl_loss + float(self.cfg.pulse_regu_coef) * regu_loss

        self.opt_pulse.zero_grad()
        loss.backward()
        pulse_grad_norm = nn.utils.clip_grad_norm_(
            list(self.pulse_posterior.parameters())
            + list(self.pulse_prior.parameters())
            + list(self.pulse_decoder.parameters()),
            1.0,
        )
        self.opt_pulse.step()

        with torch.no_grad():
            post_std = torch.exp(0.5 * post_logvar)
            prior_std = torch.exp(0.5 * prior_logvar)

        return {
            "pulse/loss": loss.detach(),
            "pulse/action_loss": action_loss.detach(),
            "pulse/kl_loss": kl_loss.detach(),
            "pulse/regu_loss": regu_loss.detach(),
            "pulse/kl_weight": torch.tensor(self.pulse_kl_weight, device=self.device),
            "pulse/post_std_mean": post_std.mean().detach(),
            "pulse/prior_std_mean": prior_std.mean().detach(),
            "pulse/post_mu_norm": post_mu.norm(dim=-1).mean().detach(),
            "pulse/prior_mu_norm": prior_mu.norm(dim=-1).mean().detach(),
            "pulse/grad_norm": pulse_grad_norm.detach() if isinstance(pulse_grad_norm, torch.Tensor) else torch.tensor(pulse_grad_norm, device=self.device),
        }

    @staticmethod
    @torch.compile
    @torch.no_grad()
    def _compute_advantage(
        td,
        critic,
        gae,
        value_norm,
        REWARD_KEY="reward",
        TERM_KEY="term",
        DONE_KEY="done",
        reward_clamp_min: float | None = 0.0,
    ):
        keys = td.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with td.view(-1) as flat:
                critic(flat)
                critic(flat["next"])

        v = td["state_value"]
        v_next = td["next", "state_value"]

        rewards = td[REWARD_KEY].sum(dim=-1, keepdim=True)
        if reward_clamp_min is not None:
            rewards = rewards.clamp_min(float(reward_clamp_min))

        adv, ret = gae(
            rewards,
            td[TERM_KEY],
            td[DONE_KEY],
            value_norm.denormalize(v),
            value_norm.denormalize(v_next),
        )

        value_norm.update(ret)
        td["adv"], td["ret"] = adv, value_norm.normalize(ret)

    @staticmethod
    @torch.compile
    def get_global_mean_std(x: torch.Tensor, mask: torch.Tensor):
        if aa.is_distributed():
            local_count = mask.sum()

            local_sum = (x * mask).sum()
            local_sum_sq = (x * x * mask).sum()

            stats = torch.stack([local_sum, local_sum_sq, local_count.float()])

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

            global_sum, global_sum_sq, global_count = stats
            global_count.clamp_min_(1)

            global_mean = global_sum / global_count
            global_var = (global_sum_sq / global_count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=0.0)).clamp(min=1e-5)
        else:
            count = mask.sum().clamp_min_(1)
            sum = (x * mask).sum()
            sum_sq = (x * x * mask).sum()

            global_mean = sum / count
            global_var = (sum_sq / count) - (global_mean * global_mean)
            global_std = torch.sqrt(global_var.clamp(min=0.0)).clamp(min=1e-5)
        return global_mean, global_std

    def _modewise_adv_norm(self, td):
        adv = td["adv"]
        is_init = td["is_init"]
        
        mask = ~is_init
        mean_mode, std_mode = self.get_global_mean_std(adv, mask)
        adv[mask] = (adv[mask] - mean_mode) / std_mode

    def state_dict(self):
        state = OrderedDict()
        for n, m in self.named_children():
            if isinstance(m, DDP):
                state[n] = m.module.state_dict()
            else:
                state[n] = m.state_dict()

        state["last_phase"] = self.cfg.phase

        state["_meta"] = {
            "current_lr": getattr(self, "current_lr", self.cfg.lr),
            "entropy_coef": getattr(self, "entropy_coef", self.cfg.entropy_coef_start),
            "reg_lambda": getattr(self, "reg_lambda", 0.0),
            "progress": getattr(self, "progress", 0.0),
            "num_updates": getattr(self, "num_updates", 0),
            "world_size": getattr(self, "world_size", 1),
        }

        return state

    def load_state_dict(self, state_dict, strict=True):
        for n, m in self.named_children():
            if n not in state_dict:
                continue
            try:
                if isinstance(m, DDP):
                    m.module.load_state_dict(state_dict.get(n, {}), strict=strict)
                else:
                    m.load_state_dict(state_dict.get(n, {}), strict=strict)
            except Exception as e:
                warnings.warn(f"Failed to load {n}: {e}")

        last_phase = state_dict.get("last_phase", "train")

        # Initialize student actor from teacher if starting from a 'train' phase checkpoint
        if last_phase == "train":
            warnings.warn("Last phase was 'train'. Performing a hard copy from `actor_teacher` to `actor_student`.")
            self.hard_copy_(self.actor_teacher, self.actor_student)

        meta = state_dict.get("_meta", {})
        if state_dict.get("last_phase") == self.cfg.phase:
            self.current_lr   = meta.get("current_lr", getattr(self, "current_lr", self.cfg.lr))
            self.entropy_coef = meta.get("entropy_coef", self.entropy_coef)
            self.reg_lambda   = meta.get("reg_lambda", self.reg_lambda)
            self.progress     = meta.get("progress", self.progress)
            self.num_updates  = meta.get("num_updates", self.num_updates)

    def load_teacher_state_dict(self, state_dict, strict=True):
        teacher_modules = ("encoder_priv", "actor_teacher")
        for name in teacher_modules:
            if name not in state_dict:
                continue
            module = getattr(self, name, None)
            if module is None:
                continue
            try:
                if isinstance(module, DDP):
                    module.module.load_state_dict(state_dict[name], strict=strict)
                else:
                    module.load_state_dict(state_dict[name], strict=strict)
            except Exception as exc:
                warnings.warn(f"Failed to load teacher module {name}: {exc}")

    def load_pulse_modules_state_dict(self, state_dict, strict=True):
        pulse_modules = ("pulse_prior", "pulse_decoder")
        for name in pulse_modules:
            if name not in state_dict:
                warnings.warn(f"Pulse checkpoint missing required module '{name}'.")
                continue
            module = getattr(self, name, None)
            if module is None:
                warnings.warn(f"Current policy has no module '{name}' to load.")
                continue
            try:
                if isinstance(module, DDP):
                    module.module.load_state_dict(state_dict[name], strict=strict)
                else:
                    module.load_state_dict(state_dict[name], strict=strict)
            except Exception as exc:
                warnings.warn(f"Failed to load pulse module {name}: {exc}")

    # Backward compatibility: keep old API name used by existing callers.
    def load_highlevel_from_pulse_state_dict(self, state_dict, strict=True):
        self.load_pulse_modules_state_dict(state_dict, strict=strict)

    def prepare_for_phase(self):
        if self.cfg.phase == "pulse":
            self._freeze_module(self.encoder_priv)
            self._freeze_module(self.actor_teacher)
            self._freeze_module(self.adapt_module)
            return
        if self.cfg.phase == "highlevel":
            self._freeze_module(self.encoder_priv)
            self._freeze_module(self.actor_teacher)
            self._freeze_module(self.adapt_module)
            self._freeze_module(self.pulse_posterior)
            self._freeze_module(self.pulse_prior)
            self._freeze_module(self.pulse_decoder)
            return

    @staticmethod
    def _freeze_module(module):
        base = module.module if isinstance(module, DDP) else module
        base.requires_grad_(False)
        base.eval()

    @staticmethod
    def _sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def _diag_gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = logvar_p - logvar_q + (var_q + (mu_q - mu_p).square()) / var_p - 1.0
        return 0.5 * kl.sum(dim=-1, keepdim=True)

    @staticmethod
    def soft_copy_(src_module: nn.Module, dst_module: nn.Module, tau: float):
        src = src_module.module if isinstance(src_module, DDP) else src_module
        dst = dst_module.module if isinstance(dst_module, DDP) else dst_module

        with torch.no_grad():
            src_params = dict(src.named_parameters())
            for name, dst_param in dst.named_parameters():
                if name in src_params:
                    src_param = src_params[name]
                    # The requires_grad status of dst_param is maintained
                    dst_param.data.copy_(
                        tau * src_param.data + (1.0 - tau) * dst_param.data
                    )
    
    @staticmethod
    def hard_copy_(src_module: nn.Module, dst_module: nn.Module):
        PPOPolicy.soft_copy_(src_module, dst_module, 1.0)
