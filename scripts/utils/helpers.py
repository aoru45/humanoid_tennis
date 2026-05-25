import torch
import torch.nn as nn
import hydra
import numpy as np
import time
import logging
import os
import datetime
import sys
import importlib

from typing import Sequence, Dict, Any, Tuple, List
from pathlib import Path
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.envs.transforms import VecNorm

from termcolor import colored
from collections import OrderedDict
import imageio
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig
import humanoid_tennis.learning
from humanoid_tennis.utils.wandb import parse_checkpoint_path
import humanoid_tennis


_LEGACY_CKPT_ALIAS_READY = False


def _install_legacy_checkpoint_aliases() -> None:
    """Allow loading checkpoints pickled with old module path `active_adaptation.*`."""
    global _LEGACY_CKPT_ALIAS_READY
    if _LEGACY_CKPT_ALIAS_READY:
        return

    # Ensure key policy modules are imported before aliasing.
    for module_name in (
        "humanoid_tennis.learning",
        "humanoid_tennis.learning.ppo",
        "humanoid_tennis.learning.ppo.common",
        "humanoid_tennis.learning.ppo.ppo",
        "humanoid_tennis.learning.modules",
        "humanoid_tennis.learning.modules.distributions",
    ):
        try:
            importlib.import_module(module_name)
        except Exception:
            pass

    # Alias all already-loaded humanoid_tennis modules to active_adaptation namespace.
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == "humanoid_tennis" or name.startswith("humanoid_tennis."):
            legacy_name = "active_adaptation" + name[len("humanoid_tennis") :]
            sys.modules.setdefault(legacy_name, module)

    _LEGACY_CKPT_ALIAS_READY = True


def _resolve_local_checkpoint_path(path: str | None) -> str | None:
    if path is None or str(path).startswith("run:"):
        return path
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return str(p)
    if HydraConfig.initialized():
        cwd = HydraConfig.get().runtime.cwd
        return str((Path(cwd) / p).resolve())
    return str(p.resolve())

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


class ObsNorm(ModBase):
    def __init__(self, in_keys, out_keys, locs, scales):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        
        self.loc = nn.ParameterDict({k: nn.Parameter(locs[k]) for k in in_keys})
        self.scale = nn.ParameterDict({k: nn.Parameter(scales[k]) for k in out_keys})
        self.requires_grad_(False)

    def forward(self, tensordict: TensorDictBase):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            obs = tensordict.get(in_key, None)
            if obs is not None:
                loc = self.loc[in_key]
                scale = self.scale[out_key]
                tensordict.set(out_key, (obs - loc) / scale)
        return tensordict
    
    @classmethod
    def from_vecnorm(cls, vecnorm: VecNorm, keys):
        in_keys = []
        out_keys = []
        for in_key, out_key in zip(vecnorm.in_keys, vecnorm.out_keys):
            if in_key in keys:
                in_keys.append(in_key)
                out_keys.append(out_key)
        return cls(
            in_keys=in_keys,
            out_keys=out_keys,
            locs=vecnorm.loc,
            scales=vecnorm.scale
        )


def _load_vecnorm_state(vecnorm: VecNorm, ckpt_state: Dict[str, Any]) -> Tuple[bool, str]:
    # First try strict/full loading for true resume of identical observation spaces.
    try:
        vecnorm.load_state_dict(ckpt_state)
        return True, "Loaded VecNorm state fully."
    except Exception as full_exc:
        full_err = str(full_exc)

    # Fallback for cross-phase initialization: only load compatible observation-group stats.
    current_state = vecnorm.state_dict()
    ckpt_extra = ckpt_state.get("_extra_state", None)
    current_extra = current_state.get("_extra_state", None)
    if not isinstance(ckpt_extra, dict) or not isinstance(current_extra, dict):
        return False, f"Failed full VecNorm load and no compatible _extra_state fallback. full_error={full_err}"

    def _group_names(extra: Dict[str, Any]) -> List[str]:
        names = set()
        for key in extra.keys():
            if isinstance(key, str) and key.endswith("_sum"):
                names.add(key[: -len("_sum")])
        return sorted(names)

    loaded_groups: List[str] = []
    skipped_groups: List[str] = []
    for group in _group_names(current_extra):
        sum_key = f"{group}_sum"
        ssq_key = f"{group}_ssq"
        cnt_key = f"{group}_count"
        trio = (sum_key, ssq_key, cnt_key)

        if any(k not in ckpt_extra for k in trio):
            skipped_groups.append(f"{group}(missing)")
            continue
        if any(k not in current_extra for k in trio):
            skipped_groups.append(f"{group}(missing_current)")
            continue

        ckpt_sum = ckpt_extra[sum_key]
        ckpt_ssq = ckpt_extra[ssq_key]
        ckpt_cnt = ckpt_extra[cnt_key]
        cur_sum = current_extra[sum_key]
        cur_ssq = current_extra[ssq_key]
        cur_cnt = current_extra[cnt_key]

        if not (
            torch.is_tensor(ckpt_sum)
            and torch.is_tensor(ckpt_ssq)
            and torch.is_tensor(ckpt_cnt)
            and torch.is_tensor(cur_sum)
            and torch.is_tensor(cur_ssq)
            and torch.is_tensor(cur_cnt)
        ):
            skipped_groups.append(f"{group}(non_tensor)")
            continue

        same_shape = (
            ckpt_sum.shape == cur_sum.shape
            and ckpt_ssq.shape == cur_ssq.shape
            and ckpt_cnt.shape == cur_cnt.shape
        )
        if not same_shape:
            skipped_groups.append(
                f"{group}(shape_ckpt={tuple(ckpt_sum.shape)}/{tuple(ckpt_ssq.shape)}/{tuple(ckpt_cnt.shape)}"
                f",cur={tuple(cur_sum.shape)}/{tuple(cur_ssq.shape)}/{tuple(cur_cnt.shape)})"
            )
            continue

        current_extra[sum_key] = ckpt_sum.to(device=cur_sum.device, dtype=cur_sum.dtype)
        current_extra[ssq_key] = ckpt_ssq.to(device=cur_ssq.device, dtype=cur_ssq.dtype)
        current_extra[cnt_key] = ckpt_cnt.to(device=cur_cnt.device, dtype=cur_cnt.dtype)
        loaded_groups.append(group)

    if len(loaded_groups) == 0:
        return False, f"Failed full VecNorm load; no compatible groups found. full_error={full_err}"

    current_state["_extra_state"] = current_extra
    try:
        vecnorm.load_state_dict(current_state)
    except Exception as partial_exc:
        return (
            False,
            "Failed full VecNorm load and partial compatible-group load also failed. "
            f"full_error={full_err}; partial_error={partial_exc}",
        )

    return (
        True,
        "Loaded VecNorm partially with compatible groups only. "
        f"loaded={loaded_groups}, skipped={skipped_groups}",
    )

class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> int:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)
    
    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()

import torch.distributed as dist
from torchrl._utils import _append_last
from torchrl.envs.transforms.transforms import _sum_left

import humanoid_tennis as aa


class SymmetricVecNorm(VecNorm):
    """VecNorm that updates running stats with both sample and mirrored sample."""

    def __init__(self, *args, symmetry_transforms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._symmetry_transforms = symmetry_transforms or {}

    def _get_symmetry(self, key):
        transform = self._symmetry_transforms.get(key)
        if transform is None and isinstance(key, tuple):
            transform = self._symmetry_transforms.get(key[-1])
        return transform

    def _update(self, key, value, N) -> torch.Tensor:
        _sum = self._td.get(_append_last(key, "_sum"))
        _ssq = self._td.get(_append_last(key, "_ssq"))
        _count = self._td.get(_append_last(key, "_count"))

        value_sum = _sum_left(value, _sum)
        value_ssq = _sum_left(value.pow(2), _ssq)
        count_update = N

        symmetry = self._get_symmetry(key)
        if symmetry is not None:
            if symmetry.perm.numel() != value.shape[-1]:
                raise RuntimeError(
                    f"Symmetry dim mismatch for key '{key}': "
                    f"perm={symmetry.perm.numel()} vs value={value.shape[-1]}"
                )
            value_sum = value_sum + symmetry(value_sum, sign=True)
            value_ssq = value_ssq + symmetry(value_ssq, sign=False)
            count_update = 2 * N

        if not self.frozen:
            _sum *= self.decay
            _sum += value_sum
            self._td.set_(_append_last(key, "_sum"), _sum)

            _ssq *= self.decay
            _ssq += value_ssq
            self._td.set_(_append_last(key, "_ssq"), _ssq)

            _count *= self.decay
            _count += count_update
            self._td.set_(_append_last(key, "_count"), _count)

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)


def make_env_policy(cfg: DictConfig, return_checkpoint_state: bool = False):
    OmegaConf.set_struct(cfg, False)
    from humanoid_tennis.envs import SimpleEnv
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, StepCounter
    # Propagate top-level/app headless flag into task.viewer for MJLab GUI.
    if "app" in cfg and "headless" in cfg.app:
        cfg.task.viewer.headless = cfg.app.headless
    elif "headless" in cfg:
        cfg.task.viewer.headless = cfg.headless
    aa.print("import SimpleEnv done")
    env_cls_path = str(cfg.task.get("env_class", "humanoid_tennis.envs.locomotion.SimpleEnv"))
    if env_cls_path == "humanoid_tennis.envs.locomotion.SimpleEnv":
        env_cls = SimpleEnv
    else:
        env_cls = hydra.utils.get_class(env_cls_path)
    base_env = env_cls(cfg.task)
    aa.print("SimpleEnv done")

    if cfg.checkpoint_path is not None and aa.is_main_process():
        checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path, cfg.get("wandb", None))
        checkpoint_path = _resolve_local_checkpoint_path(checkpoint_path)
        aa.print(f"Loading checkpoint from {checkpoint_path}")
        _install_legacy_checkpoint_aliases()
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}

    teacher_checkpoint_path = cfg.algo.get("teacher_checkpoint_path", None)
    if teacher_checkpoint_path is not None and aa.is_main_process():
        teacher_checkpoint_path = parse_checkpoint_path(teacher_checkpoint_path, cfg.get("wandb", None))
        teacher_checkpoint_path = _resolve_local_checkpoint_path(teacher_checkpoint_path)
        aa.print(f"Loading teacher checkpoint from {teacher_checkpoint_path}")
        _install_legacy_checkpoint_aliases()
        teacher_state_dict = torch.load(teacher_checkpoint_path, weights_only=False)
    else:
        teacher_state_dict = {}

    if aa.is_distributed():
        state_list = [state_dict, teacher_state_dict]
        dist.broadcast_object_list(state_list, src=0)
        state_dict = state_list[0] or {}
        teacher_state_dict = state_list[1] or {}
    aa.print("load checkpoint done")
    
    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])

    for obs_group_key in list(cfg.task.observation.keys()):
        if (
            obs_group_key not in policy_in_keys
            and not obs_group_key.endswith("_")
        ):
            print(colored(f"[Warn] Obs group '{obs_group_key}' not used by policy in_keys; keeping config unchanged.", "yellow"))
    
    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True) 
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    symmetry_enabled = bool(cfg.algo.get("symmetry_enabled", True))
    use_symmetry_vecnorm = symmetry_enabled

    if use_symmetry_vecnorm:
        vecnorm_symmetry = {}
        for key in obs_keys:
            obs_key = key[-1] if isinstance(key, tuple) else key
            if obs_key not in base_env.observation_funcs:
                continue
            try:
                vecnorm_symmetry[key] = base_env.observation_funcs[obs_key].symmetry_transforms().to(base_env.device)
            except NotImplementedError:
                print(colored(f"[Warn]: Obs group '{obs_key}' has no symmetry_transforms(); VecNorm keeps default update.", "yellow"))
        if len(vecnorm_symmetry):
            print(colored(f"[Info]: enable symmetric VecNorm for keys: {sorted(map(str, vecnorm_symmetry.keys()))}", "green"))
        vecnorm = SymmetricVecNorm(
            obs_keys,
            decay=0.9999,
            symmetry_transforms=vecnorm_symmetry,
        )
    else:
        print(colored("[Info]: symmetry VecNorm disabled by cfg.", "yellow"))
        vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

    if "vecnorm" in state_dict.keys():
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
        ok, msg = _load_vecnorm_state(vecnorm, state_dict["vecnorm"])
        if ok:
            print(colored(f"[Info]: {msg}", "green"))
        else:
            print(colored(f"[Warn]: {msg}", "yellow"))
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating obervation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating obervation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())
    elif cfg.vecnorm is not None:
        raise ValueError
    aa.print("create VecNorm done")

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    aa.print("TransformedEnv done")
    
    # setup policy
    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    humanoid_tennis.print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy = policy_cls(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device,
        env=env
    )
    aa.print("policy done")
    
    if "policy" in state_dict.keys():
        policy_state = state_dict["policy"]
        ckpt_phase = policy_state.get("last_phase", None) if isinstance(policy_state, dict) else None
        is_highlevel = str(getattr(cfg.algo, "phase", "")).lower() == "highlevel"
        rollout_mode = str(cfg.get("rollout_mode", "")).lower()
        if (
            rollout_mode == "pulse_random"
            and hasattr(policy, "load_pulse_modules_state_dict")
        ):
            print(
                colored(
                    "[Info]: pulse_random rollout detected. "
                    "Load pulse modules only (prior/decoder).",
                    "green",
                )
            )
            policy.load_pulse_modules_state_dict(policy_state)
        elif (
            is_highlevel
            and str(ckpt_phase).lower() == "pulse"
            and hasattr(policy, "load_highlevel_from_pulse_state_dict")
        ):
            print(
                colored(
                    "[Info]: High-level phase + pulse checkpoint detected. "
                    "Load pulse modules only (prior/decoder).",
                    "green",
                )
            )
            policy.load_highlevel_from_pulse_state_dict(policy_state)
        else:
            print(colored("[Info]: Load policy from checkpoint.", "green"))
            policy.load_state_dict(policy_state)
    if "policy" in teacher_state_dict.keys():
        print(colored("[Info]: Load teacher-only modules from teacher checkpoint.", "green"))
        policy.load_teacher_state_dict(teacher_state_dict["policy"])
    if hasattr(policy, "prepare_for_phase"):
        policy.prepare_for_phase()
    
    if cfg.checkpoint_path is not None:
        policy.broadcast_parameters([vecnorm])

    primer = policy.make_tensordict_primer()

    if primer is not None:
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
    env = TransformedEnv(env.base_env, transform)

    if return_checkpoint_state:
        return env, policy, vecnorm, primer, state_dict
    return env, policy, vecnorm, primer


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm

@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    render=False,
    keys=[("next", "stats")],
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    """
    keys = set(keys)
    keys.add(("next", "done"))

    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    trajs = []
    frames = []

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)
            tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            trajs.append(tensordict.select(*keys, strict=False).cpu())
            if render:
                frames.append(env.render("rgb_array"))
    inference_time = np.mean(inference_time[5:])
    print(f"Average inference time: {inference_time:.4f} s")

    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    compute_std_for = ["return", "survival"]
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        if k in compute_std_for:
            info[key + "_std"] = torch.std(v.float()).item()

    # log video
    if len(frames):
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        video_path = os.path.join(os.path.dirname(__file__), "..", "videos/" f"recording-{time_str}.mp4")
        fps = int(1 / env.step_dt)
        try:
            imageio.mimwrite(video_path, frames, fps=fps)
        except Exception:
            # Fallback: ensure frames are numpy arrays
            video_array = np.stack(frames)
            imageio.mimwrite(video_path, list(video_array), fps=fps)

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats
