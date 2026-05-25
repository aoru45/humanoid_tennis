from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


class LaunchBankBuffer:
    """Holds launch-bank tensors and serves cyclic/shuffled batches."""

    def __init__(self, device: str, shuffle: bool = True):
        self.device = device
        self.shuffle = bool(shuffle)
        self.level_names: list[str] = []
        self.level_pos_local: dict[str, torch.Tensor] = {}
        self.level_vel: dict[str, torch.Tensor] = {}
        self.level_ang: dict[str, torch.Tensor] = {}
        self.level_target_local: dict[str, torch.Tensor] = {}
        self.level_size: dict[str, int] = {}
        self.level_ptr: dict[str, int] = {}
        self.level_perm: dict[str, torch.Tensor | None] = {}
        self.sampling_probs: torch.Tensor | None = None
        self.last_sample_level_ids: torch.Tensor | None = None

        self.curriculum_enabled = False
        self.curriculum_start_probs: torch.Tensor | None = None
        self.curriculum_target_probs: torch.Tensor | None = None
        self.curriculum_progress_up = 0.04
        self.curriculum_progress_down = 0.0
        self.curriculum_ema_alpha = 0.05
        self.curriculum_min_level_prob = 0.05
        self.curriculum_progress = 0.0
        self.curriculum_success_ema = 0.5
        self.curriculum_last_batch_success = 0.0
        self.curriculum_update_count = 0

    def configure_shuffle(self, shuffle: bool) -> None:
        self.shuffle = bool(shuffle)
        for level in self.level_names:
            n = self.level_size[level]
            self.level_perm[level] = torch.randperm(n, device=self.device) if self.shuffle else None

    @staticmethod
    def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp_min(0.0)
        s = probs.sum().clamp_min(1.0e-6)
        return probs / s

    def _reset_levels(self) -> None:
        self.level_names = []
        self.level_pos_local = {}
        self.level_vel = {}
        self.level_ang = {}
        self.level_target_local = {}
        self.level_size = {}
        self.level_ptr = {}
        self.level_perm = {}
        self.sampling_probs = None
        self.last_sample_level_ids = None

    def _read_bank_arrays(self, launch_bank_file: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = os.path.expanduser(str(launch_bank_file))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Launch bank file not found: {path}")

        with np.load(path) as data:
            keys = set(data.keys())

            def _read(*cands: str) -> np.ndarray | None:
                for cand in cands:
                    if cand in keys:
                        return np.asarray(data[cand], dtype=np.float32)
                return None

            pos_local_np = _read("launch_pos_local", "local_pos")
            vel_np = _read("launch_vel", "vel")
            ang_np = _read("launch_ang", "ang")
            target_local_np = _read("target_bounce_local", "local_tgt")

            if pos_local_np is None or vel_np is None or ang_np is None or target_local_np is None:
                raise ValueError(
                    f"Invalid launch bank file {path}. "
                    "Expected keys: launch_pos_local, launch_vel, launch_ang, target_bounce_local."
                )
        if (
            pos_local_np.ndim != 2
            or vel_np.ndim != 2
            or ang_np.ndim != 2
            or target_local_np.ndim != 2
            or pos_local_np.shape[1] != 3
            or vel_np.shape[1] != 3
            or ang_np.shape[1] != 3
            or target_local_np.shape[1] != 3
        ):
            raise ValueError(
                f"Invalid launch bank tensor shapes from {path}: "
                f"pos={pos_local_np.shape}, vel={vel_np.shape}, ang={ang_np.shape}, target={target_local_np.shape}"
            )
        n = int(pos_local_np.shape[0])
        if n <= 0 or vel_np.shape[0] != n or ang_np.shape[0] != n or target_local_np.shape[0] != n:
                raise ValueError(
                    f"Inconsistent launch bank lengths from {path}: "
                    f"pos={pos_local_np.shape[0]}, vel={vel_np.shape[0]}, "
                    f"ang={ang_np.shape[0]}, target={target_local_np.shape[0]}"
                )

        return (
            torch.tensor(pos_local_np, dtype=torch.float32, device=self.device),
            torch.tensor(vel_np, dtype=torch.float32, device=self.device),
            torch.tensor(ang_np, dtype=torch.float32, device=self.device),
            torch.tensor(target_local_np, dtype=torch.float32, device=self.device),
        )

    def _add_level(self, level: str, launch_bank_file: str) -> None:
        pos_local, vel, ang, target_local = self._read_bank_arrays(launch_bank_file)
        n = int(pos_local.shape[0])
        self.level_names.append(level)
        self.level_pos_local[level] = pos_local
        self.level_vel[level] = vel
        self.level_ang[level] = ang
        self.level_target_local[level] = target_local
        self.level_size[level] = n
        self.level_ptr[level] = 0
        self.level_perm[level] = torch.randperm(n, device=self.device) if self.shuffle else None

    def load(self, launch_bank_file: str) -> None:
        self._reset_levels()
        self._add_level("default", launch_bank_file)
        self.sampling_probs = torch.ones((1,), dtype=torch.float32, device=self.device)
        self.curriculum_enabled = False

    def load_levels(
        self,
        *,
        easy_file: str | None = None,
        medium_file: str | None = None,
        hard_file: str | None = None,
    ) -> None:
        self._reset_levels()
        if easy_file:
            self._add_level("easy", easy_file)
        if medium_file:
            self._add_level("medium", medium_file)
        if hard_file:
            self._add_level("hard", hard_file)
        if len(self.level_names) == 0:
            raise ValueError("No launch bank files provided for multi-level loading.")
        base = torch.ones((len(self.level_names),), dtype=torch.float32, device=self.device)
        self.sampling_probs = self._normalize_probs(base)
        self.curriculum_enabled = False

    def set_sampling_probs(self, probs: torch.Tensor) -> None:
        if len(self.level_names) == 0:
            raise RuntimeError("Launch banks are not loaded.")
        if probs.numel() != len(self.level_names):
            raise ValueError(f"Expected {len(self.level_names)} probabilities, got {probs.numel()}.")
        self.sampling_probs = self._normalize_probs(probs.to(device=self.device, dtype=torch.float32))

    def enable_curriculum(
        self,
        *,
        start_probs: tuple[float, float, float],
        target_probs: tuple[float, float, float],
        progress_up: float,
        progress_down: float,
        ema_alpha: float,
        min_level_prob: float,
    ) -> None:
        # Curriculum semantics are defined for easy/medium/hard split only.
        expected = ["easy", "medium", "hard"]
        if self.level_names != expected:
            self.curriculum_enabled = False
            return
        start = torch.tensor(start_probs, device=self.device, dtype=torch.float32)
        target = torch.tensor(target_probs, device=self.device, dtype=torch.float32)
        self.curriculum_start_probs = self._normalize_probs(start)
        self.curriculum_target_probs = self._normalize_probs(target)
        self.curriculum_progress_up = max(0.0, float(progress_up))
        # Allow recovery: curriculum can move down when success drops.
        self.curriculum_progress_down = max(0.0, float(progress_down))
        self.curriculum_ema_alpha = min(max(float(ema_alpha), 0.0), 1.0)
        self.curriculum_min_level_prob = min(max(float(min_level_prob), 0.0), 0.30)
        self.curriculum_progress = 0.0
        self.curriculum_success_ema = 0.5
        self.curriculum_last_batch_success = 0.0
        self.curriculum_update_count = 0
        self.sampling_probs = self.curriculum_start_probs.clone()
        self.curriculum_enabled = True

    def _next_indices(self, level: str, num_samples: int) -> torch.Tensor:
        size = int(self.level_size[level])
        if size <= 0:
            raise RuntimeError("Launch bank is empty.")
        ids = torch.empty((num_samples,), dtype=torch.long, device=self.device)
        filled = 0
        while filled < num_samples:
            ptr = int(self.level_ptr[level])
            if ptr >= size:
                ptr = 0
                if self.shuffle:
                    self.level_perm[level] = torch.randperm(size, device=self.device)
                self.level_ptr[level] = 0
                ptr = 0
            take = min(size - ptr, num_samples - filled)
            src = slice(ptr, ptr + take)
            perm = self.level_perm[level]
            if perm is None:
                ids[filled : filled + take] = torch.arange(
                    ptr, ptr + take, dtype=torch.long, device=self.device
                )
            else:
                ids[filled : filled + take] = perm[src]
            self.level_ptr[level] = ptr + take
            filled += take
        return ids

    def get_last_sample_level_ids(self) -> torch.Tensor | None:
        if self.last_sample_level_ids is None:
            return None
        return self.last_sample_level_ids.clone()

    def get_curriculum_state(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.sampling_probs is None:
            return out
        for i, level in enumerate(self.level_names):
            out[f"prob_{level}"] = float(self.sampling_probs[i].item())
        out["progress"] = float(self.curriculum_progress)
        out["success_ema"] = float(self.curriculum_success_ema)
        out["success_batch"] = float(self.curriculum_last_batch_success)
        out["update_count"] = float(self.curriculum_update_count)
        return out

    def update_curriculum(self, *, level_ids: torch.Tensor, success: torch.Tensor) -> None:
        if not self.curriculum_enabled or self.sampling_probs is None:
            return
        if level_ids.numel() == 0 or success.numel() == 0:
            return
        valid = (level_ids >= 0) & (level_ids < len(self.level_names))
        if not valid.any():
            return
        succ = float(success[valid].float().mean().item())
        self.curriculum_last_batch_success = succ
        self.curriculum_update_count += 1
        alpha = self.curriculum_ema_alpha
        self.curriculum_success_ema = (1.0 - alpha) * self.curriculum_success_ema + alpha * float(succ)
        ema = min(max(float(self.curriculum_success_ema), 0.0), 1.0)

        # Bidirectional curriculum around a neutral success level.
        # Above center -> increase difficulty; below center -> decrease.
        center = 0.5
        if ema >= center:
            up = self.curriculum_progress_up * ((ema - center) / max(1.0 - center, 1.0e-6))
            delta = up
        else:
            down = self.curriculum_progress_down * ((center - ema) / max(center, 1.0e-6))
            delta = -down
        self.curriculum_progress += delta
        self.curriculum_progress = float(min(max(self.curriculum_progress, 0.0), 1.0))
        assert self.curriculum_start_probs is not None
        assert self.curriculum_target_probs is not None
        mix = (1.0 - self.curriculum_progress) * self.curriculum_start_probs + self.curriculum_progress * self.curriculum_target_probs
        floor = self.curriculum_min_level_prob
        mix = mix.clamp_min(floor)
        self.sampling_probs = self._normalize_probs(mix)

    def sample(
        self,
        *,
        env_origins: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.level_names) == 0:
            raise RuntimeError("Launch bank is empty. Please provide a valid offline launch bank file.")
        if self.sampling_probs is None:
            raise RuntimeError("Sampling probabilities are not initialized.")
        if len(self.level_names) == 1:
            level = self.level_names[0]
            bank_ids = self._next_indices(level, num_samples)
            self.last_sample_level_ids = torch.zeros((num_samples,), dtype=torch.long, device=self.device)
            return (
                self.level_pos_local[level][bank_ids] + env_origins,
                self.level_vel[level][bank_ids],
                self.level_ang[level][bank_ids],
                self.level_target_local[level][bank_ids] + env_origins,
            )

        probs = self._normalize_probs(self.sampling_probs)
        level_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        self.last_sample_level_ids = level_ids

        pos = torch.zeros((num_samples, 3), dtype=torch.float32, device=self.device)
        vel = torch.zeros((num_samples, 3), dtype=torch.float32, device=self.device)
        ang = torch.zeros((num_samples, 3), dtype=torch.float32, device=self.device)
        target = torch.zeros((num_samples, 3), dtype=torch.float32, device=self.device)
        for li, level in enumerate(self.level_names):
            mask = level_ids == li
            k = int(mask.sum().item())
            if k == 0:
                continue
            out_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            bank_ids = self._next_indices(level, k)
            pos[out_ids] = self.level_pos_local[level][bank_ids] + env_origins[out_ids]
            vel[out_ids] = self.level_vel[level][bank_ids]
            ang[out_ids] = self.level_ang[level][bank_ids]
            target[out_ids] = self.level_target_local[level][bank_ids] + env_origins[out_ids]
        return pos, vel, ang, target


@dataclass(frozen=True)
class ContactSensorHandles:
    racket_ball: Any = None
    ball_net: Any = None
    ball_court: Any = None
    racket_body: Any = None
    racket_velocity: Any = None

    @classmethod
    def from_scene(cls, scene) -> "ContactSensorHandles":
        sensors = getattr(scene, "sensors", {})
        return cls(
            racket_ball=scene["racket_ball_contact"] if "racket_ball_contact" in sensors else None,
            ball_net=scene["ball_net_contact"] if "ball_net_contact" in sensors else None,
            ball_court=scene["ball_court_contact"] if "ball_court_contact" in sensors else None,
            racket_body=scene["racket_body_contact"] if "racket_body_contact" in sensors else None,
            racket_velocity=(
                scene["robot/tennis_racket_center_global_linvel"]
                if "robot/tennis_racket_center_global_linvel" in sensors
                else None
            ),
        )
