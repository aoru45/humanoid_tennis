import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _unwrap_base_env(env):
    base_env = env
    for _ in range(16):
        if hasattr(base_env, "sim") and hasattr(base_env, "scene"):
            return base_env
        if not hasattr(base_env, "base_env"):
            break
        base_env = base_env.base_env
    raise RuntimeError("Failed to unwrap base env from transformed env stack.")


@dataclass
class _RecordSession:
    id: int
    iter_start: int
    env_ids_cpu: torch.Tensor
    env_ids_device: torch.Tensor
    qpos: np.ndarray
    qvel: np.ndarray
    root_state: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    iter_idx: np.ndarray
    rollout_step: np.ndarray
    cursor: int = 0


class TrainStateRecorder:
    def __init__(
        self,
        env,
        *,
        interval: int,
        num_envs: int,
        num_steps: int,
        output_dir: str,
        seed: int = 0,
        start_iter: int = 0,
        enabled: bool = True,
    ):
        self.enabled = bool(enabled) and interval > 0 and num_envs > 0 and num_steps > 0
        self.interval = int(interval)
        self.num_envs = int(num_envs)
        self.num_steps = int(num_steps)
        self.output_dir = output_dir
        self.start_iter = int(start_iter)

        self.base_env = _unwrap_base_env(env)
        self.asset = self.base_env.scene["robot"]
        self._rng = np.random.default_rng(int(seed))
        self._session: Optional[_RecordSession] = None
        self._session_id = 0
        self.last_dump_path: Optional[str] = None

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(
                "TrainStateRecorder enabled: interval=%d, envs=%d, steps=%d, output_dir=%s",
                self.interval,
                self.num_envs,
                self.num_steps,
                self.output_dir,
            )
        else:
            logging.info("TrainStateRecorder disabled.")

    def maybe_start(self, iter_idx: int):
        if not self.enabled or self._session is not None:
            return

        global_iter = self.start_iter + int(iter_idx)
        if global_iter % self.interval != 0:
            return

        env_count = min(self.num_envs, self.base_env.num_envs)
        env_ids = self._rng.choice(self.base_env.num_envs, size=env_count, replace=False)
        env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        env_ids_device = env_ids_cpu.to(self.base_env.device)

        qpos = self.base_env.sim.data.qpos[env_ids_device]
        qvel = self.base_env.sim.data.qvel[env_ids_device]
        root_state = self._get_root_state()[env_ids_device]
        joint_pos = self.asset.data.joint_pos[env_ids_device]
        joint_vel = self.asset.data.joint_vel[env_ids_device]

        session = _RecordSession(
            id=self._session_id,
            iter_start=global_iter,
            env_ids_cpu=env_ids_cpu,
            env_ids_device=env_ids_device,
            qpos=np.empty((self.num_steps, env_count, qpos.shape[-1]), dtype=np.float32),
            qvel=np.empty((self.num_steps, env_count, qvel.shape[-1]), dtype=np.float32),
            root_state=np.empty((self.num_steps, env_count, root_state.shape[-1]), dtype=np.float32),
            joint_pos=np.empty((self.num_steps, env_count, joint_pos.shape[-1]), dtype=np.float32),
            joint_vel=np.empty((self.num_steps, env_count, joint_vel.shape[-1]), dtype=np.float32),
            iter_idx=np.empty((self.num_steps,), dtype=np.int64),
            rollout_step=np.empty((self.num_steps,), dtype=np.int64),
        )
        self._session_id += 1
        self._session = session
        logging.info(
            "TrainStateRecorder started session %d at iter=%d with env_ids=%s",
            session.id,
            global_iter,
            env_ids_cpu.tolist(),
        )

    def on_step(self, iter_idx: int, rollout_step: int):
        if not self.enabled or self._session is None:
            return None

        s = self._session
        if s.cursor >= self.num_steps:
            return None

        env_ids = s.env_ids_device
        s.qpos[s.cursor] = self.base_env.sim.data.qpos[env_ids].detach().cpu().float().numpy()
        s.qvel[s.cursor] = self.base_env.sim.data.qvel[env_ids].detach().cpu().float().numpy()
        s.root_state[s.cursor] = self._get_root_state()[env_ids].detach().cpu().float().numpy()
        s.joint_pos[s.cursor] = self.asset.data.joint_pos[env_ids].detach().cpu().float().numpy()
        s.joint_vel[s.cursor] = self.asset.data.joint_vel[env_ids].detach().cpu().float().numpy()
        s.iter_idx[s.cursor] = self.start_iter + int(iter_idx)
        s.rollout_step[s.cursor] = int(rollout_step)
        s.cursor += 1

        if s.cursor >= self.num_steps:
            return self._save_and_clear(partial=False)
        return None

    def flush(self):
        if not self.enabled or self._session is None:
            return None
        if self._session.cursor == 0:
            self._session = None
            return None
        return self._save_and_clear(partial=True)

    def _get_root_state(self):
        return torch.cat(
            [
                self.asset.data.root_link_pos_w,
                self.asset.data.root_link_quat_w,
                self.asset.data.root_link_lin_vel_w,
                self.asset.data.root_link_ang_vel_w,
            ],
            dim=-1,
        )

    def _save_and_clear(self, partial: bool):
        s = self._session
        if s is None:
            return None
        count = s.cursor
        suffix = "partial" if partial else "full"
        filename = (
            f"iter_{s.iter_start:07d}_session_{s.id:04d}"
            f"_envs_{s.env_ids_cpu.numel()}_steps_{count:04d}_{suffix}.npz"
        )
        save_path = os.path.join(self.output_dir, filename)

        payload = {
            "qpos": s.qpos[:count],
            "qvel": s.qvel[:count],
            "root_state": s.root_state[:count],
            "joint_pos": s.joint_pos[:count],
            "joint_vel": s.joint_vel[:count],
            "iter_idx": s.iter_idx[:count],
            "rollout_step": s.rollout_step[:count],
            "env_ids": s.env_ids_cpu.numpy(),
            "joint_names": np.asarray(self.asset.joint_names),
            "body_names": np.asarray(self.asset.body_names),
            "robot_name": np.asarray([self.base_env.cfg.robot.name]),
            "step_dt": np.asarray([float(self.base_env.step_dt)], dtype=np.float32),
            "physics_dt": np.asarray([float(self.base_env.physics_dt)], dtype=np.float32),
            "decimation": np.asarray([int(self.base_env.decimation)], dtype=np.int32),
            "env_spacing": np.asarray([float(getattr(self.base_env.scene, "env_spacing", 2.5))], dtype=np.float32),
        }
        np.savez_compressed(save_path, **payload)

        self.last_dump_path = save_path
        self._session = None
        logging.info("TrainStateRecorder saved: %s", save_path)
        return save_path
