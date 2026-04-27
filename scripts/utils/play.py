import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
import time
from pathlib import Path
from omegaconf import OmegaConf


from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential

from humanoid_tennis.learning import ALGOS
from humanoid_tennis.utils.export import export_onnx
from scripts.utils.helpers import EpisodeStats, make_env_policy, ObsNorm

def play(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    

    env, policy, vecnorm, _ = make_env_policy(cfg)
    if hasattr(policy, "step_schedule"):
        policy.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)

    if cfg.export_policy:
        import copy
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        _policy = TensorDictSequential(obs_norm, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        os.makedirs(os.path.join(FILE_PATH, "..", "exports", f"{cfg.task.name}-{time_str}"), exist_ok=True)
        path = os.path.join(FILE_PATH, "..", "exports", f"{cfg.task.name}-{time_str}", "policy.pt")
        torch.save(_policy, path)

        meta = {}
        meta["action_scaling"] = dict(cfg.task.action.get("action_scaling"))
        # meta["stiffness"] = dict(cfg.task.robot.stiffness)
        # meta["damping"] = dict(cfg.task.robot.damping)
        # meta["effort_limit"] = dict(cfg.task.robot.effort_limit)
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_mode = cfg.get("rollout_mode", "eval")
    rollout_max_steps = int(cfg.get("rollout_max_steps", -1))
    rollout_record_path = str(cfg.get("rollout_record_path", "")).strip()
    rollout_record_env_id = int(cfg.get("rollout_record_env_id", 0))
    rollout_target_fps = float(cfg.get("rollout_target_fps", 0.0))
    rollout_period_s = (1.0 / rollout_target_fps) if rollout_target_fps > 0.0 else 0.0
    next_tick_s = None
    policy = policy.get_rollout_policy(rollout_mode)
    record_qpos = []
    record_qvel = []
    record_enabled = len(rollout_record_path) > 0

    td_ = env.reset()
    
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for i in itertools.count():
            td_ = policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())

            if rollout_period_s > 0.0:
                now_s = time.perf_counter()
                if next_tick_s is None:
                    next_tick_s = now_s + rollout_period_s
                sleep_s = next_tick_s - now_s
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                    next_tick_s += rollout_period_s
                else:
                    # If simulation is slower than target playback, resync to avoid drift.
                    next_tick_s = now_s

            if record_enabled:
                base_env = getattr(env, "base_env", env)
                num_envs = int(base_env.num_envs)
                env_id = min(max(int(rollout_record_env_id), 0), max(0, num_envs - 1))
                qpos_t = base_env.sim.data.qpos[env_id].detach().cpu().float().numpy().copy()
                qvel_t = base_env.sim.data.qvel[env_id].detach().cpu().float().numpy().copy()
                record_qpos.append(qpos_t)
                record_qvel.append(qvel_t)

            if rollout_max_steps > 0 and (i + 1) >= rollout_max_steps:
                break

    if record_enabled:
        if len(record_qpos) == 0:
            print("[WARN] rollout_record_path is set but no frames were recorded.")
        else:
            base_env = getattr(env, "base_env", env)
            qpos = np.asarray(record_qpos, dtype=np.float32)
            qvel = np.asarray(record_qvel, dtype=np.float32)
            if qpos.shape[-1] < 7 or qvel.shape[-1] < 6:
                print(
                    "[WARN] Recorded qpos/qvel dims are too small for motion format export: "
                    f"qpos={qpos.shape}, qvel={qvel.shape}"
                )
            else:
                root_pos = qpos[:, :3]
                root_rot_wxyz = qpos[:, 3:7]
                root_rot_xyzw = np.concatenate([root_rot_wxyz[:, 1:], root_rot_wxyz[:, :1]], axis=-1)
                dof_pos = qpos[:, 7:]
                dof_vel = qvel[:, 6:]
                step_dt = float(base_env.step_dt)
                physics_dt = float(base_env.physics_dt)
                fps = 1.0 / max(step_dt, 1.0e-8)

                save_path = Path(rollout_record_path).expanduser()
                if not save_path.is_absolute():
                    save_path = (Path.cwd() / save_path).resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(save_path),
                    root_pos=root_pos.astype(np.float32),
                    root_rot=root_rot_xyzw.astype(np.float32),
                    dof_pos=dof_pos.astype(np.float32),
                    dof_vel=dof_vel.astype(np.float32),
                    qpos=qpos.astype(np.float32),
                    qvel=qvel.astype(np.float32),
                    fps=np.array([fps], dtype=np.float32),
                    step_dt=np.array([step_dt], dtype=np.float32),
                    physics_dt=np.array([physics_dt], dtype=np.float32),
                )
                print(
                    f"[INFO] Saved offline rollout: {save_path} | frames={qpos.shape[0]}, "
                    f"fps={fps:.3f}, step_dt={step_dt:.4f}, physics_dt={physics_dt:.6f}"
                )
    
    env.close()
