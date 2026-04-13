import torch
# import warp
import hydra
import numpy as np

import einops
import logging
import os
import sys
import time
import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
import torch.distributed as dist

import active_adaptation as aa
from active_adaptation.utils.torchrl import SyncDataCollector, TDTimeBuffer

# local import
from scripts.utils.helpers import make_env_policy, EpisodeStats, evaluate
from scripts.utils.train_record import TrainStateRecorder
from torchrl.envs.utils import set_exploration_type, ExplorationType
from active_adaptation.utils.wandb import finish_wandb_run, init_wandb_run

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    if aa.is_distributed():
        dist.init_process_group(
            backend="nccl",
            world_size=aa.get_world_size(),
            rank=aa.get_local_rank(),
        )
        cfg.seed = cfg.seed + aa.get_local_rank() * 10000

    env, policy, vecnorm, primer = make_env_policy(cfg)

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)
    start_iter = cfg.get("start_iter", 0)
    start_frame = start_iter * frames_per_batch

    need_logging = aa.is_main_process() and cfg.wandb.get("mode", "disabled") != "disabled"
    if need_logging:
        default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        run = init_wandb_run(cfg.wandb, config=cfg, name=default_run_name)
        if run.id:
            run.name = f"{run.id}-{default_run_name}"
        else:
            run.name = default_run_name
        setproctitle(run.name)

        cfg_save_path = os.path.join(run.dir, "cfg.yaml")
        OmegaConf.save(cfg, cfg_save_path)
        run.save(cfg_save_path, policy="now")
        run.save(os.path.join(run.dir, "config.yaml"), policy="now")

        import inspect
        import shutil
        source_path = inspect.getfile(policy.__class__)
        target_path = os.path.join(run.dir, source_path.split("/")[-1])
        shutil.copy(source_path, target_path)
        run.save(target_path, policy="now")

        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        logging.info(f"Log interval: {log_interval} steps")

        stats_keys = [
            k for k in env.reward_spec.keys(True, True) 
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(stats_keys, device=env.device)

        def save(policy, checkpoint_name: str):
            ckpt_path = os.path.join(run.dir, f"{checkpoint_name}.pt")
            state_dict = OrderedDict()
            state_dict["wandb"] = {"name": run.name, "id": run.id}
            state_dict["policy"] = policy.state_dict()
            state_dict["env"] = env.state_dict()
            state_dict["cfg"] = cfg
            if "vecnorm" in locals():
                state_dict["vecnorm"] = vecnorm.state_dict()
            torch.save(state_dict, ckpt_path)
            run.save(ckpt_path, policy="now", base_path=run.dir)
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")

        def should_save(i):
            if not aa.is_main_process():
                return False
            return i > 0 and i % save_interval == 0
    
    rollout_policy = policy.get_rollout_policy("train")
    env_frames = 0
    carry = env.reset()

    assert env.training
    if aa.is_main_process():
        progress = tqdm(range(total_iters))
    else:
        progress = range(total_iters)

    N = env.num_envs
    T = cfg.algo.train_every
    data_buf = TDTimeBuffer(N, T, device=policy.device)

    train_recorder = None
    if aa.is_main_process():
        train_record_cfg = cfg.get("train_record", {})
        record_output_dir = train_record_cfg.get("output_dir", "train_records")
        if not os.path.isabs(record_output_dir):
            record_output_dir = os.path.join(os.getcwd(), record_output_dir)
        train_recorder = TrainStateRecorder(
            env,
            interval=int(train_record_cfg.get("interval", 0)),
            num_envs=int(train_record_cfg.get("envs", 16)),
            num_steps=int(train_record_cfg.get("steps", 256)),
            output_dir=record_output_dir,
            seed=int(cfg.seed),
            start_iter=int(start_iter),
            enabled=True,
        )

    for i in progress:
        start = time.perf_counter()
        if train_recorder is not None:
            train_recorder.maybe_start(i)

        with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
            torch.compiler.cudagraph_mark_step_begin() # for compiled policy
            for t in range(cfg.algo.train_every):
                carry = rollout_policy(carry)

                td, carry = env.step_and_maybe_reset(carry)
                if train_recorder is not None:
                    rollout_step = (start_iter + i) * cfg.algo.train_every + t
                    train_recorder.on_step(i, rollout_step)

                # deal with value
                policy.critic(td)
                policy.critic(td["next"])
                td.get(("next", "state_value"))[:] = torch.where(
                    td["next", "done"], 
                    td["state_value"],
                    td["next", "state_value"]
                )

                # clean up tensordict
                td["next"] = td["next"].exclude(*rollout_policy.in_keys)
                private_keys = [key for key in td.keys(True, True) if isinstance(key, str) and key.startswith('_')]
                td = td.exclude(*private_keys, "priv_pred", "priv_feature")

                data_buf.write_step(t, td)

            data = data_buf.td

        rollout_time = time.perf_counter() - start
        training_start = time.perf_counter()

        if hasattr(policy, "step_schedule"):
            policy.step_schedule(i / total_iters, i)
        if hasattr(env, "step_schedule"):
            env.step_schedule(i / total_iters, i)
        
        train_carry = policy.train_op(data, vecnorm)

        if need_logging:
            info = {}
            env_frames += data.numel()
            episode_stats.add(data)

            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            
            info.update(train_carry)
            info.update(env.extra)
            info.update(env.stats_ema)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["rollout_fps"] = data.numel() / rollout_time * aa.get_world_size()
            info["training_time"] = time.perf_counter() - training_start
        
            if save_interval and save_interval > 0 and should_save(i):
                save(policy, f"checkpoint_{i}")

            run.log(info, step=i)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    if aa.is_main_process():
        if train_recorder is not None:
            train_recorder.flush()

        if need_logging:
            save(policy, "checkpoint_final")

            policy_eval = policy.get_rollout_policy("eval")
            info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
            run.log(info, step = total_iters)

            finish_wandb_run(run)
    exit(0)


if __name__ == "__main__":
    main()
