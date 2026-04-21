import argparse
import os
import sys
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, StepCounter, TransformedEnv

# Add project root to path for local imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.envs import SimpleEnv


def build_cfg(args):
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "cfg")
    cfg_dir = os.path.abspath(cfg_dir)
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                f"task={args.task}",
                f"+exp={args.exp}",
                "wandb.mode=disabled",
                "checkpoint_path=null",
                f"task.num_envs={args.num_envs}",
            ],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.headless = True
    cfg.app.headless = True
    cfg.task.viewer.headless = True
    cfg.task.viewer.debug_visualization_enabled = False
    cfg.task.command.debug_draw = False
    cfg.task.sim.device = args.device
    if "launch_bank_file" in cfg.task.command:
        cfg.task.command.launch_bank_file = ""
    if "require_launch_bank" in cfg.task.command:
        cfg.task.command.require_launch_bank = False

    def _set_range(name: str, value):
        if value is None:
            return
        lo, hi = float(value[0]), float(value[1])
        if lo >= hi:
            raise ValueError(f"Invalid range for {name}: [{lo}, {hi}]")
        cfg.task.command[name] = [lo, hi]

    _set_range("launcher_x_range", args.launcher_x_range)
    _set_range("launcher_y_range", args.launcher_y_range)
    _set_range("launcher_z_range", args.launcher_z_range)
    _set_range("strike_x_range", args.strike_x_range)
    _set_range("strike_y_range", args.strike_y_range)
    _set_range("strike_z_range", args.strike_z_range)
    _set_range("flight_t_range", args.flight_t_range)
    _set_range("launch_speed_range", args.launch_speed_range)
    _set_range("incoming_bounce_x_range", args.incoming_bounce_x_range)
    _set_range("incoming_bounce_y_range", args.incoming_bounce_y_range)
    _set_range("target_x_range", args.target_x_range)
    _set_range("target_y_range", args.target_y_range)
    return cfg


def build_env(cfg):
    base_env = SimpleEnv(cfg.task)
    transform = Compose(InitTracker(), StepCounter())
    env = TransformedEnv(base_env, transform)
    env.set_seed(int(cfg.seed))
    return env


def main():
    parser = argparse.ArgumentParser(description="Offline-generate tennis launch bank (.npz).")
    parser.add_argument("--task", type=str, default="G1/G1_tennis_highlevel")
    parser.add_argument("--exp", type=str, default="highlevel")
    parser.add_argument("--num-envs", type=int, default=64, help="Sampling parallelism in env construction.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512, help="How many launches to sample per call.")
    parser.add_argument("--print-every", type=int, default=1024)
    parser.add_argument("--launcher-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--launcher-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--launcher-z-range", type=float, nargs=2, default=None)
    parser.add_argument("--strike-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--strike-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--strike-z-range", type=float, nargs=2, default=None)
    parser.add_argument("--flight-t-range", type=float, nargs=2, default=None)
    parser.add_argument("--launch-speed-range", type=float, nargs=2, default=None)
    parser.add_argument("--incoming-bounce-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--incoming-bounce-y-range", type=float, nargs=2, default=None)
    parser.add_argument("--target-x-range", type=float, nargs=2, default=None)
    parser.add_argument("--target-y-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="data/tennis_launch_bank/highlevel_launch_bank.npz",
        help="Output .npz path. Relative paths are resolved from repo root.",
    )
    args = parser.parse_args()

    cfg = build_cfg(args)
    env = build_env(cfg)
    cmd = env.base_env.command_manager

    if not hasattr(cmd, "_sample_ball_launch"):
        raise RuntimeError("Command does not provide _sample_ball_launch; cannot generate launch bank.")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(repo_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    env_ids_all = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    total = int(args.num_samples)
    batch_size = max(1, int(args.batch_size))
    produced = 0

    pos_local_list: List[torch.Tensor] = []
    vel_list: List[torch.Tensor] = []
    ang_list: List[torch.Tensor] = []
    tgt_local_list: List[torch.Tensor] = []

    print(f"[INFO] Sampling launch bank: total={total}, batch={batch_size}, device={env.device}")

    try:
        while produced < total:
            n = min(batch_size, total - produced)
            repeats = max(1, (n + env.num_envs - 1) // env.num_envs)
            env_ids = env_ids_all.repeat(repeats)[:n]
            with torch.no_grad():
                pos_w, vel, ang, tgt_w = cmd._sample_ball_launch(env_ids)
            origins = env.base_env.scene.env_origins[env_ids]

            pos_local_list.append((pos_w - origins).detach().cpu())
            vel_list.append(vel.detach().cpu())
            ang_list.append(ang.detach().cpu())
            tgt_local_list.append((tgt_w - origins).detach().cpu())

            produced += n
            if args.print_every > 0 and (produced % int(args.print_every) == 0 or produced == total):
                print(f"[INFO] sampled {produced}/{total}")
    finally:
        try:
            env.close()
        except TypeError:
            env.base_env.close()

    pos_local = torch.cat(pos_local_list, dim=0).numpy().astype(np.float32, copy=False)
    vel = torch.cat(vel_list, dim=0).numpy().astype(np.float32, copy=False)
    ang = torch.cat(ang_list, dim=0).numpy().astype(np.float32, copy=False)
    tgt_local = torch.cat(tgt_local_list, dim=0).numpy().astype(np.float32, copy=False)

    np.savez_compressed(
        output_path,
        launch_pos_local=pos_local,
        launch_vel=vel,
        launch_ang=ang,
        target_bounce_local=tgt_local,
    )
    print(f"[INFO] Saved launch bank: {output_path}")
    print(f"[INFO] Shapes: pos={pos_local.shape}, vel={vel.shape}, ang={ang.shape}, target={tgt_local.shape}")


if __name__ == "__main__":
    main()
