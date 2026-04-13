import torch
import os
import sys
import hydra
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf
from scripts.utils.play import play
from scripts.utils.eval import eval
from active_adaptation.utils.wandb import (
    load_run_cfg_and_checkpoint,
    load_wandb_cfg_from_yaml,
)

FILE_PATH = os.path.dirname(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    parser.add_argument("-o", "--teleop", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("-s", "--success", action="store_true", default=False)  # test success rate
    args = parser.parse_args()

    wandb_cfg = load_wandb_cfg_from_yaml(os.path.join(FILE_PATH, "..", "cfg", "eval.yaml"))
    cfg, download = load_run_cfg_and_checkpoint(
        args.run_path,
        wandb_cfg=wandb_cfg,
        root_dir=os.path.join(os.path.dirname(__file__), "wandb"),
        iteration=args.iterations,
    )

    print(f"Loading run {download.run_name}")
    print(f"Downloading {download.checkpoint_name}")
    OmegaConf.set_struct(cfg, False)

    cfg["checkpoint_path"] = download.checkpoint_path
    cfg["vecnorm"] = "eval"

    if args.teleop:
        cfg["task"]["command"]["teleop"] = True

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        # cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        if args.terrain:
            cfg["task"]["terrain"] = _cfg.task.terrain
        if args.command:
            cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags
    
    if args.play:
        if not args.success:
            cfg["app"]["headless"] = False
            cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        play(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 16
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            cfg["app"]["headless"] = False
        eval(cfg)
    exit(0)

if __name__ == "__main__":
    main()
