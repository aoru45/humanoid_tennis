import argparse
import os
import sys

import torch
import hydra
from omegaconf import OmegaConf

# Add project root to path for `scripts.*` imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.play import play


def load_cfg(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    if "task" in cfg:
        return cfg

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    config_name = os.path.splitext(os.path.basename(cfg_path))[0]
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=[
                "task=G1/G1_tracking",
                "+exp=pulse",
            ],
        )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Random pulse prior rollout with Viser viewer.")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-envs", default=16, type=int)
    parser.add_argument("--temp", default=1.0, type=float)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    OmegaConf.set_struct(cfg, False)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    cfg["checkpoint_path"] = args.checkpoint
    cfg["vecnorm"] = "eval"
    cfg["export_policy"] = False
    cfg["perf_test"] = False
    cfg["rollout_mode"] = "pulse_random"

    cfg["headless"] = False
    cfg["app"]["headless"] = False
    cfg["task"]["num_envs"] = int(args.num_envs)
    cfg["task"]["viewer"]["headless"] = False
    cfg["task"]["viewer"]["debug_visualization_enabled"] = True
    cfg["task"]["viewer"]["camera_tracking_enabled"] = False
    cfg["task"]["command"]["show_tracking_debug"] = False
    cfg["task"]["command"]["show_reference_motion"] = False
    cfg["task"]["termination"] = {}
    cfg["task"]["command"]["body_z_terminate_thres"] = 0.0
    cfg["task"]["command"]["gravity_terminate_thres"] = 0.0
    cfg["task"]["max_episode_length"] = int(1e9)
    cfg["task"]["sim"]["device"] = device

    if "algo" not in cfg:
        cfg["algo"] = {}
    cfg["algo"]["pulse_prior_temp"] = float(args.temp)

    print(f"[INFO] device={device}, num_envs={args.num_envs}, pulse_prior_temp={args.temp}")
    print("[INFO] Rollout mode: pulse_random")
    print("[INFO] Press Ctrl+C to stop.")
    play(cfg)


if __name__ == "__main__":
    main()
