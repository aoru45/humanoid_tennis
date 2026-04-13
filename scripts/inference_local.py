import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

# Add project root to path for `scripts.*` imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.play import play


def main():
    parser = argparse.ArgumentParser(description="Local inference with Viser viewer.")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-envs", default=16, type=int)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(cfg, False)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    cfg["checkpoint_path"] = args.checkpoint
    cfg["vecnorm"] = "eval"
    cfg["export_policy"] = False
    cfg["perf_test"] = False
    cfg["headless"] = False
    cfg["app"]["headless"] = False
    cfg["task"]["num_envs"] = int(args.num_envs)
    cfg["task"]["viewer"]["headless"] = False
    cfg["task"]["viewer"]["debug_visualization_enabled"] = True
    cfg["task"]["viewer"]["camera_tracking_enabled"] = False
    cfg["task"]["command"]["dataset"]["mem_paths"] = ["tennis"]
    cfg["task"]["command"]["dataset"]["path_weights"] = [1.0]
    cfg["task"]["command"]["dataset"]["fix_ds"] = 0
    cfg["task"]["command"]["show_tracking_debug"] = False
    cfg["task"]["command"]["show_reference_motion"] = True
    cfg["task"]["command"]["reference_alpha"] = 0.65
    cfg["task"]["sim"]["device"] = device

    print(f"[INFO] device={device}, num_envs={args.num_envs}")
    print("[INFO] dataset=tennis only")
    print("[INFO] Press Ctrl+C to stop.")
    play(cfg)


if __name__ == "__main__":
    main()
