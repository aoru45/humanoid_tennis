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
    parser.add_argument("--robot", default=None, type=str, help="Override robot name.")
    parser.add_argument("--dataset", default="run_tennis_subset", type=str, help="Motion dataset mem_path to visualize.")
    parser.add_argument("--fix-ds", default=0, type=int, help="Fixed dataset index for deterministic visualization.")
    parser.add_argument("--reference-alpha", default=0.65, type=float, help="Reference ghost transparency.")
    parser.add_argument("--no-reference", action="store_true", help="Disable reference motion ghost.")
    parser.add_argument("--show-tracking-debug", action="store_true", help="Show tracking debug overlays.")
    parser.add_argument("--viewer-max-fps", default=45.0, type=float, help="Maximum Viser update FPS.")
    parser.add_argument("--target-fps", default=50.0, type=float, help="Wall-clock rollout playback FPS. <=0 runs as fast as possible.")
    parser.add_argument("--with-tennis-scene", action="store_true", help="Keep tennis court/ball entities if present in cfg.")
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
    cfg["rollout_target_fps"] = float(args.target_fps)
    cfg["headless"] = False
    cfg["app"]["headless"] = False
    cfg["task"]["num_envs"] = int(args.num_envs)
    if args.robot is not None:
        cfg["task"]["robot"]["name"] = args.robot
    cfg["task"]["viewer"]["headless"] = False
    cfg["task"]["viewer"]["debug_visualization_enabled"] = True
    cfg["task"]["viewer"]["camera_tracking_enabled"] = False
    cfg["task"]["viewer"]["max_fps"] = float(args.viewer_max_fps)
    if not args.with_tennis_scene and "tennis" in cfg["task"]:
        cfg["task"].pop("tennis")
    cfg["task"]["command"]["dataset"]["mem_paths"] = [args.dataset]
    cfg["task"]["command"]["dataset"]["path_weights"] = [1.0]
    cfg["task"]["command"]["dataset"]["fix_ds"] = int(args.fix_ds)
    cfg["task"]["command"]["show_tracking_debug"] = bool(args.show_tracking_debug)
    cfg["task"]["command"]["show_reference_motion"] = not bool(args.no_reference)
    cfg["task"]["command"]["reference_alpha"] = float(args.reference_alpha)
    cfg["task"]["sim"]["device"] = device

    print(f"[INFO] device={device}, num_envs={args.num_envs}")
    print(f"[INFO] robot={cfg['task']['robot']['name']}")
    print(f"[INFO] dataset={args.dataset}, fix_ds={args.fix_ds}")
    print(f"[INFO] reference_motion={not bool(args.no_reference)}, reference_alpha={args.reference_alpha}")
    print(f"[INFO] tennis_scene={bool(args.with_tennis_scene)}")
    print("[INFO] Press Ctrl+C to stop.")
    play(cfg)


if __name__ == "__main__":
    main()
