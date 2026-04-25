import argparse
import os
import sys
import subprocess
from pathlib import Path

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
    parser.add_argument("--robot-name", default=None, type=str)
    parser.add_argument("--step-dt", default=None, type=float, help="Override simulation step_dt (e.g. 0.02)")
    parser.add_argument("--physics-dt", default=None, type=float, help="Override physics_dt")
    parser.add_argument("--viewer-max-fps", default=30.0, type=float, help="Viser max update FPS. <=0 means no throttle.")
    parser.add_argument("--viewer-debug", action="store_true", help="Enable heavy debug visualization overlays.")
    parser.add_argument("--playback-fps", default=0.0, type=float, help="Delay playback to fixed FPS for smoother viewing. <=0 disables pacing.")
    parser.add_argument("--offline-record", default="", type=str, help="Save rollout to npz (motion format) for offline replay.")
    parser.add_argument("--offline-steps", default=2000, type=int, help="Number of env steps to record when --offline-record is set.")
    parser.add_argument("--offline-env-id", default=0, type=int, help="Env id to record in offline mode.")
    parser.add_argument("--offline-headless", action="store_true", help="Force headless during recording for max speed.")
    parser.add_argument("--offline-replay", action="store_true", help="Auto-launch replay_motion_npz.py after recording.")
    parser.add_argument("--offline-replay-speed", default=1.0, type=float, help="Replay speed multiplier.")
    parser.add_argument("--offline-replay-viewer-max-fps", default=45.0, type=float, help="Viewer max FPS for offline replay.")
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
    cfg["task"]["viewer"]["debug_visualization_enabled"] = bool(args.viewer_debug)
    cfg["task"]["viewer"]["camera_tracking_enabled"] = False
    cfg["task"]["viewer"]["max_fps"] = float(args.viewer_max_fps)
    cfg["task"]["command"]["show_tracking_debug"] = False
    cfg["task"]["command"]["show_reference_motion"] = False
    command_target = str(cfg["task"]["command"].get("_target_", ""))
    if "motion_tracking" in command_target:
        cfg["task"]["command"]["init_from_default_pose"] = True
    cfg["task"]["termination"] = {}
    cfg["task"]["command"]["body_z_terminate_thres"] = 0.0
    cfg["task"]["command"]["gravity_terminate_thres"] = 0.0
    cfg["task"]["max_episode_length"] = int(1e9)
    cfg["task"]["sim"]["device"] = device
    if args.step_dt is not None:
        cfg["task"]["sim"]["step_dt"] = args.step_dt
    if args.physics_dt is not None:
        cfg["task"]["sim"]["mujoco_physics_dt"] = args.physics_dt
        cfg["task"]["sim"]["isaac_physics_dt"] = args.physics_dt
    if args.robot_name:
        cfg["task"]["robot"]["name"] = args.robot_name

    if "algo" not in cfg:
        cfg["algo"] = {}
    cfg["algo"]["pulse_prior_temp"] = float(args.temp)
    cfg["rollout_target_fps"] = float(args.playback_fps)

    offline_record = str(args.offline_record).strip()
    offline_mode = len(offline_record) > 0
    if offline_mode:
        if args.offline_steps <= 0:
            raise ValueError("--offline-steps must be > 0 when --offline-record is set.")
        cfg["rollout_record_path"] = offline_record
        cfg["rollout_record_env_id"] = int(args.offline_env_id)
        cfg["rollout_max_steps"] = int(args.offline_steps)
        if args.offline_headless:
            cfg["headless"] = True
            cfg["app"]["headless"] = True
            cfg["task"]["viewer"]["headless"] = True
        # Don't artificially pace during offline recording.
        cfg["rollout_target_fps"] = 0.0

    print(
        f"[INFO] device={device}, num_envs={args.num_envs}, pulse_prior_temp={args.temp}, "
        f"viewer_max_fps={float(args.viewer_max_fps):.1f}, viewer_debug={bool(args.viewer_debug)}, "
        f"playback_fps={float(args.playback_fps):.1f}"
    )
    if offline_mode:
        print(
            f"[INFO] offline_record={offline_record}, offline_steps={int(args.offline_steps)}, "
            f"offline_env_id={int(args.offline_env_id)}, offline_headless={bool(args.offline_headless)}"
        )
    if args.robot_name:
        print(f"[INFO] robot_name={args.robot_name}")
    print("[INFO] Rollout mode: pulse_random")
    if "motion_tracking" in command_target:
        print("[INFO] Init mode: default cfg pose for all envs (no dataset/random init pose)")
    print("[INFO] Press Ctrl+C to stop.")
    play(cfg)

    if offline_mode and args.offline_replay:
        record_path = Path(offline_record).expanduser()
        if not record_path.is_absolute():
            record_path = (Path.cwd() / record_path).resolve()
        replay_cmd = [
            sys.executable,
            "scripts/data_process/replay_motion_npz.py",
            str(record_path),
            "--robot",
            str(cfg["task"]["robot"]["name"]),
            "--device",
            str(device),
            "--physics-dt",
            str(args.physics_dt if args.physics_dt is not None else cfg["task"]["sim"]["mujoco_physics_dt"]),
            "--speed",
            str(float(args.offline_replay_speed)),
            "--viewer-max-fps",
            str(float(args.offline_replay_viewer_max_fps)),
            "--loop",
        ]
        print("[INFO] Launch offline replay:", " ".join(replay_cmd))
        subprocess.run(replay_cmd, check=True)


if __name__ == "__main__":
    main()
