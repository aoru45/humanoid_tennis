import argparse
import os
import sys
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to path for local imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from active_adaptation.utils.tennis_launch_generator import (
    LaunchPhysicsConfig,
    LaunchSamplerConfig,
    LaunchTrajectorySampler,
)
from active_adaptation.envs.mdp.commands.highlevel_tennis import HighLevelTennisConfig


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
            ],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _resolve_range(name: str, value, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    lo, hi = float(value[0]), float(value[1])
    if lo >= hi:
        raise ValueError(f"Invalid range for {name}: [{lo}, {hi}]")
    return (lo, hi)


def main():
    parser = argparse.ArgumentParser(description="Offline-generate tennis launch bank (.npz).")
    parser.add_argument("--task", type=str, default="G1/G1_tennis_highlevel")
    parser.add_argument("--exp", type=str, default="highlevel")
    parser.add_argument("--num-envs", type=int, default=64, help="Unused (kept for backward compatibility).")
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
    parser.add_argument("--launch-spin-rps-range", type=float, nargs=2, default=None)
    parser.add_argument("--angle-deg-range", type=float, nargs=2, default=None)
    parser.add_argument("--min-vz", type=float, default=None)
    parser.add_argument("--min-forward-speed", type=float, default=None)
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
    cmd_cfg = HighLevelTennisConfig.from_any(cfg.task.command.config)
    court_cfg = cmd_cfg.court

    defaults = LaunchSamplerConfig()
    physics_cfg = LaunchPhysicsConfig(
        ball_radius=float(cmd_cfg.ball_radius),
        ball_mass=float(cmd_cfg.ball_mass),
        air_density=float(cmd_cfg.air_density),
        air_drag_k=float(cmd_cfg.air_drag_k),
        drag_coef=float(cmd_cfg.drag_coef),
        lift_spin_scale=float(cmd_cfg.lift_spin_scale),
        spin_damping_coef=float(cmd_cfg.spin_damping_coef),
        net_height=float(court_cfg.net_height),
        gravity_z=-9.81,
    )
    sampler_cfg = LaunchSamplerConfig(
        launcher_x_range=_resolve_range("launcher_x_range", args.launcher_x_range, defaults.launcher_x_range),
        launcher_y_range=_resolve_range("launcher_y_range", args.launcher_y_range, defaults.launcher_y_range),
        launcher_z_range=_resolve_range("launcher_z_range", args.launcher_z_range, defaults.launcher_z_range),
        strike_x_range=_resolve_range("strike_x_range", args.strike_x_range, defaults.strike_x_range),
        strike_y_range=_resolve_range("strike_y_range", args.strike_y_range, defaults.strike_y_range),
        strike_z_range=_resolve_range("strike_z_range", args.strike_z_range, defaults.strike_z_range),
        flight_t_range=_resolve_range("flight_t_range", args.flight_t_range, defaults.flight_t_range),
        launch_speed_range=_resolve_range("launch_speed_range", args.launch_speed_range, defaults.launch_speed_range),
        launch_spin_rps_range=_resolve_range(
            "launch_spin_rps_range", args.launch_spin_rps_range, defaults.launch_spin_rps_range
        ),
        angle_deg_range=_resolve_range("angle_deg_range", args.angle_deg_range, defaults.angle_deg_range),
        incoming_bounce_x_range=_resolve_range(
            "incoming_bounce_x_range", args.incoming_bounce_x_range, defaults.incoming_bounce_x_range
        ),
        incoming_bounce_y_range=_resolve_range(
            "incoming_bounce_y_range", args.incoming_bounce_y_range, defaults.incoming_bounce_y_range
        ),
        target_x_range=_resolve_range("target_x_range", args.target_x_range, defaults.target_x_range),
        target_y_range=_resolve_range("target_y_range", args.target_y_range, defaults.target_y_range),
        min_vz=(float(args.min_vz) if args.min_vz is not None else float(defaults.min_vz)),
        min_forward_speed=(
            float(args.min_forward_speed) if args.min_forward_speed is not None else float(defaults.min_forward_speed)
        ),
        physics_dt=float(cfg.task.sim.get("mujoco_physics_dt", defaults.physics_dt)),
    )

    sampler = LaunchTrajectorySampler(device=args.device, physics=physics_cfg, config=sampler_cfg)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(repo_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = int(args.num_samples)
    batch_size = max(1, int(args.batch_size))
    produced = 0

    pos_local_list: List[torch.Tensor] = []
    vel_list: List[torch.Tensor] = []
    ang_list: List[torch.Tensor] = []
    tgt_local_list: List[torch.Tensor] = []

    print(f"[INFO] Sampling launch bank: total={total}, batch={batch_size}, device={args.device}")

    while produced < total:
        n_target = min(batch_size, total - produced)
        n = n_target
        sample_ok = False
        last_err: Exception | None = None
        while not sample_ok:
            try:
                with torch.no_grad():
                    pos_local, vel, ang, tgt_local = sampler.sample(n)
                sample_ok = True
            except RuntimeError as err:
                last_err = err
                if n > 1:
                    n = max(1, n // 2)
                    print(f"[WARN] sampler.sample failed for batch={n_target}, retry with smaller batch={n}: {err}")
                    continue
                # n==1: allow a few retries before surfacing the error.
                retried = False
                for retry_i in range(8):
                    try:
                        with torch.no_grad():
                            pos_local, vel, ang, tgt_local = sampler.sample(1)
                        sample_ok = True
                        retried = True
                        if retry_i > 0:
                            print(f"[WARN] single-sample retry succeeded after {retry_i + 1} attempts.")
                        break
                    except RuntimeError as err_single:
                        last_err = err_single
                if not retried:
                    raise RuntimeError(
                        f"Failed to sample even a single launch after retries. Last error: {last_err}"
                    ) from last_err

        pos_local_list.append(pos_local.detach().cpu())
        vel_list.append(vel.detach().cpu())
        ang_list.append(ang.detach().cpu())
        tgt_local_list.append(tgt_local.detach().cpu())
        produced += int(pos_local.shape[0])

        if args.print_every > 0 and (produced % int(args.print_every) == 0 or produced == total):
            print(f"[INFO] sampled {produced}/{total}")

    pos_local = torch.cat(pos_local_list, dim=0)[:total].numpy().astype(np.float32, copy=False)
    vel = torch.cat(vel_list, dim=0)[:total].numpy().astype(np.float32, copy=False)
    ang = torch.cat(ang_list, dim=0)[:total].numpy().astype(np.float32, copy=False)
    tgt_local = torch.cat(tgt_local_list, dim=0)[:total].numpy().astype(np.float32, copy=False)
    speed = np.linalg.norm(vel, axis=1)
    forward_speed = -vel[:, 1]
    speed_hi = float(sampler_cfg.launch_speed_range[1])
    near_hi_ratio = float((speed >= (speed_hi - 1.0e-3)).mean())

    np.savez_compressed(
        output_path,
        launch_pos_local=pos_local,
        launch_vel=vel,
        launch_ang=ang,
        target_bounce_local=tgt_local,
        sim_physics_dt=np.array(sampler_cfg.physics_dt, dtype=np.float32),
        air_drag_k=np.array(physics_cfg.air_drag_k, dtype=np.float32),
        drag_coef=np.array(physics_cfg.drag_coef, dtype=np.float32),
        lift_spin_scale=np.array(physics_cfg.lift_spin_scale, dtype=np.float32),
        spin_damping_coef=np.array(physics_cfg.spin_damping_coef, dtype=np.float32),
        air_density=np.array(physics_cfg.air_density, dtype=np.float32),
        ball_mass=np.array(physics_cfg.ball_mass, dtype=np.float32),
        ball_radius=np.array(physics_cfg.ball_radius, dtype=np.float32),
        net_height=np.array(physics_cfg.net_height, dtype=np.float32),
    )
    print(f"[INFO] Saved launch bank: {output_path}")
    print(f"[INFO] Shapes: pos={pos_local.shape}, vel={vel.shape}, ang={ang.shape}, target={tgt_local.shape}")
    print(
        "[INFO] Launch stats: "
        f"speed p50/p90=({np.percentile(speed, 50):.2f}, {np.percentile(speed, 90):.2f}) "
        f"forward p50/p90=({np.percentile(forward_speed, 50):.2f}, {np.percentile(forward_speed, 90):.2f}) "
        f"near_hi_ratio={near_hi_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
