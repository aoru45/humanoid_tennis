import argparse
import contextlib
import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Add project root to path for local imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.helpers import make_env_policy


def resolve_launch_bank_subset_path(args) -> tuple[str, str]:
    subset = str(args.mode).strip().lower()
    if subset == "mid":
        subset = "medium"
    if subset not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unsupported mode: {args.mode}")
    bank_path = os.path.join(
        str(args.launch_bank_root),
        f"launch_bank_{subset}.npz",
    )
    return subset, bank_path


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
                f"task.num_envs={args.num_envs}",
            ],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.headless = False
    cfg.app.headless = False
    cfg.task.viewer.headless = False
    cfg.task.viewer.debug_visualization_enabled = bool(args.viewer_debug)
    cfg.task.viewer.camera_tracking_enabled = False
    cfg.task.command.debug_draw = bool(args.command_debug_draw)
    cfg.task.sim.device = args.device
    if args.seed is not None:
        cfg.seed = int(args.seed)
    ckpt_arg = str(args.checkpoint_path).strip()
    if ckpt_arg and ckpt_arg.lower() not in {"none", "null"}:
        cfg.checkpoint_path = ckpt_arg
    else:
        cfg.checkpoint_path = None
    subset, bank_path = resolve_launch_bank_subset_path(args)
    cfg.task.command.launch_bank_file = bank_path
    # Debug script uses a single subset bank; disable mixed multi-bank loading.
    cfg.task.command.launch_bank_easy_file = ""
    cfg.task.command.launch_bank_medium_file = ""
    cfg.task.command.launch_bank_hard_file = ""
    if args.disable_randomization and "randomization" in cfg.task:
        cfg.task.randomization = {}
    if args.disable_fall_over_termination and "termination" in cfg.task and "fall_over" in cfg.task.termination:
        del cfg.task.termination["fall_over"]
    args._resolved_launch_bank_subset = subset
    args._resolved_launch_bank_path = bank_path
    return cfg


def build_env_policy(cfg):
    env, policy, _vecnorm, _primer = make_env_policy(cfg)
    if hasattr(policy, "step_schedule"):
        policy.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)
    return env, policy



def apply_my_rollout(td, policy):
    policy.actor_highlevel(td)
    policy.highlevel_action_split(td)
    policy.pulse_prior(td)
    policy.highlevel_latent_barrier(td)
    # prior_z = td["pulse_z"]
    # td["prior_z"] = prior_z
    # td["delta_z"] = torch.zeros_like(prior_z)
    # td["pulse_z"] = td["prior_z"] + td["delta_z"]
    policy.pulse_decoder(td)
    policy.highlevel_wrist_residual(td)
    policy.pulse_action_head(td)
    return td


def td_flag(td, key, env_id: int, threshold: float = 0.5) -> int:
    try:
        value = td.get(key)
    except Exception:
        return 0
    if value is None:
        return 0
    scalar = value[env_id].reshape(-1)[0]
    return int((scalar > threshold).item())


def install_launch_cache(
    env: TransformedEnv,
    cmd,
    cache_size: int,
    refill_size: int,
    warm_start_size: int,
) -> bool:
    if cache_size <= 0 or not hasattr(cmd, "_sample_ball_launch"):
        return False

    original_sampler = cmd._sample_ball_launch
    device = env.device
    env_ids_all = torch.arange(env.num_envs, dtype=torch.long, device=device)
    cache = {
        "local_pos": None,
        "vel": None,
        "ang": None,
        "local_tgt": None,
        "idx": 0,
        "n": 0,
    }

    def refill(min_count: int):
        target_cache = max(1, int(cache_size))
        chunk_size = int(refill_size) if refill_size > 0 else target_cache
        chunk_size = max(1, min(target_cache, chunk_size))
        if cache["local_pos"] is None and warm_start_size > 0:
            chunk_size = max(1, min(target_cache, int(warm_start_size)))
        n = max(chunk_size, int(min_count))
        repeats = max(1, (n + env.num_envs - 1) // env.num_envs)
        sample_env_ids = env_ids_all.repeat(repeats)[:n]
        with torch.no_grad():
            pos_w, vel, ang, tgt_w = original_sampler(sample_env_ids)
        origins = env.base_env.scene.env_origins[sample_env_ids]
        cache["local_pos"] = pos_w - origins
        cache["vel"] = vel
        cache["ang"] = ang
        cache["local_tgt"] = tgt_w - origins
        cache["idx"] = 0
        cache["n"] = n

    def cached_sampler(env_ids: torch.Tensor):
        num_req = int(env_ids.numel())
        pos_local = torch.zeros((num_req, 3), device=device, dtype=torch.float32)
        vel = torch.zeros((num_req, 3), device=device, dtype=torch.float32)
        ang = torch.zeros((num_req, 3), device=device, dtype=torch.float32)
        tgt_local = torch.zeros((num_req, 3), device=device, dtype=torch.float32)

        filled = 0
        while filled < num_req:
            available = cache["n"] - cache["idx"] if cache["local_pos"] is not None else 0
            if available <= 0:
                refill(num_req - filled)
                available = cache["n"] - cache["idx"]

            take = min(available, num_req - filled)
            src = slice(cache["idx"], cache["idx"] + take)
            dst = slice(filled, filled + take)
            pos_local[dst] = cache["local_pos"][src]
            vel[dst] = cache["vel"][src]
            ang[dst] = cache["ang"][src]
            tgt_local[dst] = cache["local_tgt"][src]
            cache["idx"] += take
            filled += take

        origins = env.base_env.scene.env_origins[env_ids]
        return pos_local + origins, vel, ang, tgt_local + origins

    cmd._sample_ball_launch = cached_sampler
    return True


def run(args):
    cfg = build_cfg(args)
    env, policy = build_env_policy(cfg)
    cmd = env.base_env.command_manager
    cache_enabled = install_launch_cache(
        env,
        cmd,
        int(args.launch_cache_size),
        int(args.launch_cache_refill_size),
        int(args.launch_cache_warm_start),
    )
    sleep_dt = float(args.realtime_scale) * float(env.base_env.step_dt)
    min_loop_dt = 0.0
    if float(args.viewer_max_fps) > 0.0:
        min_loop_dt = 1.0 / float(args.viewer_max_fps)

    print(f"[INFO] task={args.task}, exp={args.exp}, num_envs={env.num_envs}, device={env.device}")
    if str(env.device) != str(args.device):
        print(
            f"[WARN] requested device={args.device}, actual env device={env.device}. "
            "Current env backend may ignore task.sim.device."
        )
    print(f"[INFO] checkpoint_path={cfg.checkpoint_path}")
    print(
        "[INFO] disable_randomization="
        f"{bool(args.disable_randomization)}, "
        f"disable_fall_over_termination={bool(args.disable_fall_over_termination)}"
    )
    print("[INFO] Viser launched by env viewer. Press Ctrl+C to stop.")
    print(
        "[INFO] viewer throttle:",
        f"max_fps={float(args.viewer_max_fps):.1f}",
        f"min_loop_dt={min_loop_dt:.4f}s",
    )
    print(
        f"[INFO] Launch bank subset: {args._resolved_launch_bank_subset}, "
        f"path={args._resolved_launch_bank_path}"
    )
    if cache_enabled:
        print(
            "[INFO] Launch cache enabled: "
            f"size={int(args.launch_cache_size)}, "
            f"refill={int(args.launch_cache_refill_size)}, "
            f"warm_start={int(args.launch_cache_warm_start)}"
        )

    carry = env.reset()
    step = 0
    next_forced_reset = args.reset_every if args.reset_every > 0 else -1
    
    try:
        while True:
            loop_t0 = time.perf_counter()
            if args.steps > 0 and step >= args.steps:
                break

            carry = apply_my_rollout(carry, policy)

            if hasattr(env, "step_and_maybe_reset") and (not args.no_auto_reset):
                td, carry = env.step_and_maybe_reset(carry)
                done = td.get(("next", "done"))
                if done.any() and args.print_reset_ids:
                    reset_mask = done.to(dtype=torch.bool).reshape(env.num_envs)
                    reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                    print(f"[reset] env_ids={reset_ids}")
            else:
                td = env.step(carry)
                done = td.get(("next", "done"))
                carry = td.get("next")
                if done.any() and (not args.no_auto_reset):
                    carry = env.reset(carry)
                    if args.print_reset_ids:
                        reset_mask = done.to(dtype=torch.bool).reshape(env.num_envs)
                        reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                        print(f"[reset] env_ids={reset_ids}")

            if args.print_every > 0 and (step % args.print_every == 0):
                env_id = min(max(0, args.env_id), env.num_envs - 1)
                ball_pos = cmd.ball.data.root_link_pos_w[env_id].detach().cpu()
                env_origin = env.base_env.scene.env_origins[env_id].detach().cpu()
                ball_pos_local = ball_pos - env_origin
                ball_vel = cmd.ball.data.root_link_lin_vel_w[env_id].detach().cpu()
                print(
                    f"[step {step:06d}] env={env_id} "
                    f"ball_pos_local=({ball_pos_local[0]:+.3f}, {ball_pos_local[1]:+.3f}, {ball_pos_local[2]:+.3f}) "
                    f"ball_vel=({ball_vel[0]:+.3f}, {ball_vel[1]:+.3f}, {ball_vel[2]:+.3f}) "
                    f"hit={int(cmd.has_hit[env_id].item())} "
                    f"bounce={int(cmd.has_bounce[env_id].item())} "
                    f"success={int(cmd.success[env_id].item())} "
                    f"task_step={int(cmd.task_step[env_id].item())} "
                    f"done={int(done[env_id].item())} "
                    f"truncated={td_flag(td, ('next', 'truncated'), env_id)} "
                    f"terminated={td_flag(td, ('next', 'terminated'), env_id)} "
                    f"term_miss={td_flag(td, ('next', 'stats', 'termination', 'miss_ball_termination'), env_id)} "
                    f"term_net={td_flag(td, ('next', 'stats', 'termination', 'net_hit_termination'), env_id)} "
                    f"term_out={td_flag(td, ('next', 'stats', 'termination', 'ball_out_of_bounds_termination'), env_id)} "
                    f"term_fall={td_flag(td, ('next', 'stats', 'termination', 'fall_over'), env_id)}"
                )

            if next_forced_reset > 0 and (step + 1) == next_forced_reset:
                carry = env.reset()
                next_forced_reset += args.reset_every

            target_loop_dt = max(float(sleep_dt), float(min_loop_dt))
            if target_loop_dt > 0:
                delay = target_loop_dt - (time.perf_counter() - loop_t0)
                if delay > 0:
                    time.sleep(delay)
            step += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            env.close()
        except TypeError:
            # TorchRL may forward `raise_if_closed`, while project env close() has no kwargs.
            env.base_env.close()


def main():
    parser = argparse.ArgumentParser(description="Debug high-level tennis trajectories in Viser under training-consistent env setup.")
    parser.add_argument("--task", type=str, default="G1/G1_tennis_highlevel")
    parser.add_argument("--exp", type=str, default="highlevel")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default="null", help="Policy checkpoint path. Use 'null' for random init policy.")
    parser.add_argument("--steps", type=int, default=-1, help="-1 means run until Ctrl+C.")
    parser.add_argument("--reset-every", type=int, default=0, help="Force reset all envs every N steps. <=0 disables.")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--print-reset-ids", action="store_true", help="Print env IDs that are reset on done.")
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--max-episode-length", type=int, default=0, help="Override task.max_episode_length if >0.")
    parser.add_argument("--command-max-task-steps", type=int, default=0, help="Override task.command.max_task_steps if >0.")
    parser.add_argument(
        "--mode",
        type=str,
        default="easy",
        choices=["easy", "mid", "hard"],
        help="Select launch bank subset. 'mid' maps to 'medium'.",
    )
    parser.add_argument(
        "--launch-bank-root",
        type=str,
        default="data/tennis_launch_bank/highlevel_subsets",
        help="Directory containing launch_bank_easy/medium/hard.npz.",
    )
    parser.add_argument("--launch-cache-size", type=int, default=0, help="Pre-generate launch states in batches for faster resets.")
    parser.add_argument("--launch-cache-refill-size", type=int, default=32, help="Per-refill sample batch size for launch cache.")
    parser.add_argument("--launch-cache-warm-start", type=int, default=8, help="Initial cache batch size on first reset.")
    parser.add_argument("--no-auto-reset", action="store_true", help="Do not auto-reset on done; keep stepping current episode.")
    parser.add_argument("--disable-randomization", action="store_true", help="Disable task randomization for simplified debugging.")
    parser.add_argument("--disable-fall-over-termination", action="store_true", help="Disable fall_over termination for simplified debugging.")
    parser.add_argument("--realtime-scale", type=float, default=0., help="sleep = step_dt * scale; set 0 for fastest.")
    parser.add_argument(
        "--viewer-max-fps",
        type=float,
        default=45.0,
        help="Throttle loop to at most this FPS to keep Viser responsive. <=0 disables throttling.",
    )
    parser.add_argument("--viewer-debug", action="store_true", help="Enable Viser debug overlays.")
    parser.add_argument("--command-debug-draw", action="store_true", help="Enable command debug draw points.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
