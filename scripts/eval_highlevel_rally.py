from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Add project root for `scripts.*` absolute imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils.helpers import make_env_policy


def _count_indicator(x: torch.Tensor) -> torch.Tensor:
    return (x.abs() > 1.0e-6).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate highlevel tennis with per-serve rally metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/tennis_highlevel/checkpoints/checkpoint_final.pt",
        help="Path to highlevel checkpoint (.pt).",
    )
    parser.add_argument(
        "--bank-file",
        type=str,
        default="data/tennis_launch_bank/highlevel_subsets/launch_bank_easy.npz",
        help="Launch bank used for serve sampling.",
    )
    parser.add_argument("--episodes", type=int, default=512, help="Number of rallies to evaluate.")
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel env count.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=128)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    bank_path = Path(args.bank_file).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"launch bank not found: {bank_path}")

    with hydra.initialize(config_path="../cfg", job_name="eval_highlevel_rally", version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["task=G1/G1_tennis_highlevel", "+exp=highlevel"])

    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(args.seed)
    cfg.checkpoint_path = str(ckpt_path)
    cfg.vecnorm = "eval"
    cfg.rollout_mode = "eval"
    cfg.task.num_envs = int(args.num_envs)
    cfg.app.headless = True
    cfg.task.viewer.headless = True

    # Force one-serve-per-episode, matching the user's success definition.
    cfg.task.command.config.episode.relaunch_on_success = False
    cfg.task.command.config.episode.max_consecutive_returns_before_finish = 1

    # Evaluate with easy launch bank only.
    cfg.task.command.config.launch.bank.file = str(bank_path)
    cfg.task.command.config.launch.bank.easy_file = None
    cfg.task.command.config.launch.bank.medium_file = None
    cfg.task.command.config.launch.bank.hard_file = None
    cfg.task.command.config.launch.bank.use_curriculum = False

    # Expose hit flag as an episode term metric.
    cfg.task.reward.term.episode_has_hit = {"weight": 1.0}

    env, policy, _, _ = make_env_policy(cfg)
    policy = policy.get_rollout_policy("eval")

    if hasattr(policy, "step_schedule"):
        policy.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)

    target_episodes = max(1, int(args.episodes))
    report_every = max(1, int(args.progress_every))
    next_report = report_every
    done_episodes = 0
    success = 0
    has_hit = 0
    pass_net = 0
    has_bounce = 0
    fail_miss = 0
    fail_net = 0
    fail_out = 0
    fail_style = 0
    fail_racket_pre = 0
    fail_racket_post = 0
    fall = 0

    td_ = env.reset()
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        while done_episodes < target_episodes:
            td_ = policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            done_mask = td["next", "done"].squeeze(-1)
            if not bool(done_mask.any()):
                continue

            stats = td["next", "stats"]
            term = stats["term"]

            success_t = stats["success"].squeeze(-1)[done_mask]
            success += int((success_t > 0.5).sum().item())
            has_hit += int(_count_indicator(term["episode_has_hit"].squeeze(-1)[done_mask]).sum().item())
            pass_net += int(_count_indicator(term["episode_pass_net"].squeeze(-1)[done_mask]).sum().item())
            has_bounce += int(_count_indicator(term["episode_has_bounce"].squeeze(-1)[done_mask]).sum().item())
            fail_miss += int(_count_indicator(term["episode_fail_miss"].squeeze(-1)[done_mask]).sum().item())
            fail_net += int(_count_indicator(term["episode_fail_net"].squeeze(-1)[done_mask]).sum().item())
            fail_out += int(_count_indicator(term["episode_fail_out"].squeeze(-1)[done_mask]).sum().item())
            fail_style += int(_count_indicator(term["episode_stroke_style_violation"].squeeze(-1)[done_mask]).sum().item())
            fail_racket_pre += int(
                _count_indicator(term["episode_fail_racket_body_pre_hit"].squeeze(-1)[done_mask]).sum().item()
            )
            fail_racket_post += int(
                _count_indicator(term["episode_fail_racket_body_post_hit"].squeeze(-1)[done_mask]).sum().item()
            )
            fall += int(_count_indicator(term["episode_fall"].squeeze(-1)[done_mask]).sum().item())

            done_episodes += int(done_mask.sum().item())
            if done_episodes >= next_report or done_episodes >= target_episodes:
                hit_rate = 100.0 * has_hit / max(done_episodes, 1)
                success_rate = 100.0 * success / max(done_episodes, 1)
                print(
                    f"[MJLab Rally Eval] episodes={done_episodes} hit_rate={hit_rate:.2f}% "
                    f"success_rate={success_rate:.2f}%"
                )
                while next_report <= done_episodes:
                    next_report += report_every

    env.close()

    hit_rate = 100.0 * has_hit / max(done_episodes, 1)
    success_rate = 100.0 * success / max(done_episodes, 1)

    print("\n=== MJLab Rally Eval (Per-Serve) ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"launch_bank: {bank_path}")
    print(f"episodes: {done_episodes}")
    print(f"hit_rate: {has_hit}/{done_episodes} = {hit_rate:.2f}%")
    print(f"success_rate: {success}/{done_episodes} = {success_rate:.2f}%")
    print("failure_breakdown:")
    print(f"  miss: {fail_miss}")
    print(f"  net: {fail_net}")
    print(f"  out: {fail_out}")
    print(f"  style: {fail_style}")
    print(f"  racket_body_pre_hit: {fail_racket_pre}")
    print(f"  racket_body_post_hit: {fail_racket_post}")
    print(f"  fall: {fall}")
    print("aux:")
    print(f"  pass_net: {pass_net}")
    print(f"  has_bounce: {has_bounce}")


if __name__ == "__main__":
    main()
