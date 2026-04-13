#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv run python scripts/inference_local.py \
  --cfg outputs/2026-04-13/15-43-12-G1TRACKING-ppo/wandb/run-20260413_154341-8aqcoput/files/cfg.yaml \
  --checkpoint outputs/2026-04-13/15-43-12-G1TRACKING-ppo/wandb/run-20260413_154341-8aqcoput/files/checkpoint_1500.pt \
  --num-envs 8 \
  --device cuda:0
