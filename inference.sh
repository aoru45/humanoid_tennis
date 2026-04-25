#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

uv run python scripts/inference_local.py \
  --cfg outputs/track_seed/cfg.yaml \
  --checkpoint outputs/track_seed/checkpoint_final.pt \
  --num-envs 8 \
  --device cuda:0
