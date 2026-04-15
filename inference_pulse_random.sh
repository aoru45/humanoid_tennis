#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-8}"
PULSE_TEMP="${PULSE_TEMP:-1.0}"
CFG_PATH="${CFG_PATH:-cfg/train.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-15/15-08-02-G1TRACKING-ppo/wandb/run-20260415_150832-b4vugxit/files/checkpoint_1950.pt}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CFG_PATH}"
  exit 1
fi

echo "[INFO] Launch pulse random inference"
echo "[INFO] checkpoint=${CHECKPOINT_PATH}"
echo "[INFO] cfg=${CFG_PATH}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] pulse_prior_temp=${PULSE_TEMP}"

uv run python scripts/inference_pulse_random.py \
  --cfg "${CFG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --temp "${PULSE_TEMP}"
