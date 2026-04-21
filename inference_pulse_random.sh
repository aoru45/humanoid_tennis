#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-8}"
PULSE_TEMP="${PULSE_TEMP:-1.0}"
CFG_PATH="${CFG_PATH:-cfg/train.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-18/01-01-39-tracking-pulse-tennis-0418-0101/wandb/run-20260418_010213-fy4oipc6/files/checkpoint_final.pt}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-17/23-05-53-tracking-pulse-only-tennis-0417-2305/wandb/run-20260417_230629-nxafmogr/files/checkpoint_final.pt}"

USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket}"

echo "[INFO] Launch pulse random inference"
echo "[INFO] checkpoint=${CHECKPOINT_PATH}"
echo "[INFO] cfg=${CFG_PATH}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] pulse_prior_temp=${PULSE_TEMP}"
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"

uv run python scripts/inference_pulse_random.py \
  --cfg "${CFG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --temp "${PULSE_TEMP}" \
  --robot-name "${ROBOT_NAME}"
