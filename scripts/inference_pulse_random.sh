#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-23/15-18-37-tracking-pulse-run-tennis-0423-1518/checkpoints/checkpoint_final.pt}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-8}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"

if [[ "${CHECKPOINT_PATH}" != /* ]]; then
  CHECKPOINT_PATH="${REPO_ROOT}/${CHECKPOINT_PATH}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

uv run python "${REPO_ROOT}/scripts/inference_pulse_random.py" \
  --cfg "${REPO_ROOT}/cfg/train.yaml" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --temp 1.0 \
  --step-dt 0.02 \
  --physics-dt 0.0005 \
  --viewer-max-fps 12 \
  --playback-fps 4 \
  --robot-name "${ROBOT_NAME}"
