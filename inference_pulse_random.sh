#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-8}"
PULSE_TEMP="${PULSE_TEMP:-1.0}"
CFG_PATH="${CFG_PATH:-cfg/train.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-15/17-34-58-tracking-pulse-only-tennis-0415-1734/wandb/run-20260415_173529-b4bx1esr/files/checkpoint_2400.pt}"
USE_RACKET="${USE_RACKET:-0}"
ROBOT_NAME="${ROBOT_NAME:-}"

USE_RACKET_NORM="$(printf '%s' "${USE_RACKET}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${ROBOT_NAME}" ]]; then
  case "${USE_RACKET_NORM}" in
    1|true|yes|y|on)
      ROBOT_NAME="g1_col_full_self_racket"
      ;;
    0|false|no|n|off|"")
      ROBOT_NAME="g1_col_full_self"
      ;;
    *)
      echo "[ERROR] Invalid USE_RACKET='${USE_RACKET}', expected 0/1 or true/false."
      exit 1
      ;;
  esac
fi

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
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"

uv run python scripts/inference_pulse_random.py \
  --cfg "${CFG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --temp "${PULSE_TEMP}" \
  --robot-name "${ROBOT_NAME}"
