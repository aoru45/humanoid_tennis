#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_DIR="${RUN_DIR:-outputs/2026-06-20/15-15-57-tracking-stage1-tennis-0620-1515}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/checkpoint_final.pt}"
CFG="${CFG:-${RUN_DIR}/wandb/offline-run-20260620_151736-p419y2pj/files/cfg.yaml}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-4}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
DATASET="${DATASET:-run_tennis_subset}"
FIX_DS="${FIX_DS:-0}"
REFERENCE_ALPHA="${REFERENCE_ALPHA:-0.65}"
VIEWER_MAX_FPS="${VIEWER_MAX_FPS:-45}"
TARGET_FPS="${TARGET_FPS:-50}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="uv run --no-sync python"
  fi
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CFG}" ]]; then
  echo "[ERROR] cfg not found: ${CFG}" >&2
  exit 1
fi

${PYTHON_BIN} scripts/inference_local.py \
  --cfg "${CFG}" \
  --checkpoint "${CKPT}" \
  --device "${DEVICE}" \
  --num-envs "${NUM_ENVS}" \
  --robot "${ROBOT_NAME}" \
  --dataset "${DATASET}" \
  --fix-ds "${FIX_DS}" \
  --reference-alpha "${REFERENCE_ALPHA}" \
  --viewer-max-fps "${VIEWER_MAX_FPS}" \
  --target-fps "${TARGET_FPS}" \
  "$@"
