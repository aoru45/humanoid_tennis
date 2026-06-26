#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MOTION="${MOTION:-dataset/tennis}"
DEVICE="${DEVICE:-cuda:0}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
SPEED="${SPEED:-1.0}"
VIEWER_MAX_FPS="${VIEWER_MAX_FPS:-45}"
PHYSICS_DT="${PHYSICS_DT:-0.0005}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOOP_ARG=()
if [[ "${LOOP:-1}" == "1" ]]; then
  LOOP_ARG+=(--loop)
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="uv run --no-sync python"
  fi
fi

${PYTHON_BIN} scripts/data_process/replay_motion_npz.py "${MOTION}" \
  --robot "${ROBOT_NAME}" \
  --device "${DEVICE}" \
  --speed "${SPEED}" \
  --physics-dt "${PHYSICS_DT}" \
  --viewer-max-fps "${VIEWER_MAX_FPS}" \
  "${LOOP_ARG[@]}"
