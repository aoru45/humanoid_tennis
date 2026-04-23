#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK="${TASK:-G1/G1_tennis_highlevel}"
EXP="${EXP:-highlevel}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-10240}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-data/tennis_launch_bank/highlevel_launch_bank.npz}"

# Close-feed defaults: bias serves toward robot side to reduce long chasing.
# All ranges are "min max".
LAUNCHER_X_RANGE="${LAUNCHER_X_RANGE:--4.0 4.0}"
LAUNCHER_Y_RANGE="${LAUNCHER_Y_RANGE:-7.0 8.8}"
STRIKE_X_RANGE="${STRIKE_X_RANGE:--4.0 4.0}"
STRIKE_Y_RANGE="${STRIKE_Y_RANGE:--12.0 -4.0}"
INCOMING_BOUNCE_X_RANGE="${INCOMING_BOUNCE_X_RANGE:--4.0 4.0}"
INCOMING_BOUNCE_Y_RANGE="${INCOMING_BOUNCE_Y_RANGE:--12 -4.0}"
FLIGHT_T_RANGE="${FLIGHT_T_RANGE:-0.70 1.00}"
LAUNCH_SPEED_RANGE="${LAUNCH_SPEED_RANGE:-12.0 30.0}"

read -r LAUNCHER_X_MIN LAUNCHER_X_MAX <<< "${LAUNCHER_X_RANGE}"
read -r LAUNCHER_Y_MIN LAUNCHER_Y_MAX <<< "${LAUNCHER_Y_RANGE}"
read -r STRIKE_X_MIN STRIKE_X_MAX <<< "${STRIKE_X_RANGE}"
read -r STRIKE_Y_MIN STRIKE_Y_MAX <<< "${STRIKE_Y_RANGE}"
read -r IN_BOUNCE_X_MIN IN_BOUNCE_X_MAX <<< "${INCOMING_BOUNCE_X_RANGE}"
read -r IN_BOUNCE_Y_MIN IN_BOUNCE_Y_MAX <<< "${INCOMING_BOUNCE_Y_RANGE}"
read -r FLIGHT_T_MIN FLIGHT_T_MAX <<< "${FLIGHT_T_RANGE}"
read -r LAUNCH_SPEED_MIN LAUNCH_SPEED_MAX <<< "${LAUNCH_SPEED_RANGE}"

echo "[INFO] Generating launch bank with close-feed defaults"
echo "[INFO] task=${TASK} exp=${EXP} output=${OUTPUT}"
echo "[INFO] strike_y_range=[${STRIKE_Y_MIN}, ${STRIKE_Y_MAX}] incoming_bounce_y_range=[${IN_BOUNCE_Y_MIN}, ${IN_BOUNCE_Y_MAX}]"
echo "[INFO] Launch bank will include incoming_first_bounce_local + incoming_first_bounce_time_s"

PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --num-envs "${NUM_ENVS}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range "${LAUNCHER_X_MIN}" "${LAUNCHER_X_MAX}" \
  --launcher-y-range "${LAUNCHER_Y_MIN}" "${LAUNCHER_Y_MAX}" \
  --strike-x-range "${STRIKE_X_MIN}" "${STRIKE_X_MAX}" \
  --strike-y-range "${STRIKE_Y_MIN}" "${STRIKE_Y_MAX}" \
  --incoming-bounce-x-range "${IN_BOUNCE_X_MIN}" "${IN_BOUNCE_X_MAX}" \
  --incoming-bounce-y-range "${IN_BOUNCE_Y_MIN}" "${IN_BOUNCE_Y_MAX}" \
  --flight-t-range "${FLIGHT_T_MIN}" "${FLIGHT_T_MAX}" \
  --launch-speed-range "${LAUNCH_SPEED_MIN}" "${LAUNCH_SPEED_MAX}" \
  --output "${OUTPUT}"
