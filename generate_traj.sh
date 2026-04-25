#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK="${TASK:-G1/G1_tennis_highlevel}"
EXP="${EXP:-highlevel}"
MODE="${MODE:-three_subset}"  # single | three_subset
NUM_ENVS="${NUM_ENVS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-data/tennis_launch_bank/highlevel_launch_bank.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-data/tennis_launch_bank/highlevel_subsets}"
REBOUND_FILTER="${REBOUND_FILTER:-false}"  # true | false

# Common launch ranges.
LAUNCHER_X_RANGE="${LAUNCHER_X_RANGE:--4.0 4.0}"
LAUNCHER_Y_RANGE="${LAUNCHER_Y_RANGE:-7.0 8.8}"
FLIGHT_T_RANGE="${FLIGHT_T_RANGE:-0.70 1.00}"
LAUNCH_SPEED_RANGE="${LAUNCH_SPEED_RANGE:-12.0 24.0}"

# Single-bank ranges (legacy mode).
STRIKE_X_RANGE="${STRIKE_X_RANGE:--4.0 4.0}"
STRIKE_Y_RANGE="${STRIKE_Y_RANGE:--9.5 -6.5}"
INCOMING_BOUNCE_X_RANGE="${INCOMING_BOUNCE_X_RANGE:--4.0 4.0}"
INCOMING_BOUNCE_Y_RANGE="${INCOMING_BOUNCE_Y_RANGE:--12.0 -7.0}"

# Three-subset sample counts (default: each subset has NUM_SAMPLES samples).
NUM_SAMPLES_EASY="${NUM_SAMPLES_EASY:-${NUM_SAMPLES}}"
NUM_SAMPLES_MEDIUM="${NUM_SAMPLES_MEDIUM:-${NUM_SAMPLES}}"
NUM_SAMPLES_HARD="${NUM_SAMPLES_HARD:-${NUM_SAMPLES}}"

# Easy: feed close to racket zone.
EASY_STRIKE_X_RANGE="${EASY_STRIKE_X_RANGE:--0.8 1.2}"
EASY_STRIKE_Y_RANGE="${EASY_STRIKE_Y_RANGE:--9.6 -7.2}"
EASY_INCOMING_BOUNCE_X_RANGE="${EASY_INCOMING_BOUNCE_X_RANGE:--1.2 1.6}"
EASY_INCOMING_BOUNCE_Y_RANGE="${EASY_INCOMING_BOUNCE_Y_RANGE:--10.2 -7.4}"

# Medium: wider but still biased to reachable area.
MEDIUM_STRIKE_X_RANGE="${MEDIUM_STRIKE_X_RANGE:--2.2 2.2}"
MEDIUM_STRIKE_Y_RANGE="${MEDIUM_STRIKE_Y_RANGE:--10.8 -6.0}"
MEDIUM_INCOMING_BOUNCE_X_RANGE="${MEDIUM_INCOMING_BOUNCE_X_RANGE:--2.8 2.8}"
MEDIUM_INCOMING_BOUNCE_Y_RANGE="${MEDIUM_INCOMING_BOUNCE_Y_RANGE:--11.2 -5.8}"

# Hard: near full-court spread.
HARD_STRIKE_X_RANGE="${HARD_STRIKE_X_RANGE:--4.0 4.0}"
HARD_STRIKE_Y_RANGE="${HARD_STRIKE_Y_RANGE:--12.0 -4.0}"
HARD_INCOMING_BOUNCE_X_RANGE="${HARD_INCOMING_BOUNCE_X_RANGE:--3.8 3.8}"
HARD_INCOMING_BOUNCE_Y_RANGE="${HARD_INCOMING_BOUNCE_Y_RANGE:--11.5 -3.5}"

OUTPUT_EASY="${OUTPUT_EASY:-${OUTPUT_DIR}/launch_bank_easy.npz}"
OUTPUT_MEDIUM="${OUTPUT_MEDIUM:-${OUTPUT_DIR}/launch_bank_medium.npz}"
OUTPUT_HARD="${OUTPUT_HARD:-${OUTPUT_DIR}/launch_bank_hard.npz}"

parse_range() {
  local range="$1"
  local -n out_min="$2"
  local -n out_max="$3"
  read -r out_min out_max <<< "${range}"
}

generate_bank() {
  local name="$1"
  local output_path="$2"
  local num_samples="$3"
  local strike_x_range="$4"
  local strike_y_range="$5"
  local in_bounce_x_range="$6"
  local in_bounce_y_range="$7"

  local LAUNCHER_X_MIN LAUNCHER_X_MAX LAUNCHER_Y_MIN LAUNCHER_Y_MAX
  local STRIKE_X_MIN STRIKE_X_MAX STRIKE_Y_MIN STRIKE_Y_MAX
  local IN_BOUNCE_X_MIN IN_BOUNCE_X_MAX IN_BOUNCE_Y_MIN IN_BOUNCE_Y_MAX
  local FLIGHT_T_MIN FLIGHT_T_MAX LAUNCH_SPEED_MIN LAUNCH_SPEED_MAX

  parse_range "${LAUNCHER_X_RANGE}" LAUNCHER_X_MIN LAUNCHER_X_MAX
  parse_range "${LAUNCHER_Y_RANGE}" LAUNCHER_Y_MIN LAUNCHER_Y_MAX
  parse_range "${strike_x_range}" STRIKE_X_MIN STRIKE_X_MAX
  parse_range "${strike_y_range}" STRIKE_Y_MIN STRIKE_Y_MAX
  parse_range "${in_bounce_x_range}" IN_BOUNCE_X_MIN IN_BOUNCE_X_MAX
  parse_range "${in_bounce_y_range}" IN_BOUNCE_Y_MIN IN_BOUNCE_Y_MAX
  parse_range "${FLIGHT_T_RANGE}" FLIGHT_T_MIN FLIGHT_T_MAX
  parse_range "${LAUNCH_SPEED_RANGE}" LAUNCH_SPEED_MIN LAUNCH_SPEED_MAX

  mkdir -p "$(dirname "${output_path}")"
  echo "[INFO] [${name}] samples=${num_samples} output=${output_path}"
  echo "[INFO] [${name}] strike_x=[${STRIKE_X_MIN}, ${STRIKE_X_MAX}] strike_y=[${STRIKE_Y_MIN}, ${STRIKE_Y_MAX}]"
  echo "[INFO] [${name}] in_bounce_x=[${IN_BOUNCE_X_MIN}, ${IN_BOUNCE_X_MAX}] in_bounce_y=[${IN_BOUNCE_Y_MIN}, ${IN_BOUNCE_Y_MAX}]"
  echo "[INFO] [${name}] rebound_filter=${REBOUND_FILTER}"


  PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/generate_tennis_launch_bank.py \
    --task "${TASK}" \
    --exp "${EXP}" \
    --num-envs "${NUM_ENVS}" \
    --device "${DEVICE}" \
    --num-samples "${num_samples}" \
    --batch-size "${BATCH_SIZE}" \
    --launcher-x-range "${LAUNCHER_X_MIN}" "${LAUNCHER_X_MAX}" \
    --launcher-y-range "${LAUNCHER_Y_MIN}" "${LAUNCHER_Y_MAX}" \
    --strike-x-range "${STRIKE_X_MIN}" "${STRIKE_X_MAX}" \
    --strike-y-range "${STRIKE_Y_MIN}" "${STRIKE_Y_MAX}" \
    --incoming-bounce-x-range "${IN_BOUNCE_X_MIN}" "${IN_BOUNCE_X_MAX}" \
    --incoming-bounce-y-range "${IN_BOUNCE_Y_MIN}" "${IN_BOUNCE_Y_MAX}" \
    --flight-t-range "${FLIGHT_T_MIN}" "${FLIGHT_T_MAX}" \
    --launch-speed-range "${LAUNCH_SPEED_MIN}" "${LAUNCH_SPEED_MAX}" \
    --output "${output_path}"
}

if [[ "${MODE}" == "single" ]]; then
  echo "[INFO] MODE=single"
  generate_bank "single" "${OUTPUT}" "${NUM_SAMPLES}" \
    "${STRIKE_X_RANGE}" "${STRIKE_Y_RANGE}" \
    "${INCOMING_BOUNCE_X_RANGE}" "${INCOMING_BOUNCE_Y_RANGE}"
  exit 0
fi

echo "[INFO] MODE=three_subset"
echo "[INFO] task=${TASK} exp=${EXP} output_dir=${OUTPUT_DIR}"
echo "[INFO] split: easy=${NUM_SAMPLES_EASY} medium=${NUM_SAMPLES_MEDIUM} hard=${NUM_SAMPLES_HARD}"

generate_bank "easy" "${OUTPUT_EASY}" "${NUM_SAMPLES_EASY}" \
  "${EASY_STRIKE_X_RANGE}" "${EASY_STRIKE_Y_RANGE}" \
  "${EASY_INCOMING_BOUNCE_X_RANGE}" "${EASY_INCOMING_BOUNCE_Y_RANGE}"

generate_bank "medium" "${OUTPUT_MEDIUM}" "${NUM_SAMPLES_MEDIUM}" \
  "${MEDIUM_STRIKE_X_RANGE}" "${MEDIUM_STRIKE_Y_RANGE}" \
  "${MEDIUM_INCOMING_BOUNCE_X_RANGE}" "${MEDIUM_INCOMING_BOUNCE_Y_RANGE}"

generate_bank "hard" "${OUTPUT_HARD}" "${NUM_SAMPLES_HARD}" \
  "${HARD_STRIKE_X_RANGE}" "${HARD_STRIKE_Y_RANGE}" \
  "${HARD_INCOMING_BOUNCE_X_RANGE}" "${HARD_INCOMING_BOUNCE_Y_RANGE}"

MANIFEST_PATH="${OUTPUT_DIR}/launch_bank_manifest.txt"
cat > "${MANIFEST_PATH}" << EOF
task=${TASK}
exp=${EXP}
easy=${OUTPUT_EASY}
medium=${OUTPUT_MEDIUM}
hard=${OUTPUT_HARD}
EOF

echo "[INFO] Wrote manifest: ${MANIFEST_PATH}"
echo "[INFO] Training config fields:"
echo "  task.command.launch_bank_easy_file=${OUTPUT_EASY}"
echo "  task.command.launch_bank_medium_file=${OUTPUT_MEDIUM}"
echo "  task.command.launch_bank_hard_file=${OUTPUT_HARD}"
