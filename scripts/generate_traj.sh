#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TASK="${TASK:-G1/G1_tennis_highlevel}"
EXP="${EXP:-highlevel}"
DEVICE="${DEVICE:-cuda:0}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-data/tennis_launch_bank/highlevel_subsets}"

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON_BIN="uv run python"
else
  PYTHON_BIN="python"
fi

EASY_OUT="${OUTPUT_DIR}/launch_bank_easy.npz"
MEDIUM_OUT="${OUTPUT_DIR}/launch_bank_medium.npz"
HARD_OUT="${OUTPUT_DIR}/launch_bank_hard.npz"

mkdir -p "${OUTPUT_DIR}"

# Easy: prioritize learnable near-robot bounces, but allow a small mid-court tail.
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -3.8 3.8 \
  --launcher-y-range 2.0 5.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -1.2 1.2 \
  --strike-y-range -9.6 -6.6 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range -2.0 2.0 \
  --incoming-bounce-y-range -10.8 -5.8 \
  --flight-t-range 1.00 1.50 \
  --launch-speed-range 5.2 12.0 \
  --launch-spin-rps-range -6.0 6.0 \
  --angle-deg-range 8.0 24.0 \
  --min-vz 1.2 \
  --min-forward-speed 2.6 \
  --output "${EASY_OUT}"

# Medium: back-court dominant on robot side.
# Court reference (tennis_court.xml): near service line at y=-6.40, near baseline at y=-11.885.
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.0 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -1.8 1.8 \
  --strike-y-range -9.4 -5.8 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range -2.8 2.8 \
  --incoming-bounce-y-range -10.8 -6.2 \
  --flight-t-range 0.85 1.45 \
  --launch-speed-range 6.8 16.0 \
  --launch-spin-rps-range -8.0 8.0 \
  --angle-deg-range 8.0 23.0 \
  --min-vz 1.2 \
  --min-forward-speed 2.8 \
  --output "${MEDIUM_OUT}"

# Hard: broader than medium but still mostly back-court (avoid front-court near net).
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -3.2 3.2 \
  --strike-y-range -9.6 -5.4 \
  --strike-z-range 0.90 1.35 \
  --incoming-bounce-x-range -3.6 3.6 \
  --incoming-bounce-y-range -11.0 -5.8 \
  --flight-t-range 0.70 1.35 \
  --launch-speed-range 7.5 18.0 \
  --launch-spin-rps-range -9.0 9.0 \
  --angle-deg-range 8.0 26.0 \
  --min-vz 1.0 \
  --min-forward-speed 2.4 \
  --output "${HARD_OUT}"

cat > "${OUTPUT_DIR}/launch_bank_manifest.txt" << EOF
task=${TASK}
exp=${EXP}
easy=${EASY_OUT}
medium=${MEDIUM_OUT}
hard=${HARD_OUT}
EOF

echo "[INFO] Done. Use these in training:"
echo "  task.command.config.launch.bank.easy_file=${EASY_OUT}"
echo "  task.command.config.launch.bank.medium_file=${MEDIUM_OUT}"
echo "  task.command.config.launch.bank.hard_file=${HARD_OUT}"
echo "  task.command.config.launch.bank.use_curriculum=true"
