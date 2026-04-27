#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK="${TASK:-G1/G1_tennis_highlevel}"
EXP="${EXP:-highlevel}"
DEVICE="${DEVICE:-cuda:0}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-data/tennis_launch_bank/highlevel_subsets}"

EASY_OUT="${OUTPUT_DIR}/launch_bank_easy.npz"
MEDIUM_OUT="${OUTPUT_DIR}/launch_bank_medium.npz"
HARD_OUT="${OUTPUT_DIR}/launch_bank_hard.npz"

mkdir -p "${OUTPUT_DIR}"

# Easy: prioritize learnable near-robot bounces, but allow a small mid-court tail.
./.venv/bin/python scripts/generate_tennis_launch_bank.py \
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

# Medium: cover most of return side, including near-net service-box region.
./.venv/bin/python scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.0 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -1.8 1.8 \
  --strike-y-range -8.6 -2.6 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range -2.8 2.8 \
  --incoming-bounce-y-range -10.2 -2.8 \
  --flight-t-range 0.90 1.35 \
  --launch-speed-range 6.8 16.0 \
  --launch-spin-rps-range -8.0 8.0 \
  --angle-deg-range 8.0 23.0 \
  --min-vz 1.6 \
  --min-forward-speed 3.2 \
  --output "${MEDIUM_OUT}"

# Hard: full legal incoming half-court coverage, including close-to-net bounces.
./.venv/bin/python scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -3.2 3.2 \
  --strike-y-range -7.2 -1.6 \
  --strike-z-range 0.90 1.35 \
  --incoming-bounce-x-range -3.6 3.6 \
  --incoming-bounce-y-range -9.8 -0.8 \
  --flight-t-range 0.75 1.20 \
  --launch-speed-range 7.5 18.0 \
  --launch-spin-rps-range -9.0 9.0 \
  --angle-deg-range 8.0 26.0 \
  --min-vz 1.2 \
  --min-forward-speed 2.8 \
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
