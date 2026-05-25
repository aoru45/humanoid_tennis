#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TASK="${TASK:-G1/G1_tennis_highlevel}"
EXP="${EXP:-highlevel}"
DEVICE="${DEVICE:-cuda:0}"
NUM_SAMPLES="${NUM_SAMPLES:-10240}"
BATCH_SIZE="${BATCH_SIZE:-10240}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_subsets}"
LOG_FAILURE_REASONS="${LOG_FAILURE_REASONS:-1}"

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
MEDIUM_LEFT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_medium_left.npz"
MEDIUM_RIGHT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_medium_right.npz"
HARD_LEFT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_hard_left.npz"
HARD_RIGHT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_hard_right.npz"
HARD_PLUS_LEFT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_hard_plus_left.npz"
HARD_PLUS_RIGHT_TMP="${OUTPUT_DIR}/.tmp_launch_bank_hard_plus_right.npz"
HARD_PLUS_PERCENT="${HARD_PLUS_PERCENT:-30}"

mkdir -p "${OUTPUT_DIR}"

LOG_ARGS=()
if [[ "${LOG_FAILURE_REASONS}" == "1" ]]; then
  LOG_ARGS+=(--log-failure-reasons)
fi

# Easy: prioritize learnable near-robot bounces, but allow a small mid-court tail.
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  "${LOG_ARGS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -3.8 3.8 \
  --launcher-y-range 2.0 5.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -1.2 1.2 \
  --strike-y-range -9.6 -6.6 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range -2.0 2.0 \
  --incoming-bounce-y-range -10.8 -5.4 \
  --flight-t-range 1.00 1.50 \
  --launch-speed-range 5.0 12.4 \
  --launch-spin-rps-range -7.0 7.0 \
  --angle-deg-range 7.5 24.5 \
  --min-vz 1.0 \
  --min-forward-speed 2.2 \
  --output "${EASY_OUT}"

MID_LEFT_SAMPLES=$(( NUM_SAMPLES / 2 ))
MID_RIGHT_SAMPLES=$(( NUM_SAMPLES - MID_LEFT_SAMPLES ))

# Medium (de-centered): two lateral clusters, avoid center-lane overfit.
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  "${LOG_ARGS[@]}" \
  --num-samples "${MID_LEFT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.0 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -1.8 -0.25 \
  --strike-y-range -9.4 -5.8 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range -2.8 -0.45 \
  --incoming-bounce-y-range -10.8 -5.6 \
  --target-x-range -3.6 -0.7 \
  --target-y-range 8.0 11.0 \
  --flight-t-range 0.85 1.45 \
  --launch-speed-range 6.6 16.2 \
  --launch-spin-rps-range -8.5 8.5 \
  --angle-deg-range 7.5 23.5 \
  --min-vz 1.0 \
  --min-forward-speed 2.4 \
  --output "${MEDIUM_LEFT_TMP}"

${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  "${LOG_ARGS[@]}" \
  --num-samples "${MID_RIGHT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.0 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range 0.25 1.8 \
  --strike-y-range -9.4 -5.8 \
  --strike-z-range 0.95 1.30 \
  --incoming-bounce-x-range 0.45 2.8 \
  --incoming-bounce-y-range -10.8 -5.6 \
  --target-x-range 0.7 3.6 \
  --target-y-range 8.0 11.0 \
  --flight-t-range 0.85 1.45 \
  --launch-speed-range 6.6 16.2 \
  --launch-spin-rps-range -8.5 8.5 \
  --angle-deg-range 7.5 23.5 \
  --min-vz 1.0 \
  --min-forward-speed 2.4 \
  --output "${MEDIUM_RIGHT_TMP}"

OUTPUT_DIR_ENV="${OUTPUT_DIR}" ${PYTHON_BIN} - << 'PY'
import numpy as np
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR_ENV"])
left = root / ".tmp_launch_bank_medium_left.npz"
right = root / ".tmp_launch_bank_medium_right.npz"
out = root / "launch_bank_medium.npz"

dl = np.load(left)
dr = np.load(right)
keys = list(dl.files)
merged = {}
for k in keys:
    a = dl[k]
    b = dr[k]
    if a.ndim == 0 and b.ndim == 0:
        # Scalar metadata (e.g. sim_physics_dt): keep one copy.
        merged[k] = a
    else:
        merged[k] = np.concatenate([a, b], axis=0)
n = merged[keys[0]].shape[0]
perm = np.random.default_rng(42).permutation(n)
for k in keys:
    arr = merged[k]
    if arr.ndim >= 1 and arr.shape[0] == n:
        merged[k] = arr[perm]
np.savez_compressed(out, **merged)
left.unlink(missing_ok=True)
right.unlink(missing_ok=True)
print(f"[INFO] merged medium bank -> {out} (n={n})")
PY

if ! [[ "${HARD_PLUS_PERCENT}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] HARD_PLUS_PERCENT must be integer in [0, 95], got: ${HARD_PLUS_PERCENT}"
  exit 1
fi
if (( HARD_PLUS_PERCENT < 0 || HARD_PLUS_PERCENT > 95 )); then
  echo "[ERROR] HARD_PLUS_PERCENT out of range [0, 95], got: ${HARD_PLUS_PERCENT}"
  exit 1
fi

HARD_PLUS_SAMPLES=$(( NUM_SAMPLES * HARD_PLUS_PERCENT / 100 ))
HARD_BASE_SAMPLES=$(( NUM_SAMPLES - HARD_PLUS_SAMPLES ))
HARD_BASE_LEFT_SAMPLES=$(( HARD_BASE_SAMPLES / 2 ))
HARD_BASE_RIGHT_SAMPLES=$(( HARD_BASE_SAMPLES - HARD_BASE_LEFT_SAMPLES ))
HARD_PLUS_LEFT_SAMPLES=$(( HARD_PLUS_SAMPLES / 2 ))
HARD_PLUS_RIGHT_SAMPLES=$(( HARD_PLUS_SAMPLES - HARD_PLUS_LEFT_SAMPLES ))

# Hard (base): wider lateral clusters, still avoid center dominance.
${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  "${LOG_ARGS[@]}" \
  --num-samples "${HARD_BASE_LEFT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range -3.2 -0.35 \
  --strike-y-range -9.6 -5.4 \
  --strike-z-range 0.90 1.35 \
  --incoming-bounce-x-range -3.6 -0.65 \
  --incoming-bounce-y-range -11.0 -5.4 \
  --target-x-range -4.0 -1.0 \
  --target-y-range 8.1 11.3 \
  --flight-t-range 0.70 1.35 \
  --launch-speed-range 7.2 18.4 \
  --launch-spin-rps-range -10.0 10.0 \
  --angle-deg-range 7.5 26.5 \
  --min-vz 0.9 \
  --min-forward-speed 2.1 \
  --output "${HARD_LEFT_TMP}"

${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
  --task "${TASK}" \
  --exp "${EXP}" \
  --device "${DEVICE}" \
  "${LOG_ARGS[@]}" \
  --num-samples "${HARD_BASE_RIGHT_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --launcher-x-range -4.0 4.0 \
  --launcher-y-range 3.5 7.5 \
  --launcher-z-range 1.6 2.5 \
  --strike-x-range 0.35 3.2 \
  --strike-y-range -9.6 -5.4 \
  --strike-z-range 0.90 1.35 \
  --incoming-bounce-x-range 0.65 3.6 \
  --incoming-bounce-y-range -11.0 -5.4 \
  --target-x-range 1.0 4.0 \
  --target-y-range 8.1 11.3 \
  --flight-t-range 0.70 1.35 \
  --launch-speed-range 7.2 18.4 \
  --launch-spin-rps-range -10.0 10.0 \
  --angle-deg-range 7.5 26.5 \
  --min-vz 0.9 \
  --min-forward-speed 2.1 \
  --output "${HARD_RIGHT_TMP}"

if (( HARD_PLUS_SAMPLES > 0 )); then
  # Hard+ (reachable-boundary): push to wider lateral boundary + shorter reaction window.
  ${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
    --task "${TASK}" \
    --exp "${EXP}" \
    --device "${DEVICE}" \
    "${LOG_ARGS[@]}" \
    --num-samples "${HARD_PLUS_LEFT_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --launcher-x-range -4.1 4.1 \
    --launcher-y-range 3.7 7.8 \
    --launcher-z-range 1.6 2.5 \
    --strike-x-range -3.9 -1.1 \
    --strike-y-range -9.9 -5.0 \
    --strike-z-range 0.90 1.36 \
    --incoming-bounce-x-range -4.1 -1.2 \
    --incoming-bounce-y-range -11.1 -5.1 \
    --target-x-range -4.1 -1.25 \
    --target-y-range 8.3 11.4 \
    --flight-t-range 0.66 1.18 \
    --launch-speed-range 8.0 21.0 \
    --launch-spin-rps-range -12.0 12.0 \
    --angle-deg-range 6.8 27.8 \
    --min-vz 0.8 \
    --min-forward-speed 2.2 \
    --output "${HARD_PLUS_LEFT_TMP}"

  ${PYTHON_BIN} scripts/generate_tennis_launch_bank.py \
    --task "${TASK}" \
    --exp "${EXP}" \
    --device "${DEVICE}" \
    "${LOG_ARGS[@]}" \
    --num-samples "${HARD_PLUS_RIGHT_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --launcher-x-range -4.1 4.1 \
    --launcher-y-range 3.7 7.8 \
    --launcher-z-range 1.6 2.5 \
    --strike-x-range 1.1 3.9 \
    --strike-y-range -9.9 -5.0 \
    --strike-z-range 0.90 1.36 \
    --incoming-bounce-x-range 1.2 4.1 \
    --incoming-bounce-y-range -11.1 -5.1 \
    --target-x-range 1.25 4.1 \
    --target-y-range 8.3 11.4 \
    --flight-t-range 0.66 1.18 \
    --launch-speed-range 8.0 21.0 \
    --launch-spin-rps-range -12.0 12.0 \
    --angle-deg-range 6.8 27.8 \
    --min-vz 0.8 \
    --min-forward-speed 2.2 \
    --output "${HARD_PLUS_RIGHT_TMP}"
fi

OUTPUT_DIR_ENV="${OUTPUT_DIR}" ${PYTHON_BIN} - << 'PY'
import numpy as np
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR_ENV"])
out = root / "launch_bank_hard.npz"
cands = [
    root / ".tmp_launch_bank_hard_left.npz",
    root / ".tmp_launch_bank_hard_right.npz",
    root / ".tmp_launch_bank_hard_plus_left.npz",
    root / ".tmp_launch_bank_hard_plus_right.npz",
]
banks = [np.load(p) for p in cands if p.exists()]
if not banks:
    raise RuntimeError("No hard temp banks found to merge.")

keys = list(banks[0].files)
merged = {}
for k in keys:
    arrays = [b[k] for b in banks]
    if arrays[0].ndim == 0:
        merged[k] = arrays[0]
    else:
        merged[k] = np.concatenate(arrays, axis=0)
n = merged[keys[0]].shape[0]
perm = np.random.default_rng(43).permutation(n)
for k in keys:
    arr = merged[k]
    if arr.ndim >= 1 and arr.shape[0] == n:
        merged[k] = arr[perm]
np.savez_compressed(out, **merged)
for b in banks:
    b.close()
for p in cands:
    p.unlink(missing_ok=True)
print(f"[INFO] merged hard bank(+hard+) -> {out} (n={n})")
PY

cat > "${OUTPUT_DIR}/launch_bank_manifest.txt" << EOF
task=${TASK}
exp=${EXP}
easy=${EASY_OUT}
medium=${MEDIUM_OUT}
hard=${HARD_OUT}
hard_plus_percent=${HARD_PLUS_PERCENT}
EOF

echo "[INFO] Done. Use these in training:"
echo "  task.command.config.launch.bank.easy_file=${EASY_OUT}"
echo "  task.command.config.launch.bank.medium_file=${MEDIUM_OUT}"
echo "  task.command.config.launch.bank.hard_file=${HARD_OUT}"
echo "  task.command.config.launch.bank.use_curriculum=true"
