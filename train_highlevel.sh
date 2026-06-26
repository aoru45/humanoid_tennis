#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

RUN_NAME="${RUN_NAME:-tennis-highlevel-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
BOUNDARY_RUN_NAME="${BOUNDARY_RUN_NAME:-${RUN_NAME}-boundary}"
BOUNDARY_RUN_NAME_SLUG="$(printf '%s' "${BOUNDARY_RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
NPROC="${NPROC:-2}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-flash128}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
LAUNCH_BANK_DIR="${LAUNCH_BANK_DIR:-${REPO_ROOT}/tennis_launch_bank/highlevel_subsets}"
LAUNCH_BANK_EASY_FILE="${LAUNCH_BANK_EASY_FILE:-${LAUNCH_BANK_DIR}/launch_bank_easy.npz}"
LAUNCH_BANK_MEDIUM_FILE="${LAUNCH_BANK_MEDIUM_FILE:-${LAUNCH_BANK_DIR}/launch_bank_medium.npz}"
LAUNCH_BANK_HARD_FILE="${LAUNCH_BANK_HARD_FILE:-${LAUNCH_BANK_DIR}/launch_bank_hard.npz}"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-}"  # optional single-bank fallback
MJ_CCD_ITER="${MJ_CCD_ITER:-128}"
MJ_SOLVER_BUDGET="${MJ_SOLVER_BUDGET:-160000000}"
ENABLE_REACHABLE_BOUNDARY_STAGE="${ENABLE_REACHABLE_BOUNDARY_STAGE:-1}"
STAGE1_TOTAL_FRAMES="${STAGE1_TOTAL_FRAMES:-4000000000}"
STAGE2_TOTAL_FRAMES="${STAGE2_TOTAL_FRAMES:-1000000000}"

DEFAULT_HYDRA_RUN_DIR="./outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"
DEFAULT_BOUNDARY_HYDRA_RUN_DIR="./outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)-${BOUNDARY_RUN_NAME_SLUG}"
BOUNDARY_HYDRA_RUN_DIR="${BOUNDARY_HYDRA_RUN_DIR:-$DEFAULT_BOUNDARY_HYDRA_RUN_DIR}"


PULSE_CKPT="${PULSE_CKPT:-${PULSE_PTH:-${REPO_ROOT}/outputs/2026-06-23/21-54-25-stage2-pulse-fixedxml-0623-2154/checkpoints/checkpoint_final.pt}}"
HIGHLEVEL_RESUME_CKPT="${HIGHLEVEL_RESUME_CKPT:-${RESUME_PTH:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${HIGHLEVEL_RESUME_CKPT:-${PULSE_CKPT}}}"
TORCHRUN_CMD=()
if [[ -x "./.venv/bin/torchrun" ]]; then
  TORCHRUN_CMD=("./.venv/bin/torchrun")
else
  TORCHRUN_CMD=(uv run --no-sync torchrun)
fi

WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/renym_warp_cache/motion_tracking_tennis}"
WARP_DISABLE_PCH="${WARP_DISABLE_PCH:-1}"
WARP_PER_RANK_CACHE="${WARP_PER_RANK_CACHE:-1}"
mkdir -p "${WARP_CACHE_PATH}"
export WARP_CACHE_PATH WARP_DISABLE_PCH WARP_PER_RANK_CACHE

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH is empty. Set PULSE_CKPT for fresh high-level training, or HIGHLEVEL_RESUME_CKPT to resume."
  exit 1
fi
if [[ "${CHECKPOINT_PATH}" != run:* ]] && [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

echo "[INFO] High-level run dir: ${HYDRA_RUN_DIR}"
echo "[INFO] Warp cache dir: ${WARP_CACHE_PATH}"
echo "[INFO] Warp disable PCH: ${WARP_DISABLE_PCH}; per-rank cache: ${WARP_PER_RANK_CACHE}"
if [[ -n "${HIGHLEVEL_RESUME_CKPT}" ]]; then
  echo "[INFO] Resuming high-level checkpoint: ${CHECKPOINT_PATH}"
else
  echo "[INFO] Initializing high-level policy from Pulse checkpoint: ${CHECKPOINT_PATH}"
fi

"${TORCHRUN_CMD[@]}" --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=highlevel" \
  "task.robot.name=${ROBOT_NAME}" \
  "checkpoint_path=${CHECKPOINT_PATH}" \
  "task.command.config.launch.bank.file=${LAUNCH_BANK_FILE}" \
  "task.command.config.launch.bank.easy_file=${LAUNCH_BANK_EASY_FILE}" \
  "task.command.config.launch.bank.medium_file=${LAUNCH_BANK_MEDIUM_FILE}" \
  "task.command.config.launch.bank.hard_file=${LAUNCH_BANK_HARD_FILE}" \
  "task.command.config.launch.bank.use_curriculum=true" \
  "save_interval=100" \
  "start_iter=0" \
  "total_frames=${STAGE1_TOTAL_FRAMES}" \
  "task.num_envs=${NUM_ENVS}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "wandb.mode=${WANDB_MODE}" \
  "+wandb.entity=${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}" \
  "resume_load_train_state=False" \
  "resume_wandb=False" \
  "resume_load_env=False" \
  "resume_load_rng=False" \
  "task.sim.isaac_physics_dt=0.0005" \
  "task.sim.mujoco_physics_dt=0.0005" \
  "$@"
