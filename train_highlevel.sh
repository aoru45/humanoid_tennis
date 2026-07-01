#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

RUN_NAME="${RUN_NAME:-tennis-highlevel-newxml-pulseonly-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-6144}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket}"
ACTION_MAX_DELAY="${ACTION_MAX_DELAY:-0}"
TOTAL_FRAMES="${TOTAL_FRAMES:-${HIGHLEVEL_TOTAL_FRAMES:-4000000000}}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-flash128}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"

LAUNCH_BANK_DIR="${LAUNCH_BANK_DIR:-${REPO_ROOT}/tennis_launch_bank/highlevel_subsets}"
LAUNCH_BANK_EASY_FILE="${LAUNCH_BANK_EASY_FILE:-${LAUNCH_BANK_DIR}/launch_bank_easy.npz}"
LAUNCH_BANK_MEDIUM_FILE="${LAUNCH_BANK_MEDIUM_FILE:-${LAUNCH_BANK_DIR}/launch_bank_medium.npz}"
LAUNCH_BANK_HARD_FILE="${LAUNCH_BANK_HARD_FILE:-${LAUNCH_BANK_DIR}/launch_bank_hard.npz}"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-}"

# Default to the old high-level checkpoint. With LOAD_HIGHLEVEL_PULSE_ONLY=true,
# only its frozen Stage-2 pulse prior/decoder are loaded; the high-level actor is
# reinitialized and trained from scratch for the current XML.
PULSE_SOURCE_CKPT="${PULSE_SOURCE_CKPT:-${HIGHLEVEL_PULSE_CKPT:-${REPO_ROOT}/highlevel.pt}}"
LOAD_HIGHLEVEL_PULSE_ONLY="${LOAD_HIGHLEVEL_PULSE_ONLY:-true}"

DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

if [[ "${PULSE_SOURCE_CKPT}" != run:* ]] && [[ ! -f "${PULSE_SOURCE_CKPT}" ]]; then
  echo "[ERROR] Pulse source checkpoint not found: ${PULSE_SOURCE_CKPT}"
  exit 1
fi

echo "[INFO] High-level run dir: ${HYDRA_RUN_DIR}"
echo "[INFO] Pulse source checkpoint: ${PULSE_SOURCE_CKPT}"
echo "[INFO] load_highlevel_pulse_only=${LOAD_HIGHLEVEL_PULSE_ONLY}"
echo "[INFO] robot_name=${ROBOT_NAME}; action.max_delay=${ACTION_MAX_DELAY}"
echo "[INFO] launch_bank_dir=${LAUNCH_BANK_DIR}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=highlevel" \
  "task.robot.name=${ROBOT_NAME}" \
  "task.action.max_delay=${ACTION_MAX_DELAY}" \
  "checkpoint_path=${PULSE_SOURCE_CKPT}" \
  "algo.load_highlevel_pulse_only=${LOAD_HIGHLEVEL_PULSE_ONLY}" \
  "task.command.config.launch.bank.file=${LAUNCH_BANK_FILE}" \
  "task.command.config.launch.bank.easy_file=${LAUNCH_BANK_EASY_FILE}" \
  "task.command.config.launch.bank.medium_file=${LAUNCH_BANK_MEDIUM_FILE}" \
  "task.command.config.launch.bank.hard_file=${LAUNCH_BANK_HARD_FILE}" \
  "task.command.config.launch.bank.use_curriculum=true" \
  "save_interval=100" \
  "start_iter=0" \
  "total_frames=${TOTAL_FRAMES}" \
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
