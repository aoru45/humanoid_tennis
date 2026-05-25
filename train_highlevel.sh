#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-tennis-highlevel-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
BOUNDARY_RUN_NAME="${BOUNDARY_RUN_NAME:-${RUN_NAME}-boundary}"
BOUNDARY_RUN_NAME_SLUG="$(printf '%s' "${BOUNDARY_RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
LAUNCH_BANK_DIR="${LAUNCH_BANK_DIR:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_subsets}"
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


PULSE_PTH="/mnt/data/xueaoru/motion_tracking/outputs/2026-04-23/15-18-37-tracking-pulse-run-tennis-0423-1518/checkpoints/checkpoint_final.pt"
# RESUME_PTH="/mnt/data/xueaoru/motion_tracking/outputs/2026-04-27/00-49-30-tennis-highlevel-0427-0049/checkpoints/checkpoint_8400.pt"
# RESUME_PTH="/mnt/data/xueaoru/motion_tracking/outputs/2026-04-29/00-25-16-tennis-highlevel-0429-0025/checkpoints/checkpoint_5700.pt"
# RESUME_PTH="/mnt/data/xueaoru/motion_tracking/outputs/2026-04-29/22-27-45-tennis-highlevel-0429-2227/checkpoints/checkpoint_6700.pt"
RESUME_PTH="/mnt/data/xueaoru/motion_tracking/outputs/2026-05-08/20-44-49-tennis-highlevel-0508-2044/checkpoints/checkpoint_400.pt"

echo "[INFO] Stage-1 run dir: ${HYDRA_RUN_DIR}"
uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=highlevel" \
  "task.robot.name=g1_col_full_self_racket" \
  "checkpoint_path=${RESUME_PTH}" \
  "task.command.config.launch.bank.file=${LAUNCH_BANK_FILE}" \
  "task.command.config.launch.bank.easy_file=${LAUNCH_BANK_EASY_FILE}" \
  "task.command.config.launch.bank.medium_file=${LAUNCH_BANK_MEDIUM_FILE}" \
  "task.command.config.launch.bank.hard_file=${LAUNCH_BANK_HARD_FILE}" \
  "task.command.config.launch.bank.use_curriculum=true" \
  "save_interval=100" \
  "start_iter=0" \
  "total_frames=${STAGE1_TOTAL_FRAMES}" \
  "task.num_envs=6144" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "wandb.mode=online" \
  "+wandb.entity=aoru45" \
  "wandb.project=gentle_humanoid" \
  "wandb.name=${RUN_NAME}" \
  "resume_load_train_state=False" \
  "resume_wandb=False" \
  "resume_load_env=False" \
  "resume_load_rng=False" \
  "task.sim.isaac_physics_dt=0.0005" \
  "task.sim.mujoco_physics_dt=0.0005"
