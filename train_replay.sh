#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-tennis-replay-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_ENVS="${NUM_ENVS:-6144}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-05-03/00-19-54-tennis-replay-0503-0019/checkpoints/checkpoint_3200.pt}"
LAUNCH_BANK_DIR="${LAUNCH_BANK_DIR:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_subsets}"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-}"
LAUNCH_BANK_EASY_FILE="${LAUNCH_BANK_EASY_FILE:-${LAUNCH_BANK_DIR}/launch_bank_easy.npz}"
LAUNCH_BANK_MEDIUM_FILE="${LAUNCH_BANK_MEDIUM_FILE:-${LAUNCH_BANK_DIR}/launch_bank_medium.npz}"
LAUNCH_BANK_HARD_FILE="${LAUNCH_BANK_HARD_FILE:-${LAUNCH_BANK_DIR}/launch_bank_hard.npz}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"

DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=highlevel" \
  "task.robot.name=g1_col_full_self_racket" \
  "checkpoint_path=${CHECKPOINT_PATH}" \
  "+task.disable_randomization=true" \
  "task.command.config.launch.bank.file=${LAUNCH_BANK_FILE}" \
  "task.command.config.launch.bank.easy_file=${LAUNCH_BANK_EASY_FILE}" \
  "task.command.config.launch.bank.medium_file=${LAUNCH_BANK_MEDIUM_FILE}" \
  "task.command.config.launch.bank.hard_file=${LAUNCH_BANK_HARD_FILE}" \
  "task.command.config.launch.bank.use_curriculum=true" \
  "task.command.config.launch.replay.enabled=true" \
  "task.command.config.launch.replay.min_size_to_sample=1024" \
  "task.command.config.launch.replay.mix_prob_start=0.05" \
  "task.command.config.launch.replay.mix_prob_end=0.2" \
  "task.command.config.launch.replay.mix_progress_start=0.0" \
  "task.command.config.launch.replay.mix_progress_end=0.08" \
  "task.num_envs=${NUM_ENVS}" \
  "save_interval=100" \
  "start_iter=0" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "wandb.mode=${WANDB_MODE}" \
  "+wandb.entity=${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}" \
  "resume_load_train_state=False" \
  "resume_wandb=False" \
  "resume_load_env=False" \
  "resume_load_rng=False"
