#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tennis-highlevel-$(date +%m%d-%H%M)}"
USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket}"
EXP_NAME="${EXP_NAME:-highlevel}"

RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
if [[ -z "${RUN_NAME_SLUG}" ]]; then
  RUN_NAME_SLUG="run"
fi
DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

STAGE_PULSE_CKPT="${STAGE_PULSE_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-18/01-01-39-tracking-pulse-tennis-0418-0101/wandb/run-20260418_010213-fy4oipc6/files/checkpoint_final.pt}"
# STAGE_PULSE_CKPT="${STAGE_PULSE_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-17/23-05-53-tracking-pulse-only-tennis-0417-2305/wandb/run-20260417_230629-nxafmogr/files/checkpoint_final.pt}"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_launch_bank.npz}"


echo "[INFO] Launch stage-4 high-level tennis training with ${NPROC} GPUs, num_envs=${NUM_ENVS}"
echo "[INFO] WandB entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}, name=${RUN_NAME}"
echo "[INFO] Pulse checkpoint source=${STAGE_PULSE_CKPT}"
echo "[INFO] launch_bank_file=${LAUNCH_BANK_FILE}"
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"
echo "[INFO] exp=${EXP_NAME}"
echo "[INFO] hydra.run.dir=${HYDRA_RUN_DIR}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=${EXP_NAME}" \
  "task.robot.name=${ROBOT_NAME}" \
  "checkpoint_path=${STAGE_PULSE_CKPT}" \
  "task.command.launch_bank_file=${LAUNCH_BANK_FILE}" \
  "task.num_envs=${NUM_ENVS}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  wandb.mode=online \
  +wandb.entity="${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}"
