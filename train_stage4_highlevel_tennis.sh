#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-tennis-highlevel-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_launch_bank.npz}"

DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"


uv run torchrun --nproc_per_node=4 scripts/train.py \
  task=G1/G1_tennis_highlevel "+exp=highlevel" \
  "task.robot.name=g1_col_full_self_racket_noself" \
  "checkpoint_path=/mnt/data/xueaoru/motion_tracking/outputs/2026-04-26/05-23-39-tennis-highlevel-0426-0523/checkpoints/checkpoint_1500.pt" \
  "task.command.launch_bank_file=${LAUNCH_BANK_FILE}" \
  "save_interval=300" \
  "start_iter=0" \
  "task.num_envs=4096" \
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
