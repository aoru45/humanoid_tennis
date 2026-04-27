#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-tracking-pulse-run-tennis-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
STAGE2_CKPT="${STAGE2_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/track_seed/checkpoint_final.pt}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}}"

if [[ "${STAGE2_CKPT}" != run:* ]] && [[ ! -f "${STAGE2_CKPT}" ]]; then
  echo "[ERROR] Stage2 checkpoint not found: ${STAGE2_CKPT}"
  exit 1
fi

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking "+exp=pulse" \
  "task.robot.name=${ROBOT_NAME}" \
  "checkpoint_path=${STAGE2_CKPT}" \
  "task.num_envs=${NUM_ENVS}" \
  "task.sim.isaac_physics_dt=0.005" \
  "task.sim.mujoco_physics_dt=0.005" \
  "task.command.dataset.mem_paths=[run_tennis_subset]" \
  "task.command.dataset.path_weights=[1.0]" \
  "task.command.body_z_terminate_thres=0.35" \
  "task.command.body_z_terminate_patterns=[\"pelvis\",\".*ankle_roll.*\"]" \
  "task.action.max_delay=0" \
  "algo.pulse_kl_coef_start=2.0e-2" \
  "algo.pulse_kl_coef_end=5.0e-3" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "wandb.mode=online" \
  "+wandb.entity=aoru45" \
  "wandb.project=gentle_humanoid" \
  "wandb.name=${RUN_NAME}"
