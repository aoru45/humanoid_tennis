#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

RUN_NAME="${RUN_NAME:-tracking-pulse-run-tennis-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-flash128}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE2_CKPT:-${REPO_ROOT}/outputs/2026-06-20/15-15-57-tracking-stage1-tennis-0620-1515/checkpoints/checkpoint_final.pt}}"
if [[ -z "${HYDRA_RUN_DIR:-}" ]]; then
  HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
fi
TORCHRUN_CMD=()
if [[ -x "./.venv/bin/torchrun" ]]; then
  TORCHRUN_CMD=("./.venv/bin/torchrun")
else
  TORCHRUN_CMD=(uv run --no-sync torchrun)
fi

if [[ "${STAGE1_CKPT}" != run:* ]] && [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[ERROR] Stage-1 checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

"${TORCHRUN_CMD[@]}" --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking "+exp=pulse" \
  "task.robot.name=${ROBOT_NAME}" \
  "checkpoint_path=${STAGE1_CKPT}" \
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
  "wandb.mode=${WANDB_MODE}" \
  "+wandb.entity=${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}" \
  "$@"
