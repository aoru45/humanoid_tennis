#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-tracking-stage1-tennis-$(date +%m%d-%H%M)}"
RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket_noself}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENTITY="${WANDB_ENTITY:-flash128}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
if [[ -z "${HYDRA_RUN_DIR:-}" ]]; then
  HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
fi
TORCHRUN_CMD=()
if [[ -x "./.venv/bin/torchrun" ]]; then
  TORCHRUN_CMD=("./.venv/bin/torchrun")
else
  TORCHRUN_CMD=(uv run --no-sync torchrun)
fi

"${TORCHRUN_CMD[@]}" --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking "+exp=train" \
  "task.robot.name=${ROBOT_NAME}" \
  "task.num_envs=${NUM_ENVS}" \
  "task.sim.isaac_physics_dt=0.005" \
  "task.sim.mujoco_physics_dt=0.005" \
  "task.command.dataset.mem_paths=[seed_g1,run_tennis_subset]" \
  "task.command.dataset.path_weights=[0.9,0.1]" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "wandb.mode=${WANDB_MODE}" \
  "+wandb.entity=${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}"
