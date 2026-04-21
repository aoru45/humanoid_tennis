#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tracking-stage1-tennis-$(date +%m%d-%H%M)}"
USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket}"

RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
if [[ -z "${RUN_NAME_SLUG}" ]]; then
  RUN_NAME_SLUG="run"
fi
DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

required=(
  "dataset/lafan_all/meta_motion.json"
  "dataset/amass_all/meta_motion.json"
  "dataset/100style/meta_motion.json"
  "dataset/amass_hard/meta_motion.json"
  "dataset/real_vr/meta_motion.json"
  "dataset/tennis/meta_motion.json"
)

for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing dataset file: ${path}"
    exit 1
  fi
done

echo "[INFO] Launch stage-1 training with ${NPROC} GPUs, num_envs=${NUM_ENVS}"
echo "[INFO] WandB entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}, name=${RUN_NAME}"
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"
echo "[INFO] hydra.run.dir=${HYDRA_RUN_DIR}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking +exp=train \
  "task.robot.name=${ROBOT_NAME}" \
  'task.command.dataset.mem_paths=[lafan_all,amass_all,100style,amass_hard,real_vr,tennis]' \
  'task.command.dataset.path_weights=[0.25,0.25,0.25,0.05,0.05,0.15]' \
  "task.num_envs=${NUM_ENVS}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  wandb.mode=online \
  +wandb.entity="${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}"
 
