#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tracking-stage2-adapt-tennis-$(date +%m%d-%H%M)}"

# Stage-1 checkpoint source (MUST set this):
# 1) Local file path example:
#    STAGE1_CKPT="/abs/path/to/checkpoint_final.pt"
# 2) WandB run reference example:
#    STAGE1_CKPT="run:axell-wppr/gentle_humanoid/<run_id>"
STAGE1_CKPT="${STAGE1_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-13/23-02-27-G1TRACKING-ppo/wandb/run-20260413_230258-5o9gouvd/files/checkpoint_final.pt}"

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

if [[ "${STAGE1_CKPT}" != run:* ]]; then
  if [[ "${STAGE1_CKPT}" == "/replace/with/your/stage1_checkpoint.pt" ]]; then
    echo "[ERROR] Please set STAGE1_CKPT to your Stage-1 checkpoint path."
    echo "        Example:"
    echo "        STAGE1_CKPT=/abs/path/to/checkpoint_final.pt bash $0"
    exit 1
  fi
  if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "[ERROR] Stage-1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
  fi
fi

echo "[INFO] Launch stage-2 adapt training with ${NPROC} GPUs, num_envs=${NUM_ENVS}"
echo "[INFO] WandB entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}, name=${RUN_NAME}"
echo "[INFO] Stage-1 checkpoint source=${STAGE1_CKPT}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking +exp=adapt \
  "checkpoint_path=${STAGE1_CKPT}" \
  'task.command.dataset.mem_paths=[lafan_all,amass_all,100style,amass_hard,real_vr,tennis]' \
  'task.command.dataset.path_weights=[0.25,0.25,0.25,0.05,0.05,0.15]' \
  "task.num_envs=${NUM_ENVS}" \
  wandb.mode=online \
  +wandb.entity="${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}" 
 
