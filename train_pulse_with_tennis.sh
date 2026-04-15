#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tracking-pulse-tennis-$(date +%m%d-%H%M)}"

# Stage-2 adapt checkpoint source.
# This checkpoint already contains the teacher modules inherited from Stage-1
# (`encoder_priv`, `actor_teacher`) as well as the frozen Stage-2 estimator
# (`adapt_module` / `priv_pred`).
# 1) Local file path example:
#    STAGE2_CKPT="/abs/path/to/checkpoint_final.pt"
# 2) WandB run reference example:
#    STAGE2_CKPT="run:axell-wppr/gentle_humanoid/<run_id>"
STAGE2_CKPT="${STAGE2_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-14/19-55-05-G1TRACKING-ppo/wandb/run-20260414_195537-lgsyzih4/files/checkpoint_final.pt}"

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

if [[ "${STAGE2_CKPT}" != run:* ]]; then
  if [[ "${STAGE2_CKPT}" == "/replace/with/your/stage2_checkpoint.pt" ]]; then
    echo "[ERROR] Please set STAGE2_CKPT to your Stage-2 estimator checkpoint path."
    exit 1
  fi
  if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "[ERROR] Stage-2 estimator checkpoint not found: ${STAGE2_CKPT}"
    exit 1
  fi
fi

echo "[INFO] Launch pulse distillation with ${NPROC} GPUs, num_envs=${NUM_ENVS}"
echo "[INFO] WandB entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}, name=${RUN_NAME}"
echo "[INFO] Stage-2 checkpoint source=${STAGE2_CKPT}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking +exp=pulse \
  "checkpoint_path=${STAGE2_CKPT}" \
  'task.command.dataset.mem_paths=[lafan_all,amass_all,100style,amass_hard,real_vr,tennis]' \
  'task.command.dataset.path_weights=[0.25,0.25,0.25,0.05,0.05,0.15]' \
  "task.command.body_z_terminate_thres=0.35" \
  'task.command.body_z_terminate_patterns=["pelvis",".*ankle_roll.*"]' \
  "task.action.max_delay=0" \
  "algo.pulse_kl_coef_start=2.0e-2" \
  "algo.pulse_kl_coef_end=5.0e-3" \
  "task.num_envs=${NUM_ENVS}" \
  wandb.mode=online \
  +wandb.entity="${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}"
