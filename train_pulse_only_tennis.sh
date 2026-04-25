#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tracking-pulse-only-tennis-$(date +%m%d-%H%M)}"
USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-}"
MJ_NCONMAX="${MJ_NCONMAX:-128}"
MJ_NJMAX="${MJ_NJMAX:-512}"
MJ_ITER="${MJ_ITER:-10}"
MJ_LS_ITER="${MJ_LS_ITER:-20}"
MJ_CCD_ITER="${MJ_CCD_ITER:-12}"
MJ_MULTICCD="${MJ_MULTICCD:-false}"

USE_RACKET_NORM="$(printf '%s' "${USE_RACKET}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${ROBOT_NAME}" ]]; then
  case "${USE_RACKET_NORM}" in
    1|true|yes|y|on)
      ROBOT_NAME="g1_col_full_self_racket_noself"
      ;;
    0|false|no|n|off|"")
      ROBOT_NAME="g1_col_full_self"
      ;;
    *)
      echo "[ERROR] Invalid USE_RACKET='${USE_RACKET}', expected 0/1 or true/false."
      exit 1
      ;;
  esac
fi

RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
if [[ -z "${RUN_NAME_SLUG}" ]]; then
  RUN_NAME_SLUG="run"
fi
DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

# Stage-2 adapt checkpoint source.
# This checkpoint already contains the teacher modules inherited from Stage-1
# (`encoder_priv`, `actor_teacher`) as well as the frozen Stage-2 estimator
# (`adapt_module` / `priv_pred`).
# 1) Local file path example:
#    STAGE2_CKPT="/abs/path/to/checkpoint_final.pt"
# 2) WandB run reference example:
#    STAGE2_CKPT="run:axell-wppr/gentle_humanoid/<run_id>"
STAGE2_CKPT="${STAGE2_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-15/19-26-41-tracking-stage1-tennis-0415-1926/wandb/run-20260415_192712-dih8z6j9/files/checkpoint_final.pt}"

required=(
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
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"
echo "[INFO] hydra.run.dir=${HYDRA_RUN_DIR}"
echo "[INFO] dataset=tennis only"
echo "[INFO] mujoco(light): nconmax=${MJ_NCONMAX}, njmax=${MJ_NJMAX}, iter=${MJ_ITER}, ls_iter=${MJ_LS_ITER}, ccd_iter=${MJ_CCD_ITER}, multiccd=${MJ_MULTICCD}"

uv run torchrun --nproc_per_node="${NPROC}" scripts/train.py \
  task=G1/G1_tracking +exp=pulse \
  "task.robot.name=${ROBOT_NAME}" \
  "checkpoint_path=${STAGE2_CKPT}" \
  task.sim.isaac_physics_dt=0.005 \
  task.sim.mujoco_physics_dt=0.005 \
  "++task.sim.nconmax=${MJ_NCONMAX}" \
  "++task.sim.njmax=${MJ_NJMAX}" \
  "++task.sim.mujoco_iterations=${MJ_ITER}" \
  "++task.sim.mujoco_ls_iterations=${MJ_LS_ITER}" \
  "++task.sim.mujoco_ccd_iterations=${MJ_CCD_ITER}" \
  "++task.sim.mujoco_multiccd=${MJ_MULTICCD}" \
  'task.command.dataset.mem_paths=[tennis]' \
  'task.command.dataset.path_weights=[1.0]' \
  "task.command.body_z_terminate_thres=0.35" \
  'task.command.body_z_terminate_patterns=["pelvis",".*ankle_roll.*"]' \
  "task.action.max_delay=2" \
  "algo.pulse_kl_coef_start=2.0e-2" \
  "algo.pulse_kl_coef_end=5.0e-3" \
  "task.num_envs=${NUM_ENVS}" \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  wandb.mode=online \
  +wandb.entity="${WANDB_ENTITY}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=${RUN_NAME}"
