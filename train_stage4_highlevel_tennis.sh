#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NPROC="${NPROC:-4}"
NUM_ENVS="${NUM_ENVS:-4096}"
WANDB_ENTITY="${WANDB_ENTITY:-aoru45}"
WANDB_PROJECT="${WANDB_PROJECT:-gentle_humanoid}"
RUN_NAME="${RUN_NAME:-tennis-highlevel-$(date +%m%d-%H%M)}"
USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-}"
EXP_NAME="${EXP_NAME:-highlevel}"

USE_RACKET_NORM="$(printf '%s' "${USE_RACKET}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${ROBOT_NAME}" ]]; then
  case "${USE_RACKET_NORM}" in
    1|true|yes|y|on)
      ROBOT_NAME="g1_col_full_self_racket"
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

if [[ "${ROBOT_NAME}" != "g1_col_full_self_racket" && "${ROBOT_NAME}" != "g1_racket" ]]; then
  echo "[ERROR] Stage-4 high-level tennis requires a racket-enabled robot."
  echo "        Please use ROBOT_NAME=g1_col_full_self_racket (or g1_racket)."
  exit 1
fi

if [[ ! -f "cfg/exp/${EXP_NAME}.yaml" ]]; then
  echo "[ERROR] Exp config not found: cfg/exp/${EXP_NAME}.yaml"
  exit 1
fi

RUN_NAME_SLUG="$(printf '%s' "${RUN_NAME}" | tr '[:space:]/' '__' | tr -cd '[:alnum:]_.-')"
if [[ -z "${RUN_NAME_SLUG}" ]]; then
  RUN_NAME_SLUG="run"
fi
DEFAULT_HYDRA_RUN_DIR="./outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}-${RUN_NAME_SLUG}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"

# Pulse checkpoint source (recommended to initialize pulse prior/decoder).
#
# 1) Local file path example:
#    STAGE_PULSE_CKPT="/abs/path/to/checkpoint_final.pt"
# 2) WandB run reference example:
#    STAGE_PULSE_CKPT="run:axell-wppr/gentle_humanoid/<run_id>"
STAGE_PULSE_CKPT="${STAGE_PULSE_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-18/01-01-39-tracking-pulse-tennis-0418-0101/wandb/run-20260418_010213-fy4oipc6/files/checkpoint_final.pt}"
# STAGE_PULSE_CKPT="${STAGE_PULSE_CKPT:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-17/23-05-53-tracking-pulse-only-tennis-0417-2305/wandb/run-20260417_230629-nxafmogr/files/checkpoint_final.pt}"
LAUNCH_BANK_FILE="${LAUNCH_BANK_FILE:-/mnt/data/xueaoru/motion_tracking/data/tennis_launch_bank/highlevel_launch_bank.npz}"

if [[ "${STAGE_PULSE_CKPT}" != run:* ]]; then
  if [[ "${STAGE_PULSE_CKPT}" == "/replace/with/your/pulse_checkpoint.pt" ]]; then
    echo "[ERROR] Please set STAGE_PULSE_CKPT to your pulse checkpoint path."
    exit 1
  fi
  if [[ ! -f "${STAGE_PULSE_CKPT}" ]]; then
    echo "[ERROR] Pulse checkpoint not found: ${STAGE_PULSE_CKPT}"
    exit 1
  fi
fi

if [[ ! -f "${LAUNCH_BANK_FILE}" ]]; then
  echo "[ERROR] Launch bank file not found: ${LAUNCH_BANK_FILE}"
  exit 1
fi

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
