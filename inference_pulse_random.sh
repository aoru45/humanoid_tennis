#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-8}"
PULSE_TEMP="${PULSE_TEMP:-1.0}"
CFG_PATH="${CFG_PATH:-cfg/train.yaml}"
STEP_DT="${STEP_DT:-0.02}"
PHYSICS_DT="${PHYSICS_DT:-0.0005}"
VIEWER_MAX_FPS="${VIEWER_MAX_FPS:-12}"
VIEWER_DEBUG="${VIEWER_DEBUG:-0}"
PLAYBACK_FPS="${PLAYBACK_FPS:-4}"
OFFLINE="${OFFLINE:-0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-3000}"
OFFLINE_ENV_ID="${OFFLINE_ENV_ID:-0}"
OFFLINE_HEADLESS="${OFFLINE_HEADLESS:-1}"
OFFLINE_REPLAY="${OFFLINE_REPLAY:-1}"
OFFLINE_REPLAY_SPEED="${OFFLINE_REPLAY_SPEED:-1.0}"
OFFLINE_REPLAY_VIEWER_MAX_FPS="${OFFLINE_REPLAY_VIEWER_MAX_FPS:-45}"
OFFLINE_RECORD_PATH="${OFFLINE_RECORD_PATH:-./outputs/offline_replay/pulse_random_$(date +%Y%m%d_%H%M%S).npz}"

#CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/pulse_more/checkpoint_38700.pt}"
#CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/data/xueaoru/motion_tracking/outputs/2026-04-23/15-18-37-tracking-pulse-run-tennis-0423-1518/checkpoints/checkpoint_final.pt}"

USE_RACKET="${USE_RACKET:-1}"
ROBOT_NAME="${ROBOT_NAME:-g1_col_full_self_racket}"

echo "[INFO] Launch pulse random inference"
echo "[INFO] checkpoint=${CHECKPOINT_PATH}"
echo "[INFO] cfg=${CFG_PATH}"
echo "[INFO] device=${DEVICE}"
echo "[INFO] num_envs=${NUM_ENVS}"
echo "[INFO] pulse_prior_temp=${PULSE_TEMP}"
echo "[INFO] STEP_DT=${STEP_DT}"
echo "[INFO] physics_dt=${PHYSICS_DT}"
echo "[INFO] viewer_max_fps=${VIEWER_MAX_FPS}"
echo "[INFO] viewer_debug=${VIEWER_DEBUG}"
echo "[INFO] playback_fps=${PLAYBACK_FPS}"
echo "[INFO] offline=${OFFLINE}"
echo "[INFO] offline_steps=${OFFLINE_STEPS}"
echo "[INFO] offline_record_path=${OFFLINE_RECORD_PATH}"
echo "[INFO] robot_name=${ROBOT_NAME} (USE_RACKET=${USE_RACKET})"

cmd=(
  uv run python scripts/inference_pulse_random.py
  --cfg "${CFG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --device "${DEVICE}"
  --num-envs "${NUM_ENVS}"
  --temp "${PULSE_TEMP}"
  --step-dt "${STEP_DT}"
  --physics-dt "${PHYSICS_DT}"
  --viewer-max-fps "${VIEWER_MAX_FPS}"
  --playback-fps "${PLAYBACK_FPS}"
  --robot-name "${ROBOT_NAME}"
)

case "$(printf '%s' "${VIEWER_DEBUG}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    cmd+=(--viewer-debug)
    ;;
esac

case "$(printf '%s' "${OFFLINE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    cmd+=(--offline-record "${OFFLINE_RECORD_PATH}")
    cmd+=(--offline-steps "${OFFLINE_STEPS}")
    cmd+=(--offline-env-id "${OFFLINE_ENV_ID}")
    case "$(printf '%s' "${OFFLINE_HEADLESS}" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|y|on)
        cmd+=(--offline-headless)
        ;;
    esac
    case "$(printf '%s' "${OFFLINE_REPLAY}" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|y|on)
        cmd+=(--offline-replay)
        cmd+=(--offline-replay-speed "${OFFLINE_REPLAY_SPEED}")
        cmd+=(--offline-replay-viewer-max-fps "${OFFLINE_REPLAY_VIEWER_MAX_FPS}")
        ;;
    esac
    ;;
esac

"${cmd[@]}"
