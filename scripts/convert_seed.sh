#!/usr/bin/env bash
set -euo pipefail

# Convert seed_refetch/g1 CSV motions to motion_tracking training NPZ.
# Pipeline:
# 1) CSV -> raw NPZ (fast FK, batched + parallel workers)
# 2) raw NPZ -> tracking NPZ
# 3) tracking NPZ -> mem dataset (optional)

usage() {
  cat <<'USAGE'
Usage:
  bash convert_seed.sh [options]

Options:
  --seed-csv-root PATH      Root dir of input CSV files. Default: /mnt/data/xueaoru/seed_refetch/g1/csv
  --converter-repo PATH     unitree_rl_mjlab repo path. Default: /mnt/data/xueaoru/unitree_rl_mjlab
  --raw-npz-dir PATH        Output dir for intermediate raw NPZ. Default: data/seed_g1_raw_npz
  --tracking-npz-dir PATH   Output dir for training-format NPZ. Default: data/seed_g1_tracking_npz
  --mem-dataset-dir PATH    Output dir for mem dataset. Default: dataset/seed_g1
  --input-fps N             Input CSV FPS. Default: 120
  --output-fps N            Output FPS. Default: 50
  --position-scale S        Scale for root xyz when CSV is seed format. Default: 0.01
  --device STR              Fallback device if --device-list not set. Default: cuda:0
  --device-list LIST        Comma-separated devices for stage-1 workers. Example: cuda:0,cuda:1,cpu
  --workers N               Parallel worker count for stage-1 batch conversion. Default: 1
  --progress-sec N          Progress heartbeat interval (seconds). Default: 5
  --stage2-workers N        Parallel worker count for stage-2 raw->tracking. Default: same as --workers
  --stage2-progress-sec N   Stage-2 progress heartbeat interval (seconds). Default: same as --progress-sec
  --stage3-progress-sec N   Stage-3 heartbeat interval while building mem (seconds). Default: same as --progress-sec
  --max-files N             Convert only first N CSV files (0 = all). Default: 0
  --start-index N           Skip first N CSV files before conversion. Default: 0
  --overwrite               Overwrite existing raw/tracking NPZ
  --skip-stage1             Skip stage-1 CSV->raw conversion and reuse existing raw NPZ
  --skip-stage2             Skip stage-2 raw->tracking conversion and reuse existing tracking NPZ
  --no-mem                  Skip mem-dataset build
  --disable-filter          Disable quality filter when building mem-dataset
  --segment-len N           Segment length for mem-dataset. Default: 1000
  --fail-fast               Stop on first worker-level failure
  --dry-run                 Print stage-1 worker commands and exit
  -h, --help                Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

SEED_CSV_ROOT="${SEED_CSV_ROOT:-/mnt/data/xueaoru/seed_refetch/g1/csv}"
CONVERTER_REPO="${CONVERTER_REPO:-/mnt/data/xueaoru/unitree_rl_mjlab}"
RAW_NPZ_DIR="${RAW_NPZ_DIR:-${PROJECT_ROOT}/data/seed_g1_raw_npz}"
TRACKING_NPZ_DIR="${TRACKING_NPZ_DIR:-${PROJECT_ROOT}/data/seed_g1_tracking_npz}"
MEM_DATASET_DIR="${MEM_DATASET_DIR:-${PROJECT_ROOT}/dataset/seed_g1}"
INPUT_FPS="${INPUT_FPS:-120}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
POSITION_SCALE="${POSITION_SCALE:-0.01}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_LIST="${DEVICE_LIST:-}"
WORKERS="${WORKERS:-1}"
PROGRESS_SEC="${PROGRESS_SEC:-5}"
MAX_FILES="${MAX_FILES:-0}"
START_INDEX="${START_INDEX:-0}"
SEGMENT_LEN="${SEGMENT_LEN:-1000}"
STAGE2_WORKERS="${STAGE2_WORKERS:-}"
STAGE2_PROGRESS_SEC="${STAGE2_PROGRESS_SEC:-}"
STAGE3_PROGRESS_SEC="${STAGE3_PROGRESS_SEC:-}"

OVERWRITE=0
BUILD_MEM_DATASET=1
DISABLE_FILTER=0
FAIL_FAST=0
DRY_RUN=0
SKIP_STAGE1=0
SKIP_STAGE2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-csv-root) SEED_CSV_ROOT="$2"; shift 2 ;;
    --converter-repo) CONVERTER_REPO="$2"; shift 2 ;;
    --raw-npz-dir) RAW_NPZ_DIR="$2"; shift 2 ;;
    --tracking-npz-dir) TRACKING_NPZ_DIR="$2"; shift 2 ;;
    --mem-dataset-dir) MEM_DATASET_DIR="$2"; shift 2 ;;
    --input-fps) INPUT_FPS="$2"; shift 2 ;;
    --output-fps) OUTPUT_FPS="$2"; shift 2 ;;
    --position-scale) POSITION_SCALE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --device-list) DEVICE_LIST="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --progress-sec) PROGRESS_SEC="$2"; shift 2 ;;
    --stage2-workers) STAGE2_WORKERS="$2"; shift 2 ;;
    --stage2-progress-sec) STAGE2_PROGRESS_SEC="$2"; shift 2 ;;
    --stage3-progress-sec) STAGE3_PROGRESS_SEC="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --start-index) START_INDEX="$2"; shift 2 ;;
    --segment-len) SEGMENT_LEN="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --skip-stage1) SKIP_STAGE1=1; shift ;;
    --skip-stage2) SKIP_STAGE2=1; shift ;;
    --no-mem) BUILD_MEM_DATASET=0; shift ;;
    --disable-filter) DISABLE_FILTER=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${WORKERS}" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
  echo "[ERROR] --workers must be a positive integer, got: ${WORKERS}"
  exit 1
fi
if ! [[ "${PROGRESS_SEC}" =~ ^[0-9]+$ ]] || (( PROGRESS_SEC < 1 )); then
  echo "[ERROR] --progress-sec must be a positive integer, got: ${PROGRESS_SEC}"
  exit 1
fi
if [[ -z "${STAGE2_WORKERS}" ]]; then
  STAGE2_WORKERS="${WORKERS}"
fi
if [[ -z "${STAGE2_PROGRESS_SEC}" ]]; then
  STAGE2_PROGRESS_SEC="${PROGRESS_SEC}"
fi
if [[ -z "${STAGE3_PROGRESS_SEC}" ]]; then
  STAGE3_PROGRESS_SEC="${PROGRESS_SEC}"
fi
if ! [[ "${STAGE2_WORKERS}" =~ ^[0-9]+$ ]] || (( STAGE2_WORKERS < 1 )); then
  echo "[ERROR] --stage2-workers must be a positive integer, got: ${STAGE2_WORKERS}"
  exit 1
fi
if ! [[ "${STAGE2_PROGRESS_SEC}" =~ ^[0-9]+$ ]] || (( STAGE2_PROGRESS_SEC < 1 )); then
  echo "[ERROR] --stage2-progress-sec must be a positive integer, got: ${STAGE2_PROGRESS_SEC}"
  exit 1
fi
if ! [[ "${STAGE3_PROGRESS_SEC}" =~ ^[0-9]+$ ]] || (( STAGE3_PROGRESS_SEC < 1 )); then
  echo "[ERROR] --stage3-progress-sec must be a positive integer, got: ${STAGE3_PROGRESS_SEC}"
  exit 1
fi
if ! [[ "${MAX_FILES}" =~ ^[0-9]+$ ]] || (( MAX_FILES < 0 )); then
  echo "[ERROR] --max-files must be a non-negative integer, got: ${MAX_FILES}"
  exit 1
fi
if ! [[ "${START_INDEX}" =~ ^[0-9]+$ ]] || (( START_INDEX < 0 )); then
  echo "[ERROR] --start-index must be a non-negative integer, got: ${START_INDEX}"
  exit 1
fi

if [[ -z "${DEVICE_LIST}" ]]; then
  DEVICE_LIST="${DEVICE}"
fi
DEVICE_LIST="$(printf '%s' "${DEVICE_LIST}" | tr -d '[:space:]')"
if [[ -z "${DEVICE_LIST}" ]]; then
  echo "[ERROR] --device-list resolved to empty"
  exit 1
fi

FAST_STAGE1_SCRIPT="${PROJECT_ROOT}/scripts/data_process/convert_seed_csv_batch_fk.py"
TRACKING_CONVERTER_SCRIPT="${PROJECT_ROOT}/scripts/data_process/convert_tennis_to_tracking_dataset.py"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

if [[ ! -d "${SEED_CSV_ROOT}" ]]; then
  echo "[ERROR] seed csv root not found: ${SEED_CSV_ROOT}"
  exit 1
fi
if [[ ! -d "${CONVERTER_REPO}" ]]; then
  echo "[ERROR] converter repo not found: ${CONVERTER_REPO}"
  exit 1
fi
if [[ ! -f "${FAST_STAGE1_SCRIPT}" ]]; then
  echo "[ERROR] stage-1 script not found: ${FAST_STAGE1_SCRIPT}"
  exit 1
fi
if [[ ! -f "${TRACKING_CONVERTER_SCRIPT}" ]]; then
  echo "[ERROR] tracking converter script not found: ${TRACKING_CONVERTER_SCRIPT}"
  exit 1
fi

mkdir -p "${RAW_NPZ_DIR}" "${TRACKING_NPZ_DIR}"
if [[ "${BUILD_MEM_DATASET}" == "1" ]]; then
  mkdir -p "${MEM_DATASET_DIR}"
fi

if [[ "${DEVICE_LIST}" == *cuda* ]]; then
  if ! "${PYTHON_BIN}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "[WARN] CUDA unavailable, fallback to CPU for stage-1."
    DEVICE="cpu"
    DEVICE_LIST="cpu"
  fi
fi

select_worker_device() {
  local worker_idx="$1"
  IFS=',' read -r -a devices <<< "${DEVICE_LIST}"
  local n="${#devices[@]}"
  if (( n <= 0 )); then
    echo "${DEVICE}"
    return 0
  fi
  local pick=$(( worker_idx % n ))
  echo "${devices[$pick]}"
}

mapfile -t CSV_FILES < <(find "${SEED_CSV_ROOT}" -type f -name '*.csv' | LC_ALL=C sort)
TOTAL_FOUND="${#CSV_FILES[@]}"
if (( TOTAL_FOUND == 0 )); then
  echo "[ERROR] no csv files found under: ${SEED_CSV_ROOT}"
  exit 1
fi

if (( START_INDEX > 0 )); then
  if (( START_INDEX >= TOTAL_FOUND )); then
    echo "[ERROR] start-index (${START_INDEX}) >= total files (${TOTAL_FOUND})"
    exit 1
  fi
  CSV_FILES=("${CSV_FILES[@]:START_INDEX}")
fi

if (( MAX_FILES > 0 )) && (( ${#CSV_FILES[@]} > MAX_FILES )); then
  CSV_FILES=("${CSV_FILES[@]:0:MAX_FILES}")
fi

TOTAL_STAGE1="${#CSV_FILES[@]}"
if (( TOTAL_STAGE1 == 0 )); then
  echo "[ERROR] no files selected after start/max filtering."
  exit 1
fi

echo "[INFO] seed csv root     : ${SEED_CSV_ROOT}"
echo "[INFO] converter repo    : ${CONVERTER_REPO}"
echo "[INFO] raw npz dir       : ${RAW_NPZ_DIR}"
echo "[INFO] tracking npz dir  : ${TRACKING_NPZ_DIR}"
echo "[INFO] mem dataset dir   : ${MEM_DATASET_DIR}"
echo "[INFO] input/output fps  : ${INPUT_FPS}/${OUTPUT_FPS}"
echo "[INFO] position scale    : ${POSITION_SCALE}"
echo "[INFO] device list       : ${DEVICE_LIST}"
echo "[INFO] workers           : ${WORKERS}"
echo "[INFO] stage2 workers    : ${STAGE2_WORKERS}"
echo "[INFO] total csv found   : ${TOTAL_FOUND}"
echo "[INFO] csv to process    : ${TOTAL_STAGE1}"

if [[ "${SKIP_STAGE1}" == "1" ]]; then
  echo "[INFO] stage-1/3: skipped by --skip-stage1 (reuse existing raw npz)."
else
  echo "[INFO] stage-1/3: csv->raw npz (fast FK batch)..."
fi

STATUS_FILE="$(mktemp)"
STAGE1_TMP_DIR="$(mktemp -d)"
cleanup_tmp() {
  rm -f "${STATUS_FILE}" || true
  rm -rf "${STAGE1_TMP_DIR}" || true
}
trap cleanup_tmp EXIT

if [[ "${SKIP_STAGE1}" != "1" ]]; then
  ACTIVE_WORKERS="${WORKERS}"
  if (( ACTIVE_WORKERS > TOTAL_STAGE1 )); then
    ACTIVE_WORKERS="${TOTAL_STAGE1}"
  fi

  declare -a CHUNK_FILES=()
  declare -a WORKER_PIDS=()
  declare -a WORKER_LOGS=()
  declare -a WORKER_DEVS=()

  for (( w=0; w<ACTIVE_WORKERS; w++ )); do
    chunk_file="${STAGE1_TMP_DIR}/chunk_${w}.txt"
    : > "${chunk_file}"
    CHUNK_FILES+=("${chunk_file}")
  done

  for i in "${!CSV_FILES[@]}"; do
    w=$(( i % ACTIVE_WORKERS ))
    printf '%s\n' "${CSV_FILES[$i]}" >> "${CHUNK_FILES[$w]}"
  done

  for (( w=0; w<ACTIVE_WORKERS; w++ )); do
    chunk_file="${CHUNK_FILES[$w]}"
    if [[ ! -s "${chunk_file}" ]]; then
      continue
    fi

    worker_device="$(select_worker_device "${w}")"
    worker_log="${STAGE1_TMP_DIR}/worker_${w}.log"
    cmd=(
      "${PYTHON_BIN}" "${FAST_STAGE1_SCRIPT}"
      --csv-list "${chunk_file}"
      --raw-npz-dir "${RAW_NPZ_DIR}"
      --mjlab-repo "${CONVERTER_REPO}"
      --input-fps "${INPUT_FPS}"
      --output-fps "${OUTPUT_FPS}"
      --position-scale "${POSITION_SCALE}"
      --device "${worker_device}"
      --status-file "${STATUS_FILE}"
      --log-every 100
    )
    if [[ "${OVERWRITE}" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    if [[ "${FAIL_FAST}" == "1" ]]; then
      cmd+=(--fail-fast)
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[DRY_RUN] worker=%d device=%s cmd: ' "${w}" "${worker_device}"
      printf '%q ' "${cmd[@]}"
      printf '\n'
    else
      ( "${cmd[@]}" > "${worker_log}" 2>&1 ) &
      WORKER_PIDS+=("$!")
      WORKER_LOGS+=("${worker_log}")
      WORKER_DEVS+=("${worker_device}")
    fi
  done

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] stage-1 commands printed."
    echo "[DRY_RUN] skip stage-2/3."
    exit 0
  fi

  WORKER_COUNT="${#WORKER_PIDS[@]}"
  if (( WORKER_COUNT == 0 )); then
    echo "[ERROR] no stage-1 workers launched."
    exit 1
  fi

  start_ts="$(date +%s)"
  while true; do
    alive=0
    for pid in "${WORKER_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done

    done_now="$(wc -l < "${STATUS_FILE}" 2>/dev/null || echo 0)"
    ok_now="$(awk -F '\t' '$1=="OK"{c++} END{print c+0}' "${STATUS_FILE}")"
    skip_now="$(awk -F '\t' '$1=="SKIP"{c++} END{print c+0}' "${STATUS_FILE}")"
    fail_now="$(awk -F '\t' '$1=="FAIL"{c++} END{print c+0}' "${STATUS_FILE}")"
    elapsed="$(( $(date +%s) - start_ts ))"
    echo "[INFO] stage-1 progress: done=${done_now}/${TOTAL_STAGE1} ok=${ok_now} skip=${skip_now} fail=${fail_now} alive=${alive}/${WORKER_COUNT} elapsed=${elapsed}s"

    if (( alive == 0 )); then
      break
    fi
    sleep "${PROGRESS_SEC}"
  done

  WORKER_FAILED=0
  for i in "${!WORKER_PIDS[@]}"; do
    pid="${WORKER_PIDS[$i]}"
    if ! wait "${pid}"; then
      WORKER_FAILED=1
      echo "[WARN] stage-1 worker failed: idx=${i} device=${WORKER_DEVS[$i]} log=${WORKER_LOGS[$i]}"
      tail -n 40 "${WORKER_LOGS[$i]}" || true
      if [[ "${FAIL_FAST}" == "1" ]]; then
        break
      fi
    fi
  done

  CONVERTED="$(awk -F '\t' '$1=="OK"{c++} END{print c+0}' "${STATUS_FILE}")"
  SKIPPED="$(awk -F '\t' '$1=="SKIP"{c++} END{print c+0}' "${STATUS_FILE}")"
  FAILED="$(awk -F '\t' '$1=="FAIL"{c++} END{print c+0}' "${STATUS_FILE}")"

  echo "[INFO] csv->raw summary: converted=${CONVERTED} skipped=${SKIPPED} failed=${FAILED} worker_failed=${WORKER_FAILED}"
  if (( WORKER_FAILED != 0 )) && [[ "${FAIL_FAST}" == "1" ]]; then
    exit 1
  fi
  if (( FAILED > 0 )) && [[ "${FAIL_FAST}" == "1" ]]; then
    exit 1
  fi
fi

RAW_COUNT="$(find "${RAW_NPZ_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if (( RAW_COUNT == 0 )); then
  echo "[ERROR] no raw npz files available in: ${RAW_NPZ_DIR}"
  exit 1
fi

if [[ "${SKIP_STAGE2}" == "1" ]]; then
  echo "[INFO] stage-2/3: skipped by --skip-stage2 (reuse existing tracking npz)."
else
  USE_STAGE2_INPUT_LIST=1
  if [[ "${SKIP_STAGE1}" == "1" ]] && (( START_INDEX == 0 )) && (( MAX_FILES == 0 )); then
    USE_STAGE2_INPUT_LIST=0
    echo "[INFO] stage-2 prep: full raw dir mode (skip input-list build, raw_count=${RAW_COUNT})."
  fi

  if (( USE_STAGE2_INPUT_LIST == 1 )); then
    STAGE2_LIST_FILE="${STAGE1_TMP_DIR}/stage2_input_list.txt"
    : > "${STAGE2_LIST_FILE}"
    stage2_prep_start="${SECONDS}"
    stage2_prep_last_log="${SECONDS}"
    stage2_prep_total="${#CSV_FILES[@]}"
    stage2_prep_scanned=0
    echo "[INFO] stage-2 prep: building input list from selected CSV files (no existence check)..."
    for csv_path in "${CSV_FILES[@]}"; do
      stage2_prep_scanned=$((stage2_prep_scanned + 1))
      stem="$(basename "${csv_path%.*}")"
      printf '%s\n' "${stem}.npz" >> "${STAGE2_LIST_FILE}"
      if (( SECONDS - stage2_prep_last_log >= STAGE2_PROGRESS_SEC )); then
        elapsed=$((SECONDS - stage2_prep_start))
        echo "[INFO] stage-2 prep progress: scanned=${stage2_prep_scanned}/${stage2_prep_total} elapsed=${elapsed}s"
        stage2_prep_last_log="${SECONDS}"
      fi
    done
    stage2_prep_elapsed=$((SECONDS - stage2_prep_start))
    echo "[INFO] stage-2 prep done: scanned=${stage2_prep_total} elapsed=${stage2_prep_elapsed}s"
  fi

  echo "[INFO] stage-2/3: raw->tracking (batch)..."
  track_cmd=(
    "${PYTHON_BIN}" scripts/data_process/convert_tennis_to_tracking_dataset.py
    --input-dir "${RAW_NPZ_DIR}"
    --output-dir "${TRACKING_NPZ_DIR}"
    --output-fps "${OUTPUT_FPS}"
    --workers "${STAGE2_WORKERS}"
    --progress-sec "${STAGE2_PROGRESS_SEC}"
  )
  if (( USE_STAGE2_INPUT_LIST == 1 )); then
    track_cmd+=(--input-list "${STAGE2_LIST_FILE}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    track_cmd+=(--overwrite)
  fi
  (cd "${PROJECT_ROOT}" && "${track_cmd[@]}")
fi

TRACKING_COUNT="$(find "${TRACKING_NPZ_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"
if (( TRACKING_COUNT == 0 )); then
  echo "[ERROR] no tracking npz files available in: ${TRACKING_NPZ_DIR}"
  exit 1
fi

if [[ "${BUILD_MEM_DATASET}" == "1" ]]; then
  echo "[INFO] stage-3/3: tracking->mem (batch)..."
  stage3_log="${STAGE1_TMP_DIR}/stage3_mem.log"
  if [[ -t 1 ]]; then
    echo "[INFO] stage-3 mode: live tqdm output (interactive terminal)."
    (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" - "${TRACKING_NPZ_DIR}" "${MEM_DATASET_DIR}" "${SEGMENT_LEN}" "${DISABLE_FILTER}" <<'PY'
import sys
from pathlib import Path
from scripts.data_process.convert_tennis_to_tracking_dataset import _build_mem_dataset

tracking_dir = Path(sys.argv[1])
mem_dir = Path(sys.argv[2])
segment_len = int(sys.argv[3])
disable_filter = bool(int(sys.argv[4]))
mem_dir.mkdir(parents=True, exist_ok=True)

_build_mem_dataset(
    converted_root=tracking_dir,
    mem_path=mem_dir,
    segment_len=segment_len,
    disable_filter=disable_filter,
)
print(f"[mem] built at {mem_dir}")
PY
    ) 2>&1 | tee "${stage3_log}"
  else
    echo "[INFO] stage-3 mode: non-interactive heartbeat."
    (
      cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" - "${TRACKING_NPZ_DIR}" "${MEM_DATASET_DIR}" "${SEGMENT_LEN}" "${DISABLE_FILTER}" <<'PY'
import json
import sys
from pathlib import Path
from scripts.data_process.convert_tennis_to_tracking_dataset import _build_mem_dataset

tracking_dir = Path(sys.argv[1])
mem_dir = Path(sys.argv[2])
segment_len = int(sys.argv[3])
disable_filter = bool(int(sys.argv[4]))
mem_dir.mkdir(parents=True, exist_ok=True)

_build_mem_dataset(
    converted_root=tracking_dir,
    mem_path=mem_dir,
    segment_len=segment_len,
    disable_filter=disable_filter,
)
print(f"[mem] built at {mem_dir}")
PY
    ) > "${stage3_log}" 2>&1 &
    stage3_pid=$!
    stage3_start_ts="$(date +%s)"
    while kill -0 "${stage3_pid}" 2>/dev/null; do
      elapsed="$(( $(date +%s) - stage3_start_ts ))"
      echo "[INFO] stage-3 progress: elapsed=${elapsed}s"
      sleep "${STAGE3_PROGRESS_SEC}"
    done
    if ! wait "${stage3_pid}"; then
      echo "[ERROR] stage-3 failed, log: ${stage3_log}"
      tail -n 80 "${stage3_log}" || true
      exit 1
    fi
    cat "${stage3_log}"
  fi
fi

echo "[INFO] done (3-stage batch pipeline)."
echo "[INFO] raw npz      : ${RAW_NPZ_DIR}"
echo "[INFO] tracking npz : ${TRACKING_NPZ_DIR}"
if [[ "${BUILD_MEM_DATASET}" == "1" ]]; then
  echo "[INFO] mem dataset  : ${MEM_DATASET_DIR}"
fi
