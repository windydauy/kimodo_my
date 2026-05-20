#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one G1 training CSV clip:
# 1) Convert CSV -> qpos NPZ
# 2) Convert qpos NPZ -> ee-pose adapter NPZ
# 3) Convert adapter NPZ -> constraints JSON
# 4) Resolve overview prompt and generate G1 motion
# 5) Validate constraints on generated CSV
# 6) (Optional) launch viser overlay comparison (original CSV vs generated)

CSV_PATH="${1:-dataset/g1/csv/230418/sub10_largebox_000.csv}"
RUN_NAME="${2:-$(basename "${CSV_PATH}" .csv)}"
KEYFRAME_STEP="${KEYFRAME_STEP:-30}"
SOURCE_FPS="${SOURCE_FPS:-120}"
TARGET_FPS="${TARGET_FPS:-30}"
TIMELINE_JSONL="${TIMELINE_JSONL:-dataset/timelines.jsonl}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.2}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
ROOT_CONSTRAINT_MODE="${ROOT_CONSTRAINT_MODE:-xyzyaw}"
DISTILL_CONFIG="${DISTILL_CONFIG:-}"
DISTILL_CKPT="${DISTILL_CKPT:-}"

if [[ "${ROOT_CONSTRAINT_MODE}" != "xyzyaw" && "${ROOT_CONSTRAINT_MODE}" != "root2d" && "${ROOT_CONSTRAINT_MODE}" != "none" ]]; then
  echo "Error: ROOT_CONSTRAINT_MODE must be one of: xyzyaw, root2d, none"
  exit 1
fi

DEFAULT_PROMPT="A person stands and performs a natural whole-body motion."
if [[ -z "${PROMPT:-}" ]]; then
  echo "Resolving overview prompt from timeline ..."
  PROMPT="$(PYTHONPATH=. python scripts/resolve_timeline_overview_prompt.py \
    --timeline_jsonl "${TIMELINE_JSONL}" \
    --clip "${CSV_PATH}" \
    --fallback "${DEFAULT_PROMPT}" \
    --print_source)"
else
  echo "Using prompt from PROMPT environment override."
fi

mkdir -p "${OUT_DIR}"

SOURCE_QPOS_NPZ="${OUT_DIR}/source_qpos.npz"
ADAPTER_NPZ="${OUT_DIR}/source_ee_pose_adapter.npz"
CONSTRAINTS_JSON="${OUT_DIR}/constraints_ee_pose.json"
GEN_CONSTRAINTS_JSON="${OUT_DIR}/constraints_for_generation.json"
OUTPUT_STEM="${OUT_DIR}/g1_generated"
CSV_OUT="${OUTPUT_STEM}.csv"
XML_PATH="kimodo/assets/skeletons/g1skel34/xml/g1.xml"

echo "[1/6] Convert G1 CSV to qpos NPZ ..."
PYTHONPATH=. python scripts/g1_csv_to_qpos_npz.py \
  --input "${CSV_PATH}" \
  --output "${SOURCE_QPOS_NPZ}" \
  --input-fps "${SOURCE_FPS}"

echo "[2/6] Convert qpos NPZ to ee-pose adapter NPZ ..."
PYTHONPATH=. python scripts/custom_motion_qpos_to_ee_pose_npz.py \
  --input "${SOURCE_QPOS_NPZ}" \
  --output "${ADAPTER_NPZ}"

echo "[3/6] Convert adapter NPZ to ee-pose constraints JSON ..."
PYTHONPATH=. python scripts/npz_to_ee_pose_constraints.py \
  --input "${ADAPTER_NPZ}" \
  --output "${CONSTRAINTS_JSON}" \
  --target-fps "${TARGET_FPS}" \
  --keyframe-step "${KEYFRAME_STEP}" \
  --root-constraint-mode "${ROOT_CONSTRAINT_MODE}"
cp "${CONSTRAINTS_JSON}" "${GEN_CONSTRAINTS_JSON}"

echo "[4/6] Compute duration and generate G1 motion ..."
DURATION=$(python3 - <<PY
import numpy as np
d=np.load("${ADAPTER_NPZ}", allow_pickle=True)
print(float(d["root_global_6d"].shape[0]) / float(d["fps"]))
PY
)
echo "duration_sec=${DURATION}"

if [[ -z "${DIFFUSION_STEPS}" ]]; then
  if [[ -n "${DISTILL_CONFIG}" ]]; then
    DIFFUSION_STEPS=$(python3 - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("${DISTILL_CONFIG}")
print(int(cfg.distillation.student_steps))
PY
)
  else
    DIFFUSION_STEPS=100
  fi
fi

DISTILL_FLAGS=()
if [[ -n "${DISTILL_CONFIG}" || -n "${DISTILL_CKPT}" ]]; then
  if [[ -z "${DISTILL_CONFIG}" || -z "${DISTILL_CKPT}" ]]; then
    echo "Error: DISTILL_CONFIG and DISTILL_CKPT must be provided together."
    exit 1
  fi
  DISTILL_FLAGS+=(--distill_config "${DISTILL_CONFIG}")
  DISTILL_FLAGS+=(--distill_ckpt "${DISTILL_CKPT}")
fi

echo "Prompt: ${PROMPT}"
PYTHONPATH=. python scripts/generate_g1_with_first_heading.py "${PROMPT}" \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --constraints "${GEN_CONSTRAINTS_JSON}" \
  --heading_source_npz "${SOURCE_QPOS_NPZ}" \
  "${DISTILL_FLAGS[@]}" \
  --output "${OUTPUT_STEM}"

if [[ ! -f "${CSV_OUT}" ]]; then
  echo "Error: CSV not found at ${CSV_OUT}"
  exit 1
fi

echo "[5/6] Check constraints on MuJoCo CSV result ..."
PYTHONPATH=. python scripts/check_ee_constraints_mujoco.py \
  --csv "${CSV_OUT}" \
  --constraints "${CONSTRAINTS_JSON}" \
  --xml "${XML_PATH}"

if [[ "${SKIP_VIEWER}" == "1" ]]; then
  echo "[6/6] Skip viewer (SKIP_VIEWER=1)."
  exit 0
fi

echo "[6/6] Launch viser overlay viewer ..."
PYTHONPATH=. python scripts/viser_compare_g1csv_generated.py \
  --original-csv "${CSV_PATH}" \
  --original-fps "${SOURCE_FPS}" \
  --generated-csv "${CSV_OUT}" \
  --generated-npz "${OUTPUT_STEM}.npz" \
  --generated-fps "${TARGET_FPS}" \
  --port "${VISER_PORT}" \
  --original-color "${ORIGINAL_COLOR}" \
  --generated-color "${GENERATED_COLOR}" \
  --ghost-opacity "${GHOST_OPACITY}"

