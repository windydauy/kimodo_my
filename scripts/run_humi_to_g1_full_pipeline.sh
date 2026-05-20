#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one raw HUMI trajectory JSON:
# 1) Convert HUMI JSON -> ee-pose adapter NPZ
# 2) Convert adapter NPZ -> Kimodo ee-pose constraints JSON
# 3) Compute duration
# 4) Resolve/use prompt and generate G1 motion with the official Kimodo G1 model
# 5) Validate constraints against generated MuJoCo qpos CSV
# 6) (Optional) open viser overlay viewer

INPUT="${1:-hf_humi_raw/proposal/raw_trajectories/recording_000.json}"
if [[ -d "${INPUT}" ]]; then
  JSON_PATH="${INPUT%/}/raw_trajectories/recording_000.json"
elif [[ -f "hf_humi_raw/${INPUT}/raw_trajectories/recording_000.json" ]]; then
  JSON_PATH="hf_humi_raw/${INPUT}/raw_trajectories/recording_000.json"
else
  JSON_PATH="${INPUT}"
fi

ACTION_DIR="$(basename "$(dirname "$(dirname "${JSON_PATH}")")")"
if [[ "$(basename "$(dirname "${JSON_PATH}")")" != "raw_trajectories" ]]; then
  ACTION_DIR="$(basename "$(dirname "${JSON_PATH}")")"
fi
RUN_NAME="${2:-${ACTION_DIR}_$(basename "${JSON_PATH}" .json)}"
KEYFRAME_STEP="${KEYFRAME_STEP:-10}"
TARGET_FPS="${TARGET_FPS:-30}"
SOURCE_FPS="${SOURCE_FPS:-}"
TIMELINE_JSONL="${TIMELINE_JSONL:-}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.35}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
ROOT_CONSTRAINT_MODE="${ROOT_CONSTRAINT_MODE:-xyzyaw}"
DISTILL_CONFIG="${DISTILL_CONFIG:-}"
DISTILL_CKPT="${DISTILL_CKPT:-}"
MAX_GENERATED_FRAMES="${MAX_GENERATED_FRAMES:-300}"
MAX_DURATION_SEC="${MAX_DURATION_SEC:-}"

if [[ "${ROOT_CONSTRAINT_MODE}" != "xyzyaw" && "${ROOT_CONSTRAINT_MODE}" != "root2d" && "${ROOT_CONSTRAINT_MODE}" != "none" ]]; then
  echo "Error: ROOT_CONSTRAINT_MODE must be one of: xyzyaw, root2d, none"
  exit 1
fi

IK_JSON_PATH="${IK_JSON_PATH:-}"
SHOW_IK_GT="${SHOW_IK_GT:-0}"
if [[ "${SHOW_IK_GT}" == "1" && -z "${IK_JSON_PATH}" && "$(basename "$(dirname "${JSON_PATH}")")" == "raw_trajectories" ]]; then
  CANDIDATE_IK_JSON="$(dirname "$(dirname "${JSON_PATH}")")/ik_recomputed/$(basename "${JSON_PATH}")"
  if [[ -f "${CANDIDATE_IK_JSON}" ]]; then
    IK_JSON_PATH="${CANDIDATE_IK_JSON}"
  fi
fi

ACTION_NAME="$(basename "$(dirname "$(dirname "${JSON_PATH}")")" | tr '_-' '  ')"
if [[ "$(basename "$(dirname "${JSON_PATH}")")" != "raw_trajectories" ]]; then
  ACTION_NAME="$(basename "$(dirname "${JSON_PATH}")" | tr '_-' '  ')"
fi
DEFAULT_PROMPT="A person performs a ${ACTION_NAME} motion with natural full-body movement."

if [[ -z "${PROMPT:-}" ]]; then
  if [[ -n "${TIMELINE_JSONL}" && -f "${TIMELINE_JSONL}" ]]; then
    echo "Resolving overview prompt from timeline ..."
    PROMPT="$(PYTHONPATH=. python scripts/resolve_timeline_overview_prompt.py \
      --timeline_jsonl "${TIMELINE_JSONL}" \
      --clip "${JSON_PATH}" \
      --fallback "${DEFAULT_PROMPT}" \
      --print_source)"
  else
    echo "Using fallback prompt derived from HUMI action name."
    PROMPT="${DEFAULT_PROMPT}"
  fi
else
  echo "Using prompt from PROMPT environment override."
fi

mkdir -p "${OUT_DIR}"

ADAPTER_NPZ="${OUT_DIR}/source_ee_pose_adapter.npz"
CONSTRAINTS_JSON="${OUT_DIR}/constraints_ee_pose.json"
GEN_CONSTRAINTS_JSON="${OUT_DIR}/constraints_for_generation.json"
OUTPUT_STEM="${OUT_DIR}/g1_generated"
CSV_PATH="${OUTPUT_STEM}.csv"
XML_PATH="kimodo/assets/skeletons/g1skel34/xml/g1.xml"

echo "[1/6] Convert HUMI JSON to ee-pose adapter NPZ ..."
SOURCE_FPS_FLAGS=()
if [[ -n "${SOURCE_FPS}" ]]; then
  SOURCE_FPS_FLAGS+=(--fps "${SOURCE_FPS}")
fi
TRUNCATE_FLAGS=()
if [[ -n "${MAX_GENERATED_FRAMES}" || -n "${MAX_DURATION_SEC}" ]]; then
MAX_SOURCE_FRAMES=$(python - <<PY
import json
import math
from pathlib import Path
path = Path("${JSON_PATH}")
with path.open("r", encoding="utf-8") as f:
    ep = json.load(f)["episode"]
if "${SOURCE_FPS}":
    src_fps = float("${SOURCE_FPS}")
else:
    ts = [float(x["timestamp"]) for x in ep]
    dt = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    src_fps = float(1.0 / (sum(dt) / len(dt)))
if "${MAX_GENERATED_FRAMES}":
    max_sec = float("${MAX_GENERATED_FRAMES}") / float("${TARGET_FPS}")
else:
    max_sec = float("${MAX_DURATION_SEC}")
# Keep enough source frames so generated frame indices stay within the requested duration.
print(max(1, min(len(ep), int(math.floor(max_sec * src_fps)))))
PY
)
  TRUNCATE_FLAGS+=(--max-source-frames "${MAX_SOURCE_FRAMES}")
  echo "Truncating HUMI source to ${MAX_SOURCE_FRAMES} frames for requested max duration/frames."
fi
PYTHONPATH=. python scripts/humi_json_to_ee_pose_npz.py \
  --input "${JSON_PATH}" \
  --output "${ADAPTER_NPZ}" \
  "${SOURCE_FPS_FLAGS[@]}" \
  "${TRUNCATE_FLAGS[@]}"

echo "[2/6] Convert adapter NPZ to ee-pose constraints JSON ..."
PYTHONPATH=. python scripts/npz_to_ee_pose_constraints.py \
  --input "${ADAPTER_NPZ}" \
  --output "${CONSTRAINTS_JSON}" \
  --target-fps "${TARGET_FPS}" \
  --keyframe-step "${KEYFRAME_STEP}" \
  --root-constraint-mode "${ROOT_CONSTRAINT_MODE}"
cp "${CONSTRAINTS_JSON}" "${GEN_CONSTRAINTS_JSON}"

echo "[3/6] Compute duration from adapter NPZ ..."
DURATION=$(python - <<PY
import numpy as np
d=np.load("${ADAPTER_NPZ}", allow_pickle=True)
if "${MAX_GENERATED_FRAMES}":
    print(float("${MAX_GENERATED_FRAMES}") / float("${TARGET_FPS}"))
else:
    print(float(d["root_global_6d"].shape[0]) / float(d["fps"]))
PY
)
echo "duration_sec=${DURATION}"

if [[ -z "${DIFFUSION_STEPS}" ]]; then
  if [[ -n "${DISTILL_CONFIG}" ]]; then
    DIFFUSION_STEPS=$(python - <<PY
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

echo "[4/6] Generate G1 motion with constraints and true first heading ..."
echo "Prompt: ${PROMPT}"
PYTHONPATH=. python scripts/generate_g1_with_first_heading.py "${PROMPT}" \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --constraints "${GEN_CONSTRAINTS_JSON}" \
  --heading_source_npz "${ADAPTER_NPZ}" \
  "${DISTILL_FLAGS[@]}" \
  --output "${OUTPUT_STEM}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "Error: CSV not found at ${CSV_PATH}"
  exit 1
fi

echo "[5/6] Check constraints on MuJoCo CSV result ..."
PYTHONPATH=. python scripts/check_ee_constraints_mujoco.py \
  --csv "${CSV_PATH}" \
  --constraints "${CONSTRAINTS_JSON}" \
  --xml "${XML_PATH}"

echo "[5.5/6] Check dense HUMI EE tracking on MuJoCo CSV result ..."
PYTHONPATH=. python scripts/check_humi_ee_tracking_mujoco.py \
  --csv "${CSV_PATH}" \
  --humi-json "${JSON_PATH}" \
  --generated-fps "${TARGET_FPS}" \
  --xml "${XML_PATH}"

if [[ "${SKIP_VIEWER}" == "1" ]]; then
  echo "[6/6] Skip viewer (SKIP_VIEWER=1)."
  exit 0
fi

echo "[6/6] Launch viser overlay viewer ..."
IK_VIEWER_FLAGS=()
if [[ -n "${IK_JSON_PATH}" ]]; then
  IK_VIEWER_FLAGS+=(--original-ik-json "${IK_JSON_PATH}")
fi
PYTHONPATH=. python scripts/viser_compare_humi_generated.py \
  --original-json "${JSON_PATH}" \
  "${IK_VIEWER_FLAGS[@]}" \
  --generated-csv "${CSV_PATH}" \
  --generated-npz "${OUTPUT_STEM}.npz" \
  --generated-fps "${TARGET_FPS}" \
  --max-duration-sec "${DURATION}" \
  --port "${VISER_PORT}" \
  --original-color "${ORIGINAL_COLOR}" \
  --generated-color "${GENERATED_COLOR}" \
  --ghost-opacity "${GHOST_OPACITY}"
