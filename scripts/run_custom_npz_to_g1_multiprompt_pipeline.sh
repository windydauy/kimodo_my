#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for raw custom-motion qpos NPZ with multi-prompt sequencing:
# 1) Convert raw custom-motion qpos npz -> ee-pose adapter npz
# 2) Convert adapter NPZ -> Kimodo ee-pose constraints.json
# 3) Build prompt segments from timeline events and generate with true first heading
# 4) Validate constraints against MuJoCo qpos CSV
# 5) (Optional) open viser overlay viewer

NPZ_PATH="${1:-custom_motion/robot-object/sub10_largebox_000_original.npz}"
RUN_NAME="${2:-sub10_largebox_000_multiprompt}"
KEYFRAME_STEP="${KEYFRAME_STEP:-30}"
TARGET_FPS="${TARGET_FPS:-30}"
TIMELINE_JSONL="${TIMELINE_JSONL:-custom_motion/timeline_sub10.jsonl}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.2}"
NUM_TRANSITION_FRAMES="${NUM_TRANSITION_FRAMES:-5}"
SHARE_TRANSITION="${SHARE_TRANSITION:-1}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
HARD_PROJECT_OBSERVED="${HARD_PROJECT_OBSERVED:-0}"
HARD_PROJECT_PREFIX_FRAMES="${HARD_PROJECT_PREFIX_FRAMES:-0}"
HARD_PROJECT_RELEASE_FRAMES="${HARD_PROJECT_RELEASE_FRAMES:-0}"

mkdir -p "${OUT_DIR}"

ADAPTER_NPZ="${OUT_DIR}/source_ee_pose_adapter.npz"
CONSTRAINTS_JSON="${OUT_DIR}/constraints_ee_pose.json"
ROOT_PREFIX_JSON="${OUT_DIR}/constraints_root_prefix.json"
GEN_CONSTRAINTS_JSON="${OUT_DIR}/constraints_for_generation.json"
SEGMENTS_JSON="${OUT_DIR}/segments.json"
OUTPUT_STEM="${OUT_DIR}/g1_generated"
CSV_PATH="${OUTPUT_STEM}.csv"
XML_PATH="kimodo/assets/skeletons/g1skel34/xml/g1.xml"

echo "[1/6] Convert raw custom-motion NPZ to ee-pose adapter NPZ ..."
PYTHONPATH=. python scripts/custom_motion_qpos_to_ee_pose_npz.py \
  --input "${NPZ_PATH}" \
  --output "${ADAPTER_NPZ}"

echo "[2/6] Convert adapter NPZ to ee-pose constraints JSON ..."
PYTHONPATH=. python scripts/npz_to_ee_pose_constraints.py \
  --input "${ADAPTER_NPZ}" \
  --output "${CONSTRAINTS_JSON}" \
  --target-fps "${TARGET_FPS}" \
  --keyframe-step "${KEYFRAME_STEP}"

# Default generation constraints are ee-pose only.
cp "${CONSTRAINTS_JSON}" "${GEN_CONSTRAINTS_JSON}"

if [[ "${HARD_PROJECT_OBSERVED}" == "1" && "${HARD_PROJECT_PREFIX_FRAMES}" != "0" ]]; then
  echo "[2.5/6] Build dense root prefix constraints and merge for hard projection ..."
  PYTHONPATH=. python scripts/custom_motion_qpos_to_root_prefix_constraints.py \
    --input "${NPZ_PATH}" \
    --output "${ROOT_PREFIX_JSON}" \
    --target-fps "${TARGET_FPS}" \
    --prefix-frames "${HARD_PROJECT_PREFIX_FRAMES}"

  python - <<PY
import json
from pathlib import Path
ee = Path("${CONSTRAINTS_JSON}")
root = Path("${ROOT_PREFIX_JSON}")
out = Path("${GEN_CONSTRAINTS_JSON}")
ee_items = json.loads(ee.read_text(encoding="utf-8"))
root_items = json.loads(root.read_text(encoding="utf-8"))
out.write_text(json.dumps(ee_items + root_items, indent=2), encoding="utf-8")
print(f"Merged generation constraints: {out}")
PY
fi

echo "[3/6] Compute duration from adapter NPZ ..."
DURATION=$(python - <<PY
import numpy as np
d=np.load("${ADAPTER_NPZ}", allow_pickle=True)
print(float(d["root_global_6d"].shape[0]) / float(d["fps"]))
PY
)
echo "duration_sec=${DURATION}"

echo "[4/6] Generate G1 motion with multi-prompt sequencing and true first heading ..."
SHARE_FLAG="--share_transition"
if [[ "${SHARE_TRANSITION}" == "0" ]]; then
  SHARE_FLAG="--no-share-transition"
fi
HARD_PROJECT_FLAGS=()
if [[ "${HARD_PROJECT_OBSERVED}" == "1" && "${HARD_PROJECT_PREFIX_FRAMES}" != "0" ]]; then
  HARD_PROJECT_FLAGS+=(--hard_project_observed_motion)
  HARD_PROJECT_FLAGS+=(--hard_project_prefix_frames "${HARD_PROJECT_PREFIX_FRAMES}")
  if [[ "${HARD_PROJECT_RELEASE_FRAMES}" != "0" ]]; then
    HARD_PROJECT_FLAGS+=(--hard_project_release_frames "${HARD_PROJECT_RELEASE_FRAMES}")
  fi
fi

PYTHONPATH=. python scripts/generate_g1_with_first_heading_multiprompt.py \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --num_transition_frames "${NUM_TRANSITION_FRAMES}" \
  "${SHARE_FLAG}" \
  --timeline_jsonl "${TIMELINE_JSONL}" \
  --constraints "${GEN_CONSTRAINTS_JSON}" \
  --heading_source_npz "${NPZ_PATH}" \
  --save_segments_json "${SEGMENTS_JSON}" \
  "${HARD_PROJECT_FLAGS[@]}" \
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

if [[ "${SKIP_VIEWER}" == "1" ]]; then
  echo "[6/6] Skip viewer (SKIP_VIEWER=1)."
  exit 0
fi

echo "[6/6] Launch viser overlay viewer ..."
PYTHONPATH=. python scripts/viser_compare_custom_motion_generated.py \
  --original-npz "${NPZ_PATH}" \
  --generated-csv "${CSV_PATH}" \
  --generated-npz "${OUTPUT_STEM}.npz" \
  --generated-fps "${TARGET_FPS}" \
  --port "${VISER_PORT}" \
  --original-color "${ORIGINAL_COLOR}" \
  --generated-color "${GENERATED_COLOR}" \
  --ghost-opacity "${GHOST_OPACITY}"
