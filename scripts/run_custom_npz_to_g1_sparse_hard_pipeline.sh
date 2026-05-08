#!/usr/bin/env bash
set -euo pipefail

# Sparse-keyframe hard-constraint pipeline (no manual prompt input):
# 1) Convert raw custom-motion qpos npz -> ee-pose adapter npz
# 2) Convert adapter NPZ -> sparse Kimodo ee-pose constraints.json
# 3) Resolve overview prompt from timeline (or use PROMPT override)
# 4) Generate with hard projection on sparse constrained frames only
# 5) Validate constraints against MuJoCo qpos CSV
# 6) (Optional) open viser overlay viewer

NPZ_PATH="${1:-custom_motion/robot-object/sub10_largebox_000_original.npz}"
RUN_NAME="${2:-sub10_largebox_000_sparse_hard}"
KEYFRAME_STEP="${KEYFRAME_STEP:-10}"
TARGET_FPS="${TARGET_FPS:-30}"
TIMELINE_JSONL="${TIMELINE_JSONL:-custom_motion/timeline_sub10.jsonl}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.2}"

DEFAULT_PROMPT="A person standing upright leans forward and reaches down with both arms to grasp a large, heavy black crate. The person performs a deep bend at the hips and knees, maintaining a stable stance while lowering their torso to align their hands with the top edges of the object."

if [[ -z "${PROMPT:-}" ]]; then
  echo "Resolving overview prompt from timeline ..."
  PROMPT="$(PYTHONPATH=. python scripts/resolve_timeline_overview_prompt.py \
    --timeline_jsonl "${TIMELINE_JSONL}" \
    --clip "${NPZ_PATH}" \
    --fallback "${DEFAULT_PROMPT}" \
    --print_source)"
else
  echo "Using prompt from PROMPT environment override."
fi

mkdir -p "${OUT_DIR}"

ADAPTER_NPZ="${OUT_DIR}/source_ee_pose_adapter.npz"
CONSTRAINTS_JSON="${OUT_DIR}/constraints_ee_pose.json"
OUTPUT_STEM="${OUT_DIR}/g1_generated"
CSV_PATH="${OUTPUT_STEM}.csv"
XML_PATH="kimodo/assets/skeletons/g1skel34/xml/g1.xml"

echo "[1/6] Convert raw custom-motion NPZ to ee-pose adapter NPZ ..."
PYTHONPATH=. python scripts/custom_motion_qpos_to_ee_pose_npz.py \
  --input "${NPZ_PATH}" \
  --output "${ADAPTER_NPZ}"

echo "[2/6] Convert adapter NPZ to sparse ee-pose constraints JSON ..."
PYTHONPATH=. python scripts/npz_to_ee_pose_constraints.py \
  --input "${ADAPTER_NPZ}" \
  --output "${CONSTRAINTS_JSON}" \
  --target-fps "${TARGET_FPS}" \
  --keyframe-step "${KEYFRAME_STEP}"

echo "[3/6] Compute duration from adapter NPZ ..."
DURATION=$(python - <<PY
import numpy as np
d=np.load("${ADAPTER_NPZ}", allow_pickle=True)
print(float(d["root_global_6d"].shape[0]) / float(d["fps"]))
PY
)
echo "duration_sec=${DURATION}"

echo "[4/6] Generate G1 motion with sparse hard constraints and true first heading ..."
echo "Prompt: ${PROMPT}"
PYTHONPATH=. python scripts/generate_g1_with_first_heading.py "${PROMPT}" \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --constraints "${CONSTRAINTS_JSON}" \
  --heading_source_npz "${NPZ_PATH}" \
  --hard_project_observed_motion \
  --hard_project_prefix_frames 0 \
  --hard_project_release_frames 0 \
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
