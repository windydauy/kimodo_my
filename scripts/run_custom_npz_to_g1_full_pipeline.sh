#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for raw custom-motion qpos NPZ:
# 1) Convert raw custom-motion qpos npz -> ee-pose adapter npz
# 2) Convert adapter NPZ -> Kimodo ee-pose constraints.json
# 3) Resolve overview prompt (or use PROMPT override) and generate G1 motion
# 4) Validate constraints against MuJoCo qpos CSV
# 5) (Optional) open viser overlay viewer

NPZ_PATH="${1:-custom_motion/robot-object/sub10_largebox_000_original.npz}"
RUN_NAME="${2:-sub10_largebox_000}"
KEYFRAME_STEP="${KEYFRAME_STEP:-30}"
TARGET_FPS="${TARGET_FPS:-30}"
TIMELINE_JSONL="${TIMELINE_JSONL:-custom_motion/timeline_sub10.jsonl}"
SKIP_VIEWER="${SKIP_VIEWER:-0}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
VISER_PORT="${VISER_PORT:-8080}"
ORIGINAL_COLOR="${ORIGINAL_COLOR:-0,200,0}"
GENERATED_COLOR="${GENERATED_COLOR:-255,120,0}"
GHOST_OPACITY="${GHOST_OPACITY:-0.2}"
HARD_PROJECT_OBSERVED="${HARD_PROJECT_OBSERVED:-0}"
HARD_PROJECT_PREFIX_FRAMES="${HARD_PROJECT_PREFIX_FRAMES:-0}"
HARD_PROJECT_RELEASE_FRAMES="${HARD_PROJECT_RELEASE_FRAMES:-0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-}"
ROOT_CONSTRAINT_MODE="${ROOT_CONSTRAINT_MODE:-}"
INCLUDE_ROOT_XYZYAW="${INCLUDE_ROOT_XYZYAW:-}"
DISTILL_CONFIG="${DISTILL_CONFIG:-}"
DISTILL_CKPT="${DISTILL_CKPT:-}"

if [[ -z "${ROOT_CONSTRAINT_MODE}" ]]; then
  if [[ -n "${INCLUDE_ROOT_XYZYAW}" ]]; then
    if [[ "${INCLUDE_ROOT_XYZYAW}" == "1" ]]; then
      ROOT_CONSTRAINT_MODE="xyzyaw"
    else
      ROOT_CONSTRAINT_MODE="none"
    fi
  else
    ROOT_CONSTRAINT_MODE="xyzyaw"
  fi
fi

if [[ "${ROOT_CONSTRAINT_MODE}" != "xyzyaw" && "${ROOT_CONSTRAINT_MODE}" != "root2d" && "${ROOT_CONSTRAINT_MODE}" != "none" ]]; then
  echo "Error: ROOT_CONSTRAINT_MODE must be one of: xyzyaw, root2d, none"
  exit 1
fi

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
ROOT_PREFIX_JSON="${OUT_DIR}/constraints_root_prefix.json"
GEN_CONSTRAINTS_JSON="${OUT_DIR}/constraints_for_generation.json"
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
  --keyframe-step "${KEYFRAME_STEP}" \
  --root-constraint-mode "${ROOT_CONSTRAINT_MODE}"

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

echo "[4/6] Generate G1 motion with constraints and true first heading ..."
echo "Prompt: ${PROMPT}"
HARD_PROJECT_FLAGS=()
if [[ "${HARD_PROJECT_OBSERVED}" == "1" && "${HARD_PROJECT_PREFIX_FRAMES}" != "0" ]]; then
  HARD_PROJECT_FLAGS+=(--hard_project_observed_motion)
  HARD_PROJECT_FLAGS+=(--hard_project_prefix_frames "${HARD_PROJECT_PREFIX_FRAMES}")
  if [[ "${HARD_PROJECT_RELEASE_FRAMES}" != "0" ]]; then
    HARD_PROJECT_FLAGS+=(--hard_project_release_frames "${HARD_PROJECT_RELEASE_FRAMES}")
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

PYTHONPATH=. python scripts/generate_g1_with_first_heading.py "${PROMPT}" \
  --model Kimodo-G1-RP-v1 \
  --duration "${DURATION}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --constraints "${GEN_CONSTRAINTS_JSON}" \
  --heading_source_npz "${NPZ_PATH}" \
  "${DISTILL_FLAGS[@]}" \
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
