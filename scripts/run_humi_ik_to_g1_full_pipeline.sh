#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one HUMI ik_recomputed JSON:
# 1) Extract retargeted five-point poses (default: realized_target)
# 2) Reuse run_humi_to_g1_full_pipeline.sh for random segmenting,
#    ee-pose constraints, generation, checks, and viewer.

INPUT="${1:-hf_humi_raw/proposal/ik_recomputed/recording_000.json}"
RUN_NAME="${2:-$(basename "$(dirname "$(dirname "${INPUT}")")")_ik_$(basename "${INPUT}" .json)}"
IK_POSE_SOURCE="${IK_POSE_SOURCE:-realized_target}"
OUT_DIR="${OUT_DIR:-scripts/pipeline_outputs/${RUN_NAME}}"
EXTRACTED_JSON="${OUT_DIR}/ik_${IK_POSE_SOURCE}_pose.json"

if [[ "$(basename "$(dirname "${INPUT}")")" == "raw_trajectories" ]]; then
  CANDIDATE_IK_JSON="$(dirname "$(dirname "${INPUT}")")/ik_recomputed/$(basename "${INPUT}")"
  if [[ -f "${CANDIDATE_IK_JSON}" ]]; then
    echo "Input is raw_trajectories; using matching ik_recomputed JSON: ${CANDIDATE_IK_JSON}"
    INPUT="${CANDIDATE_IK_JSON}"
  else
    echo "Error: input is raw_trajectories, but matching IK JSON was not found: ${CANDIDATE_IK_JSON}"
    exit 1
  fi
fi

ACTION_NAME="$(basename "$(dirname "$(dirname "${INPUT}")")" | tr '_-' '  ')"
if [[ -z "${PROMPT:-}" ]]; then
  export PROMPT="A person performs a ${ACTION_NAME} motion with natural full-body movement."
fi

mkdir -p "${OUT_DIR}"

echo "[0/6] Extract HUMI IK ${IK_POSE_SOURCE} poses ..."
PYTHONPATH=. python scripts/ik_recomputed_json_to_pose_json.py \
  --input "${INPUT}" \
  --output "${EXTRACTED_JSON}" \
  --pose-source "${IK_POSE_SOURCE}"

export OUT_DIR
export SHOW_IK_GT="${SHOW_IK_GT:-0}"
if [[ "${SHOW_IK_GT}" == "1" ]]; then
  export IK_JSON_PATH="${INPUT}"
fi

bash scripts/run_humi_to_g1_full_pipeline.sh "${EXTRACTED_JSON}" "${RUN_NAME}"
