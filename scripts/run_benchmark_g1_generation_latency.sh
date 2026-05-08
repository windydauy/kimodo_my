#!/usr/bin/env bash
set -euo pipefail

# One-command wrapper for scripts/benchmark_g1_generation_latency.py
# Default target: ee + root_xyzyaw (soft constraints, no hard projection).

MODEL="${MODEL:-Kimodo-G1-RP-v1}"
RUN_NAME="${RUN_NAME:-sub10_largebox_000_ee_xyzyaw}"
PROMPT="${PROMPT:-A robot bends down to grasp a box on the ground, adjusts its grip while crouching, and then releases it before standing fully upright.}"
DURATION="${DURATION:-3.98}"
FUTURE_FRAMES="${FUTURE_FRAMES:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-0}"
WINDOW_STRIDE="${WINDOW_STRIDE:-0}"
WINDOW_START="${WINDOW_START:-0}"
NUM_WINDOWS="${NUM_WINDOWS:-0}"
CONSTRAINTS="${CONSTRAINTS:-scripts/pipeline_outputs/${RUN_NAME}/constraints_for_generation.json}"
HEADING_SOURCE_NPZ="${HEADING_SOURCE_NPZ:-custom_motion/robot-object/sub10_largebox_000_original.npz}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-30}"
CFG_TYPE="${CFG_TYPE:-separated}"
CFG_WEIGHT_TEXT="${CFG_WEIGHT_TEXT:-2.0}"
CFG_WEIGHT_CONSTRAINT="${CFG_WEIGHT_CONSTRAINT:-2.0}"
OUTPUT_JSON="${OUTPUT_JSON:-scripts/pipeline_outputs/${RUN_NAME}/benchmark_latency.json}"
DISTILL_CONFIG="${DISTILL_CONFIG:-}"
DISTILL_CKPT="${DISTILL_CKPT:-}"

HF_HOME="${HF_HOME:-./huggingface}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "Error: constraints file not found: ${CONSTRAINTS}" >&2
  exit 1
fi

if [[ ! -f "${HEADING_SOURCE_NPZ}" ]]; then
  echo "Error: heading source npz not found: ${HEADING_SOURCE_NPZ}" >&2
  exit 1
fi

echo "=== G1 Generation Latency Benchmark ==="
echo "MODEL=${MODEL}"
echo "RUN_NAME=${RUN_NAME}"
echo "DURATION=${DURATION}"
echo "FUTURE_FRAMES=${FUTURE_FRAMES}"
echo "WINDOW_SIZE=${WINDOW_SIZE}, WINDOW_STRIDE=${WINDOW_STRIDE}, WINDOW_START=${WINDOW_START}, NUM_WINDOWS=${NUM_WINDOWS}"
echo "CONSTRAINTS=${CONSTRAINTS}"
echo "HEADING_SOURCE_NPZ=${HEADING_SOURCE_NPZ}"
echo "DIFFUSION_STEPS=${DIFFUSION_STEPS}, WARMUP=${WARMUP}, REPEATS=${REPEATS}"
echo "CFG_TYPE=${CFG_TYPE}, CFG_WEIGHT=[${CFG_WEIGHT_TEXT}, ${CFG_WEIGHT_CONSTRAINT}]"
if [[ -n "${DISTILL_CONFIG}" || -n "${DISTILL_CKPT}" ]]; then
  echo "DISTILL_CONFIG=${DISTILL_CONFIG}"
  echo "DISTILL_CKPT=${DISTILL_CKPT}"
fi
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo

DISTILL_FLAGS=()
if [[ -n "${DISTILL_CONFIG}" || -n "${DISTILL_CKPT}" ]]; then
  if [[ -z "${DISTILL_CONFIG}" || -z "${DISTILL_CKPT}" ]]; then
    echo "Error: DISTILL_CONFIG and DISTILL_CKPT must be provided together." >&2
    exit 1
  fi
  DISTILL_FLAGS+=(--distill_config "${DISTILL_CONFIG}")
  DISTILL_FLAGS+=(--distill_ckpt "${DISTILL_CKPT}")
fi

python scripts/benchmark_g1_generation_latency.py \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --duration "${DURATION}" \
  --future_frames "${FUTURE_FRAMES}" \
  --window_size "${WINDOW_SIZE}" \
  --window_stride "${WINDOW_STRIDE}" \
  --window_start "${WINDOW_START}" \
  --num_windows "${NUM_WINDOWS}" \
  --constraints "${CONSTRAINTS}" \
  --heading_source_npz "${HEADING_SOURCE_NPZ}" \
  --diffusion_steps "${DIFFUSION_STEPS}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --cfg_type "${CFG_TYPE}" \
  --cfg_weight "${CFG_WEIGHT_TEXT}" "${CFG_WEIGHT_CONSTRAINT}" \
  "${DISTILL_FLAGS[@]}" \
  --output_json "${OUTPUT_JSON}"
