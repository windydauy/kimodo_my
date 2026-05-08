#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-kimodo/distillation/configs/distill_g1_100_to_20.yaml}
RESUME_PATH=${RESUME_PATH:-}
USE_TMUX=${USE_TMUX:-0}
SESSION_NAME=${SESSION_NAME:-distill_g1}
LOG_DIR=${LOG_DIR:-logs}

export HF_HOME=${HF_HOME:-./huggingface}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Keep W&B visualization online unless overridden by user.
export WANDB_MODE=${WANDB_MODE:-online}
# Distillation training uses raw root trajectories by default; disable
# dataset-time root smoothing to avoid unstable native sparse solver paths.
export KIMODO_DISABLE_ROOT_SMOOTH=${KIMODO_DISABLE_ROOT_SMOOTH:-1}
# Extra diagnostics for intermittent native crashes.
export KIMODO_DEBUG_LOG_CSV_LOAD=${KIMODO_DEBUG_LOG_CSV_LOAD:-0}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

CMD=(python scripts/train_distill_g1_100_to_20.py --config "${CONFIG_PATH}")
if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

if [[ "${USE_TMUX}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  TS=$(date +"%Y%m%d_%H%M%S")
  LOG_FILE="${LOG_DIR}/distill_${SESSION_NAME}_${TS}.log"

  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[tmux] session already exists: ${SESSION_NAME}"
    echo "[tmux] attach with: tmux attach -t ${SESSION_NAME}"
    exit 1
  fi

  TMUX_CMD="${CMD[*]} 2>&1 | tee '${LOG_FILE}'"
  tmux new-session -d -s "${SESSION_NAME}" "${TMUX_CMD}"
  echo "[tmux] started session: ${SESSION_NAME}"
  echo "[tmux] log file: ${LOG_FILE}"
  echo "[tmux] attach: tmux attach -t ${SESSION_NAME}"
  echo "[tmux] detach: Ctrl+b then d"
  echo "[tmux] stop: tmux kill-session -t ${SESSION_NAME}"
else
  echo "[distill] running: ${CMD[*]}"
  "${CMD[@]}"
fi
