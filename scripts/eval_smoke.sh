#!/usr/bin/env bash
# Smoke test: pi05_libero_finetuned on LIBERO-Spatial task 0, 1 episode.
# Proves the full chain (policy load -> env -> rollout -> success/metrics) end to end.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vf0s}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_logs/smoke}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export MUJOCO_GL=egl
mkdir -p "$OUT_DIR"

lerobot-eval \
    --output_dir="$OUT_DIR" \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids='[0]' \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --policy.path=lerobot/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --env.max_parallel_tasks=1
