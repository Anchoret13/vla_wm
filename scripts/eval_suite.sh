#!/usr/bin/env bash
# Stage-1 per-suite eval (plan §2): ONE suite per invocation (no shared-fate crash),
# explicit --seed, project-local LIBERO config, fingerprint dumped next to eval_info.json.
#
# Usage:  bash scripts/eval_suite.sh <suite> [SEED] [BATCH] [EPS]
#   e.g.  bash scripts/eval_suite.sh libero_spatial 1000 5 10
set -euo pipefail

SUITE="${1:?usage: eval_suite.sh <libero_spatial|libero_object|libero_goal|libero_10|libero_90> [seed] [batch] [eps]}"
SEED="${2:-1000}"
BATCH="${3:-5}"
EPS="${4:-10}"
ENV_NAME="${ENV_NAME:-vf0s}"
POLICY="${POLICY:-lerobot/pi05_libero_finetuned}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/../eval_logs/stage1/${SUITE}_seed${SEED}}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$(python -m lcwm.libero_paths)"
echo ">>> LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"

mkdir -p "$OUT_DIR"

lerobot-eval \
    --output_dir="$OUT_DIR" \
    --seed="$SEED" \
    --env.type=libero \
    --env.task="$SUITE" \
    --eval.batch_size="$BATCH" \
    --eval.n_episodes="$EPS" \
    --policy.path="$POLICY" \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --env.max_parallel_tasks=1

python -m lcwm.fingerprint "$OUT_DIR" \
    "{\"run\": {\"suite\": \"$SUITE\", \"seed\": $SEED, \"batch\": $BATCH, \"eps_per_task\": $EPS, \"policy\": \"$POLICY\"}}"
echo ">>> done: $OUT_DIR"
