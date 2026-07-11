#!/usr/bin/env bash
# Full LIBERO reproduction: 4 suites x 10 tasks x 10 episodes = 400 episodes.
# Reproduces LeRobot's reported pi0.5 numbers (~97.5% avg).
#
# GPU memory: pi0.5 policy ~7.5 GB (bf16) + ~2-2.7 GB per async env worker
# (own CUDA context + MuJoCo EGL). So BATCH=10 needs ~23 GB FREE. This box shares
# one RTX 5090 with other jobs -- check `nvidia-smi` and lower BATCH if free mem is small.
#   BATCH=10 -> ~23 GB   BATCH=5 -> ~20 GB   BATCH=2 -> ~13 GB   BATCH=1 -> ~10 GB
set -euo pipefail

ENV_NAME="${ENV_NAME:-vf0s}"
BATCH="${BATCH:-10}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_logs/full_pi05_libero}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOM
mkdir -p "$OUT_DIR"

lerobot-eval \
    --output_dir="$OUT_DIR" \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size="$BATCH" \
    --eval.n_episodes=10 \
    --policy.path=lerobot/pi05_libero_finetuned \
    --policy.n_action_steps=10 \
    --policy.compile_model=false \
    --env.max_parallel_tasks=1
