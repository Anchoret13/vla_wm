#!/usr/bin/env bash
# Recreate the `vf0s` conda env: PyTorch pi0.5 (LeRobot) + LIBERO.
# Target box: RTX 5090 (sm_120) -> requires cu128 torch.
# WARNING: this removes an existing env named $ENV_NAME (default vf0s) with no backup.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vf0s}"
PY_VER="${PY_VER:-3.12}"
# repo is .../vla_wm/vla_wm ; lerobot lives at the sibling .../vla_wm/lerobot
LEROBOT_DIR="${LEROBOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/lerobot}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo ">>> [1/6] (re)create env '$ENV_NAME' (python $PY_VER)"
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python="$PY_VER" -y

echo ">>> [2/6] torch cu128 (RTX 5090 / sm_120; range matches lerobot pin)"
conda run -n "$ENV_NAME" pip install "torch>=2.7,<2.12" "torchvision>=0.22,<0.27" \
    --index-url https://download.pytorch.org/whl/cu128

echo ">>> [3/6] native cmake<4 (egl_probe build needs it; pip's cmake fails under build isolation)"
conda install -n "$ENV_NAME" -c conda-forge "cmake<4" -y

echo ">>> [4/6] clone lerobot if missing -> $LEROBOT_DIR"
if [ ! -d "$LEROBOT_DIR" ]; then
    git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi

echo ">>> [5/6] install lerobot .[pi,libero]  (activate so native cmake is on PATH)"
conda activate "$ENV_NAME"
pip install -e "${LEROBOT_DIR}[pi,libero]"

echo ">>> [6/6] verify (imports + pi05 + libero + EGL render)"
export MUJOCO_GL=egl
python - <<'PY'
import torch, lerobot
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
import lerobot.envs.libero  # noqa: F401
import mujoco
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size="1 1 1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); r.update_scene(mujoco.MjData(m)); r.render()
assert torch.cuda.is_available(), "CUDA not available"
print("OK | torch", torch.__version__, "| cap", torch.cuda.get_device_capability(),
      "| lerobot", lerobot.__version__)
PY

echo ">>> done.  use with:  conda activate $ENV_NAME"
