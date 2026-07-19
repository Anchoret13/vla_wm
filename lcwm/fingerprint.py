"""Environment fingerprint — dropped into every results dir.

Editable installs drift silently (plan §8); this records exactly what produced a number:
git commits of this repo and the editable lerobot, package versions, GPU/driver, and the
resolved LIBERO paths (which bddl/init/assets actually got used).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from lcwm.libero_paths import REPO_ROOT, ensure_project_libero_config


def _git(repo: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return "?"
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def collect_fingerprint(extra: dict | None = None) -> dict:
    ensure_project_libero_config()
    import torch  # noqa: PLC0415 — keep module import light
    from libero.libero import get_libero_path  # after config is ensured

    import lerobot

    lerobot_repo = Path(lerobot.__file__).resolve().parent.parent.parent  # src/lerobot -> repo
    fp = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "vla_wm": _git(REPO_ROOT),
            "lerobot": _git(lerobot_repo),
        },
        "packages": {
            n: _pkg(n)
            for n in ("lerobot", "hf-libero", "transformers", "robosuite",
                      "mujoco", "numpy", "gymnasium", "torchcodec", "safetensors")
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": (torch.cuda.get_device_capability(0)
                           if torch.cuda.is_available() else None),
        },
        "libero_paths": {
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
            **{k: get_libero_path(k)
               for k in ("benchmark_root", "bddl_files", "init_states", "assets")},
        },
        "env_vars": {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYTORCH_CUDA_ALLOC_CONF")},
    }
    if extra:
        fp.update(extra)
    return fp


def write_fingerprint(out_dir: str | Path, extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fingerprint.json"
    path.write_text(json.dumps(collect_fingerprint(extra), indent=2, default=str))
    return path


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    print(write_fingerprint(out_dir, extra))
