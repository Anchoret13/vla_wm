"""Project-local LIBERO path config — hermetic, never touches the shared ~/.libero.

Why: ~/.libero/config.yaml on this box belongs to an older UW-checkout install used by
other projects (env OZ00MS). It points bddl/init files at that checkout and `datasets`
at a directory that no longer exists. hf-libero honors LIBERO_CONFIG_PATH, so we keep
our own config dir inside the repo (gitignored) and point everything at the packaged
hf-libero files inside the active environment. Paths are computed, not hand-written,
so the config regenerates correctly after any env rebuild.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_LIBERO_DIR = REPO_ROOT / ".libero"
# Sibling of the repo dir: .../vla_wm/datasets/libero (large files stay out of the repo)
DATASETS_DIR = REPO_ROOT.parent / "datasets" / "libero"
ASSETS_DIR = Path.home() / ".cache" / "libero" / "assets"  # where hf-libero downloads


def _packaged_libero_root() -> Path:
    """Locate site-packages/libero/libero of the *active* environment without importing it."""
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero not installed in this environment")
    return Path(list(spec.submodule_search_locations)[0]) / "libero"


def ensure_project_libero_config() -> Path:
    """Write (if needed) the project-local config and set LIBERO_CONFIG_PATH.

    Must run before `import libero.libero`. Returns the config dir. Idempotent.
    """
    root = _packaged_libero_root()
    cfg = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "assets": str(ASSETS_DIR),
        "datasets": str(DATASETS_DIR),
    }
    PROJECT_LIBERO_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg_file = PROJECT_LIBERO_DIR / "config.yaml"
    if not cfg_file.exists() or yaml.safe_load(cfg_file.read_text()) != cfg:
        cfg_file.write_text(yaml.safe_dump(cfg))
    os.environ["LIBERO_CONFIG_PATH"] = str(PROJECT_LIBERO_DIR)
    return PROJECT_LIBERO_DIR


if __name__ == "__main__":
    d = ensure_project_libero_config()
    print(d)  # shell scripts capture this for `export LIBERO_CONFIG_PATH`
