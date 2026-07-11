# vla_wm · `pi_explore`

Running **PyTorch π0.5 (pi0.5)** on the **LIBERO** simulator via **HuggingFace LeRobot**.
Starting point for VLA + world-model exploration (pivoted away from OpenVLA).

## Status

| Item | State |
|---|---|
| `vf0s` conda env (py3.12) rebuilt as the single torch env | ✅ |
| lerobot 0.6.1 editable at `../lerobot` with `.[pi,libero]` | ✅ |
| torch 2.11.0+cu128 (RTX 5090 / sm_120) | ✅ |
| EGL headless render + pi05 + libero imports | ✅ |
| Smoke eval: `pi05_libero_finetuned` on LIBERO-Spatial task 0 → **100% success** | ✅ ([results/smoke_eval_info.json](results/smoke_eval_info.json)) |
| Full 400-ep eval | ⛔ **blocked** — shared GPU occupied by another job (`rx0`, ~22 GB used, ~9 GB free); batch=10 needs ~23 GB free |

## Layout

```
pi_explore/
├── scripts/
│   ├── setup_env.sh    # recreate the vf0s env from scratch
│   ├── eval_smoke.sh   # 1 task × 1 episode smoke test
│   └── eval_full.sh    # 4 suites × 10 tasks × 10 ep = 400 episodes
├── results/            # small metrics kept in-repo
├── environment.yml     # conda snapshot of vf0s (record; setup_env.sh is source of truth)
└── requirements-freeze.txt
```
External (not in this repo): lerobot source `../lerobot`, eval outputs/videos `../eval_logs/` (gitignored).

## Run

```bash
conda activate vf0s
export MUJOCO_GL=egl
bash scripts/eval_smoke.sh            # smoke
BATCH=10 bash scripts/eval_full.sh    # full reproduction (needs ~23 GB free GPU)
```

Checkpoint used: [`lerobot/pi05_libero_finetuned`](https://huggingface.co/lerobot/pi05_libero_finetuned)
(reproduces LIBERO 97.5% avg per LeRobot docs). It is embodiment-matched to LIBERO's Franka.

## Gotchas learned (why the scripts look the way they do)

- **RTX 5090 = sm_120 needs cu128 torch.** Install from `https://download.pytorch.org/whl/cu128`
  (default PyPI cu126 wheels lack sm_120 kernels).
- **`egl_probe` / `hf-egl-probe` need a *native* cmake.** `conda install -c conda-forge "cmake<4"`,
  **not** `pip install cmake` — pip's cmake is a python-wrapper that fails inside pip build isolation
  (`ModuleNotFoundError: No module named 'cmake'`). System gcc/g++/make + EGL headers must already exist.
- **Always `export MUJOCO_GL=egl`** for headless offscreen rendering.
- **`--policy.compile_model=false`.** The finetuned checkpoint's config has `compile_model=true`
  (`max-autotune`), which torch.compiles the 3.7B model on first run: slow, and floods logs with
  non-fatal Triton `out of resource` warnings on sm_120 (shared-mem limit 101376 B). Eager is
  numerically equivalent and has predictable memory.
- **GPU memory.** Policy ≈ 7.5 GB (bf16). `--eval.batch_size=N` spawns **N async env workers**,
  each ~2–2.7 GB GPU (own CUDA context + MuJoCo EGL). So batch=10 ≈ 23 GB. Size the batch to the
  *free* GPU memory — this box shares one 5090 with other jobs.
