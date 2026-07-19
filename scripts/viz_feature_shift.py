#!/usr/bin/env python
"""Visualize real-vs-constant-prompt image hidden states (tap-source [DEC] diagnostics).

For frames captured along a policy rollout, compares last-layer image-token features
under three prompts on the SAME frame:
  real   — the task's true instruction
  const  — the constant tap prompt (lcwm.taps.CONSTANT_PROMPT)
  alt    — a semantically different task's instruction (control: is the shift
           content-specific, or just "any language"?)

Outputs (results/viz/feature_shift/):
  shift_t{T}.png    per-camera spatial heatmaps of per-token cosine distance,
                    overlaid on the frame: real-vs-const | real-vs-alt
  pca_t{T}.png      joint-PCA token scatter with const->real displacement arrows
  hist.png          per-token cos-dist distributions across captured steps
  summary.json      scalar stats incl. spatial correlation of the two shift maps
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.chassis import Pi05Runner, make_task_env, make_task_suite  # noqa: E402
from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.taps import CONSTANT_PROMPT  # noqa: E402

SUITE, TASK_ID, ALT_TASK_ID = "libero_10", 0, 2  # t0 living-room vs t2 kitchen/stove
CAPTURE_STEPS = [0, 80, 160]
GRID = 16  # SigLIP 224/14


@torch.no_grad()
def img_tokens(runner, obs, prompt):
    """(n_cams, 256, d) valid image tokens under `prompt` for one obs."""
    pre = prefix_forward(runner.policy, runner._obs_to_policy_batch(obs, prompt))
    h = pre.hidden[0, : pre.n_img_tokens].float()
    valid = pre.pad_masks[0, : pre.n_img_tokens].bool()
    hv = h[valid]
    assert hv.shape[0] % (GRID * GRID) == 0, hv.shape
    return hv.reshape(-1, GRID * GRID, h.shape[-1])  # (n_cams, 256, d)


def cosdist_map(a, b):
    """(256,d),(256,d) -> (16,16) per-token cosine distance."""
    c = 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return c.reshape(GRID, GRID).cpu().numpy()


def main() -> None:
    out_dir = REPO_ROOT / "results" / "viz" / "feature_shift"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Pi05Runner(suite_name=SUITE)
    suite = make_task_suite(SUITE)
    real_desc = suite.get_task(TASK_ID).language
    alt_desc = suite.get_task(ALT_TASK_ID).language
    print(f"real: {real_desc!r}\nconst: {CONSTANT_PROMPT!r}\nalt:  {alt_desc!r}")

    # -- rollout, capturing frames ------------------------------------------
    env = make_task_env(SUITE, TASK_ID)
    obs, _ = env.reset(seed=1000)
    captured = {}
    runner.reset()
    for t in range(max(CAPTURE_STEPS) + 1):
        if t in CAPTURE_STEPS:
            captured[t] = deepcopy(obs)
        action = runner.select_action(obs, real_desc)
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    env.close()
    print(f"captured steps: {sorted(captured)}")

    cams = ["agentview", "wrist"]
    all_maps, summary = {}, {}
    for t, ob in captured.items():
        tok = {p: img_tokens(runner, ob, d)
               for p, d in [("real", real_desc), ("const", CONSTANT_PROMPT), ("alt", alt_desc)]}
        n_cams = tok["real"].shape[0]
        per_cam = []
        for c in range(min(n_cams, 2)):
            m_const = cosdist_map(tok["real"][c], tok["const"][c])
            m_alt = cosdist_map(tok["real"][c], tok["alt"][c])
            corr = float(np.corrcoef(m_const.ravel(), m_alt.ravel())[0, 1])
            per_cam.append({"const": m_const, "alt": m_alt, "corr": corr})
        all_maps[t] = {"per_cam": per_cam, "obs": ob, "tok": tok}
        summary[t] = {
            cams[c]: {
                "mean_cos_dist_const": float(per_cam[c]["const"].mean()),
                "max_cos_dist_const": float(per_cam[c]["const"].max()),
                "mean_cos_dist_alt": float(per_cam[c]["alt"].mean()),
                "max_cos_dist_alt": float(per_cam[c]["alt"].max()),
                "shiftmap_corr_const_vs_alt": per_cam[c]["corr"],
            }
            for c in range(len(per_cam))
        }

    vmax = max(m["per_cam"][c][k].max()
               for m in all_maps.values() for c in range(len(m["per_cam"]))
               for k in ("const", "alt"))

    # -- spatial heatmaps ----------------------------------------------------
    for t, m in all_maps.items():
        n_cams = len(m["per_cam"])
        fig, axes = plt.subplots(n_cams, 3, figsize=(11, 3.6 * n_cams))
        axes = np.atleast_2d(axes)
        frames = [m["obs"]["pixels"]["image"], m["obs"]["pixels"]["image2"]]
        for c in range(n_cams):
            for j, (title, img, heat) in enumerate([
                (f"{cams[c]} frame (t={t})", frames[c], None),
                ("cos dist: real vs CONST", frames[c], m["per_cam"][c]["const"]),
                (f"real vs ALT-instr (r={m['per_cam'][c]['corr']:.2f})",
                 frames[c], m["per_cam"][c]["alt"]),
            ]):
                ax = axes[c, j]
                ax.imshow(img)
                if heat is not None:
                    im = ax.imshow(heat, cmap="magma", alpha=0.55, vmin=0, vmax=vmax,
                                   extent=(0, img.shape[1], img.shape[0], 0),
                                   interpolation="bilinear")
                    fig.colorbar(im, ax=ax, fraction=0.046)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        fig.suptitle(f'real="{real_desc[:60]}"  const="{CONSTANT_PROMPT}"', fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"shift_t{t}.png", dpi=150)
        plt.close(fig)

    # -- PCA displacement (agentview, mid-capture) ---------------------------
    t_mid = sorted(all_maps)[len(all_maps) // 2]
    tok = all_maps[t_mid]["tok"]
    a, b = tok["const"][0], tok["real"][0]  # (256, d) each
    joint = torch.cat([a, b], 0)
    joint = joint - joint.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(joint, q=2)
    pa, pb = (a - a.mean(0)) @ v, (b - b.mean(0)) @ v
    pa, pb = pa.cpu().numpy(), pb.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(*pa.T, s=8, alpha=0.5, label="const prompt")
    ax.scatter(*pb.T, s=8, alpha=0.5, label="real prompt")
    for i in range(0, 256, 4):
        ax.annotate("", xy=pb[i], xytext=pa[i],
                    arrowprops={"arrowstyle": "->", "lw": 0.4, "alpha": 0.35})
    ax.legend()
    ax.set_title(f"agentview tokens, joint PCA (t={t_mid}): const → real displacement")
    fig.tight_layout()
    fig.savefig(out_dir / f"pca_t{t_mid}.png", dpi=150)
    plt.close(fig)

    # -- histograms ----------------------------------------------------------
    fig, axes = plt.subplots(1, len(all_maps), figsize=(4 * len(all_maps), 3.2),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (t, m) in zip(axes, sorted(all_maps.items())):
        for c in range(len(m["per_cam"])):
            ax.hist(m["per_cam"][c]["const"].ravel(), bins=40, alpha=0.55,
                    label=f"{cams[c]} vs const")
            ax.hist(m["per_cam"][c]["alt"].ravel(), bins=40, alpha=0.55,
                    label=f"{cams[c]} vs alt", histtype="step", lw=1.5)
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xlabel("per-token cos dist")
        if t == sorted(all_maps)[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
