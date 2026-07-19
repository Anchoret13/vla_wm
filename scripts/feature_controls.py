#!/usr/bin/env python
"""§11.1 feature/identity controls: same-frame prompt battery (framework_design.md).

On identical frames from a libero_10 t0 rollout, compares last-layer image tokens
under: canonical t0 | 3 paraphrases of t0 | canonical t1 (SAME scene, different
target objects) | unrelated t2 | constant prompt.

Reports (results/feature_controls/):
- quantiles + tail fractions of per-token cos dist (not mean-only)
- shift-DIRECTION cosine matrix: are δ(para), δ(t1), δ(t2), δ(const) collinear?
  computed on tail tokens (top-quartile shift norm) and on all tokens
- corrected joint-PCA (single shared centering) with explained variance
- SigLIP pre-trunk (embed_image) row as the prompt-invariant reference (sanity)
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

SUITE, TASK_ID = "libero_10", 0
CAPTURE_STEPS = [0, 80, 160]
GRID = 16

PARAPHRASES = [
    "place the alphabet soup and the tomato sauce into the basket",
    "pick up the alphabet soup and the tomato sauce and put them in the basket",
    "move the soup can and the tomato sauce can into the basket",
]


@torch.no_grad()
def img_tokens_lastlayer(runner, obs, prompt):
    pre = prefix_forward(runner.policy, runner._obs_to_policy_batch(obs, prompt))
    h = pre.hidden[0, : pre.n_img_tokens].float()
    valid = pre.pad_masks[0, : pre.n_img_tokens].bool()
    return h[valid]  # (n_valid, d)


@torch.no_grad()
def img_tokens_siglip(runner, obs, prompt):
    """SigLIP pre-trunk (embed_image output): prompt-invariant by construction."""
    batch = runner._obs_to_policy_batch(obs, prompt)
    images, img_masks = runner.policy._preprocess_images(batch)
    model = runner.policy.model
    toks = [model.paligemma_with_expert.embed_image(img)[0].float()
            for img, m in zip(images, img_masks) if bool(m[0])]
    return torch.cat(toks, dim=0)


def stats(dists: torch.Tensor) -> dict:
    q = torch.quantile(dists, torch.tensor([0.5, 0.9, 0.99], device=dists.device))
    return {
        "mean": float(dists.mean()), "p50": float(q[0]), "p90": float(q[1]),
        "p99": float(q[2]), "frac_gt_0.3": float((dists > 0.3).float().mean()),
        "frac_gt_0.6": float((dists > 0.6).float().mean()),
    }


def main() -> None:
    out_dir = REPO_ROOT / "results" / "feature_controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Pi05Runner(suite_name=SUITE)
    suite = make_task_suite(SUITE)
    canon = suite.get_task(TASK_ID).language
    t1 = suite.get_task(1).language        # same LIVING_ROOM_SCENE2
    t2 = suite.get_task(2).language        # unrelated KITCHEN_SCENE3
    prompts = {
        "canon": canon,
        **{f"para{i}": p for i, p in enumerate(PARAPHRASES)},
        "t1_same_scene": t1,
        "t2_unrelated": t2,
        "const": CONSTANT_PROMPT,
    }
    print(json.dumps(prompts, indent=1))

    env = make_task_env(SUITE, TASK_ID)
    obs, _ = env.reset(seed=1000)
    captured, t = {}, 0
    runner.reset()
    for t in range(max(CAPTURE_STEPS) + 1):
        if t in CAPTURE_STEPS:
            captured[t] = deepcopy(obs)
        obs, _r, term, trunc, _i = env.step(runner.select_action(obs, canon))
        if term or trunc:
            break
    env.close()

    report: dict = {"prompts": prompts, "steps": {}}
    for t, ob in captured.items():
        h = {name: img_tokens_lastlayer(runner, ob, p) for name, p in prompts.items()}
        # SigLIP sanity: identical regardless of prompt (compute twice, compare)
        s1 = img_tokens_siglip(runner, ob, canon)
        s2 = img_tokens_siglip(runner, ob, t2)
        siglip_dev = float((s1 - s2).abs().max())

        base = h["canon"]
        entries, deltas = {}, {}
        for name in prompts:
            if name == "canon":
                continue
            d = 1 - torch.nn.functional.cosine_similarity(base, h[name], dim=-1)
            entries[name] = stats(d)
            deltas[name] = h[name] - base  # (n_tok, d) shift vectors

        # direction analysis: mean cosine between per-token shift vectors of two
        # conditions, over all tokens and over the joint top-25%-norm tail
        names = list(deltas)
        norm = {n: deltas[n].norm(dim=-1) for n in names}
        tail_mask = None
        for n in names:
            thr = torch.quantile(norm[n], 0.75)
            m = norm[n] > thr
            tail_mask = m if tail_mask is None else (tail_mask | m)
        dir_all = np.zeros((len(names), len(names)))
        dir_tail = np.zeros_like(dir_all)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                ca = torch.nn.functional.cosine_similarity(deltas[a], deltas[b], dim=-1)
                dir_all[i, j] = float(ca.mean())
                dir_tail[i, j] = float(ca[tail_mask].mean())

        report["steps"][t] = {
            "siglip_prompt_invariance_maxdev": siglip_dev,
            "cos_dist_vs_canon": entries,
            "shift_direction_cos": {
                "names": names,
                "all_tokens": dir_all.tolist(),
                "tail_tokens": dir_tail.tolist(),
            },
        }

        # direction matrix figure
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mat, ttl in [(axes[0], dir_all, "all tokens"),
                             (axes[1], dir_tail, "tail (top-25% shift) tokens")]:
            im = ax.imshow(mat, vmin=-0.2, vmax=1.0, cmap="RdBu_r")
            ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(names)), names, fontsize=8)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"shift-direction cos ({ttl}), t={t}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / f"direction_t{t}.png", dpi=150)
        plt.close(fig)

        # corrected joint PCA: shared centering + explained variance
        fig, ax = plt.subplots(figsize=(7, 6))
        sel = ["const", "t1_same_scene", "para0"]
        stack = torch.cat([base] + [h[n] for n in sel], dim=0)
        mu = stack.mean(0, keepdim=True)
        u, s, v = torch.pca_lowrank(stack - mu, q=8)
        ev = (s**2 / (stack - mu).pow(2).sum()).cpu().numpy()
        pj = {n: ((h[n] - mu) @ v[:, :2]).cpu().numpy() for n in ["canon"] + sel}
        for n, c in zip(["canon"] + sel, ["k", "tab:blue", "tab:red", "tab:green"]):
            ax.scatter(*pj[n].T, s=7, alpha=0.5, c=c, label=n)
        for i in range(0, base.shape[0], 8):
            ax.annotate("", xy=pj["t1_same_scene"][i], xytext=pj["canon"][i],
                        arrowprops={"arrowstyle": "->", "lw": 0.4, "alpha": 0.3,
                                    "color": "tab:red"})
        ax.legend(fontsize=8)
        ax.set_title(f"joint PCA t={t} (shared centering) "
                     f"EV: pc1 {ev[0]:.2f}, pc2 {ev[1]:.2f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"pca_corrected_t{t}.png", dpi=150)
        plt.close(fig)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["steps"][CAPTURE_STEPS[0]], indent=1)[:2000])
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
