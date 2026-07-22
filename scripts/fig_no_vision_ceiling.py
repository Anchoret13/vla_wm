#!/usr/bin/env python
"""Evidence figure: the expert-demo assay cannot demonstrate visual dynamics.

The no-vision baseline (tokens zeroed; proprio + action blocks + history +
task-id retained) matches every vision model on all future readouts. Kept as a
standing exhibit for the data-regime conclusion of 2026-07-22.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "wm_v2" / "no_vision_ceiling.png"

# Okabe-Ito (CVD-safe); fixed assignment, baselines wear neutral ink
C_SIGLIP, C_REAL = "#0072B2", "#D55E00"
C_NOVIS, C_NOTASK = "#1a1a1a", "#767676"


def load():
    runs = {}
    for p in glob.glob(str(REPO / "results/wm_v2/*.json")):
        r = json.load(open(p))
        c = r["config"]
        key = (c["arm"], c["tag"], c["seed"])
        runs[key] = r["diagnostics"]
    return runs


def main() -> None:
    runs = load()
    groups = [("siglip", "ema"), ("siglip", "emavar"),
              ("real_ll", "ema"), ("real_ll", "emavar")]
    labels = ["SigLIP\nEMA", "SigLIP\nEMA+var", "canon-LL\nEMA",
              "canon-LL\nEMA+var"]
    colors = [C_SIGLIP, C_SIGLIP, C_REAL, C_REAL]
    fills = ["full", "none", "full", "none"]
    novis = runs[("siglip", "taskonly", 0)]   # artifact tag; = no-vision
    notask = runs[("siglip", "notask", 0)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    def scatter_panel(ax, metric, title, ylabel, better):
        for i, ((arm, tag), col, fs) in enumerate(zip(groups, colors, fills)):
            for seed in (0, 1):
                v = runs[(arm, tag, seed)][metric]
                ax.plot(i, v, "o", ms=9, mfc=col if fs == "full" else "white",
                        mec=col, mew=1.8)
        ax.axhline(novis[metric], color=C_NOVIS, lw=1.6)
        ax.axhline(notask[metric], color=C_NOTASK, lw=1.4, ls=(0, (4, 3)))
        ax.text(3.45, novis[metric], " no-vision", color=C_NOVIS, fontsize=8.5,
                va="center", ha="left")
        ax.text(3.45, notask[metric], " no-task", color=C_NOTASK, fontsize=8.5,
                va="center", ha="left")
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_xlim(-0.6, 4.6)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.02, better, transform=ax.transAxes, fontsize=7.5,
                color="#767676")

    scatter_panel(axes[0], "f1_fut_k1",
                  "Future predicate F1 (k=1)", "F1", "higher = better")
    scatter_panel(axes[1], "qmse_fut_k1",
                  "Future proprio MSE (k=1)", "MSE", "lower = better")

    ax = axes[2]
    ks = [1, 2, 3, 4]
    for (arm, tag), col, fs in zip(groups, colors, fills):
        ys = [sum(runs[(arm, tag, s)][f"f1_fut_k{k}"] for s in (0, 1)) / 2
              for k in ks]
        ax.plot(ks, ys, "-o", color=col, ms=5, lw=1.8,
                mfc=col if fs == "full" else "white", mec=col)
    ax.plot(ks, [novis[f"f1_fut_k{k}"] for k in ks], "-", color=C_NOVIS,
            lw=1.6)
    ax.plot(ks, [notask[f"f1_fut_k{k}"] for k in ks], ls=(0, (4, 3)),
            color=C_NOTASK, lw=1.4)
    ax.text(4.05, novis["f1_fut_k4"], " no-vision", color=C_NOVIS,
            fontsize=8.5, va="center")
    ax.set_xticks(ks)
    ax.set_xlabel("rollout horizon k (decision steps)", fontsize=9)
    ax.set_ylabel("F1", fontsize=9)
    ax.set_title("Future predicate F1 over horizon", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=C_SIGLIP, mec=C_SIGLIP,
                   label="SigLIP (filled=EMA, open=EMA+var)"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_REAL, mec=C_REAL,
                   label="canonical-LL"),
        plt.Line2D([], [], color=C_NOVIS, label="no-vision baseline "
                   "(proprio+action+history+task-id)"),
        plt.Line2D([], [], color=C_NOTASK, ls=(0, (4, 3)),
                   label="no-task baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Expert-demo assay cannot demonstrate visual dynamics: "
                 "the no-vision baseline matches every vision model",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
