#!/usr/bin/env python
"""How much of the deployed Phi' head's held-out correlation is the split?

    # the assigned diagnostic: D0 alone (96 episodes), >= 5 episode-disjoint splits
    python scripts/diag_v253_phi_splits.py --splits 8

    # plus the deployed head's own bundle (D0 + the four D1 runs, 192 episodes),
    # seed 251 first so the deployed fit is REPRODUCED by this pipeline
    python scripts/diag_v253_phi_splits.py --splits 8 --deployed-splits 5

WHAT THIS ANSWERS.  The head that every `--advantage predicted` arm of the v251 round
reads its Phi' off (results/v251_interface/chain3_lr2_m1_*/phi_head.pt, digest
4ac74b90...) was fitted ONCE, on ONE episode-disjoint split, and reported a
decision-region held-out Pearson of 0.3886 (Spearman 0.1291, 0.8439 over all
boundaries).  A single split is a single draw.  The 24-episode predecessor of this
same construction moved over [-0.21, +0.89] across five splits and was positive in
only 3 of 5 (fit_phi_head's own docstring in scripts/train_v251_interface.py).  This
script refits the IDENTICAL head - same construction, same inputs, same region, same
hyperparameters, same normalised coordinates - over several episode-disjoint splits
and reports the spread.

WHAT IT DOES NOT ANSWER.  Nothing about whether the world model helps, whether the
transition is learned, or whether any arm of the round beats another.  The spread of
this head's held-out correlation bounds how well the QUANTITY every predicted
advantage is read off can be estimated at all; it says nothing on its own about the
rollout that produces the latent the head is read on.

HOW A SPLIT IS VARIED, and nothing else with it.  `fit_phi_head` derives its
episode-disjoint holdout from `stable_fraction(seed, episode_key)` and seeds the
optimiser from the same integer, so the split and the init move together; that is the
deployed protocol and it is kept.  Every other input - the tapes, the labels, the
model-pair normaliser, the region mask, steps/batch/lr/weight-decay/holdout-fraction -
is held fixed across splits and across the deployed-head recomputation.  The fit
itself is imported from scripts/train_v251_interface.py rather than reimplemented, so
there is no second copy of the head to drift.

ZERO ENVIRONMENT STEPS.  Tapes, labels and traces already on disk; CPU only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from train_v251_interface import (  # noqa: E402
    build_bundle,
    fit_phi_head,
    load_run,
    pearson,
    sha256_file,
    spearman,
    stable_fraction,
    state_dict_digest,
)

OUT = REPO / "results" / "v253_phi_splits"

D0 = "results/v250_chain3_round/chain3_lr2_d0_2026-09-12T224639Z"
D1 = [
    "results/v250_chain3_round/chain3_lr2_d1_i0_2026-09-12T231912Z",
    "results/v250_chain3_round/chain3_lr2_d1_i1_2026-09-12T232801Z",
    "results/v250_chain3_round/chain3_lr2_d1_i2_2026-09-12T233655Z",
    "results/v250_chain3_round/chain3_lr2_d1_i3_2026-09-12T234542Z",
]
MODEL_PAIR = "results/v241_model_pair/evolve1_2026-09-12T235711Z/model_pair.pt"
#: The head the v251 round actually deployed, and its recorded single-split metrics.
DEPLOYED_HEAD = "results/v251_interface/chain3_lr2_m1_2026-09-12T235743Z/phi_head.pt"


# --------------------------------------------------------------------------- bundles


def load_bundle(tapes: Sequence[str], task: str, mu: Tensor, sd: Tensor,
                dims: tuple[int, int, int], region_from_chunk: int
                ) -> dict[str, Any]:
    """Every input `fit_phi_head` reads, built exactly as train_v251_interface builds it."""
    episodes, records = [], []
    for run_dir in tapes:
        eps, record = load_run(REPO / run_dir if not Path(run_dir).is_absolute()
                               else Path(run_dir), task, dims)
        episodes.extend(eps)
        records.append(record)
        print(f"loaded {record['episodes']} episodes / {record['transitions']} rows "
              f"from {record['run_dir']}", flush=True)
    seen: set[str] = set()
    for ep in episodes:
        if ep.key in seen:
            raise ValueError(f"duplicate episode key {ep.key}: overlapping tapes")
        seen.add(ep.key)
    bundle = build_bundle(episodes, mu, sd)
    bound_episode = torch.repeat_interleave(
        torch.arange(bundle.n_episodes), bundle.ep_len + 1)
    bound_chunk = torch.cat([torch.arange(int(n) + 1) for n in bundle.ep_len])
    if int(bound_chunk.numel()) != int(bundle.z_bound.shape[0]):
        raise AssertionError("boundary bookkeeping")
    region = bound_chunk >= int(region_from_chunk)
    return {
        "bundle": bundle, "records": records, "bound_episode": bound_episode,
        "bound_chunk": bound_chunk, "region": region,
        "tape_digests": sorted(r["tape_sha256"] for r in records),
    }


def split_episodes(seed: int, keys: Sequence[str], holdout_fraction: float
                   ) -> set[int]:
    """The holdout episode set `fit_phi_head` will derive, recomputed for reporting.

    Byte-for-byte the same rule as fit_phi_head; the counts it returns are asserted
    against fit_phi_head's own metrics, so a drift between the two is a hard failure
    rather than a silently different split.
    """
    ranked = sorted(((stable_fraction(seed, key), e) for e, key in enumerate(keys)))
    holdout = {e for frac, e in ranked if frac < holdout_fraction}
    if not holdout:
        holdout.add(ranked[0][1])
    if len(holdout) == len(ranked):
        holdout.discard(ranked[-1][1])
    return holdout


# ------------------------------------------------------------------------ one split


def run_split(data: dict[str, Any], seed: int, *, steps: int, batch: int,
              holdout_fraction: float) -> dict[str, Any]:
    bundle = data["bundle"]
    region, bound_episode = data["region"], data["bound_episode"]
    t0 = time.time()
    net, metrics = fit_phi_head(
        bundle.z_bound, bundle.phi_bound, bound_episode, bundle.keys,
        seed=seed, steps=steps, batch=batch, lr=1e-3, weight_decay=1e-2,
        holdout_fraction=holdout_fraction, region=region)
    wall = time.time() - t0

    hold_eps = split_episodes(seed, bundle.keys, holdout_fraction)
    is_hold = torch.tensor([int(e) in hold_eps for e in bound_episode.tolist()])
    tr_idx = torch.nonzero(~is_hold & region).flatten()
    ho_idx = torch.nonzero(is_hold & region).flatten()
    all_ho = torch.nonzero(is_hold).flatten()
    # the split this script reports must be the split the head was fitted on
    if (int(tr_idx.numel()) != int(metrics["train_boundaries"])
            or int(ho_idx.numel()) != int(metrics["holdout_boundaries"])
            or len(hold_eps) != int(metrics["holdout_episodes"])):
        raise AssertionError(
            f"seed {seed}: recomputed split {(int(tr_idx.numel()), int(ho_idx.numel()), len(hold_eps))} "
            f"!= fit_phi_head's {(metrics['train_boundaries'], metrics['holdout_boundaries'], metrics['holdout_episodes'])}"
        )
    with torch.no_grad():
        pred = torch.cat([net(bundle.z_bound[i:i + 1024]).squeeze(-1)
                          for i in range(0, int(bundle.z_bound.shape[0]), 1024)])
    phi = bundle.phi_bound
    row = {
        "seed": seed,
        "wallclock_s": round(wall, 1),
        "digest": state_dict_digest(net.state_dict()),
        "train_episodes": bundle.n_episodes - len(hold_eps),
        "holdout_episodes": len(hold_eps),
        "region_train_boundaries": int(tr_idx.numel()),
        "region_holdout_boundaries": int(ho_idx.numel()),
        "all_holdout_boundaries": int(all_ho.numel()),
        "region_holdout_pearson": metrics["holdout_corr"],
        "region_holdout_spearman": metrics["holdout_spearman"],
        "all_boundary_holdout_pearson": metrics["holdout_corr_all_boundaries"],
        "all_boundary_holdout_spearman": spearman(
            pred[all_ho].numpy(), phi[all_ho].numpy()),
        "region_train_pearson": metrics["train_corr"],
        "region_train_mse": metrics["train_mse"],
        "region_holdout_mse": metrics["holdout_mse"],
        "all_boundary_holdout_mse": float(((pred[all_ho] - phi[all_ho]) ** 2).mean()),
        "holdout_phi_std": float(phi[ho_idx].std(unbiased=False)),
        "fit_metrics": metrics,
    }
    print(f"[seed {seed}] region holdout r={row['region_holdout_pearson']:+.4f} "
          f"rho={row['region_holdout_spearman']:+.4f} "
          f"all-boundary r={row['all_boundary_holdout_pearson']:+.4f} "
          f"({row['train_episodes']}/{row['holdout_episodes']} eps, "
          f"{row['region_train_boundaries']}/{row['region_holdout_boundaries']} region bnds, "
          f"{wall:.0f}s)", flush=True)
    return row


def spread(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": min(vals), "max": max(vals),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n_positive": sum(1 for v in vals if v > 0),
        "values": vals,
    }


# ----------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--model-pair", default=MODEL_PAIR)
    ap.add_argument("--d0-tapes", nargs="+", default=[D0])
    ap.add_argument("--deployed-tapes", nargs="+", default=[D0, *D1])
    ap.add_argument("--deployed-head", default=DEPLOYED_HEAD)
    ap.add_argument("--base-seed", type=int, default=251,
                    help="first split seed; the deployed head used 251")
    ap.add_argument("--splits", type=int, default=8,
                    help="episode-disjoint splits on the D0 bundle")
    ap.add_argument("--deployed-splits", type=int, default=5,
                    help="splits on the deployed head's own bundle (D0 + D1); the "
                         "first is seed 251, which reproduces the deployed fit. 0 skips")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--region-from-chunk", type=int, default=26)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out-root", type=Path, default=OUT)
    ap.add_argument("--tag", default="chain3_lr2_d0_multisplit")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    if a.threads:
        torch.set_num_threads(int(a.threads))
    if a.splits < 5:
        raise SystemExit("--splits must be >= 5: one split is what this diagnostic exists to replace")

    pair_path = (REPO / a.model_pair).resolve()
    pair = torch.load(pair_path, weights_only=False, map_location="cpu")
    dims = tuple(int(x) for x in pair["dims"])
    mu = pair["mu"].detach().float().cpu()
    sd = pair["sd"].detach().float().cpu()
    normaliser_digest = state_dict_digest({"mu": mu, "sd": sd})
    if pair["task"] != a.task:
        raise SystemExit(f"{pair_path}: task {pair['task']!r} != {a.task!r}")

    deployed = torch.load((REPO / a.deployed_head).resolve(), weights_only=False,
                          map_location="cpu")
    if deployed.get("normaliser_digest") != normaliser_digest:
        raise SystemExit("the deployed head was fitted under a different normaliser")

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = Path(a.out_root) / a.tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- the assigned diagnostic: D0 alone, >= 5 episode-disjoint splits ----------
    d0 = load_bundle(a.d0_tapes, a.task, mu, sd, dims, a.region_from_chunk)
    print(f"D0 bundle: {d0['bundle'].n_episodes} episodes, "
          f"{int(d0['bundle'].z_bound.shape[0])} boundaries, "
          f"{int(d0['region'].sum())} in the decision region "
          f"(chunk >= {a.region_from_chunk})", flush=True)
    d0_rows = [run_split(d0, a.base_seed + i, steps=a.steps, batch=a.batch,
                         holdout_fraction=a.holdout) for i in range(a.splits)]
    d0_stats = {k: spread(d0_rows, k) for k in
                ("region_holdout_pearson", "region_holdout_spearman",
                 "all_boundary_holdout_pearson", "all_boundary_holdout_spearman",
                 "region_holdout_mse")}
    del d0["bundle"]

    # ---- the deployed head's own bundle, its own split first ---------------------
    dep_rows: list[dict[str, Any]] = []
    dep_stats: dict[str, Any] = {}
    reproduction: dict[str, Any] = {"attempted": bool(a.deployed_splits)}
    if a.deployed_splits:
        dep = load_bundle(a.deployed_tapes, a.task, mu, sd, dims, a.region_from_chunk)
        print(f"deployed bundle: {dep['bundle'].n_episodes} episodes, "
              f"{int(dep['bundle'].z_bound.shape[0])} boundaries, "
              f"{int(dep['region'].sum())} in the decision region", flush=True)
        if dep["tape_digests"] != sorted(deployed.get("tape_digests", [])):
            raise SystemExit("--deployed-tapes are not the tapes the deployed head saw")
        dep_rows = [run_split(dep, a.base_seed + i, steps=a.steps, batch=a.batch,
                              holdout_fraction=a.holdout)
                    for i in range(a.deployed_splits)]
        dep_stats = {k: spread(dep_rows, k) for k in
                     ("region_holdout_pearson", "region_holdout_spearman",
                      "all_boundary_holdout_pearson", "all_boundary_holdout_spearman",
                      "region_holdout_mse")}
        own = dep_rows[0]
        rec = dict(deployed.get("metrics", {}))
        reproduction = {
            "attempted": True,
            "deployed_digest": deployed.get("digest"),
            "recomputed_digest": own["digest"],
            "bitwise_identical": own["digest"] == deployed.get("digest"),
            "recorded": {
                "holdout_corr": rec.get("holdout_corr"),
                "holdout_spearman": rec.get("holdout_spearman"),
                "holdout_corr_all_boundaries": rec.get("holdout_corr_all_boundaries"),
                "train_boundaries": rec.get("train_boundaries"),
                "holdout_boundaries": rec.get("holdout_boundaries"),
                "holdout_episodes": rec.get("holdout_episodes"),
            },
            "recomputed": {
                "holdout_corr": own["region_holdout_pearson"],
                "holdout_spearman": own["region_holdout_spearman"],
                "holdout_corr_all_boundaries": own["all_boundary_holdout_pearson"],
                "train_boundaries": own["region_train_boundaries"],
                "holdout_boundaries": own["region_holdout_boundaries"],
                "holdout_episodes": own["holdout_episodes"],
            },
            "split_identical": (
                own["region_train_boundaries"] == rec.get("train_boundaries")
                and own["region_holdout_boundaries"] == rec.get("holdout_boundaries")
                and own["holdout_episodes"] == rec.get("holdout_episodes")),
        }
        del dep["bundle"]

    summary = {
        "schema": "v253_phi_splits_v1",
        "utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "question": (
            "Over how wide a range does the deployed Phi' head's held-out correlation "
            "move when only the episode-disjoint split changes?"),
        "answers_nothing_about": (
            "whether the world model helps, whether T_theta is learned, or any "
            "between-arm difference in the v251 round"),
        "env_steps": 0,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "config": {
            "task": a.task,
            "steps": a.steps, "batch": a.batch, "lr": 1e-3, "weight_decay": 1e-2,
            "holdout_fraction": a.holdout,
            "region_from_chunk": a.region_from_chunk,
            "region_restricted": True,
            "split_rule": "stable_fraction(seed, episode_key) < holdout_fraction, "
                          "episode-disjoint; seed also seeds the init and the batch RNG",
            "base_seed": a.base_seed,
            "seeds_d0": [a.base_seed + i for i in range(a.splits)],
            "seeds_deployed_bundle": [a.base_seed + i for i in range(a.deployed_splits)],
            "head": "Sequential(LayerNorm(z), Linear(z,256), GELU, Linear(256,1))",
            "fit_imported_from": "scripts/train_v251_interface.fit_phi_head",
            "target": "Phi' from lcwm/v250_progress.tape_progress",
            "space": "model-pair normalised latents",
            "model_pair": str(pair_path),
            "normaliser_digest": normaliser_digest,
            "steps_reduced_from_deployed": a.steps != 3000,
        },
        "d0": {
            "tapes": [str((REPO / t).resolve()) for t in a.d0_tapes],
            "runs": d0["records"],
            "tape_digests": d0["tape_digests"],
            "episodes": int(d0["bound_episode"].max()) + 1,
            "boundaries": int(d0["bound_chunk"].numel()),
            "region_boundaries": int(d0["region"].sum()),
            "region_fraction": float(d0["region"].float().mean()),
            "splits": d0_rows,
            "spread": d0_stats,
        },
        "deployed_bundle": {
            "tapes": [str((REPO / t).resolve()) for t in a.deployed_tapes],
            "splits": dep_rows,
            "spread": dep_stats,
            "reproduction_of_deployed_head": reproduction,
        },
        "provenance": {
            "argv": list(sys.argv if argv is None else ["diag_v253_phi_splits", *argv]),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "train_v251_interface_sha256": sha256_file(
                REPO / "scripts" / "train_v251_interface.py"),
            "v250_progress_sha256": sha256_file(REPO / "lcwm" / "v250_progress.py"),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    lines = [
        f"Phi' head multi-split diagnostic - {a.tag} - {started}",
        f"head: LayerNorm -> Linear({dims[0]},256) -> GELU -> Linear(256,1), "
        f"{a.steps} steps, batch {a.batch}, lr 1e-3, wd 1e-2, holdout {a.holdout}, "
        f"region chunk >= {a.region_from_chunk}",
        "",
        f"D0 only: {summary['d0']['episodes']} episodes, "
        f"{summary['d0']['boundaries']} boundaries, "
        f"{summary['d0']['region_boundaries']} in region",
        f"{'seed':>6} {'tr/ho eps':>10} {'tr/ho bnd':>12} {'region r':>10} "
        f"{'region rho':>11} {'all-bnd r':>10}",
    ]
    for r in d0_rows:
        lines.append(
            f"{r['seed']:>6} {r['train_episodes']:>4}/{r['holdout_episodes']:<5} "
            f"{r['region_train_boundaries']:>5}/{r['region_holdout_boundaries']:<6} "
            f"{r['region_holdout_pearson']:>+10.4f} {r['region_holdout_spearman']:>+11.4f} "
            f"{r['all_boundary_holdout_pearson']:>+10.4f}")
    for name, st in d0_stats.items():
        if st.get("n"):
            lines.append(f"  {name}: min {st['min']:+.4f} median {st['median']:+.4f} "
                         f"mean {st['mean']:+.4f} max {st['max']:+.4f} "
                         f"sd {st['std']:.4f}  positive {st['n_positive']}/{st['n']}")
    if dep_rows:
        lines += ["", "deployed bundle (D0 + four D1 runs), seed 251 = the deployed fit:",
                  f"{'seed':>6} {'tr/ho eps':>10} {'tr/ho bnd':>12} {'region r':>10} "
                  f"{'region rho':>11} {'all-bnd r':>10}"]
        for r in dep_rows:
            lines.append(
                f"{r['seed']:>6} {r['train_episodes']:>4}/{r['holdout_episodes']:<5} "
                f"{r['region_train_boundaries']:>5}/{r['region_holdout_boundaries']:<6} "
                f"{r['region_holdout_pearson']:>+10.4f} "
                f"{r['region_holdout_spearman']:>+11.4f} "
                f"{r['all_boundary_holdout_pearson']:>+10.4f}")
        for name, st in dep_stats.items():
            if st.get("n"):
                lines.append(f"  {name}: min {st['min']:+.4f} median {st['median']:+.4f} "
                             f"mean {st['mean']:+.4f} max {st['max']:+.4f} "
                             f"sd {st['std']:.4f}  positive {st['n_positive']}/{st['n']}")
        lines.append(f"  reproduction of the deployed head: bitwise "
                     f"{reproduction['bitwise_identical']}, split identical "
                     f"{reproduction['split_identical']}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
