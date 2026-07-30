#!/usr/bin/env python
"""H7.0(d) — tracked deterministic analysis for behavior_factorial_v1.

Reconstructs from records.jsonl, machine-readably: per-task SR / mean Q /
per-object completion / D_final / A_D per arm; the registered S_prefix
contrasts with fixed-seed 10,000-resample cluster bootstrap CIs, exact
sign-flip p-values, and Holm correction; and the analysis configuration.

Per the 2026-07-29 post-H4 review: SR is primary, Q partial credit, and
D/A_D are retained only as a declared arbitrary-order diagnostic (the BDDL
goals are unordered conjunctions). S_prefix statistics are kept as the
registered analysis with that caveat recorded in the artifact itself.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = REPO_ROOT / "results" / "behavior_factorial_v1"
BOOT_SEED = 20260729
BOOT_N = 10_000


def cluster_bootstrap(deltas: list[float]):
    rng = random.Random(BOOT_SEED)
    n = len(deltas)
    means = sorted(
        sum(rng.choice(deltas) for _ in range(n)) / n
        for _ in range(BOOT_N)
    )
    return means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N)]


def sign_flip_p(deltas: list[float]) -> float:
    observed = abs(sum(deltas))
    count = 0
    for signs in itertools.product((1, -1), repeat=len(deltas)):
        if abs(sum(s * d for s, d in zip(signs, deltas))) >= observed - 1e-12:
            count += 1
    return count / 2 ** len(deltas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="h4_dev_matrix_v1")
    parser.add_argument("--panel", default="development")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="default: <panel>/analysis_<run_id>.json",
    )
    args = parser.parse_args()
    records_path = ROOT / args.panel / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip() and json.loads(line)["run_id"] == args.run_id
    ]
    arms = sorted({r["arm"] for r in records})
    tasks = sorted({r["task"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    by = {(r["task"], r["seed"], r["arm"]): r for r in records}
    assert len(records) == len(arms) * len(tasks) * len(seeds), (
        "incomplete matrix"
    )

    per_task = {}
    for task in tasks:
        per_task[task] = {}
        for arm in arms:
            rows = [by[(task, seed, arm)] for seed in seeds]
            per_task[task][arm] = {
                "sr": sum(r["success"] for r in rows) / len(rows),
                "success_count": sum(r["success"] for r in rows),
                "mean_q": sum(r["q_final"] for r in rows) / len(rows),
                "mean_d_final": sum(r["d_final"] for r in rows) / len(rows),
                "mean_a_d": sum(r["a_d"] for r in rows) / len(rows),
                "success_steps": [
                    r["steps"] for r in rows if r["success"]
                ],
                "atom_first_completion": [
                    r["atom_first_completion"] for r in rows
                ],
            }

    n_atoms_total = sum(int(t[5]) for t in tasks)
    s_prefix = {
        arm: [
            sum(by[(t, seed, arm)]["d_final"] for t in tasks)
            / n_atoms_total
            for seed in seeds
        ]
        for arm in arms
    }
    contrasts = {}
    registered = [
        ("rec_wm", "stock"),
        ("rec_wm", "rec_random"),
        ("rec_wm", "cur_wm"),
    ]
    pvals = {}
    for a, b in registered:
        if a not in s_prefix or b not in s_prefix:
            continue
        deltas = [
            s_prefix[a][i] - s_prefix[b][i] for i in range(len(seeds))
        ]
        lo, hi = cluster_bootstrap(deltas)
        p = sign_flip_p(deltas)
        name = f"{a}-{b}"
        pvals[name] = p
        contrasts[name] = {
            "per_seed_delta": deltas,
            "mean_delta": sum(deltas) / len(deltas),
            "bootstrap_95ci": [lo, hi],
            "sign_flip_p": p,
        }
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    for i, (name, p) in enumerate(ordered):
        contrasts[name]["holm_adjusted_alpha"] = 0.05 / (m - i)
        contrasts[name]["holm_reject"] = p <= 0.05 / (m - i)

    output = {
        "schema": "behavior_factorial_analysis_v1",
        "run_id": args.run_id,
        "config": {
            "bootstrap_resamples": BOOT_N,
            "bootstrap_seed": BOOT_SEED,
            "sign_flip": "exact, all 2^n sign patterns",
            "correction": "Holm over the three registered contrasts",
            "independence_unit": "environment seed",
            "n_seeds": len(seeds),
        },
        "metric_caveat": (
            "goals are unordered BDDL conjunctions; D_final/A_D/S_prefix "
            "are arbitrary-order diagnostics (2026-07-29 review); SR is "
            "primary, Q is partial credit"
        ),
        "per_task": per_task,
        "s_prefix_per_seed": s_prefix,
        "registered_contrasts": contrasts,
    }
    out = args.output or (
        ROOT / args.panel / f"analysis_{args.run_id}.json"
    )
    out.write_text(json.dumps(output, indent=1))
    print(f"-> {out}")
    for task in tasks:
        line = " | ".join(
            f"{arm}: SR={per_task[task][arm]['success_count']}"
            f"/{len(seeds)} Q={per_task[task][arm]['mean_q']:.2f}"
            for arm in arms
        )
        print(task, line)
    for name, c in contrasts.items():
        print(
            f"{name}: Δ={c['mean_delta']:+.3f} "
            f"CI=[{c['bootstrap_95ci'][0]:+.3f},"
            f"{c['bootstrap_95ci'][1]:+.3f}] p={c['sign_flip_p']:.3f} "
            f"holm_reject={c['holm_reject']}"
        )


if __name__ == "__main__":
    main()
