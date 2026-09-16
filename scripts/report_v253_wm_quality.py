#!/usr/bin/env python
"""Prediction-quality readout, with intervals, for the EVOLVE-1 deployed model pair.

    python scripts/report_v253_wm_quality.py --stage all

WHAT THIS ANSWERS.  The 2026-09-12 round shipped four transitions (M0 stale,
M0_cont base-continue, M1 updated, M1_shuf updated-with-shuffled-actions) and
scored the POLICIES they produced.  It never published the OBJECT axis: whether
the deployment update changed the MODEL at all.  `results/v241_model_pair/
evolve1_2026-09-12T235711Z/summary.json` carries held-out rollout MSE, the
identity (`predict no change`) floor - a control contracted in v241, NOT a
registered endpoint: the round's preregistration.json contains no model-quality
endpoint at all - and the action-shuffle control at H = 1/3/10 on the old and the
new distribution - every one a POINT
ESTIMATE with no interval, so no read of it can distinguish "the update moved the
model" from "the update moved the third decimal".

This script adds the intervals and the paired contrasts, on CPU, with ZERO
environment steps.

    stage a   Recompute every committed quantity from the checkpoint and the tapes,
              assert the recomputation reproduces summary.json, then attach
              EPISODE-bootstrap 95% percentile intervals.  Rows inside an episode
              are autocorrelated (one chain3 episode is 75 chunks of one stalled
              trajectory), so the resampling unit is the held-out episode, never
              the row and never the window.  The MSE is a ratio of sums
              (sum of squared error over all elements / element count), exactly as
              build_v241_model_pair.evaluate_model computes it, and every bootstrap
              replicate recomputes that ratio on the resampled episodes.

              The three paired contrasts, all on the SAME resampled episode index
              so the pairing is preserved:
                M1 - M0_cont   did the new DATA move the model, at matched budget
                M1 - M0        did the update move the model at all, vs the stale one
                M1 - M1_shuf   is the transition using the ACTIONS

    stage b   Predicted-vs-realised Phi' at H = 10: roll T_theta ten chunks forward
              from a held-out (z_t, b_t) under the ACTUALLY EXECUTED actions, read
              the deployed frozen Phi' head off the PREDICTED latent, and correlate
              it with the REALISED Phi' label at that boundary.  Two controls are
              computed on the identical rows, because the correlation alone cannot
              separate the roll from the state it started in:
                identity   the same head read off z_t, no roll at all
                oracle     the same head read off the REAL latent at t+10c
              A predicted-latent correlation that does not beat `identity` says the
              roll contributed nothing; `oracle` is the head's own ceiling on those
              rows.  Reported on the decision region (the boundaries at chunk >= 26
              the head was FITTED on and the interface acts in) and over all
              boundaries separately, as train_v251_interface always reports them.

WHICH AXIS.  All of it is the OBJECT axis of CLAUDE.md: T_theta rolled forward
under an action and heads read off the PREDICTED latent.  Stage a additionally
speaks to the DATA axis (both distributions are deployment tapes; update_holdout
is the new one).  NONE of it speaks to the task or the placement axis, and none of
it is evidence that the policy improved: whether the update changed the MODEL and
whether it changed the POLICY are separate questions, and this script answers only
the first.  The deployment endpoints are the round driver's to report.

WHAT WOULD OVERTURN EACH CONCLUSION.
  * any MSE contrast - a 95% interval that spans zero says the point estimate is
    not resolved at this holdout size (21 base / 18 update episodes); a second
    model pair trained from a different seed landing outside the interval says the
    interval covers sampling over episodes but not over initialisations, which it
    does not, because there is exactly one pair on disk.
  * "the model beats predict-no-change" - mse >= identity_mse at that horizon.
  * "the transition uses the actions" - M1 and M1_shuf agreeing.
  * the predicted-Phi' correlation - the identity control matching it, which
    places the correlation in the start state rather than in the roll.
  * every one of these - the head and the model pair being single fits: one split,
    one seed, no cross-seed replication (CLAUDE.md rule 9).

NOT PRE-REGISTERED.  The round's preregistration.json governs the DEPLOYMENT
endpoints.  These are model-side diagnostics computed after the fact from
artifacts that already exist; they carry bootstrap percentile intervals and
bootstrap tail probabilities, which are descriptive, and no multiplicity
correction is applied to them.  They are not one of the round's registered
families and must not be reported as if they were.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from build_v241_model_pair import (  # noqa: E402
    load_role,
    rollout_mse_for_episode,
    sha256_file,
    split_role,
)
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402
from train_v251_interface import (  # noqa: E402
    build_bundle,
    episode_beliefs,
    load_run,
    make_phi_head,
    pearson,
    roll_latent,
    spearman,
    stable_fraction,
    state_dict_digest,
)

DEFAULT_PAIR = (
    REPO / "results" / "v241_model_pair" / "evolve1_2026-09-12T235711Z" / "model_pair.pt"
)
DEFAULT_PHI_HEAD = (
    REPO / "results" / "v251_interface" / "chain3_lr2_m1_2026-09-12T235743Z" / "phi_head.pt"
)
ROUND = REPO / "results" / "v250_chain3_round"
DEFAULT_BASE_TAPES = [ROUND / "chain3_lr2_d0_2026-09-12T224639Z" / "tape.pt"]
DEFAULT_UPDATE_TAPES = [
    ROUND / "chain3_lr2_d1_i0_2026-09-12T231912Z" / "tape.pt",
    ROUND / "chain3_lr2_d1_i1_2026-09-12T232801Z" / "tape.pt",
    ROUND / "chain3_lr2_d1_i2_2026-09-12T233655Z" / "tape.pt",
    ROUND / "chain3_lr2_d1_i3_2026-09-12T234542Z" / "tape.pt",
]
DEFAULT_OUT = REPO / "results" / "v253_wm_quality"

ARMS = ("stale", "base_continue", "updated", "updated_shuffled")
#: Round-facing names, so the artifact reads in the same vocabulary as the arms.
ARM_ALIAS = {
    "stale": "M0",
    "base_continue": "M0_cont",
    "updated": "M1",
    "updated_shuffled": "M1_shuf",
}
CONTRASTS = (
    ("updated", "base_continue"),
    ("updated", "stale"),
    ("updated", "updated_shuffled"),
    ("base_continue", "stale"),
)
SPLITS = ("base_holdout", "update_holdout")
#: The split rule the committed pair was built under; asserted, not chosen here.
SPLIT_SEED = 241
SPLIT_FRACTION = 0.2
#: Paired readout contrasts at H = 10.  `predicted_latent - identity_no_roll` is the
#: refuting ablation for "the roll carries the correlation"; `- oracle` says how much
#: of the head's own ceiling on these rows the roll reaches.
PAIRED_READOUTS = (
    ("predicted_latent", "identity_no_roll"),
    ("predicted_latent", "oracle_real_future_latent"),
)
#: The Phi' head split and decision region the deployed head was fitted under.
PHI_SEED = 251
PHI_HOLDOUT = 0.2
PHI_REGION_FROM_CHUNK = 26


def git_value(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def ci(samples: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "sd": float(samples.std(ddof=1))}


def tail_probability(samples: np.ndarray) -> float:
    """Two-sided percentile-bootstrap tail probability. NOT a pre-registered test."""
    below = float(np.mean(samples <= 0.0))
    above = float(np.mean(samples >= 0.0))
    return float(min(1.0, 2.0 * min(below, above)))


# --------------------------------------------------------------- stage a: rollout MSE


def per_episode_sse(
    model: RawLatentWM,
    episodes: Sequence[Any],
    horizon: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Per-episode squared-error sums under build_v241's own donor pairing.

    evaluate_model sorts by key, drops episodes shorter than the horizon, and pairs
    each episode with the NEXT one in that order as its action-shuffle donor.  The
    pairing is reproduced here and then held FIXED across bootstrap replicates: the
    donor is a property of the evaluation, not of the resample.
    """
    model.eval().cpu()
    ordered = sorted(episodes, key=lambda ep: ep.key)
    eligible = [ep for ep in ordered if ep.n >= horizon]
    if len(eligible) < 2:
        raise ValueError("need at least two eligible episodes")
    learned, shuffled, identity, counts, windows, keys = [], [], [], [], [], []
    for i, ep in enumerate(eligible):
        donor = eligible[(i + 1) % len(eligible)]
        ls, ss, ids, n = rollout_mse_for_episode(model, ep, donor, horizon, mu, sd)
        learned.append(ls)
        shuffled.append(ss)
        identity.append(ids)
        counts.append(n)
        windows.append(max(ep.n - horizon + 1, 0))
        keys.append(ep.key)
    return {
        "learned_sse": np.asarray(learned, dtype=np.float64),
        "shuffled_sse": np.asarray(shuffled, dtype=np.float64),
        "identity_sse": np.asarray(identity, dtype=np.float64),
        "elements": np.asarray(counts, dtype=np.float64),
        "windows": np.asarray(windows, dtype=np.float64),
        "keys": keys,
    }


def ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Ratio-of-sums over a bootstrap index matrix [B, E] -> [B]."""
    return num[idx].sum(axis=1) / den[idx].sum(axis=1)


def stage_a(
    pair: dict[str, Any],
    base_tapes: Sequence[Path],
    update_tapes: Sequence[Path],
    committed: dict[str, Any],
    horizons: Sequence[int],
    bootstrap: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    dims = tuple(int(x) for x in pair["dims"])
    mu = pair["mu"].detach().float().cpu()
    sd = pair["sd"].detach().float().cpu()
    task = str(pair["task"])

    base_eps, _, base_dims, _ = load_role(
        list(base_tapes), "base", task, None, None, False
    )
    update_eps, _, update_dims, _ = load_role(
        list(update_tapes), "update", task, base_dims, None, False
    )
    if base_dims != dims or update_dims != dims:
        raise SystemExit(f"tape dims {base_dims}/{update_dims} != pair dims {dims}")
    _, base_holdout = split_role(base_eps, SPLIT_FRACTION, SPLIT_SEED)
    _, update_holdout = split_role(update_eps, SPLIT_FRACTION, SPLIT_SEED)
    holdouts = {"base_holdout": base_holdout, "update_holdout": update_holdout}

    # --- reproduction check 1: the split we recovered IS the committed split -------
    split_check: dict[str, Any] = {}
    committed_split = committed["provenance"]["split"]
    for name, eps in holdouts.items():
        got = sorted(ep.key for ep in eps)
        want = sorted(rec["key"] for rec in committed_split[name])
        split_check[name] = {
            "recovered_episodes": len(got),
            "committed_episodes": len(want),
            "keys_match": got == want,
        }
    if not all(v["keys_match"] for v in split_check.values()):
        raise SystemExit(f"holdout split does not reproduce: {split_check}")

    models: dict[str, RawLatentWM] = {}
    for arm in ARMS:
        net = RawLatentWM(*dims)
        net.load_state_dict(pair["models"][arm], strict=True)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        models[arm] = net

    raw: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    for arm in ARMS:
        raw[arm] = {}
        for split, eps in holdouts.items():
            raw[arm][split] = {}
            for h in horizons:
                print(f"stage a: {ARM_ALIAS[arm]} {split} H={h}", flush=True)
                raw[arm][split][h] = per_episode_sse(models[arm], eps, h, mu, sd)

    # --- reproduction check 2: the point estimates match summary.json --------------
    committed_models = committed["metrics"]["models"]
    repro_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for split in SPLITS:
            for h in horizons:
                r = raw[arm][split][h]
                got = {
                    "mse": float(r["learned_sse"].sum() / r["elements"].sum()),
                    "identity_mse": float(r["identity_sse"].sum() / r["elements"].sum()),
                    "action_shuffle_mse": float(
                        r["shuffled_sse"].sum() / r["elements"].sum()
                    ),
                    "elements": float(r["elements"].sum()),
                    "windows": float(r["windows"].sum()),
                }
                got["x_identity"] = got["mse"] / got["identity_mse"]
                want = committed_models[arm][split][str(h)]
                for field, value in got.items():
                    reference = float(want[field])
                    delta = value - reference
                    repro_rows.append(
                        {
                            "arm": ARM_ALIAS[arm],
                            "split": split,
                            "horizon": h,
                            "field": field,
                            "recomputed": value,
                            "committed": reference,
                            "abs_delta": abs(delta),
                            "rel_delta": abs(delta) / abs(reference)
                            if reference != 0
                            else 0.0,
                        }
                    )
    max_rel = max(row["rel_delta"] for row in repro_rows)
    max_abs = max(row["abs_delta"] for row in repro_rows)
    worst = max(repro_rows, key=lambda row: row["rel_delta"])
    reproduction = {
        "split": split_check,
        "fields_checked": len(repro_rows),
        "max_abs_delta": max_abs,
        "max_rel_delta": max_rel,
        "tolerance_rel": tolerance,
        "worst": worst,
        "passed": bool(max_rel <= tolerance),
    }

    # --- episode bootstrap --------------------------------------------------------
    rng = np.random.default_rng(seed)
    estimates: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for split in SPLITS:
        n_eps = len(holdouts[split])
        for h in horizons:
            # One index matrix per (split, horizon), SHARED by every arm, so the
            # contrasts below are paired on the episode and the difference interval
            # is not inflated by independent resampling of the two arms.
            idx = rng.integers(0, n_eps, size=(bootstrap, n_eps))
            boot_mse: dict[str, np.ndarray] = {}
            for arm in ARMS:
                r = raw[arm][split][h]
                if len(r["elements"]) != n_eps:
                    raise SystemExit("episode count changed between arms")
                m = ratio(r["learned_sse"], r["elements"], idx)
                ident = ratio(r["identity_sse"], r["elements"], idx)
                shuf = ratio(r["shuffled_sse"], r["elements"], idx)
                boot_mse[arm] = m
                node = estimates.setdefault(ARM_ALIAS[arm], {}).setdefault(split, {})
                node[str(h)] = {
                    "episodes": n_eps,
                    "elements": float(r["elements"].sum()),
                    "windows": int(r["windows"].sum()),
                    "mse": float(r["learned_sse"].sum() / r["elements"].sum()),
                    "mse_ci95": ci(m),
                    "identity_mse": float(r["identity_sse"].sum() / r["elements"].sum()),
                    "identity_mse_ci95": ci(ident),
                    "x_identity": float(
                        r["learned_sse"].sum() / r["identity_sse"].sum()
                    ),
                    "x_identity_ci95": ci(m / ident),
                    "mse_minus_identity": float(
                        (r["learned_sse"].sum() - r["identity_sse"].sum())
                        / r["elements"].sum()
                    ),
                    "mse_minus_identity_ci95": ci(m - ident),
                    "mse_minus_identity_tail_p": tail_probability(m - ident),
                    "action_shuffle_mse": float(
                        r["shuffled_sse"].sum() / r["elements"].sum()
                    ),
                    "action_shuffle_mse_ci95": ci(shuf),
                    "action_shuffle_over_learned": float(
                        r["shuffled_sse"].sum() / r["learned_sse"].sum()
                    ),
                    "action_shuffle_over_learned_ci95": ci(shuf / m),
                }
            for left, right in CONTRASTS:
                label = f"{ARM_ALIAS[left]}_minus_{ARM_ALIAS[right]}"
                d = boot_mse[left] - boot_mse[right]
                point = float(
                    raw[left][split][h]["learned_sse"].sum()
                    / raw[left][split][h]["elements"].sum()
                    - raw[right][split][h]["learned_sse"].sum()
                    / raw[right][split][h]["elements"].sum()
                )
                contrasts.setdefault(label, {}).setdefault(split, {})[str(h)] = {
                    "episodes": n_eps,
                    "delta_mse": point,
                    "delta_mse_ci95": ci(d),
                    "tail_p_two_sided": tail_probability(d),
                    "fraction_negative": float(np.mean(d < 0.0)),
                    "direction": "lower is better for the left arm",
                }
    return {
        "reproduction": reproduction,
        "bootstrap_resamples": bootstrap,
        "bootstrap_unit": "held-out episode, with replacement; ratio-of-sums recomputed",
        "arms": estimates,
        "contrasts": contrasts,
        "holdout_episode_keys": {
            name: sorted(ep.key for ep in eps) for name, eps in holdouts.items()
        },
    }


# ------------------------------------------------- stage b: predicted-vs-realised Phi'


def corr_bootstrap(
    readouts: dict[str, np.ndarray],
    real: np.ndarray,
    ep_index: np.ndarray,
    n_eps: int,
    rng: np.random.Generator,
    bootstrap: int,
    paired: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Pearson/Spearman per readout, plus PAIRED deltas, resampling EPISODES.

    Every readout is scored on the same resampled episodes in the same replicate, so
    `predicted_latent - identity_no_roll` is a paired difference: it answers "does the
    roll add anything over reading the head off the start state", which two separate
    intervals cannot answer.
    """
    n_rows = int(len(real))
    if n_rows < 3:
        return {"n_rows": n_rows}
    rows_by_ep = [np.nonzero(ep_index == e)[0] for e in range(n_eps)]
    present = [e for e in range(n_eps) if len(rows_by_ep[e]) > 0]
    draws: dict[str, dict[str, list[float]]] = {
        name: {"pearson": [], "spearman": []} for name in readouts
    }
    for _ in range(bootstrap):
        pick = rng.integers(0, len(present), size=len(present))
        rows = np.concatenate([rows_by_ep[present[p]] for p in pick])
        b = real[rows]
        for name, values in readouts.items():
            a = values[rows]
            pv, sv = pearson(a, b), spearman(a, b)
            draws[name]["pearson"].append(np.nan if pv is None else pv)
            draws[name]["spearman"].append(np.nan if sv is None else sv)
    out: dict[str, Any] = {
        "n_rows": n_rows,
        "n_episodes": len(present),
        "bootstrap_resamples": bootstrap,
        "readouts": {},
        "paired_deltas": {},
    }
    for name, values in readouts.items():
        node: dict[str, Any] = {
            "pearson": pearson(values, real),
            "spearman": spearman(values, real),
            "mse_vs_realised_phi": float(np.mean((values - real) ** 2)),
        }
        for stat in ("pearson", "spearman"):
            arr = np.asarray(draws[name][stat], dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                node[f"{stat}_ci95"] = ci(arr)
        out["readouts"][name] = node
    for left, right in paired:
        if left not in readouts or right not in readouts:
            continue
        for stat in ("pearson", "spearman"):
            la = np.asarray(draws[left][stat], dtype=np.float64)
            ra = np.asarray(draws[right][stat], dtype=np.float64)
            keep = np.isfinite(la) & np.isfinite(ra)
            if not keep.any():
                continue
            d = la[keep] - ra[keep]
            point_left = pearson(readouts[left], real) if stat == "pearson" \
                else spearman(readouts[left], real)
            point_right = pearson(readouts[right], real) if stat == "pearson" \
                else spearman(readouts[right], real)
            out["paired_deltas"].setdefault(f"{left}_minus_{right}", {})[stat] = {
                "delta": None if point_left is None or point_right is None
                else float(point_left - point_right),
                "ci95": ci(d),
                "tail_p_two_sided": tail_probability(d),
            }
    return out


def stage_b(
    pair: dict[str, Any],
    phi_head_path: Path,
    run_dirs: Sequence[Path],
    arm: str,
    horizon: int,
    bootstrap: int,
    seed: int,
    wm_holdout_keys: Sequence[str] | None,
    row_batch: int,
) -> dict[str, Any]:
    dims = tuple(int(x) for x in pair["dims"])
    zdim = dims[0]
    mu = pair["mu"].detach().float().cpu()
    sd = pair["sd"].detach().float().cpu()
    task = str(pair["task"])

    episodes = []
    run_records = []
    for run_dir in run_dirs:
        eps, record = load_run(run_dir, task, dims)
        episodes.extend(eps)
        run_records.append(record)
    bundle = build_bundle(episodes, mu, sd)

    bound_episode = torch.repeat_interleave(
        torch.arange(bundle.n_episodes), bundle.ep_len + 1
    )
    bound_chunk = torch.cat([torch.arange(int(n) + 1) for n in bundle.ep_len])

    # --- the deployed head, in the coordinates it was fitted in --------------------
    loaded = torch.load(phi_head_path, weights_only=False, map_location="cpu")
    head = make_phi_head(zdim)
    head.load_state_dict(loaded["state_dict"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    head_digest = state_dict_digest(head.state_dict())
    normaliser_digest = state_dict_digest({"mu": mu, "sd": sd})
    head_provenance = {
        "path": str(phi_head_path),
        "sha256": sha256_file(phi_head_path),
        "state_digest": head_digest,
        "recorded_digest": loaded.get("digest"),
        "zdim": int(loaded.get("zdim", -1)),
        "normaliser_digest_recomputed": normaliser_digest,
        "normaliser_digest_recorded": loaded.get("normaliser_digest"),
        "normaliser_match": normaliser_digest == loaded.get("normaliser_digest"),
    }

    # --- reproduction check 3: the head's own committed holdout correlations -------
    ranked = sorted(((stable_fraction(PHI_SEED, k), e) for e, k in enumerate(bundle.keys)))
    holdout_eps = {e for frac, e in ranked if frac < PHI_HOLDOUT}
    if not holdout_eps:
        holdout_eps.add(ranked[0][1])
    if len(holdout_eps) == len(ranked):
        holdout_eps.discard(ranked[-1][1])
    is_holdout = torch.tensor([int(e) in holdout_eps for e in bound_episode.tolist()])
    region = bound_chunk >= PHI_REGION_FROM_CHUNK
    with torch.no_grad():
        phi_hat = torch.cat(
            [
                head(bundle.z_bound[i : i + 1024]).squeeze(-1)
                for i in range(0, int(bundle.z_bound.shape[0]), 1024)
            ]
        )
    hold_region = torch.nonzero(is_holdout & region).flatten()
    hold_all = torch.nonzero(is_holdout).flatten()
    head_check = {
        "holdout_episodes": len(holdout_eps),
        "holdout_boundaries_region": int(hold_region.numel()),
        "holdout_boundaries_all": int(hold_all.numel()),
        "region_boundaries": int(region.sum()),
        "holdout_corr_region": pearson(
            phi_hat[hold_region].numpy(), bundle.phi_bound[hold_region].numpy()
        ),
        "holdout_corr_all_boundaries": pearson(
            phi_hat[hold_all].numpy(), bundle.phi_bound[hold_all].numpy()
        ),
        "holdout_spearman_region": spearman(
            phi_hat[hold_region].numpy(), bundle.phi_bound[hold_region].numpy()
        ),
        "holdout_mse_region": float(
            ((phi_hat[hold_region] - bundle.phi_bound[hold_region]) ** 2).mean()
        ),
    }

    # --- which episodes are held out from WHAT -------------------------------------
    # The head's split (seed 251, over the 192 bundle episodes) and the model pair's
    # split (seed 241, over role+env_seed groups) are DIFFERENT splits.  A row is
    # only fully held out when its episode sits outside both, so all three sets are
    # reported and the doubly-held-out one is the primary.
    short_to_index = {key: e for e, key in enumerate(bundle.keys)}
    wm_hold_idx: set[int] = set()
    unmatched: list[str] = []
    for key in wm_holdout_keys or []:
        sha, _, eid = key.partition(":")
        short = f"{sha[:12]}:{eid}"
        if short in short_to_index:
            wm_hold_idx.add(short_to_index[short])
        else:
            unmatched.append(key)
    if unmatched:
        raise SystemExit(f"WM holdout keys absent from the bundle: {unmatched[:5]}")
    head_hold_idx = {int(e) for e in holdout_eps}
    episode_sets = {
        "head_and_wm_holdout": sorted(head_hold_idx & wm_hold_idx),
        "head_holdout": sorted(head_hold_idx),
        "wm_holdout": sorted(wm_hold_idx),
    }

    # --- roll the transition and read the head off the PREDICTED latent ------------
    model = RawLatentWM(*dims)
    model.load_state_dict(pair["models"][arm], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    selected = sorted(set().union(*[set(v) for v in episode_sets.values()]))
    beliefs = episode_beliefs(model, bundle, torch.device("cpu"))

    rows_list: list[int] = []
    for e in selected:
        n_e = int(bundle.ep_len[e])
        if n_e < horizon:
            continue
        base = int(bundle.ep_row_start[e])
        rows_list.extend(range(base, base + n_e - horizon + 1))
    rows = torch.tensor(rows_list, dtype=torch.long)
    bound_now = bundle.row_boundary[rows]
    bound_future = bound_now + horizon

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, int(rows.numel()), row_batch):
            chunk = rows[i : i + row_batch]
            b0 = beliefs[chunk]
            z0 = bundle.z_bound[bundle.row_boundary[chunk]]
            offsets = torch.arange(horizon)
            ub = bundle.u_base[chunk.unsqueeze(1) + offsets.unsqueeze(0)]
            dl = bundle.delta[chunk.unsqueeze(1) + offsets.unsqueeze(0)]
            # delta = the recorded residual, so u_base + delta is the action the
            # deployment ACTUALLY executed: this is a prediction of the realised
            # future, not of a counterfactual one.
            z_pred = roll_latent(model, b0, z0, ub, dl, horizon)
            preds.append(head(z_pred).squeeze(-1))
            print(
                f"stage b: rolled {min(i + row_batch, int(rows.numel()))}"
                f"/{int(rows.numel())} windows",
                flush=True,
            )
    phi_pred = torch.cat(preds).numpy()
    phi_real = bundle.phi_bound[bound_future].numpy()
    phi_identity = phi_hat[bound_now].numpy()
    phi_oracle = phi_hat[bound_future].numpy()
    row_episode = bundle.row_episode[rows].numpy()
    target_chunk = bound_chunk[bound_future].numpy()

    rng = np.random.default_rng(seed)
    readouts = {
        "predicted_latent": phi_pred,
        "identity_no_roll": phi_identity,
        "oracle_real_future_latent": phi_oracle,
    }
    results: dict[str, Any] = {}
    for set_name, eps_in_set in episode_sets.items():
        member = np.isin(row_episode, np.asarray(eps_in_set, dtype=np.int64))
        for region_name, region_mask in (
            # The head was FITTED on boundaries at chunk >= 26 and is READ here at the
            # target boundary, so the region is defined by the target chunk t + Hc.
            ("decision_region", target_chunk >= PHI_REGION_FROM_CHUNK),
            ("all_boundaries", np.ones_like(target_chunk, dtype=bool)),
        ):
            mask = member & region_mask
            if not mask.any():
                continue
            local_eps = row_episode[mask]
            remap = {e: i for i, e in enumerate(sorted(set(local_eps.tolist())))}
            ep_index = np.asarray([remap[int(e)] for e in local_eps], dtype=np.int64)
            node = corr_bootstrap(
                {name: values[mask] for name, values in readouts.items()},
                phi_real[mask],
                ep_index,
                len(remap),
                rng,
                bootstrap,
                PAIRED_READOUTS,
            )
            node["realised_phi_variance"] = float(np.var(phi_real[mask]))
            node["realised_phi_mean"] = float(np.mean(phi_real[mask]))
            results.setdefault(set_name, {})[region_name] = node
    return {
        "arm": ARM_ALIAS[arm],
        "arm_internal": arm,
        "horizon_chunks": horizon,
        "actions": "recorded u_base + recorded delta = the executed action",
        "phi_head": head_provenance,
        "head_reproduction": head_check,
        "episode_sets": {k: len(v) for k, v in episode_sets.items()},
        "episode_set_note": (
            "head_holdout is episode-disjoint for the Phi' head (seed 251); "
            "wm_holdout is episode-disjoint for the transition (seed 241, "
            "role+env_seed groups); head_and_wm_holdout is the intersection and is "
            "the only set held out from both fits"
        ),
        "windows": int(rows.numel()),
        "runs": run_records,
        "results": results,
    }


# ------------------------------------------------------------------------------ main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pair", type=Path, default=DEFAULT_PAIR)
    ap.add_argument("--phi-head", type=Path, default=DEFAULT_PHI_HEAD)
    ap.add_argument("--base-tapes", type=Path, nargs="+", default=DEFAULT_BASE_TAPES)
    ap.add_argument("--update-tapes", type=Path, nargs="+", default=DEFAULT_UPDATE_TAPES)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--phi-horizon", type=int, default=10)
    ap.add_argument("--arm", default="updated", choices=list(ARMS))
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--phi-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=253)
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="max tolerated RELATIVE deviation from summary.json")
    ap.add_argument("--row-batch", type=int, default=128)
    ap.add_argument("--stage", choices=["a", "b", "all"], default="all")
    ap.add_argument("--threads", type=int, default=12,
                    help="the committed build used 12; CPU reduction order depends on it")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", default="evolve1")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    if a.threads:
        torch.set_num_threads(int(a.threads))
    torch.manual_seed(a.seed)

    pair_path = a.pair.resolve()
    committed_path = pair_path.parent / "summary.json"
    committed = json.loads(committed_path.read_text())
    pair = torch.load(pair_path, weights_only=False, map_location="cpu")
    if sha256_file(pair_path) != committed["checkpoint_sha256"]:
        raise SystemExit(f"{pair_path}: sha256 differs from the committed summary")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.out_root.resolve() / f"{a.tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "schema": "v253_wm_quality_v1",
        "utc": stamp,
        "task": str(pair["task"]),
        "dims": [int(x) for x in pair["dims"]],
        "axis": "object (T_theta rolled forward, heads read off the predicted latent)",
        "env_steps": 0,
        "device": "cpu",
        "model_pair": str(pair_path),
        "model_pair_sha256": committed["checkpoint_sha256"],
        "preregistered": False,
        "preregistration_note": (
            "the round's preregistration.json governs the DEPLOYMENT endpoints; these "
            "are post-hoc model-side diagnostics with descriptive bootstrap intervals "
            "and no multiplicity correction"
        ),
        "provenance": {
            "argv": sys.argv,
            "cli": shlex.join(sys.argv),
            "git": git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_value("status", "--short")),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": int(a.threads),
            "seed": a.seed,
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    wm_keys: list[str] | None = None
    if a.stage in ("a", "all"):
        result = stage_a(
            pair,
            a.base_tapes,
            a.update_tapes,
            committed,
            sorted(a.horizons),
            a.bootstrap,
            a.seed,
            a.tolerance,
        )
        report["stage_a"] = result
        wm_keys = sorted(
            result["holdout_episode_keys"]["base_holdout"]
            + result["holdout_episode_keys"]["update_holdout"]
        )
        rep = result["reproduction"]
        print(
            f"reproduction: {rep['fields_checked']} fields, max rel deviation "
            f"{rep['max_rel_delta']:.3e}, PASS={rep['passed']}",
            flush=True,
        )
        if not rep["passed"]:
            raise SystemExit("point estimates do not reproduce summary.json; stopping")

    if a.stage in ("b", "all"):
        if wm_keys is None:
            split = committed["provenance"]["split"]
            wm_keys = sorted(
                [r["key"] for r in split["base_holdout"]]
                + [r["key"] for r in split["update_holdout"]]
            )
        run_dirs = [p.parent if p.name == "tape.pt" else p
                    for p in list(a.base_tapes) + list(a.update_tapes)]
        report["stage_b"] = stage_b(
            pair,
            a.phi_head.resolve(),
            run_dirs,
            a.arm,
            a.phi_horizon,
            a.phi_bootstrap,
            a.seed + 1,
            wm_keys,
            a.row_batch,
        )

    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
