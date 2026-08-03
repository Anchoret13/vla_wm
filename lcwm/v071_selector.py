"""V7.1.4A — compiled frozen selector (2026-08-03.md).

Pure functions over ensemble predictions; no I/O, no simulator. The
outcome ensemble and LCWM stay byte-frozen; this module only compiles
the registered decision rule.

Prediction vector layout (from scripts/train_v071_3_outcome.py):
  [0] success_by_100 (prob)   MASKED from selection (constant labels)
  [1..4] q@{10,30,60,100}     diagnostic only, never selective
  [5] q_valid_mean            supported component 2
  [6] p_valid_100             supported component 1
  [7] neg_damage/10           MASKED from selection (constant labels)
  [8] neg_tau_next/100        supported component 3

Paired differences use the SAME ensemble member on candidate and stock
sides: Delta_{i,d}^{(k)} = yhat_d^{(k)}(z,u_i) - yhat_d^{(k)}(z,u_0);
L/U = mean_k +- std_k. With K=5 this is a frozen ensemble margin band /
LCB heuristic, not a calibrated confidence interval; the word
"confidence" below names the registered rule only.

Domain projection before any comparison: p_valid_100, q_valid_mean,
q@h -> [0,1]; normalized neg_tau_next -> [-1.01, 0]. Raw values are
retained in the ledger row.
"""

from __future__ import annotations

import hashlib

import numpy as np

MASKED = {"success_by_100": 0, "neg_damage": 7}
DIAGNOSTIC_Q_IDX = [1, 2, 3, 4]
# registered supported lexicographic order (name, index, tolerance in
# projected/normalized units — frozen replay-noise tolerances; tau's
# raw tolerance 7.15 is divided by the registered /100 target scale)
SELECT_ORDER = (("p_valid_100", 6, 0.0),
                ("q_valid_mean", 5, 0.041666666666666664),
                ("neg_tau_next", 8, 0.0715))


def project(vec: np.ndarray) -> np.ndarray:
    """Project one 9-dim prediction to registered target domains."""
    v = np.asarray(vec, dtype=np.float64).copy()
    v[0] = np.clip(v[0], 0.0, 1.0)
    for i in DIAGNOSTIC_Q_IDX:
        v[i] = np.clip(v[i], 0.0, 1.0)
    v[5] = np.clip(v[5], 0.0, 1.0)
    v[6] = np.clip(v[6], 0.0, 1.0)
    v[7] = np.clip(v[7], -1.01, 0.0)   # neg_damage/10, nonpositive
    v[8] = np.clip(v[8], -1.01, 0.0)
    return v


def paired_band(cand: np.ndarray, stock: np.ndarray,
                idx: int) -> dict:
    """Same-member paired difference band for one component.
    cand/stock: (K, 9) PROJECTED member predictions."""
    delta = cand[:, idx] - stock[:, idx]
    m, s = float(delta.mean()), float(delta.std())
    return {"mean": m, "std": s, "L": m - s, "U": m + s}


def sha_rank(run_id: str, source_id: str, decision: int,
             candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(
        f"{run_id}|{source_id}|{decision}|{candidate_id}"
        .encode()).digest()[:8], "big")


def classify_candidate(cand_proj: np.ndarray,
                       stock_proj: np.ndarray) -> dict:
    """Ternary lexicographic confidence rule vs stock.

    Returns eligibility, per-component bands, decisive component.
    A masked head can never appear here — SELECT_ORDER excludes it by
    construction; an assertion still guards the invariant.
    """
    bands = {}
    for name, idx, tol in SELECT_ORDER:
        assert name not in MASKED, "masked head in selector order"
        b = paired_band(cand_proj, stock_proj, idx)
        b["tol"] = tol
        bands[name] = b
        if b["L"] > tol:
            return {"eligibility": "confidence_positive",
                    "decisive_component": name, "bands": bands}
        if b["U"] < -tol:
            return {"eligibility": "ineligible_negative",
                    "decisive_component": name, "bands": bands}
        if -tol <= b["L"] and b["U"] <= tol:
            continue                      # supported tie -> next
        return {"eligibility": "ineligible_uncertain",
                "decisive_component": name, "bands": bands}
    return {"eligibility": "tied_ineligible",
            "decisive_component": None, "bands": bands}


def lex_compare(a_mean: np.ndarray, b_mean: np.ndarray) -> int:
    """Lexicographic comparison of PROJECTED ensemble means over the
    supported components with frozen tolerances."""
    for _name, idx, tol in SELECT_ORDER:
        if a_mean[idx] > b_mean[idx] + tol:
            return 1
        if b_mean[idx] > a_mean[idx] + tol:
            return -1
    return 0


def select(preds: dict, stock_id: str, *, run_id: str,
           source_id: str, decision: int) -> dict:
    """Frozen selector over one anchor's candidate bank.

    preds: candidate_id -> (K, 9) RAW ensemble member predictions
    (stock_id must be a key). Returns the pre-outcome ledger row body.
    """
    assert stock_id in preds
    raw = {c: np.asarray(p, dtype=np.float64)
           for c, p in preds.items()}
    proj = {c: np.stack([project(m) for m in p])
            for c, p in raw.items()}
    rows = {}
    positives = []
    for cid in sorted(preds):
        if cid == stock_id:
            continue
        cls = classify_candidate(proj[cid], proj[stock_id])
        assert cls["decisive_component"] not in MASKED, \
            "masked head decisive"
        rows[cid] = {
            **cls,
            "raw_mean": raw[cid].mean(0).tolist(),
            "raw_std": raw[cid].std(0).tolist(),
            "proj_mean": proj[cid].mean(0).tolist(),
        }
        if cls["eligibility"] == "confidence_positive":
            positives.append(cid)
    if not positives:
        selected, reason = stock_id, "abstain_no_confidence_positive"
    else:
        best = [positives[0]]
        for cid in positives[1:]:
            c = lex_compare(proj[cid].mean(0), proj[best[0]].mean(0))
            if c > 0:
                best = [cid]
            elif c == 0:
                best.append(cid)
        selected = min(best, key=lambda c: sha_rank(
            run_id, source_id, decision, c))
        reason = ("sha_tie_break" if len(best) > 1
                  else "lex_max_confidence_positive")
    return {
        "run_id": run_id, "source_id": source_id,
        "decision": decision, "stock_id": stock_id,
        "selected": selected,
        "intervention": selected != stock_id,
        "selection_reason": reason,
        "n_confidence_positive": len(positives),
        "stock_raw_mean": raw[stock_id].mean(0).tolist(),
        "stock_proj_mean": proj[stock_id].mean(0).tolist(),
        "candidates": rows,
        "masked_components": sorted(MASKED),
        "select_order": [n for n, _i, _t in SELECT_ORDER],
    }
