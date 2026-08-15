#!/usr/bin/env python
"""V7.5 Stage 3 — the binding pre-D gate on the frozen V7.5C checkpoint.

Registered in plan_and_progress/2026-08-09.md ("V7.5 Stages 2-5 and
routing"), with the binding quantity fixed by the Stage 1b finding.

MEASUREMENT SPACE. T1/T4/T5 are evaluated on the CENTERED POOLED
interface vector `c_t` -- the quantity the policy actually consumes --
because that is the space the inherited V7.4A thresholds (0.30, 0.5x,
1.5x) were registered in. T2/T3 are evaluated as RELATIVE token-space
contrasts, because that is the space `HIST_GATE = 1e-4` was registered
in and the space the V7.5C per-epoch `hist_rel` readout measured the
recurrence collapse in. Both are reported for T2 so the two records
stay comparable.

  T1 identity      BINDING. Median pairwise distance over distinct real
                   anchors. Failure = content collapse, the defect
                   class that invalidated V7.3D.
  T2 history       BINDING for V7.5 (the change vs V7.4, which had it
                   measured-only). Stage 1b showed a different-anchor
                   contrast passes on a channel that is open but
                   carries nothing; only the reset contrast separates
                   "live recurrence" from "live wiring".
  T3 order         MEASURED.
  T4 paraphrase    BINDING where crossed support exists.
  T5 goal sep      BINDING where crossed support exists.
  rollout          MEASURED: the registered rollout term's true-vs-
                   shuffled margin at the final checkpoint.

gate_pass = T1 and T2 and (T4/T5 where supported). A false verdict
halts V7.5 before Stages 4-5 with the status named.

SOLE PRODUCER of the policy-side artifacts (the V7.4 architecture
decision): `anchor_states_v075.pt` (pre-pool token states for every
anchor x {real, reset, shuffled} plus every train demo-rehearsal row)
and `centering_v075_token_query.pt`, both sha-bound in the report.
The TokenQueryPool init is sha-seeded from the policy trainer's
namespace so the gate's pools are byte-identical to every arm's init.

Demo states unroll `ep["obs_10"]` at the WM's 10-action decision
stride, NOT the 50-stride `rows` -- the V7.3D defect.

Output: <date>_v075_gate_r1/
Tensor-only: reads the policy-stage h-cache, runs no environment and
no policy forward.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v075_schedule as S  # noqa: E402
from lcwm.self_predict import LatentPredictor, latent_distance  # noqa: E402
from lcwm.v067_lineage import load_v067, sha256_file  # noqa: E402
from lcwm.v074_interface import CenteredState, TokenQueryPool  # noqa: E402
from lcwm.v075_state import V075State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RID = "v075_gate_r1"
GATE_MODE = "v075_binding_gate"
STATE_SCHEMA = "v075_policy_states_v1"
INTERFACE = "token_query"
TQP_SEED_NS = "v075_policy_r1|tqp"
# frozen BEFORE the gate runs
T1_MIN = 0.30
HIST_GATE = 1e-4        # relative token-space reset contrast
EPS_V74 = 0.05          # the v074 absolute centered-pooled rule
D_MIN = 5
T4_MAX_RATIO = 0.5
T5_MIN_RATIO = 1.5
TBPTT, ROLL_K, ROLL_STARTS = 16, 3, 4


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def sha_seed(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcwm", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    lcwm_root = None
    for p in sorted(RESULTS.glob("*_v075_lcwm_r1")):
        lcwm_root = p
    if lcwm_root is None:
        sys.exit("no *_v075_lcwm_r1 root — run train_v075_lcwm.py first")
    ckpt = args.lcwm or (lcwm_root / "checkpoints" / "final.pt")
    if not ckpt.exists():
        sys.exit(f"{ckpt} missing — V7.5C has not produced final.pt")
    out = RESULTS / f"{run_date()}_{RID}"
    report_path = out / "gate_report.json"
    if report_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {report_path} (pass --force)")
    out.mkdir(parents=True, exist_ok=True)

    hman = json.loads((lcwm_root / "hcache_manifest.json").read_text())
    if hman.get("stage") != "policy":
        sys.exit("the h-cache was built with --stage lcwm; the gate "
                 "needs the teach anchors and demo rows: rerun "
                 "scripts/build_v075_hcache.py --stage policy")
    hidx = json.loads((lcwm_root / "hcache_index.json").read_text())

    # ---- the complete policy anchor set: acq (train+dev) + teach -----
    man = json.loads(
        (S.v074_root() / "anchor_manifest.json").read_text())
    assert man.get("schema") == "v074_anchor_manifest_v1"
    anchors, n_budget = {}, 0
    for a in man["anchors"]:
        if a["kind"] not in ("acq", "teach"):
            n_budget += 1
            continue
        ak = f"{a['source_id']}_d{int(a['decision'])}"
        assert ak not in anchors, f"duplicate manifest anchor {ak}"
        anchors[ak] = {"task": a["task"], "source_id": a["source_id"],
                       "decision": int(a["decision"]),
                       "kind": a["kind"], "role": a["role"]}
    aks = sorted(anchors)
    counts = defaultdict(int)
    for r in anchors.values():
        counts[(r["kind"], r["role"])] += 1
    assert dict(counts) == {("acq", "train"): 24, ("acq", "dev"): 9,
                            ("teach", "teacher"): 9}, dict(counts)

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = V075State().to(device)
    model.load_state_dict(ck["model"])
    model.eval()          # freezes hist_center: the gate never observes
    pred = LatentPredictor().to(device)
    pred.load_state_dict(ck["predictor"])
    pred.eval()
    assert not model.hist_center.training

    h_lru, src_lru = {}, {}

    def hget(key):
        if key not in h_lru:
            if len(h_lru) > 300:
                h_lru.clear()
            h_lru[key] = torch.load(hidx[key], weights_only=False)
        d = h_lru[key]
        return (d["h"][None].float().to(device),
                d["mask"][None].to(device))

    def rows_of(sid):
        if sid not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[sid] = torch.load(
                S.sources_for(S.UNIVERSE) / f"{sid}.pt",
                weights_only=False)["rows"]
        return src_lru[sid]

    def act_of(row):
        aa = row["chunk_norm"][None, :10].float().to(device)
        am = (torch.arange(10, device=device)[None]
              < row["executed_len"])
        return aa, am

    @torch.no_grad()
    def unroll(ak, tvv, order=None, collect=False):
        """State at the anchor. `order` permutes the PREFIX decisions
        only (order-only control: same length, same (obs, action)
        pairs, same terminal anchor observation)."""
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        rows_ = rows_of(sid)
        seq = list(range(d)) if order is None else list(order)
        z, trace, acts = None, [], []
        for pos, dd in enumerate(seq):
            h, m = hget(f"{tvv}::src::{sid}::d{dd}")
            if z is None:
                z = model.initial_state(h, m)
            else:
                aa, am = act_of(rows_[seq[pos - 1]])
                z = model.step(z, aa, h, m, action_mask=am)
            if collect:
                trace.append(z)
                if pos < len(seq) - 1:
                    acts.append(act_of(rows_[dd]))
        h, m = hget(f"{tvv}::src::{sid}::d{d}")
        if z is None:
            z = model.initial_state(h, m)
        else:
            aa, am = act_of(rows_[seq[-1]])
            z = model.step(z, aa, h, m, action_mask=am)
        if collect:
            if trace:
                acts.append((aa, am))
            trace.append(z)
            return z, trace, acts
        return z

    @torch.no_grad()
    def reset_state(ak, tvv):
        a = anchors[ak]
        h, m = hget(f"{tvv}::src::{a['source_id']}::d{a['decision']}")
        return model.initial_state(h, m)

    @torch.no_grad()
    def demo_tokens(ep, tid, di, ri):
        """Demo tokens at row ri, unrolled at the WM's OWN 10-action
        decision stride over ep["obs_10"] (the V7.3D construction)."""
        target_t = 50 * ri
        seq = [o for o in ep["obs_10"] if o["t"] <= target_t]
        z = None
        for j, o in enumerate(seq):
            h, m = hget(f"demo::{tid}::{di}::t{o['t']}")
            if z is None:
                z = model.initial_state(h, m)
            else:
                ae = torch.as_tensor(
                    np.asarray(seq[j - 1]["chunk10_env"])).float()
                aa = ep_norm(ep, ae)[None].to(device)
                n_a = aa.shape[1]
                if n_a < 10:
                    aa = torch.cat([aa, torch.zeros(
                        1, 10 - n_a, 7, device=device)], dim=1)
                am = (torch.arange(10, device=device)[None] < n_a)
                z = model.step(z, aa[:, :10], h, m, action_mask=am)
        return z

    def ep_norm(ep, a_env):
        mean = torch.as_tensor(ep["action_mean"]).float()
        std = torch.as_tensor(ep["action_std_eps"]).float()
        return (a_env - mean) / std

    # ---- build every token state -------------------------------------
    tokens, text_goal = {}, defaultdict(dict)
    for ak in aks:
        a = anchors[ak]
        cp0 = f"{S.canon_goal(a['task'])}_p0"
        tokens[f"real::{ak}"] = unroll(ak, cp0).cpu()
        tokens[f"reset::{ak}"] = reset_state(ak, cp0).cpu()
        rng = np.random.default_rng(sha_seed(f"{RID}|shuffle|{ak}"))
        tokens[f"shuffled::{ak}"] = (
            unroll(ak, cp0, order=list(rng.permutation(a["decision"])))
            if a["decision"] > 1 else tokens[f"real::{ak}"].clone()).cpu()
        if a["kind"] == "acq":      # teach anchors stay uncrossed
            for g, tvv in S.queries_for(a):
                if tvv == cp0:
                    continue
                tokens[f"text::{ak}::{tvv}"] = unroll(ak, tvv).cpu()
                text_goal[ak][tvv] = g
    print(f"[gate] {len(aks)} anchors, {len(tokens)} anchor states",
          flush=True)

    dman = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_keys = []
    for e in dman["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] != "train":
            continue
        tid, di = e["task_id"], e["demo"]
        for ri in range(len(ep["rows"])):
            k = "demo::" + json.dumps([tid, di, ri])
            tokens[k] = demo_tokens(ep, tid, di, ri).cpu()
            demo_keys.append(k)
    print(f"[gate] {len(demo_keys)} demo-rehearsal states", flush=True)

    # ---- attend (trainer-seeded init) + center ------------------------
    tqp = TokenQueryPool(384, seed=sha_seed(TQP_SEED_NS)).eval()
    with torch.no_grad():
        pools = {k: tqp(v.float())[0] for k, v in tokens.items()}
    center = CenteredState.fit([pools[f"real::{ak}"] for ak in aks]
                               + [pools[k] for k in demo_keys])
    cvec = {k: center.apply(v) for k, v in pools.items()}

    def cd(ka, kb):
        return float(torch.linalg.vector_norm(cvec[ka] - cvec[kb]))

    def rel_tok(ka, kb):
        a_, b_ = tokens[ka].float(), tokens[kb].float()
        return float((a_ - b_).norm() / a_.norm().clamp_min(1e-12))

    # ---- T1 (centered pooled) ----------------------------------------
    pair = [cd(f"real::{i}", f"real::{j}")
            for x, i in enumerate(aks) for j in aks[x + 1:]]
    t1_med = med(pair)
    t1 = {"space": "centered_pooled", "median": t1_med, "n": len(pair),
          "threshold": T1_MIN, "binding": True,
          "pass": bool(t1_med >= T1_MIN)}

    # ---- T2 / T3 -------------------------------------------------------
    sup = [ak for ak in aks if anchors[ak]["decision"] >= D_MIN]
    t2_rel = [rel_tok(f"real::{ak}", f"reset::{ak}") for ak in sup]
    t3_rel = [rel_tok(f"real::{ak}", f"shuffled::{ak}") for ak in sup]
    t2_pool = [cd(f"real::{ak}", f"reset::{ak}") for ak in sup]
    t3_pool = [cd(f"real::{ak}", f"shuffled::{ak}") for ak in sup]
    t2 = {"space": "relative_token", "median": med(t2_rel),
          "n": len(t2_rel), "threshold": HIST_GATE, "binding": True,
          "quantity": "||z_real - z_reset|| / ||z_real||",
          "why_binding": "Stage 1b: a different-anchor contrast passes "
                         "on an open channel that carries nothing",
          "centered_pooled_median": med(t2_pool),
          "v074_rule_n_pass": sum(1 for v in t2_pool if v >= EPS_V74),
          "v074_rule_n_support": len(sup),
          "pass": bool(med(t2_rel) is not None
                       and med(t2_rel) >= HIST_GATE)}
    t3 = {"space": "relative_token", "median": med(t3_rel),
          "n": len(t3_rel), "binding": False,
          "centered_pooled_median": med(t3_pool),
          "v074_rule_n_pass": sum(1 for v in t3_pool if v >= EPS_V74)}

    # ---- T4 / T5 (centered pooled) -------------------------------------
    para, sep = [], []
    for ak in aks:
        cg = S.canon_goal(anchors[ak]["task"])
        for tvv, g in text_goal[ak].items():
            (para if g == cg else sep).append(
                cd(f"real::{ak}", f"text::{ak}::{tvv}"))
    p_med, s_med = med(para), med(sep)
    t4 = {"space": "centered_pooled", "median": p_med, "n": len(para),
          "limit": (T4_MAX_RATIO * t1_med) if t1_med else None,
          "binding": bool(para),
          "pass": bool(para and p_med <= T4_MAX_RATIO * t1_med)}
    t5 = {"space": "centered_pooled", "median": s_med, "n": len(sep),
          "limit": (T5_MIN_RATIO * p_med) if p_med else None,
          "ratio_to_paraphrase": (s_med / p_med)
          if (p_med and p_med > 0) else None,
          "binding": bool(sep and para),
          "pass": bool(sep and para and s_med >= T5_MIN_RATIO * p_med)}

    # ---- rollout margin at the final checkpoint (measured) -------------
    r_true, r_ctrl = [], []
    with torch.no_grad():
        for ak in aks:
            cp0 = f"{S.canon_goal(anchors[ak]['task'])}_p0"
            _z, trace, acts = unroll(ak, cp0, collect=True)
            n = len(trace)
            last = n - 1 - ROLL_K
            if last < 0 or not acts:
                continue
            firstj = max(0, min(last, n - 1 - TBPTT))
            cand = list(range(firstj, last + 1))
            if len(cand) > ROLL_STARTS:
                stp = (len(cand) - 1) / (ROLL_STARTS - 1)
                cand = sorted({cand[min(len(cand) - 1, round(i * stp))]
                               for i in range(ROLL_STARTS)})
            hc = model.hist_center
            for j in cand:
                zr = trace[j]
                for i in range(ROLL_K):
                    if j + i >= len(acts):
                        break
                    aa, am = acts[j + i]
                    zr = model.t(zr, model.e_a(aa, am))
                    pz = hc.apply_frozen(pred(zr))
                    r_true.append(float(latent_distance(
                        pz, hc.apply_frozen(trace[j + i + 1]),
                        "cosine").mean()))
                    far = (j + i + 1 + n // 2) % n
                    r_ctrl.append(float(latent_distance(
                        pz, hc.apply_frozen(trace[far]),
                        "cosine").mean()))
    roll = {"n": len(r_true), "binding": False,
            "true": float(np.mean(r_true)) if r_true else None,
            "shuffled": float(np.mean(r_ctrl)) if r_ctrl else None,
            "margin": float(np.mean(r_ctrl) - np.mean(r_true))
            if r_true else None,
            "rule": "shuffled MINUS true; <= 0 means the registered "
                    "rollout term trained nothing"}

    binding = [t1["pass"], t2["pass"]]
    binding += [t["pass"] for t in (t4, t5) if t["binding"]]
    gate_pass = all(binding)

    # ---- artifacts, sha-bound ------------------------------------------
    spath = out / "anchor_states_v075.pt"
    cpath = out / f"centering_v075_{INTERFACE}.pt"
    lcwm_sha = sha256_file(ckpt)
    torch.save({"schema": STATE_SCHEMA, "lcwm_sha": lcwm_sha,
                "tokens": tokens}, spath)
    center.save(cpath)
    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO_ROOT).stdout.strip()
    rows = [{"anchor": ak, **anchors[ak],
             "d_reset_rel": rel_tok(f"real::{ak}", f"reset::{ak}"),
             "d_shuffled_rel": rel_tok(f"real::{ak}", f"shuffled::{ak}"),
             "d_reset_pooled": cd(f"real::{ak}", f"reset::{ak}"),
             "in_history_support": anchors[ak]["decision"] >= D_MIN,
             "n_crossed_texts": len(text_goal[ak])} for ak in aks]
    report = {
        "schema": "v075_gate_report_v1", "mode": GATE_MODE,
        "interface": INTERFACE, "run_id": RID, "git_sha": git_sha,
        "lcwm": str(ckpt.relative_to(REPO_ROOT)),
        "lcwm_sha256": lcwm_sha,
        "universe": S.UNIVERSE, "n_anchors": len(aks),
        "n_budget_rows_skipped": n_budget,
        "n_demo_states": len(demo_keys),
        "tqp_seed_namespace": TQP_SEED_NS,
        "measurement_spaces": {
            "T1_T4_T5": "centered pooled c_t (the V7.4A threshold "
                        "space, and what the policy consumes)",
            "T2_T3": "relative token contrast (the space HIST_GATE "
                     "and the V7.5C hist_rel readout live in); the "
                     "v074 absolute centered-pooled rule is reported "
                     "alongside for comparability"},
        "centering": {"path": cpath.name, "sha256": sha256_file(cpath),
                      "sigma": center.sigma,
                      "mu_norm": float(center.mu.norm()),
                      "fit_set": "real anchor pools + train demo pools"},
        "anchor_states": {"path": spath.name,
                          "sha256": sha256_file(spath),
                          "schema": STATE_SCHEMA},
        "thresholds": {"T1_MIN": T1_MIN, "HIST_GATE": HIST_GATE,
                       "EPS_V74": EPS_V74, "D_MIN": D_MIN,
                       "T4_MAX_RATIO": T4_MAX_RATIO,
                       "T5_MIN_RATIO": T5_MIN_RATIO},
        "T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5,
        "rollout": roll, "gate_pass": gate_pass,
        "routing": ("Stages 4-5 may proceed" if gate_pass else
                    "HALT before Stages 4-5 — registered cheap "
                    "negative; name the failing item, do not sweep"),
        "anchor_rows": rows}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"[gate] -> {report_path}")
    print(f"  centering: ||mu||={float(center.mu.norm()):.4f} "
          f"sigma={center.sigma:.6e} "
          f"(fit on {len(aks)} anchor + {len(demo_keys)} demo pools)")
    for name, t in (("T1 identity  ", t1), ("T2 history   ", t2),
                    ("T3 order     ", t3), ("T4 paraphrase", t4),
                    ("T5 goal sep  ", t5)):
        v = t["median"]
        print(f"  {name} {'BINDING' if t['binding'] else 'measured':8s} "
              f"median={(f'{v:.6g}' if v is not None else 'unsup'):>12} "
              f"n={t['n']:<4} "
              f"{'PASS' if t.get('pass') else ('FAIL' if t['binding'] else '--')}")
    if roll["n"]:
        print(f"  rollout       measured true={roll['true']:.6f} "
              f"shuffled={roll['shuffled']:.6f} "
              f"margin={roll['margin']:+.6f}")
    print(f"\n[gate] gate_pass={gate_pass} — {report['routing']}")


if __name__ == "__main__":
    main()
