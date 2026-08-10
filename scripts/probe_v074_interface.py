#!/usr/bin/env python
"""V7.4A — complete-anchor content assertion probe (registered
instrument: plan_and_progress/2026-08-09.md V7.4A, thresholds T1-T5).

--universe v073 is the A-STAGE DIAGNOSTIC: the instrument runs on the
frozen V7.3C LCWM over the COMPLETE V7.3 real-anchor set — all 6
unique grounded-ledger anchors PLUS all 9 model-ledger anchors
INCLUDING the 8 abstentions — as instrument validation, report-only
(the 2026-08-07 audit numbers are prior data). Per anchor it builds
the real recurrent pool (exact train_v073_policy.py semantics:
executed_len masking, prefix caching), the reset pool (initial_state
at the anchor observation), the order-only shuffled pool (V7.3D
construction, seed namespace v074_probe|shuffle|<ak>), and — for
v073-family anchors — one pool per (goal, text) crossed query
resolved through the V7.3B semantic_query_index (unresolvable pairs
are recorded, never guessed; v072T-family anchors are canonical-only
and excluded from T4/T5 support). Everything is centered by the
registered CenteredState fitted uniformly on real anchor pools + the
STORED demo-rehearsal pools (never recomputed here), then:

  T1 identity    median pairwise distinct-real-anchor dist >= 0.30
  T2 history     ||c_real - c_reset||    >= 0.05 for >= 2/3 (d >= 5)
  T3 order       ||c_real - c_shuffled|| >= 0.05 for >= 1/3 (d >= 5)
  T4 paraphrase  median same-goal text dist <= 0.5 x T1 median
  T5 goals       median alt-goal dist >= 1.5 x paraphrase median

Registered A-stage consequence (recorded here, enforced by the V7.4
execution order): the pooled interface failing T1 or T2 commits V7.4
to the fallback interface. --interface token_query pools with a FRESH
TokenQueryPool under torch.manual_seed(0): an UNTRAINED query is a
LOWER BOUND on the fallback, not the trained fallback itself.

Also consumes lcwm.v074_policy_schedule as a dry run over the real
ledger anchors (diagnostic-only exposure-cap validation ahead of
V7.4D). Output under --out: probe_report_<interface>.json,
run_manifest_<interface>.json, centering_<interface>.pt (the pooled
and token_query runs of the registered two-run procedure never
collide), plus ONE shared interface-independent probe_states.pt
token cache reused by both runs.

--universe v074 is the BINDING PRE-D GATE under the dated pre-C
amendment (2026-08-09.md "V7.4A EXECUTED"): the instrument runs on
the frozen V7.4C final checkpoint over the COMPLETE V7.4
policy-state set — every acq (train + dev) and teach anchor of the
frozen V7.4B anchor manifest (SHORTFALL / family_shortfall budget
rows skipped, never refilled) plus the demo-rehearsal states
RECOMPUTED with the probed checkpoint (the stored v073 demo pools
belong to the retired V7.3C checkpoint). --interface token_query is
REQUIRED: the A-stage pooled run recorded consequence
commit_fallback_token_query and the registered fallback switch is
SPENT. The TokenQueryPool init is sha-seeded from the policy
trainer's namespace so the gate and every V7.4D arm start
byte-identical. AMENDED verdict: gate_pass = T1 pass AND (T4 pass
or unsupported) AND (T5 pass or unsupported); T2/T3 are computed
and recorded measured-and-named next to the A-stage pooled values,
never gating. Outputs into the SAME dated interface_diag root:
probe_report_v074_token_query.json (mode "binding_gate"),
run_manifest_v074_token_query.json, centering_v074_token_query.pt,
probe_states_v074.pt (resumable token cache) and
anchor_states_v074.pt in EXACTLY the trainer's
v074_policy_states_v1 token schema, consumed by V7.4D instead of a
trainer-built cache.
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

from lcwm import v073_schedule as S  # noqa: E402
from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v074_interface import (CenteredState,  # noqa: E402
                                 TokenQueryPool)
from lcwm.v074_policy_schedule import (  # noqa: E402
    build_matched_policy_schedule)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
TEACH = RESULTS / "2026-08-07_v073_teacher_r1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
POLICY = RESULTS / "2026-08-07_v073_policy_r1"
LCWM_FINAL = RESULTS / "2026-08-06_v073_lcwm_r1" / "checkpoints" \
    / "final.pt"
OUT_DEFAULT = RESULTS / "2026-08-09_v074_interface_diag_r1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
SEQ_REF = Path("/home/stargazer/Desktop/vla_wm/datasets"
               "/seq_prefix_cache_v1/task0_demo0.pt")
RID = "v074_probe"
STATES_SCHEMA = "v074_probe_states_v1"
REPORT_SCHEMA = "v074_probe_report_v1"
STATE_SCHEMA_V73 = "v073_policy_states_v2"
STATE_SCHEMA_V74 = "v074_policy_states_v1"
# BYTE-MATCH scripts/train_v074_policy.py: its RID "v074_policy_r1"
# and tqp_seed = sha_seed(f"{RID}|tqp") — the gate's TokenQueryPool
# init must be identical to every V7.4D arm's init.
TQP_SEED_NS = "v074_policy_r1|tqp"
D_MIN, EPS, T1_MIN = 5, 0.05, 0.30
SCHED_STEPS = 300


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def load_ledger_anchors():
    """Real-anchor records keyed by bare anchor id: the 6 unique
    grounded-ledger anchors + ALL 9 model-ledger rows (abstentions
    included). Source loading mirrors train_v073_policy.py: v073
    family from V73/sources, v072T family from V72T/teacher_sources."""
    anchors, g_origin = {}, {}
    for g in [json.loads(x) for x in
              (TEACH / "grounded_ledger.jsonl").open()]:
        ak = g["anchor"]
        assert g["origin"] in ("v073B", "v072T_released"), g["origin"]
        g_origin[ak] = g["origin"]
        rec = anchors.setdefault(ak, {
            "task": g["task"], "source_id": ak.rsplit("_d", 1)[0],
            "decision": int(ak.rsplit("_d", 1)[1]),
            "family": ("v073" if g["origin"] == "v073B"
                       else "v072T"), "roles": []})
        if "grounded" not in rec["roles"]:
            rec["roles"].append("grounded")
    n_grounded = len(anchors)
    assert n_grounded == 6, f"grounded anchors {n_grounded} != 6"
    model_rows = [json.loads(x) for x in
                  (TEACH / "model_ledger_pre_outcome.jsonl").open()]
    assert len(model_rows) == 9, f"model rows {len(model_rows)} != 9"
    for r in model_rows:
        ak = r["anchor"]
        assert ak == f"{r['source_id']}_d{r['decision']}", ak
        rec = anchors.setdefault(ak, {
            "task": r["task"], "source_id": r["source_id"],
            "decision": r["decision"], "family": "v073",
            "roles": []})
        rec["roles"].append(r["mode"])
    assert len(anchors) == n_grounded + 9, \
        "grounded/model anchor sets overlap"
    return anchors, g_origin, model_rows


def load_text_index(index_path=V73 / "semantic_query_index.jsonl"):
    """PER-ANCHOR verified (goal, text) crossing from a semantic
    query index (V7.3B by default; the v074 gate passes the V7.4B
    index): anchor -> {tvid: goal_id}, plus the (anchor-invariant)
    tvid -> prompt string map. The index rows carry the anchor
    dimension — only pairs the build verified AT an anchor count as
    crossed there (seam review: the text_variant-granular reading
    extrapolated crossing to the 9 teacher-source anchors the index
    never covered)."""
    lang, by_anchor = {}, defaultdict(dict)
    for line in index_path.open():
        row = json.loads(line)
        tv = row["text_variant_id"]
        assert lang.get(tv, row["language"]) == row["language"], \
            f"conflicting language strings for {tv}"
        lang[tv] = row["language"]
        by_anchor[row["anchor"]][tv] = row["goal_id"]
    return lang, dict(by_anchor)


def v074_lcwm_default() -> Path:
    """Newest *_v074_lcwm_r1 final checkpoint (sorted-glob rule)."""
    root = None
    for p in sorted(RESULTS.glob("*_v074_lcwm_r1")):
        root = p
    ckpt = None if root is None else root / "checkpoints" \
        / "final.pt"
    if ckpt is None or not ckpt.exists():
        sys.exit("--universe v074 default --lcwm not found: no "
                 f"{RESULTS}/*_v074_lcwm_r1/checkpoints/final.pt — "
                 "the binding gate runs on the frozen V7.4C "
                 "final-step checkpoint; run train_v074_lcwm.py to "
                 "completion first (or pass --lcwm explicitly)")
    return ckpt


def demo_tok_key(rt) -> str:
    """train_v074_policy.py state-cache key for a (task_id, demo,
    row) rehearsal triple — byte-identical string."""
    return "demo::" + json.dumps(list(rt))


def load_v074_anchors():
    """Real V7.4 anchors from the frozen V7.4B anchor manifest: all
    acq (train + dev) + teach rows. SHORTFALL / family_shortfall
    rows are under-filled-budget records, not anchors — skipped,
    never refilled (the v074_schedule.load_universe discipline)."""
    from lcwm.v074_schedule import v074_root
    root = v074_root()
    man = json.loads((root / "anchor_manifest.json").read_text())
    assert man.get("schema") == "v074_anchor_manifest_v1", \
        man.get("schema")
    anchors, n_budget_rows = {}, 0
    for a in man["anchors"]:
        if a["kind"] not in ("acq", "teach"):
            n_budget_rows += 1
            continue
        ak = f"{a['source_id']}_d{int(a['decision'])}"
        assert ak not in anchors, f"duplicate manifest anchor {ak}"
        anchors[ak] = {
            "task": a["task"], "source_id": a["source_id"],
            "decision": int(a["decision"]), "kind": a["kind"],
            "role": a["role"], "blocker_family": a.get("family")}
    counts = defaultdict(int)
    for r in anchors.values():
        counts[(r["kind"], r["role"])] += 1
    # the frozen V7.4B budget: 24 train + 9 dev acq, 9 teach
    assert dict(counts) == {("acq", "train"): 24, ("acq", "dev"): 9,
                            ("teach", "teacher"): 9}, dict(counts)
    return root, anchors, n_budget_rows


def load_demo_rows():
    """Train-split demo-rehearsal episodes + full (tid, demo, row)
    enumeration, exactly as train_v074_policy.py builds them."""
    from lcwm.v067_lineage import load_v067
    dm = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in dm["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] == "train":
            demo_eps[(e["task_id"], e["demo"])] = ep
    demo_rows = [(tid, di, ri) for (tid, di) in sorted(demo_eps)
                 for ri in range(len(demo_eps[(tid, di)]["rows"]))]
    return demo_eps, demo_rows


def main_v074_gate(args) -> None:
    """BINDING pre-D gate (registration + the dated pre-C amendment
    "V7.4A EXECUTED" in plan_and_progress/2026-08-09.md): T1 binding,
    T4/T5 binding where crossed support exists, T2/T3 measured and
    named next to the A-stage pooled values, never gating. A
    gate_pass=false report halts V7.4D (the fallback switch is
    SPENT; enforcement lives with the V7.4D consumers)."""
    from lcwm import v074_schedule as S74
    from lcwm.v067_lineage import sha256_file

    if args.interface != "token_query":
        sys.exit(
            "--universe v074 refuses --interface "
            f"{args.interface!r}: the A-stage pooled run recorded "
            "consequence commit_fallback_token_query "
            "(probe_report_pooled.json) and the registered fallback "
            "switch is SPENT — the binding pre-D gate runs on "
            "token_query only")
    if args.lcwm is None:
        args.lcwm = v074_lcwm_default()
    report_path = args.out / "probe_report_v074_token_query.json"
    if report_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {report_path} "
                 "(pass --force)")
    pooled_path = args.out / "probe_report_pooled.json"
    if not pooled_path.exists():
        sys.exit(f"A-stage pooled report {pooled_path} not found — "
                 "the gate quotes its T2/T3 values and its "
                 "commit_fallback_token_query consequence")
    pooled = json.loads(pooled_path.read_text())
    a_cons = pooled.get("a_stage_decision", {}).get("consequence")
    if a_cons != "commit_fallback_token_query":
        sys.exit(f"A-stage consequence {a_cons!r} != "
                 "'commit_fallback_token_query' — the token_query "
                 "binding gate is only defined under the recorded "
                 "fallback commitment")

    args.out.mkdir(parents=True, exist_ok=True)
    lcwm_sha = sha256_file(args.lcwm)
    v74_root, anchors, n_budget_rows = load_v074_anchors()
    aks = sorted(anchors)
    tvid_lang, index_by_anchor = load_text_index(
        v74_root / "semantic_query_index.jsonl")
    demo_eps, demo_rows = load_demo_rows()
    demo_needed = [demo_tok_key(rt) for rt in demo_rows]

    # ---- enumerate needed token states -----------------------------
    perms = {}
    for ak in aks:
        hist = list(range(anchors[ak]["decision"]))
        gg = torch.Generator().manual_seed(
            sha_seed(f"{RID}|shuffle|{ak}"))
        perms[ak] = ([hist[i] for i in torch.randperm(
            len(hist), generator=gg).tolist()] if hist else [])
    text_goal, unresolved = defaultdict(dict), {}
    for ak in aks:
        verified = index_by_anchor.get(ak, {})
        for gid, tvid in S74.queries_for(
                {"universe": "v074", "task": anchors[ak]["task"]},
                None):
            # crossed ONLY where the V7.4B index verified this
            # (goal, text) AT this anchor — teach anchors have no
            # index rows and stay uncrossed (same per-anchor rule
            # as the v073 path)
            if tvid not in verified or tvid not in tvid_lang:
                unresolved.setdefault(ak, []).append(tvid)
                continue
            assert verified[tvid] == gid, (ak, tvid)
            text_goal[ak][tvid] = gid
    needed = {}
    for ak in aks:
        needed[ak] = {"real": f"{ak}::real", "reset": f"{ak}::reset",
                      "shuffled": f"{ak}::shuffled"}
        for tvid in text_goal.get(ak, {}):
            needed[ak][f"text::{tvid}"] = f"{ak}::text::{tvid}"

    # ---- token cache (resumable; includes demo states) -------------
    cache_path = args.out / "probe_states_v074.pt"
    tokens = {}
    if cache_path.exists():
        cb = torch.load(cache_path, weights_only=False)
        if (cb.get("schema") == STATES_SCHEMA
                and cb.get("lcwm_sha") == lcwm_sha
                and cb.get("universe") == "v074"):
            tokens = cb["tokens"]

    def save_cache():
        tmp = args.out / "probe_states_v074.pt.tmp"
        torch.save({"schema": STATES_SCHEMA, "universe": "v074",
                    "lcwm_sha": lcwm_sha, "tokens": tokens}, tmp)
        tmp.replace(cache_path)

    missing = [k for ks in needed.values() for k in ks.values()
               if k not in tokens]
    missing += [k for k in demo_needed if k not in tokens]
    if missing:
        # GPU only when token states must actually be recomputed
        if not torch.cuda.is_available():
            sys.exit(f"GPU required: {len(missing)} token states "
                     f"need real-prompt prefix features through "
                     f"frozen pi0.5")
        device = torch.device("cuda")
        from lcwm.chassis import Pi05Runner
        from lcwm.sampler import prefix_forward
        from lcwm.seq_prefix_cache import normalize_actions
        ref = torch.load(SEQ_REF, weights_only=False)
        ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]
        wm = V06State().to(device)
        wm.load_state_dict(torch.load(
            args.lcwm, weights_only=False)["model"])
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        runner = Pi05Runner(suite_name="libero_10")
        pfx_cache = {}

        @torch.no_grad()
        def prefix_of(key, obs, lang):
            if key not in pfx_cache:
                if len(pfx_cache) > 120:
                    pfx_cache.clear()
                b = runner._obs_to_policy_batch(obs, lang)
                pfx_cache[key] = prefix_forward(runner.policy, b)
            return pfx_cache[key]

        @torch.no_grad()
        def unroll(rows, seq, instr, tag):
            """Recurrent V7.3 unroll over row indices `seq` (last =
            anchor observation); returns the 4 state tokens. Pairing
            and masking exactly as train_v073_policy.py: the action
            into step j is rows[seq[j-1]]'s first 10 under the
            executed_len mask."""
            z = None
            for j, dd in enumerate(seq):
                pfx = prefix_of(f"{tag}|{dd}", rows[dd]["obs"],
                                instr)
                h, m = pfx.hidden.float(), pfx.pad_masks.bool()
                if z is None:
                    z = wm.initial_state(h, m)
                else:
                    prev = rows[seq[j - 1]]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < int(prev.get("executed_len", 10)))
                    z = wm.step(z, aa, h, m, action_mask=am)
            return z.detach().cpu()

        @torch.no_grad()
        def demo_tokens(ep, ri, tag):
            """Demo tokens at row ri, unrolled at the WM's OWN
            10-action decision stride over ep["obs_10"] — the
            train_v073_policy.py::demo_pool construction with the
            final mean-pool dropped, byte-matching
            train_v074_policy.py::demo_tokens."""
            target_t = 50 * ri
            seq = [o for o in ep["obs_10"] if o["t"] <= target_t]
            z = None
            for j, o in enumerate(seq):
                pfx = prefix_of(f"{tag}|t{o['t']}", o["obs"],
                                ep["language"])
                h, m = pfx.hidden.float(), pfx.pad_masks.bool()
                if z is None:
                    z = wm.initial_state(h, m)
                else:
                    ae = np.asarray(seq[j - 1]["chunk10_env"])
                    aa = normalize_actions(
                        torch.from_numpy(ae).float(),
                        ref_mean, ref_std)[None].to(device)
                    n_a = aa.shape[1]
                    if n_a < 10:
                        aa = torch.cat([aa, torch.zeros(
                            1, 10 - n_a, 7, device=device)], dim=1)
                    am = (torch.arange(10, device=device)[None]
                          < n_a)
                    z = wm.step(z, aa[:, :10], h, m, action_mask=am)
            return z.detach().cpu()

        src_dir = S74.sources_for("v074")
        for ak in aks:
            ks = needed[ak]
            if all(k in tokens for k in ks.values()):
                continue
            rec = anchors[ak]
            src = torch.load(src_dir / f"{rec['source_id']}.pt",
                             weights_only=False)
            rows, instr = src["rows"], src["language_canonical"]
            d = rec["decision"]
            assert int(rows[d]["decision"]) == d, ak
            tag = f"{ak}|canon"
            if ks["real"] not in tokens:
                tokens[ks["real"]] = unroll(
                    rows, list(range(d + 1)), instr, tag)
            if ks["reset"] not in tokens:
                tokens[ks["reset"]] = unroll(rows, [d], instr, tag)
            if ks["shuffled"] not in tokens:
                tokens[ks["shuffled"]] = unroll(
                    rows, perms[ak] + [d], instr, tag)
            for tvid in sorted(text_goal.get(ak, {})):
                key = ks[f"text::{tvid}"]
                if key in tokens:
                    continue
                lang = tvid_lang[tvid]
                if lang == instr:
                    tokens[key] = tokens[ks["real"]]
                else:
                    tokens[key] = unroll(rows, list(range(d + 1)),
                                         lang, f"{ak}|{tvid}")
            pfx_cache.clear()
            save_cache()
            print(f"[state] {ak}: {len(ks)} pools cached",
                  flush=True)
        for (tid, di) in sorted(demo_eps):
            ep = demo_eps[(tid, di)]
            keys_ep = [demo_tok_key((tid, di, ri))
                       for ri in range(len(ep["rows"]))]
            if all(k in tokens for k in keys_ep):
                continue
            for ri, key in enumerate(keys_ep):
                if key not in tokens:
                    tokens[key] = demo_tokens(ep, ri,
                                              f"demo{tid}_{di}")
            pfx_cache.clear()
            save_cache()
            print(f"[state] demo {tid}_{di}: {len(keys_ep)} rows "
                  f"cached", flush=True)

    # ---- attend (trainer-seeded init) + center ---------------------
    tqp_seed = sha_seed(TQP_SEED_NS)
    tqp = TokenQueryPool(384, seed=tqp_seed).eval()
    pools, demo_pools = {}, {}
    with torch.no_grad():
        for ks in needed.values():
            for key in ks.values():
                pools[key] = tqp(tokens[key].float())[0]
        for key in demo_needed:
            demo_pools[key] = tqp(tokens[key].float())[0]
    center = CenteredState.fit(
        [pools[f"{ak}::real"] for ak in aks]
        + [demo_pools[k] for k in demo_needed])
    centering_path = args.out / "centering_v074_token_query.pt"
    center.save(centering_path)
    cvec = {k: center.apply(v).reshape(-1)
            for k, v in pools.items()}

    def dist(ka, kb):
        return float(torch.linalg.vector_norm(cvec[ka] - cvec[kb]))

    # ---- thresholds -------------------------------------------------
    pair_d = [dist(f"{a}::real", f"{b}::real")
              for i, a in enumerate(aks) for b in aks[i + 1:]]
    t1_med = float(np.median(pair_d))
    t1 = {"median": t1_med, "threshold": T1_MIN,
          "n_anchors": len(aks), "n_pairs": len(pair_d),
          "binding": True, "pass": bool(t1_med >= T1_MIN)}

    sup = [ak for ak in aks if anchors[ak]["decision"] >= D_MIN]
    d_reset = {ak: dist(f"{ak}::real", f"{ak}::reset") for ak in aks}
    d_shuf = {ak: dist(f"{ak}::real", f"{ak}::shuffled")
              for ak in aks}

    def hist_thresh(dd, num, den, name):
        if not sup:
            return {"rule": name, "pass": "unsupported",
                    "n_support": 0}
        n_pass = sum(1 for ak in sup if dd[ak] >= EPS)
        return {"rule": name, "eps": EPS, "n_support": len(sup),
                "n_pass": n_pass,
                "pass": bool(den * n_pass >= num * len(sup))}

    t2 = hist_thresh(d_reset, 2, 3,
                     ">=0.05 for >=2/3 of anchors with d>=5")
    t3 = hist_thresh(d_shuf, 1, 3,
                     ">=0.05 for >=1/3 of anchors with d>=5")
    for t_ in (t2, t3):
        t_["binding"] = False
        t_["role"] = "measured_and_named"

    para_all, alt_all, per_anchor_45 = [], [], {}
    for ak in aks:
        tg = text_goal.get(ak, {})
        if not tg:
            continue
        by_goal = defaultdict(list)
        for tvid, gid in tg.items():
            by_goal[gid].append(f"{ak}::text::{tvid}")
        pd_, ad_ = [], []
        goals = sorted(by_goal)
        for gid in goals:
            ks = sorted(by_goal[gid])
            pd_ += [dist(ks[i], ks[j]) for i in range(len(ks))
                    for j in range(i + 1, len(ks))]
        for i, ga in enumerate(goals):
            for gb in goals[i + 1:]:
                ad_ += [dist(ka, kb) for ka in sorted(by_goal[ga])
                        for kb in sorted(by_goal[gb])]
        para_all += pd_
        alt_all += ad_
        per_anchor_45[ak] = {
            "median_paraphrase": (float(np.median(pd_)) if pd_
                                  else None),
            "median_alt_goal": (float(np.median(ad_)) if ad_
                                else None)}
    excluded_45 = [ak for ak in aks if not text_goal.get(ak)]
    if para_all:
        t4_med = float(np.median(para_all))
        t4 = {"median_paraphrase": t4_med,
              "bound": 0.5 * t1_med, "n_dists": len(para_all),
              "n_anchors": len(per_anchor_45),
              "pass": bool(t4_med <= 0.5 * t1_med)}
    else:
        t4_med = None
        t4 = {"pass": "unsupported", "n_dists": 0}
    if alt_all and t4_med is not None:
        t5_med = float(np.median(alt_all))
        t5 = {"median_alt_goal": t5_med,
              "bound": 1.5 * t4_med, "n_dists": len(alt_all),
              "n_anchors": len(per_anchor_45),
              "pass": bool(t5_med >= 1.5 * t4_med)}
    else:
        t5 = {"pass": "unsupported", "n_dists": len(alt_all)}
    for t_ in (t4, t5):
        t_["binding"] = "where_crossed_support_exists"
    thresholds = {"T1_identity": t1, "T2_history": t2,
                  "T3_order": t3, "T4_paraphrase": t4,
                  "T5_goal_separation": t5}

    # ---- AMENDED binding verdict ------------------------------------
    def pass_or_unsupported(t_):
        return t_["pass"] is True or t_["pass"] == "unsupported"

    gate_pass = bool(t1["pass"] is True
                     and pass_or_unsupported(t4)
                     and pass_or_unsupported(t5))
    gate = {
        "rule": "AMENDED binding pre-D gate (2026-08-09.md pre-C "
                "amendment): gate_pass = T1 pass AND (T4 pass or "
                "unsupported) AND (T5 pass or unsupported); T2/T3 "
                "measured and named, never gating; the fallback "
                "switch is SPENT — gate_pass=false halts V7.4D",
        "gate_pass": gate_pass, "T1": t1["pass"],
        "T4": t4["pass"], "T5": t5["pass"]}

    # ---- T2/T3: measured and named vs the A-stage pooled values ----
    moved = t2["pass"] is True and t3["pass"] is True
    measured = {
        "status": "measured_and_named (amended rule: never gating)",
        "v074c_T2_history": t2, "v074c_T3_order": t3,
        "a_stage_pooled_T2_history":
            pooled["thresholds"]["T2_history"],
        "a_stage_pooled_T3_order":
            pooled["thresholds"]["T3_order"],
        "a_stage_report": pooled_path.name,
        "a_stage_report_sha256": sha256_file(pooled_path),
        "named_interpretation": (
            "V7.4C training moved the recurrence: real-vs-reset "
            "and real-vs-shuffled distances clear the registered "
            "epsilon on the required anchor fractions." if moved
            else
            "recurrence functionally inert at inference — standing "
            "limitation carried verbatim into every V7.4 result "
            "interpretation, INCLUDING any Phase-1 win: a win "
            "under inert recurrence is a win for "
            "observation+language state content, not for recurrent "
            "history integration, and will be claimed as exactly "
            "that and nothing more.")}

    # ---- anchor_states_v074.pt: EXACT trainer token schema ---------
    toks_out = {}
    for ak in aks:
        for sfx in ("real", "reset", "shuffled"):
            t_ = tokens[f"{ak}::{sfx}"]
            assert t_.dim() == 3 and t_.shape[-1] == 384, \
                (ak, sfx, tuple(t_.shape))
            toks_out[f"{sfx}::{ak}"] = t_.cpu()
    for key in demo_needed:
        toks_out[key] = tokens[key].cpu()
    anchor_states_path = args.out / "anchor_states_v074.pt"
    torch.save({"schema": STATE_SCHEMA_V74, "lcwm_sha": lcwm_sha,
                "tokens": toks_out}, anchor_states_path)

    notes = [
        "BINDING pre-D gate on the frozen V7.4C final checkpoint "
        "over the complete V7.4 policy-state set.",
        "token_query REQUIRED: the A-stage pooled run committed "
        "commit_fallback_token_query; the registered fallback "
        "switch is SPENT.",
        "TokenQueryPool init sha-seeded from the policy trainer's "
        f"namespace {TQP_SEED_NS!r} — byte-identical to every "
        "V7.4D arm's init (train_v074_policy.py).",
        "demo states RECOMPUTED with the probed checkpoint (the "
        "stored v073 demo pools belong to the retired V7.3C "
        "checkpoint).",
        "teach anchors have no semantic_query_index rows and stay "
        "uncrossed (same per-anchor admission rule as v073).",
    ]

    anchor_rows = []
    for ak in aks:
        rec = anchors[ak]
        tg = text_goal.get(ak, {})
        row = {"anchor": ak, "task": rec["task"],
               "source_id": rec["source_id"],
               "decision": rec["decision"], "kind": rec["kind"],
               "role": rec["role"],
               "blocker_family": rec["blocker_family"],
               "d_reset": d_reset[ak], "d_shuffled": d_shuf[ak],
               "in_history_support": rec["decision"] >= D_MIN,
               "n_text_pools": len(tg),
               "crossed_pairs": {
                   tvid: {"goal": tg[tvid],
                          "equals_real_pool": bool(torch.equal(
                              tokens[f"{ak}::text::{tvid}"],
                              tokens[f"{ak}::real"]))}
                   for tvid in sorted(tg)},
               "unresolved_pairs": unresolved.get(ak, []),
               **per_anchor_45.get(ak, {})}
        if not tg:
            row["crossed_texts"] = "excluded_no_index_rows"
        anchor_rows.append(row)

    report = {
        "schema": REPORT_SCHEMA, "run_id": RID,
        "mode": "binding_gate", "universe": "v074",
        "interface": "token_query", "lcwm_sha256": lcwm_sha,
        "gate_pass": gate_pass, "gate": gate,
        "measured_and_named": measured,
        "a_stage_commitment": {
            "report": pooled_path.name,
            "report_sha256": measured["a_stage_report_sha256"],
            "a_stage_decision": pooled["a_stage_decision"]},
        "notes": notes, "anchors": anchor_rows,
        "thresholds": thresholds,
        "t4_t5_uncrossed_anchors": excluded_45,
        "unresolved_pairs_total": sum(
            len(v) for v in unresolved.values()),
        "tqp_seed": tqp_seed, "tqp_seed_namespace": TQP_SEED_NS,
        "fit": {"n_real": len(aks), "n_demo": len(demo_rows),
                "sigma": center.sigma,
                "mu_norm": float(center.mu.norm()),
                "mu_absmax": float(center.mu.abs().max()),
                "pooling": "token_query_trainer_seeded_init"}}
    report_path.write_text(json.dumps(report, indent=2))

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    manifest_path = args.out / "run_manifest_v074_token_query.json"
    manifest_path.write_text(json.dumps({
        "schema": "v074_probe_manifest_v1", "run_id": RID,
        "git_sha": git_sha, "universe": "v074",
        "mode": "binding_gate", "interface": "token_query",
        "lcwm_path": str(args.lcwm), "lcwm_sha256": lcwm_sha,
        "gate_pass": gate_pass,
        "tqp_seed": tqp_seed, "tqp_seed_namespace": TQP_SEED_NS,
        "anchor_counts": {"acq_train": 24, "acq_dev": 9, "teach": 9,
                          "real_total": len(aks),
                          "budget_rows_skipped": n_budget_rows},
        "demo_rows": len(demo_rows),
        "fit": report["fit"],
        # sha-bind every consumed decision input and every output
        "input_sha256": {
            "anchor_manifest": sha256_file(
                v74_root / "anchor_manifest.json"),
            "semantic_query_index": sha256_file(
                v74_root / "semantic_query_index.jsonl"),
            "recovery_ledger": sha256_file(
                v74_root / "recovery_ledger.jsonl"),
            "outcome_support": sha256_file(
                v74_root / "outcome_support.json"),
            "data_run_manifest": sha256_file(
                v74_root / "run_manifest.json"),
            "source_split": sha256_file(
                v74_root / "source_split.json"),
            "demo_manifest": sha256_file(
                DEMO_DIR / "manifest.json"),
            "a_stage_pooled_report": sha256_file(pooled_path)},
        "output_sha256": {
            "probe_report": sha256_file(report_path),
            "centering": sha256_file(centering_path),
            "probe_states": sha256_file(cache_path),
            "anchor_states": sha256_file(anchor_states_path)},
    }, indent=2))

    # ---- console ----------------------------------------------------
    print(f"\n[probe] universe=v074 interface=token_query "
          f"mode=binding_gate anchors={len(aks)} "
          f"(acq 33, teach 9) demo_rows={len(demo_rows)}",
          flush=True)
    hdr = (f"{'anchor':40s} {'d':>3s} {'kind/role':14s} "
           f"{'reset':>7s} {'shuf':>7s} txt")
    print(hdr + "\n" + "-" * len(hdr))
    for ak in aks:
        rec = anchors[ak]
        print(f"{ak:40s} {rec['decision']:3d} "
              f"{rec['kind'] + '/' + rec['role']:14s} "
              f"{d_reset[ak]:7.4f} {d_shuf[ak]:7.4f} "
              f"{len(text_goal.get(ak, {})):3d}")
    for name, t_ in thresholds.items():
        verdict = ("UNSUPPORTED" if t_["pass"] == "unsupported"
                   else "PASS" if t_["pass"] else "FAIL")
        stat = {k: v for k, v in t_.items() if k != "pass"}
        print(f"[{verdict:11s}] {name}: {stat}")
    print(f"[gate] gate_pass={gate_pass} (T1 binding; T4/T5 "
          f"binding where crossed support exists; T2/T3 "
          f"measured-and-named)")
    print(f"-> {report_path}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="v073",
                    choices=("v073", "v074"))
    ap.add_argument("--lcwm", type=Path, default=None,
                    help="probed LCWM checkpoint (default: v073 -> "
                         "the frozen V7.3C final; v074 -> newest "
                         "*_v074_lcwm_r1/checkpoints/final.pt)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--interface", default="pooled",
                    choices=("pooled", "token_query"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.universe == "v074":
        # BINDING pre-D gate (amended rule); the v073 A-stage path
        # below stays byte-identical in behavior
        main_v074_gate(args)
        return
    if args.lcwm is None:
        args.lcwm = LCWM_FINAL
    # interface-specific outputs; ONLY the token cache is shared
    # between the pooled and token_query runs (review: identical
    # filenames made the registered two-run procedure collide)
    report_path = args.out / f"probe_report_{args.interface}.json"
    if report_path.exists() and not args.force:
        sys.exit(f"refusing to overwrite {report_path} "
                 "(pass --force)")
    from lcwm.v067_lineage import sha256_file

    args.out.mkdir(parents=True, exist_ok=True)
    lcwm_sha = sha256_file(args.lcwm)

    anchors, g_origin, model_rows = load_ledger_anchors()
    aks = sorted(anchors)
    tvid_lang, index_by_anchor = load_text_index()

    # STORED demo pools (registered: never recomputed by the probe)
    blob = torch.load(POLICY / "anchor_states.pt",
                      weights_only=False)
    assert blob.get("schema") == STATE_SCHEMA_V73, blob.get("schema")
    demo_pools = blob["demo_pools"]
    demo_rows = sorted(tuple(json.loads(k)) for k in demo_pools)

    # ---- v074 schedule dry run (cheap, before any GPU work) --------
    sched = build_matched_policy_schedule(
        {ak: {"task": anchors[ak]["task"],
              "source_id": anchors[ak]["source_id"]}
         for ak in aks if "grounded" in anchors[ak]["roles"]},
        {r["anchor"]: {"task": r["task"],
                       "source_id": r["source_id"],
                       "mode": r["mode"]} for r in model_rows},
        demo_rows, SCHED_STEPS, RID)
    sched_check = {
        "schema": sched["schema"], "steps": SCHED_STEPS,
        "grounded_hist": sched["histograms"]["grounded"],
        "model_hist": sched["histograms"]["model"],
        "n_active_model_slots": sum(
            1 for _a, act in sched["model"] if act),
        "note": "diagnostic-only dry run of "
                "lcwm.v074_policy_schedule on the V7.3 ledgers; "
                "exposure caps asserted inside the builder"}

    # ---- enumerate needed token states -----------------------------
    perms = {}
    for ak in aks:
        hist = list(range(anchors[ak]["decision"]))
        gg = torch.Generator().manual_seed(
            sha_seed(f"{RID}|shuffle|{ak}"))
        perms[ak] = ([hist[i] for i in torch.randperm(
            len(hist), generator=gg).tolist()] if hist else [])
    text_goal, unresolved = defaultdict(dict), {}
    for ak in aks:
        if anchors[ak]["family"] != "v073":
            continue
        verified = index_by_anchor.get(ak, {})
        for gid, tvid in S.queries_for(
                {"universe": "v073", "task": anchors[ak]["task"]},
                None):
            # crossed ONLY where the V7.3B index verified this
            # (goal, text) AT this anchor — teacher-source anchors
            # have no index rows and stay uncrossed
            if tvid not in verified or tvid not in tvid_lang:
                unresolved.setdefault(ak, []).append(tvid)
                continue
            assert verified[tvid] == gid, (ak, tvid)
            text_goal[ak][tvid] = gid
    needed = {}
    for ak in aks:
        needed[ak] = {"real": f"{ak}::real", "reset": f"{ak}::reset",
                      "shuffled": f"{ak}::shuffled"}
        for tvid in text_goal.get(ak, {}):
            needed[ak][f"text::{tvid}"] = f"{ak}::text::{tvid}"

    # ---- token cache (interface-independent; resumable) ------------
    cache_path = args.out / "probe_states.pt"
    tokens = {}
    if cache_path.exists():
        cb = torch.load(cache_path, weights_only=False)
        if (cb.get("schema") == STATES_SCHEMA
                and cb.get("lcwm_sha") == lcwm_sha
                and cb.get("universe") == args.universe):
            tokens = cb["tokens"]

    def save_cache():
        tmp = args.out / "probe_states.pt.tmp"
        torch.save({"schema": STATES_SCHEMA,
                    "universe": args.universe,
                    "lcwm_sha": lcwm_sha, "tokens": tokens}, tmp)
        tmp.replace(cache_path)

    missing = [k for ks in needed.values() for k in ks.values()
               if k not in tokens]
    if missing:
        # GPU only when token states must actually be recomputed; a
        # complete cache (e.g. the token_query re-run) is CPU-only
        if not torch.cuda.is_available():
            sys.exit(f"GPU required: {len(missing)} token states "
                     f"need real-prompt prefix features through "
                     f"frozen pi0.5")
        device = torch.device("cuda")
        from lcwm.chassis import Pi05Runner
        from lcwm.sampler import prefix_forward
        wm = V06State().to(device)
        wm.load_state_dict(torch.load(
            args.lcwm, weights_only=False)["model"])
        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        runner = Pi05Runner(suite_name="libero_10")
        pfx_cache = {}

        @torch.no_grad()
        def prefix_of(key, obs, lang):
            if key not in pfx_cache:
                if len(pfx_cache) > 120:
                    pfx_cache.clear()
                b = runner._obs_to_policy_batch(obs, lang)
                pfx_cache[key] = prefix_forward(runner.policy, b)
            return pfx_cache[key]

        @torch.no_grad()
        def unroll(rows, seq, instr, tag):
            """Recurrent V7.3 unroll over row indices `seq` (last =
            anchor observation); returns the 4 state tokens. Pairing
            and masking exactly as train_v073_policy.py: the action
            into step j is rows[seq[j-1]]'s first 10 under the
            executed_len mask."""
            z = None
            for j, dd in enumerate(seq):
                pfx = prefix_of(f"{tag}|{dd}", rows[dd]["obs"],
                                instr)
                h, m = pfx.hidden.float(), pfx.pad_masks.bool()
                if z is None:
                    z = wm.initial_state(h, m)
                else:
                    prev = rows[seq[j - 1]]
                    aa = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < int(prev.get("executed_len", 10)))
                    z = wm.step(z, aa, h, m, action_mask=am)
            return z.detach().cpu()

        for ak in aks:
            ks = needed[ak]
            if all(k in tokens for k in ks.values()):
                continue
            rec = anchors[ak]
            src_dir = (V73 / "sources" if rec["family"] == "v073"
                       else V72T / "teacher_sources")
            src = torch.load(src_dir / f"{rec['source_id']}.pt",
                             weights_only=False)
            rows, instr = src["rows"], src["language_canonical"]
            d = rec["decision"]
            assert int(rows[d]["decision"]) == d, ak
            tag = f"{ak}|canon"
            if ks["real"] not in tokens:
                tokens[ks["real"]] = unroll(
                    rows, list(range(d + 1)), instr, tag)
            if ks["reset"] not in tokens:
                tokens[ks["reset"]] = unroll(rows, [d], instr, tag)
            if ks["shuffled"] not in tokens:
                tokens[ks["shuffled"]] = unroll(
                    rows, perms[ak] + [d], instr, tag)
            for tvid in sorted(text_goal.get(ak, {})):
                key = ks[f"text::{tvid}"]
                if key in tokens:
                    continue
                lang = tvid_lang[tvid]
                if lang == instr:
                    tokens[key] = tokens[ks["real"]]
                else:
                    tokens[key] = unroll(rows, list(range(d + 1)),
                                         lang, f"{ak}|{tvid}")
            pfx_cache.clear()
            save_cache()
            print(f"[state] {ak}: {len(ks)} pools cached",
                  flush=True)

    # ---- pool + center ---------------------------------------------
    if args.interface == "token_query":
        torch.manual_seed(0)   # registered fresh-init seed
        tqp = TokenQueryPool(384).eval()
    else:
        tqp = None
    pools = {}
    with torch.no_grad():
        for ks in needed.values():
            for key in ks.values():
                z = tokens[key].float()
                pools[key] = (tqp(z)[0] if tqp is not None
                              else z.mean(dim=1)[0])
    center = CenteredState.fit(
        [pools[f"{ak}::real"] for ak in aks]
        + [demo_pools[k] for k in sorted(demo_pools)])
    centering_path = args.out / f"centering_{args.interface}.pt"
    center.save(centering_path)
    cvec = {k: center.apply(v).reshape(-1)
            for k, v in pools.items()}

    def dist(ka, kb):
        return float(torch.linalg.vector_norm(cvec[ka] - cvec[kb]))

    # ---- thresholds -------------------------------------------------
    pair_d = [dist(f"{a}::real", f"{b}::real")
              for i, a in enumerate(aks) for b in aks[i + 1:]]
    t1_med = float(np.median(pair_d))
    t1 = {"median": t1_med, "threshold": T1_MIN,
          "n_anchors": len(aks), "n_pairs": len(pair_d),
          "pass": bool(t1_med >= T1_MIN)}

    sup = [ak for ak in aks if anchors[ak]["decision"] >= D_MIN]
    d_reset = {ak: dist(f"{ak}::real", f"{ak}::reset") for ak in aks}
    d_shuf = {ak: dist(f"{ak}::real", f"{ak}::shuffled")
              for ak in aks}

    def hist_thresh(dd, num, den, name):
        if not sup:
            return {"rule": name, "pass": "unsupported",
                    "n_support": 0}
        n_pass = sum(1 for ak in sup if dd[ak] >= EPS)
        return {"rule": name, "eps": EPS, "n_support": len(sup),
                "n_pass": n_pass,
                "pass": bool(den * n_pass >= num * len(sup))}

    t2 = hist_thresh(d_reset, 2, 3,
                     ">=0.05 for >=2/3 of anchors with d>=5")
    t3 = hist_thresh(d_shuf, 1, 3,
                     ">=0.05 for >=1/3 of anchors with d>=5")

    para_all, alt_all, per_anchor_45 = [], [], {}
    for ak in aks:
        tg = text_goal.get(ak, {})
        if not tg:
            continue
        by_goal = defaultdict(list)
        for tvid, gid in tg.items():
            by_goal[gid].append(f"{ak}::text::{tvid}")
        pd_, ad_ = [], []
        goals = sorted(by_goal)
        for gid in goals:
            ks = sorted(by_goal[gid])
            pd_ += [dist(ks[i], ks[j]) for i in range(len(ks))
                    for j in range(i + 1, len(ks))]
        for i, ga in enumerate(goals):
            for gb in goals[i + 1:]:
                ad_ += [dist(ka, kb) for ka in sorted(by_goal[ga])
                        for kb in sorted(by_goal[gb])]
        para_all += pd_
        alt_all += ad_
        per_anchor_45[ak] = {
            "median_paraphrase": (float(np.median(pd_)) if pd_
                                  else None),
            "median_alt_goal": (float(np.median(ad_)) if ad_
                                else None)}
    excluded_45 = [ak for ak in aks
                   if anchors[ak]["family"] == "v072T"]
    if para_all:
        t4_med = float(np.median(para_all))
        t4 = {"median_paraphrase": t4_med,
              "bound": 0.5 * t1_med, "n_dists": len(para_all),
              "n_anchors": len(per_anchor_45),
              "pass": bool(t4_med <= 0.5 * t1_med)}
    else:
        t4_med = None
        t4 = {"pass": "unsupported", "n_dists": 0}
    if alt_all and t4_med is not None:
        t5_med = float(np.median(alt_all))
        t5 = {"median_alt_goal": t5_med,
              "bound": 1.5 * t4_med, "n_dists": len(alt_all),
              "n_anchors": len(per_anchor_45),
              "pass": bool(t5_med >= 1.5 * t4_med)}
    else:
        t5 = {"pass": "unsupported", "n_dists": len(alt_all)}
    thresholds = {"T1_identity": t1, "T2_history": t2,
                  "T3_order": t3, "T4_paraphrase": t4,
                  "T5_goal_separation": t5}

    # ---- registered A-stage consequence (recorded, report-only) ----
    if args.interface == "pooled":
        if "unsupported" in (t1["pass"], t2["pass"]):
            # empty support prints unsupported, never a verdict —
            # the fallback commits only on a MEASURED failure
            a_stage = {
                "rule": "pooled failing T1 or T2 at A-stage "
                        "commits V7.4 to the fallback interface "
                        "(registered frozen decision rule, step 1)",
                "pooled_t1_t2_ok": "unsupported",
                "consequence": "unsupported_no_decision"}
        else:
            ok = t1["pass"] is True and t2["pass"] is True
            a_stage = {
                "rule": "pooled failing T1 or T2 at A-stage "
                        "commits V7.4 to the fallback interface "
                        "(registered frozen decision rule, step 1)",
                "pooled_t1_t2_ok": ok,
                "consequence": ("keep_pooled_candidate" if ok else
                                "commit_fallback_token_query")}
    else:
        a_stage = {"note": "token_query diagnostic run; the frozen "
                           "decision rule reads the POOLED run"}

    # ---- reimplementation check vs the stored V7.3 trainer pools ---
    consistency = {}
    pm = json.loads((POLICY / "run_manifest.json").read_text())
    if args.interface == "pooled" \
            and pm["lcwm_final_sha"] == lcwm_sha:
        for ak in aks:
            skeys = []
            if "grounded" in anchors[ak]["roles"]:
                skeys.append(f"{g_origin[ak]}::{ak}")
            if set(anchors[ak]["roles"]) & {"model_teacher"}:
                skeys.append(f"model::{ak}")
            for sk in skeys:
                if sk in blob["pools"]:
                    consistency[sk] = float(
                        (pools[f"{ak}::real"].cpu()
                         - blob["pools"][sk].reshape(-1))
                        .abs().max())

    notes = [
        "A-stage diagnostic: report-only instrument validation on "
        "the frozen V7.3C LCWM and V7.3 anchor set.",
        "v072T-family anchors are canonical-only: no crossed texts "
        "exist for them; excluded from T4/T5 support.",
        "T4/T5 crossing is admitted per anchor only where the V7.3B "
        "semantic_query_index verified that (goal, text) AT that "
        "anchor; teacher-source anchors have no index rows and are "
        "uncrossed by construction.",
        "demo/retention schedule histograms omitted from "
        "sched_check (uniform interleave); counts live in the "
        "schedule artifact.",
    ]
    if args.interface == "token_query":
        notes += [
            "token_query pools with a FRESH TokenQueryPool "
            "(torch.manual_seed(0)): an UNTRAINED query is a lower "
            "bound on the registered fallback, not the trained "
            "fallback.",
            "demo pools remain the STORED mean-pooled artifacts "
            "(registration: never recomputed), so the centering fit "
            "mixes mean-pooled demo states with token-query real "
            "states in this mode."]

    anchor_rows = []
    for ak in aks:
        rec = anchors[ak]
        tg = text_goal.get(ak, {})
        row = {"anchor": ak, "task": rec["task"],
               "source_id": rec["source_id"],
               "decision": rec["decision"],
               "family": rec["family"], "roles": rec["roles"],
               "d_reset": d_reset[ak], "d_shuffled": d_shuf[ak],
               "in_history_support": rec["decision"] >= D_MIN,
               "n_text_pools": len(tg),
               "crossed_pairs": {
                   tvid: {"goal": tg[tvid],
                          "equals_real_pool": bool(torch.equal(
                              tokens[f"{ak}::text::{tvid}"],
                              tokens[f"{ak}::real"]))}
                   for tvid in sorted(tg)},
               "unresolved_pairs": unresolved.get(ak, []),
               **per_anchor_45.get(ak, {})}
        if rec["family"] == "v072T":
            row["crossed_texts"] = "excluded_canonical_only"
        anchor_rows.append(row)

    report = {
        "schema": REPORT_SCHEMA, "run_id": RID,
        "mode": "A_stage_diagnostic", "universe": args.universe,
        "interface": args.interface, "lcwm_sha256": lcwm_sha,
        "notes": notes, "anchors": anchor_rows,
        "thresholds": thresholds, "a_stage_decision": a_stage,
        "t4_t5_excluded_anchors": excluded_45,
        "unresolved_pairs_total": sum(
            len(v) for v in unresolved.values()),
        "trainer_pool_consistency_max_abs": consistency,
        "sched_check": sched_check,
        "fit": {"n_real": len(aks), "n_demo": len(demo_pools),
                "sigma": center.sigma,
                "mu_norm": float(center.mu.norm()),
                "mu_absmax": float(center.mu.abs().max()),
                "mixed_pooling":
                    bool(args.interface == "token_query")}}
    report_path.write_text(json.dumps(report, indent=2))

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    manifest_path = args.out / f"run_manifest_{args.interface}.json"
    manifest_path.write_text(json.dumps({
        "schema": "v074_probe_manifest_v1", "run_id": RID,
        "git_sha": git_sha, "universe": args.universe,
        "interface": args.interface, "lcwm_path": str(args.lcwm),
        "lcwm_sha256": lcwm_sha,
        "anchor_counts": {"grounded": 6, "model": 9,
                          "real_total": len(aks)},
        "fit": report["fit"],
        # sha-bind every consumed decision input (review finding)
        "input_sha256": {
            "grounded_ledger": sha256_file(
                TEACH / "grounded_ledger.jsonl"),
            "model_ledger": sha256_file(
                TEACH / "model_ledger_pre_outcome.jsonl"),
            "semantic_query_index": sha256_file(
                V73 / "semantic_query_index.jsonl"),
            "anchor_states": sha256_file(
                POLICY / "anchor_states.pt")},
        "centering_sha": sha256_file(centering_path),
        "states_cache_sha": sha256_file(cache_path),
        "report_sha": sha256_file(report_path)}, indent=2))

    # ---- console ----------------------------------------------------
    print(f"\n[probe] universe={args.universe} "
          f"interface={args.interface} anchors={len(aks)} "
          f"(grounded 6, model 9)", flush=True)
    hdr = (f"{'anchor':46s} {'d':>3s} {'roles':14s} "
           f"{'reset':>7s} {'shuf':>7s} txt")
    print(hdr + "\n" + "-" * len(hdr))
    for ak in aks:
        rec = anchors[ak]
        print(f"{ak:46s} {rec['decision']:3d} "
              f"{'/'.join(rec['roles']):14s} "
              f"{d_reset[ak]:7.4f} {d_shuf[ak]:7.4f} "
              f"{len(text_goal.get(ak, {})):3d}")
    for name, t in thresholds.items():
        verdict = ("UNSUPPORTED" if t["pass"] == "unsupported"
                   else "PASS" if t["pass"] else "FAIL")
        stat = {k: v for k, v in t.items() if k != "pass"}
        print(f"[{verdict:11s}] {name}: {stat}")
    if args.interface == "pooled":
        print(f"[a-stage] {a_stage['consequence']}")
    print(f"-> {report_path}", flush=True)


if __name__ == "__main__":
    main()
