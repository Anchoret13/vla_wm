#!/usr/bin/env python
"""V7.4D — fine-tune pi0.5 through grounded + model-generated targets
under the registered V7.4A interface (plan_and_progress/2026-08-09.md).

Derived from scripts/train_v073_policy.py. Registered deltas ONLY:

  interface   --interface {pooled,token_query} is REQUIRED and is
              ENFORCED against the A-stage consequence read from
              probe_report_pooled.json in the binding-gate root
              (commit_fallback_token_query -> token_query,
              keep_pooled_candidate -> pooled,
              unsupported_no_decision -> refuse); a mismatch
              refuses. Training additionally REQUIRES the pre-D
              BINDING GATE: probe_report_v074_<interface>.json with
              mode "binding_gate", gate_pass true, matching
              interface, and lcwm_sha256 equal to the deployed
              V7.4C final.pt. Both report shas land in the
              manifest. pooled: c = CenteredState.apply(Pool(z)).
              token_query: ONE TokenQueryPool per arm, sha-seeded
              from the stable literal namespace "<RID>|tqp"
              (identical across arms AND identical to the seed the
              gate mirrors), instantiated at stock init, trained
              jointly with LCProj + action_out_proj, and run
              IN-GRAPH every step over the cached PRE-POOL token
              states [1, 4, 384] (the lcwm/v074_interface caching
              constraint).
  gate        anchor_states_v074.pt (schema v074_policy_states_v1)
  artifacts   and centering_v074_<interface>.pt are CONSUMED from
              the binding-gate root, sha-verified against the
              gate's run_manifest_v074_<interface>.json, and
              lcwm-sha-chained to the deployed checkpoint. This
              trainer NEVER re-unrolls token states and NEVER
              refits mu/sigma; only trainer-side composites
              (attended pools, centered residuals, W_c biases) are
              derived in-graph from the cached tokens. Missing
              artifacts, a broken sha chain, or an anchor/demo
              census that does not cover the schedule refuse
              loudly with the offending keys NAMED. On resume the
              stored manifest shas are re-verified against the
              current artifacts before any step.
  c = 0       the constant-state arm correction_bc uses c = 0
              exactly (the centered analogue of the V7.3
              global-mean arm — NOT the mean through W_c).
  schedule    lcwm.v074_policy_schedule.build_matched_policy_schedule
              over the grounded anchors + the FULL teacher-anchor
              set (eligible AND abstain); at abstention slots the
              model/random term is computed WITH WEIGHT EXACTLY ZERO
              (present, not skipped: a uniform weighted_anchor_fm
              call scaled by 0.0, identical graph for lc_full and
              lc_random). Blob torch.saved; on resume it is
              re-verified and len(grounded) == 300 is asserted.
  readouts    the A/B/C + reset/shuffled attribution readouts
              aggregate over the FULL scheduled anchor set (median
              in the logs + per-anchor rows in every checkpoint,
              with separate support counts for the full set vs the
              grounded-target subset); reset/shuffled variants come
              per anchor from the gate blob (V7.3D order-only
              construction, built by the gate — never here).
  step-50     over all scheduled anchors,
  gate        Var(W_c c) / (||mean(W_c c)||^2 + Var(W_c c)) >= 0.5;
              failure HALTS the run with a named status written into
              the manifest (the registered fallback path is
              orchestrated outside; this trainer only refuses to
              continue). Enforced on the recurrent-state arms;
              recorded for all. Per-checkpoint W_c c statistics are
              saved for the content audit.

Everything else inherited unchanged from V7.3D: five arms (stock is
the untrained deployment baseline), trainable = bias-free zero-init
LCProj + stock-initialized action_out_proj (+ the arm's
TokenQueryPool under token_query), L = grounded_first10 +
model_first10 + suffix_trust + retention_trust + demo_FM through
lcwm/v073_policy_loss.py, 300 AdamW steps lr 1e-4 wd 1e-4 clip 1.0,
checkpoints + readouts every 50, final-step rule, videos for every
scheduled training-state environment evaluation (v074 sources).
Output: results/libero_loho_public_v1/<date>_v074_policy_r1/
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v073_policy_loss import (suffix_trust,  # noqa: E402
                                   weighted_anchor_fm)
from lcwm.v074_interface import (CenteredState,  # noqa: E402
                                 TokenQueryPool)
from lcwm.v074_policy_schedule import (  # noqa: E402
    build_matched_policy_schedule, verify_matched_policy_schedule)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
# released-tranche fallthrough roots (provenance-labeled evidence)
V73TEACH = RESULTS / "2026-08-07_v073_teacher_r1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RID = "v074_policy_r1"
STEPS, LR, WD, CLIP, CKPT_EVERY = 300, 1e-4, 1e-4, 1.0, 50
# registered V7.4 CRN bases (2026-08-09.md, frozen)
T_BASE, R_BASE, D_BASE = 974_000, 974_500_000, 974_900_000
D_Z, GATE_STEP, GATE_MIN = 384, 50, 0.5
STATE_SCHEMA = "v074_policy_states_v1"
PROBE_REPORT_SCHEMA = "v074_probe_report_v1"
GATE_MODE = "binding_gate"
# registered frozen decision rule: A-stage consequence -> the ONE
# interface this trainer may deploy (anything else refuses)
CONSEQ_TO_INTERFACE = {
    "commit_fallback_token_query": "token_query",
    "keep_pooled_candidate": "pooled",
}
ARMS = {
    "correction_bc": {"state": "zero", "model_mass": False},
    "lc_grounded": {"state": "recurrent", "model_mass": False},
    "lc_full": {"state": "recurrent", "model_mass": "model"},
    "lc_random": {"state": "recurrent", "model_mass": "random"},
}


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
                          ).stdout.strip()


def latest_root(pattern: str) -> Path:
    hits = sorted(RESULTS.glob(pattern))
    if not hits:
        sys.exit(f"missing required run root {RESULTS}/{pattern}")
    return hits[-1]


def demo_tok_key(rt) -> str:
    """State-cache key for a (task_id, demo, row) rehearsal triple."""
    return "demo::" + json.dumps(list(rt))


class LCProj(nn.Module):
    """Bias-free zero-init state -> AdaRMS projection (v0.7)."""

    def __init__(self, d_z: int = D_Z, width: int = 1024):
        super().__init__()
        self.lin = nn.Linear(d_z, width, bias=False)
        nn.init.zeros_(self.lin.weight)

    def forward(self, pool):
        return self.lin(pool)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--interface", required=True,
                    choices=("pooled", "token_query"),
                    help="deployed interface; ENFORCED to equal "
                         "the A-stage committed decision and the "
                         "binding-gate interface")
    ap.add_argument("--probe-report", type=Path, default=None,
                    help="binding-gate report probe_report_v074_"
                         "<interface>.json (default: auto-discover "
                         "in the latest *_v074_interface_diag_r1 "
                         "root); its parent must also hold the "
                         "A-stage probe_report_pooled.json and the "
                         "gate artifacts")
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import freeze_pi05_base
    from lcwm.sampler import prefix_forward
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.v067_lineage import load_v067, sha256_file

    LCWM = latest_root("*_v074_lcwm_r1")
    TEACH = latest_root("*_v074_teacher_r1")
    V74 = latest_root("*_v074_data_r1")
    lcwm_ckpt = LCWM / "checkpoints" / "final.pt"
    lcwm_sha = sha256_file(lcwm_ckpt)

    # ---------- A-stage consequence + binding pre-D gate ------------
    # tqp seed: stable literal namespace (this exact value is what
    # the gate mirrors for its init-attended centering fit)
    tqp_seed = sha_seed(f"{RID}|tqp")
    gate_report_path = args.probe_report or (
        latest_root("*_v074_interface_diag_r1")
        / f"probe_report_v074_{args.interface}.json")
    if not gate_report_path.exists():
        sys.exit(f"binding-gate report {gate_report_path} not "
                 f"found (V7.4D may not start before the pre-D "
                 f"gate)")
    gate_root = gate_report_path.parent
    a_stage_path = gate_root / "probe_report_pooled.json"
    if not a_stage_path.exists():
        sys.exit(f"A-stage report {a_stage_path} not found")
    a_report = json.loads(a_stage_path.read_text())
    if a_report.get("schema") != PROBE_REPORT_SCHEMA:
        sys.exit(f"A-stage report schema "
                 f"{a_report.get('schema')!r} != "
                 f"{PROBE_REPORT_SCHEMA!r}")
    conseq = (a_report.get("a_stage_decision")
              or {}).get("consequence")
    if conseq == "unsupported_no_decision":
        sys.exit("A-stage consequence unsupported_no_decision: no "
                 "committed interface exists; refusing to train")
    required_if = CONSEQ_TO_INTERFACE.get(conseq)
    if required_if is None:
        sys.exit(f"unknown A-stage consequence {conseq!r} in "
                 f"{a_stage_path} (known: "
                 f"{sorted(CONSEQ_TO_INTERFACE)} + "
                 f"unsupported_no_decision)")
    if required_if != args.interface:
        sys.exit(f"A-stage consequence {conseq!r} commits "
                 f"interface {required_if!r}; refusing "
                 f"--interface {args.interface!r}")
    gate_report = json.loads(gate_report_path.read_text())
    if gate_report.get("mode") != GATE_MODE:
        sys.exit(f"{gate_report_path} mode "
                 f"{gate_report.get('mode')!r} != {GATE_MODE!r} "
                 f"(an A-stage diagnostic cannot authorize V7.4D)")
    if gate_report.get("interface") != args.interface:
        sys.exit(f"binding-gate interface "
                 f"{gate_report.get('interface')!r} != "
                 f"--interface {args.interface!r}")
    if gate_report.get("gate_pass") is not True:
        sys.exit(f"binding gate gate_pass="
                 f"{gate_report.get('gate_pass')!r}: V7.4D is "
                 f"halted by the registered pre-D gate")
    if gate_report.get("lcwm_sha256") != lcwm_sha:
        sys.exit(f"binding gate ran on lcwm "
                 f"{gate_report.get('lcwm_sha256')!r} but "
                 f"{lcwm_ckpt} has sha {lcwm_sha}")
    gate_man_path = (gate_root
                     / f"run_manifest_v074_{args.interface}.json")
    states_path = gate_root / "anchor_states_v074.pt"
    center_path = gate_root / f"centering_v074_{args.interface}.pt"
    absent = [str(p) for p in (gate_man_path, states_path,
                               center_path) if not p.exists()]
    if absent:
        sys.exit("binding-gate artifacts missing: "
                 + ", ".join(absent))
    gate_man = json.loads(gate_man_path.read_text())
    if gate_man.get("lcwm_sha256") != lcwm_sha:
        sys.exit(f"gate manifest lcwm_sha256 "
                 f"{gate_man.get('lcwm_sha256')!r} != deployed "
                 f"{lcwm_sha}")

    def _str_values(obj, acc):
        if isinstance(obj, dict):
            for v in obj.values():
                _str_values(v, acc)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _str_values(v, acc)
        elif isinstance(obj, str):
            acc.add(obj)

    def _key_values(obj, key, acc):
        if isinstance(obj, dict):
            for kk, v in obj.items():
                if kk == key:
                    acc.append(v)
                _key_values(v, key, acc)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _key_values(v, key, acc)

    gate_bound = set()
    _str_values(gate_man, gate_bound)
    gate_report_sha = sha256_file(gate_report_path)
    states_sha = sha256_file(states_path)
    center_sha = sha256_file(center_path)
    for nm, sha in ((gate_report_path.name, gate_report_sha),
                    (states_path.name, states_sha),
                    (center_path.name, center_sha)):
        if sha not in gate_bound:
            sys.exit(f"{nm} sha256 {sha} is not bound by "
                     f"{gate_man_path}")
    gate_tqp = []
    _key_values(gate_report, "tqp_seed", gate_tqp)
    _key_values(gate_man, "tqp_seed", gate_tqp)
    if any(v != tqp_seed for v in gate_tqp):
        sys.exit(f"gate TokenQueryPool seed(s) {gate_tqp} != "
                 f"trainer seed {tqp_seed} "
                 f"(namespace {RID}|tqp)")

    # gate-produced token states + centering: loaded, sha-verified
    # above; NEVER rebuilt here (a stale blob refuses, not rebuilds)
    tok_blob = torch.load(states_path, weights_only=False)
    if (tok_blob.get("schema") != STATE_SCHEMA
            or tok_blob.get("lcwm_sha") != lcwm_sha):
        sys.exit(f"{states_path}: schema/lcwm mismatch "
                 f"({tok_blob.get('schema')!r}, "
                 f"{tok_blob.get('lcwm_sha')!r}); this trainer "
                 f"never re-unrolls — rerun the binding gate")
    toks = tok_blob["tokens"]
    center = CenteredState.load(center_path)
    out = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        out = p
    OUT = out or RESULTS / f"{run_date()}_{RID}"

    device = torch.device("cuda")
    torch.manual_seed(0)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock_aop = {k: v.detach().clone()
                 for k, v in aop.state_dict().items()}
    stock_head = copy.deepcopy(aop).to(device)
    stock_head.load_state_dict(stock_aop)
    for p_ in stock_head.parameters():
        p_.requires_grad_(False)
    stock_head.eval()

    # No V06State instance here: token states arrive verified from
    # the gate blob and this trainer never runs the WM recurrence.

    def norm_once(x, is_norm):
        if is_norm:
            cn = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        else:
            cn = normalize_actions(
                torch.from_numpy(np.asarray(x)).float(),
                ref_mean, ref_std)
        if cn.shape[0] < cfg.chunk_size:
            cn = torch.cat([cn, torch.zeros(
                cfg.chunk_size - cn.shape[0], cn.shape[1])], 0)
        return cn[:cfg.chunk_size].to(device)

    # ---------- grounded targets ------------------------------------
    grounded = [json.loads(x) for x in
                (TEACH / "grounded_ledger.jsonl").open()]
    # rel_chunks: released-tranche chunk stores, lazy.
    # anchors: akey (origin::ak) -> {task, source_id, decision,
    #   bare, instr, rows, targets}.
    # meta: bare ak -> {task, source_id, decision, instr, rows}
    #   (shared between the grounded and teacher anchor sets).
    rel_chunks, anchors, meta = {}, {}, {}
    for g in grounded:
        ak = g["anchor"]
        sid = ak.rsplit("_d", 1)[0]
        d = int(ak.rsplit("_d", 1)[1])
        if g["origin"] == "v074B":
            sh = torch.load(V74 / "shards" / f"{ak}.pt",
                            weights_only=False)
            tr = next(t for t in sh["transitions"]
                      if t["branch_key"] == g["branch_key"])
            ch = norm_once(
                tr["chunk_norm"] if tr["chunk_norm"] is not None
                else tr["actions_env"],
                tr["chunk_norm"] is not None)
            src_dir = V74 / "sources"
        elif g["origin"] in ("v073T_released", "v072T_released"):
            store, src_dir = {
                "v073T_released": (V73TEACH / "candidate_chunks.pt",
                                   V73 / "sources"),
                "v072T_released": (V72T / "candidate_chunks.pt",
                                   V72T / "teacher_sources"),
            }[g["origin"]]
            if g["origin"] not in rel_chunks:
                rel_chunks[g["origin"]] = torch.load(
                    store, weights_only=False)
            ch = norm_once(
                rel_chunks[g["origin"]][f"{ak}_{g['branch_key']}"],
                True)
        else:
            sys.exit(f"unknown grounded origin {g['origin']!r} "
                     f"(known: v074B, v073T_released, "
                     f"v072T_released)")
        akey = f"{g['origin']}::{ak}"
        if akey not in anchors:
            src = torch.load(src_dir / f"{sid}.pt",
                             weights_only=False)
            anchors[akey] = {
                "task": g["task"], "source_id": sid, "decision": d,
                "bare": ak, "instr": src["language_canonical"],
                "rows": src["rows"], "targets": {}}
            meta.setdefault(ak, {
                "task": g["task"], "source_id": sid, "decision": d,
                "instr": src["language_canonical"],
                "rows": src["rows"]})
        anchors[akey]["targets"][g["branch_key"]] = ch
    # ---------- FULL teacher set (eligible AND abstention) ----------
    model_rows = [json.loads(x) for x in
                  (TEACH / "model_ledger_pre_outcome.jsonl").open()]
    rand_led = {r["anchor"]: r for r in
                [json.loads(x) for x in
                 (TEACH / "matched_random_ledger.jsonl").open()]}
    teach_chunks = torch.load(TEACH / "candidate_chunks.pt",
                              weights_only=False)
    model_anchors = {}
    for r in model_rows:
        ak, sid, d = r["anchor"], r["source_id"], r["decision"]
        if ak not in meta:
            src = torch.load(V74 / "sources" / f"{sid}.pt",
                             weights_only=False)
            meta[ak] = {"task": r["task"], "source_id": sid,
                        "decision": d,
                        "instr": src["language_canonical"],
                        "rows": src["rows"]}
        model_w = {c: w for c, w in r["weights"].items() if w > 0}
        random_w = ({c: w for c, w
                     in rand_led[ak]["weights"].items() if w > 0}
                    if ak in rand_led else {})
        if r["mode"] == "model_teacher":
            assert model_w and random_w, \
                f"eligible anchor {ak} with empty weights"
        model_anchors[ak] = {
            "task": r["task"], "source_id": sid, "decision": d,
            "mode": r["mode"], "model_w": model_w,
            "random_w": random_w,
            "chunks": {c: norm_once(teach_chunks[f"{ak}_{c}"], True)
                       for c in r["weights"]}}

    # ---------- demo rehearsal (targets + prefixes; tokens: gate) ---
    dm = json.loads((DEMO_DIR / "manifest.json").read_text())
    demo_eps = {}
    for e in dm["episodes"]:
        ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
        if ep["split"] == "train":
            demo_eps[(e["task_id"], e["demo"])] = ep
    demo_keys = sorted(demo_eps)
    demo_rows = [(tid, di, ri) for (tid, di) in demo_keys
                 for ri in range(len(demo_eps[(tid, di)]["rows"]))]

    # ---------- frozen prefixes (token states come from the gate) ---
    pfx_cache = {}

    @torch.no_grad()
    def prefix_of(key, obs, lang):
        if key not in pfx_cache:
            if len(pfx_cache) > 120:
                pfx_cache.clear()
            b = runner._obs_to_policy_batch(obs, lang)
            pfx_cache[key] = (prefix_forward(runner.policy, b), b)
        return pfx_cache[key]

    # Token states (real/reset/shuffled per bare anchor + demo rows,
    # V7.3D order-only shuffles) come EXCLUSIVELY from the verified
    # gate blob loaded above — the registered no-re-unroll rule.
    _dev = {}

    def tokd(key):
        t = _dev.get(key)
        if t is None:
            t = toks[key].to(device).float()
            _dev[key] = t
        return t

    # ---------- matched schedule (v074 module) ----------------------
    sched_path = OUT / "matched_training_schedule.pt"
    g_sched = {akey: {"task": e["task"], "source_id": e["source_id"]}
               for akey, e in anchors.items()}
    t_sched = {ak: {"task": m["task"], "source_id": m["source_id"],
                    "mode": m["mode"]}
               for ak, m in model_anchors.items()}
    if sched_path.exists():
        sched = torch.load(sched_path, weights_only=False)
        verify_matched_policy_schedule(sched, g_sched, t_sched,
                                       demo_rows)
    else:
        sched = build_matched_policy_schedule(
            g_sched, t_sched, demo_rows, STEPS, RID)
        torch.save(sched, sched_path)
    assert len(sched["grounded"]) == 300, \
        f"grounded stream {len(sched['grounded'])} != 300"
    # full scheduled anchor set: gate + readout support (delta 4/5)
    sched_bares = sorted({anchors[a]["bare"]
                          for a in sched["grounded"]}
                         | {a for a, _x in sched["model"]})
    g_by_bare = {}
    for akey in sorted(anchors):
        g_by_bare.setdefault(anchors[akey]["bare"], anchors[akey])
    # gate census must cover the schedule (set composition; the
    # refusal NAMES the keys — a KeyError mid-run names nothing)
    need = [f"{v}::{bb}" for bb in sched_bares
            for v in ("real", "reset", "shuffled")]
    need += sorted({demo_tok_key(rt) for rt in
                    list(sched["demo"]) + list(sched["retention"])})
    miss = [kk for kk in need if kk not in toks]
    if miss:
        sys.exit(f"gate anchor_states census does not cover the "
                 f"schedule: {len(miss)}/{len(need)} keys missing "
                 f"from {states_path.name}: " + ", ".join(miss[:20])
                 + (" ..." if len(miss) > 20 else ""))

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    # every sha the manifest binds, recomputed from the CURRENT
    # artifacts (written once; re-verified on every resume)
    manifest_bindings = {
        "a_stage_report_sha256": sha256_file(a_stage_path),
        "gate_report_sha256": gate_report_sha,
        "gate_manifest_sha256": sha256_file(gate_man_path),
        "anchor_states_sha256": states_sha,
        "centering_sha256": center_sha,
        "schedule_sha256": sha256_file(sched_path),
        "teacher_seal_sha256": sha256_file(
            TEACH / "ledger_seal.json"),
        "lcwm_final_sha256": lcwm_sha,
    }
    mp = OUT / "run_manifest.json"
    if mp.exists():
        prev = json.loads(mp.read_text())
        if prev.get("interface") != args.interface:
            sys.exit(f"run root {OUT} already committed interface "
                     f"{prev.get('interface')!r}; refusing "
                     f"{args.interface!r}")
        stored = prev.get("artifact_sha256") or {}
        stale = [kk for kk in manifest_bindings
                 if stored.get(kk) != manifest_bindings[kk]]
        if stale:
            sys.exit(f"resume revalidation failed in {OUT}: "
                     + "; ".join(
                         f"{kk} stored {stored.get(kk)!r} != "
                         f"current {manifest_bindings[kk]!r}"
                         for kk in stale))
    else:
        mp.write_text(json.dumps({
            "schema": "v074_policy_manifest_v1", "run_schema":
            "v074", "run_id": RID, "git_sha": git_sha,
            "arms": ARMS,
            "interface": args.interface,
            "interface_decision": {
                "a_stage_report": f"{gate_root.name}/"
                                  f"{a_stage_path.name}",
                "a_stage_report_sha256": manifest_bindings[
                    "a_stage_report_sha256"],
                "a_stage_consequence": conseq,
                "required_interface": required_if},
            "binding_gate": {
                "report": f"{gate_root.name}/"
                          f"{gate_report_path.name}",
                "report_sha256": gate_report_sha,
                "manifest": f"{gate_root.name}/"
                            f"{gate_man_path.name}",
                "manifest_sha256": manifest_bindings[
                    "gate_manifest_sha256"],
                "mode": GATE_MODE, "gate_pass": True,
                "lcwm_sha256": lcwm_sha},
            "tqp_seed": tqp_seed,
            "tqp_seed_namespace": f"{RID}|tqp",
            "centering": {"path": f"{gate_root.name}/"
                                  f"{center_path.name}",
                          "sha256": center_sha,
                          "sigma": center.sigma,
                          "provenance": "binding-gate artifact, "
                                        "sha-verified; never "
                                        "refit here"},
            "anchor_states": {"path": f"{gate_root.name}/"
                                      f"{states_path.name}",
                              "sha256": states_sha,
                              "provenance": "binding-gate "
                                            "artifact, "
                                            "sha-verified; never "
                                            "re-unrolled here"},
            "trainable": "LCProj(bias-free, zero-init) + "
                         "action_out_proj (stock init)"
                         + (" + TokenQueryPool (sha-seeded)"
                            if args.interface == "token_query"
                            else ""),
            "frozen": "PrefixVLM, LCWM, all other expert blocks",
            "optimizer": {"steps": STEPS, "lr": LR, "wd": WD,
                          "clip": CLIP, "ckpt_every": CKPT_EVERY,
                          "selection": "final step ONLY"},
            "gate": {"step": GATE_STEP, "min_ratio": GATE_MIN,
                     "statistic": "Var(Wc c) / (||mean(Wc c)||^2 "
                                  "+ Var(Wc c)) over all scheduled "
                                  "anchors",
                     "enforced_on": "recurrent-state arms only "
                                    "(lc_grounded, lc_full, "
                                    "lc_random); recorded for all "
                                    "arms",
                     "scoping_rationale":
                         "correction_bc trains with c = 0 exactly "
                         "(registered constant-state control), so "
                         "Var(Wc c) is identically zero by "
                         "construction and the collapse statistic "
                         "is undefined for it; the gate exists to "
                         "catch interface collapse in arms that "
                         "consume real centered state content"},
            "seed_bases": {"T": T_BASE, "R": R_BASE, "D": D_BASE},
            "grounded_rows": len(grounded),
            "grounded_anchors": len(anchors),
            "teacher_anchors": len(model_anchors),
            "teacher_eligible": sum(
                1 for m in model_anchors.values()
                if m["mode"] == "model_teacher"),
            "scheduled_anchor_set": sched_bares,
            "roots": {"data": V74.name, "teacher": TEACH.name,
                      "lcwm": LCWM.name,
                      "gate": gate_root.name},
            "teacher_seal": json.loads(
                (TEACH / "ledger_seal.json").read_text()),
            "lcwm_final_sha": lcwm_sha,
            "schedule_sha256": manifest_bindings[
                "schedule_sha256"],
            "artifact_sha256": manifest_bindings,
        }, indent=2))

    def record_gate(arm_, entry):
        gp = OUT / "gate_status.json"
        cur = json.loads(gp.read_text()) if gp.exists() else {}
        cur[arm_] = entry
        gp.write_text(json.dumps(cur, indent=2))

    def train_arm(arm):
        cfg_a = ARMS[arm]
        aop.load_state_dict(stock_aop)
        for p in aop.parameters():
            p.requires_grad_(True)
        lc = LCProj().to(device)
        tqp = None
        if args.interface == "token_query":
            tqp = TokenQueryPool(D_Z, seed=tqp_seed).to(device)
        params = list(lc.parameters()) + list(aop.parameters())
        groups = {"lc_proj": list(lc.parameters()),
                  "action_out_proj": list(aop.parameters())}
        if tqp is not None:
            params += list(tqp.parameters())
            groups["token_query"] = list(tqp.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
        logs, grad_report = [], {}
        zero_c = torch.zeros(1, D_Z, device=device)

        def c_of(key):
            """Centered interface state for a cached token entry.
            token_query attends IN-GRAPH on every call (the
            lcwm/v074_interface caching constraint — only pre-pool
            tokens are ever cached)."""
            t = tokd(key)
            pool = t.mean(dim=1) if tqp is None else tqp(t)
            return center.apply(pool)

        def bias_if(key):
            """Interface bias with REAL cached content (used by the
            reset/shuffled substitutions and the gate)."""
            return lc(c_of(key))

        def bias_arm(key):
            if cfg_a["state"] == "zero":
                return lc(zero_c)   # c = 0 exactly (registered)
            return bias_if(key)

        @torch.no_grad()
        def wcc_stats():
            """W_c c statistics over ALL scheduled anchors (real
            centered content through this arm's projection)."""
            b = torch.cat([bias_if(f"real::{bb}")
                           for bb in sched_bares])
            mu_b = b.mean(dim=0)
            var = (b - mu_b).pow(2).sum(dim=1).mean()
            mn2 = mu_b.pow(2).sum()
            den = float(var + mn2)
            return {"n_anchors": len(sched_bares),
                    "var": float(var), "mean_norm_sq": float(mn2),
                    "ratio": (float(var / (var + mn2)) if den > 0
                              else 0.0),
                    "degenerate": bool(den == 0.0),
                    "per_anchor_norm": {
                        bb: float(x) for bb, x in zip(
                            sched_bares, b.norm(dim=1).tolist())}}

        def d10(a, b):
            return float((a[:10, :7] - b[:10, :7]).abs().mean())

        @torch.no_grad()
        def readouts(step_):
            """A/B/C LC attribution + reset/shuffled substitution +
            grounded-target reproduction, same-noise, aggregated
            over the FULL scheduled anchor set (registered delta 4):
            medians for the logs, per-anchor rows for the
            checkpoint."""
            from lcwm.lc_flow import sample_chunks_lc
            from lcwm.sampler import sample_chunks
            from lcwm.v067_lineage import flow_noise
            nz = flow_noise(sha_seed(f"{RID}|abc|{step_}"),
                            cfg.chunk_size,
                            cfg.max_action_dim).to(device)
            zb = torch.zeros(1, 1024, device=device)
            rows_, agg = [], defaultdict(list)
            for bare in sched_bares:
                mm = meta[bare]
                pf, pb = prefix_of(f"{bare}|{mm['decision']}",
                                   mm["rows"][mm["decision"]]["obs"],
                                   mm["instr"])
                chA = sample_chunks_lc(runner.policy, pb,
                                       bias_arm(f"real::{bare}"),
                                       n=1, noise=nz, prefix=pf)[0]
                chB = sample_chunks_lc(runner.policy, pb, zb, n=1,
                                       noise=nz, prefix=pf)[0]
                live = runner.policy.model.action_out_proj
                try:
                    runner.policy.model.action_out_proj = stock_head
                    chC = sample_chunks(runner.policy, pb, n=1,
                                        noise=nz, prefix=pf)[0]
                finally:
                    runner.policy.model.action_out_proj = live
                chR = sample_chunks_lc(runner.policy, pb,
                                       bias_if(f"reset::{bare}"),
                                       n=1, noise=nz, prefix=pf)[0]
                chS = sample_chunks_lc(runner.policy, pb,
                                       bias_if(f"shuffled::{bare}"),
                                       n=1, noise=nz, prefix=pf)[0]
                row = {"anchor": bare,
                       "A_minus_B_state_shift": d10(chA, chB),
                       "B_minus_C_head_drift": d10(chB, chC),
                       "A_minus_reset": d10(chA, chR),
                       "A_minus_shuffled": d10(chA, chS)}
                ge = g_by_bare.get(bare)
                if ge is not None and ge["targets"]:
                    tgt = ge["targets"][
                        sorted(ge["targets"])[0]][:10, :7]
                    row["target_repro_A"] = float(
                        (chA[:10, :7] - tgt).abs().mean())
                    row["target_repro_C"] = float(
                        (chC[:10, :7] - tgt).abs().mean())
                rows_.append(row)
                for kk, vv in row.items():
                    if kk != "anchor":
                        agg[kk].append(vv)
            med = {kk: float(np.median(vv))
                   for kk, vv in agg.items()}
            # separate support counts: target_repro_* medians run
            # over the grounded-target subset only, never over the
            # full anchor set
            med["readout_n_anchors_total"] = len(rows_)
            med["readout_n_anchors_with_grounded_targets"] = sum(
                1 for r_ in rows_ if "target_repro_A" in r_)
            return med, rows_

        @torch.no_grad()
        def env_eval(step_):
            """Registered SMALL training-state environment
            evaluation: re-reach one probe anchor (rotating over the
            v074B anchors), execute 10 actions from THIS arm's
            policy, then 60 stock continuation steps. Video saved
            for every rollout regardless of outcome. Diagnostic only
            — cannot select an arm or checkpoint (final-step
            rule)."""
            from lcwm.lc_flow import sample_chunks_lc
            from lcwm.loho_public import make_public_env
            from lcwm.sampler import sample_chunks
            from lcwm.task_automaton import GoalAutomaton
            from lcwm.v067_lineage import flow_noise
            from lcwm.video_recorder import (VideoRecorder,
                                             write_index_row)
            eval_keys = [a for a in sorted(anchors)
                         if a.startswith("v074B::")]
            if not eval_keys:
                return {}
            ak_ = eval_keys[(step_ // CKPT_EVERY - 1)
                            % len(eval_keys)]
            e_ = anchors[ak_]
            sid_ = e_["source_id"]
            src_ = torch.load(V74 / "sources" / f"{sid_}.pt",
                              weights_only=False)
            task_ = e_["task"]
            gm = json.loads((RESULTS
                             / "goal_spec_manifest_v067.json")
                            .read_text())["tasks"][task_]
            sub = gm["goal_specs"][
                gm["canonical_goal_spec_id"]]["ordered_subgoals"]
            env = make_public_env(task_, 1400)
            try:
                runner.reset()
                env.reset(seed=src_["seed"])
                env._env.env.horizon = 1400
                au = GoalAutomaton(list(sub))
                au.start(env)
                au.evaluate(env, 0)
                st = 0
                for i_ in range(e_["decision"]):
                    for a_env in src_["rows"][i_]["actions_env"]:
                        env.step(a_env)
                        st += 1
                        au.evaluate(env, st)
                vr = VideoRecorder(
                    OUT / "videos"
                    / f"{arm}_step{step_:03d}_{ak_[7:]}.mp4")
                obs_ = copy.deepcopy(env._format_raw_obs(
                    env._env.env._get_observations()))
                vr.add(obs_)
                b_ = runner._obs_to_policy_batch(obs_, e_["instr"])
                pf_ = prefix_forward(runner.policy, b_)
                nz = flow_noise(sha_seed(
                    f"{RID}|eval|{arm}|{step_}"), cfg.chunk_size,
                    cfg.max_action_dim).to(device)
                ch_ = sample_chunks_lc(
                    runner.policy, b_,
                    bias_arm(f"real::{e_['bare']}"), n=1,
                    noise=nz, prefix=pf_)
                for a_env in runner.chunk_to_env(ch_[:, :10]):
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    st += 1
                    au.evaluate(env, st)
                    obs_ = copy.deepcopy(env._format_raw_obs(
                        env._env.env._get_observations()))
                    vr.add(obs_)
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                q_after = au.q_valid()
                live = runner.policy.model.action_out_proj
                sc_, hit_term = 0, False
                try:
                    runner.policy.model.action_out_proj = stock_head
                    while sc_ < 60 and not hit_term:
                        b2 = runner._obs_to_policy_batch(
                            obs_, e_["instr"])
                        p2 = prefix_forward(runner.policy, b2)
                        nz2 = flow_noise(sha_seed(
                            f"{RID}|evalcont|{step_}|{sc_}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        c2 = sample_chunks(runner.policy, b2, n=1,
                                           noise=nz2.to(device),
                                           prefix=p2)
                        for a_env in runner.chunk_to_env(
                                c2[:, :10]):
                            _o, _r, tm, tr3, _i = env.step(a_env)
                            sc_ += 1
                            au.evaluate(env, st + sc_)
                            obs_ = copy.deepcopy(
                                env._format_raw_obs(
                                    env._env.env
                                    ._get_observations()))
                            vr.add(obs_)
                            if tm or tr3:
                                hit_term = bool(tm)
                                break
                            if sc_ >= 60:
                                break
                finally:
                    runner.policy.model.action_out_proj = live
                vm = vr.close(completed=True)
                write_index_row(
                    OUT / "video_index.jsonl", vm, run_id=RID,
                    checkpoint_tag=f"{arm}_step{step_}",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256="policy_training_eval",
                    task=task_, seed=src_["seed"], arm=arm,
                    split="train_state", steps=st + sc_,
                    success=bool(hit_term),
                    ordered_progress=au.ordered_prefix(),
                    damage=au.damage_unrecovered(),
                    termination=("terminal" if hit_term
                                 else "eval_end"), root=OUT)
                return {"eval_anchor": ak_,
                        "eval_terminal": bool(hit_term),
                        "eval_q_after10": float(q_after),
                        "eval_q_final": float(au.q_valid()),
                        "eval_prefix": int(au.ordered_prefix())}
            finally:
                env.close()

        def gnorm(ps):
            tot = 0.0
            for p_ in ps:
                if p_.grad is not None:
                    tot += float((p_.grad ** 2).sum())
            return tot ** 0.5

        for k in range(STEPS):
            opt.zero_grad(set_to_none=True)
            g = torch.Generator().manual_seed(T_BASE + k)
            noise = torch.randn(1, cfg.chunk_size,
                                cfg.max_action_dim,
                                generator=g).to(device)
            time = torch.rand(1, generator=g).to(device)
            comps = {}
            # (1) grounded first-ten
            ak = sched["grounded"][k]
            e = anchors[ak]
            pfx, _b = prefix_of(f"{e['bare']}|{e['decision']}",
                                e["rows"][e["decision"]]["obs"],
                                e["instr"])
            tgts = sorted(e["targets"])
            cands = torch.stack([e["targets"][c][:, :7]
                                 for c in tgts])
            w = torch.full((len(tgts),), 1.0 / len(tgts),
                           device=device)
            bias = bias_arm(f"real::{e['bare']}")
            comps["grounded"], _ = weighted_anchor_fm(
                runner.policy, pfx, cands, w, bias, noise, time)
            # (2) model / matched-random first-ten over the FULL
            #     teacher stream; abstention slots are PRESENT with
            #     weight exactly zero (registered delta 3)
            mk, m_active = sched["model"][k]
            if cfg_a["model_mass"]:
                me = model_anchors[mk]
                mm = meta[mk]
                mpfx, _mb = prefix_of(
                    f"{mk}|{mm['decision']}",
                    mm["rows"][mm["decision"]]["obs"],
                    mm["instr"])
                mbias = bias_arm(f"real::{mk}")
                if m_active:
                    wsrc = (me["model_w"]
                            if cfg_a["model_mass"] == "model"
                            else me["random_w"])
                    assert wsrc, \
                        f"eligible anchor {mk} with empty weights"
                    cids = sorted(wsrc)
                    mc = torch.stack([me["chunks"][c][:, :7]
                                      for c in cids])
                    tot = sum(wsrc[c] for c in cids)
                    mw = torch.tensor(
                        [wsrc[c] / tot for c in cids],
                        device=device)
                    comps["model"], _ = weighted_anchor_fm(
                        runner.policy, mpfx, mc, mw, mbias,
                        noise, time)
                else:
                    # zero-mass term: same uniform candidate graph
                    # for lc_full and lc_random, contribution
                    # exactly zero, never skipped
                    cids = sorted(me["chunks"])
                    mc = torch.stack([me["chunks"][c][:, :7]
                                      for c in cids])
                    mw = torch.full((len(cids),), 1.0 / len(cids),
                                    device=device)
                    zl, _ = weighted_anchor_fm(
                        runner.policy, mpfx, mc, mw, mbias,
                        noise, time)
                    comps["model"] = 0.0 * zl
            # (3) suffix trust at the correction anchor (10:50 only,
            #     against the STOCK head)
            u0 = e["rows"][e["decision"]]["chunk_norm"]
            comps["suffix_trust"] = suffix_trust(
                runner.policy, pfx,
                norm_once(u0, True)[:, :7], bias, noise, time,
                stock_head=stock_head)
            # (4) retention trust: full 0:50 on the retention stream
            rt = sched["retention"][k]
            rep = demo_eps[(rt[0], rt[1])]
            rpfx, _rb = prefix_of(f"demo{rt[0]}_{rt[1]}|{rt[2]}",
                                  rep["rows"][rt[2]]["obs"],
                                  rep["language"])
            rbias = bias_arm(demo_tok_key(rt))
            comps["retention_trust"] = suffix_trust(
                runner.policy, rpfx,
                rep["rows"][rt[2]]["chunk_norm"].to(device)[:, :7],
                rbias, noise, time, stock_head=stock_head,
                lo=0, hi=cfg.chunk_size)
            # (5) demo flow matching under the recurrent demo state
            dt = sched["demo"][k]
            dep = demo_eps[(dt[0], dt[1])]
            dpfx, _db = prefix_of(f"demo{dt[0]}_{dt[1]}|{dt[2]}",
                                  dep["rows"][dt[2]]["obs"],
                                  dep["language"])
            dbias = bias_arm(demo_tok_key(dt))
            g2 = torch.Generator().manual_seed(D_BASE + k)
            dn = torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                             generator=g2).to(device)
            dtm = torch.rand(1, generator=g2).to(device)
            dch = dep["rows"][dt[2]]["chunk_norm"].to(device)[None,
                                                              :, :7]
            comps["demo"], _ = weighted_anchor_fm(
                runner.policy, dpfx, dch,
                torch.ones(1, device=device), dbias, dn, dtm,
                credit_t=None)      # full-50 demonstration FM
            if k == 0:
                for name, term in comps.items():
                    opt.zero_grad(set_to_none=True)
                    term.backward(retain_graph=True)
                    grad_report[name] = {
                        gname: gnorm(ps)
                        for gname, ps in groups.items()}
                assert grad_report["grounded"]["lc_proj"] >= 0
                assert grad_report["demo"]["action_out_proj"] > 0, \
                    "demo term constant on the action head"
                (OUT / f"grad_checks_{arm}.json").write_text(
                    json.dumps(grad_report, indent=2))
                opt.zero_grad(set_to_none=True)
            total = sum(comps.values())
            assert torch.isfinite(total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            if (k + 1) % CKPT_EVERY == 0:
                ws = wcc_stats()
                if (k + 1) == GATE_STEP:
                    gate_ok = ws["ratio"] >= GATE_MIN
                    enforced = cfg_a["state"] == "recurrent"
                    record_gate(arm, {
                        "step": GATE_STEP, "ratio": ws["ratio"],
                        "min_ratio": GATE_MIN, "ok": gate_ok,
                        "enforced": enforced,
                        "interface": args.interface,
                        "n_anchors": ws["n_anchors"],
                        "degenerate": ws["degenerate"]})
                    if enforced and not gate_ok:
                        torch.save(
                            {"schema": "v074_policy_ckpt_v1",
                             "arm": arm, "step": k + 1,
                             "interface": args.interface,
                             "halted": "v074_step50_gate_failed",
                             "lc_proj": lc.state_dict(),
                             "action_out_proj": aop.state_dict(),
                             "token_query": (tqp.state_dict()
                                             if tqp is not None
                                             else None),
                             "wcc_stats": ws, "logs": logs},
                            OUT / "checkpoints"
                            / f"{arm}_halt_step{k + 1:03d}.pt")
                        m_ = json.loads(mp.read_text())
                        m_.setdefault("halts", {})[arm] = {
                            "status": "v074_step50_gate_failed",
                            "step": GATE_STEP,
                            "interface": args.interface,
                            "ratio": ws["ratio"],
                            "min_ratio": GATE_MIN,
                            "wcc_var": ws["var"],
                            "wcc_mean_norm_sq":
                                ws["mean_norm_sq"]}
                        mp.write_text(json.dumps(m_, indent=2))
                        sys.exit(
                            f"[HALT {arm}] v074_step50_gate_failed:"
                            f" ratio {ws['ratio']:.3e} < {GATE_MIN}"
                            f" over {ws['n_anchors']} scheduled "
                            f"anchors (registered V7.4A "
                            f"training-time check; fallback/restart"
                            f" is orchestrated outside)")
                med, rrows = readouts(k + 1)
                row = {"step": k + 1, **med, **env_eval(k + 1),
                       **{c: float(v) for c, v in comps.items()},
                       "wcc_ratio": ws["ratio"],
                       "wcc_var": ws["var"],
                       "wcc_mean_norm_sq": ws["mean_norm_sq"]}
                with torch.no_grad():
                    row["lc_proj_norm"] = float(
                        lc.lin.weight.norm())
                    row["aop_delta"] = float(
                        (aop.weight - stock_aop["weight"]).norm())
                    if tqp is not None:
                        row["tqp_query_norm"] = float(
                            tqp.query.norm())
                logs.append(row)
                torch.save({"schema": "v074_policy_ckpt_v1",
                            "arm": arm, "step": k + 1,
                            "interface": args.interface,
                            "lc_proj": lc.state_dict(),
                            "action_out_proj": aop.state_dict(),
                            "token_query": (tqp.state_dict()
                                            if tqp is not None
                                            else None),
                            "readout_rows": rrows,
                            "wcc_stats": ws, "logs": logs},
                           OUT / "checkpoints"
                           / f"{arm}_step{k + 1:03d}.pt")
                print(f"[{arm} {k + 1}/{STEPS}] " + " ".join(
                    f"{c}={float(v):.4f}"
                    for c, v in comps.items())
                    + f" |lc|={row['lc_proj_norm']:.4f}"
                      f" |dAOP|={row['aop_delta']:.4f}"
                      f" wcc={ws['ratio']:.3e}",
                    flush=True)
        torch.save({"schema": "v074_policy_ckpt_v1", "arm": arm,
                    "step": STEPS, "interface": args.interface,
                    "lc_proj": lc.state_dict(),
                    "action_out_proj": aop.state_dict(),
                    "token_query": (tqp.state_dict()
                                    if tqp is not None else None),
                    "grad_checks": grad_report, "logs": logs},
                   OUT / "checkpoints" / f"{arm}_final.pt")
        aop.load_state_dict(stock_aop)
        return {"final": f"{arm}_final.pt",
                "grad_checks": grad_report, "logs": logs}

    results = {}
    for arm in ([args.arm] if args.arm else list(ARMS)):
        results[arm] = train_arm(arm)
        (OUT / "offline_metrics.json").write_text(
            json.dumps(results, indent=2))
    print(json.dumps({a: r["logs"][-1] for a, r in results.items()},
                     indent=1), flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
