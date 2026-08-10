#!/usr/bin/env python
"""V7.4E — fresh public-LoHo recurrent N=1 endpoint (v0.8 slice).

5 arms x 5 public tasks x seeds {1850,1860,1870,1880,1890}
= 125 rollouts, 125 videos. Arms: stock (untouched pi0.5) +
correction_bc / lc_grounded / lc_full / lc_random (each = stock pi0.5
with its trained bias-free W_c and trained action_out_proj from the
*_v074_policy_r1 final checkpoints). The frozen V7.4C LCWM drives the
recurrent state through the DEPLOYED interface read from the policy
run manifest — pooled mean or the arm-trained TokenQueryPool run on
the pre-pool token states at every decision — then the registered
centering c_t = (Pool(z) - mu_train) / sigma_train loaded from the
policy run lineage (CenteredState.load; never refitted here).
correction_bc deploys the registered constant c = 0.

NEW registered artifact (2026-08-09.md V7.4E): per-decision ONLINE
STATE TRACES for every rollout — "pre-pool tokens, c_t, and
W_c c_t norms" (registration quoted verbatim: the actual [4,384]
token states are saved, float16, not only their norms) — named in
the run manifest, so a behavioral result can never again hide a
silent interface collapse.

Deployment contract: exact public task specs, full composite prompt,
recurrent carry from episode reset, N=1 generation. No scorer,
best-of-N, atomic prompt, GoalSpec, task ID, milestone, privileged
state, or planner. Shared environment seeds, task order, per-decision
flow noise (CRN namespace structurally arm-free), horizon, evaluator;
hash-sorted arm execution order. Every rollout indexed with a video.

Primary metric: task-balanced terminal success (exact-rational).
Winner/safety rule verbatim V7.3; the frozen V7.4 routing rows are
quoted in the paired report. Only frozen final-step checkpoints.
Output: results/libero_loho_public_v1/<DATE>_v074_loho_dev_r1/
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v074_interface import (CenteredState,  # noqa: E402
                                 TokenQueryPool)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RID = "v074_loho_dev_r1"
TASK_ORDER = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
              "loho_t4_tray", "loho_t5_drawer_cabinet"]
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
SEEDS = [1850, 1860, 1870, 1880, 1890]
# registered fresh development panel (2026-08-09.md): {1700..1740}
# sealed; {1750..1790} (V7.2) and {1800..1840} (V7.3E) consumed
assert SEEDS == list(range(1850, 1900, 10))
assert not set(SEEDS) & set(range(1700, 1850, 10))
ARMS = ["stock", "correction_bc", "lc_grounded", "lc_full",
        "lc_random"]
C_ZERO_ARMS = {"correction_bc"}     # registered constant c = 0
INTERFACES = ("pooled", "token_query")
TRACE_SCHEMA = "v074_state_trace_v1"
WINNER_RULE = ("lc_full > stock under the stock-only safety rule "
               "AND strictly greater task-balanced success than "
               "correction_bc, lc_grounded, lc_random "
               "(exact-rational comparison)")
# frozen routing rows, quoted from 2026-08-09.md "Routing (frozen)"
ROUTING_ROWS = [
    "winner -> replicate on the same panel with fresh CRN noise "
    "ids; only a replicated winner opens the sealed panel;",
    "all trained arms ≤ stock, full support -> valid negative "
    "for the v0.8 bundle; localize from LoHo failures; no sweep;",
    "all trained arms ≤ stock, support cells missing -> "
    "coverage-bounded negative with cells named; no sweep;",
    "V7.4A/D halt conditions -> the slice stops at the halt with "
    "the mechanical status named; no partial behavior claims.",
]


def sha_seed(p: str) -> int:
    return int.from_bytes(hashlib.sha256(
        p.encode()).digest()[:8], "big") & ((1 << 63) - 1)


def obs_frame(env):
    return copy.deepcopy(env._format_raw_obs(
        env._env.env._get_observations()))


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
                          ).stdout.strip()


def latest_root(suffix: str, require: str | None = None) \
        -> Path | None:
    """Last sorted-glob *_<suffix> root (v073 discovery idiom).
    `require` names a member path a root must carry to count — an
    empty stub *_v074_lcwm_r1 root exists on disk and must never
    be discovered as the checkpoint lineage."""
    root = None
    for p in sorted(RESULTS.glob(f"*_{suffix}")):
        if require is None or (p / require).exists():
            root = p
    return root


def records_fallback_commit(node) -> bool:
    """True if the policy manifest's decision record carries the
    A-stage consequence literal anywhere in its (trainer-owned,
    shape-unstable) nesting."""
    if isinstance(node, dict):
        return any(records_fallback_commit(v)
                   for v in node.values())
    if isinstance(node, (list, tuple)):
        return any(records_fallback_commit(v) for v in node)
    return node == "commit_fallback_token_query"


def support_coverage() -> dict:
    """Quote the V7.4B support-cell status for the registered
    coverage-bounded routing row. Real manifest shape (verified on
    disk): shortfall lives as rows INSIDE am["anchors"] with kind
    in {family_shortfall, SHORTFALL} — there is no top-level
    shortfall key — next to a top-level task_family_coverage block
    (the LATE blocker family is known-empty for all five tasks).
    Present-file/missing-key cases are named as unavailable, never
    silent nulls."""
    data_root = latest_root("v074_data_r1")
    out = {"data_root": data_root.name if data_root else None}
    if data_root is None:
        out["status"] = "unavailable: no *_v074_data_r1 root"
        return out
    try:
        am = json.loads(
            (data_root / "anchor_manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        am = None
        out["anchor_manifest"] = f"unavailable: {e}"
    if am is not None:
        out["task_family_coverage"] = (
            am["task_family_coverage"]
            if "task_family_coverage" in am
            else "unavailable: 'task_family_coverage' key missing "
                 "in anchor_manifest.json")
        if "anchors" not in am:
            out["shortfall"] = ("unavailable: 'anchors' key "
                                "missing in anchor_manifest.json")
        else:
            per_cell = defaultdict(int)
            per_kind = defaultdict(int)
            for row in am["anchors"]:
                kind = row.get("kind")
                if kind in ("family_shortfall", "SHORTFALL"):
                    per_cell[f"{row.get('task')}|"
                             f"{row.get('family')}"] += 1
                    per_kind[kind] += 1
            out["shortfall"] = {
                "rows_per_task_family": dict(
                    sorted(per_cell.items())),
                "rows_per_kind": dict(sorted(per_kind.items()))}
    try:
        osup = json.loads(
            (data_root / "outcome_support.json").read_text())
        out["registered_task_checks"] = (
            osup["registered_task_checks"]
            if "registered_task_checks" in osup
            else "unavailable: 'registered_task_checks' key "
                 "missing in outcome_support.json")
    except (OSError, json.JSONDecodeError) as e:
        out["outcome_support"] = f"unavailable: {e}"
    return out


@torch.no_grad()
def main() -> None:
    from lcwm.chassis import Pi05Runner
    from lcwm.lc_flow import freeze_pi05_base, sample_chunks_lc
    from lcwm.loho_public import make_public_env
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.task_automaton import GoalAutomaton, terminal_success
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import (VideoRecorder, video_name,
                                     write_index_row)

    device = torch.device("cuda")
    lcwm_root = latest_root("v074_lcwm_r1",
                            require="checkpoints/final.pt")
    policy_root = latest_root("v074_policy_r1")
    if lcwm_root is None or policy_root is None:
        sys.exit("missing *_v074_lcwm_r1 (must carry checkpoints/"
                 "final.pt — a stub root without the final "
                 "checkpoint does not count) / *_v074_policy_r1 "
                 "roots — V7.4E runs only after C and D land "
                 "(execution order, 2026-08-09.md)")
    pm = json.loads((policy_root / "run_manifest.json").read_text())
    interface = pm.get("interface")
    if interface not in INTERFACES:
        sys.exit(f"policy manifest interface {interface!r} not in "
                 f"{INTERFACES} — the deployed interface must be "
                 f"read from the policy run manifest")
    if interface == "pooled":
        dec = pm.get("interface_decision") or {}
        if (pm.get("tqp_seed") is not None
                or dec.get("probe_interface") == "token_query"
                or records_fallback_commit(dec)):
            sys.exit("policy manifest deploys 'pooled' but records "
                     "a token_query commitment (A-stage consequence "
                     "commit_fallback_token_query / token_query "
                     "decision artifacts) — the registered fallback "
                     "switch is one-way; refusing pooled deployment")
    lcwm_final_sha = sha256_file(
        lcwm_root / "checkpoints" / "final.pt")
    if pm.get("lcwm_final_sha") != lcwm_final_sha:
        sys.exit(f"cross-binding failure: {lcwm_root.name} "
                 f"final.pt sha {lcwm_final_sha[:12]} != policy "
                 f"manifest lcwm_final_sha "
                 f"{str(pm.get('lcwm_final_sha'))[:12]} — the "
                 f"independently discovered roots are not one "
                 f"lineage")
    cent = pm.get("centering")
    centering_path = None
    if cent is not None:
        if not (isinstance(cent, dict) and cent.get("path")
                and cent.get("sha256")):
            sys.exit(f"policy manifest centering record {cent!r} "
                     f"lacks path/sha256 — cannot reconstruct the "
                     f"training-time centering")
        cp = Path(cent["path"])
        for cand in ([cp] if cp.is_absolute()
                     else [policy_root / cp, RESULTS / cp,
                           REPO_ROOT / cp]):
            if cand.exists() \
                    and sha256_file(cand) == cent["sha256"]:
                centering_path = cand
                break
        if centering_path is None:
            sys.exit(f"centering {cent['path']} (sha "
                     f"{cent['sha256'][:12]}) not found with "
                     f"matching content under {policy_root}, "
                     f"{RESULTS}, or {REPO_ROOT} — refusing "
                     f"filename fallback: deployment must "
                     f"reconstruct the exact training-time "
                     f"statistics, sha-verified")
    else:
        for name in (f"centering_{interface}.pt", "centering.pt"):
            if (policy_root / name).exists():
                centering_path = policy_root / name
                break
    if centering_path is None:
        sys.exit(f"no centering statistics under {policy_root} — "
                 f"the registered c_t needs the frozen train-set "
                 f"stats (CenteredState.save lineage file)")
    center = CenteredState.load(centering_path)
    centering_sha = sha256_file(centering_path)

    root = latest_root(RID) or RESULTS / f"{run_date()}_{RID}"
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "state_traces").mkdir(exist_ok=True)
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    freeze_pi05_base(runner.policy)
    for p in runner.policy.parameters():
        p.requires_grad_(False)
    aop = runner.policy.model.action_out_proj
    stock_aop = {k: v.detach().clone()
                 for k, v in aop.state_dict().items()}
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(
        lcwm_root / "checkpoints" / "final.pt",
        weights_only=False)["model"])
    wm.eval()

    ck_shas = {}
    arm_blobs = {}
    for arm in ARMS:
        if arm == "stock":
            continue
        pth = policy_root / "checkpoints" / f"{arm}_final.pt"
        ck_shas[arm] = sha256_file(pth)
        blob = torch.load(pth, weights_only=False)
        assert "lc_proj" in blob and "action_out_proj" in blob, \
            f"{arm}: checkpoint missing lc_proj/action_out_proj"
        assert blob.get("arm") in (None, arm), (arm, blob.get("arm"))
        if interface == "token_query" and arm not in C_ZERO_ARMS:
            # key presence is vacuous — the trainer writes
            # "token_query": None for non-state arms
            assert blob.get("token_query") is not None, (
                f"{arm}: token_query interface deployed but the "
                f"checkpoint carries no trained TokenQueryPool")
        arm_blobs[arm] = blob

    mp = root / "run_manifest.json"
    cent_name = (str(centering_path.relative_to(RESULTS))
                 if centering_path.is_relative_to(RESULTS)
                 else str(centering_path))
    manifest = {
        "schema": "v074_loho_dev_manifest_v1",
        "run_schema": "v074", "run_id": RID,
        "registration": "plan_and_progress/2026-08-09.md V7.4E",
        "git_sha": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=REPO_ROOT).stdout.strip(),
        "arms": ARMS, "seeds": SEEDS,
        "panel": "registered fresh development panel "
                 "{1850..1890}; sealed {1700..1740} untouched; "
                 "consumed {1750..1790}/{1800..1840} unused",
        "deployment": "recurrent full-prompt N=1; no scorer/"
                      "best-of-N/GoalSpec/taskID/atomic/planner",
        "interface": interface,
        "policy_root": policy_root.name,
        "lcwm_root": lcwm_root.name,
        "policy_manifest_sha256": sha256_file(
            policy_root / "run_manifest.json"),
        "centering": {"path": cent_name,
                      "sha256": centering_sha,
                      "sigma": center.sigma},
        "noise_crn": f"SHA256('{RID}|act|task|seed|decision')"
                     " — arm identity structurally absent",
        "arm_order": "hash-sorted per (task, seed)",
        "policy_checkpoints": ck_shas,
        "lcwm_final_sha": lcwm_final_sha,
        "c_zero_arms": sorted(C_ZERO_ARMS),
        "state_traces": {
            "schema": TRACE_SCHEMA, "dir": "state_traces",
            "pattern": f"{RID}_final_<task>_s<seed>_<arm>.pt",
            "fields": "tokens [N,4,384] float16 (the actual "
                      "pre-pool token states — registration-"
                      "named), token_norms [N,4], c [N,384] "
                      "(deployed c_t), c_norm [N], wc_c_norm "
                      "[N] per decision; empty for stock",
            "registered": "2026-08-09.md V7.4E per-decision "
                          "online state traces"},
    }
    if mp.exists():
        prev = json.loads(mp.read_text())
        binding = ["arms", "seeds", "interface", "policy_root",
                   "lcwm_root", "policy_manifest_sha256",
                   "lcwm_final_sha", "policy_checkpoints",
                   "centering"]
        bad = [k for k in binding if prev.get(k) != manifest[k]]
        if bad:
            sys.exit(f"resume refused: existing {mp} disagrees "
                     f"with the currently discovered roots/"
                     f"interface/checkpoint shas on {bad} — a "
                     f"resumed run must deploy the exact original "
                     f"lineage")
    else:
        mp.write_text(json.dumps(manifest, indent=2))

    rec_path = root / "records.jsonl"
    vindex = root / "video_index.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.open():
            r = json.loads(line)
            done.add((r["arm"], r["task"], r["seed"]))

    lc_cache, tqp_cache = {}, {}

    def lc_of(arm):
        if arm not in lc_cache:
            lin = nn.Linear(384, 1024, bias=False).to(device)
            lin.load_state_dict(
                {"weight": arm_blobs[arm]["lc_proj"]["lin.weight"]
                 .to(device)})
            lin.eval()
            lc_cache[arm] = lin
        return lc_cache[arm]

    def tqp_of(arm):
        # arm-trained fallback pool, run on the pre-pool token
        # states at EVERY decision (cached pooled outputs banned by
        # the module contract)
        if arm not in tqp_cache:
            tqp = TokenQueryPool(384).to(device)
            tqp.load_state_dict(
                {k: v.to(device) for k, v
                 in arm_blobs[arm]["token_query"].items()})
            tqp.eval()
            tqp_cache[arm] = tqp
        return tqp_cache[arm]

    for task in TASK_ORDER:
        entry = goal_manifest["tasks"][task]
        canon = entry["canonical_goal_spec_id"]
        subgoals = entry["goal_specs"][canon]["ordered_subgoals"]
        for seed in SEEDS:
            order = sorted(ARMS, key=lambda a: sha_seed(
                f"{RID}|order|{task}|{seed}|{a}"))
            for arm in order:
                if (arm, task, seed) in done:
                    continue
                if arm == "stock":
                    aop.load_state_dict(stock_aop)
                else:
                    aop.load_state_dict(
                        arm_blobs[arm]["action_out_proj"])
                env = make_public_env(task, EPISODE_LENGTH[task])
                try:
                    runner.reset()
                    env.reset(seed=seed)
                    env._env.env.horizon = EPISODE_LENGTH[task] + 50
                    instr = env.task_description
                    auto = GoalAutomaton(list(subgoals))
                    auto.start(env)
                    auto.evaluate(env, 0)
                    vr = VideoRecorder(root / "videos" / video_name(
                        RID, "final", task, seed, arm))
                    obs = obs_frame(env)
                    vr.add(obs)
                    z, prev_chunk = None, None
                    t, dec, success = 0, 0, False
                    q_traj = []
                    tr_tokens, tr_tok, tr_c, tr_wc = \
                        [], [], [], []
                    term = trunc = False
                    while t < EPISODE_LENGTH[task]:
                        batch = runner._obs_to_policy_batch(obs,
                                                            instr)
                        pfx = prefix_forward(runner.policy, batch)
                        nz = flow_noise(sha_seed(
                            f"{RID}|act|{task}|{seed}|{dec}"),
                            cfg.chunk_size, cfg.max_action_dim)
                        if arm == "stock":
                            ch = sample_chunks(
                                runner.policy, batch, n=1,
                                noise=nz.to(device), prefix=pfx)
                        else:
                            h = pfx.hidden.float()
                            m = pfx.pad_masks.bool()
                            if z is None:
                                z = wm.initial_state(h, m)
                            else:
                                am = torch.ones(
                                    1, 10, dtype=torch.bool,
                                    device=device)
                                z = wm.step(z, prev_chunk, h, m,
                                            action_mask=am)
                            # deployed interface = exact
                            # training-time state path
                            if arm in C_ZERO_ARMS:
                                c = torch.zeros(1, 384,
                                                device=device)
                            else:
                                pool = (tqp_of(arm)(z)
                                        if interface == "token_query"
                                        else z.mean(dim=1))
                                c = center.apply(pool)
                            bias = lc_of(arm)(c)
                            tr_tokens.append(z[0].half().cpu())
                            tr_tok.append(
                                z[0].norm(dim=-1).float().cpu())
                            tr_c.append(c[0].float().cpu())
                            tr_wc.append(float(bias.norm()))
                            ch = sample_chunks_lc(
                                runner.policy, batch, bias, n=1,
                                noise=nz.to(device), prefix=pfx)
                        prev_chunk = ch[:, :10].float()
                        for a_env in runner.chunk_to_env(
                                ch[:, :10]):
                            _o, _r, term, trunc, _i = env.step(
                                a_env)
                            t += 1
                            auto.evaluate(env, t)
                            q_traj.append(auto.q_valid())
                            obs = obs_frame(env)
                            vr.add(obs)
                            if terminal_success(env):
                                success = True
                            if success or term or trunc \
                                    or t >= EPISODE_LENGTH[task]:
                                break
                        dec += 1
                        if success or term or trunc:
                            break
                    vm = vr.close(completed=True)
                    # registered per-rollout online state trace —
                    # written BEFORE the record row so no resumable
                    # record can exist without its trace
                    tr_path = (root / "state_traces" /
                               f"{RID}_final_{task}_s{seed}"
                               f"_{arm}.pt")
                    c_stack = (torch.stack(tr_c) if tr_c
                               else torch.zeros(0, 384))
                    torch.save({
                        "schema": TRACE_SCHEMA, "run_id": RID,
                        "arm": arm, "task": task, "seed": seed,
                        "interface": (None if arm == "stock"
                                      else interface),
                        "c_is_constant_zero": arm in C_ZERO_ARMS,
                        "decisions": dec,
                        "tokens": (torch.stack(tr_tokens)
                                   if tr_tokens
                                   else torch.zeros(
                                       0, 4, 384,
                                       dtype=torch.float16)),
                        "token_norms": (torch.stack(tr_tok)
                                        if tr_tok
                                        else torch.zeros(0, 4)),
                        "c": c_stack,
                        "c_norm": c_stack.norm(dim=-1),
                        "wc_c_norm": torch.tensor(tr_wc),
                        "note": ("stock arm: no LC interface — "
                                 "decision count only"
                                 if arm == "stock" else None),
                    }, tr_path)
                    first_un = next(
                        (subgoals[i] for i, v in
                         enumerate(auto.prev_valid) if not v), None)
                    rec = {"arm": arm, "task": task, "seed": seed,
                           "success": bool(success), "steps": t,
                           "decisions": dec,
                           "ordered_prefix": auto.ordered_prefix(),
                           "q_final": auto.q_valid(),
                           "q_auc": float(np.mean(q_traj))
                           if q_traj else 0.0,
                           "damage": auto.damage_unrecovered(),
                           "first_unresolved": first_un,
                           "video": vm["video_path"],
                           "state_trace": str(
                               tr_path.relative_to(root))}
                    with rec_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    write_index_row(
                        vindex, vm, run_id=RID,
                        checkpoint_tag="final",
                        checkpoint_path=(
                            str(policy_root / "checkpoints"
                                / f"{arm}_final.pt")
                            if arm != "stock" else None),
                        checkpoint_sha256=ck_shas.get(arm),
                        manifest_sha256=sha256_file(mp),
                        task=task, seed=seed, arm=arm, split="dev",
                        steps=t, success=bool(success),
                        ordered_progress=auto.ordered_prefix(),
                        damage=auto.damage_unrecovered(),
                        termination=("success" if success
                                     else "horizon"), root=root)
                    print(f"[{task} s{seed}] {arm}: "
                          f"success={success} steps={t} "
                          f"q_auc={rec['q_auc']:.3f} "
                          f"dmg={rec['damage']}", flush=True)
                finally:
                    env.close()
    aop.load_state_dict(stock_aop)

    # ---------------- paired report ---------------------------------
    recs = [json.loads(x) for x in rec_path.open()]
    table = defaultdict(dict)
    for r in recs:
        table[(r["task"], r["seed"])][r["arm"]] = r

    def per_task_counts(arm, key="success"):
        """EXACT per-task tallies (numerator, denominator) — the
        registered comparison is on task-balanced rates and must not
        be decided by float summation order (review: np.mean over 5
        per-task rates gives different floats for the SAME total, so
        genuine ties were reported as wins)."""
        by = defaultdict(list)
        for (task, _s), row in table.items():
            if arm in row:
                by[task].append(row[arm][key])
        return {t_: (sum(Fraction(int(x) if isinstance(x, bool)
                                  else x).limit_denominator(10**6)
                         for x in v), len(v))
                for t_, v in by.items()}

    def balanced(arm, key="success"):
        c = per_task_counts(arm, key)
        if not c:
            return Fraction(0)
        return sum(Fraction(n, d) for n, d in c.values()) \
            / len(c)

    def per_task(arm, key="success"):
        return {t_: float(Fraction(n, d))
                for t_, (n, d) in per_task_counts(arm, key).items()}

    def gt(a, b):
        ca, cb = per_task_counts(a), per_task_counts(b)
        tasks = sorted(set(ca) & set(cb))
        bal_a, bal_b = balanced(a), balanced(b)
        pos = [t_ for t_ in tasks
               if Fraction(*ca[t_]) > Fraction(*cb[t_])]
        reg = [t_ for t_ in tasks
               if Fraction(*ca[t_]) < Fraction(*cb[t_])]
        da, db = per_task_counts(a, "damage"), \
            per_task_counts(b, "damage")
        dmg_up = [t_ for t_ in tasks
                  if Fraction(*da[t_]) > Fraction(*db[t_])]
        out = {"a": a, "b": b, "bal_a": float(bal_a),
               "bal_b": float(bal_b),
               "bal_a_exact": str(bal_a), "bal_b_exact": str(bal_b),
               "tasks_positive": pos, "task_regressions": reg,
               "damage_increase_tasks": dmg_up,
               "strictly_greater": bool(bal_a > bal_b),
               "exact_tie": bool(bal_a == bal_b)}
        if b == "stock":
            # the >=2-positive / no-regression / no-damage-increase
            # safety rule is registered RELATIVE TO STOCK only
            out["phase1_win_vs_stock"] = bool(
                bal_a > bal_b and len(pos) >= 2 and not reg
                and not dmg_up)
        return out

    pairs = [("lc_full", "stock"), ("lc_full", "correction_bc"),
             ("lc_full", "lc_grounded"), ("lc_full", "lc_random"),
             ("lc_grounded", "correction_bc"),
             ("lc_grounded", "stock"), ("correction_bc", "stock"),
             ("lc_random", "stock")]
    verdicts = {f"{a}>{b}": gt(a, b) for a, b in pairs}
    winner = bool(
        verdicts["lc_full>stock"].get("phase1_win_vs_stock")
        and verdicts["lc_full>correction_bc"]["strictly_greater"]
        and verdicts["lc_full>lc_grounded"]["strictly_greater"]
        and verdicts["lc_full>lc_random"]["strictly_greater"])
    report = {
        "v074_development_winner": winner,
        "winner_rule": WINNER_RULE,
        "interface": interface,
        "task_balanced_success": {a: per_task(a) for a in ARMS},
        "task_balanced_overall": {a: str(balanced(a))
                                  for a in ARMS},
        "task_balanced_q_auc": {a: per_task(a, "q_auc")
                                for a in ARMS},
        "task_balanced_damage": {a: per_task(a, "damage")
                                 for a in ARMS},
        "paired": list(verdicts.values()),
        "n_rollouts": len(recs),
        "routing_registered": {
            "source": "plan_and_progress/2026-08-09.md — V7.4E "
                      "Routing (frozen)",
            "rows": ROUTING_ROWS},
        "support_coverage": support_coverage(),
    }
    (root / "paired_report.json").write_text(
        json.dumps(report, indent=2))
    (root / "task_seed_table.json").write_text(json.dumps(
        {f"{t}|{s}": {a: r["success"] for a, r in row.items()}
         for (t, s), row in sorted(table.items())}, indent=2))
    fail = defaultdict(list)
    for r in recs:
        if not r["success"]:
            fail[r["arm"]].append({
                "task": r["task"], "seed": r["seed"],
                "first_unresolved": r["first_unresolved"],
                "ordered_prefix": r["ordered_prefix"]})
    (root / "failure_localization.json").write_text(
        json.dumps(fail, indent=2))
    print(json.dumps(report["task_balanced_success"], indent=1),
          flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
