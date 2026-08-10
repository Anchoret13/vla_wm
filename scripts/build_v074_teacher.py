#!/usr/bin/env python
"""V7.4D teacher step — u0-anchored model teacher + grounded ledger
(2026-08-09.md "V7.4D — teacher, matched schedule, policy
fine-tuning"; framework_design.md §11.6 / §13.5).

Derived from scripts/build_v073_teacher.py with ONLY the registered
V7.4 deltas:

  1. inputs: the frozen V7.4C LCWM final checkpoint
     (*_v074_lcwm_r1) and the V7.4B teacher banks/anchors
     (*_v074_data_r1, kind "teach"); dev margins are frozen on the
     v074 dev anchors under that checkpoint. Margin unrolls use the
     live prefix_forward path — the v073 hcache shortcut is dropped
     so the margin instrument and the teacher decode share one
     pipeline (canonical language == the p0 text variant, verified);
  2. ELIGIBILITY (registered change): a candidate becomes
     mode=model_teacher ONLY if its predicted registered outcome
     tuple strictly dominates the SAME anchor's predicted u0 tuple:
     diff >= 0 on ALL in-tuple components AND diff >= the frozen dev
     margin (and > 0) on >= 1 component. Identity selection
     (candidate chunk == u0's chunk) and predicted ties are
     abstention. The parent's candidate-vs-candidate lexicographic
     rule survives only as the WINNER ordering among eligible
     candidates;
  3. matched-random permutes candidate identity within the same
     eligible cells (inherited V7.3 mechanics, verbatim);
  4. ledger rows gain predicted_u0_tuple + winner_margin_components;
     unique eligible anchors are reported by task, source, phase
     (state_type) and failure mode (family|first_unresolved) in
     eligibility_report.json;
  5. ledgers sealed outcome-blind (SHAs in ledger_seal.json) BEFORE
     any execution/policy step, exactly as the parent.

Component masking inherited: damage/tau have no decoded labels; the
registered chain is success -> milestone -> q_mean, with success
re-entering the tuple ONLY if the v074 dev success ground truth is
non-constant (the parent dropped it because its dev margin was
one-sided against constant-0 GT — that drop is conditional, not
permanent). Milestone stays in-tuple as in the parent even when its
dev GT is constant (recorded as a false-positive scale).

Grounded ledger (inherited construction, v074 sourcing): the
milestone-positive replay-clean V7.4B train branches + the RELEASED
V7.3 teacher-assessment candidates whose full recomputed tuple beats
u0 in both repeats (the v073 released-winner pattern applied to the
2026-08-07 tranche). The released v073 assessment OUTCOMES enter
V7.4C training as provenance-labeled evidence per the 2026-08-03
release rule — that is the training path, not this ledger; the seal
records precisely what feeds each ledger.

Phase B (simulator): execute the v074 teacher banks once as an
assessment/replay tranche with videos and R=2 sibling-CRN
continuations. Outcomes may enter a future revision, never V7.4
model or policy weights.

Output: results/libero_loho_public_v1/<date>_v074_teacher_r1/
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V73 = RESULTS / "2026-08-04_v073_data_r1"
V72T = RESULTS / "2026-08-03_v072_teacher_r1"
V73T = RESULTS / "2026-08-07_v073_teacher_r1"
RID = "v074_teacher_r1"
EPISODE_LENGTH = {"loho_t1_drawer": 700, "loho_t2_basket3": 900,
                  "loho_t3_tray": 900, "loho_t4_tray": 900,
                  "loho_t5_drawer_cabinet": 990}
# frozen second GoalSpec per task (v074 crossing instrument literal)
ALT_GOAL = {"loho_t1_drawer": "t1_back_butter",
            "loho_t2_basket3": "t2_cheese_butter_milk",
            "loho_t3_tray": "t3_sauce_ketchup_butter",
            "loho_t4_tray": "t4_right_bowl",
            "loho_t5_drawer_cabinet": "t5_front_butter"}
# registered chain order (2026-08-03): terminal success ->
# next-milestone improvement -> multi-horizon task-valid progress
CHAIN = ("success", "milestone", "q_mean")
HORIZONS, R, CONT_MAX = (10, 30, 60, 100), 2, 100
REREACH_ATOL = 2e-3


class BindingGateError(RuntimeError):
    """Registered execution order violated: the pre-D binding gate
    has not PASSED on the exact V7.4C checkpoint this script would
    score with. There is no override — the gate is not skippable by
    registration (2026-08-09.md pre-C amendment)."""


class DevMarginSupportError(RuntimeError):
    """The frozen dev-margin instrument has empty support (no dev
    anchors, no scored error rows, or every chain component masked).
    Margins may never be silently vacuous."""


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


def latest(pattern: str):
    root = None
    for p in sorted(RESULTS.glob(pattern)):
        root = p
    return root


@torch.no_grad()
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("A", "B", "ALL"),
                    default="ALL")
    args = ap.parse_args()
    from lcwm.chassis import Pi05Runner
    from lcwm.loho_public import make_public_env
    from lcwm.probe_data import body_positions
    from lcwm.sampler import prefix_forward, sample_chunks
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.snapshot import restore, snap
    from lcwm.task_automaton import (GoalAutomaton, fork_env_state,
                                     outcome_tuple,
                                     paired_preference,
                                     restore_env_state)
    from lcwm import v072_schedule as S72
    from lcwm.v067_lineage import flow_noise, sha256_file
    from lcwm.video_recorder import VideoRecorder, write_index_row
    from scripts.collect_loho_v06 import obs_q
    from scripts.collect_v067_continuations import grasp_state

    device = torch.device("cuda")
    DATA = latest("*_v074_data_r1")
    assert DATA is not None, "v074 data root missing"
    LCWM = latest("*_v074_lcwm_r1")
    assert LCWM is not None, "v074 LCWM root missing (run V7.4C)"
    ckpt_path = LCWM / "checkpoints" / "final.pt"
    assert ckpt_path.exists(), f"missing {ckpt_path}"
    # registered execution order, enforced mechanically: C -> binding
    # probe gate -> D. The gate report must exist, be a PASSED
    # binding_gate run, and be bound to THIS final.pt before any
    # scoring. Deliberately no --skip-gate-check flag: the gate is
    # not skippable by registration.
    diag = latest("*_v074_interface_diag_r1")
    if diag is None:
        raise BindingGateError(
            f"no *_v074_interface_diag_r1 root under {RESULTS} — "
            "run scripts/probe_v074_interface.py --universe v074 "
            "--interface token_query first")
    gate_path = diag / "probe_report_v074_token_query.json"
    if not gate_path.exists():
        raise BindingGateError(
            f"binding-gate report missing: {gate_path} — run "
            "scripts/probe_v074_interface.py --universe v074 "
            "--interface token_query on the V7.4C final checkpoint")
    gate = json.loads(gate_path.read_text())
    ckpt_sha = sha256_file(ckpt_path)
    if gate.get("mode") != "binding_gate":
        raise BindingGateError(
            f"{gate_path} mode={gate.get('mode')!r} != "
            "'binding_gate' (diagnostic reports do not satisfy the "
            "pre-D gate)")
    if gate.get("gate_pass") is not True:
        raise BindingGateError(
            f"binding gate did not pass "
            f"(gate_pass={gate.get('gate_pass')!r}) — V7.4D may "
            "not start; see the registered halt/fallback rule")
    if gate.get("lcwm_sha256") != ckpt_sha:
        raise BindingGateError(
            f"binding gate ran on lcwm_sha256="
            f"{gate.get('lcwm_sha256')!r} but this script would "
            f"score with {ckpt_sha} ({ckpt_path}) — re-run the "
            "gate on the checkpoint being scored")
    OUT = latest(f"*_{RID}") or RESULTS / f"{run_date()}_{RID}"
    OUT.mkdir(exist_ok=True)
    for sub in ("shards", "videos"):
        (OUT / sub).mkdir(exist_ok=True)
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]

    a_manifest = json.loads(
        (DATA / "anchor_manifest.json").read_text())
    # SHORTFALL/family_shortfall rows carry no decision — every
    # consumer below filters on "kind" and never assumes full rows
    anchors74 = a_manifest["anchors"]
    teach_anchors = [a for a in anchors74 if a["kind"] == "teach"]
    acq_sids = {a["source_id"] for a in anchors74
                if a["kind"] == "acq"}
    teach_sids = {a["source_id"] for a in teach_anchors}
    assert teach_anchors, "no v074 teacher anchors"
    assert not (teach_sids & acq_sids), "teacher source overlap"
    missing_banks = [
        a for a in teach_anchors
        if not (DATA / "teacher_banks"
                / f"{a['source_id']}_d{a['decision']}.pt").exists()]
    assert not missing_banks, \
        f"teacher banks missing (B unfinished?): {missing_banks}"
    # extended registered disjointness: v074 teacher sources must be
    # absent from EVERY universe feeding V7.4C training — v074 acq,
    # v073 acq, the v072 union, v072 teacher sources, and the v073
    # teacher sources (their assessment outcomes are RELEASED into
    # V7.4 training per the 2026-08-03 release rule)
    anchors73 = json.loads(
        (V73 / "anchor_manifest.json").read_text())["anchors"]
    train_sids = set(acq_sids)
    train_sids |= {a["source_id"] for a in anchors73
                   if a["kind"] in ("acq", "teach")}
    union72 = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                          / "union_manifest.json").read_text())
    train_sids |= set(union72["source_histories"])
    for p_ in (V72T / "teacher_sources").glob("*.pt"):
        train_sids.add(p_.stem)
    assert not (teach_sids & train_sids), \
        f"teacher sources appear in training universes: " \
        f"{teach_sids & train_sids}"
    # candidate chunk hashes must not appear among ANY chunk feeding
    # V7.4C training — all FIVE universes of the merged
    # lcwm/v074_schedule.py data, hashed from the same artifacts
    # train_v074_lcwm.py consumes:
    #   v072B          union rows' stored actions_pi05_norm
    #                  (S72.load_union, non-audit);
    #   v072T          teacher-shard env actions (kind "chunk");
    #   v073 / v074    shard chunk_norm where stored, env actions
    #                  otherwise (u0 / recovery rows);
    #   v073T_released candidate_chunks.pt.
    # Env-action rows are hashed raw AND in normalize_actions form
    # (full sequence and the padded 10-step model-entry chunk) so a
    # copied-then-normalized chunk cannot slip a byte-level check.

    def env_chunk_hashes(a_env):
        ae = np.asarray(a_env)
        hs = [hashlib.sha256(ae.tobytes()).hexdigest()]
        if len(ae):
            cn = normalize_actions(
                torch.from_numpy(ae).float(), ref_mean, ref_std)
            hs.append(hashlib.sha256(
                cn.numpy().tobytes()).hexdigest())
            c10 = cn[:10]
            if c10.shape[0] < 10:
                c10 = torch.cat([c10, torch.zeros(
                    10 - c10.shape[0], 7)], dim=0)
            hs.append(hashlib.sha256(
                c10.numpy().tobytes()).hexdigest())
        return hs

    train_chunk_shas = set()
    for shard_dir in (V73 / "shards", DATA / "shards"):
        for p_ in sorted(shard_dir.glob("*.pt")):
            sh_ = torch.load(p_, weights_only=False)
            for tr_ in sh_["transitions"]:
                if tr_.get("chunk_norm") is not None:
                    train_chunk_shas.add(hashlib.sha256(
                        np.asarray(tr_["chunk_norm"]).tobytes())
                        .hexdigest())
                else:
                    train_chunk_shas.update(
                        env_chunk_hashes(tr_["actions_env"]))
    for ch_ in torch.load(V73T / "candidate_chunks.pt",
                          weights_only=False).values():
        train_chunk_shas.add(hashlib.sha256(
            np.asarray(ch_).tobytes()).hexdigest())
    for rec_ in S72.load_union():
        if rec_["audit_only"] \
                or rec_["actions_pi05_norm"] is None:
            continue
        train_chunk_shas.add(hashlib.sha256(
            np.asarray(rec_["actions_pi05_norm"]).tobytes())
            .hexdigest())
    for p_ in sorted((V72T / "shards").glob("*.pt")):
        sh_ = torch.load(p_, weights_only=False)
        for tr_ in sh_["transitions"]:
            if tr_["kind"] == "audit":
                continue
            train_chunk_shas.update(
                env_chunk_hashes(tr_["actions_env"]))
    teach_chunk_shas = set()
    for a_ in teach_anchors:
        tb_ = torch.load(
            DATA / "teacher_banks"
            / f"{a_['source_id']}_d{a_['decision']}.pt",
            weights_only=False)
        for _cid, (_f, _l, ch_) in tb_["bank"].items():
            teach_chunk_shas.add(hashlib.sha256(
                np.asarray(ch_).tobytes()).hexdigest())
    overlap = train_chunk_shas & teach_chunk_shas
    assert not overlap, f"chunk-hash overlap: {len(overlap)}"

    runner = Pi05Runner(suite_name="libero_10")
    cfg = runner.policy.config
    wm = V06State().to(device)
    wm.load_state_dict(torch.load(ckpt_path,
                                  weights_only=False)["model"])
    wm.eval()

    def norm_chunk(cn_or_env, is_norm):
        if is_norm:
            cn = torch.as_tensor(np.asarray(cn_or_env),
                                 dtype=torch.float32,
                                 device=device)[None, :10]
        else:
            ae = torch.from_numpy(np.asarray(cn_or_env)).float()
            cn = normalize_actions(
                ae, ref_mean, ref_std)[None].to(device)[:, :10]
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        am = torch.ones(1, 10, dtype=torch.bool, device=device)
        return cn, am

    def unroll_live(src, d, instr):
        z = None
        for dd in range(d + 1):
            b = runner._obs_to_policy_batch(src["rows"][dd]["obs"],
                                            instr)
            pfx = prefix_forward(runner.policy, b)
            h, m = pfx.hidden.float(), pfx.pad_masks.bool()
            if z is None:
                z = wm.initial_state(h, m)
            else:
                prev = src["rows"][dd - 1]
                aa = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = wm.step(z, aa, h, m, action_mask=am)
        return z

    def n_sub_of(task):
        entry = goal_manifest["tasks"][task]
        cg = entry["canonical_goal_spec_id"]
        return len(entry["goal_specs"][cg]["ordered_subgoals"])

    def decode(z, cn, am, task):
        zt = wm.predict(z, cn, action_mask=am)
        o = wm.d_next(zt)
        q = torch.sigmoid(o["q_valid"][0]).cpu().numpy()
        n = n_sub_of(task)
        return {
            "success": float(torch.sigmoid(
                o["success_logit"]).reshape(-1)[0]),
            "milestone": float(torch.sigmoid(
                o["flips_01"][0, :n]).max()),
            "q_mean": float(q.mean()),
            "p_valid": float(torch.sigmoid(
                o["p_valid"].reshape(-1)[0])),
        }

    # ---- Phase A ----------------------------------------------------
    seal_path = OUT / "ledger_seal.json"
    if not seal_path.exists():
        # (1) frozen dev error margins per component on the v074 dev
        # anchors under the frozen V7.4C final.pt, plus per-component
        # ground-truth constancy (decides success's tuple membership)
        margins_path = OUT / "dev_margins.json"
        errs, gts = defaultdict(list), defaultdict(list)
        dev_anchors = sorted(
            (a for a in anchors74
             if a["kind"] == "acq" and a["role"] == "dev"),
            key=lambda a: (a["source_id"], a["decision"]))
        if not dev_anchors:
            raise DevMarginSupportError(
                "no dev acquisition anchors in "
                f"{DATA / 'anchor_manifest.json'} — the frozen "
                "dev-margin instrument has no support")
        for a in dev_anchors:
            sid, d = a["source_id"], a["decision"]
            sh = torch.load(DATA / "shards" / f"{sid}_d{d}.pt",
                            weights_only=False)
            canon = sh["goal_ids"][0]
            src = torch.load(DATA / "sources" / f"{sid}.pt",
                             weights_only=False)
            z = unroll_live(src, d, src["language_canonical"])
            for tr in sh["transitions"]:
                if tr["branch_key"] == "u0_repeat" \
                        or not tr["continuations"]:
                    continue
                cn, am = norm_chunk(
                    tr["chunk_norm"] if tr["chunk_norm"] is not None
                    else tr["actions_env"],
                    tr["chunk_norm"] is not None)
                pred = decode(z, cn, am, a["task"])
                gs = []
                for ys in tr["continuations"]:
                    c = ys[canon]
                    qv = {int(k): v for k, v in
                          c["q_at_horizons"].items()}
                    gs.append([c["p_valid_100"],
                               np.mean([qv[h_] for h_ in HORIZONS]),
                               float(c["success_by_100"])])
                gt_p = float(np.mean([g[0] for g in gs]))
                gt_q = float(np.mean([g[1] for g in gs]))
                gt_s = float(np.mean([g[2] for g in gs]))
                vs = tr["valid_seq"][canon]
                gt_m = float((np.asarray(vs[-1], dtype=bool)
                              & ~np.asarray(vs[0],
                                            dtype=bool)).any())
                errs["p_valid"].append(abs(pred["p_valid"] - gt_p))
                errs["q_mean"].append(abs(pred["q_mean"] - gt_q))
                errs["milestone"].append(
                    abs(pred["milestone"] - gt_m))
                errs["success"].append(abs(pred["success"] - gt_s))
                gts["success"].append(gt_s)
                gts["milestone"].append(gt_m)
        if not any(errs.values()):
            raise DevMarginSupportError(
                f"{len(dev_anchors)} dev anchors yielded zero "
                "scored error rows (no non-u0 branches with "
                "continuations) — margins would be vacuous")
        margins = {k: float(np.median(v)) for k, v in errs.items()}
        gt_constant = {k: (len(set(gts[k])) <= 1 if gts[k]
                           else None)
                       for k in ("success", "milestone")}
        # success re-enters the registered chain ONLY with real dev
        # label variance (parent drop-rationale is conditional);
        # milestone stays in-tuple as in the parent even one-sided.
        # A zero-support component is MASKED out of the tuple and
        # recorded — never defaulted to margin 0 (dominates() below
        # indexes margins directly, so a masked component can never
        # silently admit a candidate)
        masked_zero_support = sorted(
            k for k in CHAIN if not errs.get(k))
        comps = [k for k in CHAIN
                 if k not in masked_zero_support
                 and (k != "success"
                      or gt_constant["success"] is False)]
        if not comps:
            raise DevMarginSupportError(
                "every registered chain component is masked "
                f"(masked_zero_support={masked_zero_support}, "
                f"gt_constant={gt_constant}) — no eligibility "
                "instrument exists")
        assert all(k in margins for k in comps), (comps, margins)
        margins_path.write_text(json.dumps(
            {"margins": margins,
             "n_dev_anchors": len(dev_anchors),
             "n_dev_rows": {k: len(v) for k, v in errs.items()},
             "gt_constant": gt_constant,
             "masked_zero_support": masked_zero_support,
             "components_in_tuple": comps,
             "note": "median |error| per component on v074 dev "
                     "anchors under the frozen v074 final.pt; "
                     "frozen BEFORE any teacher decoding; live "
                     "unroll (no hcache) so margins and teacher "
                     "decode share one pipeline; success in-tuple "
                     "iff dev success GT non-constant (ordinary "
                     "continuations only; recovery-terminal rows "
                     "are a different-horizon instrument); "
                     "milestone kept as in the parent (one-sided "
                     "margin = false-positive scale when GT "
                     "constant); damage/tau masked (no labels); "
                     "zero-support components masked out of the "
                     "tuple (masked_zero_support), never margin-0"},
            indent=2))
        print(f"[margins] {margins} comps={comps} "
              f"masked={masked_zero_support}", flush=True)

        # (2) model + matched-random ledgers on the teacher banks
        led_rows, rnd_rows = [], []
        chunk_store = {}
        for a in teach_anchors:
            sid, d = a["source_id"], a["decision"]
            akey = f"{sid}_d{d}"
            tb = torch.load(DATA / "teacher_banks" / f"{akey}.pt",
                            weights_only=False)
            src = torch.load(DATA / "sources" / f"{sid}.pt",
                             weights_only=False)
            instr = src["language_canonical"]
            z = unroll_live(src, d, instr)
            preds, fams, c_sha = {}, {}, {}
            for cid, (fam, _lang, ch) in tb["bank"].items():
                cn, am = norm_chunk(ch, True)
                preds[cid] = decode(z, cn, am, a["task"])
                fams[cid] = fam
                c_sha[cid] = hashlib.sha256(
                    np.asarray(ch).tobytes()).hexdigest()
                chunk_store[f"{akey}_{cid}"] = ch
            m = margins
            u0p = preds["u0"]
            identity_cands = sorted(
                c for c in preds
                if c != "u0" and c_sha[c] == c_sha["u0"])

            def dominates(ci):
                """REGISTERED V7.4D rule (u0-anchored): eligible iff
                the predicted tuple weakly dominates u0's predicted
                tuple on ALL in-tuple components (diff >= 0) AND
                beats it by >= the frozen dev margin (and > 0, so a
                zero margin can never admit a tie) on >= 1
                component. Identity candidates (chunk == u0's chunk)
                and exact predicted ties abstain by construction.
                Returns (component, chain_rank, margin, diffs) with
                component = the EARLIEST qualifying chain rung — the
                parent's lexicographic winner ordering, now applied
                only AMONG eligible candidates."""
                if ci in identity_cands:
                    return None
                p = preds[ci]
                df = {k: p[k] - u0p[k] for k in comps}
                if any(v < 0.0 for v in df.values()):
                    return None
                # m[k] (not .get) — comps is asserted a subset of
                # the supported margin keys at freeze time
                qual = [(comps.index(k), k, df[k]) for k in comps
                        if df[k] >= m[k] and df[k] > 0.0]
                if not qual:
                    return None
                rank, k, mg = min(qual, key=lambda x: x[0])
                return (k, rank, mg, df)

            elig = {}
            for cid in preds:
                if cid == "u0":
                    continue
                r_ = dominates(cid)
                if r_ is not None:
                    elig[cid] = r_
            if elig:
                # lexicographic winner: earliest chain component
                # first, then max margin WITHIN that component
                # (cross-component margin magnitudes are not
                # comparable — inherited review rule)
                best_rank = min(v[1] for v in elig.values())
                pool_l = {c: v for c, v in elig.items()
                          if v[1] == best_rank}
                best_margin = max(v[2] for v in pool_l.values())
                winners = sorted(
                    c for c, v in pool_l.items()
                    if abs(v[2] - best_margin) < 1e-9)
                w = {c: (1.0 / len(winners) if c in winners
                         else 0.0) for c in preds}
                mode = "model_teacher"
            else:
                winners = []
                w = {c: 0.0 for c in preds}
                mode = "abstain"
            led_rows.append({
                "anchor": akey, "task": a["task"],
                "source_id": sid, "decision": d, "mode": mode,
                "predictions": preds,
                "predicted_u0_tuple": u0p,
                "eligible": {
                    c: {"component": v[0], "margin": v[2],
                        "diffs": v[3]}
                    for c, v in elig.items()},
                "winner_margin_components": {
                    c: elig[c][3] for c in winners},
                "components_in_tuple": comps,
                "identity_candidates": identity_cands,
                "weights": w, "families": fams})
            # matched-random: permute identity within eligible-mass
            # support, preserving family composition of the winners
            # (inherited V7.3 mechanics, verbatim)
            unpermutable = False
            if elig:
                win_fams = {fams[c] for c in winners}
                pool = [c for c in preds if c != "u0"
                        and fams[c] in win_fams]
                alt_pool = [c for c in pool if c not in winners]
                if not alt_pool:
                    # no alternative identity exists in the winner
                    # families: the anchor is UNPERMUTABLE — keep the
                    # weights, mark the row, and exclude its mass
                    # from the identity-control claim (registered
                    # V7.2C precedent; never silently copy the
                    # model's winner as its own control)
                    unpermutable = True
                    rw = dict(w)
                else:
                    rot = sha_seed(f"{RID}|rnd|{akey}") \
                        % len(alt_pool)
                    rnd_winners = [
                        alt_pool[(rot + k) % len(alt_pool)]
                        for k in range(len(winners))]
                    rw = {c: (sum(1 for x in rnd_winners
                                  if x == c) / len(rnd_winners))
                          for c in preds}
            else:
                rw = {c: 0.0 for c in preds}
            rnd_rows.append({"anchor": akey, "weights": rw,
                             "mode": mode,
                             "unpermutable": unpermutable})
            print(f"[teach] {akey}: {mode} "
                  f"{[c for c, ww in w.items() if ww > 0]}",
                  flush=True)

        # (2b) registered eligibility report: unique eligible anchors
        # by task, source, phase (state_type), failure mode
        # (family|first_unresolved); masked family cells named via
        # the frozen task_family_coverage (LATE came back empty)
        meta = {f"{a['source_id']}_d{a['decision']}": a
                for a in teach_anchors}
        rep = {"by_task": defaultdict(list),
               "by_source": defaultdict(list),
               "by_phase": defaultdict(list),
               "by_failure_mode": defaultdict(list)}
        for r_ in led_rows:
            if r_["mode"] != "model_teacher":
                continue
            ak, ma = r_["anchor"], meta[r_["anchor"]]
            rep["by_task"][r_["task"]].append(ak)
            rep["by_source"][r_["source_id"]].append(ak)
            rep["by_phase"][ma.get("state_type",
                                   "unknown")].append(ak)
            rep["by_failure_mode"][
                f"{ma.get('family', 'unknown')}|"
                f"{ma.get('first_unresolved', 'unknown')}"
            ].append(ak)
        elig_report = {
            "schema": "v074_teacher_eligibility_v1",
            "n_teacher_anchors": len(led_rows),
            "n_model_teacher": sum(1 for r_ in led_rows
                                   if r_["mode"]
                                   == "model_teacher"),
            "n_abstain": sum(1 for r_ in led_rows
                             if r_["mode"] == "abstain"),
            "unique_eligible_anchors": {
                k: {kk: sorted(set(vv)) for kk, vv in v.items()}
                for k, v in rep.items()},
            "task_family_coverage":
                a_manifest["task_family_coverage"],
            "note": "outcome-blind (predictions only); empty family "
                    "cells are coverage masks inherited from V7.4B "
                    "family_shortfall rows, never refilled here"}
        (OUT / "eligibility_report.json").write_text(
            json.dumps(elig_report, indent=2))
        n_by_task = {k: len(set(v))
                     for k, v in rep["by_task"].items()}
        print(f"[elig] model_teacher "
              f"{elig_report['n_model_teacher']}"
              f"/{len(led_rows)} by_task={n_by_task}", flush=True)

        # (3) grounded ledger
        grounded = []
        # V7.4B milestone-positive replay-clean train branches
        fid = json.loads((DATA / "replay_fidelity.json").read_text())
        for p_ in sorted((DATA / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            if sh["role"] != "train":
                continue
            canon = sh["goal_ids"][0]
            for tr in sh["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                vs = tr["valid_seq"][canon]
                pos = (np.asarray(vs[-1], dtype=bool)
                       & ~np.asarray(vs[0], dtype=bool)).any()
                if pos and sh["anchor"] in fid:
                    grounded.append({
                        "origin": "v074B",
                        "anchor": sh["anchor"], "task": sh["task"],
                        "branch_key": tr["branch_key"],
                        "provenance": tr["provenance"],
                        "chunk_norm_present":
                            tr["chunk_norm"] is not None})
        # released V7.3 teacher-assessment candidates: full-tuple
        # both-repeat beat over u0 (the v073 released-winner pattern
        # applied to the 2026-08-07 tranche; registered lexicographic
        # comparator with frozen tolerances via paired_preference)
        _tol = json.loads((RESULTS / "v067_support_report.json")
                          .read_text())["frozen_outcome_tolerances"]
        for p_ in sorted((V73T / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            trs = {t_["candidate_id"]: t_
                   for t_ in sh["transitions"]}
            if "u0" not in trs:
                continue
            for cid, t_ in trs.items():
                if cid == "u0" or len(t_["continuations"]) < 2:
                    continue
                verdict = paired_preference(
                    t_["continuations"],
                    trs["u0"]["continuations"], _tol)
                if verdict == 1:
                    grounded.append({
                        "origin": "v073T_released",
                        "anchor": sh["anchor"], "task": sh["task"],
                        "branch_key": cid,
                        "provenance": "released_assessment",
                        "chunk_norm_present": True})
        with (OUT / "grounded_ledger.jsonl").open("w") as f:
            for g_ in grounded:
                f.write(json.dumps(g_) + "\n")
        with (OUT / "model_ledger_pre_outcome.jsonl").open("w") as f:
            for r_ in led_rows:
                f.write(json.dumps(r_) + "\n")
        with (OUT / "matched_random_ledger.jsonl").open("w") as f:
            for r_ in rnd_rows:
                f.write(json.dumps(r_) + "\n")
        torch.save(chunk_store, OUT / "candidate_chunks.pt")
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=REPO_ROOT).stdout.strip()
        seal_path.write_text(json.dumps({
            "git_sha": git_sha,
            "run_schema": "v074",
            "data_root": DATA.name, "lcwm_root": LCWM.name,
            "lcwm_final_sha": ckpt_sha,
            "binding_gate_root": diag.name,
            "binding_gate_report": gate_path.name,
            "binding_gate_sha": sha256_file(gate_path),
            "dev_margins_sha": sha256_file(margins_path),
            "model_ledger_sha": sha256_file(
                OUT / "model_ledger_pre_outcome.jsonl"),
            "matched_random_sha": sha256_file(
                OUT / "matched_random_ledger.jsonl"),
            "grounded_sha": sha256_file(
                OUT / "grounded_ledger.jsonl"),
            "eligibility_report_sha": sha256_file(
                OUT / "eligibility_report.json"),
            "chunks_sha": sha256_file(OUT / "candidate_chunks.pt"),
            "ledger_provenance": {
                "model": "frozen v074 final.pt predictions over the "
                         "unexecuted v074 teacher banks; u0-anchored "
                         "registered dominance rule; outcome-blind",
                "matched_random": "identity permutation within the "
                                  "model ledger's eligible cells "
                                  "(winner families), inherited "
                                  "V7.3 mechanics",
                "grounded": [
                    "v074B milestone-positive replay-clean train "
                    "branches (replay_fidelity-covered anchors)",
                    "released v073 teacher-assessment candidates "
                    "beating u0 in both repeats "
                    "(paired_preference, frozen v067 tolerances)"],
                "not_in_grounded": "released v073 assessment "
                                   "OUTCOMES enter V7.4C training "
                                   "as provenance-labeled evidence "
                                   "(2026-08-03 release rule); "
                                   "v072T released winners are not "
                                   "re-listed — consumed by the "
                                   "v073 grounded ledger and "
                                   "already inside the v073/v074 "
                                   "training universes"},
            "sealed_before_any_execution": True}, indent=2))
        n_model = sum(1 for r_ in led_rows
                      if r_["mode"] == "model_teacher")
        print(f"[seal] model-teacher anchors "
              f"{n_model}/{len(led_rows)}, "
              f"grounded rows {len(grounded)}", flush=True)
    if args.phase == "A":
        return

    # ---- Phase B: execute teacher banks (assessment only) ----------
    seal = json.loads(seal_path.read_text())
    from lcwm.v067_lineage import sha256_file as shaf
    assert seal["model_ledger_sha"] == shaf(
        OUT / "model_ledger_pre_outcome.jsonl"), "ledger modified"
    chunk_store = torch.load(OUT / "candidate_chunks.pt",
                             weights_only=False)
    done = {p.stem for p in (OUT / "shards").glob("*.pt")}
    vindex = OUT / "video_index.jsonl"
    for a in teach_anchors:
        sid, d, task = a["source_id"], a["decision"], a["task"]
        akey = f"{sid}_d{d}"
        if akey in done:
            continue
        src = torch.load(DATA / "sources" / f"{sid}.pt",
                         weights_only=False)
        entry = goal_manifest["tasks"][task]
        canon_id = entry["canonical_goal_spec_id"]
        gspecs = entry["goal_specs"]
        instr = src["language_canonical"]
        env = make_public_env(task, EPISODE_LENGTH[task] + 400)
        try:
            runner.reset()
            env.reset(seed=src["seed"])
            env._env.env.horizon = EPISODE_LENGTH[task] + 500
            alt_id = ALT_GOAL[task]   # frozen v074 crossing literal
            auto0 = GoalAutomaton(
                gspecs[canon_id]["ordered_subgoals"])
            auto0.start(env)
            auto0.evaluate(env, 0)
            alt0 = GoalAutomaton(gspecs[alt_id]["ordered_subgoals"])
            alt0.start(env)          # start_pos from EPISODE RESET
            alt0.evaluate(env, 0)
            bodies = auto0.bodies
            t = 0
            for i in range(d):
                for a_env in src["rows"][i]["actions_env"]:
                    env.step(a_env)
                    t += 1
                    auto0.evaluate(env, t)
                    alt0.evaluate(env, t)
            row = src["rows"][d]
            err = float(np.abs(body_positions(
                env, list(bodies.values()))
                - row["obj_before"]).max())
            assert err < REREACH_ATOL
            snap_a = snap(env, t=t, suite_name="loho_public",
                          task_id=0)
            a_state = fork_env_state(auto0)
            alt_state = fork_env_state(alt0)
            cids = sorted(c[len(akey) + 1:]
                          for c in chunk_store
                          if c.startswith(f"{akey}_"))
            order = sorted(cids, key=lambda c: sha_seed(
                f"{RID}|order|{sid}|{d}|{c}"))
            transitions = []
            for cid in order:
                restore(env, snap_a)
                env._env.env.done = False
                ba = GoalAutomaton(
                    gspecs[canon_id]["ordered_subgoals"])
                ba.bodies, ba.start_pos = auto0.bodies, \
                    auto0.start_pos
                restore_env_state(ba, a_state)
                b_alt = GoalAutomaton(
                    gspecs[alt_id]["ordered_subgoals"])
                b_alt.bodies, b_alt.start_pos = alt0.bodies, \
                    alt0.start_pos
                restore_env_state(b_alt, alt_state)
                ch = chunk_store[f"{akey}_{cid}"]
                if cid == "u0":
                    acts = list(row["actions_env"])[:10]
                else:
                    acts = list(runner.chunk_to_env(
                        torch.as_tensor(np.asarray(ch))[None, :10]
                        .to(device)))[:10]
                vr = VideoRecorder(OUT / "videos"
                                   / f"{akey}_{cid}.mp4")
                frames = [obs_frame(env)]
                vr.add(frames[0])
                eef_seq = [obs_q(frames[0])]
                valid_seq = [list(ba.prev_valid)]
                valid_seq_alt = [list(b_alt.prev_valid)]
                a_env_l, steps = [], 0
                for a_env in acts:
                    _o, _r, tb, tr2, _i = env.step(a_env)
                    steps += 1
                    a_env_l.append(np.asarray(a_env))
                    ba.evaluate(env, steps)
                    b_alt.evaluate(env, steps)
                    valid_seq.append(list(ba.prev_valid))
                    valid_seq_alt.append(list(b_alt.prev_valid))
                    f = obs_frame(env)
                    frames.append(f)
                    vr.add(f)
                    eef_seq.append(obs_q(f))
                    if tb:
                        env._env.env.done = False
                    if tr2:
                        break
                conts = []
                b_end = snap(env, t=t + steps,
                             suite_name="loho_public", task_id=0)
                for rep in range(R):
                    restore(env, b_end)
                    env._env.env.done = False
                    ca = GoalAutomaton(
                        gspecs[canon_id]["ordered_subgoals"])
                    ca.bodies, ca.start_pos = auto0.bodies, \
                        auto0.start_pos
                    restore_env_state(ca, fork_env_state(ba))
                    ca.flips = [(-1, i_, d_) for (_s, i_, d_)
                                in ca.flips]
                    obs_c = obs_frame(env)
                    q_at, sc, cd2, stop = {}, 0, 0, False
                    while sc < CONT_MAX and not stop:
                        nz = flow_noise(sha_seed(
                            f"{RID}|cont|{sid}|{d}|{canon_id}"
                            f"|{rep}|{cd2}"), cfg.chunk_size,
                            cfg.max_action_dim)
                        bcc = runner._obs_to_policy_batch(obs_c,
                                                          instr)
                        pcc = prefix_forward(runner.policy, bcc)
                        ch2 = sample_chunks(runner.policy, bcc,
                                            n=1,
                                            noise=nz.to(device),
                                            prefix=pcc)
                        for a_env in runner.chunk_to_env(
                                ch2[:, :10]):
                            _o2, _r, tm, tr3, _i = env.step(a_env)
                            sc += 1
                            ca.evaluate(env, steps + sc)
                            obs_c = obs_frame(env)
                            if sc in HORIZONS:
                                q_at[sc] = ca.q_valid()
                            if tm or tr3:
                                stop = True
                                break
                            if sc >= CONT_MAX:
                                break
                        cd2 += 1
                    for hz in HORIZONS:
                        q_at.setdefault(hz, ca.q_valid())
                    conts.append(outcome_tuple(ca, env, steps,
                                               q_at))
                vm = vr.close(completed=True)
                write_index_row(
                    vindex, vm, run_id=RID,
                    checkpoint_tag="teacher_assessment",
                    checkpoint_path=None, checkpoint_sha256=None,
                    manifest_sha256=seal["model_ledger_sha"],
                    task=task, seed=src["seed"], arm=cid,
                    split="teacher", steps=steps,
                    success=bool(any(c["success_by_100"]
                                     for c in conts)),
                    ordered_progress=ba.ordered_prefix(),
                    damage=ba.damage_unrecovered(),
                    termination="branch_end", root=OUT)
                transitions.append({
                    "transition_id": f"{akey}_{cid}",
                    "candidate_id": cid, "anchor": akey,
                    "task": task, "frames": frames,
                    "eef_seq": eef_seq,
                    "valid_seq_canon": valid_seq,
                    "valid_seq_alt": {alt_id: valid_seq_alt},
                    "actions_env": (np.stack(a_env_l) if a_env_l
                                    else np.zeros((0, 7))),
                    "steps": steps,
                    "grasp_after": grasp_state(env, bodies),
                    "continuations": conts})
                print(f"  [{akey}] {cid} steps={steps}",
                      flush=True)
            tmp = OUT / "shards" / f"{akey}.pt.tmp"
            torch.save({"schema": "v074_teacher_shard_v1",
                        "run_schema": "v074",
                        "anchor": akey, "task": task,
                        "transitions": transitions}, tmp)
            tmp.replace(OUT / "shards" / f"{akey}.pt")
            done.add(akey)
        finally:
            env.close()
    with (OUT / "outcome_ledger.jsonl").open("w") as f:
        for p_ in sorted((OUT / "shards").glob("*.pt")):
            sh = torch.load(p_, weights_only=False)
            for tr in sh["transitions"]:
                f.write(json.dumps({
                    "anchor": sh["anchor"],
                    "candidate_id": tr["candidate_id"],
                    "steps": tr["steps"],
                    "continuations": [
                        {k: (v if k != "q_at_horizons" else
                             {str(kk): vv for kk, vv in v.items()})
                         for k, v in c.items()}
                        for c in tr["continuations"]]}) + "\n")
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
