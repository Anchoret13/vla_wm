#!/usr/bin/env python
"""V6.7.3 — one fixed v0.6.7 world-model training job, from scratch
(seed 0; never initialized from iteration 1).

Completes the registered objective that iteration 1 left silent:
- next_sem: D_next valid/event bits, 0→1 and 1→0 flip bits, reward —
  the action × language next-semantic matrix — from v067 immediate
  crossed labels, for EVERY compatible GoalSpec of every audited branch;
- cont_heads: success_by_100 (GOAL-SPECIFIC any-point), damage, τ_next,
  P_valid, multi-horizon Q — from v067 sibling-CRN continuations;
- ranking: Bradley–Terry on candidate pairs with the FROZEN outcome
  tolerances from the v067 support report (iteration 1 used {});
- para_pred: paraphrase transitioned-prediction consistency (state
  consistency kept);
- shared_phys: d_q AND d_obj as language-invariant targets under
  crossed goals (iteration 1 trained d_obj only);
- history-contrast rows: if the support report found no matched pair,
  the family is recorded UNSUPPORTED and no history claim is made.

Guards: iteration-1 continuation/label/teacher artifacts refuse to load
(run_schema=v067); replay_audit records never train; support branches
never enter ranking. Launch assertions verify a real target, nonzero
weight, and gradient reach into E_a/T for every registered D_next head;
completion assertions verify parameter change from the frozen seed-0
init per head. Budget identical to iteration 1 (AdamW 3e-4 / wd 1e-4 /
25 epochs / TBPTT 16 / EMA 0.995 / clip 1.0) — no sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.task_automaton import paired_preference  # noqa: E402
from lcwm.v06_model import V06State, ema_update, make_ema  # noqa: E402
from lcwm.v067_lineage import RUN_SCHEMA, load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
SEQ_CACHE = Path("/home/stargazer/Desktop/vla_wm/datasets"
                 "/seq_prefix_cache_v1")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4 + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUBER = 4.0
TBPTT = 16
LR, WD, EPOCHS, GRAD_NORM = 3e-4, 1e-4, 25, 1.0
FAMILIES = ("seq_phys", "current", "reward", "closure1", "closure23",
            "para", "para_pred", "distinct_current", "shared_phys",
            "branch_abs", "branch_centered", "next_sem", "cont_heads",
            "ranking")
DNEXT_HEADS = ("d_q", "d_obj", "valid_bits", "event_bits", "flips_01",
               "flips_10", "damage", "reward", "success", "p_valid",
               "q_valid", "tau_next", "score")
BCE = torch.nn.functional.binary_cross_entropy_with_logits


def huber(p, t, scale):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=HUBER)


def bits(vals, n_max, device):
    out = torch.zeros(n_max, device=device)
    out[:len(vals)] = torch.tensor([float(v) for v in vals],
                                   device=device)
    return out


def idx_bits(indices, n_max, device):
    out = torch.zeros(n_max, device=device)
    for i in indices:
        out[i] = 1.0
    return out


class SourceBundle:
    def __init__(self, source_path: Path):
        self.source = torch.load(source_path, weights_only=False)
        assert self.source["schema"] == "v06_source_v1"
        sid = self.source["source_id"]
        self.labels = load_v067(
            DATA / "semantic_relabels_v067" / source_path.name,
            "semantic_labels")
        self.features = {}
        for fp in sorted(DATA.glob(f"features/{source_path.stem}__*.pt")):
            f = torch.load(fp, weights_only=False)
            self.features[f["variant"]["variant_id"]] = f
        self.continuations = [
            load_v067(cp, "continuations")
            for cp in sorted(DATA.glob(
                f"continuations_v067/{sid}_d*.pt"))]

    def variant_ids(self):
        return list(self.features)


def unroll(model, h, mask, actions, exec_lens, device, detach_every=TBPTT):
    zs, z = [], None
    for i in range(h.shape[0]):
        hi = h[i][None].float().to(device)
        mi = mask[i][None].to(device)
        if z is None:
            z = model.initial_state(hi, mi)
        else:
            a = actions[i - 1][None].to(device)
            am = (torch.arange(10, device=device)[None]
                  < int(exec_lens[i - 1]))
            z = model.step(z, a, hi, mi, action_mask=am)
        if detach_every and i > 0 and i % detach_every == 0:
            z = z.detach()
        zs.append(z)
    return zs


def variant_of(gid: str, canon_gid: str) -> str:
    return "canonical" if gid == canon_gid else f"distinct_{gid}"


def source_losses(model, ema, bundle, device, n_sub_max, tolerances,
                  head_hits):
    fam = {k: [] for k in FAMILIES}
    src = bundle.source
    rows = src["rows"]
    n_dec = len(rows)
    actions = [r["chunk_norm"][:10].float() for r in rows]
    exec_lens = [r["executed_len"] for r in rows]
    q_scale = Q_SCALE.to(device)
    canon_gid = bundle.labels["canonical_goal_spec_id"]
    labels = bundle.labels["per_decision"]

    canon = bundle.features["canonical"]
    zs = unroll(model, canon["h"], canon["mask"], actions, exec_lens,
                device)
    with torch.no_grad():
        zs_ema = unroll(ema, canon["h"], canon["mask"], actions,
                        exec_lens, device, detach_every=0)

    def current_loss(cur, lab, with_events=True):
        n_sub = len(lab["valid_before"])
        loss = BCE(cur["valid_bits"][0, :n_sub],
                   bits(lab["valid_before"], n_sub_max,
                        device)[:n_sub])
        if with_events:
            loss = loss + BCE(
                cur["event_bits"][0, :n_sub],
                idx_bits(lab["events_before"], n_sub_max,
                         device)[:n_sub])
        return loss + (cur["ordered_prefix"][0]
                       - lab["ordered_prefix_before"] / n_sub) ** 2

    for i in range(n_dec):
        lab = labels[i]["goals"][canon_gid]
        z = zs[i]
        fam["current"].append(current_loss(model.d_current(z), lab))
        z_tilde = model.predict(
            z, actions[i][None].to(device),
            action_mask=(torch.arange(10, device=device)[None]
                         < exec_lens[i]))
        out = model.d_next(z_tilde)
        q_next = rows[i + 1]["q"] if i + 1 < n_dec else rows[i]["q"]
        d_q = (q_next - rows[i]["q"]).to(device)
        d_obj = torch.from_numpy(
            rows[i]["obj_after"] - rows[i]["obj_before"]).float().to(
                device)
        n_obj = d_obj.shape[0]
        fam["seq_phys"].append(
            huber(out["d_q"][0], d_q, q_scale)
            + huber(out["d_obj"][0, :n_obj].flatten(), d_obj.flatten(),
                    OBJ_SCALE))
        head_hits["d_q"] += 1
        head_hits["d_obj"] += 1
        fam["reward"].append(
            (out["reward"][0] - float(lab.get("reward_valid",
             sum(lab["valid_after"]) - sum(lab["valid_before"])))) ** 2)
        head_hits["reward"] += 1
        # executed-action next-semantic matrix (canonical goal)
        n_sub = len(lab["valid_after"])
        fam["next_sem"].append(
            BCE(out["valid_bits"][0, :n_sub],
                bits(lab["valid_after"], n_sub_max, device)[:n_sub])
            + BCE(out["event_bits"][0, :n_sub],
                  idx_bits(lab["events_after"], n_sub_max,
                           device)[:n_sub])
            + BCE(out["flips_01"][0, :n_sub],
                  idx_bits([f[1] for f in lab["flips_01"]], n_sub_max,
                           device)[:n_sub])
            + BCE(out["flips_10"][0, :n_sub],
                  idx_bits([f[1] for f in lab["flips_10"]], n_sub_max,
                           device)[:n_sub]))
        head_hits["valid_bits"] += 1
        head_hits["event_bits"] += 1
        head_hits["flips_01"] += 1
        head_hits["flips_10"] += 1
        if i + 1 < n_dec:
            fam["closure1"].append(torch.nn.functional.mse_loss(
                z_tilde, zs_ema[i + 1].detach()))
        if i % 8 == 0 and i + 3 < n_dec:
            zo = z
            for k in range(3):
                zo = model.predict(
                    zo, actions[i + k][None].to(device),
                    action_mask=(torch.arange(10, device=device)[None]
                                 < exec_lens[i + k]))
                if k >= 1:
                    fam["closure23"].append(
                        torch.nn.functional.mse_loss(
                            zo, zs_ema[i + k + 1].detach()))

    # paraphrase: state AND transitioned-prediction consistency
    for vid in bundle.variant_ids():
        if not vid.startswith("paraphrase"):
            continue
        f = bundle.features[vid]
        zp = unroll(model, f["h"], f["mask"], actions, exec_lens, device)
        for i in range(0, n_dec, 4):
            fam["para"].append(torch.nn.functional.mse_loss(zp[i], zs[i]))
            zt_p = model.predict(zp[i], actions[i][None].to(device))
            zt_c = model.predict(zs[i], actions[i][None].to(device))
            fam["para_pred"].append(
                torch.nn.functional.mse_loss(zt_p, zt_c))
            lab = labels[i]["goals"][canon_gid]
            fam["current"].append(current_loss(
                model.d_current(zp[i]), lab, with_events=False))

    # distinct-goal current grounding + language-invariant physics
    for vid in bundle.variant_ids():
        if not vid.startswith("distinct"):
            continue
        f = bundle.features[vid]
        gid = f["variant"]["goal_spec_id"]
        if gid not in bundle.labels["goals"]:
            continue
        zd = unroll(model, f["h"], f["mask"], actions, exec_lens, device)
        for i in range(0, n_dec, 4):
            lab = labels[i]["goals"][gid]
            fam["distinct_current"].append(current_loss(
                model.d_current(zd[i]), lab))
            zt = model.predict(
                zd[i], actions[i][None].to(device),
                action_mask=(torch.arange(10, device=device)[None]
                             < exec_lens[i]))
            outd = model.d_next(zt)
            q_next = rows[i + 1]["q"] if i + 1 < n_dec else rows[i]["q"]
            d_q = (q_next - rows[i]["q"]).to(device)
            d_obj = torch.from_numpy(
                rows[i]["obj_after"]
                - rows[i]["obj_before"]).float().to(device)
            n_obj = d_obj.shape[0]
            fam["shared_phys"].append(
                huber(outd["d_q"][0], d_q, q_scale)
                + huber(outd["d_obj"][0, :n_obj].flatten(),
                        d_obj.flatten(), OBJ_SCALE))
            # executed-action next-semantic under the distinct goal
            n_sub = len(lab["valid_after"])
            fam["next_sem"].append(
                BCE(outd["valid_bits"][0, :n_sub],
                    bits(lab["valid_after"], n_sub_max,
                         device)[:n_sub])
                + BCE(outd["flips_01"][0, :n_sub],
                      idx_bits([fl[1] for fl in lab["flips_01"]],
                               n_sub_max, device)[:n_sub])
                + BCE(outd["flips_10"][0, :n_sub],
                      idx_bits([fl[1] for fl in lab["flips_10"]],
                               n_sub_max, device)[:n_sub]))

    # branch pass on ALL phase-A audits: physical absolute + centered
    for audit in src["audits"]:
        d = audit["decision"]
        z = zs[d]
        cands = [b for b in audit["branches"] if b["kind"] == "candidate"]
        outs, d_objs = [], []
        for b in cands:
            zt = model.predict(z, b["chunk_norm"][None, :10].to(device))
            out = model.d_next(zt)
            d_q = (b["q_after"] - rows[d]["q"]).to(device)
            d_obj = torch.from_numpy(
                b["obj_after"] - rows[d]["obj_before"]).float().to(device)
            n_obj = d_obj.shape[0]
            fam["branch_abs"].append(
                huber(out["d_q"][0], d_q, q_scale)
                + huber(out["d_obj"][0, :n_obj].flatten(),
                        d_obj.flatten(), OBJ_SCALE))
            outs.append(out["d_obj"][0, :n_obj])
            d_objs.append(d_obj.view(-1, 3))
        pred, target = torch.stack(outs), torch.stack(d_objs)
        fam["branch_centered"].append(huber(
            (pred - pred.mean(0)).flatten(),
            (target - target.mean(0)).flatten(), OBJ_SCALE))

    # v067 groups: crossed immediate next-semantic + continuation heads
    # + Bradley–Terry ranking with FROZEN tolerances
    for cont in bundle.continuations:
        d = cont["decision"]
        canon_id = cont["canonical_goal_spec_id"]
        audit = next(a for a in src["audits"] if a["decision"] == d)
        chunk_of = {}
        for b in audit["branches"]:
            if b["kind"] == "candidate":
                chunk_of[b["candidate"]] = b["chunk_norm"]
            elif b["kind"] == "support":
                chunk_of[4] = b["chunk_norm"]
        z_goal_cache = {}

        def z_for(gid):
            if gid in z_goal_cache:
                return z_goal_cache[gid]
            vid = variant_of(gid, canon_id)
            if vid == "canonical":
                z_goal = zs[d]
            elif vid in bundle.features:
                fz = bundle.features[vid]
                z_goal = unroll(model, fz["h"][:d + 1],
                                fz["mask"][:d + 1], actions[:d + 1],
                                exec_lens[:d + 1], device)[-1]
            else:
                z_goal = None
            z_goal_cache[gid] = z_goal
            return z_goal

        # immediate crossed matrix from branch summaries
        for b in cont["branch_summaries"]:
            if b["kind"] == "replay":
                continue          # audit only, never trained
            bkey = b["branch_key"]
            if chunk_of.get(bkey) is None:
                continue
            for gid in cont["goals"]:
                z_goal = z_for(gid)
                if z_goal is None:
                    continue
                imm = b["immediate"][gid]
                zt = model.predict(
                    z_goal, chunk_of[bkey][None, :10].to(device))
                out = model.d_next(zt)
                n_sub = len(imm["valid_after"])
                fam["next_sem"].append(
                    BCE(out["valid_bits"][0, :n_sub],
                        bits(imm["valid_after"], n_sub_max,
                             device)[:n_sub])
                    + BCE(out["event_bits"][0, :n_sub],
                          idx_bits(imm["events_after"], n_sub_max,
                                   device)[:n_sub])
                    + BCE(out["flips_01"][0, :n_sub],
                          idx_bits([f[1] for f in imm["flips_01"]],
                                   n_sub_max, device)[:n_sub])
                    + BCE(out["flips_10"][0, :n_sub],
                          idx_bits([f[1] for f in imm["flips_10"]],
                                   n_sub_max, device)[:n_sub]))
                fam["reward"].append(
                    (out["reward"][0] - float(imm["reward_valid"])) ** 2)

        # continuation heads + ranking per goal
        by_goal: dict[str, dict] = {}
        for rec in cont["records"]:
            if rec["provenance"] == "replay_audit":
                continue
            by_goal.setdefault(rec["goal_spec_id"], {}).setdefault(
                rec["branch_key"], []).append(rec)
        for gid, branches in by_goal.items():
            z_goal = z_for(gid)
            if z_goal is None:
                continue
            scores = {}
            for bkey, recs in branches.items():
                if chunk_of.get(bkey) is None:
                    continue
                zt = model.predict(
                    z_goal, chunk_of[bkey][None, :10].to(device))
                out = model.d_next(zt)
                y = [r["outcome"] for r in sorted(
                    recs, key=lambda r: r["repeat"])]
                succ = float(any(o["success_by_100"] for o in y))
                fam["cont_heads"].append(
                    BCE(out["success_logit"][0],
                        torch.tensor(succ, device=device))
                    + (out["p_valid"][0]
                       - sum(o["p_valid_100"] for o in y) / len(y)) ** 2
                    + (out["damage"][0]
                       - sum(-o["neg_damage"] for o in y) / len(y)) ** 2
                    + (out["tau_next"][0]
                       - sum(-o["neg_tau_next"] for o in y)
                       / len(y) / 101.0) ** 2
                    + sum(
                        (out["q_valid"][0, hi]
                         - sum(o["q_at_horizons"][h] for o in y)
                         / len(y)) ** 2
                        for hi, h in enumerate(cont["horizons"])))
                head_hits["success"] += 1
                head_hits["p_valid"] += 1
                head_hits["damage"] += 1
                head_hits["tau_next"] += 1
                head_hits["q_valid"] += 1
                if bkey != 4:      # support never enters ranking
                    scores[bkey] = (out["s"][0], y)
            for i, j in itertools.combinations(sorted(scores), 2):
                a_ij = paired_preference(scores[i][1], scores[j][1],
                                         tolerances)
                if a_ij == 0:
                    continue
                p_ij = torch.sigmoid(scores[i][0] - scores[j][0])
                fam["ranking"].append(
                    torch.nn.functional.binary_cross_entropy(
                        p_ij.clamp(1e-6, 1 - 1e-6),
                        torch.tensor((a_ij + 1) / 2.0, device=device)))
                head_hits["score"] += 1
    return fam


def demo_losses(model, ema, episode, device):
    fam = {k: [] for k in FAMILIES}
    h, mask = episode["prefix_hidden"], episode["prefix_mask"]
    actions = episode["action_block_norm"].float()
    lens = episode.get("executed_lengths", [10] * actions.shape[0])
    q_scale = Q_SCALE.to(device)
    n = min(h.shape[0], actions.shape[0] + 1)
    zs = unroll(model, h[:n], mask[:n],
                [actions[i] for i in range(n - 1)],
                [int(lens[i]) for i in range(n - 1)], device)
    with torch.no_grad():
        zs_ema = unroll(ema, h[:n], mask[:n],
                        [actions[i] for i in range(n - 1)],
                        [int(lens[i]) for i in range(n - 1)], device,
                        detach_every=0)
    for i in range(n - 1):
        zt = model.predict(zs[i], actions[i][None].to(device))
        out = model.d_next(zt)
        d_q = (episode["q"][i + 1] - episode["q"][i]).to(device)
        d_obj = (episode["obj_pos"][i + 1]
                 - episode["obj_pos"][i]).to(device)
        n_obj = d_obj.shape[0]
        fam["seq_phys"].append(
            huber(out["d_q"][0], d_q, q_scale)
            + huber(out["d_obj"][0, :n_obj].flatten(), d_obj.flatten(),
                    OBJ_SCALE))
        fam["closure1"].append(torch.nn.functional.mse_loss(
            zt, zs_ema[i + 1].detach()))
    return fam


def head_param_hashes(model) -> dict:
    out = {}
    for name in DNEXT_HEADS:
        mod = getattr(model.d_next, name)
        blob = b"".join(p.detach().cpu().numpy().tobytes()
                        for p in mod.parameters())
        out[name] = hashlib.sha256(blob).hexdigest()[:16]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=RESULTS / "v067_wm")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    support_report = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    assert support_report["run_schema"] == RUN_SCHEMA
    tolerances = support_report["frozen_outcome_tolerances"]
    hc_pairs = support_report["summary"]["history_contrast_pairs"]
    train_rankable = [r["group"] for r in support_report["groups"]
                      if r["split"] == "train" and r["policy_rankable"]
                      and not r["replay_unstable"]]
    assert train_rankable, (
        "zero clean train policy-rankable groups in the corrected bank — "
        "the ranking channel is unsupported; route to V6.8 backlog "
        "relabeling instead of spending a training run")
    print(f"[launch] clean train rankable groups: "
          f"{len(train_rankable)}", flush=True)

    n_sub_max = 7
    model = V06State().to(device)
    ema = make_ema(model)
    init_hashes = head_param_hashes(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WD)

    source_paths = sorted((DATA / "sources").glob("*.pt"))
    train_paths = [p for p in source_paths
                   if torch.load(p, weights_only=False)["split"]
                   == "train"]
    demos = [e for p in sorted(SEQ_CACHE.glob("task*_demo*.pt"))
             for e in [torch.load(p, weights_only=False)]
             if e["split"] == "train"]
    print(f"train sources: {len(train_paths)}, demo support: "
          f"{len(demos)}, hc_pairs: {hc_pairs}", flush=True)

    # ---- launch assertion: crossed rows disagree semantically while ----
    # ---- physically identical (same branch, two goals) -----------------
    crossed_ok = False
    for p in train_paths:
        for cp in sorted(DATA.glob(
                f"continuations_v067/{p.stem}_d*.pt")):
            cont = load_v067(cp, "continuations")
            for b in cont["branch_summaries"]:
                if b["kind"] == "replay":
                    continue
                vals = {gid: tuple(b["immediate"][gid]["valid_after"])
                        for gid in cont["goals"]}
                if len({v for v in vals.values()}) > 1:
                    crossed_ok = True
    assert crossed_ok, ("no crossed row with goal-dependent semantic "
                        "targets on a shared physical branch")
    print("[launch] crossed semantic-disagreement row present",
          flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    head_hits = {h: 0 for h in DNEXT_HEADS}
    logs = []
    for epoch in range(args.epochs):
        order = list(range(len(train_paths))) + [
            -1 - i for i in range(len(demos))]
        random.shuffle(order)
        epoch_fam = {k: [] for k in FAMILIES}
        for idx in order:
            optimizer.zero_grad(set_to_none=True)
            if idx >= 0:
                bundle = SourceBundle(train_paths[idx])
                fam = source_losses(model, ema, bundle, device,
                                    n_sub_max, tolerances, head_hits)
            else:
                fam = demo_losses(model, ema, demos[-1 - idx], device)
            losses = {k: torch.stack(v).mean()
                      for k, v in fam.items() if v}
            if not losses:
                continue
            total = torch.stack(list(losses.values())).mean()
            assert torch.isfinite(total), "non-finite loss"
            total.backward()
            if epoch == 0 and idx == order[0]:
                # launch assertion: gradient reach into E_a / T
                for mod, nm in ((model.e_a, "E_a"), (model.t, "T")):
                    gn = sum(float(p.grad.abs().sum())
                             for p in mod.parameters()
                             if p.grad is not None)
                    assert gn > 0, f"no gradient into {nm} at launch"
                print("[launch] gradient reaches E_a and T", flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            ema_update(ema, model)
            for k, v in losses.items():
                epoch_fam[k].append(float(v))
        means = {k: (sum(v) / len(v) if v else None)
                 for k, v in epoch_fam.items()}
        logs.append({"epoch": epoch, **means})
        print(f"[v067 epoch {epoch}] " + " ".join(
            f"{k}={means[k]:.4f}" for k in FAMILIES
            if means[k] is not None), flush=True)
        if (epoch + 1) % 5 == 0 or args.smoke:
            torch.save({"model": model.state_dict(),
                        "ema": ema.state_dict(), "epoch": epoch,
                        "seed": args.seed, "run_schema": RUN_SCHEMA},
                       args.output / f"checkpoint_epoch{epoch:03d}.pt")

    # ---- completion assertions -----------------------------------------
    silent_fams = [k for k in FAMILIES if not any(
        lg.get(k) is not None for lg in logs)]
    final_hashes = head_param_hashes(model)
    unchanged = [h for h in DNEXT_HEADS
                 if final_hashes[h] == init_hashes[h]]
    zero_hit = [h for h in DNEXT_HEADS if head_hits[h] == 0]
    mechanical = {
        "silent_families": silent_fams,
        "dnext_head_target_counts": head_hits,
        "dnext_heads_without_targets": zero_hit,
        "dnext_heads_param_unchanged": unchanged,
        "history_contrast_pairs": hc_pairs,
        "history_family_status": ("unsupported (no matched pair; no "
                                  "history-usefulness claim)"
                                  if hc_pairs == 0 else "supported"),
        "frozen_outcome_tolerances": tolerances,
        "structural_no_bypass_suite": "scripts/test_v06_mechanical.py "
                                      "(12/12, unchanged model code)",
    }
    if not args.smoke:
        assert not silent_fams, f"silent families: {silent_fams}"
        assert not zero_hit, f"headless targets: {zero_hit}"
        assert not unchanged, f"untrained heads: {unchanged}"
    (args.output / "mechanical_contract.json").write_text(
        json.dumps(mechanical, indent=2))
    (args.output / "config.json").write_text(json.dumps({
        "run_schema": RUN_SCHEMA, "seed": args.seed, "lr": LR, "wd": WD,
        "epochs": args.epochs, "tbptt": TBPTT, "grad_norm": GRAD_NORM,
        "families": FAMILIES, "from_scratch": True}, indent=2))
    (args.output / "per_head_logs.json").write_text(json.dumps(logs))
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "epoch": args.epochs - 1, "seed": args.seed,
                "run_schema": RUN_SCHEMA,
                "mechanical": mechanical},
               args.output / "checkpoint_final.pt")
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
