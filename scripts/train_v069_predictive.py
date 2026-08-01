#!/usr/bin/env python
"""V6.9.1 — ONE supervision-complete predictive LCWM training job.

Gate split (V6.9.0): predictive training requires valid lineage,
source-disjoint splits, enabled targets, and gradient reach into
E_a/T/U — it does NOT require a clean ranking pair. Sibling ranking is
enabled only when eligible clean pairs exist; otherwise its loss weight
is zero, the score head is explicitly DISABLED (parameter-unchanged
allowed), and the manifest records `ranking: unsupported`. Candidate
ranking cannot veto learning the predictive state.

Data (all-135 branch loader, deduplicated per v069_branch_index):
- ordinary source trajectories: recurrence, nominal dynamics, closure;
- ALL 135 audited branch groups (60 continued + 75 immediate-only
  backlog): immediate action-conditioned physical + crossed GoalSpec
  semantics through the transitioned latent;
- the 60 corrected continuation groups: goal-conditioned success,
  damage, progress, τ_next, multi-horizon value (replay-audit rows are
  never alternatives; replay-unstable groups never supervise ranking);
- canonical/paraphrase features: state + transitioned-prediction
  consistency.

Budget frozen (v0.6 registered): seed 0, AdamW 3e-4 / wd 1e-4,
25 epochs, TBPTT 16, EMA 0.995, clip 1.0, one optimizer step per
source visit, mean over present families. No sweep.

Checkpoint selection: frozen source-held-out predictive objective
(mean of enabled-family losses on the 5 dev sources at every saved
epoch; min wins; tie → earliest). Never public LoHo behavior.

Output: results/libero_loho_public_v1/v069_predictive/
  config.json  train_manifest.json  mechanical_contract.json
  checkpoint_epochXXX.pt  checkpoint_selected.pt
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
from lcwm.v067_lineage import load_v067  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
SEQ_CACHE = Path("/home/stargazer/Desktop/vla_wm/datasets"
                 "/seq_prefix_cache_v1")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
RUN_SCHEMA_OUT = "v069"
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


def variant_of(gid: str, canon_gid: str) -> str:
    return "canonical" if gid == canon_gid else f"distinct_{gid}"


class SourceBundle:
    """One source: rows, semantic relabels, features per variant,
    continued groups, and immediate-only backlog groups (deduplicated:
    backlog groups shadowed by a continuation are dropped here)."""

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
        continued_decs = {c["decision"] for c in self.continuations}
        bl_path = DATA / "backlog_relabels_v067" / source_path.name
        self.backlog_groups = []
        if bl_path.exists():
            bl = load_v067(bl_path, "semantic_labels")
            self.backlog_groups = [
                g for g in bl["groups"]
                if g["decision"] not in continued_decs]

    def variant_ids(self):
        return list(self.features)


def unroll(model, h, mask, actions, exec_lens, device,
           detach_every=TBPTT):
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


def next_sem_loss(model, z_goal, chunk, imm, n_sub_max, device,
                  head_hits):
    zt = model.predict(z_goal, chunk[None, :10].float().to(device))
    out = model.d_next(zt)
    n_sub = len(imm["valid_after"])
    loss = (BCE(out["valid_bits"][0, :n_sub],
                bits(imm["valid_after"], n_sub_max, device)[:n_sub])
            + BCE(out["event_bits"][0, :n_sub],
                  idx_bits(imm["events_after"], n_sub_max,
                           device)[:n_sub])
            + BCE(out["flips_01"][0, :n_sub],
                  idx_bits([f[1] for f in imm["flips_01"]], n_sub_max,
                           device)[:n_sub])
            + BCE(out["flips_10"][0, :n_sub],
                  idx_bits([f[1] for f in imm["flips_10"]], n_sub_max,
                           device)[:n_sub]))
    reward = (out["reward"][0] - float(imm["reward_valid"])) ** 2
    for h_ in ("valid_bits", "event_bits", "flips_01", "flips_10",
               "reward"):
        head_hits[h_] += 1
    return loss, reward


def source_losses(model, ema, bundle, device, n_sub_max, tolerances,
                  ranking_enabled, clean_rank_groups, head_hits,
                  stats):
    fam = {k: [] for k in FAMILIES}
    src = bundle.source
    rows = src["rows"]
    n_dec = len(rows)
    actions = [r["chunk_norm"][:10].float() for r in rows]
    exec_lens = [r["executed_len"] for r in rows]
    q_scale = Q_SCALE.to(device)
    canon_gid = bundle.labels["canonical_goal_spec_id"]
    labels = bundle.labels["per_decision"]
    audit_by_d = {a["decision"]: a for a in src["audits"]}

    def chunks_of(d):
        out = {}
        for b in audit_by_d[d]["branches"]:
            if b["kind"] == "candidate":
                out[b["candidate"]] = b["chunk_norm"]
            elif b["kind"] == "support":
                out[4] = b["chunk_norm"]
        return out

    # one unroll per language variant per source visit (reused by every
    # family below — states always from episode reset)
    canon = bundle.features["canonical"]
    zs = unroll(model, canon["h"], canon["mask"], actions, exec_lens,
                device)
    with torch.no_grad():
        zs_ema = unroll(ema, canon["h"], canon["mask"], actions,
                        exec_lens, device, detach_every=0)
    zs_distinct = {}
    for vid in bundle.variant_ids():
        if vid.startswith("distinct"):
            f = bundle.features[vid]
            zs_distinct[f["variant"]["goal_spec_id"]] = unroll(
                model, f["h"], f["mask"], actions, exec_lens, device)

    def z_goal_at(gid, d):
        if gid == canon_gid:
            return zs[d]
        return zs_distinct.get(gid, [None] * n_dec)[d] \
            if gid in zs_distinct else None

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
        # executed-row next-semantic matrix (canonical)
        ns, rew = next_sem_loss(
            model, z, actions[i],
            {"valid_after": lab["valid_after"],
             "events_after": lab["events_after"],
             "flips_01": lab["flips_01"],
             "flips_10": lab["flips_10"],
             "reward_valid": lab["reward_valid"]
             if "reward_valid" in lab else
             sum(lab["valid_after"]) - sum(lab["valid_before"])},
            n_sub_max, device, head_hits)
        fam["next_sem"].append(ns)
        fam["reward"].append(rew)
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
        zp = unroll(model, f["h"], f["mask"], actions, exec_lens,
                    device)
        for i in range(0, n_dec, 4):
            fam["para"].append(
                torch.nn.functional.mse_loss(zp[i], zs[i]))
            zt_p = model.predict(zp[i], actions[i][None].to(device))
            zt_c = model.predict(zs[i], actions[i][None].to(device))
            fam["para_pred"].append(
                torch.nn.functional.mse_loss(zt_p, zt_c))
            lab = labels[i]["goals"][canon_gid]
            fam["current"].append(current_loss(
                model.d_current(zp[i]), lab, with_events=False))

    # distinct-goal current grounding + language-invariant physics
    for gid, zd in zs_distinct.items():
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

    # branch physical (ALL phase-A audits): absolute + centered
    for audit in src["audits"]:
        d = audit["decision"]
        z = zs[d]
        cands = [b for b in audit["branches"]
                 if b["kind"] == "candidate"]
        outs, d_objs = [], []
        for b in cands:
            zt = model.predict(z, b["chunk_norm"][None, :10].to(device))
            out = model.d_next(zt)
            d_q = (b["q_after"] - rows[d]["q"]).to(device)
            d_obj = torch.from_numpy(
                b["obj_after"] - rows[d]["obj_before"]).float().to(
                    device)
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

    # immediate crossed next-semantics — ALL 135 groups
    def crossed_pass(decision, branch_summaries):
        chunk_of = chunks_of(decision)
        for b in branch_summaries:
            if b["kind"] == "replay":
                continue           # audit only, never an alternative
            bkey = 4 if b["kind"] == "support" else b["candidate"]
            chunk = chunk_of.get(bkey)
            if chunk is None:
                continue
            sem_sets = set()
            for gid, imm in b["immediate"].items():
                z_goal = z_goal_at(gid, decision)
                if z_goal is None:
                    continue
                ns, rew = next_sem_loss(model, z_goal, chunk, imm,
                                        n_sub_max, device, head_hits)
                fam["next_sem"].append(ns)
                fam["reward"].append(rew)
                sem_sets.add((tuple(imm["valid_after"]),
                              tuple(imm["events_after"])))
            stats["crossed_branches"] += 1
            if len(sem_sets) > 1:
                stats["crossed_semantic_disagreements"] += 1

    for cont in bundle.continuations:
        crossed_pass(cont["decision"], cont["branch_summaries"])
    for g in bundle.backlog_groups:
        crossed_pass(g["decision"], g["branch_summaries"])

    # continuation value heads + (gated) ranking — 60 continued groups
    for cont in bundle.continuations:
        d = cont["decision"]
        canon_id = cont["canonical_goal_spec_id"]
        chunk_of = chunks_of(d)
        by_goal: dict[str, dict] = {}
        for rec in cont["records"]:
            if rec["provenance"] == "replay_audit":
                continue
            by_goal.setdefault(rec["goal_spec_id"], {}).setdefault(
                rec["branch_key"], []).append(rec)
        gid_key = f"{src['source_id']}_d{d}"
        for gid, branches in by_goal.items():
            z_goal = z_goal_at(gid, d)
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
                       - sum(o["p_valid_100"] for o in y)
                       / len(y)) ** 2
                    + (out["damage"][0]
                       - sum(-o["neg_damage"] for o in y)
                       / len(y)) ** 2
                    + (out["tau_next"][0]
                       - sum(-o["neg_tau_next"] for o in y)
                       / len(y) / 101.0) ** 2
                    + sum(
                        (out["q_valid"][0, hi]
                         - sum(o["q_at_horizons"][h] for o in y)
                         / len(y)) ** 2
                        for hi, h in enumerate(cont["horizons"])))
                for h_ in ("success", "p_valid", "damage", "tau_next",
                           "q_valid"):
                    head_hits[h_] += 1
                if bkey != 4:
                    scores[bkey] = (out["s"][0], y)
            if ranking_enabled and gid == canon_id \
                    and gid_key in clean_rank_groups:
                for i, j in itertools.combinations(sorted(scores), 2):
                    a_ij = paired_preference(
                        scores[i][1], scores[j][1], tolerances)
                    if a_ij == 0:
                        continue
                    p_ij = torch.sigmoid(
                        scores[i][0] - scores[j][0])
                    fam["ranking"].append(
                        torch.nn.functional.binary_cross_entropy(
                            p_ij.clamp(1e-6, 1 - 1e-6),
                            torch.tensor((a_ij + 1) / 2.0,
                                         device=device)))
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
            + huber(out["d_obj"][0, :n_obj].flatten(),
                    d_obj.flatten(), OBJ_SCALE))
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
                        default=RESULTS / "v069_predictive")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    lineage = json.loads((RESULTS / "v069_lineage.json").read_text())
    index = json.loads((RESULTS / "v069_branch_index.json").read_text())
    assert index["n_audited_groups"] == 135
    support = json.loads(
        (RESULTS / "v067_support_report.json").read_text())
    tolerances = support["frozen_outcome_tolerances"]
    clean_rank_groups = {
        r["group"] for r in support["groups"]
        if r["split"] == "train" and r["policy_rankable"]
        and not r["replay_unstable"]}
    ranking_enabled = bool(clean_rank_groups)
    rank_msg = (f"ENABLED ({len(clean_rank_groups)} groups)"
                if ranking_enabled
                else "unsupported -> weight 0, score head disabled")
    print(f"[gates] predictive=ENABLED  ranking={rank_msg}", flush=True)

    n_sub_max = 7
    model = V06State().to(device)
    ema = make_ema(model)
    init_hashes = head_param_hashes(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WD)

    source_paths = sorted((DATA / "sources").glob("*.pt"))
    split_of = {p: torch.load(p, weights_only=False)["split"]
                for p in source_paths}
    train_paths = [p for p in source_paths if split_of[p] == "train"]
    dev_paths = [p for p in source_paths if split_of[p] == "dev"]
    demos = [e for p in sorted(SEQ_CACHE.glob("task*_demo*.pt"))
             for e in [torch.load(p, weights_only=False)]
             if e["split"] == "train"]
    print(f"train sources: {len(train_paths)}, dev sources: "
          f"{len(dev_paths)}, demo support: {len(demos)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    head_hits = {h: 0 for h in DNEXT_HEADS}
    stats = {"crossed_branches": 0,
             "crossed_semantic_disagreements": 0}

    def eval_dev():
        with torch.no_grad():
            dev_fam = {k: [] for k in FAMILIES}
            dstats = {"crossed_branches": 0,
                      "crossed_semantic_disagreements": 0}
            dhits = {h: 0 for h in DNEXT_HEADS}
            for p in dev_paths:
                fam = source_losses(
                    model, ema, SourceBundle(p), device, n_sub_max,
                    tolerances, False, set(), dhits, dstats)
                for k, v in fam.items():
                    dev_fam[k].extend(float(x) for x in v)
            enabled = [k for k in FAMILIES
                       if k != "ranking" and dev_fam[k]]
            return (sum(sum(dev_fam[k]) / len(dev_fam[k])
                        for k in enabled) / len(enabled),
                    {k: (sum(v) / len(v) if v else None)
                     for k, v in dev_fam.items()})

    logs, ckpt_devs = [], {}
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
                                    n_sub_max, tolerances,
                                    ranking_enabled,
                                    clean_rank_groups, head_hits,
                                    stats)
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
                for mod, nm in ((model.e_a, "E_a"), (model.t, "T"),
                                (model.anchor, "U/anchor")):
                    gn = sum(float(p.grad.abs().sum())
                             for p in mod.parameters()
                             if p.grad is not None)
                    assert gn > 0, f"no gradient into {nm} at launch"
                print("[launch] gradient reaches E_a, T, U(anchor)",
                      flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           GRAD_NORM)
            optimizer.step()
            ema_update(ema, model)
            for k, v in losses.items():
                epoch_fam[k].append(float(v))
        means = {k: (sum(v) / len(v) if v else None)
                 for k, v in epoch_fam.items()}
        logs.append({"epoch": epoch, **means})
        print(f"[v069 epoch {epoch}] " + " ".join(
            f"{k}={means[k]:.4f}" for k in FAMILIES
            if means[k] is not None), flush=True)
        if (epoch + 1) % 5 == 0 or args.smoke or \
                (epoch + 1) == args.epochs:
            dev_obj, dev_detail = eval_dev()
            ckpt = args.output / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({"model": model.state_dict(),
                        "ema": ema.state_dict(), "epoch": epoch,
                        "seed": args.seed,
                        "run_schema": RUN_SCHEMA_OUT,
                        "dev_objective": dev_obj}, ckpt)
            ckpt_devs[ckpt.name] = dev_obj
            logs[-1]["dev_objective"] = dev_obj
            print(f"  [dev] epoch {epoch}: objective {dev_obj:.4f}",
                  flush=True)

    # ---- completion contract -------------------------------------------
    enabled_heads = [h for h in DNEXT_HEADS
                     if not (h == "score" and not ranking_enabled)]
    disabled_heads = [] if ranking_enabled else ["score"]
    final_hashes = head_param_hashes(model)
    unchanged = [h for h in enabled_heads
                 if final_hashes[h] == init_hashes[h]]
    zero_hit = [h for h in enabled_heads if head_hits[h] == 0]
    silent = [k for k in FAMILIES if k != "ranking" and not any(
        lg.get(k) is not None for lg in logs)]
    mechanical = {
        "gates": {"predictive": "enabled",
                  "ranking": ("enabled" if ranking_enabled
                              else "unsupported")},
        "clean_rank_groups": sorted(clean_rank_groups),
        "silent_families": silent,
        "dnext_head_target_counts": head_hits,
        "enabled_heads_without_targets": zero_hit,
        "enabled_heads_param_unchanged": unchanged,
        "explicitly_disabled_heads": disabled_heads,
        "crossed_branches_trained": stats["crossed_branches"],
        "crossed_semantic_disagreement_branches":
            stats["crossed_semantic_disagreements"],
        "replay_audit_rows_trained": 0,
        "history_contrast": "unsupported (0 matched pairs)",
    }
    if not args.smoke:
        assert not silent, f"silent families: {silent}"
        assert not zero_hit, f"headless targets: {zero_hit}"
        assert not unchanged, f"untrained enabled heads: {unchanged}"
    best_name = min(ckpt_devs, key=lambda k: (ckpt_devs[k], k))
    selected = torch.load(args.output / best_name, weights_only=False)
    torch.save(selected, args.output / "checkpoint_selected.pt")
    (args.output / "mechanical_contract.json").write_text(
        json.dumps(mechanical, indent=2))
    (args.output / "config.json").write_text(json.dumps({
        "run_schema": RUN_SCHEMA_OUT, "seed": args.seed, "lr": LR,
        "wd": WD, "epochs": args.epochs, "tbptt": TBPTT,
        "grad_norm": GRAD_NORM, "families": FAMILIES,
        "from_scratch": True,
        "selection": {"rule": "min dev objective (mean of enabled "
                              "non-ranking family losses on 5 dev "
                              "sources); tie -> earliest",
                      "selected": best_name,
                      "per_checkpoint": ckpt_devs}}, indent=2))
    (args.output / "train_manifest.json").write_text(json.dumps({
        "run_schema": RUN_SCHEMA_OUT,
        "lineage_sha256": hashlib.sha256(json.dumps(
            lineage, sort_keys=True).encode()).hexdigest(),
        "branch_index": {k: index[k] for k in (
            "n_audited_groups", "n_continued", "n_backlog",
            "n_branch_goal_rows")},
        "ranking": ("enabled" if ranking_enabled else "unsupported"),
        "frozen_outcome_tolerances": tolerances,
        "per_epoch_losses": logs}, indent=2))
    print(f"selected {best_name} "
          f"(dev {ckpt_devs[best_name]:.4f}) -> {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
