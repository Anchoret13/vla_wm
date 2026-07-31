#!/usr/bin/env python
"""V6.3 — one fixed v0.6 world-model training job (bindings in the
2026-07-30 v6 progress note; registered before any run).

π0.5 is never loaded here: h features come from the fp16 sidecars.
Loss families: seq_phys, current, reward, closure1, closure23, para,
distinct_current, shared_phys, branch_abs, branch_centered, cont_heads,
ranking. One optimizer step per source visit; mean over present families.
"""

from __future__ import annotations

import argparse
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

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
SEQ_CACHE = Path("/home/stargazer/Desktop/vla_wm/datasets"
                 "/seq_prefix_cache_v1")
GOAL_SPECS = json.loads(
    (REPO_ROOT / "results" / "libero_loho_public_v1"
     / "goal_spec_manifest.json").read_text())
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4 + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUBER = 4.0
TBPTT = 16
LR, WD, EPOCHS, GRAD_NORM = 3e-4, 1e-4, 25, 1.0
FAMILIES = ("seq_phys", "current", "reward", "closure1", "closure23",
            "para", "distinct_current", "shared_phys", "branch_abs",
            "branch_centered", "cont_heads", "ranking")


def huber(p, t, scale):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=HUBER)


def bits_tensor(bits, n_max, device):
    out = torch.zeros(n_max, device=device)
    out[:len(bits)] = torch.tensor([float(b) for b in bits],
                                   device=device)
    return out


class SourceBundle:
    """Lazy per-source data: record, labels, features per variant,
    continuation records."""

    def __init__(self, source_path: Path):
        self.source = torch.load(source_path, weights_only=False)
        sid = self.source["source_id"]
        self.labels = torch.load(DATA / "labels" / source_path.name,
                                 weights_only=False)
        self.features = {}
        for fp in sorted(DATA.glob(f"features/{source_path.stem}__*.pt")):
            f = torch.load(fp, weights_only=False)
            self.features[f["variant"]["variant_id"]] = f
        self.continuations = [
            torch.load(cp, weights_only=False)
            for cp in sorted(DATA.glob(f"continuations/{sid}_d*.pt"))]

    def variant_ids(self):
        return list(self.features)


def unroll(model, h, mask, actions, exec_lens, device, detach_every=TBPTT):
    """z_t for every decision; TBPTT detach; returns list of z."""
    zs = []
    z = None
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


def source_losses(model, ema, bundle: SourceBundle, device,
                  n_sub_max: int):
    fam = {k: [] for k in FAMILIES}
    src = bundle.source
    rows = src["rows"]
    n_dec = len(rows)
    actions = [r["chunk_norm"][:10].float() for r in rows]
    exec_lens = [r["executed_len"] for r in rows]
    q_scale = Q_SCALE.to(device)
    canon_gid = bundle.labels["goals"][0]

    canon = bundle.features["canonical"]
    zs = unroll(model, canon["h"], canon["mask"], actions, exec_lens,
                device)
    with torch.no_grad():
        zs_ema = unroll(ema, canon["h"], canon["mask"], actions,
                        exec_lens, device, detach_every=0)

    for i in range(n_dec):
        lab = bundle.labels["per_decision"][i]["goals"][canon_gid]
        z = zs[i]
        cur = model.d_current(z)
        vb = bits_tensor(lab["valid_before"], n_sub_max, device)
        eb = torch.zeros(n_sub_max, device=device)
        for e in lab["events_before"]:
            eb[e] = 1.0
        n_sub = len(lab["valid_before"])
        fam["current"].append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                cur["valid_bits"][0, :n_sub], vb[:n_sub])
            + torch.nn.functional.binary_cross_entropy_with_logits(
                cur["event_bits"][0, :n_sub], eb[:n_sub])
            + (cur["ordered_prefix"][0]
               - lab["ordered_prefix_before"] / n_sub) ** 2)
        # executed-action physical + reward through D_next
        z_tilde = model.predict(
            z, actions[i][None].to(device),
            action_mask=(torch.arange(10, device=device)[None]
                         < exec_lens[i]))
        out = model.d_next(z_tilde)
        if i + 1 < n_dec:
            q_next = rows[i + 1]["q"]
        else:
            q_next = rows[i]["q"]
        d_q = (q_next - rows[i]["q"]).to(device)
        d_obj = torch.from_numpy(
            rows[i]["obj_after"] - rows[i]["obj_before"]).float().to(
                device)
        n_obj = d_obj.shape[0]
        fam["seq_phys"].append(
            huber(out["d_q"][0], d_q, q_scale)
            + huber(out["d_obj"][0, :n_obj].flatten(), d_obj.flatten(),
                    OBJ_SCALE))
        r_target = (sum(lab["valid_after"])
                    - sum(lab["valid_before"]))
        fam["reward"].append((out["reward"][0] - float(r_target)) ** 2)
        # 1-step latent closure to the EMA posterior
        if i + 1 < n_dec:
            fam["closure1"].append(torch.nn.functional.mse_loss(
                z_tilde, zs_ema[i + 1].detach()))
        # 2/3-block open loop at every 8th decision
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

    # paraphrase passes
    for vid in bundle.variant_ids():
        if not vid.startswith("paraphrase"):
            continue
        f = bundle.features[vid]
        zp = unroll(model, f["h"], f["mask"], actions, exec_lens, device)
        for i in range(0, n_dec, 4):
            fam["para"].append(torch.nn.functional.mse_loss(
                zp[i], zs[i]))
            lab = bundle.labels["per_decision"][i]["goals"][canon_gid]
            n_sub = len(lab["valid_before"])
            vb = bits_tensor(lab["valid_before"], n_sub_max, device)
            cur = model.d_current(zp[i])
            fam["current"].append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    cur["valid_bits"][0, :n_sub], vb[:n_sub]))

    # distinct-goal passes
    for vid in bundle.variant_ids():
        if not vid.startswith("distinct"):
            continue
        f = bundle.features[vid]
        gid = f["variant"]["goal_spec_id"]
        if gid not in bundle.labels["goals"]:
            continue
        zd = unroll(model, f["h"], f["mask"], actions, exec_lens, device)
        for i in range(0, n_dec, 4):
            lab = bundle.labels["per_decision"][i]["goals"][gid]
            n_sub = len(lab["valid_before"])
            vb = bits_tensor(lab["valid_before"], n_sub_max, device)
            cur = model.d_current(zd[i])
            fam["distinct_current"].append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    cur["valid_bits"][0, :n_sub], vb[:n_sub])
                + (cur["ordered_prefix"][0]
                   - lab["ordered_prefix_before"] / n_sub) ** 2)
            zt = model.predict(
                zd[i], actions[i][None].to(device),
                action_mask=(torch.arange(10, device=device)[None]
                             < exec_lens[i]))
            outd = model.d_next(zt)
            d_obj = torch.from_numpy(
                rows[i]["obj_after"]
                - rows[i]["obj_before"]).float().to(device)
            n_obj = d_obj.shape[0]
            fam["shared_phys"].append(
                huber(outd["d_obj"][0, :n_obj].flatten(),
                      d_obj.flatten(), OBJ_SCALE))

    # branch pass: absolute + centered effects at audited decisions
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
        pred = torch.stack(outs)
        target = torch.stack(d_objs)
        fam["branch_centered"].append(huber(
            (pred - pred.mean(0)).flatten(),
            (target - target.mean(0)).flatten(), OBJ_SCALE))

    # continuation heads + ranking (accepted groups only)
    for cont in bundle.continuations:
        d = cont["decision"]
        z = zs[d]
        by_goal: dict[str, dict] = {}
        for rec in cont["records"]:
            by_goal.setdefault(rec["goal_spec_id"], {}).setdefault(
                rec["branch_index"], []).append(rec)
        audit = next(a for a in src["audits"] if a["decision"] == d)
        chunk_of = {}
        for b in audit["branches"]:
            if b["kind"] == "candidate":
                chunk_of[b["candidate"]] = b["chunk_norm"]
            elif b["kind"] == "support":
                chunk_of[4] = b["chunk_norm"]
        for gid, branches in by_goal.items():
            vid = ("canonical" if gid == canon_gid
                   else f"distinct_{gid}")
            if vid == "canonical":
                z_goal = z
            elif vid in bundle.features:
                fz = bundle.features[vid]
                z_goal = unroll(model, fz["h"][:d + 1],
                                fz["mask"][:d + 1], actions[:d + 1],
                                exec_lens[:d + 1], device)[-1]
            else:
                continue
            scores = {}
            for bi, recs in branches.items():
                if chunk_of.get(bi) is None:
                    continue
                zt = model.predict(
                    z_goal, chunk_of[bi][None, :10].to(device))
                out = model.d_next(zt)
                y = [r["outcome"] for r in sorted(
                    recs, key=lambda r: r["repeat"])]
                succ = float(any(o["success_by_100"] for o in y))
                fam["cont_heads"].append(
                    torch.nn.functional
                    .binary_cross_entropy_with_logits(
                        out["success_logit"][0],
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
                if bi != 4:  # support never enters ranking
                    scores[bi] = (out["s"][0], y)
            for i, j in itertools.combinations(sorted(scores), 2):
                a_ij = paired_preference(scores[i][1], scores[j][1], {})
                if a_ij == 0:
                    continue
                p_ij = torch.sigmoid(scores[i][0] - scores[j][0])
                fam["ranking"].append(
                    torch.nn.functional.binary_cross_entropy(
                        p_ij.clamp(1e-6, 1 - 1e-6),
                        torch.tensor((a_ij + 1) / 2.0, device=device)))
    return fam


def demo_losses(model, ema, episode, device):
    """Standard-LIBERO representation support: physical + closure only."""
    fam = {k: [] for k in FAMILIES}
    h, mask = episode["prefix_hidden"], episode["prefix_mask"]
    actions = episode["action_block_norm"].float()
    # seq_prefix_cache_v1 blocks are full 10-action strides
    lens = episode.get("executed_lengths",
                       [10] * actions.shape[0])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs, mechanics only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results"
                        / "libero_loho_public_v1" / "v06_wm")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 2
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    n_sub_max = 7
    model = V06State().to(device)
    ema = make_ema(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WD)

    source_paths = [p for p in sorted((DATA / "sources").glob("*.pt"))]
    train_paths = [p for p in source_paths
                   if torch.load(p, weights_only=False)["split"]
                   == "train"]
    demo_paths = [p for p in sorted(SEQ_CACHE.glob("task*_demo*.pt"))]
    demos = []
    for p in demo_paths:
        e = torch.load(p, weights_only=False)
        if e["split"] == "train":
            demos.append(e)
    print(f"train sources: {len(train_paths)}, demo support: "
          f"{len(demos)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
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
                                    n_sub_max)
            else:
                fam = demo_losses(model, ema, demos[-1 - idx], device)
            losses = {k: torch.stack(v).mean()
                      for k, v in fam.items() if v}
            if not losses:
                continue
            total = torch.stack(list(losses.values())).mean()
            assert torch.isfinite(total), "non-finite loss"
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            ema_update(ema, model)
            for k, v in losses.items():
                epoch_fam[k].append(float(v))
        means = {k: (sum(v) / len(v) if v else None)
                 for k, v in epoch_fam.items()}
        logs.append({"epoch": epoch, **means})
        print(f"[v06 epoch {epoch}] " + " ".join(
            f"{k}={means[k]:.4f}" for k in FAMILIES
            if means[k] is not None), flush=True)
        if (epoch + 1) % 5 == 0 or args.smoke:
            torch.save({"model": model.state_dict(),
                        "ema": ema.state_dict(), "epoch": epoch,
                        "seed": args.seed},
                       args.output / f"checkpoint_epoch{epoch:03d}.pt")
    # nonzero-weight verification per registered loss source
    silent = [k for k in FAMILIES if not any(
        lg.get(k) is not None for lg in logs)]
    if silent:
        print(f"WARNING: silent loss families: {silent}", flush=True)
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "epoch": args.epochs - 1, "seed": args.seed,
                "budget": {"lr": LR, "wd": WD, "epochs": args.epochs,
                           "tbptt": TBPTT, "grad_norm": GRAD_NORM},
                "silent_families": silent},
               args.output / "checkpoint_final.pt")
    (args.output / "per_loss_logs.json").write_text(json.dumps(logs))
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
