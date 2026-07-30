#!/usr/bin/env python
"""H7.3 — train the fixed `v05_gated_wm` world model (seed 0, registered
budget; scoping decisions in 2026-07-29-h7-progress-note.md).

Data: 18 demo episodes (H_late from seq_prefix_cache_v1 + H_early SigLIP
from seq_libero_10_v2, t-aligned) + 40 public snapshots / 160 branches with
v2 feature sidecars (canonical + paraphrase H_late). π0.5 is never loaded.
The policy interface (w_c/w_w/w_g, α) is excluded from the optimizer and
asserted bitwise unchanged — this stage trains the world model only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v05_model import V05State  # noqa: E402

SEQ_CACHE = Path("/home/stargazer/Desktop/vla_wm/datasets/seq_prefix_cache_v1")
SEQ_V2 = Path("/home/stargazer/Desktop/vla_wm/datasets/seq_libero_10_v2")
PUBLIC = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1")
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4 + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUBER = 4.0
TBPTT = 16
LR, WD, EPOCHS, GRAD_NORM = 3e-4, 1e-4, 30, 1.0
FAMILIES = ("phys", "subgoal", "reward", "dq_public", "value", "para")


def huber(p, t, scale):
    return torch.nn.functional.huber_loss(p / scale, t / scale, delta=HUBER)


def load_demo_episodes():
    episodes = []
    for path in sorted(SEQ_CACHE.glob("task*_demo*.pt")):
        data = torch.load(path, weights_only=False)
        if data["split"] not in ("train", "dev"):
            continue
        shard = SEQ_V2 / f"task{data['task_id']}_demo{data['demo']}.pt"
        seq = torch.load(shard, weights_only=False)
        t_seq = [s["t"] for s in seq["steps"]]
        t_cache = data["t"].tolist()
        assert t_seq == t_cache, f"t misalignment {path.name}"
        siglip = torch.stack([s["siglip"] for s in seq["steps"]])
        episodes.append({
            "name": path.stem, "split": data["split"],
            "h_late": data["prefix_hidden"], "mask": data["prefix_mask"],
            "h_early": siglip,
            "q": data["q"], "obj": data["obj_pos"],
            "bits": data["predicate_bits"],
            "actions": data["action_block_norm"].float(),
        })
    return episodes


def load_public_snapshots():
    items = []
    for record_path in sorted(PUBLIC.glob("loho_*.pt")):
        if record_path.name.endswith(".features.pt"):
            continue
        features_path = record_path.with_suffix(".features.pt")
        if not features_path.exists():
            continue
        record = torch.load(record_path, weights_only=False)
        features = torch.load(features_path, weights_only=False)
        n_sub = len(record["subgoals"])
        for snap in record["snapshots"]:
            feats = features["snapshots"].get(snap["decision"])
            if feats is None:
                continue
            items.append({
                "name": f"{record['task']}_{record['policy']}"
                        f"_s{record['seed']}_d{snap['decision']}",
                "split": record["split"], "n_sub": n_sub,
                "state": feats["state"],
                "branches": list(zip(snap["branches"],
                                     feats["after_branches"])),
            })
    return items


def demo_episode_loss(model, episode, device, train=True):
    T = episode["h_late"].shape[0]
    fam = {k: [] for k in FAMILIES}
    w, g = model.initial(1, device)
    detach_at = T - TBPTT if train and T - 1 > TBPTT else None
    n_atoms = episode["bits"].shape[1]
    q_scale = Q_SCALE.to(device)
    for i in range(T - 1):
        if detach_at is not None and i == detach_at - 1:
            w, g = w.detach(), g.detach()
        a = episode["actions"][i][None].to(device)
        h_early_next = episode["h_early"][i + 1][None].float().to(device)
        h_late_next = episode["h_late"][i + 1][None].float().to(device)
        m_next = episode["mask"][i + 1][None].to(device)
        q_next = episode["q"][i + 1][None].to(device)
        prior = model.physical_prior(w, a)
        w = model.step_physical(w, a, h_early_next, q_next)
        g = model.step_task(g, w, a, h_late_next, m_next)
        c = model.current(h_early_next, h_late_next, m_next, q_next)
        out = model.outcomes(prior, c, g)
        d_q = (episode["q"][i + 1] - episode["q"][i]).to(device)
        d_obj = (episode["obj"][i + 1] - episode["obj"][i]).to(device)
        n_obj = d_obj.shape[0]
        fam["phys"].append(
            huber(out["d_q"][0], d_q, q_scale)
            + huber(out["d_obj"][0, :n_obj].flatten(),
                    d_obj.flatten(), OBJ_SCALE))
        fam["subgoal"].append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                out["subgoal_logits"][0, :n_atoms],
                episode["bits"][i + 1].float().to(device)))
        r_target = (episode["bits"][i + 1].float().sum()
                    - episode["bits"][i].float().sum()).to(device)
        fam["reward"].append((out["r_hat"][0] - r_target) ** 2)
    return {k: (torch.stack(v).mean() if v else None)
            for k, v in fam.items()}


def snapshot_loss(model, item, device):
    fam = {k: [] for k in FAMILIES}
    state = item["state"]
    h_early = state["h_early"][None].float().to(device)
    h_late = state["h_late"][None].float().to(device)
    mask = state["h_late_mask"][None].to(device)
    q = state["q"][None].to(device)
    zero_a = torch.zeros(1, 10, 7, device=device)
    zero_m = torch.zeros(1, 10, dtype=torch.bool, device=device)
    w0, g0 = model.initial(1, device)
    w = model.step_physical(w0, zero_a, h_early, q, action_mask=zero_m)
    c = model.current(h_early, h_late, mask, q)
    g = model.step_task(g0, w, zero_a, h_late, mask, action_mask=zero_m)
    para = state["language_variants"]["paraphrase"]
    h_para = para["h_late"][None].float().to(device)
    m_para = para["h_late_mask"][None].to(device)
    c_p = model.current(h_early, h_para, m_para, q)
    g_p = model.step_task(g0, w, zero_a, h_para, m_para, action_mask=zero_m)
    fam["para"].append(
        torch.nn.functional.mse_loss(g, g_p)
        + torch.nn.functional.mse_loss(c, c_p))
    q_scale = Q_SCALE.to(device)
    for branch, after in item["branches"]:
        chunk10 = branch["chunk_norm"][None, :10].float().to(device)
        prior = model.physical_prior(w, chunk10)
        out = model.outcomes(prior, c, g)
        d_q = (after["q"] - state["q"]).to(device)
        d_obj = (after["obj_pos"] - state["obj_pos"]).to(device)
        n_obj = d_obj.shape[0]
        fam["phys"].append(
            huber(out["d_q"][0], d_q, q_scale)
            + huber(out["d_obj"][0, :n_obj].flatten(),
                    d_obj.flatten(), OBJ_SCALE))
        target_sub = torch.zeros(item["n_sub"], device=device)
        for idx in branch["subgoals_after_branch"]:
            target_sub[int(idx)] = 1.0
        fam["subgoal"].append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                out["subgoal_logits"][0, :item["n_sub"]], target_sub))
        fam["reward"].append(
            (out["r_hat"][0]
             - torch.tensor(branch["reward_subgoal_delta"],
                            device=device)) ** 2)
        fam["dq_public"].append(
            (out["dq_public_hat"][0]
             - torch.tensor(branch["continuation"]["dq_public"],
                            device=device)) ** 2)
        fam["value"].append(
            (out["v_hat"][0]
             - torch.tensor(branch["continuation"]["q_public"],
                            device=device)) ** 2)
    return {k: (torch.stack(v).mean() if v else None)
            for k, v in fam.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "results" / "libero_loho_public_v1"
        / "v05_gated_wm")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    demos = [e for e in load_demo_episodes() if e["split"] == "train"]
    snaps = [s for s in load_public_snapshots() if s["split"] == "train"]
    print(f"train: {len(demos)} demo episodes, {len(snaps)} public "
          f"snapshots ({sum(len(s['branches']) for s in snaps)} branches)",
          flush=True)

    model = V05State().to(device)
    frozen_names = ("w_c", "w_w", "w_g", "alpha_w", "alpha_g")
    frozen = [p.detach().clone() for n, p in model.named_parameters()
              if any(n.startswith(f) for f in frozen_names)]
    trainable = [p for n, p in model.named_parameters()
                 if not any(n.startswith(f) for f in frozen_names)]
    for n, p in model.named_parameters():
        if any(n.startswith(f) for f in frozen_names):
            p.requires_grad_(False)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)

    args.output.mkdir(parents=True, exist_ok=True)
    logs = []
    items = [("demo", i) for i in range(len(demos))] + [
        ("snap", i) for i in range(len(snaps))]
    for epoch in range(args.epochs):
        random.shuffle(items)
        epoch_fam = {k: [] for k in FAMILIES}
        for kind, index in items:
            optimizer.zero_grad(set_to_none=True)
            fam = (demo_episode_loss(model, demos[index], device)
                   if kind == "demo"
                   else snapshot_loss(model, snaps[index], device))
            losses = [v for v in fam.values() if v is not None]
            total = torch.stack(losses).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_NORM)
            optimizer.step()
            for k, v in fam.items():
                if v is not None:
                    epoch_fam[k].append(float(v))
        means = {k: (sum(v) / len(v) if v else None)
                 for k, v in epoch_fam.items()}
        logs.append({"epoch": epoch, **means})
        print(f"[v05 epoch {epoch}] " + " ".join(
            f"{k}={means[k]:.4f}" for k in FAMILIES if means[k] is not None),
            flush=True)
        if (epoch + 1) % 5 == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "seed": args.seed},
                       args.output / f"checkpoint_epoch{epoch:03d}.pt")
    current = [p.detach().clone() for n, p in model.named_parameters()
               if any(n.startswith(f) for f in frozen_names)]
    for a, b in zip(frozen, current):
        assert torch.equal(a, b), "policy interface changed in WM stage"
    # r1 launch check: grounded scores must vary within a sibling group
    # (the class of the escaped v0 wiring defect).
    with torch.no_grad():
        item = snaps[0]
        state = item["state"]
        h_early = state["h_early"][None].float().to(device)
        h_late = state["h_late"][None].float().to(device)
        mask = state["h_late_mask"][None].to(device)
        q = state["q"][None].to(device)
        zero_a = torch.zeros(1, 10, 7, device=device)
        zero_m = torch.zeros(1, 10, dtype=torch.bool, device=device)
        w0, g0 = model.initial(1, device)
        w = model.step_physical(w0, zero_a, h_early, q, action_mask=zero_m)
        c = model.current(h_early, h_late, mask, q)
        g = model.step_task(g0, w, zero_a, h_late, mask, action_mask=zero_m)
        chunks = torch.stack([b["chunk_norm"][:10]
                              for b, _ in item["branches"]]).to(device)
        n_b = chunks.shape[0]
        prior = model.physical_prior(w.expand(n_b, -1, -1), chunks)
        v = model.outcomes(prior, c.expand(n_b, -1, -1),
                           g.expand(n_b, -1, -1))["v_hat"]
        assert float(v.std()) > 0.0, (
            "grounded v_hat has zero within-sibling variance")
        print(f"sibling-variance launch check PASSED "
              f"(v_hat std {float(v.std()):.2e})", flush=True)
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1,
                "seed": args.seed, "budget": {
                    "lr": LR, "wd": WD, "epochs": args.epochs,
                    "tbptt": TBPTT, "grad_norm": GRAD_NORM}},
               args.output / "checkpoint_final.pt")
    (args.output / "per_loss_logs.json").write_text(json.dumps(logs))
    print(f"-> {args.output}", flush=True)


if __name__ == "__main__":
    main()
