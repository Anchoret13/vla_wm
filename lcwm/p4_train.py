"""P4 e2e world-model trainer (M1/M2 spec, 2026-07-28; A1 of 2026-07-29).

Reuses the existing mechanics — LCState, EMATarget/LatentPredictor,
residual_self_loss, variance_covariance_penalty, unroll_lc_history,
p4_targets — and adds only: the physical-only centered branch objective,
the source-balanced schedule, and the strict checkpoint bundle.

No π0.5 model is loaded in this stage; everything trains from cached prefix
tensors. The policy adapter (wz_hidden/wz_out) is frozen and asserted
bitwise unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from lcwm.lc_flow import LCState
from lcwm.p4_targets import branch_delta_phi
from lcwm.self_predict import (
    EMATarget,
    LatentPredictor,
    latent_distance,
    residual_self_loss,
)
from lcwm.value_heads import ValueHeads

# Registered P2-v3 optimizer budget (M2).
LR = 3e-4
WEIGHT_DECAY = 1e-4
EMA_TAU = 0.996
TBPTT = 16
VIC_ACCUM_SOURCES = 4
VIC_VAR_WEIGHT = 25.0
VIC_COV_WEIGHT = 1.0
HUBER_DELTA = 4.0
MAX_GRAD_NORM = 1.0
# Locked-v2 physical scales.
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4 + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3


def _huber(pred: Tensor, target: Tensor, scale: Tensor | float) -> Tensor:
    return torch.nn.functional.huber_loss(
        pred / scale, target / scale, delta=HUBER_DELTA
    )


def load_branch_groups(branch_dir: Path, labels_dir: Path) -> dict:
    """Admitted train branch groups by chain source, with precomputed
    sibling ΔΦ under the memo rule (valid contact groups only for ΔΦ;
    stall groups still contribute effects/predicates/reward)."""
    manifest = json.loads((branch_dir / "branch_manifest.json").read_text())
    by_source: dict[str, list[dict]] = {}
    for relative in manifest["splits"]["train"]:
        group = torch.load(branch_dir / relative, weights_only=False)
        source = group["source_trajectory_id"]
        sidecar_path = labels_dir / f"{source}.pt"
        if sidecar_path.exists():
            sidecar = torch.load(sidecar_path, weights_only=False)
            dphi, valid, violation = branch_delta_phi(group, sidecar)
        else:
            n = group["actions_norm"].shape[0]
            dphi = torch.zeros(n)
            valid = torch.zeros(n, dtype=torch.bool)
            violation = torch.zeros(n, dtype=torch.bool)
        group["branch_dphi"] = dphi
        group["branch_dphi_valid"] = valid & ~violation
        by_source.setdefault(source, []).append(group)
    for source in by_source:
        by_source[source].sort(key=lambda g: g["snapshot_id"])
    return by_source


def sequential_source_loss(
    lc, predictor, ema, heads, episode, targets, device, train=True
):
    """Per-source loss families for one sequential episode (weight-1 each
    after in-source mean over valid targets). Returns (families, states)."""
    hidden, mask = episode["prefix_hidden"], episode["prefix_mask"]
    actions = episode["actions_norm"]
    exec_mask = episode["action_exec_mask"]
    T = hidden.shape[0]
    transitions = T - 1
    detach_before = (
        T - TBPTT if train and transitions > TBPTT else None
    )
    h0, m0 = hidden[0:1].to(device), mask[0:1].to(device)
    z = lc.posterior(h0, m0)
    with torch.no_grad():
        z_tgt = ema.posterior(h0, m0)

    fam = {k: [] for k in (
        "self", "residual", "physical", "next_bits", "reward", "dphi",
        "v_next",
    )}
    states = [z]
    bits = episode["predicate_bits"]
    n_atoms = bits.shape[1]
    n_obj = episode["obj_pos"].shape[1]
    q_scale = Q_SCALE.to(device)
    for i in range(transitions):
        if detach_before is not None and i == detach_before - 1:
            z = z.detach()
        a = actions[i : i + 1].to(device)
        am = exec_mask[i : i + 1].to(device)
        h1 = hidden[i + 1 : i + 2].to(device)
        m1 = mask[i + 1 : i + 2].to(device)
        prior = lc.transition(z, a, am)
        with torch.no_grad():
            z_tgt_next = ema.step(z_tgt, a, h1, m1, am)
        pred = predictor(prior)
        fam["self"].append(latent_distance(pred, z_tgt_next, "cosine"))
        fam["residual"].append(
            residual_self_loss(pred, z_tgt_next, z_tgt)
        )
        if bool(targets["window_labeled"][i]):
            out = lc.outcome(prior)
            d_q_t = (episode["q"][i + 1] - episode["q"][i]).to(device)
            d_obj_t = (
                episode["obj_pos"][i + 1] - episode["obj_pos"][i]
            ).to(device)
            fam["physical"].append(
                _huber(out["d_q"][0], d_q_t, q_scale)
                + _huber(
                    out["d_obj"][0, :n_obj].flatten(),
                    d_obj_t.flatten(),
                    OBJ_SCALE,
                )
            )
            fam["next_bits"].append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    out["next_bits_logits"][0, :n_atoms],
                    bits[i + 1].float().to(device),
                )
            )
            vh = heads(prior)
            fam["reward"].append(
                (vh["r_hat"][0] - targets["reward"][i].to(device)) ** 2
            )
            if bool(targets["dphi_valid"][i]):
                fam["dphi"].append(
                    (vh["dphi_hat"][0] - targets["dphi"][i].to(device))
                    ** 2
                )
            v_t = targets["v_pi0"][i]
            if not torch.isnan(v_t):
                fam["v_next"].append(
                    (vh["v_hat"][0] - v_t.to(device)) ** 2
                )
        z = lc.step(z, a, h1, m1, am)
        states.append(z)
        with torch.no_grad():
            z_tgt = z_tgt_next
    families = {
        k: (torch.stack(v).mean() if v else None) for k, v in fam.items()
    }
    return families, torch.cat(states)


def branch_source_loss(lc, heads, group, device):
    """Branch families: centered physical-only sibling effects (per-channel
    resolution masks), next predicates, atom-delta reward, valid ΔΦ."""
    from lcwm.lc_flow_train import unroll_lc_history

    z = unroll_lc_history(lc, group, device=device, truncate_bptt=TBPTT)
    n = group["actions_norm"].shape[0]
    z_all = z.expand(n, -1, -1)
    actions = group["executed_actions_norm"].to(device).float()
    exec_mask = group["execution_mask"].to(device)
    prior = lc.transition(z_all, actions, exec_mask)
    out = lc.outcome(prior)
    heads_out = heads(prior)

    qc = group["restore_qc"]
    fam = {}
    q_scale = Q_SCALE.to(device)
    d_q_t = (group["q_after"] - group["q_before"][None]).to(device)
    n_obj = int(group["object_mask"].sum())
    d_obj_t = (
        group["object_pos_after"][:, :n_obj]
        - group["object_pos_before"][None, :n_obj]
    ).to(device)

    def centered(x):
        return x - x.mean(dim=0, keepdim=True)

    terms = []
    if bool(qc.get("q_resolved", False)):
        terms.append(_huber(centered(out["d_q"]), centered(d_q_t), q_scale))
    if bool(qc.get("object_resolved", False)):
        terms.append(
            _huber(
                centered(out["d_obj"][:, :n_obj].flatten(1)),
                centered(d_obj_t.flatten(1)),
                OBJ_SCALE,
            )
        )
    fam["physical"] = torch.stack(terms).mean() if terms else None

    n_atoms = int(group["atom_mask"].sum())
    fam["next_bits"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            out["next_bits_logits"][:, :n_atoms],
            group["bits_after"][:, :n_atoms].float().to(device),
        )
    )
    r_t = (
        group["bits_after"][:, :n_atoms].float().sum(-1)
        - group["bits_before"][:n_atoms].float().sum()
    ).to(device)
    fam["reward"] = ((heads_out["r_hat"] - r_t) ** 2).mean()
    valid = group["branch_dphi_valid"]
    if bool(valid.any()):
        fam["dphi"] = (
            (
                heads_out["dphi_hat"][valid.to(device)]
                - group["branch_dphi"][valid].to(device)
            )
            ** 2
        ).mean()
    else:
        fam["dphi"] = None
    return fam


def checkpoint_bundle(lc, predictor, ema, heads, optimizer, step, config):
    return {
        "lc_state": lc.state_dict(),
        "predictor": predictor.state_dict(),
        "ema": ema.state_dict(),
        "value_heads": heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "config": config,
    }


def strict_load(bundle, lc, predictor, ema, heads, optimizer=None):
    lc.load_state_dict(bundle["lc_state"], strict=True)
    predictor.load_state_dict(bundle["predictor"], strict=True)
    ema.load_state_dict(bundle["ema"], strict=True)
    heads.load_state_dict(bundle["value_heads"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(bundle["optimizer"])
    return bundle["global_step"]


def build_models(device):
    lc = LCState().to(device)
    # M1: zero-init only the d_q/d_obj output layers; freeze legacy
    # d_prog/ret heads; next_bits stays trainable.
    for name in ("dq", "dobj"):
        layer = getattr(lc.outcome, name)
        last = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)][-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
    for name in ("dprog", "ret"):
        for p in getattr(lc.outcome, name).parameters():
            p.requires_grad_(False)
    # Policy adapter frozen and asserted unchanged by the caller.
    lc.wz_hidden.requires_grad_(False)
    lc.wz_out.requires_grad_(False)
    predictor = LatentPredictor().to(device)
    ema = EMATarget(lc, tau=EMA_TAU).to(device)
    heads = ValueHeads().to(device)
    return lc, predictor, ema, heads
