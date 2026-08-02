#!/usr/bin/env python
"""V7.0.0 — coupling checks needed to bind the new experiment, computed
from the SELECTED step-300 checkpoint (hash-verified against
run_manifest_v69_dev before any statistic; the historical bias stats
were computed on final step 401 — recorded, not rewritten).

Over the strict-clean train anchors:
- recurrent and reset mean/centered bias RMS;
- matched-noise first-ten action shift (biased vs zero-bias chunks
  under identical flow noise) -> matched_noise_action_shift.npz;
- rho_state = E_i ||da_i - mean(da)||^2 / E_i ||da_i||^2.

Full intercept/weight/nonlinear decomposition is deferred to the new
arms' evaluation (not a launch gate).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import sha256_file  # noqa: E402

DATA = Path("/home/stargazer/Desktop/vla_wm/datasets/libero_loho_public_v1"
            "/v06_effect_crossed")
RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
AUDIT = RESULTS / "v070_policy" / "audit"


class SelectedToken:
    """Proof that the evaluated checkpoint was hash-verified and loaded.
    Policy statistics can only be computed with a verified token."""

    def __init__(self):
        self.verified = False
        self.step = None


def load_selected(model) -> SelectedToken:
    tok = SelectedToken()
    wz_path = RESULTS / "v069_policy" / "grounded_wz" / "wz_selected.pt"
    man = json.loads((RESULTS / "v069_eval"
                      / "run_manifest_v69_dev.json").read_text())
    assert sha256_file(wz_path) == man["wz_sha256"], \
        "selected checkpoint hash mismatch"
    wz = torch.load(wz_path, weights_only=False)
    model.w_z.load_state_dict(wz["w_z"])
    tok.verified = True
    tok.step = wz["step"]
    return tok


def compute_policy_statistics(model, token: SelectedToken, runner,
                              anchors, device):
    assert isinstance(token, SelectedToken) and token.verified, (
        "policy statistics require the hash-verified selected "
        "checkpoint to be loaded first")
    from lcwm.lc_flow import sample_chunks_lc
    from lcwm.sampler import prefix_forward

    b_rec, b_reset, shifts = [], [], []
    meta = []
    for a in anchors:
        src = a["source"]
        d = a["decision"]
        z = None
        z_pair = None
        with torch.no_grad():
            for i, row in enumerate(src["rows"]):
                if i > d:
                    break
                batch = runner._obs_to_policy_batch(
                    row["obs"], src["language_canonical"])
                prefix = prefix_forward(runner.policy, batch)
                h = prefix.hidden.float()
                m = prefix.pad_masks.bool()
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = src["rows"][i - 1]
                    aa = prev["chunk_norm"][None, :10].float().to(
                        device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, aa, h, m, action_mask=am)
                if i == d:
                    z_pair = (z.detach(),
                              model.initial_state(h, m).detach(),
                              prefix, batch)
            z_rec, z_reset, prefix, batch = z_pair
            br = model.policy_bias(z_rec)
            b0 = model.policy_bias(z_reset)
            b_rec.append(br.cpu())
            b_reset.append(b0.cpu())
            seed = int.from_bytes(hashlib.sha256(
                f"v070_audit|shift|{a['source_id']}|{d}".encode()
            ).digest()[:8], "big") & ((1 << 63) - 1)
            chunk_b = sample_chunks_lc(runner.policy, batch, br, n=1,
                                       seed=seed, prefix=prefix)
            chunk_0 = sample_chunks_lc(runner.policy, batch,
                                       torch.zeros_like(br), n=1,
                                       seed=seed, prefix=prefix)
            da = (chunk_b[0, :10] - chunk_0[0, :10]).float().cpu()
            shifts.append(da.numpy())
            meta.append(f"{a['source_id']}_d{d}")
    B, B0 = torch.cat(b_rec), torch.cat(b_reset)
    D = np.stack(shifts)                       # (n, 10, 7)
    flat = D.reshape(len(shifts), -1)
    mean_shift = flat.mean(axis=0)
    rho = float((np.linalg.norm(flat - mean_shift, axis=1) ** 2).mean()
                / max((np.linalg.norm(flat, axis=1) ** 2).mean(),
                      1e-12))
    stats = {
        "selected_step": token.step,
        "n_anchors": len(meta),
        "bias_rms_recurrent": float(B.pow(2).mean().sqrt()),
        "bias_rms_reset": float(B0.pow(2).mean().sqrt()),
        "bias_centered_rms_recurrent": float(
            (B - B.mean(0, keepdim=True)).pow(2).mean().sqrt()),
        "bias_centered_rms_reset": float(
            (B0 - B0.mean(0, keepdim=True)).pow(2).mean().sqrt()),
        "action_shift_rms": float(
            np.sqrt((flat ** 2).mean())),
        "rho_state": rho,
    }
    return stats, D, meta


def main() -> None:
    from lcwm.chassis import Pi05Runner

    device = torch.device("cuda")
    strict = json.loads(
        (AUDIT / "strict_replay_report.json").read_text())
    clean_train = set(strict["clean_anchor_lists"]["train"])

    sources = {}
    anchors = []
    for p in sorted((DATA / "corrections_v069").glob("*.pt")):
        g = torch.load(p, weights_only=False)
        akey = f"{g['source_id']}_d{g['decision']}"
        if akey not in clean_train:
            continue
        sid = g["source_id"]
        if sid not in sources:
            sources[sid] = torch.load(
                DATA / "corrections_sources_v069" / f"{sid}.pt",
                weights_only=False)
        anchors.append({"source_id": sid, "decision": g["decision"],
                        "source": sources[sid]})
    assert len(anchors) == 14, len(anchors)

    model = V06State().to(device)
    wm = torch.load(RESULTS / "v069_predictive"
                    / "checkpoint_selected.pt", weights_only=False)
    model.load_state_dict(wm["model"])
    model.eval()
    runner = Pi05Runner(suite_name="libero_10")

    token = load_selected(model)
    stats, D, meta = compute_policy_statistics(
        model, token, runner, anchors, device)
    np.savez_compressed(AUDIT / "matched_noise_action_shift.npz",
                        shifts=D, anchors=np.array(meta))
    (AUDIT / "coupling_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    print(f"-> {AUDIT}", flush=True)


if __name__ == "__main__":
    main()
