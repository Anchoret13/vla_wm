#!/usr/bin/env python
"""V7.5 Stage 0 — recurrence gain measurement on the frozen V7.3C LCWM.

The 2026-08-09 A-stage diagnostic recorded that real / reset / shuffled
TOKEN states are identical to ~1e-6 relative at every anchor: the V7.3C
recurrence is functionally inert at inference. It recorded the symptom,
not the mechanism. This run localizes the mechanism and decides ONE
registered sub-choice for the V7.5 objective change.

Structure under test (lcwm/v06_model.py::V06State.update):

    c     = Anchor(h)                       # current observation + language
    z_bar = T(z_prev, E_a(a))               # history channel
    state = LayerNorm(c + beta * tanh(R(c, z_bar))),   beta = 0.25

R.out is zero-initialized by construction ("so the state is exactly the
anchored current observation at init"). History therefore reaches the
state only through a bounded, beta-attenuated, zero-init path. The
question this run answers:

    Is the binding constraint the BOUND (beta / tanh saturation) or the
    WEIGHT (R.out magnitude)?

Both scalings act identically while tanh is in its linear regime and
diverge only once it saturates, so the beta sweep locates the plateau
and the answer is read off the operating point.

APPROXIMATION, stated up front: h is not recoverable (the V7.2/V7.3
h-caches are retired), so Anchor(h) cannot be recomputed. c is stood in
for by the cached real token states, which is valid precisely in the
regime the A-stage established -- the history term is negligible, so
state ~ LayerNorm(c) and the cached states carry the correct
post-LayerNorm scale and realistic content. The conclusion is reported
as safe only if it is stable across every real anchor used as c. The
exact online quantity is produced later by the V7.5C per-epoch
history_dependence readout; this run is the cheap pre-training decision.

Inputs : 2026-08-06_v073_lcwm_r1/checkpoints/final.pt  (model weights)
         2026-08-09_v074_interface_diag_r1/probe_states.pt (real states)
Output : <date>_v075_recurrence_diag_r1/recurrence_gain.json
Tensor-only, CPU, seed 0, no environment and no policy forward.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v06_model import V06State  # noqa: E402
from lcwm.v067_lineage import sha256_file  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LCWM = RESULTS / "2026-08-06_v073_lcwm_r1" / "checkpoints" / "final.pt"
PROBE = RESULTS / "2026-08-09_v074_interface_diag_r1" / "probe_states.pt"
RID = "v075_recurrence_diag_r1"

BETAS = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0]
GAINS = [1.0, 4.0, 16.0, 64.0, 256.0]
SEED = 0


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 distance ||a-b|| / ||a||, over the whole token block."""
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def med(xs: list[float]) -> float:
    return float(torch.tensor(xs, dtype=torch.float64).median())


def main() -> None:
    torch.manual_seed(SEED)
    out = RESULTS / f"{run_date()}_{RID}"
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(LCWM, map_location="cpu", weights_only=False)
    model = V06State()
    model.load_state_dict(ck["model"])
    model.eval()
    beta0 = model.cfg.beta

    probe = torch.load(PROBE, map_location="cpu", weights_only=False)
    tokens = probe["tokens"]
    anchors = sorted({k.rsplit("::", 1)[0] for k in tokens})
    real = {a: tokens[f"{a}::real"] for a in anchors
            if f"{a}::real" in tokens}
    reset = {a: tokens[f"{a}::reset"] for a in anchors
             if f"{a}::reset" in tokens}
    anchors = sorted(real)
    assert len(anchors) >= 4, f"need >=4 anchors, got {len(anchors)}"

    def task_of(a: str) -> str:
        return a.split("_v073")[0].split("_teacher")[0]

    # ---- weight-scale census (why the history path is attenuated) -----
    sd = ck["model"]
    census = {}
    for tag, pref in [("r.out", "r.out"), ("r.block", "r.block"),
                      ("anchor", "anchor"), ("transition", "t."),
                      ("action_enc", "e_a")]:
        ts = [v for k, v in sd.items() if k.startswith(pref)]
        n = sum(v.numel() for v in ts)
        census[tag] = {
            "l2": float(sum(float(v.float().norm()) ** 2
                            for v in ts) ** 0.5),
            "max_abs": max(float(v.float().abs().max()) for v in ts),
            "rms": float((sum(float((v.float() ** 2).sum())
                              for v in ts) / n) ** 0.5),
            "n_params": int(n)}

    rows = []
    with torch.no_grad():
        for a in anchors:
            c = real[a]                       # stand-in for Anchor(h)
            others = [b for b in anchors if task_of(b) != task_of(a)]
            if not others:
                others = [b for b in anchors if b != a]
            other = others[0]

            a0, m0 = model.null_action(c.shape[0], c.device)
            act = model.e_a(a0, m0)

            zb_self = model.t(real[a], act)
            zb_other = model.t(real[other], act)
            zb_reset = model.t(reset.get(a, real[a]), act)
            zr = torch.randn_like(real[a]) * real[a].std() + real[a].mean()
            zb_rand = model.t(zr, act)

            def pre(zb: torch.Tensor) -> torch.Tensor:
                return model.r.out(model.r.block(c, zb))

            R_self, R_other = pre(zb_self), pre(zb_other)
            R_reset, R_rand = pre(zb_reset), pre(zb_rand)

            # ---- where inside R does z_bar get lost? -----------------
            # CrossBlock is pre-norm residual: out = q1 + mlp(norm(q1)),
            # q1 = c + attended.  z_bar enters ONLY through `attended`.
            blk = model.r.block

            def decompose(zb: torch.Tensor):
                kv = blk.proj(zb.to(blk.proj.weight.dtype))
                att, _ = blk.attn(blk.norm_q(c), blk.norm_kv(kv),
                                  blk.norm_kv(kv), need_weights=False)
                q1 = c + att
                return att, q1, blk.mlp(blk.norm_mlp(q1))

            att_s, q1_s, mlp_s = decompose(zb_self)
            att_o, _, _ = decompose(zb_other)
            att_r, _, _ = decompose(zb_rand)

            def state(zb_R: torch.Tensor, beta: float,
                      gain: float = 1.0) -> torch.Tensor:
                return model.norm(c + beta * torch.tanh(gain * zb_R))

            s_self = state(R_self, beta0)
            hist_term = beta0 * torch.tanh(R_self)

            # beta sweep and weight-gain sweep on the SAME contrast
            beta_gain = {f"{b:g}": rel(state(R_self, b),
                                       state(R_other, b))
                         for b in BETAS}
            weight_gain = {f"{g:g}": rel(state(R_self, beta0, g),
                                         state(R_other, beta0, g))
                           for g in GAINS}

            rows.append({
                "anchor": a, "contrast_anchor": other,
                # --- operating point of the tanh ---
                "R_pre_max_abs": float(R_self.abs().max()),
                "R_pre_mean_abs": float(R_self.abs().mean()),
                "R_pre_frac_gt1": float((R_self.abs() > 1).float().mean()),
                # --- how much the history term contributes at all ---
                "norm_c": float(c.norm()),
                "norm_hist_term": float(hist_term.norm()),
                "hist_over_c": float(hist_term.norm()
                                     / c.norm().clamp_min(1e-12)),
                # --- how much of R actually varies with history ---
                "R_sensitivity_other": rel(R_self, R_other),
                "R_sensitivity_reset": rel(R_self, R_reset),
                "R_sensitivity_random": rel(R_self, R_rand),
                # --- upstream: z_bar enters R only via `attended` ---
                "norm_attended": float(att_s.norm()),
                "norm_mlp": float(mlp_s.norm()),
                "attended_over_c": float(att_s.norm()
                                         / c.norm().clamp_min(1e-12)),
                "attended_sensitivity_other": rel(att_s, att_o),
                "attended_sensitivity_random": rel(att_s, att_r),
                # --- output-side history gain at the trained beta ---
                "gain_other": rel(s_self, state(R_other, beta0)),
                "gain_reset": rel(s_self, state(R_reset, beta0)),
                "gain_random": rel(s_self, state(R_rand, beta0)),
                # --- observation-channel reference (vary c, fix history) ---
                "gain_observation": rel(
                    s_self, model.norm(real[other]
                                       + beta0 * torch.tanh(pre(zb_self)))),
                "beta_sweep": beta_gain,
                "weight_gain_sweep": weight_gain})

    def col(k: str) -> list[float]:
        return [r[k] for r in rows]

    summary = {
        "n_anchors": len(rows),
        "beta_trained": beta0,
        "R_pre_max_abs_median": med(col("R_pre_max_abs")),
        "R_pre_frac_gt1_max": max(col("R_pre_frac_gt1")),
        "hist_over_c_median": med(col("hist_over_c")),
        "R_sensitivity_other_median": med(col("R_sensitivity_other")),
        "gain_other_median": med(col("gain_other")),
        "gain_random_median": med(col("gain_random")),
        "gain_observation_median": med(col("gain_observation")),
        "beta_sweep_median": {
            f"{b:g}": med([r["beta_sweep"][f"{b:g}"] for r in rows])
            for b in BETAS},
        "weight_gain_sweep_median": {
            f"{g:g}": med([r["weight_gain_sweep"][f"{g:g}"] for r in rows])
            for g in GAINS}}

    # ---- mechanical verdict ------------------------------------------
    # Two independent failure sites, tested separately. A gain that is
    # already at floating-point scale carries NO regime information, so
    # the beta/weight sweeps are only read as "did this recover usable
    # gain", never as a linear-regime ratio (an earlier revision of this
    # rule read a 1.8e-10 -> 1.0e-9 step as a linear response; both
    # numbers are noise).
    USABLE = 1e-4          # gain that would register above probe noise

    summary.update({
        "attended_over_c_median": med(col("attended_over_c")),
        "attended_sensitivity_other_median": med(
            col("attended_sensitivity_other")),
        "attended_sensitivity_random_median": med(
            col("attended_sensitivity_random"))})

    sweep = summary["beta_sweep_median"]
    wsweep = summary["weight_gain_sweep_median"]
    downstream_saturated = summary["R_pre_frac_gt1_max"] >= 0.99
    upstream_starved = summary["attended_sensitivity_other_median"] < 1e-3
    beta_recovers = max(sweep.values()) >= USABLE
    weight_recovers = max(wsweep.values()) >= USABLE

    sites = []
    if upstream_starved:
        sites.append("upstream_attention_starved")
    if downstream_saturated:
        sites.append("downstream_tanh_saturated")
    summary["failure_sites"] = sites
    summary["beta_sweep_recovers_gain"] = bool(beta_recovers)
    summary["weight_sweep_recovers_gain"] = bool(weight_recovers)
    summary["usable_gain_threshold"] = USABLE
    summary["observation_over_history_ratio"] = float(
        summary["gain_observation_median"]
        / max(summary["gain_other_median"], 1e-30))

    if not sites:
        sub = "no_architectural_change__objective_pressure_only"
    elif beta_recovers or weight_recovers:
        sub = ("raise_beta" if beta_recovers else "rescale_r_out")
    else:
        sub = "desaturate_and_reinject__scalar_fixes_insufficient"
    summary["binding_constraint"] = (
        "neither_beta_nor_weight" if sites and not
        (beta_recovers or weight_recovers) else "scalar_recoverable")
    summary["registered_subchoice"] = sub

    report = {
        "schema": "v075_recurrence_gain_v1",
        "run_id": RID,
        "seed": SEED,
        "lcwm": str(LCWM.relative_to(REPO_ROOT)),
        "lcwm_sha256": sha256_file(LCWM),
        "probe_states": str(PROBE.relative_to(REPO_ROOT)),
        "probe_states_sha256": sha256_file(PROBE),
        "approximation": (
            "c stood in for by cached real token states; h is not "
            "recoverable (V7.2/V7.3 h-caches retired). Valid in the "
            "measured regime where the history term is negligible so "
            "state ~ LayerNorm(c). Conclusion reported as safe only if "
            "stable across all anchors used as c."),
        "weight_census": census,
        "summary": summary,
        "rows": rows}
    (out / "recurrence_gain.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))

    print(f"[diag] {len(rows)} anchors -> {out/'recurrence_gain.json'}")
    print("\n=== weight census (max|w| per block) ===")
    for k, v in census.items():
        print(f"  {k:12s} max|w|={v['max_abs']:.4e}  rms={v['rms']:.4e}")
    print("\n=== tanh operating point ===")
    print(f"  median max|R_pre|      {summary['R_pre_max_abs_median']:.4e}")
    print(f"  max frac(|R_pre|>1)    {summary['R_pre_frac_gt1_max']:.4e}")
    print(f"  median ||beta*tanh R|| / ||c||  "
          f"{summary['hist_over_c_median']:.4e}")
    print("\n=== history gain at trained beta (relative L2) ===")
    for k in ("gain_other", "gain_reset", "gain_random"):
        print(f"  {k:18s} {med(col(k)):.4e}")
    print(f"  {'gain_observation':18s} "
          f"{summary['gain_observation_median']:.4e}   <- reference")
    print("\n=== beta sweep (median gain_other) ===")
    for b in BETAS:
        print(f"  beta={b:8g}  {sweep[f'{b:g}']:.4e}")
    print("\n=== R.out weight-gain sweep (median gain_other) ===")
    for g in GAINS:
        print(f"  gain={g:8g}  "
              f"{summary['weight_gain_sweep_median'][f'{g:g}']:.4e}")
    print("\n=== upstream: z_bar enters R only via `attended` ===")
    print(f"  median ||attended|| / ||c||      "
          f"{summary['attended_over_c_median']:.4e}")
    print(f"  median d(attended) vs other      "
          f"{summary['attended_sensitivity_other_median']:.4e}")
    print(f"  median d(attended) vs random     "
          f"{summary['attended_sensitivity_random_median']:.4e}")
    print(f"\n[verdict] failure sites: {summary['failure_sites']}")
    print(f"[verdict] beta sweep recovers usable gain:   "
          f"{summary['beta_sweep_recovers_gain']}")
    print(f"[verdict] weight sweep recovers usable gain: "
          f"{summary['weight_sweep_recovers_gain']}")
    print(f"[verdict] observation/history gain ratio:    "
          f"{summary['observation_over_history_ratio']:.3e}")
    print(f"[verdict] registered sub-choice = "
          f"{summary['registered_subchoice']}")


if __name__ == "__main__":
    main()
