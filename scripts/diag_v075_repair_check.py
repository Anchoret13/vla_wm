#!/usr/bin/env python
"""V7.5 Stage 1b — does the repaired architecture pass history at all?

Runs the Stage 0 measurement through `lcwm/v075_state.py::V075State`
loaded with the FROZEN V7.3C weights, beside the unmodified V06State
baseline, on identical inputs.

This is a PLUMBING check, not a performance claim. The V7.3C weights
were trained for the old architecture; `r_norm` is at its default
identity init and `hist_center` is fitted from the cached probe states.
The only question it answers is the one that gates spending a training
run: with the two Stage 0 defects repaired, does a change of history
reach the output at all, or is the path still dead?

Registered pass condition (frozen here, before the run):
  the median relative output gain for a different anchor's history
  clears the Stage 0 usable threshold 1e-4, i.e. at least a ~5e5x
  improvement over the measured 1.82e-10.

Output: <date>_v075_recurrence_diag_r1/repair_check.json
Tensor-only, CPU, seed 0.
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
from lcwm.v075_state import V075State, load_v06_weights  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
LCWM = RESULTS / "2026-08-06_v073_lcwm_r1" / "checkpoints" / "final.pt"
PROBE = RESULTS / "2026-08-09_v074_interface_diag_r1" / "probe_states.pt"
RID = "v075_recurrence_diag_r1"
USABLE = 1e-4
SEED = 0


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True, text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def med(xs: list[float]) -> float:
    return float(torch.tensor(xs, dtype=torch.float64).median())


def measure(model, anchors, real, reset, centered: bool) -> list[dict]:
    """Stage 0 metrics through whichever `update` this model defines."""
    rows = []

    def task_of(a: str) -> str:
        return a.split("_v073")[0].split("_teacher")[0]

    with torch.no_grad():
        for a in anchors:
            c = real[a]
            others = [b for b in anchors if task_of(b) != task_of(a)] \
                or [b for b in anchors if b != a]
            other = others[0]
            a0, m0 = model.null_action(c.shape[0], c.device)
            act = model.e_a(a0, m0)

            def pre(z_prev: torch.Tensor) -> torch.Tensor:
                zb = model.t(z_prev, act)
                if centered:
                    zb = model.hist_center(zb)
                    return model.r_norm(model.r(c, zb))
                return model.r(c, zb)

            def state(R: torch.Tensor) -> torch.Tensor:
                return model.norm(c + model.cfg.beta * torch.tanh(R))

            R_self = pre(real[a])
            R_other = pre(real[other])
            R_reset = pre(reset.get(a, real[a]))
            zr = torch.randn_like(real[a]) * real[a].std() + real[a].mean()
            R_rand = pre(zr)
            s_self = state(R_self)

            rows.append({
                "anchor": a,
                "R_pre_max_abs": float(R_self.abs().max()),
                "R_pre_frac_gt1": float((R_self.abs() > 1).float().mean()),
                "R_sensitivity_other": rel(R_self, R_other),
                "gain_other": rel(s_self, state(R_other)),
                "gain_reset": rel(s_self, state(R_reset)),
                "gain_random": rel(s_self, state(R_rand)),
                "gain_observation": rel(
                    s_self, model.norm(real[other]
                                       + model.cfg.beta
                                       * torch.tanh(R_self)))})
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = ["R_pre_max_abs", "R_pre_frac_gt1", "R_sensitivity_other",
            "gain_other", "gain_reset", "gain_random", "gain_observation"]
    s = {k: med([r[k] for r in rows]) for k in keys}
    s["observation_over_history_ratio"] = float(
        s["gain_observation"] / max(s["gain_other"], 1e-30))
    return s


def main() -> None:
    torch.manual_seed(SEED)
    out = RESULTS / f"{run_date()}_{RID}"
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(LCWM, map_location="cpu", weights_only=False)
    probe = torch.load(PROBE, map_location="cpu", weights_only=False)
    tokens = probe["tokens"]
    anchors = sorted({k.rsplit("::", 1)[0] for k in tokens
                      if k.endswith("::real")})
    real = {a: tokens[f"{a}::real"] for a in anchors}
    reset = {a: tokens[f"{a}::reset"] for a in anchors
             if f"{a}::reset" in tokens}

    old = V06State()
    old.load_state_dict(ck["model"])
    old.eval()

    new = V075State()
    missing, _ = load_v06_weights(new, ck["model"])
    new.eval()

    # Fit the history centering on the z_bar values it will normalize.
    with torch.no_grad():
        a0, m0 = new.null_action(1, torch.device("cpu"))
        act = new.e_a(a0, m0)
        zb_all = torch.cat([new.t(tokens[k], act) for k in sorted(tokens)],
                           dim=0)
        new.hist_center.fit(zb_all)
        new.hist_center.freeze()

    rows_old = measure(old, anchors, real, reset, centered=False)
    rows_new = measure(new, anchors, real, reset, centered=True)
    s_old, s_new = summarize(rows_old), summarize(rows_new)

    passed = s_new["gain_other"] >= USABLE
    report = {
        "schema": "v075_repair_check_v1",
        "run_id": RID, "seed": SEED,
        "lcwm": str(LCWM.relative_to(REPO_ROOT)),
        "lcwm_sha256": sha256_file(LCWM),
        "probe_states_sha256": sha256_file(PROBE),
        "note": ("PLUMBING check only. V7.3C weights were trained for "
                 "the OLD architecture; r_norm is at identity init and "
                 "hist_center is fitted from the cached probe states. "
                 "Nothing here is a performance or quality claim."),
        "new_modules_initialized_fresh": sorted(missing),
        "centering": {
            "mu_norm": float(new.hist_center.mu.norm()),
            "sigma": float(new.hist_center.sigma),
            "fitted_on_states": int(zb_all.shape[0])},
        "usable_gain_threshold": USABLE,
        "registered_pass_condition": (
            "median gain_other >= 1e-4 on the repaired architecture"),
        "pass": bool(passed),
        "summary_old": s_old, "summary_new": s_new,
        "improvement_factor": float(
            s_new["gain_other"] / max(s_old["gain_other"], 1e-30)),
        "rows_old": rows_old, "rows_new": rows_new}
    (out / "repair_check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))

    w = 26
    print(f"[repair-check] {len(anchors)} anchors -> "
          f"{out/'repair_check.json'}")
    print(f"  centering fitted on {zb_all.shape[0]} states, "
          f"||mu||={float(new.hist_center.mu.norm()):.4f} "
          f"sigma={float(new.hist_center.sigma):.4f}")
    print(f"\n{'quantity':{w}} {'V7.3C (old)':>14} {'V7.5 repaired':>14}")
    print("-" * (w + 30))
    for k in ("R_pre_max_abs", "R_pre_frac_gt1", "R_sensitivity_other",
              "gain_other", "gain_reset", "gain_random",
              "gain_observation", "observation_over_history_ratio"):
        print(f"{k:{w}} {s_old[k]:14.4e} {s_new[k]:14.4e}")
    print(f"\n[repair-check] history gain improved "
          f"{report['improvement_factor']:.3e}x")
    print(f"[repair-check] registered pass condition "
          f"(median gain_other >= {USABLE:g}): "
          f"{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
