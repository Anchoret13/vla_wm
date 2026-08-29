#!/usr/bin/env python
"""Is the world model good enough to imagine in? Measured before using it.

    python scripts/probe_v159_wm_quality.py --wm <wm.pt> --tapes ...

Asked because the honest answer to "is the WM good enough" was: unknown. Every
number so far described DOWNSTREAM ranking AUC (0.617 / 0.629); the model's
accuracy as a PREDICTOR was never measured. Imagination depends on exactly that,
and on how it degrades with depth.

Three quantities, each tied to a way imagination fails:

  1. one-step error vs the identity floor
     Latents are autocorrelated, so copying z_t is already a decent prediction. If
     the model does not beat identity in the LEARNED latent, there is nothing to
     imagine with. The shuffled-action control separates "predicts the future" from
     "uses the action".

  2. compounding error over h = 1..8
     TD-MPC2 plans at horizon 3 and gives no justification beyond it working, which
     is itself a statement about how fast these models drift. v148 saw ranking AUC
     fall monotonically with depth (0.650 -> 0.609 -> 0.600 at h = 2/4/8), which is
     consistent with drift but was measured downstream, not directly.

  3. the value gap: V on imagined states vs V on the real states they predict
     This is the surface a residual actor exploits. If V(T^h(z,u)) is systematically
     higher than V(encode(o_{t+h})), then optimising a policy against the model buys
     imagined return and nothing else - the classic model-based failure, which was
     written into the actor script as its expected failure mode. Measuring the gap
     first says whether that expectation is already realised.

No environment steps.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import numpy as np, torch  # noqa: E402
from train_v152_tdmpc_wm import build_sequences  # noqa: E402
from train_v153_td_value_wm import ValueWM  # noqa: E402

OUT = REPO / "results" / "v159_wm_quality"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--max-h", type=int, default=8)
    a = ap.parse_args()
    ck = torch.load(a.wm, weights_only=False)
    Z, U, ZN, EP, TT = [], [], [], [], []
    off, dim0 = 0, None
    for tp in a.tapes:
        d = torch.load(tp, weights_only=False)
        if dim0 is None:
            dim0 = int(d["z"].shape[-1])
        assert int(d["z"].shape[-1]) == dim0
        Z.append(d["z"]); U.append(d["u"]); ZN.append(d["z_next"])
        EP.append(d["episode"] + off); TT.append(d["t"])
        off += int(d["episode"].max()) + 1
    z, u, zn = torch.cat(Z), torch.cat(U), torch.cat(ZN)
    ep, tt = torch.cat(EP), torch.cat(TT)
    mu_o, sd_o = ck["mu_o"], ck["sd_o"]
    O, ON = (z - mu_o) / sd_o, (zn - mu_o) / sd_o

    m = ValueWM(O.shape[-1], u.shape[1], u.shape[2], ck["zdim"],
                encoder=ck.get("encoder", "mlp"))
    m.load_state_dict(ck["state_dict"]); m.eval()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    print(f"WM: encoder={ck.get('encoder')} zdim={ck['zdim']} "
          f"downstream AUC {ck.get('auc', float('nan')):.3f}")

    rows = []
    g = torch.Generator().manual_seed(0)
    for h in range(1, a.max_h + 1):
        seqs = build_sequences(ep, tt, h)
        if len(seqs) < 50:
            print(f"h={h}: only {len(seqs)} sequences, stopping"); break
        with torch.no_grad():
            z0 = m.encode(O[seqs[:, 0]])
            zt = z0
            for k in range(h):
                zt = m.step(zt, u[seqs[:, k]])
            ztrue = m.encode(ON[seqs[:, h - 1]])
            ident = float(((z0 - ztrue) ** 2).mean())
            pred = float(((zt - ztrue) ** 2).mean())
            zs = z0
            for k in range(h):
                perm = torch.randperm(len(seqs), generator=g)
                zs = m.step(zs, u[seqs[:, k]][perm])
            shuf = float(((zs - ztrue) ** 2).mean())
            v_imag = m.V(zt).squeeze(-1)
            v_real = m.V(ztrue).squeeze(-1)
            gap = float((v_imag - v_real).mean())
            corr = float(np.corrcoef(v_imag.numpy(), v_real.numpy())[0, 1])
        rows.append({"h": h, "n": len(seqs), "identity": ident, "pred": pred,
                     "shuffled": shuf, "rel": pred / ident,
                     "value_gap": gap, "value_corr": corr})
        print(f"h={h}: n={len(seqs):5d}  identity {ident:.4f}  pred {pred:.4f} "
              f"({pred/ident:.3f}x)  shuffled {shuf:.4f} ({shuf/ident:.3f}x)  "
              f"V gap {gap:+.4f}  corr(V_imag,V_real) {corr:+.3f}")

    ok1 = rows and rows[0]["rel"] < 1.0
    ok2 = rows and rows[0]["shuffled"] > rows[0]["pred"]
    print(f"\nbeats identity at h=1: {ok1}   uses the action at h=1: {ok2}")
    if rows:
        usable = [r["h"] for r in rows if r["rel"] < 1.0]
        print(f"horizons where the model still beats identity: "
              f"{usable if usable else 'none'}")
    print("A positive V gap means imagined states look better than the real states "
          "they predict - the surface a residual actor exploits.")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "wm": str(a.wm), "rows": rows, "env_steps": 0,
         "beats_identity_h1": bool(ok1), "uses_action_h1": bool(ok2),
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
