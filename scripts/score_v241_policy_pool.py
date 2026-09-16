#!/usr/bin/env python
"""Blind, closed-loop policy-pool scoring for an evolving latent world model.

This script deliberately has no argument for deployed returns or success rates.
It seals rankings before any policy in the pool is evaluated in the environment.

The important coordinate contract is::

    z_model -> x_raw -> z_actor -> Delta_actor

An actor is always evaluated with the normalizer stored in its own checkpoint.
The frozen base-policy prior and value head come from base-policy deployment data
only and are shared by every model arm.  Therefore the only difference between
``stale`` and ``updated`` is the learned transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402


OUT = REPO / "results" / "v241_policy_score"
MODEL_ARMS = ("stale", "base_continue", "updated", "updated_shuffled")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_prior(zdim: int, c: int, adim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(zdim),
        nn.Linear(zdim, 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, c * adim),
    )


def make_phi(zdim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 1)
    )


def model_to_raw(z_model: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return z_model * sd + mu


def raw_to_model(x_raw: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (x_raw - mu) / sd


def actor_delta_from_raw(x_raw: torch.Tensor, actor: nn.Module,
                         actor_mu: torch.Tensor, actor_sd: torch.Tensor) -> torch.Tensor:
    """Apply an actor exactly as deployment does, irrespective of WM coordinates."""
    return actor((x_raw - actor_mu) / actor_sd)


def initial_raw_states(tapes: list[Path], task: str, zdim: int,
                       limit: int | None = None) -> torch.Tensor:
    """Read initial latent states without consulting outcome summaries."""
    rows: list[torch.Tensor] = []
    for tape in tapes:
        d = torch.load(tape, weights_only=False, map_location="cpu")
        if d.get("task") != task:
            raise ValueError(f"{tape}: task={d.get('task')!r}, expected {task!r}")
        if int(d.get("latent_dim", d["z"].shape[-1])) != zdim:
            raise ValueError(f"{tape}: latent dim mismatch")
        if d["z"].shape[-1] != zdim:
            raise ValueError(f"{tape}: z tensor dim mismatch")
        ep, tt = d["episode"].long(), d["t"].long()
        for e in sorted(set(ep.tolist())):
            idx = torch.nonzero(ep == e).flatten()
            first = idx[torch.argmin(tt[idx])]
            rows.append(d["z"][first].detach().float().cpu())
    if not rows:
        raise ValueError("no initial states found")
    out = torch.stack(rows)
    if limit is not None:
        out = out[:limit]
    return out


def load_actor(path: Path, *, task: str, dims: tuple[int, int, int],
               device: torch.device = torch.device("cpu")) -> dict[str, Any]:
    ck = torch.load(path, weights_only=False, map_location="cpu")
    zdim, c, adim = dims
    if ck.get("task") != task:
        raise ValueError(f"{path}: task mismatch")
    if tuple(ck.get("dims", ())) != dims:
        raise ValueError(f"{path}: dims={ck.get('dims')}, expected {dims}")
    actor = ResidualActor(zdim, c, adim, scale=float(ck["scale"]))
    actor.load_state_dict(ck["state_dict"])
    actor.to(device).eval()
    return {
        "path": path,
        "sha256": sha256_file(path),
        "checkpoint": ck,
        "actor": actor,
        "mu": ck["mu"].detach().float().to(device),
        "sd": ck["sd"].detach().float().to(device),
        "scale": float(ck["scale"]),
    }


@torch.no_grad()
def score_policy(model: RawLatentWM, prior: nn.Module, phi: nn.Module,
                 starts_raw: torch.Tensor, model_mu: torch.Tensor,
                 model_sd: torch.Tensor, actor_info: dict[str, Any],
                 *, horizon: int, gamma: float,
                 use_residual: bool = True, use_transition: bool = True) -> dict[str, float]:
    """Score one sustained policy under one transition model."""
    z = raw_to_model(starts_raw, model_mu, model_sd)
    c = int(actor_info["checkpoint"]["c"])
    adim = int(actor_info["checkpoint"]["adim"])
    zero_u = torch.zeros(len(z), c, adim, device=z.device, dtype=z.dtype)
    b = model.step(
        torch.zeros(len(z), model.bdim, device=z.device, dtype=z.dtype),
        model.enc(z),
        zero_u,
    )
    value = torch.zeros(len(z), device=z.device, dtype=z.dtype)
    delta_mags: list[torch.Tensor] = []

    if not use_transition:
        # Strict no-roll control. It has no policy-dependent input by design.
        return {
            "score": float(phi(z).squeeze(-1).mean()),
            "mean_abs_delta": 0.0,
        }

    for h in range(horizon):
        x_raw = model_to_raw(z, model_mu, model_sd)
        base_u = prior(z).view(-1, c, adim)
        if use_residual:
            delta = actor_delta_from_raw(
                x_raw, actor_info["actor"], actor_info["mu"], actor_info["sd"]
            )
        else:
            delta = torch.zeros_like(base_u)
        u = base_u + delta
        delta_mags.append(delta.abs().mean())
        z = model.predict(b, z, u)
        b = model.step(b, model.enc(z), u)
        value = value + (gamma ** h) * phi(z).squeeze(-1)
    return {
        "score": float(value.mean()),
        "mean_abs_delta": float(torch.stack(delta_mags).mean()),
    }


def rank_order(rows: list[dict[str, Any]], key: str) -> list[int]:
    return [int(r["k"]) for r in sorted(rows, key=lambda r: (-float(r[key]), int(r["k"])))]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pair", type=Path, required=True)
    ap.add_argument("--actor-dir", type=Path, required=True)
    ap.add_argument("--start-tapes", type=Path, nargs="+", required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-starts", type=int, default=192)
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    if a.horizon < 1:
        raise ValueError("--horizon must be positive")
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ck = torch.load(a.model_pair, weights_only=False, map_location="cpu")
    dims = tuple(int(x) for x in ck["dims"])
    if len(dims) != 3:
        raise ValueError(f"invalid dims: {dims}")
    zdim, c, adim = dims
    if ck.get("task") != a.task:
        raise ValueError(f"model task={ck.get('task')!r}, requested {a.task!r}")
    missing = [name for name in MODEL_ARMS if name not in ck["models"]]
    if missing:
        raise ValueError(f"model-pair checkpoint missing arms: {missing}")

    model_mu = ck["mu"].detach().float().to(device)
    model_sd = ck["sd"].detach().float().to(device)
    if model_mu.shape != (zdim,) or model_sd.shape != (zdim,):
        raise ValueError("invalid model normalizer")
    if not torch.isfinite(model_sd).all() or not bool((model_sd > 0).all()):
        raise ValueError("non-positive/non-finite model scale")

    prior = make_prior(zdim, c, adim)
    prior.load_state_dict(ck["prior"])
    prior.to(device).eval()
    phi = make_phi(zdim)
    phi.load_state_dict(ck["phi"])
    phi.to(device).eval()
    models: dict[str, RawLatentWM] = {}
    for name in MODEL_ARMS:
        m = RawLatentWM(zdim, c, adim)
        m.load_state_dict(ck["models"][name])
        m.to(device).eval()
        models[name] = m

    starts = initial_raw_states(a.start_tapes, a.task, zdim, a.max_starts).to(device)
    actor_paths = sorted(
        a.actor_dir.glob("actor_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not actor_paths:
        raise ValueError(f"no actor_*.pt files in {a.actor_dir}")
    actors = [load_actor(p, task=a.task, dims=dims, device=device)
              for p in actor_paths]
    scales = {round(x["scale"], 12) for x in actors}
    if len(scales) != 1:
        raise ValueError(
            f"pool is not fixed-scale ({sorted(scales)}); v241 requires the norm confound removed"
        )

    rows: list[dict[str, Any]] = []
    for info in actors:
        k = int(info["path"].stem.split("_")[-1])
        row: dict[str, Any] = {
            "k": k,
            "actor": str(info["path"]),
            "actor_sha256": info["sha256"],
            "scale": info["scale"],
        }
        for name, model in models.items():
            scored = score_policy(
                model, prior, phi, starts, model_mu, model_sd, info,
                horizon=a.horizon, gamma=a.gamma,
            )
            row[f"{name}_score"] = scored["score"]
            row[f"{name}_mean_abs_delta"] = scored["mean_abs_delta"]
        strict = score_policy(
            models["updated"], prior, phi, starts, model_mu, model_sd, info,
            horizon=a.horizon, gamma=a.gamma, use_transition=False,
        )
        blind = score_policy(
            models["updated"], prior, phi, starts, model_mu, model_sd, info,
            horizon=a.horizon, gamma=a.gamma, use_residual=False,
        )
        row["no_roll_score"] = strict["score"]
        row["action_blind_score"] = blind["score"]
        rows.append(row)

    rows.sort(key=lambda r: int(r["k"]))
    no_roll_values = [round(float(r["no_roll_score"]), 12) for r in rows]
    blind_values = [round(float(r["action_blind_score"]), 12) for r in rows]
    if len(set(no_roll_values)) != 1:
        raise AssertionError("strict no-roll control spuriously distinguishes policies")
    if len(set(blind_values)) != 1:
        raise AssertionError("strict action-blind control spuriously distinguishes policies")

    orders = {name: rank_order(rows, f"{name}_score") for name in MODEL_ARMS}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = a.out or (OUT / f"{a.task}_{stamp}")
    out.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "v241_policy_pool_score_v1",
        "utc": stamp,
        "task": a.task,
        "horizon": a.horizon,
        "gamma": a.gamma,
        "starts": len(starts),
        "fixed_scale": next(iter(scales)),
        "device": str(device),
        "deployed_outcomes_read": False,
        "model_pair": str(a.model_pair),
        "model_pair_sha256": sha256_file(a.model_pair),
        "start_tapes": [
            {"path": str(p), "sha256": sha256_file(p)} for p in a.start_tapes
        ],
        "orders": orders,
        "rows": rows,
        "provenance": {
            "argv": sys.argv,
            "git": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO,
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
            "code_sha256": sha256_file(Path(__file__)),
            "model_pair_provenance": ck.get("provenance"),
        },
    }
    (out / "ranking.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"sealed blind rankings for {len(rows)} fixed-scale policies")
    for name in MODEL_ARMS:
        vals = [float(r[f"{name}_score"]) for r in rows]
        print(f"  {name:18s}: {orders[name]}  spread={max(vals)-min(vals):.6f}")
    print(f"  no_roll/action_blind policy-invariant: yes")
    print(f"-> {out / 'ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
