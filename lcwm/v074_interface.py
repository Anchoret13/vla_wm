"""V7.4A interface layer (registration: plan_and_progress/2026-08-09.md).

CenteredState — the registered primary interface statistics:
c_t = (Pool(z_t) - mu_train) / sigma_train, with mu_train the
uniform mean over the COMPLETE V7.4 policy-state set (all real
policy-anchor pools incl. abstention + all demo-rehearsal pools —
the caller supplies the set) and sigma_train ONE scalar, the
per-dim population RMS of the centered pools. Fitted once from the
frozen V7.4C checkpoint, saved in the run lineage, never
recomputed after the first policy forward.

TokenQueryPool — the registered fallback (only alternative): one
trainable query attends over the LCWM's 4 state tokens
(single-head scaled dot-product, d=384); the attended vector
replaces Pool(z) in the same centering and W_c flow boundary.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

CENTER_SCHEMA = "v074_center_v1"


class CenteredState:
    """Frozen centering statistics for the policy-state interface."""

    def __init__(self, mu: Tensor, sigma: float):
        self.mu = torch.as_tensor(mu).detach().float().reshape(-1)
        self.sigma = float(sigma)

    @classmethod
    def fit(cls, states) -> "CenteredState":
        """Uniform mean over the given states ([384] or [1,384]
        each); sigma is the ONE scalar per-dim population RMS of the
        centered states (the registered scale control — deliberately
        not a per-dimension std). The caller owns set composition:
        exactly one pool per REAL policy anchor (grounded + model
        eligible + abstention) plus the demo-rehearsal pools —
        never __reset__/__shuffled__ controls, text variants, or
        duplicated model:: aliases (uniform weighting makes any
        duplicate an implicit reweighting)."""
        rows = [torch.as_tensor(s).detach().float().reshape(-1)
                for s in states]
        if not rows:
            raise ValueError("CenteredState.fit: empty state set")
        stacked = torch.stack(rows).cpu()
        mu = stacked.mean(dim=0)
        # mean over states of ||s - mu||^2 / d == elementwise mean
        sigma = (stacked - mu).pow(2).mean().sqrt().item()
        if sigma < 1e-8:
            raise ValueError(
                f"CenteredState.fit: degenerate sigma {sigma:.3e}")
        return cls(mu, sigma)

    def apply(self, pool: Tensor) -> Tensor:
        """(pool - mu) / sigma, broadcast over leading dims."""
        mu = self.mu.to(device=pool.device, dtype=pool.dtype)
        return (pool - mu) / self.sigma

    def save(self, path) -> None:
        torch.save({"schema": CENTER_SCHEMA, "mu": self.mu.cpu(),
                    "sigma": self.sigma}, path)

    @classmethod
    def load(cls, path) -> "CenteredState":
        blob = torch.load(path, weights_only=False)
        if blob.get("schema") != CENTER_SCHEMA:
            raise ValueError(
                f"CenteredState.load: schema "
                f"{blob.get('schema')!r} != {CENTER_SCHEMA!r}")
        # the lineage file is the single trust boundary for every
        # later policy forward — mirror fit's validation here
        obj = cls(blob["mu"], blob["sigma"])
        if not torch.isfinite(obj.mu).all() or obj.sigma < 1e-8 \
                or not math.isfinite(obj.sigma):
            raise ValueError(
                f"CenteredState.load: corrupt statistics "
                f"(sigma {obj.sigma!r})")
        return obj


class TokenQueryPool(nn.Module):
    """Registered fallback pool: one trainable query, single-head
    scaled dot-product attention over the token axis; z: [B, T, d_z]
    -> attended [B, d_z], replacing z.mean(dim=1). No output
    projection here — the zero-init W_c downstream provides the
    zero-start guarantee.

    CACHING CONSTRAINT: the attended pool depends on the trainable
    query, so the V7.3 anchor_states.pt idiom (precompute pooled
    states once, detach, reuse for 300 steps) MUST NOT be applied to
    this module's output — a cached tqp(z).detach() trains the query
    for zero steps while everything else proceeds. Cache the
    pre-pool token states [1, T, d_z] instead and run this module
    inside the graph at every step."""

    def __init__(self, d_z: int = 384, seed: int | None = None):
        super().__init__()
        self.query = nn.Parameter(torch.empty(d_z))
        if seed is None:
            nn.init.normal_(self.query, std=0.02)
        else:
            # byte-identical cross-arm init independent of the
            # caller's global RNG position (V7.3D discipline)
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.query.copy_(torch.randn(d_z, generator=g)
                                 * 0.02)
        self.d_z = d_z

    def forward(self, z: Tensor) -> Tensor:
        if z.dim() < 3 or z.shape[-1] != self.d_z:
            raise ValueError(
                f"TokenQueryPool.forward: expected [..., T, "
                f"{self.d_z}], got {tuple(z.shape)} — a pooled "
                f"[B, d] input would silently softmax over the "
                f"batch axis")
        scores = (z @ self.query) / math.sqrt(self.d_z)
        attn = torch.softmax(scores, dim=-1)
        return (attn[..., None] * z).sum(dim=-2)


def _self_test() -> None:
    import tempfile
    torch.manual_seed(0)
    d = 384
    # ---- CenteredState: fit / apply roundtrip ------------------
    states = [torch.randn(d) * 3 + 1.5 for _ in range(40)]
    states += [torch.randn(1, d) * 3 + 1.5 for _ in range(10)]
    cs = CenteredState.fit(states)
    assert cs.mu.shape == (d,) and cs.mu.dtype == torch.float32
    assert isinstance(cs.sigma, float) and cs.sigma > 1e-8
    stacked = torch.stack([s.reshape(-1) for s in states])
    out = cs.apply(stacked)
    assert out.shape == stacked.shape
    assert out.device == stacked.device
    assert out.mean(dim=0).abs().max().item() < 1e-4
    assert abs(out.pow(2).mean().sqrt().item() - 1.0) < 1e-4
    # broadcasting over leading dims + dtype preservation
    batch = torch.randn(2, 5, d, dtype=torch.float64)
    ob = cs.apply(batch)
    assert ob.shape == batch.shape and ob.dtype == torch.float64
    assert torch.allclose(cs.apply(states[0]), out[0])
    # ---- degenerate fits raise ---------------------------------
    for bad in ([], [torch.ones(d)] * 3):
        try:
            CenteredState.fit(bad)
            raise AssertionError("fit accepted degenerate input")
        except ValueError:
            pass
    # ---- save / load with schema guard -------------------------
    with tempfile.TemporaryDirectory() as td:
        p = f"{td}/center.pt"
        cs.save(p)
        cs2 = CenteredState.load(p)
        assert torch.equal(cs.mu, cs2.mu)
        assert cs.sigma == cs2.sigma
        torch.save({"schema": "bogus", "mu": cs.mu, "sigma": 1.0}, p)
        try:
            CenteredState.load(p)
            raise AssertionError("load accepted wrong schema")
        except ValueError:
            pass
        torch.save({"schema": CENTER_SCHEMA, "mu": cs.mu,
                    "sigma": 0.0}, p)
        try:
            CenteredState.load(p)
            raise AssertionError("load accepted sigma=0")
        except ValueError:
            pass
    # ---- TokenQueryPool ----------------------------------------
    tqp = TokenQueryPool(d)
    z = torch.randn(6, 4, d)
    att = tqp(z)
    assert att.shape == (6, d) and att.dtype == z.dtype
    assert att.device == z.device
    # one token -> attention returns exactly that token
    assert torch.allclose(tqp(z[:, :1]), z[:, 0], atol=1e-6)
    att.sum().backward()
    g = tqp.query.grad
    assert g is not None and g.shape == (d,)
    assert g.norm().item() > 0
    # rank guard: pooled [B, d] input must be rejected loudly
    try:
        tqp(torch.randn(1, d))
        raise AssertionError("forward accepted a pooled [B, d]")
    except ValueError:
        pass
    # seeded init is deterministic regardless of global RNG position
    torch.manual_seed(1)
    q1 = TokenQueryPool(d, seed=7).query.detach().clone()
    torch.manual_seed(2)
    torch.randn(11)
    q2 = TokenQueryPool(d, seed=7).query.detach().clone()
    assert torch.equal(q1, q2)
    print("v074_interface self-test OK")


if __name__ == "__main__":
    _self_test()
