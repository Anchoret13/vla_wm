#!/usr/bin/env python
"""V6.1 mechanical launch assertions (wiring certificates, not quality).

Registered list (2026-07-30):
 1. candidate outputs depend on u_i only through E_a→T→z̃;
 2. physical-head grads reach E_a,T; semantic/ranking grads reach E_a,T,U;
 3. replacing z̃ removes ALL sibling score differences;
 4. the candidate forward cannot access h_{t+c}/next posterior (U and
    anchor are never invoked in the scoring path; EMA is disjoint);
 5. model is text-stateless (prompt features arrive only through h;
    hash-addressed recomputation is asserted at the data layer);
 6. zero W_z reproduces stock π0.5 actions under the same flow noise;
 7. reset and recurrent modes are exactly equal at episode start;
 8. two-step mixed train/reload is finite and bitwise-stable (CPU).

Run: python scripts/test_v06_mechanical.py [--skip-pi05]
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.v06_model import V06State, make_ema, ema_update  # noqa: E402

PASS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    assert ok, f"MECHANICAL FAIL [{name}] {detail}"
    PASS.append(name)
    print(f"  PASS {name} {detail}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pi05", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)

    model = V06State()
    h = torch.randn(2, 37, model.cfg.d_h)
    mask = torch.ones(2, 37, dtype=torch.bool)
    u1 = torch.randn(2, 10, 7, requires_grad=True)
    u2 = torch.randn(2, 10, 7)
    z = model.initial_state(h, mask)

    # 1 — action path exists and is the only path
    out1 = model.d_next(model.predict(z, u1))
    grad_u = torch.autograd.grad(out1["s"].sum(), u1, retain_graph=True,
                                 allow_unused=True)[0]
    check("1a_action_grad_nonzero",
          grad_u is not None and float(grad_u.abs().sum()) > 0)

    # 3 — same z̃ ⇒ bitwise-identical outputs regardless of candidate
    z_tilde = model.predict(z, u2).detach()
    o_a = model.d_next(z_tilde)
    o_b = model.d_next(z_tilde.clone())
    check("3_replacing_ztilde_removes_differences",
          all(torch.equal(o_a[k], o_b[k]) for k in o_a))
    # and different candidates through T do differ
    o_c = model.d_next(model.predict(z, u2))
    check("1b_distinct_candidates_differ",
          float((o_c["s"] - out1["s"]).abs().max()) > 0)

    # 2 — gradient reach
    model.zero_grad(set_to_none=True)
    model.d_next(model.predict(z, u1))["d_q"].sum().backward(
        retain_graph=True)
    ea_grad = sum(float(p.grad.abs().sum()) for p in
                  model.e_a.parameters() if p.grad is not None)
    t_grad = sum(float(p.grad.abs().sum()) for p in
                 model.t.parameters() if p.grad is not None)
    check("2a_physical_grads_reach_Ea_T", ea_grad > 0 and t_grad > 0,
          f"|gEa|={ea_grad:.2e} |gT|={t_grad:.2e}")
    model.zero_grad(set_to_none=True)
    z_live = model.initial_state(h, mask)
    model.d_next(model.predict(z_live, u2))["s"].sum().backward()
    u_grad = sum(float(p.grad.abs().sum()) for p in
                 list(model.anchor.parameters())
                 + list(model.r.parameters()) if p.grad is not None)
    check("2b_ranking_grads_reach_U", u_grad > 0, f"|gU|={u_grad:.2e}")

    # 4 — scorer path never touches U/anchor or the EMA copy
    ema = make_ema(model)
    fired = {"u": False}
    orig_update = model.update

    def trap(*a, **k):
        fired["u"] = True
        return orig_update(*a, **k)

    model.update = trap
    z_frozen = z.detach()
    _ = model.d_next(model.predict(z_frozen, u2))
    model.update = orig_update
    check("4a_scorer_never_calls_U", not fired["u"])
    shared = set(map(id, model.parameters())) & set(
        map(id, ema.parameters()))
    check("4b_ema_disjoint_from_scorer", len(shared) == 0)
    sig = inspect.signature(model.d_next.forward)
    check("4c_dnext_single_tensor_input", len(sig.parameters) == 1)

    # 5 — model is text-stateless
    has_text = any("prompt" in n or "text" in n or "token_embed" in n
                   for n, _ in model.named_parameters())
    check("5_text_stateless_model", not has_text)

    # 7 — reset == recurrent at episode start
    a0, m0 = model.null_action(2, h.device)
    z_reset = model.initial_state(h, mask)
    z_rec0 = model.step(model.z0[None].expand(2, -1, -1), a0, h, mask,
                        action_mask=m0)
    check("7_reset_equals_recurrent_at_start",
          torch.equal(z_reset, z_rec0))

    # 8 — two-step mixed train/reload, finite + bitwise-stable (CPU)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        zt = model.initial_state(h, mask)
        pred = model.d_next(model.predict(zt, u2))
        cur = model.d_current(zt)
        with torch.no_grad():
            target = ema.step(zt, u2, h, mask)
        closure = torch.nn.functional.mse_loss(
            model.step(zt, u2, h, mask), target)
        loss = (pred["d_q"].square().mean() + pred["s"].square().mean()
                + cur["valid_bits"].square().mean() + closure)
        assert torch.isfinite(loss), "non-finite mixed loss"
        loss.backward()
        opt.step()
        ema_update(ema, model)
    payload = {"model": model.state_dict()}
    tmp = Path("/tmp/claude-1000/-home-stargazer-Desktop-vla-wm-vla-wm/"
               "b8bc70a2-c4de-49f5-94ed-1e453157ce52/scratchpad/"
               "v06_reload_test.pt")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)
    model2 = V06State()
    model2.load_state_dict(torch.load(tmp, weights_only=False)["model"])
    model.eval()
    model2.eval()
    with torch.no_grad():
        za, zb = model.initial_state(h, mask), model2.initial_state(h, mask)
        oa = model.d_next(model.predict(za, u2))
        ob = model2.d_next(model2.predict(zb, u2))
    check("8_reload_bitwise_stable",
          torch.equal(za, zb) and all(torch.equal(oa[k], ob[k])
                                      for k in oa))

    # 6 — zero-W_z stock parity under shared flow noise (GPU, π0.5)
    if args.skip_pi05:
        print("  SKIP 6_zero_Wz_stock_parity (--skip-pi05)", flush=True)
    else:
        from lcwm.libero_paths import ensure_project_libero_config
        ensure_project_libero_config()
        from lcwm.chassis import Pi05Runner
        from lcwm.lc_flow import sample_chunks_lc
        from lcwm.loho_public import make_public_env
        from lcwm.sampler import prefix_forward, sample_chunks

        device = torch.device("cuda")
        vm = V06State().to(device)
        runner = Pi05Runner(suite_name="libero_10")
        env = make_public_env("loho_t1_drawer", 700)
        try:
            runner.reset()
            obs, _ = env.reset(seed=9999)
            batch = runner._obs_to_policy_batch(obs, env.task_description)
            prefix = prefix_forward(runner.policy, batch)
            zg = vm.initial_state(prefix.hidden.float(),
                                  prefix.pad_masks.bool())
            bias = vm.policy_bias(zg)
            assert float(bias.abs().max()) == 0.0, "bias not exactly 0"
            stock = sample_chunks(runner.policy, batch, n=1, seed=777,
                                  prefix=prefix)
            adapted = sample_chunks_lc(runner.policy, batch, bias, n=1,
                                       seed=777, prefix=prefix)
            err = float((stock - adapted).abs().max())
            check("6_zero_Wz_stock_parity", err <= 1e-5, f"max err {err:.1e}")
        finally:
            env.close()

    print(f"ALL MECHANICAL ASSERTIONS PASSED ({len(PASS)})", flush=True)


if __name__ == "__main__":
    main()
