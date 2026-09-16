#!/usr/bin/env python
"""Mechanical CPU tests for scripts/train_v251_interface.py.

    /home/stargazer/miniconda3/envs/vf0s/bin/python -m pytest scripts/test_v251_interface.py

Every fixture is synthetic: no LIBERO, no pi0.5, no GPU, no results/ artifact is read.
The fixtures are built so the label -> Phi' -> latent chain is exactly the one the round
uses - labels.pt is passed through lcwm/v250_progress.tape_progress, and the boundary
latents are then constructed so that z_next[k] == z[k+1] holds to the byte, which is the
continuity the loader enforces on a real v250 tape.

WHAT EACH TEST PINS, and what a failure would mean:

  test_predicted_advantage_is_the_analytic_value
      the imagined roll really applies the candidate residual at EVERY step. With a
      transition whose predict() adds sum(u) to the latent and a head that reads
      coordinate 0, A must equal sum over the H-window of the candidate delta. A
      failure means the sustained rollout is not sustained, or the base/candidate rolls
      do not share (z_t, b_t).
  test_observed_uses_only_real_boundaries / test_current_is_invariant_*
      the two controls are what they are declared to be: 'observed' is the realised
      Phi' difference over the same rows, 'current' does not move when the actions or
      the horizon change. A failure means a control secretly consults the future or
      the action, which is the exact defect that would make a between-arm difference
      uninterpretable.
  test_identical_arms_give_bit_identical_actors
      nothing except the learned transition differs between two runs. A failure means
      an M1-vs-M0 difference could come from RNG, row order or the head.
  test_phi_head_is_bit_identical_across_arms
      the Phi' head is shared. A failure means the arms are being scored by different
      heads and no comparison between them is valid.
  test_zero_delta_filter_removes_exactly_the_zero_rows
      the documented row policy is the implemented one.
  test_checkpoint_round_trips
      what run_v206 reloads is what was trained (the coordinate contract).
  test_latent_dim_mismatch_fails_before_training
      an 8217-d tape against a 2073-d model pair raises instead of being silently
      dropped the way train_v205_action_belief.load_episodes:104 drops it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_v241_model_pair import make_prior  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402
from train_v251_interface import (  # noqa: E402
    Bundle,
    build_bundle,
    compute_advantage,
    eligible_rows,
    fit_phi_head,
    load_run,
    main,
    make_phi_head,
    predicted_advantage,
    state_dict_digest,
    window_actions,
)

TASK = "chain3_lr2"
ZDIM, C, ADIM = 16, 2, 3
N_EP, N_CHUNKS = 4, 12


# --------------------------------------------------------------------------- fixtures


def _labels(n_ep: int, n_chunks: int, c: int, seed: int) -> dict:
    """A labels.pt whose geometry gives a monotone, per-episode-different Phi'.

    One goal atom, two bodies. The gripper walks toward the object at an
    episode-specific speed and the predicate never fires, so Phi' stays inside the
    approach/grasp band, which is where chain3's stall lives.
    """
    g = torch.Generator().manual_seed(seed)
    boundaries = n_chunks + 1
    eef, obj, bits, ep, tt = [], [], [], [], []
    for e in range(n_ep):
        speed = 0.3 + 0.5 * float(torch.rand((), generator=g))
        target = torch.tensor([0.5, 0.0, 0.10])
        for k in range(boundaries):
            frac = min(1.0, speed * k / max(n_chunks - 1, 1))
            pos = torch.tensor([0.0, 0.0, 0.5]) * (1 - frac) + target * frac
            q = torch.zeros(25)
            q[0:3] = pos
            eef.append(q)
            obj.append(torch.stack([target, torch.tensor([0.0, 0.0, 0.10])]))
            bits.append(torch.zeros(1, dtype=torch.bool))
            ep.append(e)
            tt.append(k * c)
    return {
        "eef_proprio": torch.stack(eef),
        "obj_pos": torch.stack(obj),
        "bits": torch.stack(bits),
        "episode": torch.tensor(ep),
        "t": torch.tensor(tt),
        "object_names": ["alphabet_soup_1", "basket_1"],
        "goal_atoms": [["In", "alphabet_soup_1", "basket_1_contain_region"]],
        "milestones": ["pick", "place"],
        "events": {},
        "task": TASK,
        "c": c,
    }


def write_run(path: Path, *, seed: int, zdim: int = ZDIM, c: int = C, adim: int = ADIM,
              n_ep: int = N_EP, n_chunks: int = N_CHUNKS,
              residual_start: int | None = None, scale: float = 0.05) -> Path:
    """Write a v250-shaped run directory: tape.pt + labels.pt + trace.pt."""
    from lcwm.v250_progress import tape_progress

    path.mkdir(parents=True, exist_ok=True)
    labels = _labels(n_ep, n_chunks, c, seed)
    progress = tape_progress(labels)
    g = torch.Generator().manual_seed(seed + 977)

    z, zn, u, ub, dl, ep, tt = [], [], [], [], [], [], []
    for e in range(n_ep):
        rows = torch.nonzero(labels["episode"] == e).flatten()
        rows = rows[torch.argsort(labels["t"][rows])]
        phi_b = progress["phi"][rows]
        zb = 0.1 * torch.randn(n_chunks + 1, zdim, generator=g)
        zb[:, 0] = phi_b                      # a latent the Phi' head can actually read
        base = torch.randn(n_chunks, c, adim, generator=g)
        delta = torch.zeros(n_chunks, c, adim)
        if residual_start is not None:
            k0 = residual_start
            delta[k0:] = scale * torch.randn(n_chunks - k0, c, adim, generator=g)
        z.append(zb[:-1])
        zn.append(zb[1:])
        ub.append(base)
        dl.append(delta)
        u.append(base + delta)
        ep.append(torch.full((n_chunks,), e, dtype=torch.long))
        tt.append(torch.arange(n_chunks, dtype=torch.long) * c)

    episode, times = torch.cat(ep), torch.cat(tt)
    torch.save({"z": torch.cat(z), "u": torch.cat(u), "z_next": torch.cat(zn),
                "episode": episode, "t": times,
                "sigma": torch.zeros(len(episode)), "task": TASK, "c": c,
                "latent_dim": zdim, "proprio_dim": 25}, path / "tape.pt")
    torch.save(labels, path / "labels.pt")
    torch.save({"u_base": torch.cat(ub), "delta": torch.cat(dl),
                "episode": episode, "t": times}, path / "trace.pt")
    return path


def write_model_pair(path: Path, *, zdim: int = ZDIM, c: int = C, adim: int = ADIM,
                     identical_arms: bool = False, seed: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    for i, name in enumerate(("stale", "base_continue", "updated", "updated_shuffled")):
        torch.manual_seed(seed if identical_arms else seed + i)
        models[name] = {k: v.detach().cpu()
                        for k, v in RawLatentWM(zdim, c, adim).state_dict().items()}
    torch.manual_seed(seed + 100)
    prior = {k: v.detach().cpu()
             for k, v in make_prior(zdim, c, adim).state_dict().items()}
    phi = {k: v.detach().cpu() for k, v in make_phi_head(zdim).state_dict().items()}
    torch.save({"models": models, "prior": prior, "phi": phi,
                "dims": (zdim, c, adim), "mu": torch.zeros(zdim),
                "sd": torch.ones(zdim), "task": TASK, "sequence_len": 8,
                "metrics": {}, "provenance": {"synthetic": True}}, path)
    return path


def small_run_args(pair: Path, run: Path, out_root: Path, arm: str, advantage: str,
                   **extra: object) -> list[str]:
    argv = ["--model-pair", str(pair), "--wm-arm", arm, "--tapes", str(run),
            "--task", TASK, "--advantage", advantage, "--horizon", "3",
            "--actor-epochs", "20", "--batch", "16", "--restarts", "1",
            "--phi-steps", "20", "--phi-batch", "32", "--device", "cpu",
            # --scale is REQUIRED with no default: a forgotten scale silently trains
            # an interface an order of magnitude below the threshold for any effect
            "--scale", "0.05",
            "--out-root", str(out_root)]
    for key, value in extra.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return argv


def only_run_dir(out_root: Path) -> Path:
    dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    assert len(dirs) == 1, dirs
    return dirs[0]


# ------------------------------------------------------------- hand-built transition


class StubWM(nn.Module):
    """predict(b, z, u) = z + sum(u), broadcast over every latent coordinate.

    Belief and encoder are inert, so the H-step roll is exactly z0 + sum over the
    window of every executed action, which makes the advantage analytic.
    """

    bdim = 4

    def enc(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 8)

    def step(self, b: torch.Tensor, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return b

    def predict(self, b: torch.Tensor, z: torch.Tensor,
                u: torch.Tensor) -> torch.Tensor:
        return z + u.sum(dim=(1, 2)).unsqueeze(-1)


class FirstCoordinate(nn.Module):
    """phi_hat(z) = z[:, 0], so the head introduces no error of its own."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :1]


def loaded_bundle(tmp_path: Path, *, residual_start: int | None = 4) -> Bundle:
    run = write_run(tmp_path / "run", seed=11, residual_start=residual_start)
    episodes, _record = load_run(run, TASK, (ZDIM, C, ADIM))
    return build_bundle(episodes, torch.zeros(ZDIM), torch.ones(ZDIM))


# ----------------------------------------------------------------------------- tests


def test_loader_rejects_a_trace_that_does_not_audit(tmp_path: Path) -> None:
    run = write_run(tmp_path / "run", seed=3, residual_start=2)
    trace = torch.load(run / "trace.pt", weights_only=False)
    trace["delta"] = trace["delta"] + 1.0
    torch.save(trace, run / "trace.pt")
    with pytest.raises(ValueError, match="u_base . delta != u_exec"):
        load_run(run, TASK, (ZDIM, C, ADIM))


def test_boundary_latents_cover_every_chunk(tmp_path: Path) -> None:
    """No truncation: a 12-chunk episode yields 12 rows and 13 boundary latents."""
    bundle = loaded_bundle(tmp_path)
    assert bundle.n_episodes == N_EP
    assert bundle.n_rows == N_EP * N_CHUNKS
    assert int(bundle.z_bound.shape[0]) == N_EP * (N_CHUNKS + 1)
    assert set(bundle.ep_len.tolist()) == {N_CHUNKS}


def test_predicted_advantage_is_the_analytic_value(tmp_path: Path) -> None:
    bundle = loaded_bundle(tmp_path)
    horizon = 3
    rows, _ = eligible_rows(bundle, horizon, include_zero_delta=True)
    ub = window_actions(bundle.u_base, rows, horizon)
    dl = window_actions(bundle.delta, rows, horizon)
    z0 = bundle.z_bound[bundle.row_boundary[rows]]
    b0 = torch.zeros(len(rows), StubWM.bdim)

    got = predicted_advantage(StubWM(), FirstCoordinate(), z0, b0, ub, dl, horizon)
    want = dl.sum(dim=(1, 2, 3))
    assert torch.allclose(got, want, atol=1e-5)
    # and the base roll alone carries no candidate: a zero delta gives exactly zero
    zero = predicted_advantage(StubWM(), FirstCoordinate(), z0, b0, ub,
                               torch.zeros_like(dl), horizon)
    assert torch.equal(zero, torch.zeros_like(zero))


def test_predicted_advantage_is_sustained_not_single_chunk(tmp_path: Path) -> None:
    """Only the first window step matters iff the roll is not sustained; it is not."""
    bundle = loaded_bundle(tmp_path)
    horizon = 3
    rows, _ = eligible_rows(bundle, horizon, include_zero_delta=True)
    ub = window_actions(bundle.u_base, rows, horizon)
    dl = window_actions(bundle.delta, rows, horizon)
    z0 = bundle.z_bound[bundle.row_boundary[rows]]
    b0 = torch.zeros(len(rows), StubWM.bdim)
    first_only = dl.clone()
    first_only[:, 1:] = 0.0
    full = predicted_advantage(StubWM(), FirstCoordinate(), z0, b0, ub, dl, horizon)
    head = predicted_advantage(StubWM(), FirstCoordinate(), z0, b0, ub, first_only,
                               horizon)
    assert not torch.allclose(full, head, atol=1e-6)


def test_observed_uses_only_real_boundaries(tmp_path: Path) -> None:
    bundle = loaded_bundle(tmp_path)
    horizon = 3
    rows, _ = eligible_rows(bundle, horizon, include_zero_delta=True)
    got = compute_advantage("observed", bundle, rows, horizon)
    b_now = bundle.row_boundary[rows]
    want = bundle.phi_bound[b_now + horizon] - bundle.phi_bound[b_now]
    assert torch.equal(got, want)
    # it does not move when the actions do
    perturbed = Bundle(**{**bundle.__dict__,
                          "u": torch.randn_like(bundle.u),
                          "u_base": torch.randn_like(bundle.u_base),
                          "delta": torch.randn_like(bundle.delta)})
    assert torch.equal(compute_advantage("observed", perturbed, rows, horizon), got)


def test_current_is_invariant_to_action_and_horizon(tmp_path: Path) -> None:
    bundle = loaded_bundle(tmp_path)
    rows, _ = eligible_rows(bundle, 5, include_zero_delta=True)
    a3 = compute_advantage("current", bundle, rows, 3)
    a5 = compute_advantage("current", bundle, rows, 5)
    assert torch.equal(a3, a5)
    assert torch.equal(a3, bundle.phi_bound[bundle.row_boundary[rows]])
    perturbed = Bundle(**{**bundle.__dict__,
                          "u": torch.randn_like(bundle.u),
                          "u_base": torch.randn_like(bundle.u_base),
                          "delta": torch.randn_like(bundle.delta)})
    assert torch.equal(compute_advantage("current", perturbed, rows, 3), a3)


def test_the_three_modes_are_distinguishable(tmp_path: Path) -> None:
    bundle = loaded_bundle(tmp_path)
    horizon = 3
    rows, _ = eligible_rows(bundle, horizon, include_zero_delta=True)
    ub = window_actions(bundle.u_base, rows, horizon)
    dl = window_actions(bundle.delta, rows, horizon)
    z0 = bundle.z_bound[bundle.row_boundary[rows]]
    b0 = torch.zeros(len(rows), StubWM.bdim)
    predicted = predicted_advantage(StubWM(), FirstCoordinate(), z0, b0, ub, dl, horizon)
    observed = compute_advantage("observed", bundle, rows, horizon)
    current = compute_advantage("current", bundle, rows, horizon)
    assert not torch.allclose(predicted, observed, atol=1e-4)
    assert not torch.allclose(observed, current, atol=1e-4)


def test_zero_delta_filter_removes_exactly_the_zero_rows(tmp_path: Path) -> None:
    start = 4
    bundle = loaded_bundle(tmp_path, residual_start=start)
    horizon = 3
    kept, stats = eligible_rows(bundle, horizon, include_zero_delta=False)
    every, stats_all = eligible_rows(bundle, horizon, include_zero_delta=True)
    windows_per_ep = N_CHUNKS - horizon + 1
    assert stats_all["windows_kept"] == N_EP * windows_per_ep
    assert stats["windows"] == stats_all["windows"]
    # zero-delta windows are exactly the chunks before --residual-start-chunk
    assert stats["windows_zero_delta"] == N_EP * start
    assert stats["windows_kept"] == N_EP * (windows_per_ep - start)
    assert bool((bundle.delta[kept].abs().amax(dim=(1, 2)) > 0).all())
    dropped = sorted(set(every.tolist()) - set(kept.tolist()))
    assert all(float(bundle.delta[r].abs().max()) == 0.0 for r in dropped)
    assert bundle.row_chunk[kept].min() >= start


def test_a_base_only_tape_fails_loudly(tmp_path: Path) -> None:
    """Every row zero-delta: refuse rather than silently fit the zero residual."""
    bundle = loaded_bundle(tmp_path, residual_start=None)
    with pytest.raises(ValueError, match="zero-delta filter"):
        eligible_rows(bundle, 3, include_zero_delta=False)


def test_identical_arms_give_bit_identical_actors(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt", identical_arms=True)
    run = write_run(tmp_path / "run", seed=21, residual_start=3)
    digests = []
    for arm, tag in (("stale", "a"), ("updated", "b")):
        root = tmp_path / f"out_{tag}"
        assert main(small_run_args(pair, run, root, arm, "predicted")) == 0
        ck = torch.load(only_run_dir(root) / "actor_0.pt", weights_only=False)
        digests.append(state_dict_digest(ck["state_dict"]))
    assert digests[0] == digests[1]


def test_different_arms_give_different_actors(tmp_path: Path) -> None:
    """The counterpart: with different transitions the interface must differ."""
    pair = write_model_pair(tmp_path / "pair.pt", identical_arms=False)
    run = write_run(tmp_path / "run", seed=22, residual_start=3)
    digests = []
    for arm, tag in (("stale", "a"), ("updated", "b")):
        root = tmp_path / f"out_{tag}"
        assert main(small_run_args(pair, run, root, arm, "predicted")) == 0
        ck = torch.load(only_run_dir(root) / "actor_0.pt", weights_only=False)
        digests.append(state_dict_digest(ck["state_dict"]))
    assert digests[0] != digests[1]


def test_phi_head_is_bit_identical_across_arms(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt", identical_arms=False)
    run = write_run(tmp_path / "run", seed=23, residual_start=3)
    heads = []
    for arm, tag in (("stale", "a"), ("updated_shuffled", "b")):
        root = tmp_path / f"out_{tag}"
        assert main(small_run_args(pair, run, root, arm, "predicted")) == 0
        out = only_run_dir(root)
        head = torch.load(out / "phi_head.pt", weights_only=False)
        ck = torch.load(out / "actor_0.pt", weights_only=False)
        assert head["digest"] == ck["phi_head_digest"]
        heads.append(head["digest"])
    assert heads[0] == heads[1]


def test_assert_phi_digest_rejects_a_foreign_head(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=24, residual_start=3)
    argv = small_run_args(pair, run, tmp_path / "out", "updated", "predicted",
                          assert_phi_digest="0" * 64)
    with pytest.raises(RuntimeError, match="assert-phi-digest"):
        main(argv)


def test_checkpoint_round_trips(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=25, residual_start=3)
    root = tmp_path / "out"
    assert main(small_run_args(pair, run, root, "updated", "predicted")) == 0
    path = only_run_dir(root) / "actor_0.pt"
    ck = torch.load(path, weights_only=False)

    # exactly the fields run_v206 reads off a rawwm checkpoint
    for key in ("state_dict", "zdim", "condition", "c", "adim", "scale", "rawwm",
                "dims", "mu", "sd", "bmu", "bsd", "task", "arm"):
        assert key in ck, key
    assert ck["condition"] == "raw"
    assert tuple(ck["dims"]) == (ZDIM, C, ADIM)
    assert ck["deployed_outcomes_read"] is False

    actor = ResidualActor(ck["zdim"], ck["c"], ck["adim"], scale=ck["scale"])
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    wm = RawLatentWM(*ck["dims"])
    wm.load_state_dict(ck["rawwm"], strict=True)

    z = torch.linspace(-1.0, 1.0, ZDIM).unsqueeze(0)
    with torch.no_grad():
        first = actor(z)
        second = actor(z)
    assert torch.equal(first, second)
    reloaded = ResidualActor(ck["zdim"], ck["c"], ck["adim"], scale=ck["scale"])
    reloaded.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        assert torch.equal(reloaded(z), first)
    assert float(first.abs().max()) <= float(ck["scale"]) + 1e-6

    # the actor's normaliser IS the model pair's: the coordinate path is the identity
    pair_ck = torch.load(pair, weights_only=False)
    assert torch.equal(ck["mu"], pair_ck["mu"]) and torch.equal(ck["sd"], pair_ck["sd"])


def test_latent_dim_mismatch_fails_before_training(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt", zdim=ZDIM)
    run = write_run(tmp_path / "run", seed=26, zdim=ZDIM + 5, residual_start=3)
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="model-pair dims"):
        main(small_run_args(pair, run, root, "updated", "predicted"))
    assert not root.exists(), "an output directory was created before the dim check"


def test_observed_arm_never_touches_a_transition(tmp_path: Path) -> None:
    """The no-rollout control runs with model=None in its own advantage path."""
    bundle = loaded_bundle(tmp_path)
    rows, _ = eligible_rows(bundle, 3, include_zero_delta=True)
    got = compute_advantage("observed", bundle, rows, 3, model=None, phi_head=None,
                            beliefs=None)
    assert int(got.numel()) == int(rows.numel())
    with pytest.raises(ValueError, match="needs a model"):
        compute_advantage("predicted", bundle, rows, 3, model=None)


def test_summary_records_the_inputs_and_no_outcome(tmp_path: Path) -> None:
    import json

    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=27, residual_start=3)
    root = tmp_path / "out"
    assert main(small_run_args(pair, run, root, "stale", "observed")) == 0
    summary = json.loads((only_run_dir(root) / "summary.json").read_text())
    assert summary["deployed_outcomes_read"] is False
    assert summary["outcome_json_read"] is False
    assert summary["env_steps"] == 0
    assert summary["wm_arm"] == "stale" and summary["advantage"] == "observed"
    assert len(summary["model_pair_sha256"]) == 64
    assert summary["runs"][0]["tape_sha256"] and summary["runs"][0]["labels_sha256"]
    assert summary["runs"][0]["trace_sha256"]
    assert len(summary["phi_head"]["digest"]) == 64
    assert "holdout_corr" in summary["phi_head"]
    assert summary["advantage_distribution"]["n"] > 0
    assert 0.0 <= summary["actors"][0]["fraction_of_bound"] <= 1.0


def test_base_action_prior_branch_runs_and_changes_the_advantage(tmp_path: Path) -> None:
    """--base-action prior is implemented, not a silent fallback to 'recorded'."""
    import json

    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=28, residual_start=3)
    stats = []
    for mode in ("recorded", "prior"):
        root = tmp_path / f"out_{mode}"
        argv = small_run_args(pair, run, root, "updated", "predicted",
                              base_action=mode)
        assert main(argv) == 0
        summary = json.loads((only_run_dir(root) / "summary.json").read_text())
        assert summary["base_action"] == mode
        stats.append(summary["advantage_distribution"])
    assert stats[0]["mean"] != stats[1]["mean"]


def test_phi_readout_head_runs(tmp_path: Path) -> None:
    import json

    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=29, residual_start=3)
    root = tmp_path / "out"
    argv = small_run_args(pair, run, root, "stale", "observed", phi_readout="head")
    assert main(argv) == 0
    summary = json.loads((only_run_dir(root) / "summary.json").read_text())
    assert summary["phi_readout"] == "head"
    assert summary["advantage_distribution"]["n"] > 0


def test_include_zero_delta_keeps_every_window(tmp_path: Path) -> None:
    import json

    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=30, residual_start=4)
    root = tmp_path / "out"
    argv = small_run_args(pair, run, root, "updated", "predicted",
                          include_zero_delta=True)
    assert main(argv) == 0
    summary = json.loads((only_run_dir(root) / "summary.json").read_text())
    assert summary["include_zero_delta"] is True
    assert summary["rows"]["windows_kept"] == summary["rows"]["windows"]
    assert summary["rows"]["windows_zero_delta"] == N_EP * 4


def test_a_reused_phi_head_is_the_same_head(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=31, residual_start=3)
    first_root = tmp_path / "out_a"
    assert main(small_run_args(pair, run, first_root, "updated", "predicted")) == 0
    head_path = only_run_dir(first_root) / "phi_head.pt"
    digest = torch.load(head_path, weights_only=False)["digest"]

    second_root = tmp_path / "out_b"
    argv = small_run_args(pair, run, second_root, "stale", "observed",
                          phi_head=head_path, assert_phi_digest=digest)
    assert main(argv) == 0
    reused = torch.load(only_run_dir(second_root) / "phi_head.pt", weights_only=False)
    assert reused["digest"] == digest
    assert reused["metrics"]["reused_from"] == str(head_path.resolve())


def test_chain3_geometry_is_not_truncated_to_24_chunks(tmp_path: Path) -> None:
    """chain3@750 is 75 chunks at c=10; `L = min(min_chunks, 24)` would keep 24.

    The stall this round is about starts near chunk 26 (median last-progress step 260
    of 750), so a 24-chunk truncation would discard every informative row.
    """
    run = write_run(tmp_path / "run", seed=41, zdim=32, c=10, adim=7, n_ep=3,
                    n_chunks=75, residual_start=26)
    episodes, record = load_run(run, TASK, (32, 10, 7))
    assert record["chunks_per_episode"] == [75]
    bundle = build_bundle(episodes, torch.zeros(32), torch.ones(32))
    assert bundle.n_rows == 3 * 75
    assert int(bundle.z_bound.shape[0]) == 3 * 76
    rows, stats = eligible_rows(bundle, 5, include_zero_delta=False)
    assert stats["windows"] == 3 * 71           # k = 0..70 for H = 5
    assert stats["windows_zero_delta"] == 3 * 26
    assert stats["windows_kept"] == 3 * 45
    assert int(bundle.row_chunk[rows].max()) == 70


def test_a_phi_head_from_another_normaliser_is_refused(tmp_path: Path) -> None:
    """--assert-phi-digest pins WHICH head; it does not pin which space it belongs to.

    The head reads normalised latents. Reusing one fitted under a different mu/sd
    evaluates it in coordinates it never saw and still returns an advantage, which is
    the v235 coordinate failure in the one path that bypasses the coordinate contract.
    """
    pair_a = write_model_pair(tmp_path / "pair_a.pt", seed=5)
    other = torch.load(pair_a, weights_only=False)
    other["mu"] = torch.full((ZDIM,), 3.0)
    other["sd"] = torch.full((ZDIM,), 7.0)
    torch.save(other, tmp_path / "pair_b.pt")
    run = write_run(tmp_path / "run", seed=61, residual_start=3)

    root_a = tmp_path / "out_a"
    assert main(small_run_args(pair_a, run, root_a, "updated", "predicted")) == 0
    head = only_run_dir(root_a) / "phi_head.pt"
    stored = torch.load(head, weights_only=False)
    assert len(stored["normaliser_digest"]) == 64
    assert stored["tape_digests"]

    argv = small_run_args(tmp_path / "pair_b.pt", run, tmp_path / "out_b",
                          "updated", "predicted", phi_head=head)
    with pytest.raises(ValueError, match="coordinates it never saw"):
        main(argv)
    assert not (tmp_path / "out_b").exists()

    # a head with no normaliser provenance at all is refused rather than trusted
    stripped = dict(stored)
    stripped.pop("normaliser_digest")
    torch.save(stripped, tmp_path / "bare_head.pt")
    argv = small_run_args(pair_a, run, tmp_path / "out_c", "updated", "predicted",
                          phi_head=tmp_path / "bare_head.pt")
    with pytest.raises(ValueError, match="no normaliser provenance"):
        main(argv)


def test_short_episodes_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    """An episode with fewer than H chunks offers no window; the count must say so."""
    long_run = write_run(tmp_path / "long", seed=62, n_ep=2, n_chunks=12,
                         residual_start=2)
    short_run = write_run(tmp_path / "short", seed=63, n_ep=3, n_chunks=3,
                          residual_start=0)
    episodes = []
    for run in (long_run, short_run):
        eps, _ = load_run(run, TASK, (ZDIM, C, ADIM))
        episodes.extend(eps)
    bundle = build_bundle(episodes, torch.zeros(ZDIM), torch.ones(ZDIM))
    _rows, stats = eligible_rows(bundle, 5, include_zero_delta=True)
    assert stats["episodes"] == 5
    assert stats["episodes_shorter_than_horizon"] == 3
    assert stats["episodes_contributing"] == 2
    assert stats["windows"] == 2 * (12 - 5 + 1)


def test_no_rollout_arms_are_arm_independent(tmp_path: Path) -> None:
    """observed/current must give BIT-IDENTICAL actors under two different transitions.

    If they do not, something arm-dependent has leaked into a control that is declared
    to consult no transition, and an M1-vs-control difference would be uninterpretable.
    """
    pair = write_model_pair(tmp_path / "pair.pt", identical_arms=False)
    run = write_run(tmp_path / "run", seed=64, residual_start=3)
    for advantage in ("observed", "current"):
        digests = []
        for arm in ("stale", "updated"):
            root = tmp_path / f"out_{advantage}_{arm}"
            assert main(small_run_args(pair, run, root, arm, advantage)) == 0
            ck = torch.load(only_run_dir(root) / "actor_0.pt", weights_only=False)
            digests.append(state_dict_digest(ck["state_dict"]))
        assert digests[0] == digests[1], advantage


# --- the Phi' decision region -----------------------------------------------------
# On chain3 a nonlinear function of the timestep alone explains 97.2% of Phi' variance,
# so a head fitted over every boundary is mostly a clock and ranks stall states worse
# than a region-fitted head. These tests pin the mechanics of the restriction; they do
# not re-measure that effect, which needs real tapes.

def _phi_fixture(n_ep: int = 6, n_bound: int = 12, start: int = 6):
    """Boundaries whose Phi' is a pure clock BEFORE `start` and pure state after it."""
    torch.manual_seed(0)
    zs, phis, eps = [], [], []
    for e in range(n_ep):
        offset = float(e) * 0.1
        for b in range(n_bound):
            z = torch.zeros(ZDIM)
            z[0] = offset                       # the per-episode state signal
            z[1] = float(b)                     # the clock
            zs.append(z)
            # before `start`: Phi' is the clock and carries no state information at all
            phis.append(float(b) if b < start else 2.0 + offset)
            eps.append(e)
    return (torch.stack(zs), torch.tensor(phis), torch.tensor(eps),
            [f"ep{e}" for e in range(n_ep)],
            torch.tensor([b >= start for _ in range(n_ep) for b in range(n_bound)]))


def test_phi_region_restricts_the_rows_the_head_is_fitted_on() -> None:
    z, phi, ep, keys, region = _phi_fixture()
    _, m_all = fit_phi_head(z, phi, ep, keys, seed=3, steps=60, batch=16,
                            lr=1e-3, weight_decay=1e-2, holdout_fraction=0.34)
    _, m_reg = fit_phi_head(z, phi, ep, keys, seed=3, steps=60, batch=16,
                            lr=1e-3, weight_decay=1e-2, holdout_fraction=0.34,
                            region=region)
    assert m_all["region_restricted"] is False
    assert m_reg["region_restricted"] is True
    assert m_reg["region_boundaries"] == int(region.sum())
    # the region fit sees strictly fewer boundaries, and its reported holdout set is
    # the region one - never silently the global one
    assert m_reg["train_boundaries"] < m_all["train_boundaries"]
    assert m_reg["holdout_boundaries"] < m_all["holdout_boundaries"]
    assert "holdout_corr_all_boundaries" in m_reg


def test_phi_region_metrics_separate_region_from_all_boundaries() -> None:
    z, phi, ep, keys, region = _phi_fixture()
    _, m = fit_phi_head(z, phi, ep, keys, seed=3, steps=60, batch=16, lr=1e-3,
                        weight_decay=1e-2, holdout_fraction=0.34, region=region)
    # the two numbers are computed over different row sets, so conflating them in a
    # report would be reporting one measurement as another
    assert m["holdout_corr"] != m["holdout_corr_all_boundaries"]
    assert 0.0 < m["region_fraction"] < 1.0


def test_phi_region_is_arm_independent_and_reproducible() -> None:
    z, phi, ep, keys, region = _phi_fixture()
    a, _ = fit_phi_head(z, phi, ep, keys, seed=11, steps=60, batch=16, lr=1e-3,
                        weight_decay=1e-2, holdout_fraction=0.34, region=region)
    b, _ = fit_phi_head(z, phi, ep, keys, seed=11, steps=60, batch=16, lr=1e-3,
                        weight_decay=1e-2, holdout_fraction=0.34, region=region)
    assert state_dict_digest(a.state_dict()) == state_dict_digest(b.state_dict())


def test_phi_region_that_selects_nothing_raises() -> None:
    z, phi, ep, keys, _ = _phi_fixture()
    with pytest.raises(ValueError, match="selected no boundaries"):
        fit_phi_head(z, phi, ep, keys, seed=3, steps=10, batch=8, lr=1e-3,
                     weight_decay=1e-2, holdout_fraction=0.34,
                     region=torch.zeros(len(phi), dtype=torch.bool))


# --- the bound must match the trust region the targets came from -------------------
# Regressing a scale-0.05 actor onto scale-0.80 collector deltas asks it to represent
# targets 16x outside its own range. The calibration measured |delta| =
# 0.051/0.105/0.206 producing ZERO milestone-4 events against 0.440 producing 4/24, so
# the resulting interface cannot move the task and the round would report "training
# destroys the effect" with nothing flagged.

def test_a_scale_that_disagrees_with_the_collector_is_refused(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=11, residual_start=1)
    (run / "manifest.json").write_text(
        '{"actor_scale": 0.80, "residual_start_chunk": 1, "role": "resid"}')
    argv = small_run_args(pair, run, tmp_path / "out", "updated", "predicted")
    argv[argv.index("--scale") + 1] = "0.05"
    with pytest.raises(SystemExit, match="does not match the D1 collector scale"):
        main(argv)


def test_a_matching_scale_is_accepted(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=11, residual_start=1)
    (run / "manifest.json").write_text(
        '{"actor_scale": 0.05, "residual_start_chunk": 1, "role": "resid"}')
    assert main(small_run_args(pair, run, tmp_path / "out", "updated",
                               "predicted")) == 0


def test_a_phi_region_that_disagrees_with_the_collector_is_refused(
        tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    run = write_run(tmp_path / "run", seed=11, residual_start=1)
    (run / "manifest.json").write_text(
        '{"actor_scale": 0.05, "residual_start_chunk": 1, "role": "resid"}')
    argv = small_run_args(pair, run, tmp_path / "out", "updated", "predicted",
                          phi_region_from_chunk=7)
    with pytest.raises(SystemExit, match="does not match the collectors"):
        main(argv)


def test_runs_collected_at_different_scales_are_refused(tmp_path: Path) -> None:
    pair = write_model_pair(tmp_path / "pair.pt")
    r1 = write_run(tmp_path / "a", seed=11, residual_start=1)
    r2 = write_run(tmp_path / "b", seed=12, residual_start=1)
    (r1 / "manifest.json").write_text('{"actor_scale": 0.05, "role": "resid"}')
    (r2 / "manifest.json").write_text('{"actor_scale": 0.80, "role": "resid"}')
    argv = small_run_args(pair, r1, tmp_path / "out", "updated", "predicted")
    argv.insert(argv.index(str(r1)) + 1, str(r2))
    with pytest.raises(SystemExit, match="different residual scales"):
        main(argv)
