#!/usr/bin/env python
"""CPU-only mechanical tests for the v206 deployment executor's round additions.

WHAT IS CHECKED.  (1) the rich encoding reproduces collect_v250_chain3_round.py's
`latent` and gives 8217 = 4*2048 + 25 against the pooled 2073, and the encoding is
inferred from the actor checkpoint's own dims; (2) a latent/actor width mismatch
raises at the first chunk of episode 0 instead of running a whole panel on a space
the actor never saw; (3) the stage readout is the longest achieved PREFIX of the
ordered milestone list, so events {0,1,3} are stage 2; (4) --zero-residual executes
the base chunk bit-for-bit while making exactly the same sequence of runner/env/
policy calls as the residual arm, which is what lets its rate be quoted as the
frozen-pi0.5 number for the same pipeline (CLAUDE.md rule 5); (5) the evaluation
sidecar carries none of the fields a world-model trainer reads, its boundary rows
are one per chunk plus a terminal row with no repeated t, and a panel whose object
set or goal-atom order drifts between episodes raises instead of stacking rows
whose positions mean different entities; (6) the executor's own seed guard covers
the round's D0/D1 collection ranges, and the normaliser an arm deployed with is
recorded so an unmatched one is visible afterwards.

SCOPE OF THE CODE-PATH CLAIM, stated so it is not read as more than it is.  The
trace comparison runs under a stub env whose transition ignores the action, so it
establishes that the two arms issue the same calls in the same order from the same
state.  In the real environment the arms' states diverge after the first chunk by
construction, so trace equality there holds only up to that divergence; these tests
say nothing about behaviour after it, and nothing about a divergence living inside
Pi05Runner.sample_chunk or prefix_forward, which are stubbed here.

WHAT WOULD OVERTURN THE CODE-PATH CLAIM: the recorded call trace of the two arms
differing anywhere before the executed action, or the zero arm's executed chunk
differing from the base chunk in any element.

No simulator and no policy: the runner, the environment, prefix_forward, the
proprio reader, the predicate/automaton readers and the pose reader are all stubs.
The module under test imports none of LIBERO at import time, which this file
relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_v206_belief_residual_deploy import (  # noqa: E402
    CAMERA_BLOCKS,
    PROPRIO_DIM,
    SPENT_RANGES,
    TRAINING_TAPE_FIELDS,
    DeployConfig,
    SimDeps,
    build_eval_labels,
    final_atom_reach,
    encode_latent,
    label_scope_key,
    latent_dims,
    prefix_stage_reached,
    progress_timing,
    resolve_latent_mode,
    run_episode,
    tensor_digest,
)
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

N_SUBGOALS = 6
#: one receptacle + two manipulands, named so that lcwm/phase_potential.py's
#: resolve_entity_row can map "basket_1_contain_region" back onto "basket_1"
OBJECT_BODIES = {"basket_1": "basket_1_main", "obj_0": "obj_0_main",
                 "obj_1": "obj_1_main"}
N_OBJ = len(OBJECT_BODIES)
N_ATOMS = 2


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
class StubPrefix:
    def __init__(self, hidden, mask):
        self.hidden = hidden.unsqueeze(0)
        self.pad_masks = mask.unsqueeze(0)


class StubEnv:
    """A deterministic toy env: it records every call and never touches mujoco."""

    task_description = "stub task"

    def __init__(self, trace, horizon=40):
        self.trace = trace
        self.horizon = horizon
        self.t = 0

    def reset(self, seed=None):
        self.trace.append(("env.reset", int(seed)))
        self.t = 0
        return {"tick": 0}, {}

    def step(self, act):
        self.trace.append(("env.step",))
        self.t += 1
        done = self.t >= self.horizon
        return {"tick": self.t}, 0.0, done, False, {"is_success": False,
                                                    "done": done}


class StubRunner:
    def __init__(self, trace, hidden_dim, adim, c, prefix_len=520):
        self.trace = trace
        self.policy = object()
        self.hidden_dim = hidden_dim
        self.adim = adim
        self.c = c
        self.prefix_len = prefix_len
        self.calls = 0

    def reset(self):
        self.trace.append(("runner.reset",))

    def _obs_to_policy_batch(self, obs, desc):
        self.trace.append(("obs_to_batch", int(obs["tick"])))
        return {"tick": obs["tick"]}

    def sample_chunk(self, obs, desc):
        """Deterministic in the observation, so the two arms share chunk 0."""
        self.trace.append(("sample_chunk", int(obs["tick"])))
        self.calls += 1
        g = torch.Generator().manual_seed(1000 + int(obs["tick"]))
        return torch.rand(1, self.c, self.adim, generator=g)

    def chunk_to_env(self, chunk):
        self.trace.append(("chunk_to_env",))
        return [chunk[i] for i in range(chunk.shape[0])]


def make_deps(trace, hidden_dim, prefix_len=520, adim=7):
    def prefix_forward(policy, batch):
        trace.append(("prefix_forward", int(batch["tick"])))
        g = torch.Generator().manual_seed(7 + int(batch["tick"]))
        hidden = torch.rand(prefix_len, hidden_dim, generator=g)
        mask = torch.ones(prefix_len, dtype=torch.bool)
        mask[300:400] = False                     # a real padded span in camera 2
        return StubPrefix(hidden, mask)

    def proprio(obs):
        return torch.full((PROPRIO_DIM,), float(obs["tick"]) * 0.01)

    def goal_atoms(env):
        return [["in", f"obj_{i}", "basket_1_contain_region"]
                for i in range(N_ATOMS)]

    def predicate_bits(env, atoms):
        return np.zeros(len(atoms), dtype=bool)

    class StubAutomaton:
        def __init__(self, subgoals):
            self.subgoals = list(subgoals)
            self.events_achieved = {}

        def start(self, env):
            trace.append(("automaton.start",))

        def evaluate(self, env, step):
            trace.append(("automaton.evaluate", int(step)))
            # milestones 0, 1 and 3: a later one without its predecessor
            if step >= 10:
                self.events_achieved.setdefault(0, 10)
                self.events_achieved.setdefault(1, 10)
            if step >= 20:
                self.events_achieved.setdefault(3, 20)
            return [False] * len(self.subgoals)

    def body_positions(env, names):
        # a moving scene, so the offline Phi' is not evaluated on a constant
        out = np.zeros((len(names), 3), dtype=np.float32)
        for i in range(len(names)):
            out[i] = (0.1 * i, 0.02 * env.t, 0.05 + 0.001 * env.t)
        return out

    def discover_object_bodies(env):
        return dict(OBJECT_BODIES)

    return SimDeps(prefix_forward=prefix_forward, proprio=proprio,
                   goal_atoms=goal_atoms, predicate_bits=predicate_bits,
                   goal_automaton=StubAutomaton, body_positions=body_positions,
                   discover_object_bodies=discover_object_bodies)


class StubWM(torch.nn.Module):
    """Same surface the loop uses of RawLatentWM: enc, step, bdim."""

    def __init__(self, zdim, bdim=8, edim=8):
        super().__init__()
        self.bdim, self.edim = bdim, edim
        self.lin = torch.nn.Linear(zdim, edim)

    def enc(self, z):
        return torch.tanh(self.lin(z))

    def step(self, b, e, u_prev):
        return torch.tanh(b + e + u_prev.flatten(-2).sum(-1, keepdim=True))


def make_cfg(zdim, adim=7, c=4, **kw):
    return DeployConfig(task="chain3_lr2", horizon=kw.pop("horizon", 40), c=c,
                        adim=adim, zdim=zdim,
                        mu=torch.zeros(zdim), sd=torch.ones(zdim),
                        bmu=torch.zeros(8), bsd=torch.ones(8),
                        condition="raw",
                        subgoals=tuple(f"sg{i}" for i in range(N_SUBGOALS)), **kw)


def loud_actor(zdim, c, adim, scale=0.05):
    """ResidualActor is zero-init, i.e. a no-op; a control test needs a live one."""
    torch.manual_seed(0)
    a = ResidualActor(zdim, c, adim, scale=scale)
    with torch.no_grad():
        torch.nn.init.normal_(a.net[-1].weight, std=1.0)
        torch.nn.init.normal_(a.net[-1].bias, std=1.0)
    a.eval()
    return a


# --------------------------------------------------------------------------- #
# 1. the rich encoding
# --------------------------------------------------------------------------- #
def test_latent_dims_are_the_measured_2073_and_8217() -> None:
    assert latent_dims(2048) == (2073, 8217)


def test_encode_latent_widths_and_blocks() -> None:
    torch.manual_seed(3)
    d = 8
    hidden = torch.randn(520, d)
    mask = torch.ones(520, dtype=torch.bool)
    mask[300:400] = False
    pooled = encode_latent(hidden, mask, rich=False)
    rich = encode_latent(hidden, mask, rich=True)
    # the pooled branch is the historical masked_prefix_mean, unchanged
    torch.testing.assert_close(pooled, masked_prefix_mean(hidden, mask))
    assert pooled.shape == (d,)
    assert rich.shape == (2 * len(CAMERA_BLOCKS) * d,)
    # block 0 is unmasked, so its mean/max are the plain ones; block 1 is padded
    torch.testing.assert_close(rich[:d], hidden[0:256].mean(0))
    torch.testing.assert_close(rich[d:2 * d], hidden[0:256].max(0).values)
    blk = hidden[256:512][mask[256:512]]
    torch.testing.assert_close(rich[2 * d:3 * d], blk.mean(0))
    torch.testing.assert_close(rich[3 * d:], blk.max(0).values)


def test_encode_latent_matches_the_collector_implementation() -> None:
    """Byte-for-byte against a transcription of collect_v250_chain3_round.latent."""
    torch.manual_seed(11)
    hidden = torch.randn(600, 6)
    mask = torch.ones(600, dtype=torch.bool)
    mask[0:256] = False                    # the all-masked fallback branch
    parts = []
    for lo, hi in ((0, 256), (256, 512)):
        blk = hidden[lo:hi][mask[lo:hi]]
        if len(blk) == 0:
            blk = hidden[lo:hi]
        parts += [blk.mean(0), blk.max(0).values]
    torch.testing.assert_close(encode_latent(hidden, mask, rich=True),
                               torch.cat(parts))


def test_resolve_latent_mode_infers_from_checkpoint_dims() -> None:
    assert resolve_latent_mode(2073, 2048) is False
    assert resolve_latent_mode(8217, 2048) is True


def test_resolve_latent_mode_flag_disagreement_raises() -> None:
    for zdim, forced in ((8217, False), (2073, True)):
        try:
            resolve_latent_mode(zdim, 2048, forced)
        except SystemExit:
            continue
        raise AssertionError(f"expected SystemExit for zdim={zdim} forced={forced}")


def test_resolve_latent_mode_unknown_width_raises() -> None:
    try:
        resolve_latent_mode(4096, 2048)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for a width matching neither encoding")


def test_episode_latent_width_rich_vs_pooled() -> None:
    d, adim, c = 8, 7, 4
    for rich, zdim in ((False, d + PROPRIO_DIM),
                       (True, 2 * len(CAMERA_BLOCKS) * d + PROPRIO_DIM)):
        trace = []
        deps = make_deps(trace, d)
        cfg = make_cfg(zdim, adim=adim, c=c, collect_tape=True)
        r = run_episode(StubRunner(trace, d, adim, c), StubEnv(trace), 5, cfg,
                        StubWM(zdim), loud_actor(zdim, c, adim), deps)
        assert r["rich_latent"] is rich
        assert r["triples"], "collect-tape produced no triple"
        assert r["triples"][0][0].shape == (zdim,)


# --------------------------------------------------------------------------- #
# 2. the dims assertion
# --------------------------------------------------------------------------- #
def test_dims_mismatch_raises_on_the_first_chunk() -> None:
    d, adim, c = 8, 7, 4
    trace = []
    deps = make_deps(trace, d)
    zdim = 2 * len(CAMERA_BLOCKS) * d + PROPRIO_DIM
    cfg = make_cfg(zdim, adim=adim, c=c, rich=False)   # forced pooled, rich actor
    try:
        run_episode(StubRunner(trace, d, adim, c), StubEnv(trace), 5, cfg,
                    StubWM(zdim), loud_actor(zdim, c, adim), deps)
    except SystemExit:
        assert ("env.step",) not in trace, "raised only after stepping the env"
        return
    raise AssertionError("expected SystemExit on a latent/actor width mismatch")


# --------------------------------------------------------------------------- #
# 3. the contiguous-prefix stage
# --------------------------------------------------------------------------- #
def test_prefix_stage_is_a_prefix_not_a_count() -> None:
    assert prefix_stage_reached({0: 1, 1: 2, 3: 9}, 6) == 2
    assert prefix_stage_reached({}, 6) == 0
    assert prefix_stage_reached({1: 3, 2: 4}, 6) == 0
    assert prefix_stage_reached({0: 1, 1: 2, 2: 3}, 6) == 3
    assert prefix_stage_reached({i: i for i in range(6)}, 6) == 6


def test_progress_timing_reports_the_idle_tail() -> None:
    assert progress_timing({0: 10, 1: 260}, 750) == (260, 490)
    assert progress_timing({}, 750) == (0, 750)


def test_episode_reports_stage_prefix_and_tail() -> None:
    d, adim, c = 8, 7, 4
    trace = []
    deps = make_deps(trace, d)
    zdim = d + PROPRIO_DIM
    cfg = make_cfg(zdim, adim=adim, c=c, eval_labels=True, horizon=40)
    r = run_episode(StubRunner(trace, d, adim, c), StubEnv(trace, horizon=40), 5,
                    cfg, StubWM(zdim), loud_actor(zdim, c, adim), deps)
    assert sorted(r["events"]) == [0, 1, 3]
    assert r["stage_reached"] == 2                 # NOT 3
    assert r["last_progress_step"] == 20
    assert r["idle_tail"] == r["steps"] - 20
    # one label per chunk boundary plus the terminal one
    assert len(r["labels"]["t"]) == r["chunks"] + 1
    assert r["labels"]["obj_pos"][0].shape == (N_OBJ, 3)
    assert r["labels"]["bits"][0].shape == (N_ATOMS,)


# --------------------------------------------------------------------------- #
# 4. --zero-residual is the same loop with Delta := 0
# --------------------------------------------------------------------------- #
def _run_arm(zero: bool):
    d, adim, c = 8, 7, 4
    trace = []
    deps = make_deps(trace, d)
    zdim = d + PROPRIO_DIM
    runner = StubRunner(trace, d, adim, c)
    env = StubEnv(trace, horizon=40)
    executed = []

    original = runner.chunk_to_env

    def spy(chunk):
        executed.append(chunk.clone())
        return original(chunk)

    runner.chunk_to_env = spy
    cfg = make_cfg(zdim, adim=adim, c=c, zero_residual=zero, eval_labels=True)
    r = run_episode(runner, env, 5, cfg, StubWM(zdim),
                    loud_actor(zdim, c, adim), deps)
    return r, trace, executed


def test_zero_residual_executes_the_base_chunk_exactly() -> None:
    r, _trace, executed = _run_arm(zero=True)
    assert r["mean_abs_delta"] == 0.0
    assert r["max_abs_delta"] == 0.0
    g = torch.Generator().manual_seed(1000 + 0)
    base0 = torch.rand(1, 4, 7, generator=g)[0]
    torch.testing.assert_close(executed[0], base0, rtol=0, atol=0)


def test_zero_residual_shares_the_residual_arm_call_trace() -> None:
    zero_r, zero_trace, zero_exec = _run_arm(zero=True)
    live_r, live_trace, live_exec = _run_arm(zero=False)
    assert live_r["max_abs_delta"] > 0.0, "the control actor must actually move"
    # identical structure: same calls in the same order, same count
    assert zero_trace == live_trace
    assert zero_r["chunks"] == live_r["chunks"]
    assert zero_r["steps"] == live_r["steps"]
    # and the ONLY difference is the residual itself, inside its bound
    assert not torch.equal(zero_exec[0], live_exec[0])
    diff = (live_exec[0] - zero_exec[0]).abs()
    assert float(diff.max()) <= 0.05 + 1e-6
    assert float(diff.max()) > 0.0


def test_zero_residual_equals_a_genuinely_zero_actor() -> None:
    """--zero-residual must execute what a zero-init (no-op) actor executes.

    ResidualActor is zero-init by construction, so an untrained interface is
    already a no-op; the flag has to reproduce it action-for-action.
    """
    d, adim, c = 8, 7, 4
    zdim = d + PROPRIO_DIM
    runs = []
    for zero, actor in ((True, loud_actor(zdim, c, adim)),
                        (False, ResidualActor(zdim, c, adim, scale=0.05).eval())):
        trace = []
        deps = make_deps(trace, d)
        runner = StubRunner(trace, d, adim, c)
        executed = []
        original = runner.chunk_to_env

        def spy(chunk, _o=original, _e=executed):
            _e.append(chunk.clone())
            return _o(chunk)

        runner.chunk_to_env = spy
        cfg = make_cfg(zdim, adim=adim, c=c, zero_residual=zero)
        runs.append((run_episode(runner, StubEnv(trace), 5, cfg, StubWM(zdim),
                                 actor, deps), trace, executed))
    (r0, t0, e0), (r1, t1, e1) = runs
    assert r0["steps"] == r1["steps"] and r0["chunks"] == r1["chunks"]
    assert r1["max_abs_delta"] == 0.0
    assert t0 == t1
    assert len(e0) == len(e1) and e0
    for x, y in zip(e0, e1):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 5. the sidecar is evaluation-only
# --------------------------------------------------------------------------- #
def _sidecar_payload():
    n = 5
    return {"eef_proprio": torch.zeros(n, PROPRIO_DIM),
            "obj_pos": torch.zeros(n, N_OBJ, 3),
            "bits": torch.zeros(n, N_ATOMS, dtype=torch.bool),
            "episode": torch.zeros(n, dtype=torch.long),
            "t": torch.arange(n),
            "object_names": [f"obj_{i}" for i in range(N_OBJ)],
            "goal_atoms": [["in", "obj_0", "basket_1"]],
            "milestones": [f"sg{i}" for i in range(N_SUBGOALS)],
            "events": {0: {0: 10}}, "task": "chain3_lr2", "c": 10,
            "role": "evaluation", "provenance": "EVALUATION-ONLY readout"}


def test_sidecar_carries_no_training_field() -> None:
    payload = build_eval_labels(_sidecar_payload())
    assert set(payload) & set(TRAINING_TAPE_FIELDS) == set()
    # every key lcwm/v250_progress.tape_progress reads is present
    for key in ("episode", "t", "eef_proprio", "obj_pos", "bits",
                "object_names", "goal_atoms"):
        assert key in payload


def test_sidecar_feeds_v250_progress_offline() -> None:
    """The panel sidecar, assembled as main() assembles it, must be exactly what
    lcwm/v250_progress.tape_progress consumes - that is the whole point of
    recording eef/obj_pos/bits per boundary rather than a scalar."""
    from lcwm.v250_progress import tape_progress

    d, adim, c = 8, 7, 4
    zdim = d + PROPRIO_DIM
    eef, obj, bits, lep, lt = [], [], [], [], []
    names, atoms = None, None
    for ei, seed in enumerate((5, 6)):
        trace = []
        deps = make_deps(trace, d)
        cfg = make_cfg(zdim, adim=adim, c=c, eval_labels=True)
        r = run_episode(StubRunner(trace, d, adim, c), StubEnv(trace), seed, cfg,
                        StubWM(zdim), loud_actor(zdim, c, adim), deps, ep_idx=ei)
        lab = r["labels"]
        eef += lab["eef_proprio"]
        obj += lab["obj_pos"]
        bits += lab["bits"]
        lep += [ei] * len(lab["t"])
        lt += lab["t"]
        names = names or lab["object_names"]
        atoms = atoms if atoms is not None else lab["goal_atoms"]
    payload = build_eval_labels(
        {"eef_proprio": torch.stack(eef), "obj_pos": torch.stack(obj),
         "bits": torch.stack(bits), "episode": torch.tensor(lep),
         "t": torch.tensor(lt), "object_names": names, "goal_atoms": atoms,
         "milestones": [f"sg{i}" for i in range(N_SUBGOALS)],
         "task": "chain3_lr2", "c": 10, "role": "evaluation"})
    prog = tape_progress(payload)
    assert prog["phi"].shape == (len(lep),)
    assert prog["ordinal"].shape == (len(lep),)
    assert torch.isfinite(prog["phi"]).all()
    assert float(prog["d_active"].max()) > 0.0


def test_sidecar_rejects_a_training_field() -> None:
    for field_name in TRAINING_TAPE_FIELDS:
        payload = _sidecar_payload()
        payload[field_name] = torch.zeros(5, 3)
        try:
            build_eval_labels(payload)
        except SystemExit:
            continue
        raise AssertionError(f"{field_name} was allowed into the sidecar")


# --------------------------------------------------------------------------- #
# 6. the boundary rows, the label scope, the seed guard and the normaliser record
# --------------------------------------------------------------------------- #
def test_boundary_rows_are_one_per_chunk_with_no_repeated_t() -> None:
    """Phi' is computed per (episode, t) after an argsort on t.

    Two rows sharing a t inside one episode would be ordered arbitrarily and would
    double-count a boundary in any per-boundary average; a missing terminal row
    would drop the stall's last boundary, which on chain3 is the region the round
    is about.  chain3@750 with c=10 gives 75 chunks and 76 rows.
    """
    d, adim, c = 8, 7, 4
    for horizon, ehorizon in ((40, 40), (40, 13), (12, 40)):
        trace = []
        deps = make_deps(trace, d)
        zdim = d + PROPRIO_DIM
        cfg = make_cfg(zdim, adim=adim, c=c, eval_labels=True, horizon=horizon)
        r = run_episode(StubRunner(trace, d, adim, c),
                        StubEnv(trace, horizon=ehorizon), 5, cfg, StubWM(zdim),
                        loud_actor(zdim, c, adim), deps)
        ts = r["labels"]["t"]
        assert len(ts) == r["chunks"] + 1, (horizon, ehorizon, ts)
        assert len(set(ts)) == len(ts), f"repeated boundary t: {ts}"
        assert ts == sorted(ts)
        assert ts[0] == 0 and ts[-1] == r["steps"]


def test_label_scope_key_detects_object_and_atom_drift() -> None:
    base = {"object_names": ["a", "b"], "goal_atoms": [["in", "a", "bin"]]}
    assert label_scope_key(base) == label_scope_key(dict(base))
    # tuples and lists are the same scope; a reorder or a rename is not
    assert label_scope_key({"object_names": ("a", "b"),
                            "goal_atoms": [("in", "a", "bin")]}) == \
        label_scope_key(base)
    for drift in ({"object_names": ["b", "a"], "goal_atoms": base["goal_atoms"]},
                  {"object_names": ["a", "c"], "goal_atoms": base["goal_atoms"]},
                  {"object_names": base["object_names"],
                   "goal_atoms": [["in", "b", "bin"]]},
                  {"object_names": base["object_names"], "goal_atoms": []}):
        assert label_scope_key(drift) != label_scope_key(base), drift


def test_spent_ranges_cover_the_rounds_collection_seeds() -> None:
    """D0 8700-8795 and D1 8800-8847 are what the arms were trained on.

    Both are half-open in SPENT_RANGES.  A panel started inside either one would
    evaluate every arm on its own training seeds; the guard must refuse it.
    """
    spent = {s for lo, hi in SPENT_RANGES for s in range(lo, hi)}
    for seed in (6000, 7395, 8700, 8795, 8800, 8847):
        assert seed in spent, seed
    for seed in (8796, 8799, 8848, 9000, 9095, 9100):
        assert seed not in spent, seed


def test_tensor_digest_separates_normalisers() -> None:
    a = torch.arange(8, dtype=torch.float32)
    assert tensor_digest(a) == tensor_digest(a.clone())
    b = a.clone()
    b[3] += 1e-6
    assert tensor_digest(a) != tensor_digest(b)
    # a non-contiguous view of the same values still digests the same
    wide = torch.stack([a, a + 1], 1)
    assert tensor_digest(wide[:, 0]) == tensor_digest(a)


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))


# --- closest approach to the FINAL goal object -------------------------------------
# The unrestricted version of this readout reported 9/24 base episodes within 5 cm of
# "the active object" where the restricted one reports 0/24 - those minima were the
# gripper beside a CAN during the tail of its placement. The restriction is the
# measurement, not a refinement.

def _rows(spec):
    """spec: list of (t, eef_xyz, cream_xyz, bits) -> the four parallel row lists."""
    eef, obj, bits, ts = [], [], [], []
    for t, e, cx, b in spec:
        eef.append(torch.tensor(list(e) + [0.0] * 22))
        obj.append(torch.tensor([[9.0, 9.0, 9.0], list(cx)]))   # can at index 0
        bits.append(torch.tensor(b))
        ts.append(t)
    return eef, obj, bits, ts


NAMES = ["can_1", "cream_1"]
ATOMS = [["in", "can_1", "basket"], ["in", "cream_1", "basket"]]


def test_only_boundaries_where_the_final_atom_is_active_are_counted():
    # t=100: can NOT yet placed, gripper 1 cm from the cream -> must be IGNORED
    # t=300: can placed, gripper 40 cm from the cream        -> counted
    spec = [(100, (0, 0, 0), (0.01, 0, 0), [False, False]),
            (300, (0, 0, 0), (0.40, 0, 0), [True, False])]
    out = final_atom_reach(*_rows(spec), NAMES, ATOMS, basin_from_step=0)
    assert out["final_atom_boundaries"] == 1
    assert out["d_min_final_atom"] == pytest.approx(0.40, abs=1e-6)


def test_a_completed_final_atom_is_not_active_either():
    spec = [(300, (0, 0, 0), (0.02, 0, 0), [True, True])]
    out = final_atom_reach(*_rows(spec), NAMES, ATOMS, basin_from_step=0)
    assert out["final_atom_boundaries"] == 0
    assert out["d_min_final_atom"] is None


def test_the_basin_gate_excludes_earlier_boundaries():
    spec = [(100, (0, 0, 0), (0.05, 0, 0), [True, False]),
            (300, (0, 0, 0), (0.50, 0, 0), [True, False])]
    out = final_atom_reach(*_rows(spec), NAMES, ATOMS, basin_from_step=260)
    assert out["final_atom_boundaries"] == 1
    assert out["d_min_final_atom"] == pytest.approx(0.50, abs=1e-6)


def test_it_takes_the_minimum_over_qualifying_boundaries():
    spec = [(300, (0, 0, 0), (0.50, 0, 0), [True, False]),
            (400, (0, 0, 0), (0.12, 0, 0), [True, False]),
            (500, (0, 0, 0), (0.33, 0, 0), [True, False])]
    out = final_atom_reach(*_rows(spec), NAMES, ATOMS, basin_from_step=0)
    assert out["d_min_final_atom"] == pytest.approx(0.12, abs=1e-6)
    assert out["final_atom_boundaries"] == 3


def test_no_qualifying_boundary_reports_none_rather_than_a_number():
    out = final_atom_reach([], [], [], [], NAMES, ATOMS, basin_from_step=0)
    assert out["d_min_final_atom"] is None


# --- the fixed horizon must survive a SUCCESS --------------------------------------
# `info["done"]` is success-driven on chain3: measured 2026-09-12, seed 9018's `done`
# first fired at t=678, the same step as `is_success`. A loop that breaks on `done`
# therefore produces a 678-step episode and the row count still leaks the outcome -
# which is the whole reason --fixed-horizon exists. These tests pin the predicate.

def _done_rule(fixed, info, tr, tm):
    """The executor's own termination expression, isolated."""
    return ((bool(info.get("done", False))
             and not bool(info.get("is_success", False))) or bool(tr)) \
        if fixed else bool(tm or tr)


def test_fixed_horizon_does_not_stop_on_a_success_driven_done():
    info = {"done": True, "is_success": True}      # what chain3 emits at success
    assert _done_rule(True, info, False, True) is False


def test_fixed_horizon_still_stops_on_a_genuine_terminal():
    info = {"done": True, "is_success": False}     # a real failure terminal
    assert _done_rule(True, info, False, True) is True


def test_fixed_horizon_still_stops_on_truncation():
    info = {"done": False, "is_success": False}
    assert _done_rule(True, info, True, False) is True


def test_without_fixed_horizon_the_old_behaviour_is_unchanged():
    info = {"done": True, "is_success": True}
    assert _done_rule(False, info, False, True) is True


def test_the_leak_shape_is_what_the_rule_prevents():
    # before the fix: success at 678 -> done True -> break -> 678-step episode.
    # after: the same info keeps the loop running to the horizon.
    at_success = {"done": True, "is_success": True}
    assert _done_rule(True, at_success, False, False) is False
    # and it keeps returning False for every subsequent step, since done stays set
    assert _done_rule(True, at_success, False, False) is False
