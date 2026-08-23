"""Action 2P execution substrate (daily 2026-08-20 §4, §6 items 3-4).

Owns three things the pilot's validity depends on:

* **paired CRN execution** - the reference and every alternative are run under
  the SAME three presealed continuation keys, so each alternative has three
  matched comparisons.  Action 2P replaces the older best-of-three aggregation
  with same-key pairing precisely because reference variability must not be
  able to manufacture a positive.
* **outcome-blind pool construction** - no proposal feature, pool selection, or
  tie-break may consult the source trajectory after `tau`, its terminal mask, or
  any branch outcome.  The reference is drawn from its own key, disjoint from
  the 32 alternative keys, and is never the source's cached post-`tau` action:
  that suffix is conditioned to fail and cannot serve as a reference.
* **abort-safe segment accounting** - the ledger row is written BEFORE the
  charge, and a worst-case headroom check runs BEFORE each segment, so neither
  a kill nor an overrun can leave spent steps unrecorded.  Both failure modes
  actually happened in Stage 1R.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lcwm.libero_paths import ensure_project_libero_config

ensure_project_libero_config()

from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import restore, snap  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v081p_contract import (C_PREFIX, DEADLINE, H_PRIMARY, H_READOUTS,  # noqa: E402
                                 MIN_UNIQUE_ALTS, N_ALT_DRAWS, N_CRN,
                                 N_DIVERSITY, N_RANDOM, TAU, Subrole,
                                 alternative_keys, crn_keys, outcome_vector,
                                 reference_key)
from lcwm.v08r_contract import officiality, stratum_key  # noqa: E402


# --------------------------------------------------------------------------
# Abort-safe segment ledger with pre-segment headroom check
# --------------------------------------------------------------------------

class TechnicalHalt(SystemExit):
    """A technically-invalid source (§4): charged, never replaced, never counted
    as a normal non-failure."""


class SegmentLedger:
    """Cap accounting keyed to the ACTION, not the invocation.

    Two properties this had to regain, both learned the hard way in Stage 1R.1:

    * `spent` is derived by globbing every segment ledger under the action root,
      so a second `--run` cannot silently receive a fresh 20,120-step cap. An
      earlier version of this class re-derived only from its own file and did
      exactly that.
    * a segment writes an `open` row with its worst-case reservation BEFORE it
      steps and a `close` row with actual steps after, and re-derivation charges
      an unclosed `open` at its reservation. Charging only on completion loses
      the steps of any segment killed mid-flight.
    """

    def __init__(self, path: Path, caps: dict[str, int], action_root: Path | None = None):
        self.path, self.caps = Path(path), dict(caps)
        self.action_root = Path(action_root) if action_root else self.path.parent.parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
        self.spent = self._derive()

    #: Directory-name prefixes whose spend is charged to THEIR OWN registration
    #: and must not consume a successor's cap.  A cap belongs to a registration;
    #: a superseded or aborted registration's steps stay charged to it and stay
    #: in the project total, but deducting them from the next registration would
    #: conflate two experiments and silently shrink the successor's budget.
    QUARANTINE_PREFIXES = ("ABORTED_", "SMOKE_", "SUPERSEDED_", "HALTED_")

    def _derive(self) -> dict[str, int]:
        spent: dict[str, int] = {}
        opens: dict[str, dict] = {}
        for led in sorted(self.action_root.glob("**/segment_ledger.jsonl")):
            if any(part.startswith(self.QUARANTINE_PREFIXES)
                   for part in led.relative_to(self.action_root).parts):
                continue
            for line in led.read_text().splitlines():
                if not line:
                    continue
                r = json.loads(line)
                if r.get("phase") == "open":
                    opens[r["segment_id"]] = r
                elif r.get("phase") == "close":
                    opens.pop(r["segment_id"], None)
                    spent[r["cap_line"]] = spent.get(r["cap_line"], 0) + r["n_steps"]
        for r in opens.values():        # killed mid-segment: charge the reservation
            spent[r["cap_line"]] = spent.get(r["cap_line"], 0) + r["reservation"]
        return spent

    def _write(self, row: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def open_segment(self, segment_id: str, cap_line: str, reservation: int,
                     **row) -> None:
        ok, why = self.headroom_ok(cap_line, reservation)
        if not ok:
            raise SystemExit(f"HALT before segment: {why}")
        self._write({"phase": "open", "segment_id": segment_id,
                     "cap_line": cap_line, "reservation": int(reservation),
                     "n_steps": 0, **row})
        self.spent[cap_line] = self.spent.get(cap_line, 0) + int(reservation)

    def close_segment(self, segment_id: str, cap_line: str, reservation: int,
                      n_steps: int, **row) -> None:
        self._write({"phase": "close", "segment_id": segment_id,
                     "cap_line": cap_line, "n_steps": int(n_steps), **row})
        self.spent[cap_line] = self.spent.get(cap_line, 0) - int(reservation) + int(n_steps)
        if self.spent[cap_line] > self.caps[cap_line]:
            raise SystemExit(f"HALT: {cap_line} cap {self.caps[cap_line]} "
                             f"exceeded ({self.spent[cap_line]})")

    def headroom_ok(self, cap_line: str, worst_case: int) -> tuple[bool, str]:
        """Checked BEFORE a segment runs.  Stage 1R.1 executed a segment and
        only then discovered it had blown the line."""
        have = self.caps[cap_line] - self.spent.get(cap_line, 0)
        return (worst_case <= have,
                f"{cap_line}: worst case {worst_case} vs headroom {have}")

    @property
    def total(self) -> int:
        return sum(self.spent.values())


# --------------------------------------------------------------------------
# Anchor state
# --------------------------------------------------------------------------

@dataclass
class Anchor:
    anchor_id: str
    seed: int
    tau: int
    snapshot: object
    stratum: dict
    mask_ok: bool
    mask_why: list = field(default_factory=list)
    content_hash: str = ""
    # The source automaton, forked for every branch.  Rebuilding a fresh
    # GoalAutomaton at tau would re-base `pick_up` displacement to the tau pose,
    # so a branch would measure milestones on a different instrument than the
    # source did - the object has already moved by step 160.
    source_automaton: object = None


def _seed_all(seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sets_now(automaton, upto: int):
    ever = sorted(i for i, s in automaton.events_achieved.items() if s <= upto)
    valid = [i for i, v in enumerate(automaton.prev_valid) if v]
    dropped = {i for (s, i, d) in automaton.flips if d == -1 and s <= upto}
    return ever, valid, sorted(dropped - set(valid))


def run_source(runner, env, seed: int, ledger: SegmentLedger) -> dict:
    """One stock rollout to L, snapshotting at `tau`.  Establishes `failure@L`
    only: its post-`tau` suffix is conditioned to fail and is never a reference.
    """
    seg = f"src-{seed}"
    ledger.open_segment(seg, Subrole.SOURCE.value, DEADLINE, seed=seed,
                        subrole=Subrole.SOURCE.value)
    _seed_all(seed)
    runner.reset()
    obs, _ = env.reset(seed=seed)
    subgoals = V080_TASKS[env.task]["ordered_subgoals"]
    au = GoalAutomaton(subgoals)
    au.start(env)
    atoms = goal_atoms(env)
    instr = env.task_description
    au.evaluate(env, 0)

    anchor = None
    success_step, t, done = None, 0, False
    try:
      while not done and t < DEADLINE:
        obs, _r, term, trunc, info = env.step(runner.select_action(obs, instr))
        t += 1
        done = bool(term or trunc)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if t % 10 == 0 or done:
            au.evaluate(env, t)
            if success_step is None and predicate_bits(env, atoms).all():
                success_step = t
        if t == TAU and success_step is None:
            ever, valid, dmg = _sets_now(au, t)
            sk = stratum_key(env.task, subgoals, ever, valid, dmg)
            good, why_bad = mask_check(sk)
            snapshot = snap(env, t, "libero_10", 0, seed=seed, tau=t)
            anchor = Anchor(f"a{seed}", seed, t, snapshot, sk.to_dict(),
                            good, why_bad, content_hash=_hash_snapshot(snapshot),
                            source_automaton=au.fork())
      if done and success_step is None and t < DEADLINE:
        raise TechnicalHalt(
            f"TECHNICAL_HALT: source {seed} terminated at {t} < {DEADLINE} "
            f"without success; charged, not replaced, and not counted as a "
            f"normal non-failure")
    finally:
      ledger.close_segment(seg, Subrole.SOURCE.value, DEADLINE, t, seed=seed,
                           subrole=Subrole.SOURCE.value, anchor_id=None,
                           officiality=officiality(t, DEADLINE).value,
                           termination="deadline" if not done else "terminated")
    return {"seed": seed, "steps": t, "success_step": success_step,
            "failure_at_L": success_step is None,
            "anchor": anchor, "events": dict(au.events_achieved)}


def mask_check(sk):
    from lcwm.v081p_contract import mask_conformant
    return mask_conformant(sk.ever_achieved, sk.current_valid, sk.damaged,
                           sk.actionable, sk.unresolved)


def _hash_snapshot(s) -> str:
    """Covers physics AND the controller/robot bookkeeping that `restore` sets.

    Omitting controller state would let controller drift pass a restore check
    that exists to catch exactly that.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(s.state, dtype=np.float64).tobytes())
    for f in ("sim_ctrl", "sim_qfrc_applied", "sim_xfrc_applied",
              "sim_mocap_pos", "sim_mocap_quat", "sim_qacc_warmstart"):
        v = getattr(s, f, None)
        if v is not None:
            h.update(np.asarray(v, dtype=np.float64).tobytes())
    for f in ("controller_state", "robot_state"):
        v = getattr(s, f, None)
        if v is not None:
            h.update(json.dumps(v, sort_keys=True, default=_json_num).encode())
    h.update(str(getattr(s, "env_timestep", None)).encode())
    return h.hexdigest()


def _json_num(o):
    if isinstance(o, np.ndarray):
        return np.asarray(o, dtype=np.float64).round(12).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def verify_restore(env, anchor: "Anchor") -> tuple[bool, str]:
    """Re-snap the LIVE environment after restore and hash THAT.

    The previous check hashed `anchor.snapshot` — the stored object — and
    compared it to `anchor.content_hash`, which is derived from the same object.
    It compared a constant to itself and could never fail, making ADVANCE
    condition 2 vacuous.
    """
    fresh = snap(env, anchor.tau, "libero_10", 0)
    got = _hash_snapshot(fresh)
    return got == anchor.content_hash, got


def body_signature(env) -> np.ndarray:
    """Achieved physical state used for the replay-tolerance test: tracked body
    positions, not the commanded actions."""
    from lcwm.probe_data import body_positions, discover_object_bodies
    bodies = discover_object_bodies(env)
    names = [bodies[k] for k in sorted(bodies)]
    return body_positions(env, names).reshape(-1)


# --------------------------------------------------------------------------
# Outcome-blind pool construction
# --------------------------------------------------------------------------

def build_pool(runner, env, anchor: Anchor, root: str) -> dict:
    """Reference + 32 alternatives, deduplicated on the executed first-`c`
    actions.  Nothing here may consult the source suffix or any outcome."""
    from lcwm.sampler import sample_chunks

    restore(env, anchor.snapshot)
    runner.reset()          # never sample a pool behind a stale action queue
    obs = env._format_raw_obs(env._env.env._get_observations())
    instr = env.task_description
    o = runner._obs_to_policy_batch(obs, instr)

    ref = sample_chunks(runner.policy, o, 1, seed=reference_key(root, anchor.anchor_id))
    keys = alternative_keys(root, anchor.anchor_id)
    alts = [sample_chunks(runner.policy, o, 1, seed=k) for k in keys]

    def exec_prefix(chunk):
        return np.stack([runner.action_to_env(chunk[0, i]) for i in range(C_PREFIX)])

    ref_x = exec_prefix(ref)
    alt_x = [exec_prefix(a) for a in alts]

    seen, uniq = {}, []
    for i, x in enumerate(alt_x):
        k = np.round(x, 4).tobytes()
        if k not in seen:
            seen[k] = i
            uniq.append(i)
    return {"reference": ref_x, "alternatives": alt_x, "unique_idx": uniq,
            "n_unique": len(uniq), "alt_keys": keys,
            "ref_key": reference_key(root, anchor.anchor_id)}


def select_candidates(pool: dict, root: str, anchor_id: str) -> dict:
    """Three farthest-point-diversity candidates, then three disjoint
    matched-random ones from the remainder.  Standardized action features, a
    presealed start, and a deterministic index tie-break - no outcome input."""
    idx = list(pool["unique_idx"])
    X = np.stack([pool["alternatives"][i].reshape(-1) for i in idx])
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / np.where(sd > 1e-8, sd, 1.0)

    start = int(np.frombuffer(
        __import__("hashlib").sha256(f"{root}|fps|{anchor_id}".encode()).digest()[:4],
        dtype=np.uint32)[0] % len(idx))
    chosen = [start]
    while len(chosen) < N_DIVERSITY:
        d = np.min(np.linalg.norm(Z[:, None] - Z[chosen][None], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.lexsort((np.arange(len(idx)), -d))[0]))
    div = [idx[c] for c in chosen]

    rest = [i for i in idx if i not in div]
    rng = np.random.default_rng(int(np.frombuffer(
        __import__("hashlib").sha256(f"{root}|rand|{anchor_id}".encode()).digest()[:4],
        dtype=np.uint32)[0]))
    rand = [int(i) for i in rng.choice(rest, size=N_RANDOM, replace=False)]
    return {"diversity": div, "random": rand,
            "fps_start": start, "n_pool": len(idx)}


# --------------------------------------------------------------------------
# Paired execution
# --------------------------------------------------------------------------

def run_branch(runner, env, anchor: Anchor, prefix_actions: np.ndarray,
               crn_key: int, cap_prefix: str, cap_cont: str,
               ledger: SegmentLedger, candidate_id: str) -> dict:
    """Restore, execute the sealed 10-action prefix, then replan with stock pi0
    to `H_PRIMARY`, reading outcomes at every registered horizon.  Never steps
    officially past the deadline."""
    segp, segc = f"{candidate_id}-{crn_key}-pre", f"{candidate_id}-{crn_key}-cont"
    ledger.open_segment(segp, cap_prefix, C_PREFIX, anchor_id=anchor.anchor_id,
                        candidate_id=candidate_id, crn_key=crn_key)

    restore(env, anchor.snapshot)
    runner.reset()          # THE critical reset: pi0.5 refills its action queue
                            # only when empty, so without this a branch executes
                            # the previous branch's leftover chunk - and that
                            # happens precisely when a sibling SUCCEEDS early,
                            # which is the outcome the pilot is looking for.
    restore_ok, restored = verify_restore(env, anchor)
    subgoals = V080_TASKS[env.task]["ordered_subgoals"]
    au = (anchor.source_automaton.fork() if anchor.source_automaton is not None
          else GoalAutomaton(subgoals))
    if anchor.source_automaton is None:
        au.start(env)
    atoms = goal_atoms(env)
    instr = env.task_description
    au.evaluate(env, anchor.tau)

    t = anchor.tau
    success_step, done = None, False
    for i in range(C_PREFIX):
        obs, _r, term, trunc, info = env.step(prefix_actions[i])
        t += 1
        done = bool(term or trunc)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if done:
            break
    au.evaluate(env, t)
    # ACHIEVED post-prefix physics: the replay-tolerance test compares this
    # across the three reference repeats.  The previous signature summed the
    # commanded actions, which are identical by construction and so could never
    # detect a replay divergence.
    post_prefix = body_signature(env).tolist()
    ledger.close_segment(segp, cap_prefix, C_PREFIX, t - anchor.tau,
                         anchor_id=anchor.anchor_id, candidate_id=candidate_id,
                         crn_key=crn_key, subrole=cap_prefix,
                         officiality=officiality(t, DEADLINE).value,
                         termination="prefix_done" if not done else "terminated")

    ledger.open_segment(segc, cap_cont, H_PRIMARY, anchor_id=anchor.anchor_id,
                        candidate_id=candidate_id, crn_key=crn_key)
    runner.reset()                  # continuation starts from an empty queue
    _seed_all(crn_key)              # paired CRN starts here, identical per key
    q = getattr(getattr(runner.policy, "_action_queue", None), "__len__", lambda: 0)()
    assert q == 0, f"policy action queue not empty at continuation start: {q}"
    reads, start = {}, t
    steps_cont = 0
    while not done and (t - anchor.tau - C_PREFIX) < H_PRIMARY and t < DEADLINE:
        obs, _r, term, trunc, info = env.step(runner.select_action(obs, instr))
        t += 1
        steps_cont += 1
        done = bool(term or trunc)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if t % 10 == 0 or done:
            au.evaluate(env, t)
            if success_step is None and predicate_bits(env, atoms).all():
                success_step = t
        h = t - anchor.tau - C_PREFIX
        if h in H_READOUTS:
            reads[str(h)] = outcome_vector(
                {int(k): int(v) for k, v in au.events_achieved.items()},
                anchor.tau, t, au.damage_unrecovered(), success_step)
    for h in H_READOUTS:                     # fill readouts cut short by success
        reads.setdefault(str(h), outcome_vector(
            {int(k): int(v) for k, v in au.events_achieved.items()},
            anchor.tau, min(anchor.tau + C_PREFIX + h, t), au.damage_unrecovered(),
            success_step))
    ledger.close_segment(segc, cap_cont, H_PRIMARY, steps_cont,
                         anchor_id=anchor.anchor_id, candidate_id=candidate_id,
                         crn_key=crn_key, subrole=cap_cont,
                         officiality=officiality(t, DEADLINE).value,
                         termination="horizon" if not done else "terminated")
    assert t <= DEADLINE, f"stepped past the deadline: {t} > {DEADLINE}"
    return {"candidate_id": candidate_id, "crn_key": crn_key,
            "restore_verified": bool(restore_ok), "restored_hash": restored,
            "anchor_hash_expected": anchor.content_hash,
            "terminal_step": t, "success_step": success_step, "readouts": reads,
            "post_prefix_signature": post_prefix}
