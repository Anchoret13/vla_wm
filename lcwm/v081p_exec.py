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

class SegmentLedger:
    def __init__(self, path: Path, caps: dict[str, int]):
        self.path, self.caps = Path(path), dict(caps)
        self.spent: dict[str, int] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
        else:                       # resume-safe: re-derive from the file
            for line in self.path.read_text().splitlines():
                if line:
                    r = json.loads(line)
                    self.spent[r["cap_line"]] = self.spent.get(r["cap_line"], 0) + r["n_steps"]

    def headroom_ok(self, cap_line: str, worst_case: int) -> tuple[bool, str]:
        """Checked BEFORE a segment runs.  Stage 1R.1 executed a segment and
        only then discovered it had blown the line."""
        have = self.caps[cap_line] - self.spent.get(cap_line, 0)
        return (worst_case <= have,
                f"{cap_line}: worst case {worst_case} vs headroom {have}")

    def charge(self, cap_line: str, n_steps: int, **row) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps({"cap_line": cap_line, "n_steps": int(n_steps),
                                 **row}) + "\n")
        self.spent[cap_line] = self.spent.get(cap_line, 0) + int(n_steps)
        if self.spent[cap_line] > self.caps[cap_line]:
            raise SystemExit(f"HALT: {cap_line} cap {self.caps[cap_line]} "
                             f"exceeded ({self.spent[cap_line]})")

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
    ok, why = ledger.headroom_ok(Subrole.SOURCE.value, DEADLINE)
    if not ok:
        raise SystemExit(f"HALT before segment: {why}")

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
                            good, why_bad,
                            content_hash=_hash_snapshot(snapshot))
    ledger.charge(Subrole.SOURCE.value, t, seed=seed, subrole=Subrole.SOURCE.value,
                  anchor_id=None, officiality=officiality(t, DEADLINE).value,
                  termination="deadline" if not done else "terminated")
    return {"seed": seed, "steps": t, "success_step": success_step,
            "failure_at_L": success_step is None,
            "anchor": anchor, "events": dict(au.events_achieved)}


def mask_check(sk):
    from lcwm.v081p_contract import mask_conformant
    return mask_conformant(sk.ever_achieved, sk.current_valid, sk.damaged,
                           sk.actionable, sk.unresolved)


def _hash_snapshot(s) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(np.asarray(s.state, dtype=np.float64).tobytes())
    for f in ("sim_ctrl", "sim_qfrc_applied", "sim_xfrc_applied",
              "sim_mocap_pos", "sim_mocap_quat", "sim_qacc_warmstart"):
        v = getattr(s, f, None)
        if v is not None:
            h.update(np.asarray(v, dtype=np.float64).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Outcome-blind pool construction
# --------------------------------------------------------------------------

def build_pool(runner, env, anchor: Anchor, root: str) -> dict:
    """Reference + 32 alternatives, deduplicated on the executed first-`c`
    actions.  Nothing here may consult the source suffix or any outcome."""
    from lcwm.sampler import sample_chunks

    restore(env, anchor.snapshot)
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
    for line, worst in ((cap_prefix, C_PREFIX), (cap_cont, H_PRIMARY)):
        ok, why = ledger.headroom_ok(line, worst)
        if not ok:
            raise SystemExit(f"HALT before segment: {why}")

    restore(env, anchor.snapshot)
    restored = _hash_snapshot(anchor.snapshot)
    subgoals = V080_TASKS[env.task]["ordered_subgoals"]
    au = GoalAutomaton(subgoals)
    au.start(env)
    atoms = goal_atoms(env)
    instr = env.task_description
    au.evaluate(env, anchor.tau)

    t = anchor.tau
    success_step, done = None, False
    replay = []
    for i in range(C_PREFIX):
        obs, _r, term, trunc, info = env.step(prefix_actions[i])
        t += 1
        replay.append(float(np.abs(prefix_actions[i]).sum()))
        done = bool(term or trunc)
        if success_step is None and bool(info.get("is_success", False)):
            success_step = t
        if done:
            break
    au.evaluate(env, t)
    ledger.charge(cap_prefix, t - anchor.tau, anchor_id=anchor.anchor_id,
                  candidate_id=candidate_id, crn_key=crn_key,
                  subrole=cap_prefix, officiality=officiality(t, DEADLINE).value,
                  termination="prefix_done" if not done else "terminated")

    _seed_all(crn_key)              # paired CRN starts here, identical per key
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
    ledger.charge(cap_cont, steps_cont, anchor_id=anchor.anchor_id,
                  candidate_id=candidate_id, crn_key=crn_key, subrole=cap_cont,
                  officiality=officiality(t, DEADLINE).value,
                  termination="horizon" if not done else "terminated")
    assert t <= DEADLINE, f"stepped past the deadline: {t} > {DEADLINE}"
    return {"candidate_id": candidate_id, "crn_key": crn_key,
            "restored_hash": restored, "terminal_step": t,
            "success_step": success_step, "readouts": reads,
            "prefix_signature": replay}
