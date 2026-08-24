#!/usr/bin/env python
"""Action 2M.4 — collect B_boot-v3, the continuous-effect bank.

    python scripts/collect_v086_bank.py --plan
    python scripts/collect_v086_bank.py --run
    V086_SMOKE=1 python scripts/collect_v086_bank.py --run   # 1 group/split

Phi is evaluated at EVERY environment step, its memos are built on the source
episode through tau, and each candidate/repeat forks those memos with the
snapshot.  Nothing is rebased at the anchor.
"""
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import restore, snap  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v081p_exec import SegmentLedger, _hash_snapshot, verify_restore  # noqa: E402
from lcwm.v085_noisefloor import select_max_spread  # noqa: E402
from lcwm.v086_phase import PhasePotential, q_phi  # noqa: E402

OUT_ROOT = REPO / "results" / "v086_bank"
SMOKE = bool(os.environ.get("V086_SMOKE"))
QUOTAS = {"train": 1, "val": 1, "test": 1} if SMOKE else dict(B.QUOTAS)
SRC_LINE, BR_LINE = "source", "branch"


def _seed_all(s: int) -> None:
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _key(root: str, *p) -> int:
    h = hashlib.sha256("|".join([root, *map(str, p)]).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 31 - 1)


def _q(obs) -> np.ndarray:
    r = obs["robot_state"]
    return np.concatenate([r["eef"]["pos"], r["eef"]["quat"], r["gripper"]["qpos"]])


class Scene:
    """Per-step Phi inputs, resolved once per env construction."""

    def __init__(self, env):
        # LiberoEnv builds its inner robosuite env lazily; goal atoms and body
        # names are unavailable until then.
        if getattr(env, "_env", None) is None:
            env._ensure_env()
        self.bodies = discover_object_bodies(env)
        self.names = sorted(self.bodies)
        self.rows = [self.bodies[n] for n in self.names]
        self.atoms = [list(a) for a in goal_atoms(env)]

    def obs_row(self, env, obs):
        return (torch.from_numpy(_q(obs)).float(),
                torch.from_numpy(body_positions(env, self.rows)).float(),
                torch.from_numpy(predicate_bits(env, self.atoms)))


def run_source(runner, env, scene, seed, ledger):
    """Roll to the deadline, recording EVERY step, and fork Phi's memos at tau."""
    seg = f"src-{seed}"
    ledger.open_segment(seg, SRC_LINE, B.DEADLINE, seed=seed, subrole=SRC_LINE)
    _seed_all(seed); runner.reset()
    obs, _ = env.reset(seed=seed)
    subgoals = V080_TASKS[env.task]["ordered_subgoals"]
    au = GoalAutomaton(subgoals); au.start(env)
    q0, p0, b0 = scene.obs_row(env, obs)
    phi = PhasePotential.at_start(q0, p0, scene.names, scene.atoms)
    rec = phi.step(q0, p0, b0)
    au.evaluate(env, 0)

    anchor = None
    t, done, succ = 0, False, None
    try:
        while not done and t < B.DEADLINE:
            obs, _r, term, trunc, info = env.step(runner.select_action(obs, env.task_description))
            t += 1
            done = bool(term or trunc)
            qt, pt, bt = scene.obs_row(env, obs)
            rec = phi.step(qt, pt, bt)                # every step, no stride
            if succ is None and bool(info.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
            if t == B.TAU and succ is None:
                ever = sorted(i for i, s in au.events_achieved.items() if s <= t)
                valid = [i for i, v in enumerate(au.prev_valid) if v]
                if not ever and not valid:            # the registered eligible mask
                    anchor = {"anchor_id": f"a{seed}", "seed": seed,
                              "snapshot": snap(env, t, "libero_10", 0, seed=seed),
                              "phi": phi.fork(), "phi_tau": float(rec.phi),
                              "automaton": au.fork()}
    finally:
        ledger.close_segment(seg, SRC_LINE, B.DEADLINE, t, seed=seed,
                             subrole=SRC_LINE, termination="deadline" if not done else "term")
    # An anchor is eligible only when the source ends in failure@250 AND matched
    # the exact event mask at tau.  The snapshot is taken at tau because that is
    # when the state exists, but the source's terminal outcome is only known at
    # the deadline - so the anchor is released here, not at tau.  Dropping this
    # condition filled a whole bank (DISCARDED_2026-08-24T074430Z) with anchors
    # from episodes that later succeeded.
    return {"seed": seed, "steps": t, "failure_at_L": succ is None,
            "anchor": anchor if succ is None else None}


def run_branch(runner, env, scene, anchor, prefix, crn_key, ledger, cid):
    """Restore, fork Phi's memos, execute the prefix, then continue - Phi at
    every step of both.  The fork is what keeps approach/attachment/lift/
    transport state from being rebased at tau."""
    segp, segc = f"{cid}-{crn_key}-p", f"{cid}-{crn_key}-c"
    ledger.open_segment(segp, BR_LINE, B.C_PREFIX, anchor_id=anchor["anchor_id"],
                        candidate_id=cid, crn_key=crn_key)
    restore(env, anchor["snapshot"]); runner.reset()
    ok, _hash = verify_restore(env, type("A", (), {
        "tau": B.TAU, "content_hash": _hash_snapshot(anchor["snapshot"])})())
    phi = anchor["phi"].fork()
    au = anchor["automaton"].fork()
    obs = env._format_raw_obs(env._env.env._get_observations())
    phis = [anchor["phi_tau"]]

    t, done, succ = B.TAU, False, None
    for i in range(B.C_PREFIX):
        obs, _r, term, trunc, info = env.step(prefix[i]); t += 1
        done = bool(term or trunc)
        qt, pt, bt = scene.obs_row(env, obs)
        phis.append(float(phi.step(qt, pt, bt).phi))
        if succ is None and bool(info.get("is_success", False)):
            succ = t
        if done:
            break
    au.evaluate(env, t)
    delta24 = torch.from_numpy(body_positions(env, scene.rows)).float().reshape(-1)[:24]
    ledger.close_segment(segp, BR_LINE, B.C_PREFIX, t - B.TAU,
                         anchor_id=anchor["anchor_id"], candidate_id=cid,
                         crn_key=crn_key, subrole="prefix")

    ledger.open_segment(segc, BR_LINE, B.H, anchor_id=anchor["anchor_id"],
                        candidate_id=cid, crn_key=crn_key)
    runner.reset(); _seed_all(crn_key)
    n = 0
    while not done and n < B.H and t < B.DEADLINE:
        obs, _r, term, trunc, info = env.step(runner.select_action(obs, env.task_description))
        t += 1; n += 1
        done = bool(term or trunc)
        qt, pt, bt = scene.obs_row(env, obs)
        phis.append(float(phi.step(qt, pt, bt).phi))
        if succ is None and bool(info.get("is_success", False)):
            succ = t
        if t % 10 == 0 or done:
            au.evaluate(env, t)
    while len(phis) < 1 + B.C_PREFIX + B.H:      # hold Phi after an early terminal
        phis.append(phis[-1])
    ledger.close_segment(segc, BR_LINE, B.H, n, anchor_id=anchor["anchor_id"],
                         candidate_id=cid, crn_key=crn_key, subrole="cont")
    assert t <= B.DEADLINE

    ev = {int(k): int(v) for k, v in au.events_achieved.items()}
    start, end = B.TAU + B.C_PREFIX, B.TAU + B.C_PREFIX + B.H
    gains = {i: s for i, s in ev.items() if start < s <= end}
    legacy = {"dp": float(len(gains)),
              "ttm": float(min(gains.values()) - start if gains else B.H + 1),
              "G": float(sum(B.GAMMA ** (s - start) for s in gains.values()))}
    return {"restore_ok": bool(ok), "phis": phis, "physical": delta24,
            "continuous": q_phi(phis, B.C_PREFIX, B.H, B.GAMMA),
            "binary": {"dmg": float(au.damage_unrecovered()),
                       "succ": float(succ is not None and succ <= B.DEADLINE)},
            "legacy": legacy, "terminal_step": t}


def build_group(runner, env, scene, anchor, split, root, ledger):
    restore(env, anchor["snapshot"]); runner.reset()
    obs = env._format_raw_obs(env._env.env._get_observations())
    po = runner._obs_to_policy_batch(obs, env.task_description)
    pf = prefix_forward(runner.policy, po)
    hidden = pf.hidden[0].detach().to("cpu", torch.float16)
    hmask = pf.pad_masks[0].detach().to("cpu", torch.bool)
    ref = sample_chunks(runner.policy, po, 1, seed=_key(root, anchor["anchor_id"], "ref"),
                        prefix=pf)[0, :B.C_PREFIX].detach().float().cpu()
    raw = sample_chunks(runner.policy, po, B.N_RAW_DRAWS,
                        seed=_key(root, anchor["anchor_id"], "alt"),
                        prefix=pf)[:, :B.C_PREFIX].detach().float().cpu()
    del pf
    ref_env = runner.chunk_to_env(ref)
    raw_env = [runner.chunk_to_env(raw[i]) for i in range(B.N_RAW_DRAWS)]
    pick = select_max_spread(raw_env, ref_env)
    chunks = [ref_env] + [raw_env[i] for i in pick]
    ids = ["reference"] + [f"alt{i}" for i in range(B.N_ALTERNATIVES)]
    R = B.REPEATS[split]
    keys = [_key(root, "crn", anchor["anchor_id"], j) for j in range(R)]

    cont = torch.zeros(B.N_CANDIDATES, R, len(B.CONTINUOUS_FIELDS))
    binr = torch.zeros(B.N_CANDIDATES, R, len(B.BINARY_FIELDS))
    leg = torch.zeros(B.N_CANDIDATES, R, len(B.SECONDARY_LEGACY))
    phys = torch.zeros(B.N_CANDIDATES, R, 24)
    mask = torch.zeros(B.N_CANDIDATES, R, dtype=torch.bool)
    traces, restores = [], 0
    for ci, (cid, chunk) in enumerate(zip(ids, chunks)):
        row = []
        for ri, k in enumerate(keys):
            r = run_branch(runner, env, scene, anchor, np.asarray(chunk), k, ledger, cid)
            restores += int(r["restore_ok"])
            for fi, f in enumerate(B.CONTINUOUS_FIELDS):
                cont[ci, ri, fi] = r["continuous"][f]
            for fi, f in enumerate(B.BINARY_FIELDS):
                binr[ci, ri, fi] = r["binary"][f]
            for fi, f in enumerate(B.SECONDARY_LEGACY):
                leg[ci, ri, fi] = r["legacy"][f]
            phys[ci, ri] = r["physical"]
            mask[ci, ri] = True
            row.append(r["phis"])
        traces.append(row)
    return {"schema": B.SCHEMA, "anchor_id": anchor["anchor_id"],
            "source_id": f"{B.TASK}:{anchor['seed']}", "split": split,
            "role": B.SPLIT_ROLES[split], "prefix_hidden": hidden,
            "prefix_mask": hmask,
            "anchor_signature": torch.tensor(
                body_positions(env, scene.rows), dtype=torch.float32).reshape(-1)[:24],
            "actions_norm": torch.stack([ref] + [raw[i] for i in pick]),
            "actions_env": torch.from_numpy(np.stack(chunks)).float(),
            "candidate_ids": ids,
            "is_reference": torch.tensor([True] + [False] * B.N_ALTERNATIVES),
            "post_prefix_delta": phys, "continuous": cont, "binary": binr,
            "legacy": leg, "phi_trace": traces, "phi_tau": anchor["phi_tau"],
            "repeat_mask": mask, "crn_keys": keys, "selected": [int(i) for i in pick],
            "restores_ok": restores}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    g = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                       text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    plan = {"action": "2M.4", "bank": "B_boot-v3", "utc": stamp, "git": g,
            "git_dirty": dirty, "smoke": SMOKE, "quotas": QUOTAS,
            "repeats": B.REPEATS, "candidates": B.N_CANDIDATES,
            "branches": B.BRANCHES, "cap": B.INTERACTION_CAP,
            "cap_total": B.INTERACTION_CAP_TOTAL,
            "queues": {k: [v[0], v[-1], len(v)] for k, v in B.QUEUES.items()},
            "primary_target": B.PRIMARY_TARGET, "no_early_stop": B.NO_EARLY_STOP}
    if a.plan:
        print(json.dumps(plan, indent=2)); return 0
    if dirty and not SMOKE:
        raise SystemExit("commit before the environment run")

    out = OUT_ROOT / (f"SMOKE_{stamp}" if SMOKE else stamp)
    out.mkdir(parents=True, exist_ok=True)
    root = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    plan["root"] = root
    (out / "manifest.json").write_text(json.dumps({**plan, "status": "RUNNING"}, indent=2))
    ledger = SegmentLedger(out / "segment_ledger.jsonl", B.INTERACTION_CAP,
                           action_root=OUT_ROOT)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(B.TASK, B.DEADLINE)
    scene = Scene(env)

    groups, sources = [], []
    for split in ("train", "val", "test"):
        need, got = QUOTAS[split], 0
        for seed in B.QUEUES[split]:
            if got >= need:
                break
            s = run_source(runner, env, scene, seed, ledger)
            sources.append({"seed": seed, "split": split,
                            "failure_at_L": s["failure_at_L"],
                            "eligible": s["anchor"] is not None})
            if s["anchor"] is None:
                continue
            grp = build_group(runner, env, scene, s["anchor"], split, root, ledger)
            B.validate_group(grp)
            groups.append(grp); got += 1
            print(f"[{split}] {grp['anchor_id']} {got}/{need} "
                  f"restores {grp['restores_ok']}/{B.N_CANDIDATES * B.REPEATS[split]} "
                  f"({ledger.total}/{B.INTERACTION_CAP_TOTAL})", flush=True)
        if got < need:
            (out / "HALT.json").write_text(json.dumps(
                {"reason": "source underfill", "split": split, "got": got,
                 "need": need}, indent=2))
            raise SystemExit(f"HALT: {split} filled {got}/{need}")
    (out / "sources.json").write_text(json.dumps(sources, indent=2))
    torch.save({"schema": B.SCHEMA, "groups": groups, "root": root}, out / "groups.pt")
    (out / "manifest.json").write_text(json.dumps({**plan, "status": "COMPLETE"}, indent=2))
    print(json.dumps({"groups": len(groups), "sources": len(sources),
                      "branches": sum(B.N_CANDIDATES * B.REPEATS[g["split"]]
                                      for g in groups),
                      "steps": ledger.total, "cap": B.INTERACTION_CAP_TOTAL,
                      "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
