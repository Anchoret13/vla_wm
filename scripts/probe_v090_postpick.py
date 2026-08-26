#!/usr/bin/env python
"""Ceiling probe at a PHASE-ALIGNED anchor (post-pick, pre-place).

Every V8 anchor so far sat at a fixed `tau=160` under an all-empty event mask,
i.e. BEFORE the pick.  `pi_0`'s failures on the frozen panel are 16/18
already-picked-cannot-place.  We have been correcting a phase the failure does
not occur in.

This probe anchors at the pick event itself and asks the only question that
matters before rebuilding anything: at the state where the failure actually
happens, can a supported alternative 10-action prefix flip the outcome?

Reports the ORACLE ceiling, which upper-bounds any world model.
"""
from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm import v086_bank as B  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.snapshot import restore, snap  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS  # noqa: E402
from lcwm.v080r_panel import make_env_at  # noqa: E402
from lcwm.v085_noisefloor import select_max_spread  # noqa: E402
from lcwm.v086_phase import PhasePotential, q_phi  # noqa: E402
import collect_v086_bank as C  # noqa: E402

SEEDS = tuple(range(4700, 4740))
N_ANCHORS, N_CAND, N_REP, C_PREFIX = 16, 5, 3, 10
OUT = REPO / "results" / "v090_postpick"


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--anchors", type=int, default=N_ANCHORS)
    a = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / stamp; out.mkdir(parents=True, exist_ok=True)
    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=10)
    env = make_env_at(B.TASK, B.DEADLINE); scene = C.Scene(env)
    subgoals = V080_TASKS[B.TASK]["ordered_subgoals"]
    steps_spent, recs = 0, []

    for seed in SEEDS:
        if len(recs) >= a.anchors:
            break
        # --- source: find a failure whose pick happened, and anchor AT the pick
        seed_all(seed); runner.reset()
        obs, _ = env.reset(seed=seed)
        au = GoalAutomaton(subgoals); au.start(env)
        atoms = goal_atoms(env); au.evaluate(env, 0)
        q0, p0, b0 = scene.obs_row(env, obs)
        phi = PhasePotential.at_start(q0, p0, scene.names, scene.atoms)
        rec = phi.step(q0, p0, b0)
        anchor = None; t = 0; done = False; succ = None
        while not done and t < B.DEADLINE:
            obs, _r, term, trunc, info = env.step(runner.select_action(obs, env.task_description))
            t += 1; done = bool(term or trunc)
            qt, pt, bt = scene.obs_row(env, obs)
            rec = phi.step(qt, pt, bt)
            if succ is None and bool(info.get("is_success", False)):
                succ = t
            if t % 10 == 0 or done:
                au.evaluate(env, t)
                # anchor at the FIRST replan boundary after the pick, with room
                if (anchor is None and 0 in au.events_achieved
                        and 1 not in au.events_achieved
                        and B.DEADLINE - t >= C_PREFIX + 20):
                    anchor = {"anchor_id": f"p{seed}", "seed": seed, "tau": t,
                              "snapshot": snap(env, t, "libero_10", 0, seed=seed),
                              "phi": phi.fork(), "phi_tau": float(rec.phi),
                              "automaton": au.fork(),
                              "H": B.DEADLINE - t - C_PREFIX}
        steps_spent += t
        if anchor is None or succ is not None:
            print(f"src {seed}: fail@250={succ is None} usable_anchor={anchor is not None}", flush=True)
            continue

        # --- pool at the phase-aligned anchor, then execute ALL candidates
        restore(env, anchor["snapshot"]); runner.reset()
        o = env._format_raw_obs(env._env.env._get_observations())
        po = runner._obs_to_policy_batch(o, env.task_description)
        pf = prefix_forward(runner.policy, po)
        ref = sample_chunks(runner.policy, po, 1, seed=seed * 7 + 1, prefix=pf)[0, :C_PREFIX].detach().float().cpu()
        raw = sample_chunks(runner.policy, po, 64, seed=seed * 7 + 2, prefix=pf)[:, :C_PREFIX].detach().float().cpu()
        del pf
        refe = runner.chunk_to_env(ref)
        rawe = [runner.chunk_to_env(raw[i]) for i in range(64)]
        chunks = [refe] + [rawe[i] for i in select_max_spread(rawe, refe)]

        H = anchor["H"]
        res = np.zeros((N_CAND, N_REP)); qph = np.zeros((N_CAND, N_REP))
        for ci, ch in enumerate(chunks):
            for ri in range(N_REP):
                restore(env, anchor["snapshot"]); runner.reset()
                ph = anchor["phi"].fork(); phis = [anchor["phi_tau"]]
                obs2 = env._format_raw_obs(env._env.env._get_observations())
                tt, dn, sc = anchor["tau"], False, None
                for i in range(C_PREFIX):
                    obs2, _r, tm, tr, inf = env.step(ch[i]); tt += 1; dn = bool(tm or tr)
                    qq, pp, bb = scene.obs_row(env, obs2)
                    phis.append(float(ph.step(qq, pp, bb).phi))
                    if sc is None and bool(inf.get("is_success", False)):
                        sc = tt
                    if dn: break
                seed_all(seed * 100 + ri)
                n = 0
                while not dn and n < H and tt < B.DEADLINE:
                    obs2, _r, tm, tr, inf = env.step(runner.select_action(obs2, env.task_description))
                    tt += 1; n += 1; dn = bool(tm or tr)
                    qq, pp, bb = scene.obs_row(env, obs2)
                    phis.append(float(ph.step(qq, pp, bb).phi))
                    if sc is None and bool(inf.get("is_success", False)):
                        sc = tt
                steps_spent += (tt - anchor["tau"])
                res[ci, ri] = float(sc is not None and sc <= B.DEADLINE)
                qph[ci, ri] = q_phi(phis, C_PREFIX, H)["QPhi"]
        r = {"anchor": anchor["anchor_id"], "seed": seed, "tau": anchor["tau"], "H": H,
             "succ_rate": res.mean(1).tolist(), "qphi": qph.mean(1).tolist()}
        r["ref_rate"] = r["succ_rate"][0]
        r["best_alt_rate"] = max(r["succ_rate"][1:])
        r["flip"] = r["best_alt_rate"] > r["ref_rate"]
        recs.append(r)
        print(f"{r['anchor']} tau={r['tau']} H={H} ref={r['ref_rate']:.2f} "
              f"best_alt={r['best_alt_rate']:.2f} flip={r['flip']} ({steps_spent} steps)", flush=True)

    flips = sum(x["flip"] for x in recs)
    summary = {"utc": stamp, "anchors": len(recs), "flips": flips,
               "flip_rate": flips / max(len(recs), 1), "env_steps": steps_spent,
               "mean_tau": sum(x["tau"] for x in recs) / max(len(recs), 1),
               "records": recs,
               "comparison": {"pre_pick_tau160_ceiling": 8 / 72}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPHASE-ALIGNED (post-pick) oracle flip ceiling: {flips}/{len(recs)} = "
          f"{flips/max(len(recs),1):.1%}   vs pre-pick tau=160: 8/72 = 11.1%")
    print(f"mean tau {summary['mean_tau']:.0f} | {steps_spent} env steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
