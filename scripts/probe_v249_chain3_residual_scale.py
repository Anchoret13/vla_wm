#!/usr/bin/env python
"""Can a BOUNDED SUSTAINED residual move chain3's real bottleneck at all?

    python scripts/probe_v249_chain3_residual_scale.py --seeds 48

GOAL ANCHOR (CLAUDE.md). This is the zero-candidate diagnostic that rule 1 of the
placement-axis diagnosis demands runs BEFORE candidates, not after them. It spends
environment steps but trains nothing and selects nothing.

WHAT IT ANSWERS, and why it has to run first. chain3_lr2@750 terminal success is
2/96, but the milestone record of the D0 tape says the task is not uniformly hard:

    milestone   0 pick soup  1 place soup  2 pick sauce  3 place sauce  4 pick cheese
    episodes/96       93           93            94            92             5
    (milestone 5, place cheese: 3/96)

87 of 96 episodes stall at exactly one place - stage 4, picking up the cream
cheese. That is a single identifiable frontier, not diffuse failure.

Two facts have to be established before a world model is asked to improve anything
here, and neither costs a candidate:

  (a) does a sustained residual produce ANY spread in where episodes stop? v229
      showed single-chunk perturbations are corrected away by pi0.5's replanning,
      so the intervention unit must be a policy applied at EVERY chunk. If a
      bounded sustained residual cannot move the stage distribution on chain3,
      no predictor of its consequences can help, and that closes the round for
      one GPU-afternoon instead of after a full collection/training/eval cycle.
  (b) at what scale? D1 collection needs a trust region wide enough that u stops
      being a function of z (v122: on-policy action gain +0.0035, CI crossing 0 -
      action conditioning is unidentifiable on pure on-policy data) yet narrow
      enough that the early milestones survive. That trade-off is a number, and
      this probe measures it rather than guessing it.

PRE-REGISTERED. Arms, scales, seeds and the readout are fixed below before the
run. Every arm is reported whatever it shows; no arm is dropped, and the probe
panel (9000+) is disjoint from D0 collection (8700-8795) and from the historical
evaluation panel (8900-8995), so nothing measured here can be reused as an
evaluation result. The residual actors here are RANDOM state-conditioned
functions, not trained candidates - this measures what the action interface can
reach, not what any method achieves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lcwm.sampler import prefix_forward  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

OUT = REPO / "results" / "v249_chain3_residual_scale"
STATS = OUT / "d0_latent_stats.pt"
C = 10
PROPRIO_KEYS = ("robot_state.eef.pos", "robot_state.eef.quat",
                "robot_state.gripper.qpos", "robot_state.gripper.qvel",
                "robot_state.joints.pos", "robot_state.joints.vel")

#: Pre-registered arms. ``None`` scale is the frozen pi0.5 control. Three
#: independent inits at 0.05 measure init spread at a fixed trust region, which is
#: the quantity v230's fixed-scale pool exists to expose; 0.02/0.10/0.20 bracket it.
ARMS = [
    {"tag": "base", "scale": None, "init": None},
    {"tag": "r002_i0", "scale": 0.02, "init": 0},
    {"tag": "r005_i0", "scale": 0.05, "init": 0},
    {"tag": "r005_i1", "scale": 0.05, "init": 1},
    {"tag": "r005_i2", "scale": 0.05, "init": 2},
    {"tag": "r010_i0", "scale": 0.10, "init": 0},
    {"tag": "r020_i0", "scale": 0.20, "init": 0},
]


def proprio(obs) -> torch.Tensor:
    parts = []
    for k in PROPRIO_KEYS:
        cur = obs
        for piece in k.split("."):
            cur = cur[piece]
        parts.append(torch.as_tensor(np.asarray(cur), dtype=torch.float32).flatten())
    return torch.cat(parts)


def make_random_actor(zdim: int, adim: int, scale: float, init: int) -> ResidualActor:
    """A bounded, state-conditioned, SUSTAINED perturbation policy.

    ``ResidualActor`` zero-initialises its output layer so that a trained residual
    starts as a no-op.  A probe of what the interface can reach needs the opposite,
    so the output layer is re-initialised from a fixed seed.  Nothing here is
    trained; the actor is a fixed random function of the rich latent.

    ``GAIN`` is calibrated, not arbitrary.  At the default init the pre-tanh
    activations have std 0.10, so ``|delta|`` is only 0.077x the nominal scale and
    the nominal scale says nothing about how hard the policy is actually pushed.
    A trained residual saturates its bound instead (v163 measured 96% saturation
    at every scale), so a probe using the default init would report a trust region
    an order of magnitude narrower than the one D1 will collect under.  At
    ``GAIN=10`` the measured ratios on D0 latents are ``|delta| = 0.536 x scale``
    with a per-state spread of ``0.078 x scale``: large enough to be comparable to
    a trained actor, still genuinely state-conditioned rather than a constant bias.
    """
    GAIN = 10.0
    torch.manual_seed(24900 + 17 * init + int(round(scale * 1000)))
    act = ResidualActor(zdim, C, adim, scale=scale)
    last = act.net[-1]
    torch.nn.init.normal_(last.weight, std=GAIN / (last.in_features ** 0.5))
    torch.nn.init.zeros_(last.bias)
    act.eval()
    for p in act.parameters():
        p.requires_grad_(False)
    return act


def stage_reached(events: dict[int, int], n_milestones: int) -> int:
    """Longest achieved PREFIX of the ordered milestone list.

    A later milestone reached without its predecessors is not progress along the
    chain, so the contiguous prefix - not the count - is the readout.
    """
    ks = {int(k) for k in events}
    m = 0
    while m < n_milestones and m in ks:
        m += 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--seeds", type=int, default=48)
    ap.add_argument("--seed-start", type=int, default=9000,
                    help="probe family; disjoint from D0 (8700-8795) and the "
                         "historical evaluation panel (8900-8995)")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of the pre-registered arm tags (for resuming)")
    a = ap.parse_args()

    arms = ARMS if a.arms is None else [x for x in ARMS if x["tag"] in a.arms]
    if a.arms and len(arms) != len(a.arms):
        raise SystemExit(f"unknown arm tag in {a.arms}")

    st = torch.load(STATS, weights_only=False)
    if st["task"] != a.task or st["c"] != C:
        raise SystemExit(f"stats/task mismatch: {st['task']} c={st['c']}")
    mu, sd, zdim, adim = st["mu"], st["sd"], st["latent_dim"], st["adim"]

    L = episode_length(a.task)
    subgoals = V080_TASKS[a.task]["ordered_subgoals"]
    nms = len(subgoals)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"task={a.task} L={L} seeds={seeds[0]}-{seeds[-1]} arms={[x['tag'] for x in arms]}",
          flush=True)

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner  # noqa: E402
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    def latent(obs):
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = prefix_forward(runner.policy, po)
            h = pf.hidden[0].detach().float().cpu()
            msk = pf.pad_masks[0].detach().cpu().bool()
            parts = []
            for lo, hi in ((0, 256), (256, 512)):
                blk = h[lo:hi][msk[lo:hi]]
                if len(blk) == 0:
                    blk = h[lo:hi]
                parts += [blk.mean(0), blk.max(0).values]
            z = torch.cat(parts)
        del pf
        return torch.cat([z, proprio(obs)])

    results, t_start = {}, time.time()
    for arm in arms:
        actor = None
        if arm["scale"] is not None:
            actor = make_random_actor(zdim, adim, arm["scale"], arm["init"])
        rows, steps = [], 0
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            runner.reset()
            obs, _ = env.reset(seed=int(seed))
            au = GoalAutomaton(subgoals)
            au.start(env)
            atoms = goal_atoms(env)
            au.evaluate(env, 0)
            t, done, succ, dmag = 0, False, None, []
            while not done and t < L:
                z_t = latent(obs)
                with torch.no_grad():
                    u = runner.sample_chunk(obs, env.task_description)[0, :C]
                    u = u.detach().float().cpu()
                    if actor is not None:
                        d = actor(((z_t - mu) / sd).unsqueeze(0))[0]
                        dmag.append(float(d.abs().mean()))
                        u = u + d
                for act in runner.chunk_to_env(u):
                    obs, _r, tm, tr, inf = env.step(act)
                    t += 1
                    done = bool(tm or tr)
                    if succ is None and bool(inf.get("is_success", False)):
                        succ = t
                    if done:
                        break
                au.evaluate(env, t)
                if succ is None and predicate_bits(env, atoms).all():
                    succ = t
            steps += t
            ev = {int(k): int(v) for k, v in au.events_achieved.items()}
            rows.append({"seed": int(seed), "success": succ is not None,
                         "success_step": succ, "steps": t, "events": ev,
                         "stage": stage_reached(ev, nms),
                         "mean_abs_delta": float(np.mean(dmag)) if dmag else 0.0})
            print(f"  {arm['tag']} s{seed}: stage={rows[-1]['stage']} "
                  f"succ={rows[-1]['success']!s:5s} t={t} "
                  f"|d|={rows[-1]['mean_abs_delta']:.4f} "
                  f"[{time.time()-t_start:.0f}s]", flush=True)
        stages = [r["stage"] for r in rows]
        summ = {
            "tag": arm["tag"], "scale": arm["scale"], "init": arm["init"],
            "n": len(rows), "env_steps": steps,
            "success": sum(r["success"] for r in rows),
            "stage_hist": {str(k): stages.count(k) for k in range(nms + 1)},
            "mean_stage": float(np.mean(stages)),
            "reach_m4": sum(s >= 5 for s in stages),
            "reach_m3": sum(s >= 4 for s in stages),
            "mean_abs_delta": float(np.mean([r["mean_abs_delta"] for r in rows])),
            "episodes": rows,
        }
        results[arm["tag"]] = summ
        print(f"== {arm['tag']}: stage hist {summ['stage_hist']} "
              f"mean {summ['mean_stage']:.3f} succ {summ['success']}/{summ['n']} "
              f"|d|={summ['mean_abs_delta']:.4f}", flush=True)
        (out / f"arm_{arm['tag']}.json").write_text(json.dumps(summ, indent=2))

    manifest = {
        "utc": stamp, "task": a.task, "horizon": L, "c": C,
        "seed_start": a.seed_start, "seeds": len(seeds),
        "milestones": subgoals,
        "latent": "rich8217", "latent_dim": zdim,
        "stats_source": st["source"],
        "stats_sha256": hashlib.sha256(STATS.read_bytes()).hexdigest(),
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "episodes"}
                 for k, v in results.items()},
        "wallclock_s": time.time() - t_start,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
        "note": "zero-candidate diagnostic: random sustained residuals, nothing "
                "trained or selected; probe panel is not an evaluation panel",
    }
    (out / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
