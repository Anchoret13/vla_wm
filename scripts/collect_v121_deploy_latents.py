#!/usr/bin/env python
"""Collect deployment trajectories as (z_t, u_t, z_{t+c}) latent triples.

    python scripts/collect_v121_deploy_latents.py --task chain1b_lr2 --episodes 64

GOAL ANCHOR (CLAUDE.md). Four axes:
  task    chain1b_lr2, pi_0 = 0.438 - a low-success task, reopened because its
          earlier closure was argued from SELECTION failing, which says nothing
          about latent state prediction
  object  supplies the training data for T_theta: z_t rolled forward under the
          action actually executed
  target  the future LATENT z_{t+c}, taken from PrefixVLM at t+c. No pixels, no
          reconstruction anywhere in this pipeline
  data    trajectories collected while the frozen policy is deployed on the task,
          not offline acquisition seeds

The latent is the pooled PrefixVLM prefix hidden under the FULL prompt - the only
conditioning available at deployment. One triple per chunk boundary (c = 10 = the
deployment contract's N=1 chunk), so u_t is exactly the action the policy
committed to. Episode outcome and milestone times are stored alongside for the
p_succ head later.

Nothing is trained here; this only writes the tape.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

import numpy as np, torch  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402

C = 10
OUT = REPO / "results" / "v121_deploy_latents"

# LCState components (framework 5.1). The pooled prefix hidden is a
# language-vision alignment representation; ten steps of action barely move it,
# which is why v122 measured an action gain of only 0.81% of identity and v133
# found within-state ranking AUC ~0.5. Proprioception is the part of the state
# an action actually controls over c steps, and it was missing entirely.
PROPRIO_KEYS = ("robot_state.eef.pos", "robot_state.eef.quat",
                "robot_state.gripper.qpos", "robot_state.gripper.qvel",
                "robot_state.joints.pos", "robot_state.joints.vel")


def proprio(obs) -> torch.Tensor:
    """25-d physical state: eef pose, gripper, joints. Not a reconstruction
    target - a component of the latent, predicted in latent space like the rest."""
    parts = []
    for k in PROPRIO_KEYS:
        cur = obs
        for piece in k.split("."):
            cur = cur[piece]
        parts.append(torch.as_tensor(np.asarray(cur), dtype=torch.float32).flatten())
    return torch.cat(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--rich-latent", action="store_true",
                    help="Keep spatial structure. The default mean-pools all 567 "
                         "valid prefix tokens into one vector, which destroys "
                         "localised visual facts: measured, a probe trained DIRECTLY "
                         "on stage_reached tops out at rho 0.201 on chain3 (bar is "
                         "0.35) and no readout point rescues it - terminal 0.201, "
                         "mean 0.185, endpoints 0.120. chain1b keeps 0.5-0.6 because "
                         "its progress is visible in the arm state, while chain3's "
                         "is 'how many objects are already in the basket', which is "
                         "pure scene state. This pools each camera's 256 tokens "
                         "separately with mean AND max, so presence-of-a-patch "
                         "survives.")
    ap.add_argument("--commit", type=int, default=None,
                    help="environment steps committed per chunk. The contract's "
                         "c = 10 has been fixed for the whole project, so an "
                         "action's effect must survive ten steps of dynamics "
                         "before it reaches the next recorded latent. v226 "
                         "measured the within-state action->outcome ceiling at "
                         "chance under c = 10; this makes c a variable so that "
                         "can be tested rather than assumed.")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0],
                    help="SDE noise levels, drawn per chunk. Default [0.0] is pure "
                         "on-policy. With on-policy data u is nearly a function of "
                         "z, so action conditioning is unidentifiable: v122 measured "
                         "an action gain of +0.0035 with 95% CI [-0.0017, +0.0089] "
                         "over an action-free model. Varying sigma WITHIN episodes "
                         "decorrelates u from z, which is what makes T_theta's "
                         "action input learnable at all.")
    a = ap.parse_args()
    L = episode_length(a.task)
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = OUT / f"{a.task}_{stamp}"; out.mkdir(parents=True, exist_ok=True)
    print(f"task={a.task} L={L} episodes={len(seeds)} c={C}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    global C
    if a.commit:
        C = a.commit
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    print(f"committing {C} env steps per chunk")
    env = make_v080_env(a.task); subgoals = V080_TASKS[a.task]["ordered_subgoals"]

    Z, U, Zn, EP, TT, SG = [], [], [], [], [], []
    rng = np.random.default_rng(a.seed_start)
    episodes, steps, nsucc = [], 0, 0
    for ei, seed in enumerate(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        runner.reset(); obs, _ = env.reset(seed=int(seed))
        au = GoalAutomaton(subgoals); au.start(env); atoms = goal_atoms(env)
        au.evaluate(env, 0)
        t, done, succ = 0, False, None
        prev = None                       # (z_t, u_t) awaiting its z_{t+c}

        def latent():
            po = runner._obs_to_policy_batch(obs, env.task_description)
            with torch.no_grad():
                pf = prefix_forward(runner.policy, po)
                h = pf.hidden[0].detach().float().cpu()
                msk = pf.pad_masks[0].detach().cpu().bool()
                if a.rich_latent:
                    # vision tokens are [0, 512): two cameras of 256 each; language
                    # sits at [768, ...] and is constant within a task
                    parts = []
                    for lo, hi in ((0, 256), (256, 512)):
                        blk = h[lo:hi][msk[lo:hi]]
                        if len(blk) == 0:
                            blk = h[lo:hi]
                        parts += [blk.mean(0), blk.max(0).values]
                    z = torch.cat(parts)
                else:
                    z = masked_prefix_mean(h, msk.float())
            del pf
            return torch.cat([z, proprio(obs)])

        while not done and t < L:
            z_t = latent()
            if prev is not None:          # close the previous triple
                Z.append(prev[0]); U.append(prev[1]); Zn.append(z_t)
                EP.append(ei); TT.append(prev[2]); SG.append(prev[3])
            sg = float(a.sigmas[int(rng.integers(len(a.sigmas)))])
            with torch.no_grad():
                if sg == 0.0:
                    u = runner.sample_chunk(obs, env.task_description
                                            )[0, :C].detach().float().cpu()
                else:
                    po = runner._obs_to_policy_batch(obs, env.task_description)
                    u = sample_chunks(runner.policy, po, 1,
                                      seed=int(seed) * 7919 + t,
                                      sigma=sg)[0, :C].detach().float().cpu()
            prev = (z_t, u, t, sg)
            for act in runner.chunk_to_env(u):
                obs, _r, tm, tr, inf = env.step(act); t += 1
                done = bool(tm or tr)
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                if done: break
            au.evaluate(env, t)
            if succ is None and predicate_bits(env, atoms).all():
                succ = t
        if prev is not None and not done:
            z_t = latent()
            Z.append(prev[0]); U.append(prev[1]); Zn.append(z_t)
            EP.append(ei); TT.append(prev[2]); SG.append(prev[3])
        steps += t; nsucc += int(succ is not None)
        ev = {int(k): int(v) for k, v in au.events_achieved.items()}
        episodes.append({"idx": ei, "seed": int(seed), "success": succ is not None,
                         "success_step": succ, "steps": t, "events": ev})
        if ei % 8 == 0:
            print(f"  [{ei+1}/{len(seeds)}] {len(Z)} triples, "
                  f"{nsucc} succ, {steps} steps", flush=True)

    Zt, Ut, Znt = torch.stack(Z), torch.stack(U), torch.stack(Zn)
    torch.save({"z": Zt, "u": Ut, "z_next": Znt,
                "episode": torch.tensor(EP), "t": torch.tensor(TT),
                "sigma": torch.tensor(SG),
                "success": torch.tensor([float(e["success"]) for e in episodes]),
                "task": a.task, "c": C, "latent_dim": Zt.shape[-1],
                "proprio_dim": 25, "proprio_keys": list(PROPRIO_KEYS)},
               out / "tape.pt")
    (out / "summary.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "c": C, "episodes": len(seeds),
         "seed_range": [seeds[0], seeds[-1]], "sigmas": a.sigmas,
         "successes": nsucc,
         "rate": nsucc / len(seeds), "triples": len(Z),
         "latent_dim": int(Zt.shape[-1]), "proprio_dim": 25,
         "env_steps": steps,
         "episode_records": episodes,
         "note": "deployment trajectories; latent targets only, no reconstruction",
         "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(f"\n{len(Z)} triples, latent dim {Zt.shape[-1]}, "
          f"success {nsucc}/{len(seeds)} = {nsucc/len(seeds):.3f}; "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
