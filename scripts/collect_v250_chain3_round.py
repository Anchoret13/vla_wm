#!/usr/bin/env python
"""EVOLVE-1 chain3 round collector: fixed-horizon, instrumented, residual-capable.

    # D0 - frozen pi0.5, the base deployment tape
    python scripts/collect_v250_chain3_round.py --role base --episodes 96 --seed-start 8700

    # D1 - one sustained residual collector
    python scripts/collect_v250_chain3_round.py --role resid --actor-scale 0.05 \
        --actor-init 0 --residual-start-chunk 26 --episodes 48 --seed-start 8800

GOAL ANCHOR (CLAUDE.md). task: chain3_lr2@750, frozen pi0.5 2/96.  object/target:
this writes (z_t, u_t, z_{t+c}) rich-latent triples - the training data for an
action-conditioned latent transition, no pixels and no reconstruction anywhere.
data: every row comes from an actual deployment rollout, not an offline seed.

WHY A NEW COLLECTOR RATHER THAN collect_v246_evolve1_d1.py.  v246 encodes the
SUPERSEDED Gate-0 campaign in its type system: validate_phi_checkpoint hard-requires
``gate0_status == "PASS"`` (v246:1468), and all five artifact validators require the
literal ``semantic_binding == "mechanical_fixture_not_semantically_bound"``, which a
real 7 GB pi0.5 checkpoint cannot satisfy.  The 2026-09-12 approval removed Gate 0
as a prerequisite, so using v246 today would mean either fabricating a Gate-0 PASS
(forbidden) or rewriting its validators.  This file instead reuses the path that
actually produced the existing D0 tape - collect_v121_deploy_latents.py - and adds
only the three things the round needs.

WHAT IS NEW RELATIVE TO v121, and why each is needed.

1. FIXED HORIZON.  ``ChainEnv.step`` returns ``terminated = done or is_success``
   (lcwm/loho.py:70) but never resets, so v121's ``done = bool(tm or tr)`` ends an
   episode the moment the task is solved.  That censored D0 episodes 4 and 45 to
   59/63 rows instead of 75 and makes row count a leak of the outcome.  Here the
   loop keys off the raw ``info["done"]`` and truncation only; success is recorded
   and never acted on, so every episode yields the same 75 chunks / 76 boundaries.

2. A SUSTAINED RESIDUAL.  v229 measured that a single-chunk perturbation is inert -
   from an identical state, eight different chunks gave identical progress in 71 of
   72 cases, because pi0.5 replans and corrects it away.  The unit of intervention
   that does move chain3 is therefore a POLICY applied at EVERY chunk.  The residual
   here is a fixed, bounded, state-conditioned function of the rich latent, immutable
   within an episode, recomputed each chunk: ``u_exec = u_base + delta(z_t)``.

   ``--residual-start-chunk`` exists because of a measurement, not a hunch.  On the
   D0 tape the frozen policy places two cans by step 260 in 92-94 of 96 episodes and
   then makes NO further progress for a median of 490 of its 750 steps - 63% of the
   budget spent after the last milestone.  Perturbing from t=0 damages the part that
   already works; the informative interventions live in the stall.  The gate is a
   fixed chunk index, i.e. a schedule the deployed policy can evaluate from its own
   step counter - not a privileged state read.

3. INSTRUMENTATION FOR A PROGRESS LABEL.  Binary success (2/96) and the milestone
   index are both near-degenerate here: 88 of 96 D0 episodes end at exactly four
   milestones.  ``lcwm/phase_potential.py`` resolves INSIDE that basin
   (approach/grasp/lift/transport/placement within the active goal atom) but needs
   per-boundary eef pose, object positions and predicate bits, which no chain3 tape
   stores.  They are recorded here at negligible cost.

ARTIFACT SPLIT, which is the leakage contract.

    tape.pt      z, u_exec, z_next, episode, t, sigma  - WM training consumes ONLY this.
                 No success, no events, no policy_id, no episode length variation.
    labels.pt    per-boundary eef/obj_pos/bits + milestone events.  This is the
                 progress-annotation artifact; it stands in for the human stage
                 feedback the 2026-09-12 plan budgets, and its cost is recorded.
    trace.pt     u_base and delta per boundary, for the u_exec = u_base + delta audit.
                 Not a training input.
    outcome.json success and success_step.  Evaluation sidecar only, never a training
                 input, never read by this script's own control flow.

PRE-REGISTERED: the seed family, episode count, residual scale/init and start chunk
are command-line inputs recorded in the manifest before the run; nothing here selects
among arms or reads an outcome.
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

from lcwm.probe_data import body_positions, discover_object_bodies  # noqa: E402
from lcwm.sampler import prefix_forward, sample_chunks  # noqa: E402
from lcwm.seq_data import goal_atoms, predicate_bits  # noqa: E402
from lcwm.task_automaton import GoalAutomaton  # noqa: E402
from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402

OUT = REPO / "results" / "v250_chain3_round"
STATS = REPO / "results" / "v249_chain3_residual_scale" / "d0_latent_stats.pt"
C = 10
PROPRIO_KEYS = ("robot_state.eef.pos", "robot_state.eef.quat",
                "robot_state.gripper.qpos", "robot_state.gripper.qvel",
                "robot_state.joints.pos", "robot_state.joints.vel")
#: Matches make_random_actor in probe_v249: calibrated so |delta| ~ 0.54 * scale
#: on D0 latents, comparable to a trained actor's saturated bound, while keeping a
#: real per-state spread (0.078 * scale) so the residual is a POLICY not a bias.
ACTOR_GAIN = 10.0


def proprio(obs) -> torch.Tensor:
    parts = []
    for k in PROPRIO_KEYS:
        cur = obs
        for piece in k.split("."):
            cur = cur[piece]
        parts.append(torch.as_tensor(np.asarray(cur), dtype=torch.float32).flatten())
    return torch.cat(parts)


def make_collector_actor(zdim: int, adim: int, scale: float, init: int) -> ResidualActor:
    """A fixed, bounded, state-conditioned SUSTAINED perturbation policy.

    A D1 collector's job is to make ``u`` stop being a function of ``z`` so that
    action conditioning becomes identifiable at all - v122 measured the action gain
    on pure on-policy data at +0.0035 with a CI crossing zero.  It is deliberately
    NOT trained: a trained collector would presuppose the answer the round is
    supposed to measure.  Random, fixed, and declared.
    """
    torch.manual_seed(25000 + 17 * init + int(round(scale * 1000)))
    act = ResidualActor(zdim, C, adim, scale=scale)
    last = act.net[-1]
    torch.nn.init.normal_(last.weight, std=ACTOR_GAIN / (last.in_features ** 0.5))
    torch.nn.init.zeros_(last.bias)
    act.eval()
    for p in act.parameters():
        p.requires_grad_(False)
    return act


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--role", choices=["base", "resid"], required=True)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0],
                    help="per-chunk SDE noise for the base role. Pure on-policy "
                         "data makes action conditioning unidentifiable (v122), so "
                         "a base tape intended to train a transition needs spread.")
    ap.add_argument("--actor-scale", type=float, default=None)
    ap.add_argument("--actor-init", type=int, default=0)
    ap.add_argument("--residual-start-chunk", type=int, default=0,
                    help="first chunk index at which the residual is applied; a "
                         "fixed schedule, evaluable from the deployed step counter")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    if a.role == "resid" and a.actor_scale is None:
        raise SystemExit("--role resid requires --actor-scale")
    if a.role == "base" and a.actor_scale is not None:
        raise SystemExit("--role base takes no actor")

    st = torch.load(STATS, weights_only=False)
    if st["task"] != a.task or st["c"] != C:
        raise SystemExit(f"stats/task mismatch: {st['task']} c={st['c']}")
    mu, sd, zdim, adim = st["mu"], st["sd"], st["latent_dim"], st["adim"]

    L = episode_length(a.task)
    n_chunks = L // C
    subgoals = V080_TASKS[a.task]["ordered_subgoals"]
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = a.tag or (a.role if a.role == "base"
                    else f"resid_s{a.actor_scale}_i{a.actor_init}")
    out = OUT / f"{a.task}_{tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"task={a.task} role={a.role} L={L} chunks={n_chunks} "
          f"seeds={seeds[0]}-{seeds[-1]} tag={tag}", flush=True)

    actor = None
    if a.role == "resid":
        actor = make_collector_actor(zdim, adim, a.actor_scale, a.actor_init)

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

    Z, U, ZN, EP, TT, SG = [], [], [], [], [], []
    UB, DL = [], []
    EEF, OBJ, BITS, LEP, LT = [], [], [], [], []
    episodes, steps, t_start = [], 0, time.time()
    object_names = None
    rng = np.random.default_rng(a.seed_start)

    def close_triple(pending, z_next, ep_idx):
        """Close (z_t, u_t) with its z_{t+c}. The executed action goes to the
        tape; the base/residual split goes to the audit trace, never to training."""
        z_t, u_exec, t_t, sigma_t, u_base, delta = pending
        Z.append(z_t)
        U.append(u_exec)
        ZN.append(z_next)
        EP.append(ep_idx)
        TT.append(t_t)
        SG.append(sigma_t)
        UB.append(u_base)
        DL.append(delta)

    for ei, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner.reset()
        obs, _ = env.reset(seed=int(seed))
        bodies = discover_object_bodies(env)
        if object_names is None:
            object_names = sorted(bodies)
        elif sorted(bodies) != object_names:
            raise SystemExit(f"object set changed at ep {ei}: {sorted(bodies)}")
        body_list = [bodies[n] for n in object_names]
        atoms = goal_atoms(env)
        au = GoalAutomaton(subgoals)
        au.start(env)
        au.evaluate(env, 0)
        t, raw_done, succ = 0, False, None
        prev = None
        n_applied = 0

        def record_label(step_t):
            EEF.append(proprio(obs).clone())
            OBJ.append(torch.as_tensor(body_positions(env, body_list),
                                       dtype=torch.float32))
            BITS.append(torch.as_tensor(predicate_bits(env, atoms).copy()))
            LEP.append(ei)
            LT.append(step_t)

        for k in range(n_chunks):
            if raw_done:
                break
            z_t = latent(obs)
            record_label(t)
            if prev is not None:
                close_triple(prev, z_t, ei)
            sg = float(a.sigmas[int(rng.integers(len(a.sigmas)))]) \
                if a.role == "base" else 0.0
            with torch.no_grad():
                if sg == 0.0:
                    u_base = runner.sample_chunk(
                        obs, env.task_description)[0, :C].detach().float().cpu()
                else:
                    po = runner._obs_to_policy_batch(obs, env.task_description)
                    u_base = sample_chunks(runner.policy, po, 1,
                                           seed=int(seed) * 7919 + t,
                                           sigma=sg)[0, :C].detach().float().cpu()
                if actor is not None and k >= a.residual_start_chunk:
                    delta = actor(((z_t - mu) / sd).unsqueeze(0))[0]
                    n_applied += 1
                else:
                    delta = torch.zeros_like(u_base)
            u_exec = u_base + delta
            prev = (z_t, u_exec, t, sg, u_base, delta)
            for act in runner.chunk_to_env(u_exec):
                obs, _r, _tm, tr, inf = env.step(act)
                t += 1
                # success is RECORDED, never acted on: the horizon is fixed so that
                # row count carries no outcome information
                if succ is None and bool(inf.get("is_success", False)):
                    succ = t
                # `info["done"]` is SUCCESS-DRIVEN on this task. Measured 2026-09-12:
                # on seed 9018 `done` first fired at t=678, the same step as
                # `is_success`, and an earlier version of this loop that broke on it
                # produced a 678-step episode - i.e. row count still leaked the
                # outcome, the exact defect the fixed horizon exists to remove.
                # Stepping past it is safe: the same seed then ran the full 750 steps
                # and 75 chunks with predicates still readable ([True, True, True])
                # and the milestone record intact. So only a done that is NOT a
                # success, or a truncation, ends the episode.
                if (bool(inf.get("done", False))
                        and not bool(inf.get("is_success", False))) or bool(tr):
                    raw_done = True
                    break
            au.evaluate(env, t)
            if succ is None and predicate_bits(env, atoms).all():
                succ = t

        if prev is not None:
            z_t = latent(obs)
            record_label(t)
            close_triple(prev, z_t, ei)
        steps += t
        ev = {int(k_): int(v) for k_, v in au.events_achieved.items()}
        episodes.append({"idx": ei, "seed": int(seed), "success": succ is not None,
                         "success_step": succ, "steps": t, "events": ev,
                         "chunks_with_residual": n_applied,
                         "raw_done_early": bool(raw_done and t < L)})
        print(f"  [{ei+1}/{len(seeds)}] s{seed} t={t} rows={len(Z)} "
              f"ms={sorted(ev)} succ={succ is not None} "
              f"[{time.time()-t_start:.0f}s]", flush=True)

    Zt, Ut, ZNt = torch.stack(Z), torch.stack(U), torch.stack(ZN)
    full = [e for e in episodes if e["steps"] == L]
    torch.save({"z": Zt, "u": Ut, "z_next": ZNt,
                "episode": torch.tensor(EP), "t": torch.tensor(TT),
                "sigma": torch.tensor(SG),
                "task": a.task, "c": C, "latent_dim": int(Zt.shape[-1]),
                "proprio_dim": 25, "proprio_keys": list(PROPRIO_KEYS)},
               out / "tape.pt")
    torch.save({"eef_proprio": torch.stack(EEF), "obj_pos": torch.stack(OBJ),
                "bits": torch.stack(BITS), "episode": torch.tensor(LEP),
                "t": torch.tensor(LT), "object_names": object_names,
                "goal_atoms": atoms, "milestones": subgoals,
                "events": {e["idx"]: e["events"] for e in episodes},
                "task": a.task, "c": C,
                "provenance": "scripted privileged stand-in for a human progress "
                              "annotation; NOT an observation-only deployable signal",
                "budget_boundaries": len(LEP), "budget_episodes": len(episodes)},
               out / "labels.pt")
    torch.save({"u_base": torch.stack(UB), "delta": torch.stack(DL),
                "episode": torch.tensor(EP), "t": torch.tensor(TT)},
               out / "trace.pt")
    (out / "outcome.json").write_text(json.dumps(
        {"utc": stamp, "task": a.task, "role": a.role, "tag": tag,
         "successes": sum(e["success"] for e in episodes),
         "n": len(episodes), "episodes": episodes}, indent=2))

    dl = torch.stack(DL)
    applied = dl.abs().sum((1, 2)) > 0
    manifest = {
        "utc": stamp, "task": a.task, "role": a.role, "tag": tag,
        "horizon": L, "c": C, "chunks_per_episode": n_chunks,
        "seed_start": a.seed_start, "episodes": len(seeds),
        "seed_range": [seeds[0], seeds[-1]],
        "sigmas": a.sigmas if a.role == "base" else [0.0],
        "actor_scale": a.actor_scale, "actor_init": a.actor_init,
        "actor_gain": ACTOR_GAIN if actor is not None else None,
        "residual_start_chunk": a.residual_start_chunk,
        "rows": int(Zt.shape[0]), "latent_dim": int(Zt.shape[-1]),
        "label_boundaries": len(LEP),
        "episodes_full_horizon": len(full),
        "rows_with_residual": int(applied.sum()),
        "mean_abs_delta_applied": float(dl[applied].abs().mean()) if applied.any() else 0.0,
        "mean_abs_u_base": float(torch.stack(UB).abs().mean()),
        "env_steps": steps, "wallclock_s": time.time() - t_start,
        "stats_sha256": hashlib.sha256(STATS.read_bytes()).hexdigest(),
        "tape_sha256": hashlib.sha256((out / "tape.pt").read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256((out / "labels.pt").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
        "note": "fixed-horizon deployment collection; tape.pt carries no outcome "
                "field and every episode has the same row count",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{int(Zt.shape[0])} rows, {len(full)}/{len(episodes)} full-horizon, "
          f"succ {manifest['episodes'] and sum(e['success'] for e in episodes)}"
          f"/{len(episodes)}, |delta| {manifest['mean_abs_delta_applied']:.4f} "
          f"vs |u| {manifest['mean_abs_u_base']:.4f}\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
