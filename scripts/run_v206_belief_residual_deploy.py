#!/usr/bin/env python
"""Deploy frozen pi0.5 + a bounded residual; one loop for every arm of a round.

    # residual arm - the latent encoding is inferred from the actor's own dims
    python scripts/run_v206_belief_residual_deploy.py --actor <actor.pt> \
        --task chain3_lr2 --panel 96 --panel-start 9000 --eval-labels --tag m1

    # matched frozen-pi0.5 arm: the SAME loop with Delta := 0
    python scripts/run_v206_belief_residual_deploy.py --actor <actor.pt> \
        --task chain3_lr2 --panel 96 --panel-start 9000 --eval-labels \
        --zero-residual --tag base

GOAL ANCHOR (CLAUDE.md, five axes).
  task       chain3_lr2 @ 750 - frozen pi0.5 terminal success 2/96, with 87/96
             episodes stopping at exactly 4 of the 6 ordered milestones
  object     T_th(b_t, E_a(u)) rolled forward under the CANDIDATE action; this
             script only executes the residual that optimisation produced
  target     future LATENT; no pixel or observation reconstruction on this path
  data       deployment rollouts only
  placement  the residual sits between the frozen policy and the environment

WHAT THIS SCRIPT MEASURES.  Terminal success on a pre-registered seed panel and,
under --eval-labels, the per-episode milestone set, the contiguous-prefix stage,
the last-progress step, the idle tail, and the per-boundary eef / object-pose /
goal-predicate record that lcwm/v250_progress.py turns into Phi' offline.  Every
one of those is read out after the rollout; none of them is visible to the actor.

WHAT WOULD OVERTURN A POSITIVE READING, all on the same panel and the same loop:
the --zero-residual arm matching the residual arm, or a no-rollout / model-free
actor of the same interface capacity matching it.

WHAT IS PRE-REGISTERED.  --panel-start and --panel fix the evaluation seeds before
the run, and SPENT_RANGES - the historical fit seeds plus the 2026-09-12 round's own
D0/D1 collection seeds - is asserted disjoint from the panel.  That assertion is a
backstop for a direct invocation; scripts/eval_v251_round.py carries the task-scoped
seed-family registry and is the authority when the round is driven from there.

THREE ADDITIONS on 2026-09-12, each against a measured blocker.

1. RICH LATENT (--rich-latent / --pooled-latent; inferred from the checkpoint by
   default).  The pooled encoding here is masked_prefix_mean + 25-d proprio =
   2073-d; the chain3 round trains on the 8217-d rich encoding written by
   scripts/collect_v250_chain3_round.py (per camera block [0,256) and [256,512):
   masked mean AND masked max of the prefix hidden, four 2048-d vectors, then the
   same 25-d proprio).  Deploying an 8217-d actor through the pooled path would
   compute (o - mu)/sd with o at 2073 against mu at 8217.  encode_latent() below
   reproduces the collector's function; resolve_latent_mode() picks the encoding
   from the checkpoint's own dims and the observed prefix width, the two flags
   only override it, and a disagreement raises at the first chunk of episode 0
   rather than mid-panel.

2. EVALUATION INSTRUMENTATION (--eval-labels, off by default).  chain3's binary
   rate is 2/96, and against that base at n=96 per arm the smallest significant
   increase is 9/96 - a 4.5x rate increase - under a one-sided Fisher exact test
   (p = 0.029; 8/96 = 4.0x gives 0.050 and 7/96 = 3.5x gives 0.085), so a four-arm
   comparison read out on binary success alone is close to blind.  That arithmetic
   is a property of the panel size, not of any arm measured here.
   The sidecar adds the milestone dict from GoalAutomaton over
   V080_TASKS[task]['ordered_subgoals'], the contiguous-prefix stage (a milestone
   reached without its predecessors is not progress along the chain), the
   last-progress step and the idle tail (the measured failure mode is a median
   490-step stall out of 750), and the per-boundary rows Phi' needs.

3. FIXED HORIZON (--fixed-horizon, off by default).  ChainEnv.step returns
   terminated = done or is_success (lcwm/loho.py:70), so the default loop ends an
   episode at success and the surviving boundary count is a function of the
   outcome.  That is harmless for a binary readout and is a bias for any
   per-boundary average, so an arm comparison read on Phi' should set this flag on
   every arm; it keys off info["done"] and truncation only, as the collector does.

CODE-PATH IDENTITY, which is the only thing that makes --zero-residual a usable
base number.  CLAUDE.md rule 5: a reference number must come from the same
pipeline - on 2026-08-31 a figure from another script was used as the bar to clear
for a whole session and every judgement made against it was void.  All arms enter
run_episode(); --zero-residual sets Delta := 0 AFTER the actor forward, so the
observation batch, prefix_forward, sample_chunk, the latent build, the belief
update and the actor forward are all still executed in the same order and consume
the same RNG.  run_episode asserts torch.equal(executed, chunk) in that arm, so
"the base arm is the residual arm with a zero residual" is checked at run time
rather than asserted in prose.

LEAKAGE CONTRACT.  eval_labels.pt and outcome.json are evaluation readouts:
build_eval_labels() refuses to emit any of TRAINING_TAPE_FIELDS = (z, u, z_next),
the fields the world-model trainers consume.  --collect-tape is unchanged and
still writes a per-episode `success` field into tape.pt, so that tape is not a
valid training artifact for a round whose contract says the tape carries no
outcome - scripts/collect_v250_chain3_round.py writes that one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lcwm.v082_m0 import masked_prefix_mean  # noqa: E402
from lcwm.v08r_contract import clopper_pearson_lower, clopper_pearson_upper  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v205_action_belief import ActionBelief  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402

C = 10
PROPRIO_DIM = 25
#: Vision tokens occupy [0, 512): two cameras of 256 tokens each.  Language sits
#: above 768 and is constant within a task.  Same split as
#: collect_v121_deploy_latents.py:133-144 and collect_v250_chain3_round.py.
CAMERA_BLOCKS = ((0, 256), (256, 512))
#: The fields a world-model trainer reads out of a tape (train_v205_action_belief
#: load_episodes, train_v220_raw_latent_wm).  No evaluation sidecar may carry them.
TRAINING_TAPE_FIELDS = ("z", "u", "z_next")
OUT = REPO / "results" / "v206_belief_residual"
#: Seed ranges an evaluation panel may not touch: the historical fit ranges, plus
#: the 2026-09-12 chain3 round's own deployment collection (D0 8700-8795 from
#: `collect_v250_chain3_round.py --role base --episodes 96 --seed-start 8700`, D1
#: 8800-8847 from `--role resid --episodes 48 --seed-start 8800`).  Without the
#: latter two a direct `--panel-start 8700` would evaluate the arms on the seeds
#: their world model and interface were trained on and the assertion below would
#: still pass.  scripts/eval_v251_round.py's KNOWN_FAMILIES is the maintained
#: registry; this list is the executor's own backstop and must not fall behind it.
SPENT_RANGES = [(6000, 6064), (6200, 6216), (6300, 6396), (6500, 6548),
                (6700, 6764), (6800, 6864), (6900, 7028), (7100, 7228),
                (7300, 7396), (8700, 8796), (8800, 8848)]
#: Kept as the historical name; `run_v163_residual_deploy.py` uses the same list.
FIT_RANGES = SPENT_RANGES


# --------------------------------------------------------------------------- #
# pure helpers - no simulator, no policy, importable on a CPU-only box
# --------------------------------------------------------------------------- #
def encode_latent(hidden: torch.Tensor, mask: torch.Tensor, rich: bool) -> torch.Tensor:
    """The VLA-side latent for one prefix, WITHOUT the proprio block.

    ``rich`` reproduces collect_v250_chain3_round.py's ``latent``: per camera
    block, the masked mean and the masked max of the prefix hidden, concatenated.
    ``not rich`` is the historical pooled vector, masked_prefix_mean over the whole
    prefix.  A divergence here silently changes the latent the actor sees relative
    to the one it was trained on, so the two implementations must stay identical.
    """
    h = hidden.detach().float().cpu()
    msk = mask.detach().cpu().bool()
    if not rich:
        return masked_prefix_mean(h, msk)
    parts: list[torch.Tensor] = []
    for lo, hi in CAMERA_BLOCKS:
        blk = h[lo:hi][msk[lo:hi]]
        if len(blk) == 0:
            blk = h[lo:hi]
        parts += [blk.mean(0), blk.max(0).values]
    return torch.cat(parts)


def latent_dims(hidden_dim: int) -> tuple[int, int]:
    """(pooled, rich) full latent widths for a prefix of ``hidden_dim`` features."""
    return (hidden_dim + PROPRIO_DIM,
            2 * len(CAMERA_BLOCKS) * hidden_dim + PROPRIO_DIM)


def resolve_latent_mode(ck_zdim: int, hidden_dim: int,
                        forced: bool | None = None) -> bool:
    """Pick the encoding from the checkpoint's dims; ``forced`` only overrides.

    Returns True for the rich encoding.  Raises when the checkpoint agrees with
    neither encoding, or when an explicit flag contradicts it - the actor would
    otherwise read a latent it was never trained on.
    """
    pooled, rich = latent_dims(hidden_dim)
    if forced is None:
        if ck_zdim == pooled:
            return False
        if ck_zdim == rich:
            return True
        raise SystemExit(
            f"actor latent dim {ck_zdim} matches neither the pooled ({pooled}) nor "
            f"the rich ({rich}) encoding of a {hidden_dim}-d prefix; pass "
            f"--rich-latent/--pooled-latent only if you know which one it is")
    want = rich if forced else pooled
    if ck_zdim != want:
        raise SystemExit(
            f"--{'rich' if forced else 'pooled'}-latent asks for a {want}-d latent "
            f"but the actor checkpoint declares {ck_zdim}-d")
    return bool(forced)


def final_atom_reach(eef_rows: list, obj_rows: list, bit_rows: list,
                     t_rows: list, object_names: list, goal_atoms: list,
                     basin_from_step: int) -> dict:
    """Closest approach to the FINAL goal object, over boundaries where it is active.

    THE RESTRICTION IS THE POINT.  Taking the minimum of "distance to the active
    object" over the whole stall silently measures the gripper standing next to a CAN
    during the tail of that can's placement: on the 2026-09-12 sigma=0 base arm the
    unrestricted readout reported 9/24 episodes within 5 cm of "the active object"
    while the restricted one reported 0/24, with medians 0.365 m and 0.502 m.  The two
    answers point opposite ways - "reaches the object but cannot grasp it" versus
    "never approaches it" - so this is not a refinement.

    Reported as an evaluation-only mechanistic readout.  It is NOT the primary endpoint:
    the calibration ladder showed it moving OPPOSITE to the outcome, with scale 1.60
    giving the best approach of any arm (median 0.243 m, 20/24 seeds closer than base)
    while picking the object up half as often as scale 0.80 and never placing it.
    """
    from lcwm.phase_potential import resolve_entity_row
    if not bit_rows or not goal_atoms:
        return {"d_min_final_atom": None, "final_atom_boundaries": 0}
    row = resolve_entity_row(object_names, goal_atoms[-1][1])
    n_atoms = len(goal_atoms)
    best, n_act = None, 0
    for eef, obj, bits, tt in zip(eef_rows, obj_rows, bit_rows, t_rows):
        b = [bool(x) for x in bits]
        if len(b) != n_atoms or not all(b[:n_atoms - 1]) or b[n_atoms - 1]:
            continue                      # the final atom is not the active one here
        if int(tt) < basin_from_step:
            continue
        n_act += 1
        d = float(torch.linalg.vector_norm(
            torch.as_tensor(eef)[0:3] - torch.as_tensor(obj)[row]))
        best = d if best is None else min(best, d)
    return {"d_min_final_atom": best, "final_atom_boundaries": n_act}


def final_atom_phi(eef_rows: list, obj_rows: list, bit_rows: list, t_rows: list,
                   object_names: list, goal_atoms: list,
                   basin_from_step: int) -> dict:
    """Mean Phi' over the boundaries where the FINAL goal atom is active.

    `phi_basin_mean` is a REGISTERED secondary endpoint and nothing in this pipeline
    computed it: the driver can only read it straight through or reconstruct it from a
    per-boundary phi series with its t, and this executor wrote neither. After a full
    evaluation it would have come back `unavailable` - a registered endpoint silently
    absent from the report rather than measured.

    The restriction matches `final_atom_reach`. Phi' is `completed + phi(active)`, so a
    mean taken over boundaries with different active atoms is dominated by the integer
    `completed` term and mostly measures WHEN the second object was placed, not how
    the third sub-task went. Over final-atom-active boundaries the integer part is
    constant and the mean is the within-sub-task progress the endpoint is named for.
    """
    from lcwm.v250_progress import episode_progress
    if not bit_rows or not goal_atoms:
        return {"phi_basin_mean": None}
    recs = episode_progress(torch.stack([torch.as_tensor(x) for x in eef_rows]),
                            torch.stack([torch.as_tensor(x) for x in obj_rows]),
                            torch.stack([torch.as_tensor(x) for x in bit_rows]),
                            list(object_names), list(goal_atoms))
    n_atoms = len(goal_atoms)
    vals = []
    for rec, bits, tt in zip(recs, bit_rows, t_rows):
        b = [bool(x) for x in bits]
        if len(b) != n_atoms or not all(b[:n_atoms - 1]) or b[n_atoms - 1]:
            continue
        if int(tt) < basin_from_step:
            continue
        vals.append(float(rec.phi))
    return {"phi_basin_mean": (sum(vals) / len(vals)) if vals else None}


def prefix_stage_reached(events: dict, n_milestones: int) -> int:
    """Longest achieved PREFIX of the ordered milestone list.

    ``events`` is GoalAutomaton.events_achieved (index -> first step).  A later
    milestone without its predecessors is not progress along the chain, so
    {0, 1, 3} is stage 2, not 3.
    """
    keys = {int(k) for k in events}
    stage = 0
    for i in range(int(n_milestones)):
        if i not in keys:
            break
        stage += 1
    return stage


def progress_timing(events: dict, steps: int) -> tuple[int, int]:
    """(last-progress step, idle tail) for one episode.

    On the D0 tape the frozen policy takes its last milestone at a median step of
    260 and then spends a median 490 of 750 steps making none, so the tail is the
    quantity the round's failure mode actually lives in.
    """
    last = max((int(v) for v in events.values()), default=0)
    return last, max(int(steps) - last, 0)


def label_scope_key(labels: dict[str, Any]) -> tuple:
    """The identity of the entity axes one episode's label rows are indexed by.

    ``object_names`` orders the rows of ``obj_pos`` and ``goal_atoms`` orders the
    columns of ``bits``; lcwm/v250_progress.py resolves both by POSITION
    (``resolve_entity_row(object_names, atom[1])``).  A panel that stacks two
    episodes with different object sets or different atom orders therefore
    computes Phi' against the wrong rows and returns a number rather than an
    error.  collect_v250_chain3_round.py raises on the same drift ("object set
    changed at ep"); the evaluation panel compares this key per episode.
    """
    return (tuple(labels["object_names"]),
            tuple(tuple(a) for a in labels["goal_atoms"]))


def tensor_digest(t: torch.Tensor) -> str:
    """Short content hash of a tensor, for recording WHICH normaliser an arm used.

    The round's contract is that M1 and M0 differ only in the learned transition.
    Each arm's mu/sd come from its own checkpoint, so a per-arm normaliser fitted
    on that arm's own data would be a second difference; this executor cannot see
    the other arms, but recording the digest makes an unmatched normaliser
    detectable from the summaries afterwards rather than never.
    """
    x = t.detach().to(torch.float64).cpu().contiguous()
    return hashlib.sha1(x.numpy().tobytes()).hexdigest()[:12]


def build_eval_labels(records: dict[str, Any]) -> dict[str, Any]:
    """The evaluation sidecar payload, checked to be free of training fields.

    Keys are the ones lcwm/v250_progress.py:tape_progress reads, so Phi' can be
    computed offline for the evaluation panel with the same code the training
    annotation uses.
    """
    bad = sorted(set(records) & set(TRAINING_TAPE_FIELDS))
    if bad:
        raise SystemExit(f"evaluation sidecar carries training fields {bad}")
    return records


@dataclass(frozen=True)
class SimDeps:
    """Everything that touches LIBERO, injected so the loop is testable on CPU."""

    prefix_forward: Callable
    proprio: Callable
    goal_atoms: Callable
    predicate_bits: Callable
    goal_automaton: Callable
    body_positions: Callable
    discover_object_bodies: Callable


@dataclass
class DeployConfig:
    """One arm's deployment contract.  Identical for every arm except `zero_residual`."""

    task: str
    horizon: int
    c: int
    adim: int
    zdim: int
    mu: torch.Tensor
    sd: torch.Tensor
    bmu: torch.Tensor
    bsd: torch.Tensor
    condition: str = "belief"
    rich: bool | None = None
    zero_residual: bool = False
    collect_tape: bool = False
    eval_labels: bool = False
    fixed_horizon: bool = False
    residual_start_chunk: int = 0
    subgoals: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# the loop every arm runs
# --------------------------------------------------------------------------- #
def run_episode(runner, env, seed: int, cfg: DeployConfig, model, actor,
                deps: SimDeps, ep_idx: int = 0) -> dict:
    """One rollout.  The residual and the frozen-policy arms differ in one line."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runner.reset()
    obs, _ = env.reset(seed=int(seed))
    atoms = deps.goal_atoms(env)

    au = None
    body_list: list[str] = []
    object_names: list[str] = []
    eef_rows: list[torch.Tensor] = []
    obj_rows: list[torch.Tensor] = []
    bit_rows: list[torch.Tensor] = []
    t_rows: list[int] = []
    if cfg.eval_labels:
        bodies = deps.discover_object_bodies(env)
        object_names = sorted(bodies)
        body_list = [bodies[n] for n in object_names]
        au = deps.goal_automaton(list(cfg.subgoals))
        au.start(env)
        au.evaluate(env, 0)

    def record_label(step_t: int) -> None:
        eef_rows.append(deps.proprio(obs).clone())
        obj_rows.append(torch.as_tensor(deps.body_positions(env, body_list),
                                        dtype=torch.float32))
        bit_rows.append(torch.as_tensor(deps.predicate_bits(env, atoms).copy()))
        t_rows.append(int(step_t))

    t, done, succ, chunks = 0, False, None, 0
    dmag: list[float] = []
    dmax = 0.0
    triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    b = torch.zeros(1, model.bdim)            # belief resets each episode
    u_prev = torch.zeros(1, cfg.c, cfg.adim)  # no action precedes the first chunk
    prev = None
    while not done and t < cfg.horizon:
        po = runner._obs_to_policy_batch(obs, env.task_description)
        with torch.no_grad():
            pf = deps.prefix_forward(runner.policy, po)
            hid = pf.hidden[0].detach().float().cpu()
            msk = pf.pad_masks[0].detach().cpu().bool()
            chunk = runner.sample_chunk(obs, env.task_description
                                        )[0, :cfg.c].detach().float().cpu()
        del pf
        rich = resolve_latent_mode(cfg.zdim, int(hid.shape[-1]), cfg.rich)
        o = torch.cat([encode_latent(hid, msk, rich), deps.proprio(obs)])
        if int(o.shape[-1]) != int(cfg.zdim):
            raise SystemExit(
                f"latent width {int(o.shape[-1])} != actor dims {int(cfg.zdim)} "
                f"(rich={rich}); the actor would read a space it never saw")
        if cfg.eval_labels:
            record_label(t)
        with torch.no_grad():
            zn = ((o - cfg.mu) / cfg.sd).unsqueeze(0)
            e = model.enc(zn)
            b = model.step(b, e, u_prev)          # CAUSAL: uses u_{t-1}
            # the advantage came from T_th either way; `condition` only says
            # what the ACTOR reads at deployment
            d = actor(zn if cfg.condition == "raw"
                      else torch.cat([e, (b - cfg.bmu) / cfg.bsd], -1))
            if cfg.zero_residual or chunks < cfg.residual_start_chunk:
                # SCHEDULE, matched across arms. The interface is trained only on
                # boundaries at or after the collector's --residual-start-chunk, so
                # applying it earlier runs it at states it never saw - and the
                # 2026-09-12 basin calibration measured that an ungated sustained
                # residual is what damages the first two sub-tasks, which the frozen
                # policy otherwise completes in 24/24 episodes. The gate is a chunk
                # index, readable from the deployed step counter, not a state read.
                d = torch.zeros_like(d)
        executed = chunk.unsqueeze(0) + d
        if cfg.zero_residual and not torch.equal(executed[0], chunk):
            raise SystemExit("zero-residual arm perturbed the base chunk")
        if chunks < cfg.residual_start_chunk and not torch.equal(executed[0], chunk):
            raise SystemExit(
                f"chunk {chunks} is before --residual-start-chunk "
                f"{cfg.residual_start_chunk} but the base chunk was perturbed")
        dmag.append(float(d.abs().mean()))
        dmax = max(dmax, float(d.abs().max()))
        if cfg.collect_tape:
            if prev is not None:
                triples.append((prev[0], prev[1], o, prev[2]))
            prev = (o, executed[0].detach(), t)
        u_prev = executed.detach()
        chunks += 1
        for act in runner.chunk_to_env(executed[0]):
            if done:
                break
            obs, _r, tm, tr, inf = env.step(act)
            t += 1
            # --fixed-horizon: success is RECORDED, never acted on, so the boundary
            # count carries no outcome information.
            #
            # `info["done"]` is itself SUCCESS-DRIVEN on this task, so keying off it
            # does NOT give a fixed horizon. Measured 2026-09-12: on seed 9018 `done`
            # first fired at t=678, the same step as `is_success`, and a loop that
            # broke on it produced a 678-step episode - the outcome leak this flag
            # exists to remove. Stepping past it is safe: the same seed then ran the
            # full 750 steps with predicates still readable and the milestone record
            # intact. So under --fixed-horizon only a done that is NOT a success, or
            # a truncation, ends the episode.
            done = ((bool(inf.get("done", False))
                     and not bool(inf.get("is_success", False))) or bool(tr)) \
                if cfg.fixed_horizon else bool(tm or tr)
            if succ is None and bool(inf.get("is_success", False)):
                succ = t
        runner.reset()
        if au is not None:
            au.evaluate(env, t)
        if succ is None and deps.predicate_bits(env, atoms).all():
            succ = t

    events: dict[int, int] = {}
    stage, last_progress, idle_tail = 0, 0, 0
    if au is not None:
        if chunks:
            record_label(t)
        events = {int(k): int(v) for k, v in au.events_achieved.items()}
        stage = prefix_stage_reached(events, len(cfg.subgoals))
        last_progress, idle_tail = progress_timing(events, t)
        reach = final_atom_reach(eef_rows, obj_rows, bit_rows, t_rows,
                                 list(object_names), list(atoms),
                                 cfg.residual_start_chunk * cfg.c)
        reach.update(final_atom_phi(eef_rows, obj_rows, bit_rows, t_rows,
                                    list(object_names), list(atoms),
                                    cfg.residual_start_chunk * cfg.c))

    return {"idx": int(ep_idx), "seed": int(seed), "success": succ is not None,
            "success_step": succ, "steps": t, "chunks": chunks,
            "rich_latent": bool(rich) if chunks else None,
            "mean_abs_delta": float(np.mean(dmag)) if dmag else 0.0,
            "max_abs_delta": dmax, "events": events, "stage_reached": stage,
            "last_progress_step": last_progress, "idle_tail": idle_tail,
            **(reach if au is not None else
               {"d_min_final_atom": None, "final_atom_boundaries": 0,
                "phi_basin_mean": None}),
            "triples": triples,
            "labels": {"eef_proprio": eef_rows, "obj_pos": obj_rows,
                       "bits": bit_rows, "t": t_rows,
                       "object_names": object_names, "goal_atoms": atoms}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", type=Path, required=True)
    ap.add_argument("--task", default="chain1b_lr2")
    ap.add_argument("--panel", type=int, default=96)
    ap.add_argument("--panel-start", type=int, default=7600)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--collect-tape", action="store_true",
                    help="also write a v121-format tape of THIS run. The recorded "
                         "action is the one actually EXECUTED (chunk + Delta), so "
                         "retraining on it covers the actor's own distribution - "
                         "which a model frozen on pi0.5's tapes does not. It carries "
                         "a per-episode success field, so it is not valid input for "
                         "a round whose tape contract forbids outcome fields.")
    ap.add_argument("--zero-residual", action="store_true",
                    help="CODE-PATH CONTROL: Delta := 0, everything else identical. "
                         "The frozen-pi0.5 number on record (45/96 on chain1b) came "
                         "from the select_action path, not this chunk-sampling path, "
                         "so it is not a valid base for a residual arm.")
    ap.add_argument("--rich-latent", action="store_true",
                    help="force the 8217-d rich encoding (4 camera mean/max blocks "
                         "+ 25-d proprio). Default: inferred from the actor's dims.")
    ap.add_argument("--pooled-latent", action="store_true",
                    help="force the historical 2073-d pooled encoding.")
    ap.add_argument("--eval-labels", action="store_true",
                    help="write the evaluation sidecar (eval_labels.pt, "
                         "outcome.json): milestones, contiguous-prefix stage, "
                         "last-progress step, idle tail, and the per-boundary rows "
                         "lcwm/v250_progress.py needs for Phi'. Readouts only.")
    ap.add_argument("--residual-start-chunk", type=int, default=0,
                    help="first chunk index at which the residual is applied; before "
                         "it the executed chunk is the frozen policy's own, asserted "
                         "bit-exactly. Must match the --residual-start-chunk the "
                         "interface was TRAINED under: the interface sees only "
                         "boundaries at or after it, and the 2026-09-12 basin "
                         "calibration measured that an ungated sustained residual is "
                         "what damages the first two sub-tasks. Applied identically "
                         "to every arm, so it cannot separate them.")
    ap.add_argument("--fixed-horizon", action="store_true",
                    help="run every episode to the full horizon, keying off "
                         "info['done'] and truncation instead of ChainEnv's "
                         "terminated=done-or-success, so the boundary count does "
                         "not depend on the outcome. Set it on ALL arms or none.")
    a = ap.parse_args()
    if a.rich_latent and a.pooled_latent:
        raise SystemExit("--rich-latent and --pooled-latent are exclusive")
    forced = True if a.rich_latent else (False if a.pooled_latent else None)

    from lcwm.libero_paths import ensure_project_libero_config
    ensure_project_libero_config()
    from lcwm.probe_data import body_positions, discover_object_bodies
    from lcwm.sampler import prefix_forward
    from lcwm.seq_data import goal_atoms, predicate_bits
    from lcwm.task_automaton import GoalAutomaton
    from lcwm.v080_bench import V080_TASKS, episode_length, make_v080_env
    from collect_v121_deploy_latents import proprio
    deps = SimDeps(prefix_forward=prefix_forward, proprio=proprio,
                   goal_atoms=goal_atoms, predicate_bits=predicate_bits,
                   goal_automaton=GoalAutomaton, body_positions=body_positions,
                   discover_object_bodies=discover_object_bodies)

    L = episode_length(a.task)
    PANEL = tuple(range(a.panel_start, a.panel_start + a.panel))
    for lo, hi in SPENT_RANGES:
        assert not (set(PANEL) & set(range(lo, hi))), \
            f"panel {PANEL[0]}-{PANEL[-1]} overlaps spent seeds {lo}-{hi - 1}"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")

    ck = torch.load(a.actor, weights_only=False)
    tag = a.tag or ck["arm"]
    out = OUT / f"{a.task}_{tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    zdim, c_, adim = ck["dims"]
    if int(c_) != C:
        raise SystemExit(f"actor chunk length {c_} != executor's {C}")
    if int(ck["mu"].numel()) != int(zdim):
        raise SystemExit(f"mu has {int(ck['mu'].numel())} entries, dims say {zdim}")
    condition = ck.get("condition", "belief")
    if "rawwm" in ck:
        # v220: T_th predicts in the VLA's own latent, and the actor reads that
        # same space, so no belief is needed on the deployment path at all
        m = RawLatentWM(zdim, c_, adim)
        m.load_state_dict(ck["rawwm"])
        wm_source = "rawwm"
    elif "model" in ck:
        m = ActionBelief(zdim, c_, adim, use_action=ck["use_action"])
        m.load_state_dict(ck["model"])
        wm_source = "model"
    elif condition == "raw":
        # a no-rollout / model-free arm need not ship a transition. Its actor reads
        # the raw latent, so `e` and `b` cannot reach the executed action; the calls
        # stay in place only to keep this loop bit-identical across arms.
        m = RawLatentWM(zdim, c_, adim)
        wm_source = "absent_raw_conditioned"
    else:
        raise SystemExit("checkpoint has neither 'rawwm' nor 'model' and is not "
                         "raw-conditioned; a belief-conditioned actor needs its "
                         "transition to reproduce the space it was trained in")
    m.eval()
    actor = ResidualActor(ck["zdim"], c_, adim, scale=ck["scale"])
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    # a trainer run with --device cuda saves its normaliser as CUDA tensors; the
    # deployment latent is built on CPU, so pull them across here rather than
    # dying on "expected all tensors on the same device" at the first chunk
    mu, sd = ck["mu"].detach().cpu(), ck["sd"].detach().cpu()
    bmu, bsd = ck["bmu"].detach().cpu(), ck["bsd"].detach().cpu()
    cfg = DeployConfig(task=a.task, horizon=L, c=int(c_), adim=int(adim),
                       zdim=int(zdim), mu=mu, sd=sd,
                       bmu=bmu, bsd=bsd, condition=condition,
                       rich=forced, zero_residual=a.zero_residual,
                       collect_tape=a.collect_tape, eval_labels=a.eval_labels,
                       fixed_horizon=a.fixed_horizon,
                       residual_start_chunk=a.residual_start_chunk,
                       subgoals=tuple(V080_TASKS[a.task]["ordered_subgoals"]))
    print(f"arm={ck['arm']} action-conditioned={ck.get('use_action', True)} "
          f"scale={ck['scale']} latent={zdim}d wm={wm_source} "
          f"panel {PANEL[0]}-{PANEL[-1]}")

    from lcwm.chassis import DEFAULT_MODEL, Pi05Runner
    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(a.task)

    rows, steps, resolved_rich = [], 0, forced
    Z_, U_, ZN_, EP_, TT_ = [], [], [], [], []
    EEF, OBJ, BITS, LEP, LT = [], [], [], [], []
    object_names, atoms, scope = None, None, None
    for ei, seed in enumerate(PANEL):
        r = run_episode(runner, env, int(seed), cfg, m, actor, deps, ep_idx=ei)
        steps += r["steps"]
        if r["rich_latent"] is not None:
            resolved_rich = r["rich_latent"]
        for z_t, u_t, z_n, tt in r["triples"]:
            Z_.append(z_t)
            U_.append(u_t)
            ZN_.append(z_n)
            EP_.append(ei)
            TT_.append(tt)
        lab = r["labels"]
        if a.eval_labels:
            key = label_scope_key(lab)
            if scope is None:
                scope, object_names = key, lab["object_names"]
                atoms = lab["goal_atoms"]
            elif key != scope:
                raise SystemExit(
                    f"episode {ei} (seed {seed}) changed the label scope: "
                    f"{scope} -> {key}; obj_pos rows and bits columns are "
                    f"positional, so stacking these would compute Phi' against "
                    f"the wrong entities")
            EEF += lab["eef_proprio"]
            OBJ += lab["obj_pos"]
            BITS += lab["bits"]
            LEP += [ei] * len(lab["t"])
            LT += lab["t"]
        rows.append({k: r[k] for k in
                     ("seed", "success", "success_step", "steps", "mean_abs_delta",
                      "max_abs_delta", "chunks", "events", "stage_reached",
                      "last_progress_step", "idle_tail",
                      "d_min_final_atom", "final_atom_boundaries",
                      "phi_basin_mean")})
        extra = (f"stage={r['stage_reached']} idle={r['idle_tail']} "
                 if a.eval_labels else "")
        print(f"{tag} s{seed}: succ={r['success']!s:5s} t={r['steps']} "
              f"{extra}({steps} steps)", flush=True)

    if a.collect_tape and Z_:
        Zt, Ut, Znt = torch.stack(Z_), torch.stack(U_), torch.stack(ZN_)
        torch.save({"z": Zt, "u": Ut, "z_next": Znt,
                    "episode": torch.tensor(EP_), "t": torch.tensor(TT_),
                    "success": torch.tensor([float(r["success"]) for r in rows]),
                    "task": a.task, "c": C, "latent_dim": Zt.shape[-1],
                    "proprio_dim": PROPRIO_DIM}, out / "tape.pt")
        print(f"tape: {len(Z_)} triples of the ACTOR's own distribution")

    k = sum(r["success"] for r in rows)
    if a.eval_labels:
        torch.save(build_eval_labels(
            {"eef_proprio": torch.stack(EEF), "obj_pos": torch.stack(OBJ),
             "bits": torch.stack(BITS), "episode": torch.tensor(LEP),
             "t": torch.tensor(LT), "object_names": object_names,
             "goal_atoms": atoms, "milestones": list(cfg.subgoals),
             "events": {i: r_["events"] for i, r_ in enumerate(rows)},
             "task": a.task, "c": C, "role": "evaluation",
             "provenance": "EVALUATION-ONLY readout of a held-out deployment "
                           "panel; scripted privileged stand-in for human stage "
                           "annotation, never a training input and never visible "
                           "to the actor",
             "budget_boundaries": len(LEP), "budget_episodes": len(rows)}),
            out / "eval_labels.pt")
        (out / "outcome.json").write_text(json.dumps(
            {"utc": stamp, "task": a.task, "tag": tag, "actor": str(a.actor),
             "zero_residual": a.zero_residual, "fixed_horizon": a.fixed_horizon,
             "panel": [PANEL[0], PANEL[-1], len(PANEL)],
             "successes": k, "n": len(rows), "episodes": rows}, indent=2))

    summary = {"task": a.task, "tag": tag, "utc": stamp, "actor": str(a.actor),
               "use_action": ck.get("use_action", True), "scale": ck["scale"],
               "zero_residual": a.zero_residual,
               "latent_dim": int(zdim), "rich_latent": resolved_rich,
               "latent_mode_forced": forced is not None,
               "wm_source": wm_source, "condition": condition,
               # arm-matching evidence: M1 and M0 must share these three
               "norm_digest": {"mu": tensor_digest(mu), "sd": tensor_digest(sd),
                               "bmu": tensor_digest(bmu),
                               "bsd": tensor_digest(bsd)},
               "actor_params": int(sum(q.numel() for q in actor.parameters())),
               "fixed_horizon": a.fixed_horizon, "eval_labels": a.eval_labels,
               "residual_start_chunk": int(a.residual_start_chunk),
               # The REALISED intervention magnitude over the whole panel. `scale` is
               # only the BOUND: two arms can share it, share every flag, and still
               # execute different-sized interventions. The 2026-09-12 calibration
               # ladder attributes every endpoint to magnitude (|delta| 0.051/0.105/
               # 0.206 -> zero milestone-4 events, 0.440 -> 4/24), so an unreported
               # difference here would be read as an effect of the transition.
               "mean_abs_delta_panel": float(np.mean(
                   [r["mean_abs_delta"] for r in rows])) if rows else 0.0,
               "max_abs_delta_panel": float(max(
                   [r["max_abs_delta"] for r in rows], default=0.0)),
               "panel": [PANEL[0], PANEL[-1], len(PANEL)],
               "successes": k, "n": len(rows), "rate": k / len(rows),
               "cp95": [clopper_pearson_lower(k, len(rows)),
                        clopper_pearson_upper(k, len(rows))],
               "mean_stage_reached":
                   float(np.mean([r["stage_reached"] for r in rows]))
                   if a.eval_labels else None,
               "mean_idle_tail":
                   float(np.mean([r["idle_tail"] for r in rows]))
                   if a.eval_labels else None,
               "env_steps": steps, "episodes": rows,
               "episode_records": [{"idx": i, "seed": r["seed"],
                                    "success": r["success"],
                                    "success_step": r["success_step"],
                                    "steps": r["steps"]}
                                   for i, r in enumerate(rows)],
               "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     capture_output=True, text=True).stdout.strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{tag}: {k}/{len(rows)} = {k/len(rows):.3f} "
          f"CP95 [{summary['cp95'][0]:.3f},{summary['cp95'][1]:.3f}] "
          f"{steps} steps -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
