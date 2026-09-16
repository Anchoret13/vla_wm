#!/usr/bin/env python
"""Train the round's policy interface FROM a world model - the missing WM -> policy path.

    # M1: the updated transition
    python scripts/train_v251_interface.py --model-pair <.../model_pair.pt> \
        --wm-arm updated --advantage predicted --horizon 10 \
        --tapes results/v250_chain3_round/<d0> results/v250_chain3_round/<d1> \
        --device cuda --out-tag m1

    # M0: the stale transition, everything else byte-identical
    python scripts/train_v251_interface.py --model-pair <.../model_pair.pt> \
        --wm-arm stale --advantage predicted --horizon 10 --tapes <same> --out-tag m0

    # Q: same data, same rows, same target, no latent ever rolled forward
    python scripts/train_v251_interface.py --model-pair <.../model_pair.pt> \
        --wm-arm stale --advantage observed --horizon 10 --tapes <same> --out-tag q

WHAT THIS MEASURES.  Given ONE frozen transition arm and the round's deployment tapes,
it fits a bounded residual interface Delta_ph(z) on pi0.5's action chunk by
advantage-weighted regression onto the residual the D1 collector actually executed.
It measures nothing about deployment: no success rate, no outcome.json and no
`success` tape field is read anywhere in this file, and `outcome_json_read: False` is
recorded in the summary.  Its only output is a checkpoint that
`scripts/run_v206_belief_residual_deploy.py` can deploy (that script dispatches on the
presence of a `rawwm` key and reads mu/sd/bmu/bsd/dims from the checkpoint itself; cited
by symbol rather than by line because run_v206 is under concurrent edit today).

WHY THIS FILE EXISTS.  Before it, `model_pair.pt` was consumed by exactly two scripts:
`build_v241_model_pair.py`, which writes it, and `score_v241_policy_pool.py`, which only
RANKS an already-trained pool and contains no optimiser and no loss.  Nothing trained an
actor from a model pair, so the placement axis of CLAUDE.md - a world model that acts on
the policy rather than on a ranking - had no implementation to test.

WHY THE ADVANTAGE IS SUSTAINED RATHER THAN SINGLE-CHUNK.  v229 measured that from an
identical chain3 state, eight genuinely different action chunks produced identical
progress in 71 of 72 cases: pi0.5 replans after the chunk and corrects a single-chunk
perturbation away.  train_v220's advantage (train_v220_raw_latent_wm.py:236-247) is a
one-chunk comparison and is therefore aimed at a causally inert object.  Here the
candidate residual is applied at EVERY one of H imagined chunks, matching the unit of
intervention the D1 collector actually used (a fixed state-conditioned residual applied
at every chunk from --residual-start-chunk onward).

THE THREE ADVANTAGE MODES, all on the SAME row set so the arms stay matched:

  predicted  A = phi_hat(z~_{t+Hc} | u_base + delta) - phi_hat(z~_{t+Hc} | u_base),
             both terms produced by rolling the SELECTED arm H chunks from the same
             (z_t, b_t).  This is the arm under test; the transition is the only thing
             that differs between the M1 and M0 runs.  Head bias at z_t cancels because
             the two rolls share their start.
  observed   A = Phi'(real boundary t+Hc) - Phi'(real boundary t).  The realised
             Monte-Carlo return over the same rows.  `observed_advantage` takes no model
             argument at all - the no-rollout control cannot touch a transition even by
             accident.
  current    A = Phi'(real boundary t).  State value, no future.  It is the weakest
             control and it is the one that matters: on 2026-08-31 a state-value filter
             explained away a gain that had been attributed to a rollout.

Under the DEFAULT --phi-readout label the two controls read Phi' straight off the
supplied annotation while `predicted` necessarily reads it through the learned head, so
`predicted` vs `observed` then differs by two things - the roll, and the head's
approximation error - not one.  --phi-readout head puts the controls through the same
head and is the variant matched on that second difference; which one was run is recorded
in summary.json.  --phi-readout has no effect under --advantage predicted, whose whole
point is a head read off a latent that has no annotation.

The control arms still instantiate the selected transition, because the checkpoint's
bmu/bsd slots are filled from its beliefs, but no belief and no predicted latent enters
their advantage - `observed_advantage` and `current_advantage` have no parameter that
could carry one, and run_v206 ignores bmu/bsd under condition='raw'.

THE BASE-ACTION APPROXIMATION, stated because it is the weakest link in the causal path.
Rolling under "u_base + delta" needs u_base at IMAGINED latents, where frozen pi0.5
cannot be called.  Default `--base-action recorded` substitutes the episode's OWN
recorded base-chunk sequence: for a row at chunk k of episode e, imagined step h uses
u_base[e, k+h].  THIS ASSUMES THE FROZEN POLICY'S BASE CHUNK SEQUENCE IS UNCHANGED BY
THE RESIDUAL, WHICH IS FALSE IN GENERAL - pi0.5 replans every chunk, which is the same
replanning v229 measured.  The approximation is not defended as accurate; it is shared
bit-identically by every arm (same rows, same recorded sequence), so it cannot explain a
difference between arms, which is the only property the comparison requires of it.
`--base-action prior` instead calls model_pair's learned action prior at each imagined
latent; it is implemented but not the default because that prior generalises from train
MSE .0015 to holdout .0442 and would inject its own error into the causal path.

THE REGRESSION TARGET.  D1 rows carry a real intervention: trace.pt records u_base and
delta per boundary with u_exec = u_base + delta (verified elementwise at load, tolerance
1e-5).  The target is therefore the COLLECTOR's delta, weighted by that delta's own
advantage:

    loss = mean_i  w_i * || Delta_ph(z_i) - delta_collector_i ||^2 ,
    w    = softmax(A / temperature) * len(batch),  temperature = std(A) over all rows

This deliberately does NOT copy train_v220_raw_latent_wm.py:260
(`tgt_ = Uf[i] - Uf[i].mean(0, keepdim=True)`, the deviation of an executed chunk from
the batch mean), which is a surrogate for an intervention that was never recorded.
Here the intervention is recorded.  The temperature is the global advantage std rather
than train_v220's per-batch std so that the AWR weighting is a fixed function of the
advantage; it is recorded in summary.json.

ZERO-DELTA ROWS.  Base-role rows, and D1 rows before the collector's
--residual-start-chunk, have delta identically zero: their target is zero and they carry
no intervention.  They are dropped by default and kept only under --include-zero-delta,
which is recorded.  A run whose tapes are all base-role therefore has no training rows
under the default and fails loudly rather than silently fitting the zero function.
Episodes with fewer than H chunks offer no window at all; they are counted into
row_stats rather than dropped in silence.

THE COORDINATE CONTRACT, the thing v235 got wrong and v241 fixed.  This script trains the
actor IN THE MODEL PAIR'S NORMALISED SPACE and stores mu/sd = the model pair's mu/sd in
the actor checkpoint, so z_actor == z_model and the imagined-step path
z_model -> x_raw -> z_actor collapses to the identity.  run_v206 computes
zn = (o - mu)/sd from the checkpoint's own mu/sd and calls actor(zn), so it deploys
exactly the function that was trained.  This is asserted at save time by reloading the
checkpoint and re-running the actor on a fixed probe.

THE Phi' HEAD.  Phi' is a per-row scalar from lcwm/v250_progress.tape_progress (Phi'
= #complete goal atoms + within-atom potential, in [0, 3], with a FIXED 0.60 m approach
reference so it does not saturate in chain3's stall).  Reading it off a PREDICTED latent
needs a learned head phi_hat(z) -> Phi', fitted here on the round's real (z, Phi') pairs
with an EPISODE-DISJOINT split; the held-out correlation is reported and stored, and the
head is frozen before any advantage is computed.  The head is fitted on CPU from a fixed
seed with no dependence on --wm-arm whatsoever, so M1 and M0 receive a bit-identical
head; its digest is written to summary.json and to every actor checkpoint, and
--assert-phi-digest makes a mismatch a hard failure.  --phi-head reuses a head fitted by
an earlier run, which makes bit-identity a property of the file rather than of anyone's
determinism assumptions.  A reused head also carries the sha256 of the mu/sd it was
fitted under and is REFUSED when that disagrees with this run's model pair: the head
reads normalised latents, so --assert-phi-digest pinning WHICH head is used says nothing
about which space it belongs to, and a head reused across model pairs would otherwise be
evaluated in coordinates it never saw and still return a number.  The digests of the
tapes it was fitted on are carried too, but only recorded, since refitting on D0 and
reusing on D0+D1 is a legitimate thing to do.

WHY NOT model_pair['phi'].  What that head regresses depends on how the pair was
built, so this file cannot assume it.  Built WITHOUT --phi-labels it regresses
gamma**chunks_to_success (build_v241_model_pair.phi_target), which on chain3 at 2/96
terminal success is zero on ~98% of rows and carries no gradient inside the stall this
round is about; built WITH --phi-labels it regresses the same Phi' used here
(build_v241_model_pair.load_phi_labels / episode_phi).  In neither case does it arrive
with a held-out correlation measured on THIS row set, or with a digest this script can
pin across arms, which are the two properties the comparison needs.  A fresh head is
fitted here instead and model_pair['phi'] is never loaded.

WHAT WOULD OVERTURN A POSITIVE READING FROM THIS INTERFACE, registered before any actor
is deployed: `--advantage observed` (same data, same rows, same target, no roll) matching
`--advantage predicted`; `--advantage current` matching either, which would say a state
filter explains it; `--wm-arm stale` matching `--wm-arm updated`, which would say the
deployment update contributed nothing; and `--wm-arm updated_shuffled` matching
`--wm-arm updated`, which would say the action conditioning is decorative.  None of those
is decided here - this file only produces the arms.

PRE-REGISTERED IN THE FILENAME AND FLAGS, before any deployment: horizon, advantage mode,
base-action mode, zero-delta policy, interface scale, restart count, optimiser budget and
seed are command-line inputs recorded in summary.json together with the sha256 of the
model pair, of every tape/labels/trace artifact and of this source file.

REPO-WIDE BUGS DELIBERATELY NOT REPRODUCED.  (1) train_v205_action_belief.load_episodes
silently DROPS tapes whose latent_dim != 2073 and the caller then dies on
torch.cat([]); this loader takes the latent dim from the model pair and raises before any
training if a tape disagrees.  (2) `L = min(min_chunks, 24)` truncates episodes to 24
chunks = 240 env steps, which on chain3@750 discards 68% of the trajectory including the
whole stall basin; nothing here truncates, and the per-episode chunk counts are recorded.
(3) `s = min(int(succ_chunk), L-1)` rewrites late successes as chunk-23 successes; no
success-derived target is built at all.  (4) the trainers in this repo are CPU-only;
--device cuda is supported here because RawLatentWM.trans is ~18M parameters at the
round's 8217-d latent.
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
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from lcwm.v250_progress import tape_progress  # noqa: E402
from build_v241_model_pair import make_prior  # noqa: E402
from train_v157_residual_actor import ResidualActor  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402

OUT = REPO / "results" / "v251_interface"
MODEL_ARMS = ("stale", "base_continue", "updated", "updated_shuffled")
ADVANTAGES = ("predicted", "observed", "current")

#: Episodes per belief roll.  The roll materialises [G, L, zdim] activations and the
#: round's latent is 8217-d, so this bounds peak memory rather than throughput.
EPISODE_BATCH = 8
#: Rows per imagined rollout batch, for the same reason.
ADV_BATCH = 128
#: Fixed probe used by the save/reload self-check; no data, no RNG state.
PROBE_SEED = 251251
#: u_exec = u_base + delta is an audit identity written by the collector, not a fit.
TRACE_TOLERANCE = 1e-5
#: z_next[k] and z[k+1] are the same observation recorded twice by the collector.
CONTINUITY_TOLERANCE = 1e-4


# --------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def state_dict_digest(state: dict[str, Tensor]) -> str:
    """Content hash of a state dict, key order independent, byte exact."""
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def stable_fraction(seed: int, key: str) -> float:
    payload = f"v251:{seed}:{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip()


# ----------------------------------------------------------------------- statistics


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def distribution(values: Tensor) -> dict[str, Any]:
    v = values.detach().float().cpu()
    if v.numel() == 0:
        return {"n": 0}
    q = torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "min": float(v.min()),
        "max": float(v.max()),
        "q05": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
        "q75": float(q[3]), "q95": float(q[4]),
        "fraction_positive": float((v > 0).float().mean()),
        "fraction_zero": float((v == 0).float().mean()),
    }


# ------------------------------------------------------------------------- loading


@dataclass
class EpisodeRows:
    """One deployment episode, never truncated.

    ``z_bound`` holds n+1 boundary latents: the n tape states plus the final
    ``z_next``, so a window that ends on the last boundary needs no special case.
    ``u``/``u_base``/``delta`` hold the n chunk-indexed actions.
    """

    key: str
    run: str
    ep_id: int
    t0: int
    c: int
    z_bound: Tensor
    u: Tensor
    u_base: Tensor
    delta: Tensor
    phi_bound: Tensor
    ordinal_bound: Tensor

    @property
    def n(self) -> int:
        return int(self.u.shape[0])


@dataclass
class Bundle:
    """Flat, episode-contiguous view of every loaded run, in model-normalised space."""

    zdim: int
    c: int
    adim: int
    z_bound: Tensor          # [B, zdim], B = sum(n_e + 1)
    phi_bound: Tensor        # [B]
    u: Tensor                # [N, c, adim], N = sum(n_e)
    u_base: Tensor
    delta: Tensor
    row_episode: Tensor      # [N] episode index
    row_chunk: Tensor        # [N] chunk index within the episode
    row_boundary: Tensor     # [N] flat boundary index of the row's own boundary
    ep_row_start: Tensor     # [E]
    ep_len: Tensor           # [E]
    keys: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.u.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(self.ep_len.numel())


def _require(data: dict[str, Any], key: str, path: Path, ndim: int) -> Tensor:
    value = data.get(key)
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{path}: field {key!r} must be a rank-{ndim} tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{path}: field {key!r} contains non-finite values")
    return value.detach().cpu()


def load_run(run_dir: Path, task: str, expect_dims: tuple[int, int, int]
             ) -> tuple[list[EpisodeRows], dict[str, Any]]:
    """Load one v250 run directory (tape.pt + labels.pt + trace.pt).

    ``expect_dims`` comes from the model pair and is enforced here, before any
    training, because train_v205_action_belief.load_episodes silently drops
    mismatched tapes and leaves the caller to die on ``torch.cat([])``.
    """
    run_dir = Path(run_dir).resolve()
    if run_dir.is_file() and run_dir.name == "tape.pt":
        run_dir = run_dir.parent
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir}: not a run directory")
    paths = {name: run_dir / f"{name}.pt" for name in ("tape", "labels", "trace")}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir}: missing {name}.pt")

    tape = torch.load(paths["tape"], weights_only=False, map_location="cpu")
    labels = torch.load(paths["labels"], weights_only=False, map_location="cpu")
    trace = torch.load(paths["trace"], weights_only=False, map_location="cpu")
    # Older tapes (v121/v206) carry a per-episode `success` vector.  This file trains
    # no outcome-derived target, so the field is dropped on load and cannot be read.
    for dropped in ("success", "success_step", "outcome"):
        tape.pop(dropped, None)

    for name, obj in (("tape", tape), ("labels", labels)):
        if obj.get("task") != task:
            raise ValueError(f"{paths[name]}: task {obj.get('task')!r} != {task!r}")

    z = _require(tape, "z", paths["tape"], 2).float()
    u = _require(tape, "u", paths["tape"], 3).float()
    z_next = _require(tape, "z_next", paths["tape"], 2).float()
    episode = _require(tape, "episode", paths["tape"], 1).long()
    times = _require(tape, "t", paths["tape"], 1).long()
    if "latent_dim" not in tape or "c" not in tape:
        raise ValueError(f"{paths['tape']}: missing latent_dim/c metadata")
    zdim, c, adim = int(z.shape[1]), int(tape["c"]), int(u.shape[2])
    if int(tape["latent_dim"]) != zdim:
        raise ValueError(
            f"{paths['tape']}: latent_dim={int(tape['latent_dim'])} but z is {zdim}-d"
        )
    if int(u.shape[1]) != c:
        raise ValueError(f"{paths['tape']}: c={c} but u has {int(u.shape[1])} steps")
    if (zdim, c, adim) != expect_dims:
        raise ValueError(
            f"{paths['tape']}: tape dims {(zdim, c, adim)} != model-pair dims "
            f"{expect_dims}. Refusing to drop the tape silently the way "
            "train_v205_action_belief.load_episodes:104 does."
        )
    n = int(z.shape[0])
    if not (len(u) == len(z_next) == len(episode) == len(times) == n) or n == 0:
        raise ValueError(f"{paths['tape']}: inconsistent or empty transition fields")

    u_base = _require(trace, "u_base", paths["trace"], 3).float()
    delta = _require(trace, "delta", paths["trace"], 3).float()
    trace_ep = _require(trace, "episode", paths["trace"], 1).long()
    trace_t = _require(trace, "t", paths["trace"], 1).long()
    if u_base.shape != u.shape or delta.shape != u.shape:
        raise ValueError(f"{paths['trace']}: action shapes do not match the tape")
    if not (torch.equal(trace_ep, episode) and torch.equal(trace_t, times)):
        raise ValueError(f"{paths['trace']}: rows are not aligned with the tape")
    audit = float((u_base + delta - u).abs().max())
    if audit > TRACE_TOLERANCE:
        raise ValueError(
            f"{paths['trace']}: u_base + delta != u_exec (max {audit:.3g})"
        )

    progress = tape_progress(labels)
    phi_map: dict[tuple[int, int], float] = {}
    ord_map: dict[tuple[int, int], int] = {}
    lab_ep, lab_t = progress["episode"].long(), progress["t"].long()
    for i in range(int(lab_ep.numel())):
        key = (int(lab_ep[i]), int(lab_t[i]))
        if key in phi_map:
            raise ValueError(f"{paths['labels']}: duplicate label row {key}")
        phi_map[key] = float(progress["phi"][i])
        ord_map[key] = int(progress["ordinal"][i])

    tape_sha = sha256_file(paths["tape"])
    out: list[EpisodeRows] = []
    continuity = 0.0
    for eid in sorted({int(x) for x in episode.tolist()}):
        idx = torch.nonzero(episode == eid).flatten()
        idx = idx[torch.argsort(times[idx], stable=True)]
        te = times[idx]
        if int(idx.numel()) < 2:
            raise ValueError(f"{paths['tape']}: episode {eid} has <2 transitions")
        if not bool(torch.all(te[1:] - te[:-1] == c)):
            raise ValueError(f"{paths['tape']}: episode {eid} is not contiguous at c={c}")
        ze, une = z[idx], z_next[idx]
        mismatch = float((une[:-1] - ze[1:]).abs().max())
        continuity = max(continuity, mismatch)
        if mismatch > CONTINUITY_TOLERANCE:
            raise ValueError(
                f"{paths['tape']}: episode {eid} z_next/z continuity {mismatch:.3g}"
            )
        t0 = int(te[0])
        n_e = int(idx.numel())
        try:
            phi_b = torch.tensor([phi_map[(eid, t0 + k * c)] for k in range(n_e + 1)])
            ord_b = torch.tensor([ord_map[(eid, t0 + k * c)] for k in range(n_e + 1)],
                                 dtype=torch.long)
        except KeyError as exc:
            raise ValueError(
                f"{paths['labels']}: episode {eid} lacks a label at boundary {exc}; "
                "Phi' must cover every boundary 0..n including the terminal one. An "
                "episode that ended mid-chunk records its last label at a t that is "
                "not a multiple of c, which lands here"
            ) from exc
        out.append(EpisodeRows(
            key=f"{tape_sha[:12]}:{eid}", run=str(run_dir), ep_id=eid, t0=t0, c=c,
            z_bound=torch.cat([ze, une[-1:]], 0), u=u[idx], u_base=u_base[idx],
            delta=delta[idx], phi_bound=phi_b, ordinal_bound=ord_b,
        ))
    if not out:
        raise ValueError(f"{run_dir}: zero episodes loaded")

    record = {
        "run_dir": str(run_dir),
        "tape_sha256": tape_sha,
        "labels_sha256": sha256_file(paths["labels"]),
        "trace_sha256": sha256_file(paths["trace"]),
        "manifest_sha256": (sha256_file(run_dir / "manifest.json")
                            if (run_dir / "manifest.json").is_file() else None),
        "episodes": len(out),
        "transitions": n,
        "chunks_per_episode": sorted({ep.n for ep in out}),
        "max_z_next_continuity_error": continuity,
        "max_trace_audit_error": audit,
        "rows_with_delta": int((delta.abs().amax(dim=(1, 2)) > 0).sum()),
        "mean_abs_delta_applied": float(
            delta[delta.abs().amax(dim=(1, 2)) > 0].abs().mean()
        ) if bool((delta.abs().amax(dim=(1, 2)) > 0).any()) else 0.0,
    }
    # What the COLLECTOR actually used. Carried out so the caller can refuse a
    # --scale that disagrees with it: the trained interface's bound must match the
    # trust region the D1 deltas were drawn from, or it is regressing onto targets it
    # cannot represent. The calibration measured |delta| = 0.051/0.105/0.206 producing
    # zero milestone-4 events against 0.440 producing 4/24, so a bound set an order of
    # magnitude low yields an interface that cannot move the task and a round that
    # reports "training destroys the effect".
    man_path = run_dir / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        record["collector_actor_scale"] = man.get("actor_scale")
        record["collector_residual_start_chunk"] = man.get("residual_start_chunk")
        record["collector_role"] = man.get("role")
    return out, record


def build_bundle(episodes: Sequence[EpisodeRows], mu: Tensor, sd: Tensor) -> Bundle:
    """Flatten episodes and move every latent into the model pair's normalised space."""
    if not episodes:
        raise ValueError("no episodes to bundle")
    c = episodes[0].c
    adim = int(episodes[0].u.shape[2])
    zdim = int(episodes[0].z_bound.shape[1])
    z_parts, phi_parts, u_parts, ub_parts, dl_parts = [], [], [], [], []
    row_ep, row_k, row_b = [], [], []
    ep_row_start, ep_len = [], []
    rows = 0
    bounds = 0
    for e, ep in enumerate(episodes):
        if ep.c != c or int(ep.u.shape[2]) != adim or int(ep.z_bound.shape[1]) != zdim:
            raise ValueError(f"episode {ep.key}: dims differ from the first episode")
        n_e = ep.n
        z_parts.append(ep.z_bound)
        phi_parts.append(ep.phi_bound)
        u_parts.append(ep.u)
        ub_parts.append(ep.u_base)
        dl_parts.append(ep.delta)
        row_ep.append(torch.full((n_e,), e, dtype=torch.long))
        row_k.append(torch.arange(n_e, dtype=torch.long))
        row_b.append(bounds + torch.arange(n_e, dtype=torch.long))
        ep_row_start.append(rows)
        ep_len.append(n_e)
        rows += n_e
        bounds += n_e + 1
    z_bound = torch.cat(z_parts, 0)
    if mu.shape != (zdim,) or sd.shape != (zdim,):
        raise ValueError(f"normaliser shape {tuple(mu.shape)} != ({zdim},)")
    z_bound = (z_bound - mu) / sd
    if not bool(torch.isfinite(z_bound).all()):
        raise ValueError("non-finite latents after normalisation")
    return Bundle(
        zdim=zdim, c=c, adim=adim, z_bound=z_bound,
        phi_bound=torch.cat(phi_parts, 0), u=torch.cat(u_parts, 0),
        u_base=torch.cat(ub_parts, 0), delta=torch.cat(dl_parts, 0),
        row_episode=torch.cat(row_ep, 0), row_chunk=torch.cat(row_k, 0),
        row_boundary=torch.cat(row_b, 0),
        ep_row_start=torch.tensor(ep_row_start, dtype=torch.long),
        ep_len=torch.tensor(ep_len, dtype=torch.long),
        keys=[ep.key for ep in episodes],
    )


# ------------------------------------------------------------------- row selection


def eligible_rows(bundle: Bundle, horizon: int, include_zero_delta: bool
                  ) -> tuple[Tensor, dict[str, int]]:
    """Rows whose H-chunk window stays inside their own episode.

    A row at chunk k of an episode with n chunks is eligible when k + H <= n, so the
    window consumes actions k..k+H-1 and lands on boundary k+H, which exists for every
    k including k = n - H.  Nothing is truncated: the full 75 chunks of a chain3@750
    episode are offered, unlike the `L = min(min_chunks, 24)` convention elsewhere in
    this repo, which would discard the stall basin this round is about.

    An episode with fewer than H chunks contributes no window at all.  That is counted
    into ``episodes_shorter_than_horizon`` and printed, so a run whose tapes contain
    early-terminating episodes says how many rows it is training on and how many
    episodes reached it; the count is identical across arms, which read the same rows.
    """
    if horizon < 1:
        raise ValueError("--horizon must be >= 1")
    starts: list[Tensor] = []
    short = 0
    for e in range(bundle.n_episodes):
        n_e = int(bundle.ep_len[e])
        if n_e < horizon:
            # An episode that ended before H chunks offers no window. It is counted
            # rather than dropped in silence, which is what
            # train_v205_action_belief.load_episodes:104-108 does to a mismatched tape.
            short += 1
            continue
        base = int(bundle.ep_row_start[e])
        starts.append(base + torch.arange(n_e - horizon + 1, dtype=torch.long))
    if not starts:
        raise ValueError(
            f"no episode has at least --horizon {horizon} chunks; "
            f"episode lengths {sorted({int(x) for x in bundle.ep_len.tolist()})}"
        )
    windows = torch.cat(starts, 0)
    nonzero = bundle.delta.abs().amax(dim=(1, 2)) > 0
    kept = windows if include_zero_delta else windows[nonzero[windows]]
    stats = {
        "episodes": bundle.n_episodes,
        "episodes_shorter_than_horizon": short,
        "episodes_contributing": bundle.n_episodes - short,
        "tape_rows": bundle.n_rows,
        "windows": int(windows.numel()),
        "windows_zero_delta": int((~nonzero[windows]).sum()),
        "windows_kept": int(kept.numel()),
    }
    if int(kept.numel()) == 0:
        raise ValueError(
            "no training rows survive the zero-delta filter: every candidate window "
            "starts on a row whose recorded delta is identically zero (a base-role "
            "tape, or D1 rows before the collector's --residual-start-chunk). Pass "
            "--include-zero-delta to train on the zero target deliberately, or add a "
            "D1 run to --tapes."
        )
    return kept, stats


def window_actions(flat: Tensor, rows: Tensor, horizon: int) -> Tensor:
    """Gather [M, H, c, adim] action windows; windows never cross an episode."""
    offsets = torch.arange(horizon, device=rows.device)
    return flat[rows.unsqueeze(1) + offsets.unsqueeze(0)]


# ---------------------------------------------------------------------- the Phi' head


def make_phi_head(zdim: int) -> nn.Module:
    """phi_hat(z) -> Phi'.

    Same shape as build_v241_model_pair.make_phi, a different target and a different
    fitting protocol; see WHY NOT model_pair['phi'] in the module docstring.
    """
    return nn.Sequential(
        nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 1)
    )


def fit_phi_head(z_bound: Tensor, phi_bound: Tensor, bound_episode: Tensor,
                 keys: Sequence[str], *, seed: int, steps: int, batch: int,
                 lr: float, weight_decay: float, holdout_fraction: float,
                 region: Tensor | None = None
                 ) -> tuple[nn.Module, dict[str, Any]]:
    """Fit Phi' on real latents with an EPISODE-DISJOINT split, on CPU, from a fixed seed.

    Nothing in this function's inputs depends on --wm-arm, and it is deliberately run on
    CPU, so the M1 and M0 runs of the round receive a bit-identical head.  The caller
    asserts that by digesting the resulting state dict.

    ``region`` is a boolean mask over boundaries marking the DECISION REGION - the
    boundaries at which the interface is actually active.  It is not a refinement.
    Measured on 24 chain3 base episodes (2026-09-12), a nonlinear function of the
    timestep alone explains 97.2% of Phi' variance, because the frozen policy runs a
    stereotyped script for its first 260 steps and then stalls.  A head fitted over all
    boundaries therefore learns mostly a clock, and inside the stall - where the clock
    is flat and 92.5% of the remaining variance is across-episode at the same timestep -
    it ranks states WORSE than a head fitted on the region alone: over five
    episode-disjoint splits the decision-region held-out correlation was
    -0.912 / -0.136 / -0.411 / -0.433 / +0.054 fitting globally, against
    +0.887 / +0.764 / -0.033 / +0.361 / -0.206 fitting on the region.  The direction
    replicates (worse in 5/5, negative in 4/5); the magnitude does not, and at 24
    episodes neither construction is well determined - which is why BOTH correlations
    are always reported and the region one is the number that gates the round.

    When ``region`` is given, the head is FITTED on region boundaries only, and metrics
    are reported over the region and over all boundaries separately.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("--phi-holdout must lie strictly between zero and one")
    z_bound = z_bound.detach().float().cpu()
    phi_bound = phi_bound.detach().float().cpu()
    ranked = sorted(((stable_fraction(seed, key), e) for e, key in enumerate(keys)))
    holdout_eps = {e for frac, e in ranked if frac < holdout_fraction}
    if not holdout_eps:
        holdout_eps.add(ranked[0][1])
    if len(holdout_eps) == len(ranked):
        holdout_eps.discard(ranked[-1][1])
    if not holdout_eps or len(holdout_eps) == len(ranked):
        raise ValueError("episode-disjoint Phi' split produced an empty side")
    is_holdout = torch.tensor([int(e) in holdout_eps for e in bound_episode.tolist()])
    if region is None:
        region = torch.ones(len(phi_bound), dtype=torch.bool)
    region = region.detach().cpu().bool()
    if int(region.sum()) == 0:
        raise ValueError("the Phi' decision region selected no boundaries")
    train_idx = torch.nonzero(~is_holdout & region).flatten()
    hold_idx = torch.nonzero(is_holdout & region).flatten()
    all_hold_idx = torch.nonzero(is_holdout).flatten()
    if int(train_idx.numel()) == 0 or int(hold_idx.numel()) == 0:
        raise ValueError("Phi' split left one side without boundary rows")

    torch.manual_seed(seed)
    net = make_phi_head(int(z_bound.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed + 1)
    net.train()
    for it in range(steps):
        pick = train_idx[torch.randint(0, int(train_idx.numel()), (batch,),
                                       generator=gen)]
        loss = ((net(z_bound[pick]).squeeze(-1) - phi_bound[pick]) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if it % max(steps // 4, 1) == 0:
            print(f"  phi' head it {it}: mse {float(loss.detach()):.5f}", flush=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        pred = torch.cat([net(z_bound[i:i + 1024]).squeeze(-1)
                          for i in range(0, int(z_bound.shape[0]), 1024)])
    tr = pred[train_idx].numpy(), phi_bound[train_idx].numpy()
    ho = pred[hold_idx].numpy(), phi_bound[hold_idx].numpy()
    metrics = {
        "seed": seed, "steps": steps, "batch": batch, "lr": lr,
        "weight_decay": weight_decay, "holdout_fraction": holdout_fraction,
        "train_boundaries": int(train_idx.numel()),
        "holdout_boundaries": int(hold_idx.numel()),
        "holdout_episodes": len(holdout_eps),
        "train_mse": float(((pred[train_idx] - phi_bound[train_idx]) ** 2).mean()),
        "holdout_mse": float(((pred[hold_idx] - phi_bound[hold_idx]) ** 2).mean()),
        "train_corr": pearson(*tr),
        "holdout_corr": pearson(*ho),
        "holdout_spearman": spearman(*ho),
        # `holdout_corr` above is already region-restricted whenever a region is
        # given; this is the same head scored over EVERY held-out boundary, kept so
        # the two can never be confused for one another in a later report.
        "holdout_corr_all_boundaries": pearson(
            pred[all_hold_idx].numpy(), phi_bound[all_hold_idx].numpy()),
        "region_boundaries": int(region.sum()),
        "region_fraction": float(region.float().mean()),
        "region_restricted": bool(int(region.sum()) < len(region)),
        "phi_target_mean": float(phi_bound.mean()),
        "phi_target_std": float(phi_bound.std(unbiased=False)),
        "device": "cpu",
        "arm_dependent_inputs": False,
    }
    return net, metrics


# ------------------------------------------------------------------------ rollouts


@torch.no_grad()
def episode_beliefs(model: RawLatentWM, bundle: Bundle, device: torch.device,
                    group: int = EPISODE_BATCH) -> Tensor:
    """b_t for every tape row, from the REAL prefix under the REAL executed actions.

    Convention copied from RawLatentWM.roll: b_t consumes e(z_t) and u_{t-1}, so the
    belief at a row never sees that row's own action.  Episodes shorter than the group
    maximum are zero-padded on the right, which cannot affect any earlier state.
    """
    lengths = bundle.ep_len.tolist()
    out = torch.zeros(bundle.n_rows, model.bdim, device=device)
    for start in range(0, bundle.n_episodes, group):
        idx = list(range(start, min(start + group, bundle.n_episodes)))
        lmax = max(lengths[e] for e in idx)
        zs = torch.zeros(len(idx), lmax, bundle.zdim, device=device)
        us = torch.zeros(len(idx), lmax, bundle.c, bundle.adim, device=device)
        for g, e in enumerate(idx):
            n_e = lengths[e]
            r0 = int(bundle.ep_row_start[e])
            b0 = int(bundle.row_boundary[r0])
            zs[g, :n_e] = bundle.z_bound[b0:b0 + n_e].to(device)
            us[g, :n_e] = bundle.u[r0:r0 + n_e].to(device)
        beliefs, _ = model.roll(zs, us)
        for g, e in enumerate(idx):
            n_e = lengths[e]
            r0 = int(bundle.ep_row_start[e])
            out[r0:r0 + n_e] = beliefs[g, :n_e]
    return out


@torch.no_grad()
def roll_latent(model: RawLatentWM, b0: Tensor, z0: Tensor, u_base: Tensor,
                delta: Tensor | None, horizon: int, *, base_action: str = "recorded",
                prior: nn.Module | None = None, c: int | None = None,
                adim: int | None = None) -> Tensor:
    """z~_{t+Hc}: roll the transition H chunks with the candidate applied at EVERY step.

    ``u_base`` is [M, H, c, adim] (ignored when base_action == 'prior'); ``delta`` is the
    sustained candidate residual, or None for the base roll.  The belief update is the
    same one build_v241_model_pair.rollout_mse_for_episode uses, so imagined dynamics
    here match the dynamics that model was scored under.
    """
    if base_action not in ("recorded", "prior"):
        raise ValueError(f"unknown base-action mode {base_action!r}")
    if base_action == "prior" and prior is None:
        raise ValueError("base_action='prior' needs the model pair's action prior")
    z, b = z0, b0
    for h in range(horizon):
        if base_action == "recorded":
            base = u_base[:, h]
        else:
            base = prior(z).view(-1, int(c), int(adim))
        u = base if delta is None else base + delta[:, h]
        z = model.predict(b, z, u)
        if h + 1 < horizon:
            b = model.step(b, model.enc(z), u)
    return z


@torch.no_grad()
def predicted_advantage(model: RawLatentWM, phi_head: nn.Module, z0: Tensor, b0: Tensor,
                        u_base: Tensor, delta: Tensor, horizon: int, *,
                        base_action: str = "recorded", prior: nn.Module | None = None,
                        c: int | None = None, adim: int | None = None) -> Tensor:
    """A = phi_hat(roll under u_base + delta) - phi_hat(roll under u_base).

    Both rolls start from the same (z_t, b_t) and use the same base-action sequence, so
    the head's bias at the start state cancels and only the transition's response to the
    sustained residual survives.
    """
    kw = dict(base_action=base_action, prior=prior, c=c, adim=adim)
    z_cand = roll_latent(model, b0, z0, u_base, delta, horizon, **kw)
    z_base = roll_latent(model, b0, z0, u_base, None, horizon, **kw)
    return phi_head(z_cand).squeeze(-1) - phi_head(z_base).squeeze(-1)


def observed_advantage(phi_now: Tensor, phi_future: Tensor) -> Tensor:
    """A = Phi'(real boundary t+Hc) - Phi'(real boundary t).

    No model argument exists in this signature: the no-rollout control cannot consult a
    transition even by accident.
    """
    return phi_future - phi_now


def current_advantage(phi_now: Tensor) -> Tensor:
    """A = Phi'(real boundary t). Invariant to the action and to the horizon."""
    return phi_now.clone()


@torch.no_grad()
def compute_advantage(mode: str, bundle: Bundle, rows: Tensor, horizon: int, *,
                      model: RawLatentWM | None = None,
                      phi_head: nn.Module | None = None,
                      beliefs: Tensor | None = None,
                      prior: nn.Module | None = None,
                      base_action: str = "recorded",
                      phi_readout: str = "label",
                      device: torch.device = torch.device("cpu"),
                      batch: int = ADV_BATCH) -> Tensor:
    """Advantage for every eligible row, under one of the three registered modes."""
    if mode not in ADVANTAGES:
        raise ValueError(f"unknown advantage mode {mode!r}")
    if phi_readout not in ("label", "head"):
        raise ValueError(f"unknown --phi-readout {phi_readout!r}")
    rows = rows.cpu()
    bound_now = bundle.row_boundary[rows]
    bound_future = bound_now + horizon

    if mode in ("observed", "current"):
        def read(bound: Tensor) -> Tensor:
            if phi_readout == "label":
                return bundle.phi_bound[bound]
            if phi_head is None:
                raise ValueError("--phi-readout head needs a fitted Phi' head")
            with torch.no_grad():
                return torch.cat([
                    phi_head(bundle.z_bound[bound[i:i + batch]].to(device)
                             ).squeeze(-1).cpu()
                    for i in range(0, int(bound.numel()), batch)])

        if mode == "current":
            return current_advantage(read(bound_now))
        return observed_advantage(read(bound_now), read(bound_future))

    if model is None or phi_head is None or beliefs is None:
        raise ValueError("--advantage predicted needs a model, a Phi' head and beliefs")
    out: list[Tensor] = []
    for i in range(0, int(rows.numel()), batch):
        chunk = rows[i:i + batch]
        z0 = bundle.z_bound[bundle.row_boundary[chunk]].to(device)
        b0 = beliefs[chunk.to(beliefs.device)].to(device)
        ub = window_actions(bundle.u_base, chunk, horizon).to(device)
        dl = window_actions(bundle.delta, chunk, horizon).to(device)
        out.append(predicted_advantage(
            model, phi_head, z0, b0, ub, dl, horizon, base_action=base_action,
            prior=prior, c=bundle.c, adim=bundle.adim).cpu())
    return torch.cat(out, 0)


# ------------------------------------------------------------------------- training


def train_actor(z_rows: Tensor, targets: Tensor, advantage: Tensor, *, zdim: int,
                c: int, adim: int, scale: float, epochs: int, batch: int, seed: int,
                restart: int, lr: float, weight_decay: float,
                device: torch.device) -> tuple[ResidualActor, dict[str, Any]]:
    """Advantage-weighted regression onto the collector's recorded residual."""
    temperature = float(advantage.std(unbiased=False).clamp_min(1e-6))
    a_dev = advantage.to(device)
    torch.manual_seed(seed + restart)
    actor = ResidualActor(zdim, c, adim, scale=scale).to(device)
    opt = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(1000 * seed + restart)
    m = int(z_rows.shape[0])
    actor.train()
    last = float("nan")
    for it in range(epochs):
        pick = torch.randint(0, m, (min(batch, m),), generator=gen).to(device)
        w = torch.softmax(a_dev[pick] / temperature, 0) * int(pick.numel())
        d = actor(z_rows[pick])
        loss = (w.unsqueeze(-1).unsqueeze(-1) * (d - targets[pick]) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        last = float(loss.detach())
        if it % max(epochs // 4, 1) == 0:
            print(f"    restart {restart} it {it}: awr loss {last:.6f}", flush=True)
    actor.eval()
    with torch.no_grad():
        pred = torch.cat([actor(z_rows[i:i + 512]).cpu()
                          for i in range(0, m, 512)])
    tgt = targets.cpu()
    flat_p, flat_t = pred.flatten(1), tgt.flatten(1)
    denom = flat_p.norm(dim=1) * flat_t.norm(dim=1)
    cos = torch.where(denom > 0, (flat_p * flat_t).sum(1) / denom.clamp_min(1e-12),
                      torch.zeros_like(denom))
    stats = {
        "restart": restart,
        "final_loss": last,
        "temperature": temperature,
        "mean_abs_delta": float(pred.abs().mean()),
        "fraction_of_bound": float(pred.abs().mean()) / scale if scale > 0 else None,
        "max_abs_delta": float(pred.abs().max()),
        "mean_cosine_with_target": float(cos.mean()),
        "mean_abs_target": float(tgt.abs().mean()),
        "rows": m,
    }
    return actor, stats


# -------------------------------------------------------------------- checkpointing


def verify_checkpoint(path: Path, actor: ResidualActor, dims: tuple[int, int, int],
                      device: torch.device) -> dict[str, Any]:
    """Reload exactly what run_v206 reloads and re-run the actor on a fixed probe."""
    ck = torch.load(path, weights_only=False, map_location="cpu")
    required = {"state_dict", "zdim", "condition", "c", "adim", "scale", "rawwm",
                "dims", "mu", "sd", "bmu", "bsd", "task", "arm"}
    missing = sorted(required - set(ck))
    if missing:
        raise RuntimeError(f"{path}: checkpoint is missing run_v206 fields {missing}")
    zdim, c, adim = dims
    if tuple(ck["dims"]) != dims or int(ck["zdim"]) != zdim:
        raise RuntimeError(f"{path}: dims changed during serialization")
    probe_gen = torch.Generator().manual_seed(PROBE_SEED)
    probe = torch.randn(4, zdim, generator=probe_gen)
    reloaded = ResidualActor(int(ck["zdim"]), int(ck["c"]), int(ck["adim"]),
                             scale=float(ck["scale"]))
    reloaded.load_state_dict(ck["state_dict"])
    reloaded.eval()
    wm = RawLatentWM(zdim, c, adim)
    wm.load_state_dict(ck["rawwm"], strict=True)
    with torch.no_grad():
        want = actor.to("cpu")(probe)
        got = reloaded(probe)
    actor.to(device)
    err = float((want - got).abs().max())
    if err != 0.0:
        raise RuntimeError(f"{path}: reloaded actor differs by {err:.3g} on the probe")
    return {"probe_max_abs_delta": float(got.abs().max()), "probe_roundtrip_error": err}


# ------------------------------------------------------------------------------ cli


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="train a residual interface from a WM arm")
    ap.add_argument("--model-pair", type=Path, required=True)
    ap.add_argument("--wm-arm", choices=MODEL_ARMS, required=True)
    ap.add_argument("--tapes", type=Path, nargs="+", required=True,
                    help="v250 run directories, each with tape.pt/labels.pt/trace.pt")
    ap.add_argument("--task", default="chain3_lr2")
    ap.add_argument("--advantage", choices=ADVANTAGES, required=True)
    ap.add_argument("--horizon", type=int, default=10,
                    help="chunks the candidate residual is sustained for; v229 measured "
                         "a single-chunk perturbation to be inert (71/72 identical)")
    ap.add_argument("--base-action", choices=("recorded", "prior"), default="recorded",
                    help="'recorded' reuses the episode's own base chunks at imagined "
                         "latents (documented approximation, identical across arms); "
                         "'prior' calls model_pair's action prior (train MSE .0015, "
                         "holdout .0442)")
    ap.add_argument("--phi-readout", choices=("label", "head"), default="head",
                    help="how observed/current read Phi' off REAL latents: the supplied "
                         "annotation, or the same head the predicted arm uses. No "
                         "effect under --advantage predicted, which has no annotation "
                         "at an imagined latent and must use the head")
    ap.add_argument("--include-zero-delta", action="store_true",
                    help="keep rows whose recorded delta is identically zero; their AWR "
                         "target is the zero residual")
    ap.add_argument("--actor-epochs", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--scale", type=float, required=True,
                    help="the residual bound. REQUIRED, with no default: the basin "
                         "calibration measured |delta| = 0.051/0.105/0.206 producing "
                         "ZERO milestone-4 events and 0.440 producing 4/24, so a "
                         "forgotten scale silently trains an interface an order of "
                         "magnitude below the threshold for any effect at all.")
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--actor-weight-decay", type=float, default=1e-4)
    ap.add_argument("--phi-head", type=Path, default=None,
                    help="reuse a phi_head.pt fitted by an earlier run of this script; "
                         "refused unless it was fitted under this model pair's mu/sd")
    ap.add_argument("--phi-seed", type=int, default=251)
    ap.add_argument("--phi-steps", type=int, default=3000)
    ap.add_argument("--phi-batch", type=int, default=256)
    ap.add_argument("--phi-holdout", type=float, default=0.2)
    ap.add_argument("--phi-region-from-chunk", type=int, default=None,
                    required=False,
                    help="fit the Phi' head only on boundaries at or after this chunk "
                         "index within an episode - the DECISION REGION where the "
                         "interface is active. Default None fits on every boundary, "
                         "which on chain3 yields a head that is mostly a clock (a "
                         "nonlinear function of t alone explains 97.2%% of Phi' "
                         "variance) and ranks stall states worse than a region-fitted "
                         "head in 5/5 measured splits. Set this to the same value as "
                         "the collector's --residual-start-chunk.")
    ap.add_argument("--assert-phi-digest", default=None,
                    help="fail unless the fitted Phi' head hashes to this digest; the "
                         "round pins the M1 digest and every other arm must match it")
    ap.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=251)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--out-root", type=Path, default=OUT)
    return ap.parse_args(argv)


def validate_args(a: argparse.Namespace) -> None:
    for name in ("horizon", "actor_epochs", "batch", "restarts", "phi_steps",
                 "phi_batch"):
        if int(getattr(a, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if a.scale <= 0:
        raise ValueError("--scale must be positive")
    if not 0.0 < a.phi_holdout < 1.0:
        raise ValueError("--phi-holdout must lie strictly between zero and one")


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(choice)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    validate_args(a)
    if a.threads:
        torch.set_num_threads(int(a.threads))
    device = resolve_device(a.device)

    pair_path = Path(a.model_pair).resolve()
    pair = torch.load(pair_path, weights_only=False, map_location="cpu")
    for key in ("models", "dims", "mu", "sd", "task"):
        if key not in pair:
            raise ValueError(f"{pair_path}: model pair lacks {key!r}")
    if pair["task"] != a.task:
        raise ValueError(f"{pair_path}: task {pair['task']!r} != --task {a.task!r}")
    dims = tuple(int(x) for x in pair["dims"])
    if len(dims) != 3:
        raise ValueError(f"{pair_path}: invalid dims {dims}")
    zdim, c, adim = dims
    if a.wm_arm not in pair["models"]:
        raise ValueError(
            f"{pair_path}: arm {a.wm_arm!r} absent; present {sorted(pair['models'])}"
        )
    mu = pair["mu"].detach().float().cpu()
    sd = pair["sd"].detach().float().cpu()
    if mu.shape != (zdim,) or sd.shape != (zdim,):
        raise ValueError(f"{pair_path}: normaliser shape mismatch")
    if not bool(torch.isfinite(sd).all()) or not bool((sd > 0).all()):
        raise ValueError(f"{pair_path}: non-positive or non-finite scale")

    # --- data, loaded and dimension-checked BEFORE anything is fitted --------------
    episodes: list[EpisodeRows] = []
    run_records: list[dict[str, Any]] = []
    for run_dir in a.tapes:
        eps, record = load_run(run_dir, a.task, dims)
        episodes.extend(eps)
        run_records.append(record)
        print(f"loaded {record['episodes']} episodes / {record['transitions']} rows "
              f"from {record['run_dir']} (chunks {record['chunks_per_episode']})",
              flush=True)

    # The interface's BOUND must equal the trust region its targets were drawn from.
    # Regressing a scale-0.05 actor onto scale-0.80 collector deltas asks it to
    # represent targets 16x outside its own range; the calibration measured |delta| =
    # 0.051/0.105/0.206 producing zero milestone-4 events against 0.440 producing
    # 4/24, so the resulting interface cannot move the task at all and the round
    # would report "training destroys the effect" with nothing flagged.
    collector_scales = {r["collector_actor_scale"] for r in run_records
                        if r.get("collector_actor_scale") is not None}
    if len(collector_scales) > 1:
        raise SystemExit(f"the D1 runs were collected at different residual scales "
                         f"{sorted(collector_scales)}; a single --scale cannot match "
                         f"them and the norm confound the fixed-scale pool removed "
                         f"would be back")
    if collector_scales:
        cs = float(next(iter(collector_scales)))
        if abs(cs - float(a.scale)) > 1e-9:
            raise SystemExit(
                f"--scale {a.scale} does not match the D1 collector scale {cs}. The "
                f"interface would be bounded away from the trust region its AWR "
                f"targets come from. Pass --scale {cs}.")
    starts = {r["collector_residual_start_chunk"] for r in run_records
              if r.get("collector_residual_start_chunk") is not None
              and r.get("collector_role") == "resid"}
    if len(starts) > 1:
        raise SystemExit(f"D1 runs disagree on --residual-start-chunk {sorted(starts)}")
    if starts and a.phi_region_from_chunk is not None:
        st = int(next(iter(starts)))
        if int(a.phi_region_from_chunk) != st:
            raise SystemExit(
                f"--phi-region-from-chunk {a.phi_region_from_chunk} does not match the "
                f"collectors' --residual-start-chunk {st}; the head would be fitted on "
                f"a different region than the one the interface acts in")

    seen: set[str] = set()
    for ep in episodes:
        if ep.key in seen:
            raise ValueError(f"duplicate episode key {ep.key}: overlapping --tapes")
        seen.add(ep.key)
    bundle = build_bundle(episodes, mu, sd)
    rows, row_stats = eligible_rows(bundle, a.horizon, a.include_zero_delta)
    print(f"{bundle.n_episodes} episodes "
          f"({row_stats['episodes_contributing']} with >= H={a.horizon} chunks, "
          f"{row_stats['episodes_shorter_than_horizon']} shorter), "
          f"{bundle.n_rows} tape rows, "
          f"{row_stats['windows']} H={a.horizon} windows, "
          f"{row_stats['windows_zero_delta']} zero-delta, "
          f"{row_stats['windows_kept']} kept", flush=True)

    # --- the shared Phi' head: arm independent, CPU, fixed seed --------------------
    bound_episode = torch.repeat_interleave(
        torch.arange(bundle.n_episodes), bundle.ep_len + 1
    )
    # boundary index WITHIN its episode: episode e contributes ep_len[e] + 1 boundaries
    bound_chunk = torch.cat([torch.arange(int(n) + 1) for n in bundle.ep_len]) \
        if bundle.n_episodes else torch.zeros(0, dtype=torch.long)
    if int(bound_chunk.numel()) != int(bundle.z_bound.shape[0]):
        raise AssertionError(
            f"boundary bookkeeping: {int(bound_chunk.numel())} chunk indices for "
            f"{int(bundle.z_bound.shape[0])} boundaries")
    phi_region = (None if a.phi_region_from_chunk is None
                  else bound_chunk >= int(a.phi_region_from_chunk))
    # The head reads NORMALISED latents, so a head fitted under a different mu/sd is
    # being evaluated in coordinates it never saw - the v235 failure, in the one path
    # that bypasses the coordinate contract above.  --assert-phi-digest pins WHICH head
    # is used, not which space it belongs to, so the space is pinned separately here.
    normaliser_digest = state_dict_digest({"mu": mu, "sd": sd})
    tape_digests = sorted(r["tape_sha256"] for r in run_records)
    if a.phi_head is not None:
        loaded = torch.load(Path(a.phi_head).resolve(), weights_only=False,
                            map_location="cpu")
        if int(loaded.get("zdim", -1)) != zdim:
            raise ValueError(
                f"{a.phi_head}: Phi' head zdim {loaded.get('zdim')} != {zdim}"
            )
        stored_normaliser = loaded.get("normaliser_digest")
        if stored_normaliser is None:
            raise ValueError(
                f"{a.phi_head}: this Phi' head carries no normaliser provenance, so "
                "there is no way to check it was fitted in this model pair's "
                "normalised space. Refit it with this script rather than reusing it."
            )
        if stored_normaliser != normaliser_digest:
            raise ValueError(
                f"{a.phi_head}: Phi' head was fitted under normaliser "
                f"{stored_normaliser[:16]} but this run's model pair {pair_path} has "
                f"{normaliser_digest[:16]}. The head would be read in coordinates it "
                "never saw; refit it against this model pair."
            )
        phi_head = make_phi_head(zdim)
        phi_head.load_state_dict(loaded["state_dict"], strict=True)
        phi_head.eval()
        for p in phi_head.parameters():
            p.requires_grad_(False)
        phi_metrics = dict(loaded.get("metrics", {}))
        phi_metrics["reused_from"] = str(Path(a.phi_head).resolve())
        # Not fatal: refitting on D0 alone and reusing on D0+D1 is a legitimate thing
        # to do. Recorded so that summary.json says which rows the head actually saw.
        phi_metrics["fitted_on_these_tapes"] = (
            loaded.get("tape_digests") == tape_digests
        )
    else:
        phi_head, phi_metrics = fit_phi_head(
            bundle.z_bound, bundle.phi_bound, bound_episode, bundle.keys,
            seed=a.phi_seed, steps=a.phi_steps, batch=a.phi_batch, lr=1e-3,
            weight_decay=1e-2, holdout_fraction=a.phi_holdout, region=phi_region)
    phi_digest = state_dict_digest(phi_head.state_dict())
    if a.assert_phi_digest is not None and phi_digest != a.assert_phi_digest:
        raise RuntimeError(
            f"Phi' head digest {phi_digest} != --assert-phi-digest "
            f"{a.assert_phi_digest}; the arms of this round are not sharing one head"
        )
    for p in phi_head.parameters():
        if p.requires_grad:
            raise AssertionError("the Phi' head must be frozen before any advantage")
    print(f"phi' head digest {phi_digest[:16]} "
          f"held-out corr (decision region) {phi_metrics.get('holdout_corr')} "
          f"| (all boundaries) {phi_metrics.get('holdout_corr_all_boundaries')} "
          f"| region {phi_metrics.get('region_boundaries')} boundaries "
          f"| mse {phi_metrics.get('holdout_mse')}", flush=True)
    phi_head.to(device).eval()

    # --- the selected transition ---------------------------------------------------
    model = RawLatentWM(zdim, c, adim)
    model.load_state_dict(pair["models"][a.wm_arm], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = None
    if a.base_action == "prior":
        if "prior" not in pair:
            raise ValueError(f"{pair_path}: --base-action prior needs the shared prior")
        prior = make_prior(zdim, c, adim)
        prior.load_state_dict(pair["prior"], strict=True)
        prior.to(device).eval()
        for p in prior.parameters():
            p.requires_grad_(False)

    beliefs = episode_beliefs(model, bundle, device)
    bmu = beliefs.mean(0).cpu()
    bsd = (beliefs.std(0, unbiased=False) + 1e-6).cpu()

    advantage = compute_advantage(
        a.advantage, bundle, rows, a.horizon, model=model, phi_head=phi_head,
        beliefs=beliefs, prior=prior, base_action=a.base_action,
        phi_readout=a.phi_readout, device=device)
    if int(advantage.numel()) != int(rows.numel()):
        raise AssertionError("advantage length does not match the row set")
    adv_stats = distribution(advantage)
    print(f"advantage[{a.advantage}] mean {adv_stats['mean']:+.5f} "
          f"std {adv_stats['std']:.5f} range [{adv_stats['min']:+.5f}, "
          f"{adv_stats['max']:+.5f}] pos {adv_stats['fraction_positive']:.3f}",
          flush=True)

    z_rows = bundle.z_bound[bundle.row_boundary[rows]].to(device)
    targets = bundle.delta[rows].to(device)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = a.out_tag or f"{a.wm_arm}_{a.advantage}"
    out = Path(a.out_root) / f"{a.task}_{tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"state_dict": {k: v.cpu() for k, v in phi_head.state_dict().items()},
                "zdim": zdim, "digest": phi_digest, "metrics": phi_metrics,
                "signal": "Phi' from lcwm/v250_progress.tape_progress",
                "space": "model-pair normalised latents",
                "normaliser_digest": normaliser_digest,
                "model_pair": str(pair_path),
                "tape_digests": tape_digests}, out / "phi_head.pt")

    wm_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    arm_rows: list[dict[str, Any]] = []
    for restart in range(a.restarts):
        actor, stats = train_actor(
            z_rows, targets, advantage, zdim=zdim, c=c, adim=adim, scale=a.scale,
            epochs=a.actor_epochs, batch=a.batch, seed=a.seed, restart=restart,
            lr=a.actor_lr, weight_decay=a.actor_weight_decay, device=device)
        ck = {
            # run_v206 schema
            "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "zdim": zdim, "condition": "raw", "c": c, "adim": adim, "scale": a.scale,
            "rawwm": wm_state, "dims": dims, "mu": mu, "sd": sd, "bmu": bmu, "bsd": bsd,
            "task": a.task, "arm": f"v251_{a.wm_arm}_{a.advantage}",
            # provenance
            "wm_arm": a.wm_arm, "advantage": a.advantage, "horizon": a.horizon,
            "base_action": a.base_action, "phi_readout": a.phi_readout,
            "include_zero_delta": bool(a.include_zero_delta), "restart": restart,
            "seed": a.seed, "actor_epochs": a.actor_epochs, "batch": a.batch,
            "target": "collector delta from trace.pt, AWR weighted",
            "actor_space": "model-pair normalised; mu/sd here ARE the model pair's",
            "normaliser_source": str(pair_path),
            "model_pair_sha256": sha256_file(pair_path),
            "phi_head_digest": phi_digest,
            "phi_head_state_dict": {k: v.cpu()
                                    for k, v in phi_head.state_dict().items()},
            "phi_holdout_corr": phi_metrics.get("holdout_corr"),
            "advantage_distribution": adv_stats,
            "train_stats": stats, "rows": row_stats, "utc": stamp, "git": git_head(),
            "deployed_outcomes_read": False,
        }
        path = out / f"actor_{restart}.pt"
        torch.save(ck, path)
        stats.update(verify_checkpoint(path, actor, dims, device))
        stats["actor_sha256"] = sha256_file(path)
        stats["actor_state_digest"] = state_dict_digest(actor.state_dict())
        arm_rows.append(stats)
        print(f"  restart {restart}: mean|Delta| {stats['mean_abs_delta']:.5f} "
              f"({100 * (stats['fraction_of_bound'] or 0):.0f}% of bound) "
              f"cos(target) {stats['mean_cosine_with_target']:+.3f}", flush=True)

    summary = {
        "schema": "v251_interface_v1",
        "utc": stamp,
        "task": a.task,
        "dims": list(dims),
        "wm_arm": a.wm_arm,
        "advantage": a.advantage,
        "horizon": a.horizon,
        "base_action": a.base_action,
        "base_action_assumption": (
            "recorded: the frozen policy's base chunk sequence is assumed unchanged by "
            "the residual, which is FALSE in general (pi0.5 replans every chunk). The "
            "approximation is identical across arms and cannot explain a between-arm "
            "difference."
        ),
        "phi_readout": a.phi_readout,
        # `predicted` reads an imagined latent, which has no annotation, so it always
        # goes through the head and --phi-readout is inert for it. Recorded so the flag
        # in `flags` is not read as a property of this run's advantage.
        "phi_readout_applies": a.advantage in ("observed", "current"),
        "include_zero_delta": bool(a.include_zero_delta),
        "scale": a.scale,
        "restarts": a.restarts,
        "actor_epochs": a.actor_epochs,
        "batch": a.batch,
        "actor_lr": a.actor_lr,
        "actor_weight_decay": a.actor_weight_decay,
        "seed": a.seed,
        "device": str(device),
        "flags": {k: ([str(x) for x in v] if isinstance(v, list) else
                      str(v) if isinstance(v, Path) else v)
                  for k, v in sorted(vars(a).items())},
        "model_pair": str(pair_path),
        "model_pair_sha256": sha256_file(pair_path),
        "model_pair_provenance": pair.get("provenance"),
        "runs": run_records,
        "rows": row_stats,
        "phi_head": {"digest": phi_digest, "path": str(out / "phi_head.pt"),
                     "normaliser_digest": normaliser_digest,
                     "tape_digests": tape_digests, **phi_metrics},
        "advantage_distribution": adv_stats,
        "advantage_temperature": float(advantage.std(unbiased=False).clamp_min(1e-6)),
        "belief_stats_source": (
            f"beliefs of the {a.wm_arm!r} arm over real rows; run_v206 ignores "
            "bmu/bsd under condition='raw' and no belief enters an observed/current "
            "advantage"
        ),
        "actors": arm_rows,
        "env_steps": 0,
        "deployed_outcomes_read": False,
        "outcome_json_read": False,
        "provenance": {
            "argv": list(sys.argv if argv is None else ["train_v251_interface", *argv]),
            "git": git_head(),
            "code_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
