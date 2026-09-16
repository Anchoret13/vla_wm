#!/usr/bin/env python
"""Build the frozen/updated world-model pair for an evolving-VLA round.

This is deliberately an offline builder: it reads already-collected deployment
tapes, never launches LIBERO, never reads policy deployment rates, and writes one
self-contained checkpoint for a later blind policy-pool scorer.

The four models isolate data and compute:

    stale             M0 trained on base-train episodes
    base_continue     M0 plus the update budget, still on base only
    updated           warm-start M0, balanced base + update episodes
    updated_shuffled  same as updated, but update actions come from another episode

All four use statistics fixed from base-train current states.  The shared policy
prior and potential are trained on a selectable content-hash subset of base-train
data only, so neither can leak the new-policy distribution into an
updated-vs-stale comparison.

WHAT THE SHARED POTENTIAL IS TRAINED ON.  By default the target is the discounted
time-to-success gamma**ceil((success_step - t)/c), which requires the per-episode
outcome fields and is identically zero on any episode that never succeeds.  That
default is left in place so that nothing already built changes meaning, but it is
degenerate on chain3_lr2@750, where the frozen policy terminates 2 of 96 episodes:
the head then sees a target that is zero on 94/96 episodes and it can reach a low
MSE by predicting the constant zero.  ``--phi-labels`` replaces that target with
Phi' from ``lcwm/v250_progress.py``, computed from the round's ``labels.pt``
artifacts and joined to the tape rows on an exact (episode, t) key.  Phi' is a
declared SCRIPTED PRIVILEGED STAND-IN for per-boundary human video annotation and it
is identical across all four arms.  It is computed from simulator object poses and
goal-predicate bits rather than from the outcome sidecar; that is not the same as
being outcome-free, since Phi' reaches its maximum exactly when every goal atom
holds.  What it does establish is that with it the builder reads no success or
success_step field at all.

WHAT WOULD OVERTURN A RESULT BUILT FROM THIS CHECKPOINT.  (a) ``updated`` beating
``stale`` on update_holdout while ``base_continue`` gains the same amount says the
extra optimiser budget, not the new data, moved the transition.  (b) ``updated``
and ``updated_shuffled`` moving together says the transition is not using the
actions, since those two arms differ only in cross-episode action substitution on
the update half of each batch.  (c) a learned rollout MSE at or above
``identity_mse`` says the model has not beaten "predict no change".  (d) for the
Phi head specifically, a holdout phi_mse at or above the target's VARIANCE (reported
as phi_target_var; phi_target_std is its square root, and comparing an MSE against a
standard deviation is a unit error) says the head is fitting the mean.

PRE-REGISTERED, in the sense that they are command-line inputs recorded in the
output provenance before anything is fitted: the tape content hashes, the
role+env_seed split rule and its seed, the four arms and their step budgets, the
evaluation horizons, and which potential target was used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from lcwm.v250_progress import APPROACH_REFERENCE, tape_progress  # noqa: E402
from train_v220_raw_latent_wm import RawLatentWM  # noqa: E402


DEFAULT_OUT = REPO / "results" / "v241_model_pair"
MODEL_NAMES = ("stale", "base_continue", "updated", "updated_shuffled")
#: Per-episode record sidecars, in resolution order: (file name, list key).
#: ``summary.json``/``episode_records`` is the collect_v121_deploy_latents.py
#: contract every earlier tape in this repo ships.  ``outcome.json``/``episodes``
#: is the same per-episode rows under the name collect_v250_chain3_round.py uses,
#: which splits them off precisely so a trainer is not tempted to read them.  Only
#: ``idx`` and ``seed`` are taken from either file unless the success-based
#: potential target is in use.
SIDECARS = (("summary.json", "episode_records"), ("outcome.json", "episodes"))


@dataclass
class Episode:
    key: str
    role: str
    tape_path: str
    tape_sha256: str
    episode_id: int
    env_seed: int
    z: torch.Tensor
    u: torch.Tensor
    z_next: torch.Tensor
    t: torch.Tensor
    success: bool | None
    success_step: int | None
    #: Supplied per-row potential target, aligned one-to-one with ``z``; ``None``
    #: when no --phi-labels artifact covers this tape.
    phi: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return int(self.z.shape[0])


@dataclass(frozen=True)
class TapeRecord:
    role: str
    path: str
    sha256: str
    summary_path: str
    summary_sha256: str
    episodes: int
    transitions: int
    #: Which SIDECARS entry supplied the per-episode records.
    records_key: str = "episode_records"
    #: False when success/success_step were never read from the sidecar.
    outcome_fields_read: bool = True


@dataclass(frozen=True)
class PhiLabels:
    """One --phi-labels artifact, summarised for the output provenance."""

    path: str
    sha256: str
    directory: str
    episodes: int
    rows: int
    phi_min: float
    phi_max: float
    phi_mean: float
    violation_rows: int


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def stable_fraction(seed: int, role: str, env_seed: int) -> float:
    """Map one role/environment-seed group to a stable split coordinate."""
    payload = f"v241:{seed}:{role}:{env_seed}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def resolve_sidecar(directory: Path) -> tuple[Path, str]:
    """Locate the per-episode record sidecar that sits next to a tape."""
    for name, key in SIDECARS:
        candidate = directory / name
        if candidate.is_file():
            return candidate, key
    expected = ", ".join(f"{name} ({key})" for name, key in SIDECARS)
    raise FileNotFoundError(
        f"{directory}: no episode-record sidecar; expected one of {expected}"
    )


def load_phi_labels(
    paths: Sequence[Path] | None, expected_task: str
) -> tuple[dict[str, dict[tuple[int, int], float]] | None, list[PhiLabels]]:
    """Read the round's label artifacts into one exact (episode, t) -> Phi' table.

    Each path is a ``labels.pt`` written by ``scripts/collect_v250_chain3_round.py``
    or the directory holding one.  Tables are keyed by that directory, because
    episode ids restart at zero in every tape and are only unique within one.
    """
    if paths is None:
        return None, []
    by_dir: dict[str, dict[tuple[int, int], float]] = {}
    records: list[PhiLabels] = []
    for raw in paths:
        resolved = raw.resolve()
        if resolved.is_dir():
            resolved = resolved / "labels.pt"
        if not resolved.is_file():
            raise FileNotFoundError(f"--phi-labels: {resolved}")
        directory = str(resolved.parent)
        if directory in by_dir:
            raise ValueError(f"--phi-labels names the same tape directory twice: {directory}")
        digest = sha256_file(resolved)
        labels = torch.load(resolved, weights_only=False, map_location="cpu")
        if not isinstance(labels, dict):
            raise ValueError(f"{resolved}: label artifact is not a dict")
        if labels.get("task") != expected_task:
            raise ValueError(
                f"{resolved}: task {labels.get('task')!r} != requested {expected_task!r}"
            )
        if "goal_atoms" not in labels:
            raise ValueError(f"{resolved}: label artifact lacks goal_atoms")
        progress = tape_progress(labels)
        phi = progress["phi"].detach().float().cpu()
        episode = progress["episode"].detach().cpu().long()
        times = progress["t"].detach().cpu().long()
        if not (len(phi) == len(episode) == len(times)):
            raise ValueError(f"{resolved}: progress fields have inconsistent lengths")
        if len(phi) == 0:
            raise ValueError(f"{resolved}: label artifact is empty")
        if not torch.isfinite(phi).all():
            raise ValueError(f"{resolved}: Phi' contains non-finite values")
        n_atoms = len(labels["goal_atoms"])
        lo, hi = float(phi.min()), float(phi.max())
        if lo < 0.0 or hi > n_atoms + 1e-6:
            raise ValueError(
                f"{resolved}: Phi' leaves [0, {n_atoms}] (min {lo:.4f}, max {hi:.4f})"
            )
        table: dict[tuple[int, int], float] = {}
        for eid, step, value in zip(episode.tolist(), times.tolist(), phi.tolist()):
            key = (int(eid), int(step))
            if key in table:
                raise ValueError(
                    f"{resolved}: duplicate label row for (episode={key[0]}, t={key[1]})"
                )
            table[key] = float(value)
        by_dir[directory] = table
        violation = progress.get("violation")
        records.append(
            PhiLabels(
                path=str(resolved),
                sha256=digest,
                directory=directory,
                episodes=int(len(torch.unique(episode))),
                rows=int(len(phi)),
                phi_min=lo,
                phi_max=hi,
                phi_mean=float(phi.mean()),
                violation_rows=0 if violation is None else int(violation.sum()),
            )
        )
    return by_dir, records


def _require_tensor(data: dict[str, Any], key: str, ndim: int) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"tape field {key!r} must be a rank-{ndim} tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"tape field {key!r} contains non-finite values")
    return value.detach().float().cpu() if value.is_floating_point() else value.cpu()


def load_tape(
    path: Path,
    role: str,
    expected_task: str,
    expected_dims: tuple[int, int, int] | None,
    phi_table: dict[tuple[int, int], float] | None = None,
    require_outcomes: bool = True,
) -> tuple[list[Episode], TapeRecord, tuple[int, int, int], float]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary_path, records_key = resolve_sidecar(path.parent)
    tape_sha = sha256_file(path)
    summary_sha = sha256_file(summary_path)
    data = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: checkpoint is not a dict")
    task = data.get("task")
    if task != expected_task:
        raise ValueError(f"{path}: task {task!r} != requested {expected_task!r}")

    required_meta = ("latent_dim", "c")
    missing = [key for key in required_meta if key not in data]
    if missing:
        raise ValueError(f"{path}: missing metadata {missing}")
    z = _require_tensor(data, "z", 2)
    u = _require_tensor(data, "u", 3)
    z_next = _require_tensor(data, "z_next", 2)
    episode = _require_tensor(data, "episode", 1).long()
    times = _require_tensor(data, "t", 1).long()
    n = len(z)
    if not (len(u) == len(z_next) == len(episode) == len(times) == n):
        raise ValueError(f"{path}: transition fields have inconsistent lengths")
    if z.shape != z_next.shape:
        raise ValueError(f"{path}: z {tuple(z.shape)} != z_next {tuple(z_next.shape)}")
    zdim, c, adim = int(z.shape[1]), int(u.shape[1]), int(u.shape[2])
    dims = (zdim, c, adim)
    if int(data["latent_dim"]) != zdim:
        raise ValueError(
            f"{path}: latent_dim={data['latent_dim']} but z has dimension {zdim}"
        )
    if int(data["c"]) != c:
        raise ValueError(f"{path}: c={data['c']} but actions have {c} committed steps")
    if expected_dims is not None and dims != expected_dims:
        raise ValueError(f"{path}: dims {dims} != expected {expected_dims}")

    summary = json.loads(summary_path.read_text())
    if summary.get("task") != expected_task:
        raise ValueError(
            f"{summary_path}: task {summary.get('task')!r} != {expected_task!r}"
        )
    records = summary.get(records_key)
    if not isinstance(records, list):
        raise ValueError(f"{summary_path}: missing {records_key} list")
    by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or "idx" not in record:
            raise ValueError(f"{summary_path}: malformed episode record")
        idx = int(record["idx"])
        if idx in by_id:
            raise ValueError(f"{summary_path}: duplicate episode record {idx}")
        by_id[idx] = record

    out: list[Episode] = []
    continuity_max = 0.0
    for eid in sorted(set(int(x) for x in episode.tolist())):
        if eid not in by_id:
            raise ValueError(f"{path}: episode {eid} has no summary record")
        idx = torch.nonzero(episode == eid).flatten()
        idx = idx[torch.argsort(times[idx], stable=True)]
        if len(idx) < 2:
            raise ValueError(f"{path}: episode {eid} has fewer than two transitions")
        te = times[idx]
        if not bool(torch.all(te[1:] > te[:-1])):
            raise ValueError(f"{path}: episode {eid} has duplicate/non-monotone t")
        if not bool(torch.all(te[1:] - te[:-1] == c)):
            raise ValueError(f"{path}: episode {eid} is not contiguous at commit c={c}")
        ze, zne = z[idx], z_next[idx]
        if len(idx) > 1:
            mismatch = float((zne[:-1] - ze[1:]).abs().max())
            continuity_max = max(continuity_max, mismatch)
            if mismatch > 1e-4:
                raise ValueError(
                    f"{path}: episode {eid} z_next/z continuity mismatch {mismatch:.3g}"
                )
        rec = by_id[eid]
        if "seed" not in rec:
            raise ValueError(f"{summary_path}: episode {eid} lacks environment seed")
        try:
            env_seed = int(rec["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{summary_path}: episode {eid} has invalid environment seed "
                f"{rec['seed']!r}"
            ) from exc
        if require_outcomes:
            if "success" not in rec:
                raise ValueError(
                    f"{summary_path}: episode {eid} lacks 'success', which the "
                    "time-to-success potential target needs; a missing field would "
                    "otherwise be read as a failure and make the target silently "
                    "all-zero.  Pass --phi-labels to fit the potential without it"
                )
            success: bool | None = bool(rec.get("success", False))
            success_step_raw = rec.get("success_step")
            success_step = int(success_step_raw) if success_step_raw is not None else None
            if success and success_step is None:
                raise ValueError(
                    f"{summary_path}: successful episode {eid} lacks success_step"
                )
        else:
            # The supplied-Phi' path never reads an outcome, so the sidecar is only
            # asked for idx and seed and these stay null in the manifest.
            success, success_step = None, None
        phi_values: torch.Tensor | None = None
        if phi_table is not None:
            phi_values = torch.empty(len(idx))
            for i, step in enumerate(te.tolist()):
                lookup = (eid, int(step))
                if lookup not in phi_table:
                    raise ValueError(
                        f"{path}: tape row (episode={eid}, t={step}) has no Phi' label; "
                        f"the --phi-labels artifact for {path.parent} does not cover it"
                    )
                phi_values[i] = phi_table[lookup]
        key = f"{tape_sha}:{eid}"
        out.append(
            Episode(
                key=key,
                role=role,
                tape_path=str(path),
                tape_sha256=tape_sha,
                episode_id=eid,
                env_seed=env_seed,
                z=ze,
                u=u[idx],
                z_next=zne,
                t=te,
                success=success,
                success_step=success_step,
                phi=phi_values,
            )
        )
    extra_records = sorted(set(by_id) - {ep.episode_id for ep in out})
    if extra_records:
        raise ValueError(f"{summary_path}: records without tape transitions: {extra_records}")
    return (
        out,
        TapeRecord(
            role=role,
            path=str(path),
            sha256=tape_sha,
            summary_path=str(summary_path),
            summary_sha256=summary_sha,
            episodes=len(out),
            transitions=n,
            records_key=records_key,
            outcome_fields_read=require_outcomes,
        ),
        dims,
        continuity_max,
    )


def load_role(
    paths: Sequence[Path],
    role: str,
    task: str,
    expected_dims: tuple[int, int, int] | None,
    phi_labels: dict[str, dict[tuple[int, int], float]] | None = None,
    require_outcomes: bool = True,
) -> tuple[list[Episode], list[TapeRecord], tuple[int, int, int], float]:
    all_eps: list[Episode] = []
    records: list[TapeRecord] = []
    dims = expected_dims
    continuity = 0.0
    seen_sha: set[str] = set()
    for path in paths:
        table = (
            None if phi_labels is None else phi_labels.get(str(path.resolve().parent))
        )
        eps, rec, got_dims, mismatch = load_tape(
            path, role, task, dims, table, require_outcomes
        )
        dims = got_dims
        if rec.sha256 in seen_sha:
            raise ValueError(f"duplicate {role} tape content: {rec.path}")
        seen_sha.add(rec.sha256)
        all_eps.extend(eps)
        records.append(rec)
        continuity = max(continuity, mismatch)
    if not all_eps or dims is None:
        raise ValueError(f"no {role} episodes loaded")
    return all_eps, records, dims, continuity


def select_head_records(
    head_paths: Sequence[Path] | None, base_records: Sequence[TapeRecord]
) -> list[TapeRecord]:
    """Resolve head tapes by content hash, requiring a strict base-content subset."""
    if head_paths is None:
        return list(base_records)
    base_by_sha = {record.sha256: record for record in base_records}
    selected: list[TapeRecord] = []
    seen: set[str] = set()
    for path in head_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest = sha256_file(resolved)
        if digest not in base_by_sha:
            raise ValueError(
                f"--head-tapes content is not a --base-tapes subset: {resolved} "
                f"(sha256={digest})"
            )
        if digest in seen:
            raise ValueError(f"duplicate --head-tapes content: {resolved}")
        seen.add(digest)
        selected.append(base_by_sha[digest])
    if not selected:
        raise ValueError("--head-tapes must select at least one base tape")
    return selected


def split_role(
    episodes: Sequence[Episode], holdout_fraction: float, seed: int
) -> tuple[list[Episode], list[Episode]]:
    """Split whole same-role environment-seed groups, never individual tapes."""
    if not episodes:
        raise ValueError("cannot split an empty role")
    roles = {ep.role for ep in episodes}
    if len(roles) != 1:
        raise ValueError(f"split_role requires one role, got {sorted(roles)}")
    role = next(iter(roles))
    env_seeds = {ep.env_seed for ep in episodes}
    if len(env_seeds) < 2:
        raise ValueError(f"role {role!r} needs at least two unique environment seeds")
    ranked = sorted(
        ((stable_fraction(seed, role, env_seed), env_seed) for env_seed in env_seeds),
        key=lambda item: (item[0], item[1]),
    )
    holdout_seeds = {
        env_seed for score, env_seed in ranked if score < holdout_fraction
    }
    if not holdout_seeds:
        holdout_seeds.add(ranked[0][1])
    if len(holdout_seeds) == len(ranked):
        holdout_seeds.remove(ranked[-1][1])
    train = [ep for ep in episodes if ep.env_seed not in holdout_seeds]
    holdout = [ep for ep in episodes if ep.env_seed in holdout_seeds]
    train.sort(key=lambda ep: ep.key)
    holdout.sort(key=lambda ep: ep.key)
    train_seeds = {ep.env_seed for ep in train}
    heldout_seeds = {ep.env_seed for ep in holdout}
    if not train or not holdout:
        raise AssertionError("environment-seed split produced an empty side")
    if train_seeds & heldout_seeds:
        raise AssertionError("environment seed appears in both train and holdout")
    if train_seeds | heldout_seeds != env_seeds:
        raise AssertionError("environment-seed split lost an input group")
    if len(holdout) < 2:
        raise ValueError("each role needs at least two holdout episodes for action shuffle")
    return train, holdout


def base_statistics(episodes: Sequence[Episode]) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.cat([ep.z for ep in episodes])
    mu = z.mean(0)
    sd = z.std(0, unbiased=False).clamp_min(1e-6)
    if not torch.isfinite(mu).all() or not torch.isfinite(sd).all():
        raise ValueError("non-finite base normalization statistics")
    return mu, sd


def _normalise(value: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (value - mu) / sd


def fill_window(
    ep: Episode,
    start: int,
    sequence_len: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
    z: torch.Tensor,
    u: torch.Tensor,
    zn: torch.Tensor,
    mask: torch.Tensor,
    row: int,
) -> None:
    n = min(sequence_len, ep.n - start)
    z[row, :n] = _normalise(ep.z[start : start + n], mu, sd)
    u[row, :n] = ep.u[start : start + n]
    zn[row, :n] = _normalise(ep.z_next[start : start + n], mu, sd)
    mask[row, :n] = True


def sample_batch(
    base: Sequence[Episode],
    update: Sequence[Episode] | None,
    batch_size: int,
    sequence_len: int,
    dims: tuple[int, int, int],
    mu: torch.Tensor,
    sd: torch.Tensor,
    generator: torch.Generator,
    shuffle_generator: torch.Generator,
    shuffle_update: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zdim, c, adim = dims
    z = torch.zeros(batch_size, sequence_len, zdim)
    u = torch.zeros(batch_size, sequence_len, c, adim)
    zn = torch.zeros_like(z)
    mask = torch.zeros(batch_size, sequence_len, dtype=torch.bool)
    n_update = batch_size // 2 if update is not None else 0
    n_base = batch_size - n_update

    def choose(pool: Sequence[Episode]) -> tuple[int, Episode, int]:
        ei = int(torch.randint(0, len(pool), (), generator=generator))
        ep = pool[ei]
        start = int(torch.randint(0, ep.n, (), generator=generator))
        return ei, ep, start

    for row in range(n_base):
        _ei, ep, start = choose(base)
        fill_window(ep, start, sequence_len, mu, sd, z, u, zn, mask, row)

    update_choices: list[tuple[int, Episode, int]] = []
    if update is not None:
        if shuffle_update and len(update) < 2:
            raise ValueError("shuffled-update training needs at least two update episodes")
        for row in range(n_base, batch_size):
            choice = choose(update)
            update_choices.append(choice)
            _ei, ep, start = choice
            fill_window(ep, start, sequence_len, mu, sd, z, u, zn, mask, row)
        if shuffle_update:
            # Replace only update actions, and guarantee that every donor is a
            # different episode. Targets, states and masks remain those of the
            # original update episode.
            for offset, (ei, _ep, _start) in enumerate(update_choices):
                jump = int(
                    torch.randint(1, len(update), (), generator=shuffle_generator)
                )
                donor = update[(ei + jump) % len(update)]
                donor_start = int(
                    torch.randint(0, donor.n, (), generator=shuffle_generator)
                )
                row = n_base + offset
                n = int(mask[row].sum())
                donor_idx = (torch.arange(n) + donor_start) % donor.n
                u[row, :n] = donor.u[donor_idx]

    return z.to(device), u.to(device), zn.to(device), mask.to(device)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def world_loss(
    model: RawLatentWM,
    z: torch.Tensor,
    u: torch.Tensor,
    z_next: torch.Tensor,
    mask: torch.Tensor,
    rollout_horizon: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    b, _ = model.roll(z, u)
    pred1 = model.predict(b, z, u)
    l1 = masked_mean(((pred1 - z_next) ** 2).mean(-1), mask)

    if z.shape[1] > 1:
        inv = model.inv(torch.cat([b[:, :-1], b[:, 1:]], -1))
        inv_target = u[:, :-1].flatten(2)
        inv_mask = mask[:, :-1] & mask[:, 1:]
        linv = masked_mean(((inv - inv_target) ** 2).mean(-1), inv_mask)
    else:
        linv = l1.new_zeros(())

    horizon = min(rollout_horizon, z.shape[1])
    if horizon <= 1:
        lroll = l1.new_zeros(())
    else:
        starts = z.shape[1] - horizon + 1
        bh = b[:, :starts]
        zh = z[:, :starts]
        valid = torch.ones_like(mask[:, :starts])
        losses: list[torch.Tensor] = []
        for h in range(horizon):
            uh = u[:, h : h + starts]
            target = z_next[:, h : h + starts]
            valid = valid & mask[:, h : h + starts]
            zh = model.predict(bh, zh, uh)
            losses.append(masked_mean(((zh - target) ** 2).mean(-1), valid))
            if h + 1 < horizon:
                shape = bh.shape
                bh = model.step(
                    bh.flatten(0, 1),
                    model.enc(zh).flatten(0, 1),
                    uh.flatten(0, 1),
                ).view(shape)
        lroll = torch.stack(losses).mean()
    loss = l1 + lroll + 0.1 * linv
    return loss, {
        "one_step": float(l1.detach()),
        "rollout": float(lroll.detach()),
        "inverse": float(linv.detach()),
        "total": float(loss.detach()),
    }


def train_world(
    model: RawLatentWM,
    base: Sequence[Episode],
    update: Sequence[Episode] | None,
    steps: int,
    batch_size: int,
    sequence_len: int,
    dims: tuple[int, int, int],
    mu: torch.Tensor,
    sd: torch.Tensor,
    seed: int,
    shuffle_update: bool,
    lr: float,
    device: torch.device,
    label: str,
) -> list[dict[str, float | int]]:
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sample_gen = torch.Generator().manual_seed(seed)
    shuffle_gen = torch.Generator().manual_seed(seed + 100_000)
    logs: list[dict[str, float | int]] = []
    for step in range(steps):
        batch = sample_batch(
            base,
            update,
            batch_size,
            sequence_len,
            dims,
            mu,
            sd,
            sample_gen,
            shuffle_gen,
            shuffle_update,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss, parts = world_loss(model, *batch, rollout_horizon=3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(steps // 4, 1) == 0 or step + 1 == steps:
            row: dict[str, float | int] = {"step": step + 1, **parts}
            logs.append(row)
            print(
                f"  {label:18s} {step + 1:>6}/{steps}: "
                f"one={parts['one_step']:.5f} roll={parts['rollout']:.5f} "
                f"inv={parts['inverse']:.5f}",
                flush=True,
            )
    model.eval().cpu()
    return logs


def make_prior(zdim: int, c: int, adim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(zdim),
        nn.Linear(zdim, 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, c * adim),
    )


def make_phi(zdim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(zdim), nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 1)
    )


def phi_target(ep: Episode, gamma: float, c: int) -> torch.Tensor:
    """Discounted time-to-success, the default target.

    Identically zero on an episode that never succeeds, which makes it degenerate
    wherever terminal success is rare; ``--phi-labels`` exists for that case.
    """
    if ep.success is None:
        raise ValueError(
            f"episode {ep.key}: the success-based potential target needs the outcome "
            "fields, which were not read; pass --phi-labels or a sidecar with them"
        )
    target = torch.zeros(ep.n)
    if not ep.success or ep.success_step is None:
        return target
    remaining = (ep.success_step - ep.t).clamp_min(0).float()
    chunks = torch.ceil(remaining / c)
    live = ep.t <= ep.success_step
    target[live] = gamma ** chunks[live]
    return target


def episode_phi(ep: Episode, gamma: float, c: int) -> torch.Tensor:
    """The supplied Phi' when --phi-labels covered this tape, else time-to-success."""
    if ep.phi is None:
        return phi_target(ep, gamma, c)
    if len(ep.phi) != ep.n:
        raise ValueError(f"episode {ep.key}: Phi' has {len(ep.phi)} rows, tape has {ep.n}")
    return ep.phi


def base_rows(
    episodes: Sequence[Episode], mu: torch.Tensor, sd: torch.Tensor, gamma: float, c: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supplied = {ep.phi is not None for ep in episodes}
    if len(supplied) > 1:
        raise ValueError(
            "head episodes mix supplied Phi' labels with unlabelled episodes; the "
            "potential would then be fitted against two different quantities"
        )
    z = torch.cat([_normalise(ep.z, mu, sd) for ep in episodes])
    u = torch.cat([ep.u for ep in episodes])
    phi = torch.cat([episode_phi(ep, gamma, c) for ep in episodes])
    return z, u, phi


def fit_shared_head(
    net: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    batch_size: int,
    seed: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    output_shape: tuple[int, ...] | None,
    label: str,
) -> list[dict[str, float | int]]:
    net.to(device).train()
    x, y = x.to(device), y.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    logs: list[dict[str, float | int]] = []
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch_size,), generator=generator).to(device)
        pred = net(x[idx])
        if output_shape is not None:
            pred = pred.view(-1, *output_shape)
        else:
            pred = pred.squeeze(-1)
        loss = ((pred - y[idx]) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step + 1 == steps:
            logs.append({"step": step + 1, "mse": float(loss.detach())})
            print(
                f"  {label:18s} {step + 1:>6}/{steps}: "
                f"mse={float(loss.detach()):.5f}"
            )
    net.eval().cpu()
    return logs


def head_metrics(
    prior: nn.Module,
    phi: nn.Module,
    train: Sequence[Episode],
    holdout: Sequence[Episode],
    mu: torch.Tensor,
    sd: torch.Tensor,
    gamma: float,
    c: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with torch.no_grad():
        for name, episodes in (("train", train), ("holdout", holdout)):
            z, u, target = base_rows(episodes, mu, sd, gamma, c)
            pp = prior(z).view_as(u)
            ph = phi(z).squeeze(-1)
            result[name] = {
                "transitions": len(z),
                "prior_mse": float(((pp - u) ** 2).mean()),
                "phi_mse": float(((ph - target) ** 2).mean()),
                "phi_target_mean": float(target.mean()),
                "phi_target_std": float(target.std(unbiased=False)),
                # The bar criterion (d) names: an MSE is compared against a
                # variance, not against a standard deviation.
                "phi_target_var": float(target.var(unbiased=False)),
                "phi_target_min": float(target.min()),
                "phi_target_max": float(target.max()),
                "phi_target_positive_fraction": float((target > 0).float().mean()),
            }
    return result


def rollout_mse_for_episode(
    model: RawLatentWM,
    ep: Episode,
    donor: Episode,
    horizon: int,
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> tuple[float, float, float, int]:
    z = _normalise(ep.z, mu, sd)
    zn = _normalise(ep.z_next, mu, sd)
    u = ep.u
    donor_u = donor.u
    starts = ep.n - horizon + 1
    if starts <= 0:
        return 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        beliefs, _ = model.roll(z.unsqueeze(0), u.unsqueeze(0))
        b0 = beliefs[0, :starts]
        z0 = z[:starts]
        # Window j starts at z[j], consumes u[j:j+H], and must land on the
        # explicit target z_next[j+H-1].  Keeping this expression in z_next
        # coordinates avoids the old shifted-z off-by-one/terminal loss.
        target = zn[horizon - 1 : horizon - 1 + starts]
        donor_starts = torch.arange(starts) % (donor.n - horizon + 1)
        pred, b = z0.clone(), b0.clone()
        pred_shuffle, bs = z0.clone(), b0.clone()
        for h in range(horizon):
            uh = u[h : h + starts]
            alt_idx = donor_starts + h
            ush = donor_u[alt_idx]
            pred = model.predict(b, pred, uh)
            pred_shuffle = model.predict(bs, pred_shuffle, ush)
            if h + 1 < horizon:
                b = model.step(b, model.enc(pred), uh)
                bs = model.step(bs, model.enc(pred_shuffle), ush)
        learned_sse = float(((pred - target) ** 2).sum())
        shuffled_sse = float(((pred_shuffle - target) ** 2).sum())
        identity_sse = float(((z0 - target) ** 2).sum())
    return learned_sse, shuffled_sse, identity_sse, int(target.numel())


def evaluate_model(
    model: RawLatentWM,
    episodes: Sequence[Episode],
    horizons: Sequence[int],
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> dict[str, Any]:
    if len(episodes) < 2:
        raise ValueError("evaluation requires at least two episodes")
    model.eval().cpu()
    ordered = sorted(episodes, key=lambda ep: ep.key)
    result: dict[str, Any] = {}
    for horizon in horizons:
        learned = shuffled = identity = 0.0
        count = windows = 0
        eligible = [ep for ep in ordered if ep.n >= horizon]
        if len(eligible) < 2:
            result[str(horizon)] = {
                "windows": 0,
                "mse": None,
                "identity_mse": None,
                "x_identity": None,
                "action_shuffle_mse": None,
                "action_shuffle_over_learned": None,
            }
            continue
        for i, ep in enumerate(eligible):
            donor = eligible[(i + 1) % len(eligible)]
            ls, ss, ids, n = rollout_mse_for_episode(
                model, ep, donor, horizon, mu, sd
            )
            learned += ls
            shuffled += ss
            identity += ids
            count += n
            windows += max(ep.n - horizon + 1, 0)
        if count == 0:
            result[str(horizon)] = {
                "windows": 0,
                "mse": None,
                "identity_mse": None,
                "x_identity": None,
                "action_shuffle_mse": None,
                "action_shuffle_over_learned": None,
            }
            continue
        mse = learned / count
        idm = identity / count
        shm = shuffled / count
        result[str(horizon)] = {
            "windows": windows,
            "elements": count,
            "mse": mse,
            "identity_mse": idm,
            "x_identity": mse / idm if idm > 0 else None,
            "action_shuffle_mse": shm,
            "action_shuffle_over_learned": shm / mse if mse > 0 else None,
        }
    return result


def comparison_metrics(
    evaluations: dict[str, dict[str, Any]], horizons: Sequence[int]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "updated_vs_stale": {},
        "base_continue_vs_stale": {},
        "updated_shuffled_vs_updated": {},
        "old_retention": {},
    }
    pairs = {
        "updated_vs_stale": ("updated", "stale"),
        "base_continue_vs_stale": ("base_continue", "stale"),
        "updated_shuffled_vs_updated": ("updated_shuffled", "updated"),
    }
    for label, (left, right) in pairs.items():
        for split in ("base_holdout", "update_holdout"):
            comparisons[label][split] = {}
            for horizon in horizons:
                lm = evaluations[left][split][str(horizon)]["mse"]
                rm = evaluations[right][split][str(horizon)]["mse"]
                comparisons[label][split][str(horizon)] = {
                    "mse_delta": None if lm is None or rm is None else lm - rm,
                    "mse_ratio": (
                        None if lm is None or rm in (None, 0) else lm / rm
                    ),
                }
    for model_name in ("base_continue", "updated", "updated_shuffled"):
        comparisons["old_retention"][model_name] = {}
        for horizon in horizons:
            mm = evaluations[model_name]["base_holdout"][str(horizon)]["mse"]
            sm = evaluations["stale"]["base_holdout"][str(horizon)]["mse"]
            comparisons["old_retention"][model_name][str(horizon)] = {
                "base_holdout_mse_delta_vs_stale": (
                    None if mm is None or sm is None else mm - sm
                ),
                "base_holdout_mse_ratio_vs_stale": (
                    None if mm is None or sm in (None, 0) else mm / sm
                ),
            }
    return comparisons


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def verify_checkpoint(path: Path, dims: tuple[int, int, int], task: str) -> None:
    """Reload every scorer-facing component before declaring the build complete."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    expected_top = {
        "models",
        "prior",
        "phi",
        "dims",
        "mu",
        "sd",
        "task",
        "sequence_len",
        "metrics",
        "provenance",
    }
    if set(checkpoint) != expected_top:
        raise RuntimeError(
            f"checkpoint schema mismatch: {sorted(checkpoint)} != {sorted(expected_top)}"
        )
    if tuple(checkpoint["dims"]) != dims or checkpoint["task"] != task:
        raise RuntimeError("checkpoint task/dims changed during serialization")
    if set(checkpoint["models"]) != set(MODEL_NAMES):
        raise RuntimeError("checkpoint model-arm schema mismatch")
    zdim, c, adim = dims
    for name in MODEL_NAMES:
        probe = RawLatentWM(zdim, c, adim)
        probe.load_state_dict(checkpoint["models"][name], strict=True)
    prior = make_prior(zdim, c, adim)
    prior.load_state_dict(checkpoint["prior"], strict=True)
    phi = make_phi(zdim)
    phi.load_state_dict(checkpoint["phi"], strict=True)
    if tuple(checkpoint["mu"].shape) != (zdim,) or tuple(
        checkpoint["sd"].shape
    ) != (zdim,):
        raise RuntimeError("checkpoint normalizer shape mismatch")


def episode_manifest(episodes: Iterable[Episode]) -> list[dict[str, Any]]:
    return [
        {
            "key": ep.key,
            "role": ep.role,
            "tape_sha256": ep.tape_sha256,
            "episode_id": ep.episode_id,
            "env_seed": ep.env_seed,
            "transitions": ep.n,
            "success": ep.success,
            "success_step": ep.success_step,
        }
        for ep in sorted(episodes, key=lambda item: item.key)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tapes", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--head-tapes",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "content-hash subset of --base-tapes used for the shared prior/Phi; "
            "defaults to all base tapes"
        ),
    )
    parser.add_argument("--update-tapes", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--phi-labels",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "labels.pt artifacts (or the directories holding them) supplying the "
            "shared potential's target as Phi' from lcwm/v250_progress.py, joined to "
            "the tape rows on an exact (episode, t) key; every tape feeding the heads "
            "must be covered.  Omitted: the discounted time-to-success target, which "
            "needs the sidecar outcome fields"
        ),
    )
    parser.add_argument("--task", default="chain1b_lr2")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--commit", type=int, default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=241)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 10])
    parser.add_argument("--base-steps", type=int, default=4000)
    parser.add_argument("--update-steps", type=int, default=3000)
    parser.add_argument("--prior-steps", type=int, default=3000)
    parser.add_argument("--phi-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--wm-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=241)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must lie strictly between zero and one")
    for name in ("sequence_len", "base_steps", "update_steps", "prior_steps", "phi_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size < 4 or args.batch_size % 2:
        raise ValueError("--batch-size must be even and at least four")
    if args.head_batch_size <= 0:
        raise ValueError("--head-batch-size must be positive")
    if not args.horizons or min(args.horizons) <= 0:
        raise ValueError("--horizons must contain positive integers")
    if len(set(args.horizons)) != len(args.horizons):
        raise ValueError("--horizons must not contain duplicates")
    if args.gamma <= 0.0 or args.gamma > 1.0:
        raise ValueError("--gamma must lie in (0, 1]")


def main() -> int:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    phi_labels, phi_label_records = load_phi_labels(args.phi_labels, args.task)
    # With supplied Phi' the potential needs no outcome, so the loader is told not to
    # read one: "the builder does not use success" becomes a property of the run
    # rather than a claim about it.
    require_outcomes = phi_labels is None
    base_eps, base_records, dims, base_continuity = load_role(
        args.base_tapes, "base", args.task, None, phi_labels, require_outcomes
    )
    update_eps, update_records, update_dims, update_continuity = load_role(
        args.update_tapes, "update", args.task, dims, phi_labels, require_outcomes
    )
    assert update_dims == dims
    if args.latent_dim is not None and dims[0] != args.latent_dim:
        raise ValueError(f"loaded latent_dim {dims[0]} != --latent-dim {args.latent_dim}")
    if args.commit is not None and dims[1] != args.commit:
        raise ValueError(f"loaded c {dims[1]} != --commit {args.commit}")
    base_hashes = {record.sha256 for record in base_records}
    update_hashes = {record.sha256 for record in update_records}
    overlap = base_hashes & update_hashes
    if overlap:
        raise ValueError(f"base/update tape content overlaps: {sorted(overlap)}")
    tape_dirs_seen: dict[str, str] = {}
    for record in base_records + update_records:
        directory = str(Path(record.path).parent)
        if directory in tape_dirs_seen:
            raise ValueError(
                f"two tapes share the directory {directory}: {tape_dirs_seen[directory]} "
                f"and {record.path}.  The episode-record sidecar and any --phi-labels "
                "table are both resolved from a tape's parent directory, so the second "
                "tape would silently take the first one's environment seeds and Phi' "
                "rows on its own (episode, t) keys"
            )
        tape_dirs_seen[directory] = record.path
    head_records = select_head_records(args.head_tapes, base_records)
    head_hashes = {record.sha256 for record in head_records}
    if phi_labels is not None:
        unmatched = sorted(set(phi_labels) - set(tape_dirs_seen))
        if unmatched:
            raise ValueError(
                f"--phi-labels entries match no loaded tape directory: {unmatched}"
            )

    base_train, base_holdout = split_role(
        base_eps, args.holdout_fraction, args.split_seed
    )
    update_train, update_holdout = split_role(
        update_eps, args.holdout_fraction, args.split_seed
    )
    head_train = [ep for ep in base_train if ep.tape_sha256 in head_hashes]
    head_holdout = [ep for ep in base_holdout if ep.tape_sha256 in head_hashes]
    if not head_train:
        raise ValueError("selected head tapes have no base-train episodes")
    if not head_holdout:
        raise ValueError("selected head tapes have no base-holdout episodes")
    if phi_labels is not None:
        uncovered = sorted(
            {ep.tape_path for ep in head_train + head_holdout if ep.phi is None}
        )
        if uncovered:
            raise ValueError(f"--phi-labels does not cover head tapes: {uncovered}")
    # The split hashes role into its key, so base and update are split
    # independently: an environment seed present in both roles can be base-train
    # and update-holdout at once, which puts that layout on both sides of an
    # updated-vs-stale evaluation.  Report it rather than leave it in provenance
    # only; chain3 uses disjoint seed families (D0 8700+, D1 8800+) and should
    # print nothing.
    cross_role_overlap = sorted(
        {ep.env_seed for ep in base_eps} & {ep.env_seed for ep in update_eps}
    )
    cross_role_train_holdout = sorted(
        ({ep.env_seed for ep in base_train} & {ep.env_seed for ep in update_holdout})
        | ({ep.env_seed for ep in update_train} & {ep.env_seed for ep in base_holdout})
    )
    if cross_role_overlap:
        print(
            f"NOTE {len(cross_role_overlap)} environment seed(s) appear in both roles; "
            f"{len(cross_role_train_holdout)} of them are train in one role and holdout "
            f"in the other: {cross_role_train_holdout}",
            flush=True,
        )
    mu, sd = base_statistics(base_train)
    zdim, c, adim = dims
    print(
        f"task={args.task} dims={dims}; base {len(base_train)}/{len(base_holdout)} "
        f"train/holdout; update {len(update_train)}/{len(update_holdout)}; "
        f"heads {len(head_train)}/{len(head_holdout)}; device={device}",
        flush=True,
    )
    print(
        "phi target: "
        + (
            f"supplied Phi' from {len(phi_label_records)} label artifact(s), "
            f"outcome fields read={require_outcomes}"
            if phi_labels is not None
            else f"discounted time-to-success, gamma={args.gamma}"
        ),
        flush=True,
    )

    # Shared policy prior and potential: selected base train only, evaluated on
    # the matching selected base holdout, all in the fixed full-base frame.
    bx, bu, bphi = base_rows(head_train, mu, sd, args.gamma, c)
    torch.manual_seed(args.seed + 1)
    prior = make_prior(zdim, c, adim)
    prior_logs = fit_shared_head(
        prior,
        bx,
        bu,
        args.prior_steps,
        args.head_batch_size,
        args.seed + 11,
        args.head_lr,
        1e-4,
        device,
        (c, adim),
        "shared prior",
    )
    torch.manual_seed(args.seed + 2)
    phi = make_phi(zdim)
    phi_logs = fit_shared_head(
        phi,
        bx,
        bphi,
        args.phi_steps,
        args.head_batch_size,
        args.seed + 12,
        args.head_lr,
        1e-2,
        device,
        None,
        "shared phi",
    )
    shared_metrics = head_metrics(
        prior, phi, head_train, head_holdout, mu, sd, args.gamma, c
    )

    # M0, then three same-initialisation continuations.
    torch.manual_seed(args.seed)
    stale = RawLatentWM(zdim, c, adim)
    stale_logs = train_world(
        stale,
        base_train,
        None,
        args.base_steps,
        args.batch_size,
        args.sequence_len,
        dims,
        mu,
        sd,
        args.seed + 20,
        False,
        args.wm_lr,
        device,
        "stale/base",
    )
    base_continue = copy.deepcopy(stale)
    updated = copy.deepcopy(stale)
    updated_shuffled = copy.deepcopy(stale)
    base_continue_logs = train_world(
        base_continue,
        base_train,
        None,
        args.update_steps,
        args.batch_size,
        args.sequence_len,
        dims,
        mu,
        sd,
        args.seed + 30,
        False,
        args.wm_lr,
        device,
        "base_continue",
    )
    updated_logs = train_world(
        updated,
        base_train,
        update_train,
        args.update_steps,
        args.batch_size,
        args.sequence_len,
        dims,
        mu,
        sd,
        args.seed + 40,
        False,
        args.wm_lr,
        device,
        "updated",
    )
    updated_shuffled_logs = train_world(
        updated_shuffled,
        base_train,
        update_train,
        args.update_steps,
        args.batch_size,
        args.sequence_len,
        dims,
        mu,
        sd,
        args.seed + 40,
        True,
        args.wm_lr,
        device,
        "updated_shuffled",
    )
    models = {
        "stale": stale,
        "base_continue": base_continue,
        "updated": updated,
        "updated_shuffled": updated_shuffled,
    }

    horizons = sorted(args.horizons)
    evaluations: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        print(f"evaluating {name}", flush=True)
        evaluations[name] = {
            "base_holdout": evaluate_model(
                models[name], base_holdout, horizons, mu, sd
            ),
            "update_holdout": evaluate_model(
                models[name], update_holdout, horizons, mu, sd
            ),
        }
    metrics: dict[str, Any] = {
        "shared_base_only": shared_metrics,
        "models": evaluations,
        "comparisons": comparison_metrics(evaluations, horizons),
        "training_logs": {
            "prior": prior_logs,
            "phi": phi_logs,
            "stale": stale_logs,
            "base_continue": base_continue_logs,
            "updated": updated_logs,
            "updated_shuffled": updated_shuffled_logs,
        },
    }

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    tag = args.tag or args.task
    out = args.out_root.resolve() / f"{tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    tape_records = base_records + update_records
    provenance: dict[str, Any] = {
        "utc": stamp,
        "git": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--short")),
        "argv": sys.argv,
        "cli": shlex.join(sys.argv),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "source": {
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "raw_latent_wm": str(REPO / "scripts" / "train_v220_raw_latent_wm.py"),
            "raw_latent_wm_sha256": sha256_file(
                REPO / "scripts" / "train_v220_raw_latent_wm.py"
            ),
        },
        "inputs": [record.__dict__ for record in tape_records],
        "head_inputs": [record.__dict__ for record in head_records],
        "phi_target": {
            "source": (
                "lcwm.v250_progress.tape_progress Phi'"
                if phi_labels is not None
                else "discounted time-to-success gamma**ceil((success_step - t)/c)"
            ),
            "join": (
                "exact (episode, t); every tape row required to match exactly one "
                "label row, extra label rows (the terminal boundary) permitted"
            ),
            "labels": [record.__dict__ for record in phi_label_records],
            "approach_reference_m": (
                APPROACH_REFERENCE if phi_labels is not None else None
            ),
            "gamma": args.gamma if phi_labels is None else None,
            "outcome_fields_read": require_outcomes,
            "identical_across_arms": True,
        },
        "head_input_selection": {
            "defaulted_to_all_base": args.head_tapes is None,
            "requested_paths": (
                []
                if args.head_tapes is None
                else [str(path.resolve()) for path in args.head_tapes]
            ),
            "match": "sha256 content subset of base inputs",
        },
        "split": {
            "method": "sha256(v241:split_seed:role:env_seed)",
            "seed": args.split_seed,
            "holdout_fraction": args.holdout_fraction,
            "unit": "all episodes sharing role + env_seed",
            "train_holdout_env_seed_disjoint_asserted": True,
            "cross_role_env_seed_overlap": cross_role_overlap,
            "cross_role_train_holdout_env_seeds": cross_role_train_holdout,
            "one_tape_per_directory_asserted": True,
            "base_train": episode_manifest(base_train),
            "base_holdout": episode_manifest(base_holdout),
            "update_train": episode_manifest(update_train),
            "update_holdout": episode_manifest(update_holdout),
        },
        "contract": {
            "task": args.task,
            "dims": dims,
            "sequence_len": args.sequence_len,
            "horizons": horizons,
            "normalization": "base-train current z only; fixed for all arms",
            "prior_data": "selected head-tape episodes in base-train only",
            "phi_data": "selected head-tape episodes in base-train only",
            "phi_target": (
                "supplied Phi' labels" if phi_labels is not None else "time-to-success"
            ),
            "head_metrics": "selected head-tape episodes in base-holdout only",
            "head_tapes_default": "all base tapes when --head-tapes is omitted",
            "split_grouping": "role + summary episode-record env_seed",
            "targets": "explicit tape z_next; no shifted-z surrogate",
            "update_sampling": "balanced 50/50 base/update episodes",
            "shuffled_control": "update actions replaced cross-episode only",
            "deployed_rates_read": False,
            "max_z_next_continuity_error": {
                "base": base_continuity,
                "update": update_continuity,
            },
        },
        "hyperparameters": {
            "base_steps": args.base_steps,
            "update_steps": args.update_steps,
            "prior_steps": args.prior_steps,
            "phi_steps": args.phi_steps,
            "batch_size": args.batch_size,
            "head_batch_size": args.head_batch_size,
            "wm_lr": args.wm_lr,
            "head_lr": args.head_lr,
            "gamma": args.gamma,
            "seed": args.seed,
            "threads": args.threads,
        },
    }
    checkpoint = {
        "models": {name: cpu_state_dict(models[name]) for name in MODEL_NAMES},
        "prior": cpu_state_dict(prior),
        "phi": cpu_state_dict(phi),
        "dims": dims,
        "mu": mu.cpu(),
        "sd": sd.cpu(),
        "task": args.task,
        "sequence_len": args.sequence_len,
        "metrics": metrics,
        "provenance": provenance,
    }
    checkpoint_path = out / "model_pair.pt"
    torch.save(checkpoint, checkpoint_path)
    verify_checkpoint(checkpoint_path, dims, args.task)
    checkpoint_sha = sha256_file(checkpoint_path)
    summary = {
        "task": args.task,
        "dims": dims,
        "sequence_len": args.sequence_len,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "metrics": metrics,
        "provenance": provenance,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f"checkpoint sha256 {checkpoint_sha}\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
