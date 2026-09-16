#!/usr/bin/env python3
"""Test whether a censored v121 terminal action chunk is reproducible.

This is a development-only probe restricted to the already disclosed chain3
seed-8700 panel.  The original v121 collector did not save the action chunk
that terminated an episode, because a transition was committed only after its
next latent was available.  We reconstruct the original policy RNG stream and
require every *stored* normalized chunk to match bit-for-bit before proposing
the missing terminal chunk.  A terminal chunk is considered recoverable only
when the replay also terminates at the source-recorded step with the same
success bit.

No RGB is written.  This result cannot validate Gate 0; it only decides whether
terminal-frame recovery is mechanically defensible before a formal RGB replay
is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/v248_terminal_chunk_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/v248_terminal_chunk_mpl_cache")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402
ensure_project_libero_config()

from lcwm.chassis import DEFAULT_MODEL, Pi05Runner  # noqa: E402
from lcwm.sampler import sample_chunks  # noqa: E402
from lcwm.v080_bench import make_v080_env  # noqa: E402

SCHEMA = "v248_terminal_chunk_recovery_development_probe_v1"
COMPLETE_SCHEMA = "v248_terminal_chunk_recovery_development_complete_v1"
TASK = "chain3_lr2"
C = 10
SEED_START = 8700
ALLOWED_EPISODES = frozenset({4, 45})
EXPECTED_TAPE_SHA256 = (
    "7dbff743ea89db5fd792d317111129d188d439122a46db7384f4549fd947f86c"
)
EXPECTED_SUMMARY_SHA256 = (
    "a4dc0d5f73328247d11592ad08d55031023a2bb00bbaa3c7436e0a4a9387ef58"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, canonical_bytes(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_inputs(
    tape_path: Path, summary_path: Path, episode_ids: list[int]
) -> tuple[
    dict[str, Any],
    dict[int, list[int]],
    dict[int, dict[str, Any]],
    dict[int, float],
]:
    require(sha256_file(tape_path) == EXPECTED_TAPE_SHA256, "development tape drift")
    require(
        sha256_file(summary_path) == EXPECTED_SUMMARY_SHA256,
        "development summary drift",
    )
    require(len(set(episode_ids)) == len(episode_ids), "duplicate episode ID")
    require(set(episode_ids) <= ALLOWED_EPISODES, "only disclosed successes are allowed")
    tape = torch.load(tape_path, map_location="cpu", weights_only=False)
    require(isinstance(tape, dict), "tape is not a dictionary")
    required = {
        "z",
        "u",
        "z_next",
        "episode",
        "t",
        "sigma",
        "task",
        "c",
        "latent_dim",
    }
    require(required <= set(tape), "tape fields missing")
    require(tape["task"] == TASK and int(tape["c"]) == C, "task/c mismatch")
    require(tape["u"].ndim == 3 and tuple(tape["u"].shape[1:]) == (C, 7), "u shape")
    require(bool(torch.isfinite(tape["u"]).all()), "u contains NaN/Inf")
    rows: dict[int, list[int]] = {}
    for episode_id in episode_ids:
        selected = torch.nonzero(tape["episode"] == episode_id, as_tuple=False).flatten()
        indices = [int(value) for value in selected]
        require(indices, f"episode {episode_id} is absent")
        expected_t = list(range(0, len(indices) * C, C))
        require(
            [int(tape["t"][index]) for index in indices] == expected_t,
            f"episode {episode_id} time grid drift",
        )
        rows[episode_id] = indices

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(isinstance(summary, dict), "summary is not an object")
    require(summary.get("task") == TASK and int(summary.get("c", -1)) == C, "summary task/c")
    records = {
        int(record["idx"]): record for record in summary.get("episode_records", [])
    }
    require(sorted(records) == list(range(int(summary["episodes"]))), "episode IDs drift")
    raw_sigma_choices = [float(value) for value in summary.get("sigmas", [])]
    require(
        raw_sigma_choices == [0.0, 0.6, 1.5, 3.0],
        "development sigma choices drift",
    )
    stored_sigma_choices = [float(np.float32(value)) for value in raw_sigma_choices]
    sigma_rng = np.random.default_rng(SEED_START)
    missing_sigmas: dict[int, float] = {}
    for episode_id in sorted(records):
        record = records[episode_id]
        proposed_chunks = (int(record["steps"]) + C - 1) // C
        draws = [
            raw_sigma_choices[int(sigma_rng.integers(len(raw_sigma_choices)))]
            for _ in range(proposed_chunks)
        ]
        tape_rows = torch.nonzero(
            tape["episode"] == episode_id, as_tuple=False
        ).flatten()
        require(
            len(tape_rows) in {proposed_chunks, proposed_chunks - 1},
            f"episode {episode_id} proposal/tape count mismatch",
        )
        observed_sigmas = [float(tape["sigma"][int(index)]) for index in tape_rows]
        stored_draws = [
            stored_sigma_choices[raw_sigma_choices.index(value)] for value in draws
        ]
        require(
            observed_sigmas == stored_draws[: len(tape_rows)],
            f"episode {episode_id} sigma RNG drift",
        )
        if len(tape_rows) == proposed_chunks - 1:
            missing_sigmas[episode_id] = draws[-1]
    selected_records: dict[int, dict[str, Any]] = {}
    for episode_id in episode_ids:
        require(episode_id in records, f"summary lacks episode {episode_id}")
        record = records[episode_id]
        require(int(record["seed"]) == SEED_START + episode_id, "seed rule mismatch")
        require(record.get("success") is True, "probe episode is not successful")
        require(record.get("success_step") == record.get("steps"), "success/steps mismatch")
        missing = int(record["steps"]) - len(rows[episode_id]) * C
        require(1 <= missing < C, "episode does not have one censored partial chunk")
        selected_records[episode_id] = record
    require(set(episode_ids) <= set(missing_sigmas), "selected terminal sigma is not missing")
    return tape, rows, selected_records, missing_sigmas


@torch.no_grad()
def normalized_chunk(
    runner: Pi05Runner,
    obs: Any,
    task_description: str,
    seed: int,
    t: int,
    sigma: float,
) -> torch.Tensor:
    if sigma == 0.0:
        raw = runner.sample_chunk(obs, task_description)
    else:
        policy_observation = runner._obs_to_policy_batch(obs, task_description)
        raw = sample_chunks(
            runner.policy,
            policy_observation,
            1,
            seed=seed * 7919 + t,
            sigma=sigma,
        )
    chunk = raw[0, :C].detach().float().cpu().contiguous()
    require(chunk.shape == (C, 7), f"sampled chunk shape {tuple(chunk.shape)}")
    require(bool(torch.isfinite(chunk).all()), "sampled chunk contains NaN/Inf")
    return chunk


def replay_episode(
    runner: Pi05Runner,
    env: Any,
    tape: dict[str, Any],
    row_indices: list[int],
    record: dict[str, Any],
    missing_sigma: float,
) -> dict[str, Any]:
    episode_id = int(record["idx"])
    seed = int(record["seed"])
    source_steps = int(record["steps"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    runner.reset()
    obs, _ = env.reset(seed=seed)
    t = 0
    stored_checks = []
    terminated_step = None
    success_step = None

    for local_index, row_index in enumerate(row_indices):
        require(int(tape["t"][row_index]) == t, "row/replay time mismatch")
        stored_sigma = float(tape["sigma"][row_index])
        matching = [
            raw
            for raw in (0.0, 0.6, 1.5, 3.0)
            if float(np.float32(raw)) == stored_sigma
        ]
        require(len(matching) == 1, "stored sigma has no unique source config value")
        sigma = matching[0]
        sampled = normalized_chunk(runner, obs, env.task_description, seed, t, sigma)
        stored = tape["u"][row_index].detach().float().cpu().contiguous()
        exact = torch.equal(sampled, stored)
        max_abs = float((sampled - stored).abs().max())
        stored_checks.append(
            {
                "local_chunk_index": local_index,
                "t": t,
                "sigma": sigma,
                "exact_float32": exact,
                "max_abs_error": max_abs,
            }
        )
        if not exact:
            break
        for action in runner.chunk_to_env(stored):
            obs, _reward, terminated, truncated, info = env.step(action)
            t += 1
            if bool(info.get("is_success", False)) and success_step is None:
                success_step = t
            require(not (terminated or truncated), "stored chunk terminated unexpectedly")

    all_stored_exact = len(stored_checks) == len(row_indices) and all(
        item["exact_float32"] for item in stored_checks
    )
    recovered = None
    if all_stored_exact:
        require(t == len(row_indices) * C, "stored replay length mismatch")
        missing_chunk = normalized_chunk(
            runner, obs, env.task_description, seed, t, missing_sigma
        )
        missing_chunk_sha = hashlib.sha256(missing_chunk.numpy().tobytes()).hexdigest()
        steps_executed = 0
        for action in runner.chunk_to_env(missing_chunk):
            obs, _reward, terminated, truncated, info = env.step(action)
            t += 1
            steps_executed += 1
            if bool(info.get("is_success", False)) and success_step is None:
                success_step = t
            if terminated or truncated:
                terminated_step = t
                break
        recovered = {
            "normalized_chunk_raw_bytes_sha256": missing_chunk_sha,
            "sigma": missing_sigma,
            "steps_executed_until_termination": steps_executed,
            "terminated_step": terminated_step,
            "success_step": success_step,
            "matches_source_terminal_step": terminated_step == source_steps,
            "matches_source_success_step": success_step == int(record["success_step"]),
        }

    return {
        "episode_id": episode_id,
        "env_seed": seed,
        "stored_chunks": len(row_indices),
        "stored_steps": len(row_indices) * C,
        "source_steps": source_steps,
        "censored_terminal_steps": source_steps - len(row_indices) * C,
        "stored_chunk_checks": stored_checks,
        "all_stored_chunks_exact_float32": all_stored_exact,
        "terminal_chunk": recovered,
        "recoverable": bool(
            recovered
            and recovered["matches_source_terminal_step"]
            and recovered["matches_source_success_step"]
        ),
    }


def main() -> int:
    args = parse_args()
    tape_path = args.tape.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(tape_path.is_file() and summary_path.is_file(), "input file missing")
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = list(args.episode_id)
    tape, rows, records, missing_sigmas = load_inputs(
        tape_path, summary_path, episode_ids
    )

    runner = Pi05Runner(model_id=DEFAULT_MODEL, suite_name="libero_10", n_action_steps=C)
    env = make_v080_env(TASK)
    try:
        results = [
            replay_episode(
                runner,
                env,
                tape,
                rows[episode_id],
                records[episode_id],
                missing_sigmas[episode_id],
            )
            for episode_id in episode_ids
        ]
    finally:
        env.close()

    source_path = Path(__file__).resolve()
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_probe_complete",
        "claim_scope": "disclosed seed-8700 mechanics only; not Gate-0 evidence",
        "task": TASK,
        "c": C,
        "model_id": DEFAULT_MODEL,
        "tape_sha256": sha256_file(tape_path),
        "summary_sha256": sha256_file(summary_path),
        "episode_ids": episode_ids,
        "episodes": results,
        "all_recoverable": all(item["recoverable"] for item in results),
        "producer_source_sha256": sha256_file(source_path),
    }

    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        write_json(stage / "RESULT.json", payload)
        shutil.copy2(source_path, stage / "PRODUCER.py")
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "complete",
            "result_sha256": sha256_file(stage / "RESULT.json"),
            "producer_sha256": sha256_file(stage / "PRODUCER.py"),
        }
        write_json(stage / "COMPLETE.json", complete)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(stage, output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "all_recoverable": payload["all_recoverable"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
