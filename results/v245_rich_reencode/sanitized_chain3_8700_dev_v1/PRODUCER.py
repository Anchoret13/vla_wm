#!/usr/bin/env python
"""Outcome-blind sanitizer for frozen v121 deployment tapes.

This is stage 1 of the canonical rich-latent replay.  It opens exactly one
explicitly named ``tape.pt`` and never discovers or opens a sibling summary.
Only the registered transition tensors and non-outcome layout metadata cross
the stage boundary.  In particular, ``success`` and every other declared
outcome field are discarded without inspecting their values.

The output directory is assembled under a private sibling name and published
with one atomic rename.  A failed validation therefore cannot leave a
``COMPLETE.json`` at the requested output path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO = Path(__file__).resolve().parent.parent
SCHEMA = "v245_sanitized_deploy_tape_v1"
RESULT_SCHEMA = "v245_sanitized_deploy_tape_result_v1"
COMPLETE_SCHEMA = "v245_sanitized_deploy_tape_complete_v1"

TENSOR_FIELDS = ("z", "u", "z_next", "episode", "t", "sigma")
METADATA_FIELDS = ("task", "c", "latent_dim", "proprio_dim", "proprio_keys")
KEPT_FIELDS = frozenset((*TENSOR_FIELDS, *METADATA_FIELDS))

# These keys are names only.  Their values are never indexed, converted, counted,
# hashed, serialized, or included in a diagnostic.
DECLARED_OUTCOME_FIELDS = frozenset(
    {
        "success",
        "outcome",
        "outcomes",
        "success_step",
        "events",
        "episode_records",
        "milestones",
        "stage_reached",
    }
)


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and the tensor's contiguous CPU bytes."""
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header + raw).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_bytes(value))


def _save_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_head() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    value = process.stdout.strip()
    return value if process.returncode == 0 and value else None


def _integer_tensor(value: Any, name: str, length: int) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or len(value) != length:
        raise ValueError(f"{name} must be a tensor [{length}]")
    integer = value.detach().cpu().to(dtype=torch.long)
    if not torch.equal(value.detach().cpu(), integer.to(dtype=value.dtype)):
        raise ValueError(f"{name} contains non-integer values")
    return integer


def validate_source(source: dict[str, Any]) -> dict[str, Any]:
    """Validate transition structure without evaluating an outcome value."""
    keys = frozenset(source)
    missing = sorted(KEPT_FIELDS - keys)
    if missing:
        raise ValueError(f"source tape is missing required fields: {missing}")
    unknown = sorted(keys - KEPT_FIELDS - DECLARED_OUTCOME_FIELDS)
    if unknown:
        raise ValueError(
            "source tape has undeclared fields; classify them before sanitizing: "
            f"{unknown}"
        )

    for name in TENSOR_FIELDS:
        if not torch.is_tensor(source[name]):
            raise TypeError(f"{name} must be a tensor")
    z = source["z"]
    z_next = source["z_next"]
    u = source["u"]
    if z.ndim != 2 or len(z) == 0:
        raise ValueError(f"z must be non-empty [N,D], got {tuple(z.shape)}")
    rows, latent_dim = z.shape
    c = int(source["c"])
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if z_next.shape != z.shape:
        raise ValueError(f"z_next {tuple(z_next.shape)} != z {tuple(z.shape)}")
    if u.shape != (rows, c, 7):
        raise ValueError(f"u must have shape {(rows, c, 7)}, got {tuple(u.shape)}")
    if int(source["latent_dim"]) != latent_dim:
        raise ValueError("latent_dim metadata does not match z")
    if int(source["proprio_dim"]) != 25:
        raise ValueError("canonical v121 tape must have proprio_dim=25")
    keys_value = source["proprio_keys"]
    if not isinstance(keys_value, (list, tuple)) or len(keys_value) != 6:
        raise ValueError("proprio_keys must contain the six v121 state fields")
    if not isinstance(source["task"], str) or not source["task"]:
        raise ValueError("task must be a non-empty string")

    for name in ("z", "z_next", "u", "sigma"):
        if not bool(torch.isfinite(source[name]).all()):
            raise ValueError(f"{name} contains NaN/Inf")
    episode = _integer_tensor(source["episode"], "episode", rows)
    times = _integer_tensor(source["t"], "t", rows)
    if source["sigma"].ndim != 1 or len(source["sigma"]) != rows:
        raise ValueError(f"sigma must be a tensor [{rows}]")

    episode_ids = torch.unique_consecutive(episode).tolist()
    expected_ids = list(range(len(episode_ids)))
    if episode_ids != expected_ids:
        raise ValueError(
            "episode rows must be grouped and episode ids contiguous from zero; "
            f"got {episode_ids[:8]}{'...' if len(episode_ids) > 8 else ''}"
        )
    episode_row_counts: list[int] = []
    for episode_id in episode_ids:
        rows_for_episode = torch.nonzero(episode == episode_id, as_tuple=False).flatten()
        expected_t = torch.arange(len(rows_for_episode), dtype=torch.long) * c
        if not torch.equal(times[rows_for_episode], expected_t):
            raise ValueError(f"episode {episode_id} has non-contiguous t values")
        if len(rows_for_episode) > 1:
            left = z_next[rows_for_episode[:-1]]
            right = z[rows_for_episode[1:]]
            if tensor_sha256(left) != tensor_sha256(right) or not torch.equal(left, right):
                raise ValueError(
                    f"episode {episode_id} violates z_next[i] == z[i+1] exactly"
                )
        episode_row_counts.append(len(rows_for_episode))

    return {
        "rows": rows,
        "latent_dim": latent_dim,
        "episodes": len(episode_ids),
        "episode_ids": episode_ids,
        "episode_row_counts": episode_row_counts,
        "c": c,
    }


def sanitized_copy(source: dict[str, Any]) -> dict[str, Any]:
    """Copy only the fixed allow-list; outcome values are never touched."""
    result: dict[str, Any] = {}
    for name in TENSOR_FIELDS:
        result[name] = source[name].detach().cpu().contiguous().clone()
    for name in METADATA_FIELDS:
        value = source[name]
        result[name] = list(value) if name == "proprio_keys" else value
    return result


def verify_round_trip(source: dict[str, Any], saved: dict[str, Any]) -> None:
    if set(saved) != set(KEPT_FIELDS):
        raise ValueError("saved tape field allow-list changed during round trip")
    for name in TENSOR_FIELDS:
        if tensor_sha256(source[name]) != tensor_sha256(saved[name]):
            raise ValueError(f"{name} bytes changed during sanitization")
        if not torch.equal(source[name].detach().cpu(), saved[name]):
            raise ValueError(f"{name} values changed during sanitization")
    for name in METADATA_FIELDS:
        left = list(source[name]) if name == "proprio_keys" else source[name]
        if left != saved[name]:
            raise ValueError(f"{name} metadata changed during sanitization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tape", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = args.expected_source_sha256.lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        parser.error("--expected-source-sha256 must be 64 lowercase hex characters")
    args.expected_source_sha256 = expected
    return args


def main() -> int:
    args = parse_args()
    source_path = args.source_tape.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(source_path)
    if source_sha != args.expected_source_sha256:
        raise ValueError(
            f"source SHA mismatch: expected {args.expected_source_sha256}, got {source_sha}"
        )
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict):
        raise TypeError("source tape must contain a dictionary")
    validation = validate_source(source)
    sanitized = sanitized_copy(source)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent)
    )
    try:
        tape_path = stage / "sanitized_tape.pt"
        _save_torch(tape_path, sanitized)
        reloaded = torch.load(tape_path, map_location="cpu", weights_only=False)
        if not isinstance(reloaded, dict):
            raise TypeError("round-tripped sanitized tape is not a dictionary")
        verify_round_trip(source, reloaded)

        producer_path = stage / "PRODUCER.py"
        _write_bytes(producer_path, Path(__file__).resolve().read_bytes())
        tensor_hashes = {name: tensor_sha256(reloaded[name]) for name in TENSOR_FIELDS}
        metadata = {name: reloaded[name] for name in METADATA_FIELDS}
        utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "PASS",
            "utc": utc,
            "source": {
                "path": source_path.as_posix(),
                "sha256": source_sha,
                "bytes": source_path.stat().st_size,
                "summary_opened": False,
            },
            "field_policy": {
                "kept": sorted(KEPT_FIELDS),
                "dropped_names_without_value_access": sorted(set(source) - KEPT_FIELDS),
                "declared_outcome_fields": sorted(DECLARED_OUTCOME_FIELDS),
            },
            "validation": validation,
            "metadata": metadata,
            "tensor_sha256": tensor_hashes,
            "metadata_sha256": canonical_sha256(metadata),
            "sanitized_tape_sha256": sha256_file(tape_path),
            "producer_snapshot_sha256": sha256_file(producer_path),
            "git_head": _git_head(),
            "torch_version": torch.__version__,
        }
        result_path = stage / "RESULT.json"
        _write_json(result_path, result)
        complete = {
            "schema": COMPLETE_SCHEMA,
            "status": "atomic_success",
            "atomic_commit": True,
            "source_tape_sha256": source_sha,
            "sanitized_tape_sha256": result["sanitized_tape_sha256"],
            "result_sha256": sha256_file(result_path),
            "producer_snapshot_sha256": result["producer_snapshot_sha256"],
        }
        _write_json(stage / "COMPLETE.json", complete)
        directory_fd = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(stage, output_path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "source_sha256": source_sha,
                "sanitized_tape_sha256": result["sanitized_tape_sha256"],
                "rows": validation["rows"],
                "episodes": validation["episodes"],
                "dropped_fields": result["field_policy"][
                    "dropped_names_without_value_access"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
