"""v0.6.7 lineage guard + sibling-CRN contract (2026-07-31 V6.7.0/V6.7.1).

Two registered repairs live here:

1. run_schema guard — a repaired (v067) job must refuse to load an
   iteration-1 continuation, semantic-label, teacher, or policy manifest.
   Iteration-1 phase-A sources, feature sidecars, and the group selection
   remain reusable by hash reference; they are NOT guarded kinds.

2. sibling common randomness — the continuation RNG is a stable hash of
   (run_id, source_id, snapshot_decision, goal_spec_id, repeat,
   continuation_decision). Candidate/branch identity is structurally
   absent from the signature: sibling branches of one snapshot share the
   continuation noise stream bit-exactly by construction. The iteration-1
   defect (`+ branch_index * 40`) gave every candidate its own π0.5
   flow-noise stream, conflating candidate effect with continuation noise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

RUN_SCHEMA = "v067"

# Iteration-1 schemas that must never enter a repaired job.
ITER1_FORBIDDEN_SCHEMAS = {
    "v06_continuations_v1",   # candidate-specific continuation noise
    "v06_labels_v1",          # chunk-resolution, canonical-only success
    "v06_teachers_v1",
    "v06_gt_manifest_v1",
}

# Kinds that are regenerated under v067 and therefore guarded.
GUARDED_KINDS = ("continuations", "semantic_labels", "teachers", "policy",
                 "goal_manifest", "history_contrasts")


class Iter1ArtifactError(RuntimeError):
    """An iteration-1 (or unversioned) artifact reached a v067 job."""


def assert_v067_payload(payload: dict, path: str | Path, kind: str) -> None:
    if kind not in GUARDED_KINDS:
        raise ValueError(f"unknown guarded kind {kind!r}")
    schema = payload.get("schema")
    if schema in ITER1_FORBIDDEN_SCHEMAS:
        raise Iter1ArtifactError(
            f"{path}: schema {schema!r} is an iteration-1 {kind} artifact; "
            "the v0.6.7 repair regenerates this kind (2026-07-31 V6.7.0)")
    if payload.get("run_schema") != RUN_SCHEMA:
        raise Iter1ArtifactError(
            f"{path}: {kind} artifact has run_schema="
            f"{payload.get('run_schema')!r}, expected {RUN_SCHEMA!r}; "
            "unversioned artifacts are iteration-1 and refused")


def load_v067(path: str | Path, kind: str) -> dict:
    payload = torch.load(path, weights_only=False)
    assert_v067_payload(payload, path, kind)
    return payload


# ---- sibling common randomness ------------------------------------------

def cont_seed_v067(run_id: str, source_id: str, snap_decision: int,
                   goal_spec_id: str, repeat: int,
                   cont_decision: int) -> int:
    """63-bit continuation seed. Branch/candidate identity is NOT an
    argument — sibling candidates share the stream by construction."""
    key = (f"v067|{run_id}|{source_id}|{snap_decision}|{goal_spec_id}"
           f"|{repeat}|{cont_decision}")
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def flow_noise(seed: int, chunk_size: int, max_action_dim: int,
               n: int = 1) -> torch.Tensor:
    """Bit-exact replica of sampler.sample_chunks' seeded noise draw
    (CPU generator, float32). Pre-generating lets every continuation
    decision store the SHA of the exact tensor the sampler consumed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((n, chunk_size, max_action_dim), generator=g,
                       dtype=torch.float32)


def noise_sha(noise: torch.Tensor) -> str:
    return hashlib.sha256(
        noise.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
