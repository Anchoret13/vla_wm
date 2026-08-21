#!/usr/bin/env python
"""CPU-only contract tests for the V8.2 bootstrap collector.

These tests exercise the training-data boundary, not LIBERO or pi0.5.  They
exist to make seven otherwise easy-to-miss errors fail before a long model
fit: an unstable grouped tensor schema, an anchor-level (rather than source-
level) split, a continuation clock that starts before the executed prefix,
retaining the unexecuted 40-action suffix, flattening candidate repeats into
independent examples, a lossy save/load path, and role leakage from Action 2P.

The fixture follows the canonical bootstrap schema:

* C = 7 candidates (one reference plus six alternatives);
* actions are exactly the executed [C, 10, 7] prefix;
* train/val groups have R = 1 and test groups have R = 3;
* repeat axes stay nested inside their source/anchor group.

No simulator, model weights, or external dataset is required.
"""

from __future__ import annotations

import copy
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm import v082_bootstrap as B  # noqa: E402


CANDIDATES = 7
ACTION_STEPS = 10
ACTION_DIM = 7
SIGNATURE_DIM = 24
N_BITS = 2
N_OUTCOMES = 5
REPEATS = {"train": 1, "val": 1, "test": 3}
ROLES = {"train": "model_train", "val": "model_calib",
         "test": "selector_assess"}
OUTCOME_FIELDS = ("dmg", "succ", "dp", "ttm", "G")

PASSED: list[str] = []


def check(name: str, fn) -> None:
    fn()
    PASSED.append(name)
    print(f"  [pass] {name}", flush=True)


def expect_rejected(fn, contains: str | None = None) -> None:
    """Require a contract violation to be rejected, with a useful error."""
    try:
        fn()
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        if contains is not None:
            assert contains.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("invalid bootstrap data was accepted")


def fixture(split: str, source_id: str, anchor_id: str, *, p: int = 4) -> dict:
    """A deterministic, fully-labelled synthetic sibling group."""
    assert split in REPEATS
    r = REPEATS[split]
    full_norm = torch.arange(
        CANDIDATES * 50 * ACTION_DIM, dtype=torch.float32
    ).reshape(CANDIDATES, 50, ACTION_DIM)
    full_env = full_norm / 1000.0

    # Distinct candidate/repeat values make an accidental reshape or reduction
    # observable in both validation and round-trip checks.
    delta = torch.arange(
        CANDIDATES * r * SIGNATURE_DIM, dtype=torch.float32
    ).reshape(CANDIDATES, r, SIGNATURE_DIM)
    bits = torch.zeros(CANDIDATES, r, N_BITS, dtype=torch.bool)
    bits[:, :, 0] = torch.arange(CANDIDATES)[:, None] % 2 == 0
    outcomes = torch.zeros(CANDIDATES, r, N_OUTCOMES, dtype=torch.float32)
    outcomes[:, :, 0] = 0.0                         # damage
    outcomes[:, :, 1] = bits[:, :, 0].float()      # success
    outcomes[:, :, 2] = torch.arange(CANDIDATES)[:, None].float()
    outcomes[:, :, 3] = 81.0                       # censored ttm
    outcomes[:, :, 4] = torch.arange(r)[None].float() / 10.0

    return {
        "schema": B.SCHEMA,
        "anchor_id": anchor_id,
        "source_id": source_id,
        "split": split,
        "role": ROLES[split],
        "prefix_hidden": torch.arange(p * 2048, dtype=torch.float32).reshape(p, 2048),
        "prefix_mask": torch.ones(p, dtype=torch.bool),
        "anchor_signature": torch.arange(SIGNATURE_DIM, dtype=torch.float32),
        # Deliberately take only the executed prefix from a 50-action proposal.
        "actions_norm": full_norm[:, :ACTION_STEPS].clone(),
        "actions_env": full_env[:, :ACTION_STEPS].clone(),
        "candidate_ids": ["reference", "div0", "div1", "div2",
                          "rand0", "rand1", "rand2"],
        "is_reference": torch.tensor(
            [True, False, False, False, False, False, False]
        ),
        "post_prefix_delta": delta,
        "immediate_bits": bits,
        "outcome_h80": outcomes,
        "repeat_mask": torch.ones(CANDIDATES, r, dtype=torch.bool),
        # Extra provenance must be preserved but must not alter tensor shape.
        "provenance": {"collector": "synthetic", "policy_hash": "sha256:test"},
    }


def clone_group(group: dict) -> dict:
    return copy.deepcopy(group)


def assert_same(left, right, path: str = "root") -> None:
    """Deep equality that is strict about tensor dtype, shape, and values."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype, path
        assert left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence) and not isinstance(right, (str, bytes)), path
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_same(a, b, f"{path}[{i}]")
    else:
        assert left == right, path


def outcome_as_vector(outcome) -> torch.Tensor:
    """Normalize the helper's public mapping/vector representation."""
    if isinstance(outcome, Mapping):
        return torch.tensor([float(outcome[k]) for k in OUTCOME_FIELDS])
    return torch.as_tensor(outcome, dtype=torch.float32).reshape(-1)


def test_group_schema_and_shapes() -> None:
    for split, r in REPEATS.items():
        group = fixture(split, f"source-{split}", f"anchor-{split}")
        B.validate_bootstrap_group(group)
        assert group["prefix_hidden"].shape == (4, 2048)
        assert group["actions_norm"].shape == (CANDIDATES, 10, 7)
        assert group["post_prefix_delta"].shape == (CANDIDATES, r, 24)
        assert group["immediate_bits"].shape == (CANDIDATES, r, 2)
        assert group["outcome_h80"].shape == (CANDIDATES, r, 5)
        assert group["repeat_mask"].shape == (CANDIDATES, r)

    missing = fixture("train", "s-missing", "a-missing")
    del missing["outcome_h80"]
    expect_rejected(lambda: B.validate_bootstrap_group(missing), "outcome_h80")

    wrong_hidden = fixture("train", "s-hidden", "a-hidden")
    wrong_hidden["prefix_hidden"] = torch.zeros(4, 2047)
    expect_rejected(lambda: B.validate_bootstrap_group(wrong_hidden))

    wrong_repeat_count = fixture("test", "s-repeat", "a-repeat")
    wrong_repeat_count["post_prefix_delta"] = wrong_repeat_count[
        "post_prefix_delta"
    ][:, :1]
    wrong_repeat_count["immediate_bits"] = wrong_repeat_count["immediate_bits"][:, :1]
    wrong_repeat_count["outcome_h80"] = wrong_repeat_count["outcome_h80"][:, :1]
    wrong_repeat_count["repeat_mask"] = wrong_repeat_count["repeat_mask"][:, :1]
    expect_rejected(lambda: B.validate_bootstrap_group(wrong_repeat_count))


def test_source_disjoint_split() -> None:
    groups = [
        fixture("train", "s-train", "a-train-0"),
        fixture("train", "s-train", "a-train-1"),  # two anchors, one source: legal
        fixture("val", "s-val", "a-val"),
        fixture("test", "s-test", "a-test"),
    ]
    split_sources = {
        "train": {"s-train"},
        "val": {"s-val"},
        "test": {"s-test"},
    }
    assigned = B.split_by_source(groups, split_sources)
    B.validate_groups(assigned)
    assert {(g["source_id"], g["split"]) for g in assigned} == {
        ("s-train", "train"), ("s-val", "val"), ("s-test", "test")
    }

    overlapping = {
        "train": {"s-train"},
        "val": {"s-train", "s-val"},
        "test": {"s-test"},
    }
    expect_rejected(lambda: B.split_by_source(groups, overlapping))

    # Even if individual records are shape-valid, one source may not appear in
    # two splits. This catches the tempting anchor-level random split.
    leaked = [fixture("train", "s-shared", "a0"),
              fixture("test", "s-shared", "a1")]
    expect_rejected(lambda: B.validate_groups(leaked))


def test_continuation_clock_starts_after_prefix() -> None:
    # tau=160 and c_eff=10 imply prefix_end=170. The event at 165 belongs to
    # immediate_bits, not the continuation target. The event at 180 therefore
    # has ttm=10 (not 20), and its return discount exponent is 10 (not 20).
    out = outcome_as_vector(B.continuation_outcome(
        achieved_steps={0: 165, 1: 180},
        success_step=250,
        damage=0,
        tau=160,
        c_eff=10,
        horizon=80,
    ))
    assert out.shape == (N_OUTCOMES,)
    dmg, succ, dp, ttm, ret = [float(x) for x in out]
    assert dmg == 0.0
    assert succ == 1.0, "success at prefix_end + H must be inside H=80"
    assert dp == 1.0, "prefix events leaked into continuation progress"
    assert ttm == 10.0, "ttm clock did not start at tau+c_eff"
    gamma = float(getattr(B, "GAMMA", 0.99))
    assert math.isclose(ret, gamma ** 10, rel_tol=1e-6, abs_tol=1e-7), ret

    censored = outcome_as_vector(B.continuation_outcome(
        achieved_steps={0: 165},
        success_step=251,
        damage=0,
        tau=160,
        c_eff=10,
        horizon=80,
    ))
    assert float(censored[1]) == 0.0
    assert float(censored[2]) == 0.0
    assert float(censored[3]) == 81.0
    assert float(censored[4]) == 0.0


def test_actions_are_first_ten_only() -> None:
    valid = fixture("train", "s-actions", "a-actions")
    B.validate_bootstrap_group(valid)
    assert valid["actions_norm"].shape[1] == ACTION_STEPS

    # A byte-perfect 50-action proposal is still causally invalid here: only
    # its executed first ten may reach E_a or the stored bootstrap group.
    bad = clone_group(valid)
    bad["actions_norm"] = torch.zeros(CANDIDATES, 50, ACTION_DIM)
    bad["actions_env"] = torch.zeros(CANDIDATES, 50, ACTION_DIM)
    expect_rejected(lambda: B.validate_bootstrap_group(bad))


def test_candidate_repeats_remain_nested() -> None:
    group = fixture("test", "s-nested", "a-nested")
    B.validate_bootstrap_group(group)
    assert len(group["candidate_ids"]) == CANDIDATES
    assert group["is_reference"].shape == (CANDIDATES,)
    assert int(group["is_reference"].sum()) == 1
    assert group["post_prefix_delta"].shape[:2] == (CANDIDATES, 3)

    # The repeat values survive under candidate 0, rather than becoming three
    # pseudo-independent candidate rows.
    assert torch.equal(
        group["outcome_h80"][0, :, 4], torch.tensor([0.0, 0.1, 0.2])
    )

    flattened = clone_group(group)
    flattened["post_prefix_delta"] = flattened["post_prefix_delta"].reshape(
        CANDIDATES * 3, SIGNATURE_DIM
    )
    expect_rejected(lambda: B.validate_bootstrap_group(flattened))

    duplicate_ref = clone_group(group)
    duplicate_ref["is_reference"][1] = True
    expect_rejected(lambda: B.validate_bootstrap_group(duplicate_ref))


def test_save_load_roundtrip() -> None:
    group = fixture("test", "s-roundtrip", "a-roundtrip")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nested" / "a-roundtrip.pt"
        B.save_bootstrap_group(group, path)
        assert path.is_file()
        loaded = B.load_bootstrap_group(path)
        B.validate_bootstrap_group(loaded)
        assert_same(group, loaded)

        invalid = clone_group(group)
        invalid["actions_env"] = invalid["actions_env"][:, :9]
        expect_rejected(lambda: B.save_bootstrap_group(invalid, Path(td) / "bad.pt"))


def test_role_isolation() -> None:
    # The only legal split/role pairs for the bootstrap object. Assessment is a
    # held-out role; it is never silently folded back into model training.
    for split, role in (
        ("train", "model_train"),
        ("val", "model_calib"),
        ("test", "selector_assess"),
    ):
        B.assert_role_allowed(role)
        B.assert_split_role(split, role)

    for role, subrole in (
        ("calibration", "coverage_pilot"),  # Action 2P: permanently excluded
        ("behavior_eval", "test"),
        ("verified_correction", "train"),
    ):
        expect_rejected(lambda role=role, subrole=subrole:
                        B.assert_role_allowed(role, subrole=subrole))

    # A legal role under the wrong split is still leakage.
    expect_rejected(lambda: B.assert_split_role("train", "model_calib"))
    expect_rejected(lambda: B.assert_split_role("val", "selector_assess"))


def main() -> None:
    torch.set_num_threads(1)
    check("canonical group schema + split-specific shapes", test_group_schema_and_shapes)
    check("source-disjoint split (siblings and anchors stay grouped)",
          test_source_disjoint_split)
    check("continuation clock starts at tau+c_eff", test_continuation_clock_starts_after_prefix)
    check("only the executed first ten actions are stored", test_actions_are_first_ten_only)
    check("candidate repeats remain nested inside groups", test_candidate_repeats_remain_nested)
    check("validated save/load round-trip is lossless", test_save_load_roundtrip)
    check("training/calibration/assessment roles are isolated", test_role_isolation)
    print(f"ALL {len(PASSED)} v082 bootstrap contract tests passed", flush=True)


if __name__ == "__main__":
    main()
