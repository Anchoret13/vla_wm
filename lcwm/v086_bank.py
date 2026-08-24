"""B_boot-v3 tensor contract for Action 2M.4.

The readout change is the point of this bank.  Earlier V8 banks scored a
branch with a deterministic lexicographic label over `(dmg, succ, dp, ttm, G)`
evaluated at the automaton's 10-step cadence; Action 2M.3 showed that label is
quantized far coarser than the candidate effect it must resolve - `G` took ten
distinct values across 900 branches.

Here the primary target is a CONTINUOUS action consequence: the phase potential
evaluated at EVERY environment step, discounted, and averaged over shared-seed
repeats.  The deterministic target for one `(state, candidate)` is the candidate
mean across its repeats; repeats estimate that target and are retained for
calibration, never converted into conflicting per-rollout labels for one
deterministic model output.
"""

from __future__ import annotations

TASK = "chain1b_lr2"
DEADLINE = 250
TAU = 160
C_PREFIX = 10
H = 80
assert TAU + C_PREFIX + H == DEADLINE

SCHEMA = "v086_bank_group_v1"
GAMMA = 0.99

N_RAW_DRAWS = 64
N_ALTERNATIVES = 4
N_CANDIDATES = 1 + N_ALTERNATIVES

QUOTAS = {"train": 48, "val": 8, "test": 8}
REPEATS = {"train": 4, "val": 6, "test": 6}
QUEUES = {"train": tuple(range(4000, 4108)),      # 108 sources
          "val": tuple(range(4108, 4131)),        #  23
          "test": tuple(range(4131, 4154))}       #  23
SPLIT_ROLES = {"train": "model_train", "val": "model_calib",
               "test": "selector_assess"}

#: Every seed family already consumed by v082/v083/v084/v085.  Asserted here so
#: a source-disjointness violation is an import error, not a silent overlap.
SPENT = (frozenset(range(3400, 3440)) | frozenset(range(3480, 3500))
         | frozenset(range(3500, 3576)) | frozenset(range(3600, 3706))
         | frozenset(range(3800, 3816)) | frozenset(range(3900, 3930))
         | frozenset(range(3940, 3978)))
RESERVED = frozenset(range(3200, 3400))
assert not (set(sum((list(q) for q in QUEUES.values()), [])) & (SPENT | RESERVED))

BRANCHES = sum(QUOTAS[s] * N_CANDIDATES * REPEATS[s] for s in QUOTAS)   # 1,440
SOURCE_CAP = sum(len(q) for q in QUEUES.values()) * DEADLINE            # 38,500
BRANCH_CAP = BRANCHES * (C_PREFIX + H)                                 # 129,600
INTERACTION_CAP = {"source": SOURCE_CAP, "branch": BRANCH_CAP}
INTERACTION_CAP_TOTAL = SOURCE_CAP + BRANCH_CAP                        # 168,100
assert BRANCHES == 1440 and INTERACTION_CAP_TOTAL == 168_100

#: Primary target and the secondary vector retained only for reporting.
PRIMARY_TARGET = "QPhi"
CONTINUOUS_FIELDS = ("GPhi_exec", "dPhi_exec", "GPhi_cont_20", "GPhi_cont_40",
                     "GPhi_cont_80", "dPhi_cont_20", "dPhi_cont_40",
                     "dPhi_cont_80", "QPhi")
BINARY_FIELDS = ("dmg", "succ")
SECONDARY_LEGACY = ("dp", "ttm", "G")

REQUIRED_FIELDS = {
    "schema", "anchor_id", "source_id", "split", "role",
    "prefix_hidden", "prefix_mask", "anchor_signature",
    "actions_norm", "actions_env", "candidate_ids", "is_reference",
    "post_prefix_delta", "continuous", "binary", "legacy",
    "phi_trace", "phi_tau", "repeat_mask",
}

#: "No label-count or noise-floor condition may stop the run before training."
NO_EARLY_STOP = ("collection and training proceed regardless of how many "
                 "candidate/reference differences appear; only a technical "
                 "fault, source underfill, or the interaction cap may halt")


def assert_split_role(split: str, role: str) -> None:
    if split not in SPLIT_ROLES:
        raise ValueError(f"unknown split {split!r}")
    if SPLIT_ROLES[split] != role:
        raise ValueError(f"split {split!r} requires role {SPLIT_ROLES[split]!r}, "
                         f"got {role!r}")


def validate_group(g: dict) -> None:
    missing = REQUIRED_FIELDS - set(g)
    if missing:
        raise ValueError(f"group missing {sorted(missing)}")
    assert_split_role(g["split"], g["role"])
    C, R = N_CANDIDATES, REPEATS[g["split"]]
    if len(g["candidate_ids"]) != C:
        raise ValueError(f"expected {C} candidates, got {len(g['candidate_ids'])}")
    if sum(bool(x) for x in g["is_reference"]) != 1:
        raise ValueError("exactly one candidate must be the reference")
    for name, cols in (("continuous", CONTINUOUS_FIELDS),
                       ("binary", BINARY_FIELDS), ("legacy", SECONDARY_LEGACY)):
        t = g[name]
        if tuple(t.shape) != (C, R, len(cols)):
            raise ValueError(f"{name} must be [{C},{R},{len(cols)}], got {tuple(t.shape)}")
    if tuple(g["post_prefix_delta"].shape) != (C, R, 24):
        raise ValueError("post_prefix_delta must be [C,R,24]")
    if tuple(g["repeat_mask"].shape) != (C, R):
        raise ValueError("repeat_mask must be [C,R]")
    if len(g["phi_trace"]) != C or any(len(x) != R for x in g["phi_trace"]):
        raise ValueError("phi_trace must be C x R per-step traces")


def candidate_mean_targets(g: dict) -> dict:
    """The deterministic model target: candidate mean across shared-seed repeats.

    `delta` is the candidate-minus-reference mean consequence, which is what the
    paired-effect head regresses directly.
    """
    import torch

    m = g["repeat_mask"].float().unsqueeze(-1)
    denom = m.sum(1).clamp(min=1.0)
    cont = (g["continuous"] * m).sum(1) / denom          # [C, F]
    binr = (g["binary"] * m).sum(1) / denom              # [C, 2] repeat-mean rates
    phys = (g["post_prefix_delta"] * m).sum(1) / denom   # [C, 24]
    qi = CONTINUOUS_FIELDS.index(PRIMARY_TARGET)
    S = cont[:, qi]
    ref = int(torch.nonzero(g["is_reference"].bool()).flatten()[0])
    return {"continuous_mean": cont, "binary_rate": binr, "physical_mean": phys,
            "S": S, "delta": S - S[ref], "reference_index": ref}
