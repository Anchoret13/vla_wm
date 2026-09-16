#!/usr/bin/env python
"""Mechanical CPU-only tests for scripts/eval_v251_round.py.

WHAT THESE VERIFY.  Every number the driver reports is produced by an arithmetic
path checked here against a value computed by hand or against a published
reference, and every refusal the driver is supposed to make is triggered:

  * exact McNemar against hand-computed binomial tail values, including the
    b = 3, c = 0 -> one-sided 0.125 case that scripts/run_v088_behavior.py:31
    already records as the ceiling of a small discordant table;
  * Clopper-Pearson against the exact 2/96 interval [0.0025330, 0.0732368] and
    against the 0/20 one-sided 0.1391 bound recorded in lcwm/v08r_contract.py:331;
  * Holm ordering, adjusted alpha and monotone adjusted p on a fixed p-vector;
  * the seed guard REFUSING an evaluation panel that overlaps a collection family,
    with no override, and accepting a declared probe overlap only when declared;
  * a non-zero subprocess return code aborting the driver instead of being
    discarded (the defect at scripts/run_v208_deploy_loop.py:101/113/122), and an
    output directory that does not carry this run's tag being rejected rather than
    silently accepted;
  * --dry-run writing the preregistration and nothing else, and taking no
    environment step;
  * every arm being invoked WITH the evaluation instrumentation, and the four
    defects an adversarial pass found on 2026-09-12: a sidecar written without
    --eval-labels being read as a whole-panel zero instead of as missing
    instrumentation, a phi series joined against a last-progress step that came
    from a different file, a phi series with no time base being reduced to its
    final boundary under the basin endpoint's name, and a binary endpoint
    arriving as the string "False".

No simulator, no GPU, no torch: the fixtures are synthetic JSON and trivial
subprocesses. `checkpoint_fingerprint` is the only torch path in the driver and is
not exercised here - it is CPU-only but needs a real checkpoint, which this file
must not fabricate.

Run: /home/stargazer/miniconda3/envs/vf0s/bin/python -m pytest scripts/test_eval_v251_round.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_v251_round as E

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except Exception:                                # pragma: no cover
    scipy_stats = None
    HAVE_SCIPY = False


# --------------------------------------------------------------------------
# exact McNemar
# --------------------------------------------------------------------------
def test_mcnemar_five_zero_is_one_over_thirty_two():
    r = E.exact_mcnemar(5, 0)
    assert r["n_discordant"] == 5
    assert r["p_one_sided_a_greater"] == pytest.approx(1 / 32)       # 0.03125
    assert r["p_two_sided"] == pytest.approx(2 / 32)                 # 0.0625


def test_mcnemar_three_zero_matches_the_recorded_ceiling():
    # run_v088_behavior.py:31 - "even a perfect fix gives McNemar p=0.125"
    r = E.exact_mcnemar(3, 0)
    assert r["p_one_sided_a_greater"] == pytest.approx(0.125)
    assert r["p_two_sided"] == pytest.approx(0.25)


def test_mcnemar_eight_one_hand_computed():
    r = E.exact_mcnemar(8, 1)
    assert r["p_two_sided"] == pytest.approx(2 * (1 + 9) / 512)      # 0.0390625
    assert r["p_one_sided_a_greater"] == pytest.approx((9 + 1) / 512)


def test_mcnemar_twelve_four_hand_computed():
    r = E.exact_mcnemar(12, 4)
    expect = 2 * (1 + 16 + 120 + 560 + 1820) / 2 ** 16
    assert r["p_two_sided"] == pytest.approx(expect)
    assert r["b"] == 12 and r["c"] == 4


def test_mcnemar_no_discordant_pairs_is_one():
    r = E.exact_mcnemar(0, 0)
    assert r["p_two_sided"] == 1.0 and r["p_one_sided_a_greater"] == 1.0
    assert "no discordant pairs" in r["note"]


def test_mcnemar_is_symmetric_two_sided_and_directional_one_sided():
    a, b = E.exact_mcnemar(7, 2), E.exact_mcnemar(2, 7)
    assert a["p_two_sided"] == pytest.approx(b["p_two_sided"])
    assert a["p_one_sided_a_greater"] < b["p_one_sided_a_greater"]


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        E.exact_mcnemar(-1, 3)


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy absent; own value still reported")
def test_mcnemar_agrees_with_scipy_binomtest():
    for b, c in [(5, 0), (3, 0), (8, 1), (12, 4), (2, 2), (1, 2), (30, 17)]:
        r = E.exact_mcnemar(b, c)
        ref = float(scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue)
        assert r["p_two_sided"] == pytest.approx(ref, abs=1e-12)
        assert r["scipy_agrees"] is True


# --------------------------------------------------------------------------
# Clopper-Pearson
# --------------------------------------------------------------------------
def test_clopper_pearson_two_over_ninetysix_reference_interval():
    lo, hi = E.clopper_pearson(2, 96)
    assert lo == pytest.approx(0.0025330467891847633, abs=1e-9)
    assert hi == pytest.approx(0.07323683079498775, abs=1e-9)


def test_clopper_pearson_one_sided_zero_of_twenty_is_the_recorded_bound():
    # lcwm/v08r_contract.py:331 - "0/20 -> 0.1391"
    assert E.clopper_pearson_upper(0, 20, 0.05) == pytest.approx(0.1391, abs=5e-5)


def test_clopper_pearson_edges():
    assert E.clopper_pearson(0, 96)[0] == 0.0
    assert E.clopper_pearson(96, 96)[1] == 1.0
    lo, hi = E.clopper_pearson(48, 96)
    assert lo < 0.5 < hi


def test_clopper_pearson_contains_the_point_estimate():
    for k in (0, 1, 2, 5, 13, 47, 96):
        lo, hi = E.clopper_pearson(k, 96)
        assert lo <= k / 96 <= hi


# --------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------
def test_holm_ordering_on_a_fixed_p_vector():
    out = E.holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.4}, alpha=0.05)
    assert [out[k]["rank"] for k in ("a", "b", "c", "d")] == [1, 2, 3, 4]
    assert out["a"]["alpha_adjusted"] == pytest.approx(0.05 / 4)
    assert out["b"]["alpha_adjusted"] == pytest.approx(0.05 / 3)
    assert out["c"]["alpha_adjusted"] == pytest.approx(0.05 / 2)
    assert out["d"]["alpha_adjusted"] == pytest.approx(0.05 / 1)
    # step-down: b fails at 0.05/3, so c and d cannot be rejected even though
    # c's own alpha would admit it
    assert out["a"]["reject"] is True
    assert [out[k]["reject"] for k in ("b", "c", "d")] == [False, False, False]


def test_holm_adjusted_p_is_monotone_and_hand_computed():
    out = E.holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.4}, alpha=0.05)
    assert out["a"]["p_adjusted"] == pytest.approx(0.004)
    assert out["b"]["p_adjusted"] == pytest.approx(0.06)
    assert out["c"]["p_adjusted"] == pytest.approx(0.06)     # monotone, not 0.06->0.06
    assert out["d"]["p_adjusted"] == pytest.approx(0.4)
    seq = [out[k]["p_adjusted"] for k in ("a", "b", "c", "d")]
    assert seq == sorted(seq)


def test_holm_single_test_is_uncorrected():
    out = E.holm({"only": 0.04}, alpha=0.05)
    assert out["only"]["alpha_adjusted"] == pytest.approx(0.05)
    assert out["only"]["p_adjusted"] == pytest.approx(0.04)
    assert out["only"]["reject"] is True


def test_holm_ties_are_deterministic_by_name():
    out = E.holm({"z": 0.01, "a": 0.01}, alpha=0.05)
    assert out["a"]["rank"] == 1 and out["z"]["rank"] == 2


# --------------------------------------------------------------------------
# Wilcoxon / bootstrap
# --------------------------------------------------------------------------
def test_wilcoxon_exact_small_case_hand_computed():
    # all five differences positive: T+ = 15, and P(T+ >= 15) = 1/32
    r = E.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r["n_nonzero"] == 5
    assert r["t_plus"] == pytest.approx(15.0)
    assert r["p_two_sided"] == pytest.approx(2 / 32)
    assert r["method"].startswith("exact")


def test_wilcoxon_discards_zero_pairs():
    r = E.wilcoxon_signed_rank([0.0, 0.0, 1.0, 2.0, 3.0])
    assert r["n_nonzero"] == 3 and r["n_zero_pairs"] == 2


def test_wilcoxon_all_ties_is_degenerate_not_significant():
    r = E.wilcoxon_signed_rank([0.0] * 20)
    assert r["p_two_sided"] == 1.0 and r["n_nonzero"] == 0


@pytest.mark.skipif(not HAVE_SCIPY, reason="scipy absent; own value still reported")
def test_wilcoxon_agrees_with_scipy_exact_and_approx():
    import random
    rng = random.Random(1234)
    cases = [[1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0],
             [rng.gauss(0.2, 1.0) for _ in range(40)],
             [float(rng.randint(-2, 3)) for _ in range(96)]]      # ties on purpose
    for diffs in cases:
        r = E.wilcoxon_signed_rank(diffs)
        nz = [d for d in diffs if d != 0.0]
        ref = float(scipy_stats.wilcoxon(nz, zero_method="wilcox",
                                         alternative="two-sided").pvalue)
        assert r["p_two_sided"] == pytest.approx(ref, abs=1e-6)


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    diffs = [0.1 * ((i % 7) - 3) for i in range(96)]
    a, b = E.paired_bootstrap(diffs), E.paired_bootstrap(diffs)
    assert a["ci95"] == b["ci95"]                 # fixed seed, reused implementation
    assert a["ci95"][0] <= a["mean"] <= a["ci95"][1]
    assert a["resamples"] == E.BOOT_N and a["seed"] == E.BOOT_SEED


# --------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------
def test_min_discordant_counts_are_five_and_six():
    assert E.mcnemar_min_discordant(0.05, two_sided=False) == 5      # 0.5**5 = .03125
    assert E.mcnemar_min_discordant(0.05, two_sided=True) == 6       # 2*.5**6 = .03125


def test_power_block_declares_the_binary_endpoint_estimation_only_at_96():
    pw = E.power_block(96)["binary_endpoint"]
    assert pw["expected_base_successes_at_n"] == 2
    assert pw["min_treated_count_one_sided"] == 7
    assert pw["min_treated_count_two_sided"] == 8
    assert pw["is_estimation_endpoint_only"] is True
    assert "ESTIMATION endpoint" in pw["statement"]
    assert "CANNOT DETECT" in pw["statement"]


def test_detectable_paired_effect_matches_the_closed_form():
    mde = E.min_detectable_paired_effect(96)
    expect = (E.norm_ppf(0.975) + E.norm_ppf(0.80)) / math.sqrt(96)
    assert mde["cohen_dz"] == pytest.approx(expect)
    assert mde["cohen_dz"] == pytest.approx(0.2859, abs=1e-3)


def test_norm_ppf_round_trips():
    for p in (0.025, 0.5, 0.8, 0.975):
        assert E.norm_cdf(E.norm_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_fisher_one_sided_is_a_probability():
    p = E.fisher_one_sided_greater(9, 96, 2, 96)
    assert 0.0 < p < 0.05
    assert E.fisher_one_sided_greater(2, 96, 2, 96) > 0.05


# --------------------------------------------------------------------------
# the seed guard
# --------------------------------------------------------------------------
def test_guard_refuses_a_panel_overlapping_a_collection_family():
    with pytest.raises(SystemExit) as exc:
        E.check_panel(8750, 96, E.KNOWN_FAMILIES)
    msg = str(exc.value)
    assert "d0_rich_tape" in msg and "collection" in msg
    assert "no override" in msg


def test_guard_collection_overlap_cannot_be_acknowledged_away():
    with pytest.raises(SystemExit):
        E.check_panel(8700, 96, E.KNOWN_FAMILIES, acknowledge_reuse=True)


def test_guard_refuses_a_probe_overlap_until_it_is_declared():
    with pytest.raises(SystemExit) as exc:
        E.check_panel(9030, 24, E.KNOWN_FAMILIES)
    assert "--acknowledge-panel-reuse" in str(exc.value)
    out = E.check_panel(9030, 24, E.KNOWN_FAMILIES, acknowledge_reuse=True)
    assert out["clean"] is False
    assert out["declared_reuse"][0]["family"] == "v249_v250_calibration"


def test_guard_passes_a_disjoint_panel_and_records_what_it_checked():
    out = E.check_panel(9100, 96, E.KNOWN_FAMILIES)
    assert out["clean"] is True and out["declared_reuse"] == []
    assert len(out["checked_families"]) == len(E.KNOWN_FAMILIES)
    assert out["panel"] == [9100, 9195, 96]


def test_guard_catches_a_one_seed_touch_at_the_boundary():
    with pytest.raises(SystemExit):
        E.check_panel(8795, 1, E.KNOWN_FAMILIES)              # 8795 is still D0
    # 8796-8799 sits in the gap between D0 (..8795) and D1 (8800..)
    assert E.check_panel(8796, 4, E.KNOWN_FAMILIES)["clean"] is True


def test_families_from_manifest_reads_both_shapes(tmp_path):
    a = tmp_path / "manifest.json"
    a.write_text(json.dumps({"seed_range": [9500, 9547], "role": "resid"}))
    fam = E.families_from_manifest(a)[0]
    assert (fam.lo, fam.hi, fam.role) == (9500, 9548, "collection")
    b = tmp_path / "other.json"
    b.write_text(json.dumps({"seed_start": 9600, "episodes": 48}))
    fam = E.families_from_manifest(b)[0]
    assert (fam.lo, fam.hi) == (9600, 9648)


def test_families_from_manifest_raises_rather_than_registering_nothing(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"task": "chain3_lr2"}))
    with pytest.raises(SystemExit):
        E.families_from_manifest(p)


def test_manifest_family_actually_blocks_the_panel(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"seed_range": [9100, 9147]}))
    fams = list(E.KNOWN_FAMILIES) + E.families_from_manifest(p)
    E.check_panel(9100, 96, E.KNOWN_FAMILIES)                 # clean without it
    with pytest.raises(SystemExit):
        E.check_panel(9100, 96, fams)                         # blocked with it


# --------------------------------------------------------------------------
# subprocess handling
# --------------------------------------------------------------------------
def _arm_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"episodes": []}))
    return d


def test_nonzero_return_code_aborts_the_driver(tmp_path):
    cmd = [sys.executable, "-c", "import sys; print('work'); sys.exit(3)"]
    with pytest.raises(E.ArmFailed) as exc:
        E.execute_arm("M1", cmd, tmp_path / "arm.log", {}, "tagX")
    assert "exited 3" in str(exc.value)
    assert (tmp_path / "arm.log").read_text().strip() == "work"


def test_successful_arm_resolves_its_directory_from_its_own_stdout(tmp_path):
    d = _arm_dir(tmp_path, "chain3_lr2_tagX_2026")
    cmd = [sys.executable, "-c", f"print('12 steps -> {d}')"]
    got = E.execute_arm("M1", cmd, tmp_path / "arm.log", {}, "tagX")
    assert got == d


def test_out_dir_marker_form_is_accepted(tmp_path):
    d = _arm_dir(tmp_path, "run_tagY_z")
    assert E.resolve_out_dir(f"noise\nOUT_DIR={d}\n", "tagY", 0.0) == d


def test_a_directory_without_this_runs_tag_is_rejected(tmp_path):
    d = _arm_dir(tmp_path, "some_other_run")
    with pytest.raises(E.ArmFailed) as exc:
        E.resolve_out_dir(f"-> {d}\n", "tagX", 0.0)
    assert "tag" in str(exc.value)


def test_a_stale_directory_is_rejected_by_its_mtime(tmp_path):
    d = _arm_dir(tmp_path, "chain3_tagX_old")
    future = 2 ** 31                              # the subprocess "started" later
    with pytest.raises(E.ArmFailed) as exc:
        E.resolve_out_dir(f"-> {d}\n", "tagX", future)
    assert "predates" in str(exc.value)


def test_no_resolvable_path_aborts_rather_than_globbing(tmp_path):
    with pytest.raises(E.ArmFailed) as exc:
        E.resolve_out_dir("finished, wrote everything\n", "tagX", 0.0)
    assert "no globbing" in str(exc.value)


def test_base_arm_command_carries_zero_residual_and_the_others_do_not():
    base = E.arm_command("base", Path("/x/a.pt"), 96, 9100, "t")
    m1 = E.arm_command("M1", Path("/x/a.pt"), 96, 9100, "t")
    assert "--zero-residual" in base and "--zero-residual" not in m1
    for cmd in (base, m1):
        assert cmd[2] == E.DEPLOY
        assert cmd[cmd.index("--panel") + 1] == "96"
        assert cmd[cmd.index("--panel-start") + 1] == "9100"


# --------------------------------------------------------------------------
# endpoint extraction
# --------------------------------------------------------------------------
def test_contiguous_stage_is_not_the_milestone_count():
    # milestones 0,1,2,3 then a skip to 5: the contiguous stage is 4, not 5
    got = E.derive_from_events({"0": 70, "1": 140, "2": 210, "3": 260, "5": 600})
    assert got["stage"] == 4.0
    assert got["milestone4"] == 0.0
    assert got["last_progress_step"] == 600.0


def test_milestone_four_attainment_and_empty_events():
    got = E.derive_from_events({0: 70, 1: 140, 2: 210, 3: 260, 4: 460})
    assert got["milestone4"] == 1.0 and got["stage"] == 5.0
    empty = E.derive_from_events({})
    assert empty == {"milestone4": 0.0, "stage": 0.0, "last_progress_step": 0.0}


def test_phi_basin_mean_uses_only_boundaries_after_the_last_milestone():
    rec = {"phi": [1.0, 2.0, 3.0, 4.0], "phi_t": [0, 100, 300, 500]}
    assert E.derive_phi_basin(rec, 260.0) == pytest.approx(3.5)
    assert E.derive_phi_basin(rec, 900.0) == pytest.approx(4.0)   # empty basin rule
    assert E.derive_phi_basin({"phi_basin_mean": 2.25}, 0.0) == pytest.approx(2.25)
    assert E.derive_phi_basin({}, 0.0) is None


def _write_sidecar(d: Path, seeds, m4_seeds=(), phi=2.0, name="progress.json"):
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in seeds:
        ev = {"0": 70, "1": 140, "2": 210, "3": 260}
        if s in m4_seeds:
            ev["4"] = 460
        rows.append({"seed": s, "success": False, "events": ev,
                     "phi_basin_mean": phi + (0.01 if s in m4_seeds else 0.0),
                     "d_min_final_atom": 0.10 if s in m4_seeds else 0.50})
    (d / name).write_text(json.dumps({"episodes": rows}))
    return d


def test_read_arm_endpoints_derives_every_powered_endpoint(tmp_path):
    d = _write_sidecar(tmp_path / "arm", range(9100, 9108), m4_seeds={9101, 9103})
    got = E.read_arm_endpoints(d)
    assert got["n"] == 8 and got["unavailable"] == []
    assert got["per_seed"][9101]["milestone4"] == 1.0
    assert got["per_seed"][9100]["milestone4"] == 0.0
    assert got["per_seed"][9100]["stage"] == 4.0
    assert got["per_seed"][9101]["stage"] == 5.0
    assert got["per_seed"][9100]["last_progress_step"] == 260.0
    assert got["per_seed"][9101]["last_progress_step"] == 460.0
    assert got["source"] == [str(d / "progress.json")]


def test_read_arm_endpoints_reports_a_missing_endpoint_instead_of_dropping_it(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(
        {"episodes": [{"seed": 9100, "success": True, "events": {"0": 70}}]}))
    got = E.read_arm_endpoints(d)
    assert got["unavailable"] == ["phi_basin_mean", "d_min_final_atom"]
    assert got["per_seed"][9100]["stage"] == 1.0


def test_read_arm_endpoints_refuses_an_unreadable_arm(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"task": "chain3_lr2"}))
    with pytest.raises(E.ArmFailed):
        E.read_arm_endpoints(d)


def test_read_arm_endpoints_never_imputes_a_missing_seed(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "progress.json").write_text(json.dumps({"episodes": [
        {"seed": 9100, "events": {"0": 70}, "phi_basin_mean": 2.0,
         "d_min_final_atom": 0.4},
        {"events": {"0": 70}}]}))          # a record with no seed is skipped
    got = E.read_arm_endpoints(d)
    assert list(got["per_seed"]) == [9100]
    assert got["unavailable"] == ["success"]


def test_read_arm_endpoints_merges_the_sidecar_with_summary_json(tmp_path):
    # the realistic split: the progress sidecar has the per-boundary quantities
    # and no terminal outcome, summary.json has the terminal outcome and nothing
    # else. Taking only one file would drop a registered endpoint.
    d = tmp_path / "arm"
    d.mkdir()
    (d / "progress.json").write_text(json.dumps({"episodes": [
        {"seed": s, "events": {"0": 70, "1": 140, "2": 210, "3": 260},
         "phi_basin_mean": 2.1, "d_min_final_atom": 0.42}
        for s in range(9100, 9104)]}))
    (d / "summary.json").write_text(json.dumps({"episodes": [
        {"seed": s, "success": s == 9102, "steps": 750}
        for s in range(9100, 9104)]}))
    got = E.read_arm_endpoints(d)
    assert got["unavailable"] == []
    assert got["per_seed"][9102]["success"] == 1.0
    assert got["per_seed"][9100]["success"] == 0.0
    assert got["per_seed"][9100]["phi_basin_mean"] == pytest.approx(2.1)
    assert got["per_seed"][9100]["stage"] == 4.0
    assert len(got["source"]) == 2


# --------------------------------------------------------------------------
# contrasts end to end, on synthetic per-seed vectors
# --------------------------------------------------------------------------
def _arm(name, per_seed):
    return E.ArmResult(name=name, role="synthetic", actor="<none>", sha256=None,
                       per_seed=per_seed)


def _rows(m4_flags, phi_offset=0.0):
    return {9100 + i: {"success": 0.0, "milestone4": float(f), "stage": 4.0 + f,
                       "last_progress_step": 260.0 + 200.0 * f,
                       "phi_basin_mean": 2.0 + phi_offset + 0.1 * f}
            for i, f in enumerate(m4_flags)}


def test_binary_contrast_reports_the_discordant_table_with_the_p_value():
    a = _arm("M1", _rows([1] * 12 + [0] * 84))
    b = _arm("M0", _rows([0] * 12 + [0] * 84))
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["milestone4"], "panel test")
    assert c["a_count"] == 12 and c["b_count"] == 0
    assert c["discordant"]["b"] == 12 and c["discordant"]["c"] == 0
    assert c["p"] == pytest.approx(2 * 0.5 ** 12)
    assert c["overturned_by"] == E.OVERTURN[("M1", "M0")]
    assert c["endpoint_caveat"]


def test_continuous_contrast_carries_an_interval_and_a_direction_count():
    a = _arm("M1", _rows([1] * 20 + [0] * 76, phi_offset=0.05))
    b = _arm("Q", _rows([0] * 96))
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["phi_basin_mean"], "panel test")
    assert c["ci95"][0] <= c["mean_difference"] <= c["ci95"][1]
    assert c["n_better"] == 96 and c["n_worse"] == 0
    # every pair improves, so the tail is tiny - but it must never be printed or
    # stored as exactly zero
    assert 0.0 < c["p"] < 1e-10
    assert E.fmt_p(c["p"]).startswith(("1", "2", "3", "4", "5", "6", "7", "8", "9"))


def test_contrast_pairs_by_seed_and_ignores_unshared_seeds():
    a = _arm("M1", _rows([1, 1, 0, 0]))
    b = _arm("M0", {k: v for k, v in _rows([0, 0, 0, 0]).items() if k != 9103})
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["milestone4"], "panel test")
    assert c["n_paired"] == 3


def test_contrast_with_no_shared_seeds_is_unavailable_not_zero():
    a = _arm("M1", _rows([1, 1]))
    b = _arm("M0", {9200: dict(milestone4=0.0)})
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["milestone4"], "panel test")
    assert c["available"] is False


def test_holm_over_the_registered_primary_family_uses_every_contrast():
    # The family is one test per registered primary contrast, so it grew from 3 to 5
    # when `rand` and `M1_shuf` were registered. Holm is applied over the WHOLE family,
    # which is the point: adding controls makes each individual test harder to pass,
    # and that cost is paid deliberately rather than by dropping the controls.
    fam = [f"milestone4:{a}-vs-{b}" for a, b in E.PRIMARY_CONTRASTS]
    assert len(fam) == len(E.PRIMARY_CONTRASTS) == 5
    pvals = dict(zip(fam, [0.004, 0.03, 0.2, 0.5, 0.9]))
    out = E.holm(pvals)
    assert out[fam[0]]["family_size"] == len(E.PRIMARY_CONTRASTS)
    # 0.004 < 0.05/5 = 0.01 rejects; 0.03 > 0.05/4 = 0.0125 does not
    assert out[fam[0]]["reject"] is True and out[fam[1]]["reject"] is False


# --------------------------------------------------------------------------
# the driver: dry run
# --------------------------------------------------------------------------
def test_dry_run_writes_the_preregistration_and_nothing_else(tmp_path):
    rc = E.main(["--panel", "96", "--panel-start", "9100",
                 "--out-root", str(tmp_path)])
    assert rc == 0
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1 and runs[0].name.endswith("_dryrun")
    assert [p.name for p in runs[0].iterdir()] == ["preregistration.json"]


def test_preregistration_commits_the_panel_endpoints_and_power(tmp_path):
    E.main(["--panel", "96", "--panel-start", "9100", "--out-root", str(tmp_path)])
    doc = json.loads(next(next(tmp_path.iterdir()).glob("*.json")).read_text())
    assert doc["schema"] == E.SCHEMA_PREREG
    assert doc["panel"] == {"start": 9100, "count": 96, "seeds": [9100, 9195],
                            "committed_before_any_episode": True,
                            "rule": doc["panel"]["rule"]}
    assert [a["name"] for a in doc["arms"]] == list(E.ARM_ORDER)
    assert doc["arms"][0]["zero_residual"] is True
    assert doc["endpoint_hierarchy"]["primary"] == ["milestone4"]
    assert doc["endpoint_hierarchy"]["estimation"] == ["success"]
    assert doc["comparisons"]["primary"] == [list(c) for c in E.PRIMARY_CONTRASTS]
    assert doc["multiplicity"]["method"] == "Holm step-down"
    assert doc["power"]["binary_endpoint"]["is_estimation_endpoint_only"] is True
    assert doc["seed_guard"]["clean"] is True
    assert doc["task"] == "chain3_lr2" and doc["horizon"] == 750


def test_dry_run_refuses_an_overlapping_panel_before_writing_anything(tmp_path):
    with pytest.raises(SystemExit):
        E.main(["--panel", "96", "--panel-start", "8700",
                "--out-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_dry_run_and_execute_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        E.main(["--panel", "96", "--panel-start", "9100", "--dry-run", "--execute",
                "--out-root", str(tmp_path)])


def test_execute_refuses_missing_checkpoints_before_any_environment_step(tmp_path):
    with pytest.raises(SystemExit) as exc:
        E.main(["--panel", "96", "--panel-start", "9100", "--execute",
                "--out-root", str(tmp_path)])
    assert "missing" in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def test_panel_and_panel_start_have_no_defaults():
    with pytest.raises(SystemExit):
        E.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        E.build_parser().parse_args(["--panel", "96"])


def test_report_names_the_conditions_and_the_overturning_arm(tmp_path):
    a = _arm("M1", _rows([1] * 12 + [0] * 84))
    b = _arm("M0", _rows([0] * 96))
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["milestone4"], "panel 9100-9195")
    c["family"] = "primary"
    summary = {"task": "chain3_lr2", "mode": "executed", "panel": [9100, 9195, 96],
               "arms": {"M1": {"successes": 3, "n": 96, "rate": 3 / 96,
                               "cp95": list(E.clopper_pearson(3, 96))}},
               "contrasts": [c], "unavailable_endpoints": [],
               "preregistration": {"power": E.power_block(96)},
               "conditions": "frozen pi0.5; one panel; one init"}
    text = E.format_report(summary)
    assert "ESTIMATION endpoint" in text
    assert "discordant +12 -0" in text
    assert "overturned by:" in text
    assert "conditions:" in text
    assert "endpoint notes" in text
    assert E.ENDPOINT_BY_KEY["milestone4"].caveat in text
    assert "goal achieved" not in text.lower()
    for banned in ("breakthrough", "confirmed", "decisive", "achieved"):
        assert banned not in text.lower()


def test_terminal_success_lines_are_marked_as_not_a_test():
    a = _arm("M1", _rows([1] * 6 + [0] * 90))
    b = _arm("base", _rows([0] * 96))
    for arm in (a, b):
        for row in arm.per_seed.values():
            row["success"] = row["milestone4"]
    c = E.contrast(a, b, E.ENDPOINT_BY_KEY["success"], "panel 9100-9195")
    c["family"] = "estimation"
    summary = {"task": "chain3_lr2", "mode": "executed", "panel": [9100, 9195, 96],
               "arms": {}, "contrasts": [c], "unavailable_endpoints": [],
               "preregistration": {"power": E.power_block(96)},
               "conditions": "frozen pi0.5"}
    text = E.format_report(summary)
    assert "this p is not a registered test at this n" in text


def test_power_block_records_the_holm_adjusted_detectable_effect():
    pw = E.power_block(96)
    plain = pw["ordinal_and_continuous_endpoints"]["cohen_dz"]
    holm_step = pw["ordinal_and_continuous_at_holm_first_step"]["cohen_dz"]
    assert holm_step > plain                    # a smaller alpha needs a bigger effect
    assert holm_step == pytest.approx(0.349, abs=2e-3)


# --------------------------------------------------------------------------
# the driver: a full four-arm pass with the deployment subprocess stubbed out
# --------------------------------------------------------------------------
FAKE_ARM = """
import json, sys
from pathlib import Path
tag, seeds, m4 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
d = Path(sys.argv[4]) / ("chain3_lr2_" + tag + "_stamp")
d.mkdir(parents=True)
rows, prog = [], []
for i in range(seeds):
    s = 9100 + i
    hit = i < m4
    ev = {"0": 70, "1": 140, "2": 210, "3": 260}
    if hit:
        ev["4"] = 460
    rows.append({"seed": s, "success": bool(i < m4 // 4), "steps": 750})
    prog.append({"seed": s, "events": ev, "phi_basin_mean": 2.0 + 0.1 * hit,
                 "d_min_final_atom": 0.10 if hit else 0.50})
(d / "summary.json").write_text(json.dumps({"episodes": rows}))
(d / "progress.json").write_text(json.dumps({"episodes": prog}))
print("done -> " + str(d))
"""


def test_full_four_arm_pass_pairs_holm_and_reports(tmp_path, monkeypatch):
    """No simulator: each arm is a trivial subprocess writing a synthetic sidecar.

    This exercises exactly the path that would otherwise only run under the
    simulator - return-code checking, output-dir resolution from the arm's own
    stdout, seed pairing across four arms, Holm over the registered families, and
    the two artifacts.
    """
    script = tmp_path / "fake_arm.py"
    script.write_text(FAKE_ARM)
    for name in ("m1", "m0", "m0cont", "q", "m1shuf", "rand"):
        (tmp_path / f"{name}.pt").write_bytes(b"synthetic-checkpoint-" + name.encode())
    m4_by_arm = {"base": 5, "M0": 6, "M1": 24, "Q": 7, "M1_shuf": 6, "rand": 8, "M0_cont": 6}
    runs = tmp_path / "runs"

    def fake_cmd(arm, actor, panel, panel_start, tag, task=E.TASK, start=0):
        return [sys.executable, str(script), tag, str(panel),
                str(m4_by_arm[arm]), str(runs)]

    monkeypatch.setattr(E, "arm_command", fake_cmd)
    out_root = tmp_path / "out"
    rc = E.main(["--panel", "96", "--panel-start", "9100", "--execute",
                 "--no-check-matched", "--out-root", str(out_root),
                 "--m1", str(tmp_path / "m1.pt"), "--m0", str(tmp_path / "m0.pt"),
                 "--q", str(tmp_path / "q.pt"),
                 "--m1-shuf", str(tmp_path / "m1shuf.pt"),
                 "--rand", str(tmp_path / "rand.pt"),
                "--m0-cont", str(tmp_path / "m0cont.pt")])
    assert rc == 0
    run_dir = next(out_root.iterdir())
    got = {p.name for p in run_dir.iterdir()}
    assert {"preregistration.json", "summary.json", "report.txt"} <= got
    assert {f"{a}.log" for a in E.ARM_ORDER} <= got

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema"] == E.SCHEMA_SUMMARY
    assert summary["preregistration_sha256"]
    assert set(summary["arms"]) == set(E.ARM_ORDER)
    assert summary["arms"]["M1"]["n"] == 96
    assert summary["unavailable_endpoints"] == []
    # the run-level block is present and names what it could NOT check: this
    # fake arm's summary.json carries the episode rows and nothing else
    assert summary["run_matched"]["hard_failures"] == []
    assert "eval_labels" in summary["run_matched"]["not_verifiable"]
    assert summary["degenerate_arms"]["hard_failures"] == []

    # M1-vs-M0 is now CONTEXT: it confounds the transition with 75% more optimiser
    # steps. The Holm-corrected dynamics contrast is M1-vs-M0_cont.
    c = next(x for x in summary["contrasts"] if x["contrast"] == "milestone4:M1-vs-M0")
    assert c["a_count"] == 24 and c["b_count"] == 6
    assert c["discordant"] == {**c["discordant"], "b": 18, "c": 0}
    key = "milestone4:M1-vs-M0_cont"
    c = next(x for x in summary["contrasts"] if x["contrast"] == key)
    assert c["holm"]["family_size"] == len(E.PRIMARY_CONTRASTS)
    assert summary["holm"]["primary"][key]["p_adjusted"] <= 1.0

    # every primary contrast carries its interval, its conditions and its refutation
    for x in summary["contrasts"]:
        if x.get("available"):
            assert x["panel"].startswith("panel 9100-9195")
            assert x["overturned_by"]
    text = (run_dir / "report.txt").read_text()
    assert "ESTIMATION endpoint" in text and "overturned by:" in text


def test_the_driver_aborts_when_one_arm_fails_instead_of_reporting_three(
        tmp_path, monkeypatch):
    for name in ("m1", "m0", "m0cont", "q", "m1shuf", "rand"):
        (tmp_path / f"{name}.pt").write_bytes(b"x")

    def fake_cmd(arm, actor, panel, panel_start, tag, task=E.TASK, start=0):
        if arm == "M1":
            return [sys.executable, "-c", "import sys; sys.exit(7)"]
        return [sys.executable, "-c", "print('-> /nonexistent')"]

    monkeypatch.setattr(E, "arm_command", fake_cmd)
    with pytest.raises(E.ArmFailed):
        E.main(["--panel", "8", "--panel-start", "9100", "--execute",
                "--no-check-matched", "--out-root", str(tmp_path / "out"),
                "--m1", str(tmp_path / "m1.pt"), "--m0", str(tmp_path / "m0.pt"),
                "--q", str(tmp_path / "q.pt"),
                "--m1-shuf", str(tmp_path / "m1shuf.pt"),
                "--rand", str(tmp_path / "rand.pt"),
                "--m0-cont", str(tmp_path / "m0cont.pt")])


# --------------------------------------------------------------------------
# the executor contract: instrumentation, and what the arms actually ran with
# --------------------------------------------------------------------------
def test_every_arm_is_invoked_with_the_evaluation_instrumentation():
    """Without --eval-labels the executor writes events={} for every episode.

    run_v206_belief_residual_deploy.py:396-404 builds no automaton unless the
    flag is set, so events, stage_reached and last_progress_step are 0 - values,
    not nulls. The driver would read them as measurements and report the primary
    endpoint as 0/96 vs 0/96 on every contrast. Without --fixed-horizon an
    episode stops at success (lcwm/loho.py:70) and the boundary count carries the
    outcome, which biases every per-boundary average by the arm's own success.
    """
    for arm in E.ARM_ORDER:
        cmd = E.arm_command(arm, Path("/x/a.pt"), 96, 9100, "t")
        assert "--eval-labels" in cmd, arm
        assert "--fixed-horizon" in cmd, arm


def test_a_sidecar_without_eval_labels_reads_as_zero_on_every_endpoint(tmp_path):
    """The exact bytes run_v206 writes without --eval-labels, read back."""
    d = tmp_path / "arm"
    d.mkdir()
    rows = [{"seed": 9100 + i, "success": False, "success_step": None, "steps": 750,
             "chunks": 75, "events": {}, "stage_reached": 0,
             "last_progress_step": 0, "idle_tail": 0} for i in range(4)]
    (d / "summary.json").write_text(json.dumps({"episodes": rows}))
    got = E.read_arm_endpoints(d)
    # nothing is None, so nothing is reported unavailable: this is why the flag
    # is passed by arm_command and verified again after the run
    assert got["per_seed"][9100]["stage"] == 0.0
    assert got["unavailable"] == ["phi_basin_mean", "d_min_final_atom"]

    class _Arm:
        name, per_seed = "M1", got["per_seed"]

    hard = E.degenerate_arms([_Arm()], {"per_arm": {"M1": {}}})["hard_failures"]
    assert len(hard) == 1 and "--eval-labels" in hard[0]
    ok = E.degenerate_arms([_Arm()], {"per_arm": {"M1": {"eval_labels": True}}})
    assert ok["hard_failures"] == [] and ok["flagged"][0]["arm"] == "M1"


def _meta(**kw):
    base = {"scale": 0.05, "condition": "raw", "latent_dim": 8217,
            "rich_latent": True, "use_action": True, "actor_params": 2_200_000,
            "fixed_horizon": True, "eval_labels": True, "wm_source": "rawwm",
            "panel": [9100, 9195, 96], "zero_residual": False,
            "residual_start_chunk": 26, "mean_abs_delta_panel": 0.44,
            "max_abs_delta_panel": 0.80,
            "norm_digest": {"mu": "aa", "sd": "bb", "bmu": "cc", "bsd": "dd"}}
    base.update(kw)
    return base


def test_run_matched_accepts_a_matched_set_and_records_what_it_could_not_check():
    metas = {"base": _meta(zero_residual=True), "M0": _meta(), "M1": _meta(),
             "Q": _meta()}
    got = E.check_run_matched(metas, 9100, 96)
    assert got["hard_failures"] == []
    assert got["compared"] == [n for n in E.ARM_ORDER
                               if n != "base" and n in metas]
    assert got["not_verifiable"] == []
    blind = {"M1": {"norm_digest": {}}, "M0": {"norm_digest": {}}}
    assert "scale" in E.check_run_matched(blind, 9100, 96)["not_verifiable"]


def test_run_matched_catches_each_way_the_arms_can_stop_being_one_experiment():
    def fails(metas):
        return " | ".join(E.check_run_matched(metas, 9100, 96)["hard_failures"])

    good = {"base": _meta(zero_residual=True), "M0": _meta(), "M1": _meta()}
    assert fails(good) == ""
    # the panel actually stepped is not the panel registered
    bad = dict(good, M1=_meta(panel=[7600, 7695, 96]))
    assert "ran panel" in fails(bad)
    # the base arm did not zero its residual / a treated arm did
    assert "zero_residual" in fails(dict(good, base=_meta(zero_residual=False)))
    assert "zero_residual" in fails(dict(good, M1=_meta(zero_residual=True)))
    # the instrumentation the endpoints are read out of was off
    assert "--eval-labels" in fails(dict(good, M0=_meta(eval_labels=False)))
    assert "--fixed-horizon" in fails(dict(good, M0=_meta(fixed_horizon=False)))
    # the arms deployed under different latent normalisers, or different capacity
    other = _meta()
    other["norm_digest"] = dict(other["norm_digest"], sd="ZZ")
    assert "latent normaliser sd" in fails(dict(good, M1=other))
    assert "actor_params" in fails(dict(good, Q=_meta(actor_params=17)))
    assert "latent_dim" in fails(dict(good, Q=_meta(latent_dim=2073)))


def test_run_matched_does_not_fail_on_the_belief_normaliser():
    """bmu/bsd are produced BY the transition, so M1 and M0 may differ there."""
    m0 = _meta()
    m0["norm_digest"] = dict(m0["norm_digest"], bmu="x1", bsd="x2")
    got = E.check_run_matched({"M1": _meta(), "M0": m0}, 9100, 96)
    assert got["hard_failures"] == []
    assert {e["field"] for e in got["expected_differences"]} == {
        "norm_digest.bmu", "norm_digest.bsd"}


def test_read_run_meta_reads_the_executors_own_record(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(_meta(zero_residual=True)))
    got = E.read_run_meta(d)
    assert got["panel"] == [9100, 9195, 96] and got["zero_residual"] is True
    assert got["norm_digest"]["mu"] == "aa" and got["eval_labels"] is True


# --------------------------------------------------------------------------
# the joins that produce a number instead of an error
# --------------------------------------------------------------------------
def test_phi_basin_is_joined_against_events_carried_by_a_different_file(tmp_path):
    """The realistic split: phi/t in progress.json, events in summary.json.

    Deriving each file's row separately and then merging them computes the basin
    against last_progress_step = 0, i.e. every boundary, and files it under the
    endpoint named "mean Phi' over the stall basin".
    """
    d = tmp_path / "arm"
    d.mkdir()
    (d / "progress.json").write_text(json.dumps({"episodes": [
        {"seed": 9100, "phi": [1.0, 2.0, 3.0, 4.0], "phi_t": [0, 100, 300, 500]}]}))
    (d / "summary.json").write_text(json.dumps({"episodes": [
        {"seed": 9100, "success": False,
         "events": {"0": 70, "1": 140, "2": 210, "3": 260}}]}))
    row = E.read_arm_endpoints(d)["per_seed"][9100]
    assert row["last_progress_step"] == 260.0
    assert row["phi_basin_mean"] == pytest.approx(3.5)     # not 3.0


def test_a_phi_series_with_no_time_base_is_unavailable_not_its_last_boundary():
    rec = {"phi": [1.0, 2.0, 3.0, 4.0]}
    assert E.derive_phi_basin(rec, 260.0) is None
    # the one reconstruction allowed: the full-horizon boundary grid
    full = {"phi": [float(i) for i in range(E.N_BOUNDARIES)]}
    assert E.N_BOUNDARIES == 76
    got = E.derive_phi_basin(full, 700.0)                  # boundaries t = 710..750
    assert got == pytest.approx((71 + 72 + 73 + 74 + 75) / 5)
    assert E.derive_phi_basin({"phi": [1.0], "phi_t": None}, None) is None


def test_a_phi_series_and_its_time_base_must_be_the_same_length():
    with pytest.raises(E.ArmFailed):
        E.derive_phi_basin({"phi": [1.0, 2.0, 3.0], "phi_t": [0, 10]}, 0.0)


def test_a_seed_appearing_twice_in_one_sidecar_is_refused(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "progress.json").write_text(json.dumps({"episodes": [
        {"seed": 9100, "success": True, "events": {"0": 70, "4": 460}},
        {"seed": 9100, "success": False, "events": {"0": 70}}]}))
    with pytest.raises(E.ArmFailed) as exc:
        E.read_arm_endpoints(d)
    assert "more than one record" in str(exc.value)


def test_a_binary_endpoint_is_not_coerced_from_a_string():
    """float(bool("False")) is 1.0 - the failure mode this refuses."""
    assert E._as_binary(True, "success") == 1.0
    assert E._as_binary(0, "success") == 0.0
    for bad in ("False", "0", "", None, 2, 0.5):
        with pytest.raises(E.ArmFailed):
            E._as_binary(bad, "success")


# --------------------------------------------------------------------------
# claims the driver may not make without having checked them
# --------------------------------------------------------------------------
def test_an_unreadable_transition_is_not_reported_as_a_difference():
    def fp(**fields):
        return {"fields": dict(fields), "actor_param_shapes": {"w": [4, 4]},
                "actor_n_params": 16, "transition_key": None,
                "transition_sha256": fields.pop("_sha", None)}

    both_blind = {"M1": fp(zdim=8217), "M0": fp(zdim=8217)}
    got = E.check_matched(both_blind)
    assert got["m1_m0_transition_identical"] is None
    assert "NOT" in got["m1_m0_transition_comparison"]
    assert got["hard_failures"] == []                      # it cannot fail a check
    assert "transition_sha256" in got["not_verifiable"]

    one = {"M1": dict(fp(zdim=8217), transition_sha256="a"),
           "M0": fp(zdim=8217)}
    assert "transition_sha256" in E.check_matched(one)["hard_failures"]

    same = {"M1": dict(fp(zdim=8217), transition_sha256="a"),
            "M0": dict(fp(zdim=8217), transition_sha256="a")}
    got = E.check_matched(same)
    assert got["m1_m0_transition_identical"] is True
    assert "transition_sha256" in got["hard_failures"]

    diff = {"M1": dict(fp(zdim=8217), transition_sha256="a"),
            "M0": dict(fp(zdim=8217), transition_sha256="b")}
    got = E.check_matched(diff)
    assert got["m1_m0_transition_comparison"] == "differ"
    assert got["hard_failures"] == []


def test_a_field_absent_from_every_checkpoint_is_not_a_passed_check():
    fp = {"fields": {"zdim": 8217}, "actor_param_shapes": {}, "actor_n_params": 0,
          "transition_key": None, "transition_sha256": None}
    got = E.check_matched({"M1": dict(fp), "M0": dict(fp)})
    assert "scale" in got["not_verifiable"] and "dims" in got["not_verifiable"]


def test_holm_divides_by_the_registered_family_not_the_tests_that_ran():
    """An endpoint missing from the sidecar must not loosen the correction."""
    pv = {"stage:M1-vs-M0": 0.01, "stage:M1-vs-Q": 0.02}
    shrunk = E.holm(pv)
    registered = E.holm(pv, E.ALPHA, 9)
    assert shrunk["stage:M1-vs-M0"]["alpha_adjusted"] == pytest.approx(0.025)
    assert registered["stage:M1-vs-M0"]["alpha_adjusted"] == pytest.approx(0.05 / 9)
    assert registered["stage:M1-vs-M0"]["p_adjusted"] == pytest.approx(0.09)
    assert registered["stage:M1-vs-M0"]["n_tests_run"] == 2
    with pytest.raises(ValueError):
        E.holm(pv, E.ALPHA, 1)


def test_the_registered_family_sizes_follow_the_endpoint_tiers():
    # The families are (endpoints in a tier) x (primary contrasts). Both grew when
    # `rand` and `M1_shuf` were registered, so the sizes are asserted against the
    # constants rather than against the numbers they happened to have at four arms.
    prim = len([e for e in E.ENDPOINTS if e.tier == "primary"])
    sec = len([e for e in E.ENDPOINTS if e.tier == "secondary"])
    assert prim == 1
    assert sec == 4          # stage, last_progress_step, phi_basin_mean, d_min_final_atom
    assert len(E.PRIMARY_CONTRASTS) == 5
    assert prim * len(E.PRIMARY_CONTRASTS) == 5
    assert sec * len(E.PRIMARY_CONTRASTS) == 20


def test_the_power_statement_and_the_module_docstring_agree():
    pb = E.power_block(96)["binary_endpoint"]
    assert pb["min_treated_count_two_sided"] == 8
    assert pb["min_treated_count_one_sided"] == 7
    assert pb["rate_multiple_two_sided"] == pytest.approx(4.0)
    # the unpaired Fisher figure the executor's own docstring records
    assert pb["min_treated_count_unpaired_fisher_one_sided"] == 9
    doc = E.__doc__
    assert "8/96, 4.0x" in doc and "7/96 against a 2/96 base, a 3.5x" in doc


def test_the_driver_refuses_a_panel_the_arms_did_not_actually_run(
        tmp_path, monkeypatch):
    """Four arms agreeing with each other is not four arms on the panel.

    The fake arm ignores --panel-start and writes 9100+, as an executor that fell
    back to its own default panel would. The arms pair perfectly with one another
    and the run is still void.
    """
    script = tmp_path / "fake_arm.py"
    script.write_text(FAKE_ARM)
    for name in ("m1", "m0", "m0cont", "q", "m1shuf", "rand"):
        (tmp_path / f"{name}.pt").write_bytes(b"x")
    runs = tmp_path / "runs"

    def fake_cmd(arm, actor, panel, panel_start, tag, task=E.TASK, start=0):
        return [sys.executable, str(script), tag, str(panel), "5", str(runs)]

    monkeypatch.setattr(E, "arm_command", fake_cmd)
    with pytest.raises(SystemExit) as exc:
        E.main(["--panel", "96", "--panel-start", "9200", "--execute",
                "--no-check-matched", "--out-root", str(tmp_path / "out"),
                "--m1", str(tmp_path / "m1.pt"), "--m0", str(tmp_path / "m0.pt"),
                "--q", str(tmp_path / "q.pt"),
                "--m1-shuf", str(tmp_path / "m1shuf.pt"),
                "--rand", str(tmp_path / "rand.pt"),
                "--m0-cont", str(tmp_path / "m0cont.pt")])
    assert "did not run the pre-registered panel" in str(exc.value)


# --------------------------------------------------------------------------
# a four-arm pass whose stub writes the executor's real summary.json shape
# --------------------------------------------------------------------------
FAKE_ARM_RUNMETA = """
import json, sys
from pathlib import Path
tag, seeds, m4 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root, zero = sys.argv[4], sys.argv[5]
d = Path(root) / ("chain3_lr2_" + tag + "_stamp")
d.mkdir(parents=True)
rows, prog = [], []
for i in range(seeds):
    s = 9100 + i
    hit = i < m4
    ev = {"0": 70, "1": 140, "2": 210, "3": 260}
    if hit:
        ev["4"] = 460
    rows.append({"seed": s, "success": bool(i < m4 // 4), "steps": 750})
    # 76 boundaries and no explicit t: the full-horizon grid, reconstructed
    prog.append({"seed": s, "events": ev, "phi": [2.0 + 0.1 * hit] * 76,
                 "d_min_final_atom": 0.10 if hit else 0.50})
(d / "summary.json").write_text(json.dumps({
    "episodes": rows, "panel": [9100, 9100 + seeds - 1, seeds],
    "zero_residual": zero == "1", "eval_labels": True, "fixed_horizon": True,
    "scale": 0.05, "condition": "raw", "latent_dim": 8217, "rich_latent": True,
    "use_action": True, "actor_params": 2200000, "wm_source": "rawwm",
    "residual_start_chunk": 26,
    "mean_abs_delta_panel": (0.0 if zero == "1" else 0.44),
    "max_abs_delta_panel": (0.0 if zero == "1" else 0.80),
    "norm_digest": {"mu": "aa", "sd": "bb", "bmu": "cc", "bsd": "dd"}}))
(d / "progress.json").write_text(json.dumps({"episodes": prog}))
print("done -> " + str(d))
"""


def _run_with_runmeta(tmp_path, monkeypatch, zero_for, panel=24):
    script = tmp_path / "fake_arm_meta.py"
    script.write_text(FAKE_ARM_RUNMETA)
    for name in ("m1", "m0", "m0cont", "q", "m1shuf", "rand"):
        (tmp_path / f"{name}.pt").write_bytes(b"x" + name.encode())
    m4_by_arm = {"base": 1, "M0": 2, "M1": 8, "Q": 2, "M1_shuf": 2, "rand": 3, "M0_cont": 2}
    runs = tmp_path / "runs"

    def fake_cmd(arm, actor, panel_, panel_start, tag, task=E.TASK, start=0):
        return [sys.executable, str(script), tag, str(panel_),
                str(m4_by_arm[arm]), str(runs),
                "1" if arm in zero_for else "0"]

    monkeypatch.setattr(E, "arm_command", fake_cmd)
    out_root = tmp_path / "out"
    rc = E.main(["--panel", str(panel), "--panel-start", "9100", "--execute",
                 "--no-check-matched", "--out-root", str(out_root),
                 "--m1", str(tmp_path / "m1.pt"), "--m0", str(tmp_path / "m0.pt"),
                 "--q", str(tmp_path / "q.pt"),
                 "--m1-shuf", str(tmp_path / "m1shuf.pt"),
                 "--rand", str(tmp_path / "rand.pt"),
                "--m0-cont", str(tmp_path / "m0cont.pt")])
    return rc, json.loads((next(out_root.iterdir()) / "summary.json").read_text())


def test_a_matched_set_passes_the_run_level_check_end_to_end(tmp_path, monkeypatch):
    rc, summary = _run_with_runmeta(tmp_path, monkeypatch, zero_for={"base"})
    assert rc == 0
    rm = summary["run_matched"]
    assert rm["hard_failures"] == [] and rm["not_verifiable"] == []
    assert rm["per_arm"]["base"]["zero_residual"] is True
    assert rm["per_arm"]["M1"]["zero_residual"] is False
    # phi came back as a 76-boundary series with no t and was still evaluated
    assert summary["unavailable_endpoints"] == []
    c = next(x for x in summary["contrasts"]
             if x["contrast"] == "phi_basin_mean:M1-vs-M0_cont")
    assert c["available"]
    # secondary family = (secondary endpoints) x len(PRIMARY_CONTRASTS)
    n_sec = len([e for e in E.ENDPOINTS if e.tier == "secondary"])
    assert c["holm"]["family_size"] == n_sec * len(E.PRIMARY_CONTRASTS)


def test_the_run_level_check_aborts_a_base_arm_that_did_not_zero(tmp_path,
                                                                 monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_with_runmeta(tmp_path, monkeypatch, zero_for=set())
    assert "zero_residual" in str(exc.value)
    assert "refusing to report" in str(exc.value)


# --- the deployment schedule gate --------------------------------------------------
# The interface is trained only on boundaries at or after the collector's
# --residual-start-chunk. If the driver deploys it from chunk 0 it runs at states it
# never saw, and the 2026-09-12 basin calibration measured that an ungated sustained
# residual is what damages the first two sub-tasks. The gate must reach EVERY arm,
# base included, or it separates the arms by something other than the transition.

def test_every_arm_receives_the_same_schedule_gate() -> None:
    for arm in ("M1", "M0", "Q", "M1_shuf", "base"):
        cmd = E.arm_command(arm, Path("/tmp/a.pt"), 288, 9100, f"t_{arm}",
                            E.TASK, 26)
        assert "--residual-start-chunk" in cmd
        assert cmd[cmd.index("--residual-start-chunk") + 1] == "26"


def test_the_schedule_gate_defaults_to_zero_and_is_explicit() -> None:
    cmd = E.arm_command("M1", Path("/tmp/a.pt"), 288, 9100, "t", E.TASK)
    # present even at 0, so a deployment can never be silently ungated
    assert "--residual-start-chunk" in cmd
    assert cmd[cmd.index("--residual-start-chunk") + 1] == "0"


def test_the_base_arm_still_carries_zero_residual_alongside_the_gate() -> None:
    cmd = E.arm_command("base", Path("/tmp/a.pt"), 288, 9100, "t", E.TASK, 26)
    assert "--zero-residual" in cmd
    assert "--eval-labels" in cmd and "--fixed-horizon" in cmd


# --- endpoint direction ------------------------------------------------------------
# `d_min_final_atom` is registered with direction=-1: a CLOSER approach is better.
# Before this was applied, an arm that approached the object more closely printed as
# "worse" and its bootstrap interval carried the wrong sign.

def test_a_smaller_is_better_endpoint_reports_closer_as_better():
    ep = E.ENDPOINT_BY_KEY["d_min_final_atom"]
    assert ep.direction == -1
    # arm A gets 0.10 m, arm B 0.50 m: A is closer, so A is BETTER
    a = _arm("M1", {s: {"d_min_final_atom": 0.10} for s in range(9100, 9108)})
    b = _arm("M0", {s: {"d_min_final_atom": 0.50} for s in range(9100, 9108)})
    c = E.contrast(a, b, ep, "panel test")
    assert c["n_better"] == 8 and c["n_worse"] == 0
    assert c["oriented"] is True
    assert c["raw_mean_difference"] == pytest.approx(-0.40)
    assert c["mean_difference"] == pytest.approx(+0.40)


def test_a_larger_is_better_endpoint_is_left_unoriented():
    ep = E.ENDPOINT_BY_KEY["last_progress_step"]
    assert ep.direction == +1
    a = _arm("M1", {s: {"last_progress_step": 460.0} for s in range(9100, 9104)})
    b = _arm("M0", {s: {"last_progress_step": 260.0} for s in range(9100, 9104)})
    c = E.contrast(a, b, ep, "panel test")
    assert c["n_better"] == 4 and c["oriented"] is False
    assert c["mean_difference"] == pytest.approx(c["raw_mean_difference"])


# --- the untrained arm is matched structurally, not by training provenance ---------
# `rand` has no phi head, no model pair, no advantage and no optimiser budget, because
# being untrained is its defining property. Comparing those fields against the trained
# arms aborted the round on the first real run. It must still be compared on everything
# that makes it the same kind of function in the same space.

def _fp(transition="t0", **fields):
    base = {"dims": [8217, 10, 7], "zdim": 8217, "scale": 0.8, "condition": "raw",
            "task": "chain3_lr2"}
    base.update(fields)
    return {"fields": base, "actor_param_shapes": [[256, 8217], [70, 256]],
            "actor_n_params": 2200000, "transition_sha256": transition}


def test_the_untrained_arm_is_exempt_from_training_provenance_only():
    trained = _fp(horizon=10, actor_epochs=3000, advantage="predicted",
                  phi_head_digest="H", model_pair_sha256="P", phi_readout="head")
    # M1 and M0 must carry DIFFERENT transitions - that is the round's only
    # registered difference, and identical ones are correctly a hard failure
    m0 = dict(trained, transition_sha256="t1")
    fps = {"M1": trained, "M0": m0, "rand": _fp()}
    got = E.check_matched(fps)
    assert got["hard_failures"] == []
    assert "rand" in got["untrained_exempt"]
    assert "horizon" in got["untrained_exempt"]["rand"]
    assert got["trained_compared"] == ["M0", "M1"]
    assert "rand" in got["compared"]


def test_the_untrained_arm_is_still_structurally_checked():
    trained = _fp(horizon=10, phi_head_digest="H", model_pair_sha256="P")
    bad_rand = _fp(scale=0.05)              # a different trust region entirely
    got = E.check_matched({"M1": trained,
                           "M0": dict(trained, transition_sha256="t1"),
                           "rand": bad_rand})
    assert "scale" in got["hard_failures"]


def test_a_trained_arm_with_a_foreign_phi_head_still_aborts():
    a = _fp(horizon=10, phi_head_digest="H", model_pair_sha256="P")
    b = _fp(transition="t1", horizon=10, phi_head_digest="DIFFERENT",
            model_pair_sha256="P")
    got = E.check_matched({"M1": a, "M0": b, "rand": _fp()})
    assert "phi_head_digest" in got["hard_failures"]
