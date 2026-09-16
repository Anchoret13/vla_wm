#!/usr/bin/env python
"""Mechanical CPU-only checks on scripts/build_v241_model_pair.py.

WHAT THIS MEASURES.  Four properties of the builder that a later between-arm
comparison silently depends on, each on synthetic fixtures, none of which import
LIBERO, touch a GPU, or read a real tape:

  1. the supplied-Phi' join is by (episode, t) and not by row position, every tape
     row must find exactly one label row, and the extra terminal label row that
     collect_v250_chain3_round.py always writes (76 boundaries against 75
     transitions) is permitted;
  2. --phi-labels actually changes the fitted potential and is recorded in the
     output provenance, and the run that uses it reads no outcome field at all;
  3. an 8217-d tape - the rich-latent width the chain3 round collects - flows
     through the loader, the normaliser, both heads and the transition unchanged;
  4. the seed-clustered split puts every episode of an environment seed wholly on
     one side, over many seeds and holdout fractions.

WHAT WOULD OVERTURN THE THING THEY GUARD.  If test_join_is_by_key_not_position
passed while the loader aligned positionally, a label artifact whose rows are not
already sorted by (episode, t) would attach the wrong target to every row and
nothing downstream would show it.  If matched budgets failed, an updated-vs-stale
difference would be attributable to optimiser steps rather than to data.

PRE-REGISTERED: fixture shapes and seeds are fixed constants in this file; no test
selects among builder configurations after reading a result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pytest  # noqa: E402
import torch  # noqa: E402

import build_v241_model_pair as B  # noqa: E402
from lcwm.v250_progress import APPROACH_REFERENCE, tape_progress  # noqa: E402

TASK = "synthetic_lr2"
C = 2
ADIM = 3
OBJECT_NAMES = ["obj_a", "obj_b", "basket_1"]
GOAL_ATOMS = [
    ["in", "obj_a", "basket_1_contain_region"],
    ["in", "obj_b", "basket_1_contain_region"],
]
#: Per-boundary EEF->obj_a distances, in metres.  All are above NEAR_DISTANCE and
#: below APPROACH_REFERENCE, so every boundary stays in the approach branch and
#: Phi' = 0.2 * (1 - d / 0.60) is strictly decreasing in d and distinct per row.
DISTANCES = (0.55, 0.48, 0.41, 0.34, 0.27, 0.20, 0.13)


def write_tape(
    directory: Path,
    episodes: dict[int, int],
    zdim: int,
    n_chunks: int,
    seed: int,
    sidecar: str = "summary",
    outcome_fields: bool = True,
) -> Path:
    """One synthetic tape plus its per-episode record sidecar.

    ``episodes`` maps episode idx -> environment seed.  z/z_next are built from a
    single per-episode state path, so the loader's continuity check sees an exact
    match, and t advances by exactly C.
    """
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    z, u, z_next, ep_ids, times = [], [], [], [], []
    for eid in sorted(episodes):
        states = torch.randn(n_chunks + 1, zdim, generator=generator)
        z.append(states[:-1])
        z_next.append(states[1:])
        u.append(torch.randn(n_chunks, C, ADIM, generator=generator))
        ep_ids.extend([eid] * n_chunks)
        times.extend([k * C for k in range(n_chunks)])
    torch.save(
        {
            "z": torch.cat(z),
            "u": torch.cat(u),
            "z_next": torch.cat(z_next),
            "episode": torch.tensor(ep_ids),
            "t": torch.tensor(times),
            "task": TASK,
            "c": C,
            "latent_dim": zdim,
        },
        directory / "tape.pt",
    )
    records = []
    for eid, env_seed in sorted(episodes.items()):
        record = {"idx": eid, "seed": env_seed, "steps": n_chunks * C}
        if outcome_fields:
            record["success"] = False
            record["success_step"] = None
        records.append(record)
    if sidecar == "summary":
        payload, name = {"task": TASK, "episode_records": records}, "summary.json"
    else:
        payload, name = {"task": TASK, "episodes": records}, "outcome.json"
    (directory / name).write_text(json.dumps(payload, indent=2))
    return directory / "tape.pt"


def write_labels(
    directory: Path,
    episode_ids: list[int],
    n_chunks: int,
    shuffle: bool = True,
    t_offset: int = 0,
    duplicate: bool = False,
    drop_episode: int | None = None,
) -> Path:
    """A labels.pt in the collect_v250_chain3_round.py shape.

    Rows are deliberately stored out of (episode, t) order unless ``shuffle`` is
    off: a positional join would then attach the wrong Phi' to every row, which is
    exactly what test_join_is_by_key_not_position looks for.
    """
    directory.mkdir(parents=True, exist_ok=True)
    eef, obj, bits, eps, times = [], [], [], [], []
    for eid in episode_ids:
        if eid == drop_episode:
            continue
        for k in range(n_chunks + 1):  # one more boundary than transitions
            d = DISTANCES[(k + eid) % len(DISTANCES)]
            q = torch.zeros(25)
            q[0] = d  # eef at (d, 0, 0), obj_a at the origin
            eef.append(q)
            positions = torch.zeros(3, 3)
            positions[1, 0] = 1.0  # obj_b parked far away
            positions[2, 0] = 2.0  # basket
            obj.append(positions)
            bits.append(torch.zeros(2, dtype=torch.bool))
            eps.append(eid)
            times.append(k * C + t_offset)
    if duplicate:
        eef.append(eef[0])
        obj.append(obj[0])
        bits.append(bits[0])
        eps.append(eps[0])
        times.append(times[0])
    order = list(range(len(eps)))
    if shuffle:
        order = order[1::2] + order[0::2][::-1]
    payload = {
        "eef_proprio": torch.stack([eef[i] for i in order]),
        "obj_pos": torch.stack([obj[i] for i in order]),
        "bits": torch.stack([bits[i] for i in order]),
        "episode": torch.tensor([eps[i] for i in order]),
        "t": torch.tensor([times[i] for i in order]),
        "object_names": list(OBJECT_NAMES),
        "goal_atoms": [list(atom) for atom in GOAL_ATOMS],
        "task": TASK,
        "c": C,
    }
    torch.save(payload, directory / "labels.pt")
    return directory / "labels.pt"


def build_round(
    root: Path,
    zdim: int = 16,
    n_chunks: int = 6,
    labels: bool = True,
    sidecar: str = "summary",
    outcome_fields: bool = True,
):
    """A two-role fixture: 8 base episodes over 4 seeds, 6 update over 3."""
    base_eps = {i: 900 + i // 2 for i in range(8)}
    update_eps = {i: 950 + i // 2 for i in range(6)}
    base = write_tape(root / "base", base_eps, zdim, n_chunks, seed=11,
                      sidecar=sidecar, outcome_fields=outcome_fields)
    update = write_tape(root / "update", update_eps, zdim, n_chunks, seed=12,
                        sidecar=sidecar, outcome_fields=outcome_fields)
    if labels:
        write_labels(root / "base", sorted(base_eps), n_chunks)
        write_labels(root / "update", sorted(update_eps), n_chunks)
    return base, update


def expected_phi(labels_path: Path) -> dict[tuple[int, int], float]:
    labels = torch.load(labels_path, weights_only=False, map_location="cpu")
    progress = tape_progress(labels)
    return {
        (int(e), int(t)): float(v)
        for e, t, v in zip(
            progress["episode"].tolist(), progress["t"].tolist(), progress["phi"].tolist()
        )
    }


# --------------------------------------------------------------- the (episode, t) join


def test_join_is_by_key_not_position(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 16, 6, seed=1)
    labels = write_labels(tmp_path / "d", [0, 1], 6, shuffle=True)
    table, records = B.load_phi_labels([labels], TASK)
    episodes, _record, _dims, _mismatch = B.load_tape(
        tape, "base", TASK, None, table[str(tmp_path / "d")], True
    )
    want = expected_phi(labels)
    assert records[0].rows == 14 and records[0].episodes == 2
    for ep in episodes:
        assert ep.phi is not None and len(ep.phi) == ep.n
        for i, step in enumerate(ep.t.tolist()):
            assert ep.phi[i].item() == pytest.approx(want[(ep.episode_id, int(step))])
    first = episodes[0]
    analytic = 0.2 * (1.0 - DISTANCES[0] / APPROACH_REFERENCE)
    assert first.phi[0].item() == pytest.approx(analytic, abs=1e-6)
    assert len({round(v.item(), 6) for v in first.phi}) == first.n


def test_join_accepts_the_extra_terminal_label_row(tmp_path):
    """75 transitions against 76 boundaries is the collector's normal output."""
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 8, 5, seed=2)
    labels = write_labels(tmp_path / "d", [0, 1], 5, shuffle=False)
    table, _ = B.load_phi_labels([labels], TASK)
    episodes, _r, _d, _m = B.load_tape(
        tape, "base", TASK, None, table[str(tmp_path / "d")], True
    )
    assert sum(ep.n for ep in episodes) == 10
    assert len(table[str(tmp_path / "d")]) == 12


def test_join_misalignment_raises(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 8, 5, seed=3)
    labels = write_labels(tmp_path / "d", [0, 1], 5, t_offset=1)
    table, _ = B.load_phi_labels([labels], TASK)
    with pytest.raises(ValueError, match="no Phi' label"):
        B.load_tape(tape, "base", TASK, None, table[str(tmp_path / "d")], True)


def test_join_missing_episode_raises(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 8, 5, seed=4)
    labels = write_labels(tmp_path / "d", [0, 1], 5, drop_episode=1)
    table, _ = B.load_phi_labels([labels], TASK)
    with pytest.raises(ValueError, match="no Phi' label"):
        B.load_tape(tape, "base", TASK, None, table[str(tmp_path / "d")], True)


def test_duplicate_label_row_raises(tmp_path):
    write_tape(tmp_path / "d", {0: 900}, 8, 5, seed=5)
    labels = write_labels(tmp_path / "d", [0], 5, duplicate=True)
    with pytest.raises(ValueError, match="duplicate label row"):
        B.load_phi_labels([labels], TASK)


def test_label_task_mismatch_raises(tmp_path):
    labels = write_labels(tmp_path / "d", [0], 5)
    with pytest.raises(ValueError, match="task"):
        B.load_phi_labels([labels], "another_task")


def test_phi_labels_accepts_a_directory(tmp_path):
    write_labels(tmp_path / "d", [0], 4)
    table, records = B.load_phi_labels([tmp_path / "d"], TASK)
    assert set(table) == {str(tmp_path / "d")}
    assert records[0].path.endswith("labels.pt")


def test_uncovered_head_tape_raises(tmp_path, monkeypatch):
    base, update = build_round(tmp_path, labels=False)
    write_labels(tmp_path / "update", list(range(6)), 6)
    monkeypatch.setattr(sys, "argv", cli(tmp_path, base, update, "uncovered")
                        + ["--phi-labels", str(tmp_path / "update" / "labels.pt")])
    with pytest.raises(ValueError, match="does not cover head tapes"):
        B.main()


def test_unmatched_phi_label_path_raises(tmp_path, monkeypatch):
    base, update = build_round(tmp_path, labels=True)
    stray = write_labels(tmp_path / "stray", [0], 6)
    monkeypatch.setattr(sys, "argv", cli(tmp_path, base, update, "stray")
                        + ["--phi-labels", str(stray)])
    with pytest.raises(ValueError, match="match no loaded tape directory"):
        B.main()


def test_mixed_labelled_and_unlabelled_heads_raise(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 8, 5, seed=6)
    labels = write_labels(tmp_path / "d", [0, 1], 5)
    table, _ = B.load_phi_labels([labels], TASK)
    labelled, _r, _d, _m = B.load_tape(tape, "base", TASK, None, table[str(tmp_path / "d")])
    plain, _r2, _d2, _m2 = B.load_tape(tape, "base", TASK, None, None)
    mu = torch.zeros(8)
    sd = torch.ones(8)
    with pytest.raises(ValueError, match="mix supplied"):
        B.base_rows([labelled[0], plain[1]], mu, sd, 0.9, C)


# --------------------------------------------------------------- sidecars and outcomes


def test_outcome_json_sidecar_is_accepted(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 8, 5, seed=7, sidecar="outcome")
    episodes, record, _dims, _m = B.load_tape(tape, "base", TASK, None)
    assert record.records_key == "episodes"
    assert record.summary_path.endswith("outcome.json")
    assert {ep.env_seed for ep in episodes} == {900, 901}


def test_missing_sidecar_raises(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900}, 8, 5, seed=8)
    (tmp_path / "d" / "summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="no episode-record sidecar"):
        B.load_tape(tape, "base", TASK, None)


def test_outcome_free_sidecar_needs_phi_labels(tmp_path):
    """Without a target from labels, a sidecar with no 'success' must fail loudly."""
    tape = write_tape(
        tmp_path / "d", {0: 900}, 8, 5, seed=9, sidecar="outcome", outcome_fields=False
    )
    with pytest.raises(ValueError, match="lacks 'success'"):
        B.load_tape(tape, "base", TASK, None, None, True)
    episodes, record, _d, _m = B.load_tape(tape, "base", TASK, None, None, False)
    assert record.outcome_fields_read is False
    assert all(ep.success is None and ep.success_step is None for ep in episodes)


def test_success_target_refuses_unread_outcomes(tmp_path):
    tape = write_tape(tmp_path / "d", {0: 900}, 8, 5, seed=10, outcome_fields=False)
    episodes, _r, _d, _m = B.load_tape(tape, "base", TASK, None, None, False)
    with pytest.raises(ValueError, match="needs the outcome fields"):
        B.phi_target(episodes[0], 0.9, C)


# --------------------------------------------------------------- the seed-clustered split


def test_split_never_shares_an_environment_seed():
    for split_seed in range(40):
        for fraction in (0.1, 0.2, 0.35, 0.5, 0.75):
            episodes = [
                B.Episode(
                    key=f"k{i}",
                    role="base",
                    tape_path="t",
                    tape_sha256="s",
                    episode_id=i,
                    env_seed=900 + i // 3,
                    z=torch.zeros(2, 4),
                    u=torch.zeros(2, C, ADIM),
                    z_next=torch.zeros(2, 4),
                    t=torch.tensor([0, C]),
                    success=None,
                    success_step=None,
                )
                for i in range(18)
            ]
            train, holdout = B.split_role(episodes, fraction, split_seed)
            train_seeds = {ep.env_seed for ep in train}
            holdout_seeds = {ep.env_seed for ep in holdout}
            assert not train_seeds & holdout_seeds
            assert train_seeds | holdout_seeds == {900 + i for i in range(6)}
            assert len(train) + len(holdout) == 18
            assert len({ep.key for ep in train} & {ep.key for ep in holdout}) == 0
            for seed_value in holdout_seeds:
                assert sum(ep.env_seed == seed_value for ep in holdout) == 3


# --------------------------------------------------------------- the 8217-d path


def test_rich_latent_flows_through():
    """8217 = 4 x 2048 camera mean/max + 25 proprio, the chain3 round's width."""
    zdim = 8217
    assert "2073" not in Path(B.__file__).read_text()
    with torch.no_grad():
        episodes = [
            B.Episode(
                key=f"k{i}",
                role="base",
                tape_path="t",
                tape_sha256="s",
                episode_id=i,
                env_seed=900 + i,
                z=torch.randn(3, zdim),
                u=torch.randn(3, C, ADIM),
                z_next=torch.randn(3, zdim),
                t=torch.tensor([0, C, 2 * C]),
                success=None,
                success_step=None,
                phi=torch.tensor([0.5, 1.5, 2.5]),
            )
            for i in range(2)
        ]
        mu, sd = B.base_statistics(episodes)
        assert mu.shape == (zdim,) and sd.shape == (zdim,)
        z, u, phi = B.base_rows(episodes, mu, sd, 0.9, C)
        assert z.shape == (6, zdim) and u.shape == (6, C, ADIM) and phi.shape == (6,)
        assert torch.allclose(phi, torch.tensor([0.5, 1.5, 2.5] * 2))
        prior = B.make_prior(zdim, C, ADIM)
        head = B.make_phi(zdim)
        assert prior(z[:2]).shape == (2, C * ADIM)
        assert head(z[:2]).squeeze(-1).shape == (2,)
    model = B.RawLatentWM(zdim, C, ADIM)
    assert model.zdim == zdim
    logs = B.train_world(
        model, episodes, None, 1, 4, 2, (zdim, C, ADIM), mu, sd,
        7, False, 1e-3, torch.device("cpu"), "rich",
    )
    assert logs and logs[-1]["step"] == 1
    assert all(p.device.type == "cpu" for p in model.parameters())
    result = B.evaluate_model(model, episodes, [1], mu, sd)
    assert result["1"]["windows"] == 6 and result["1"]["mse"] is not None


def test_loader_keeps_every_chunk(tmp_path):
    """No 24-chunk truncation: a 75-chunk episode must arrive with 75 rows."""
    tape = write_tape(tmp_path / "d", {0: 900, 1: 901}, 12, 75, seed=13)
    episodes, record, dims, _m = B.load_tape(tape, "base", TASK, None)
    assert [ep.n for ep in episodes] == [75, 75]
    assert record.transitions == 150 and dims == (12, C, ADIM)
    assert episodes[0].t[-1].item() == 74 * C


def test_sample_batch_lands_on_the_requested_device():
    episodes = [
        B.Episode(
            key=f"k{i}", role="base", tape_path="t", tape_sha256="s", episode_id=i,
            env_seed=900 + i, z=torch.randn(4, 6), u=torch.randn(4, C, ADIM),
            z_next=torch.randn(4, 6), t=torch.tensor([0, C, 2 * C, 3 * C]),
            success=None, success_step=None,
        )
        for i in range(3)
    ]
    device = torch.device("cpu")
    batch = B.sample_batch(
        episodes, episodes, 4, 3, (6, C, ADIM), torch.zeros(6), torch.ones(6),
        torch.Generator().manual_seed(0), torch.Generator().manual_seed(1), True, device,
    )
    assert all(tensor.device == device for tensor in batch)
    assert batch[0].shape == (4, 3, 6) and batch[1].shape == (4, 3, C, ADIM)


# --------------------------------------------------------------- end to end


def cli(root: Path, base: Path, update: Path, tag: str) -> list[str]:
    return [
        "build_v241_model_pair.py",
        "--base-tapes", str(base),
        "--update-tapes", str(update),
        "--task", TASK,
        "--latent-dim", "16",
        "--commit", str(C),
        "--holdout-fraction", "0.5",
        "--sequence-len", "3",
        "--horizons", "1", "3",
        "--base-steps", "2",
        "--update-steps", "2",
        "--prior-steps", "2",
        "--phi-steps", "2",
        "--batch-size", "4",
        "--head-batch-size", "8",
        "--threads", "2",
        "--out-root", str(root / "out"),
        "--tag", tag,
    ]


def run_builder(root: Path, monkeypatch, tag: str, extra: list[str], **kwargs) -> dict:
    base, update = build_round(root, **kwargs)
    monkeypatch.setattr(sys, "argv", cli(root, base, update, tag) + extra)
    assert B.main() == 0
    out = sorted((root / "out").glob(f"{tag}_*"))
    assert len(out) == 1
    return torch.load(out[0] / "model_pair.pt", weights_only=False, map_location="cpu")


def test_phi_labels_change_the_head_and_are_recorded(tmp_path, monkeypatch):
    labelled = run_builder(
        tmp_path / "a", monkeypatch, "labelled",
        ["--phi-labels", str(tmp_path / "a" / "base"), str(tmp_path / "a" / "update")],
    )
    default = run_builder(tmp_path / "b", monkeypatch, "default", [])

    lp = labelled["provenance"]["phi_target"]
    dp = default["provenance"]["phi_target"]
    assert lp["source"].startswith("lcwm.v250_progress")
    assert lp["approach_reference_m"] == APPROACH_REFERENCE
    assert lp["outcome_fields_read"] is False
    assert [record["rows"] for record in lp["labels"]] == [56, 42]
    assert dp["source"].startswith("discounted time-to-success")
    assert dp["labels"] == [] and dp["outcome_fields_read"] is True
    assert labelled["provenance"]["contract"]["phi_target"] == "supplied Phi' labels"

    # The fixture has no successes, which is the chain3 situation: the default
    # target is constant zero and the supplied one is not.
    lm = labelled["metrics"]["shared_base_only"]
    dm = default["metrics"]["shared_base_only"]
    for split in ("train", "holdout"):
        assert dm[split]["phi_target_std"] == 0.0
        assert dm[split]["phi_target_positive_fraction"] == 0.0
        assert lm[split]["phi_target_std"] > 0.0
        assert lm[split]["phi_target_positive_fraction"] == 1.0
        assert 0.0 < lm[split]["phi_target_mean"] < 3.0
    same = [
        torch.equal(labelled["phi"][k], default["phi"][k]) for k in labelled["phi"]
    ]
    assert not all(same), "the fitted potential did not move with the target"
    for key in labelled["prior"]:
        assert torch.equal(labelled["prior"][key], default["prior"][key])

    assert set(labelled["models"]) == set(B.MODEL_NAMES)
    assert tuple(labelled["dims"]) == (16, C, ADIM)
    assert labelled["mu"].shape == (16,) and labelled["sd"].shape == (16,)
    manifest = labelled["provenance"]["split"]["base_train"]
    assert all(row["success"] is None for row in manifest)
    assert labelled["provenance"]["split"]["cross_role_env_seed_overlap"] == []


def test_normalisation_uses_base_train_current_states_only(tmp_path, monkeypatch):
    checkpoint = run_builder(tmp_path, monkeypatch, "norm", [])
    keys = {row["key"] for row in checkpoint["provenance"]["split"]["base_train"]}
    tape = torch.load(tmp_path / "base" / "tape.pt", weights_only=False, map_location="cpu")
    episodes, _r, _d, _m = B.load_tape(tmp_path / "base" / "tape.pt", "base", TASK, None)
    train = [ep for ep in episodes if ep.key in keys]
    assert 0 < len(train) < len(episodes)
    z = torch.cat([ep.z for ep in train])
    assert torch.allclose(checkpoint["mu"], z.mean(0), atol=1e-6)
    assert torch.allclose(checkpoint["sd"], z.std(0, unbiased=False).clamp_min(1e-6), 1e-6)
    assert not torch.allclose(checkpoint["mu"], tape["z"].mean(0), atol=1e-6)


def test_updated_and_base_continue_get_matched_budgets(tmp_path, monkeypatch):
    calls: list[dict] = []
    original = B.train_world

    def spy(model, base, update, steps, batch_size, sequence_len, dims, mu, sd,
            seed, shuffle_update, lr, device, label):
        calls.append({
            "label": label, "steps": steps, "batch": batch_size, "seq": sequence_len,
            "lr": lr, "seed": seed, "shuffle": shuffle_update,
            "base": len(base), "update": None if update is None else len(update),
        })
        return original(model, base, update, steps, batch_size, sequence_len, dims,
                        mu, sd, seed, shuffle_update, lr, device, label)

    monkeypatch.setattr(B, "train_world", spy)
    checkpoint = run_builder(tmp_path, monkeypatch, "budget", [])
    by_label = {call["label"]: call for call in calls}
    cont, upd, shuf = by_label["base_continue"], by_label["updated"], by_label["updated_shuffled"]
    for field in ("steps", "batch", "seq", "lr", "base"):
        assert cont[field] == upd[field] == shuf[field]
    assert cont["update"] is None and upd["update"] == shuf["update"] > 0
    assert upd["seed"] == shuf["seed"] and upd["shuffle"] is False and shuf["shuffle"] is True
    logs = checkpoint["metrics"]["training_logs"]
    assert logs["base_continue"][-1]["step"] == logs["updated"][-1]["step"]
    assert logs["updated_shuffled"][-1]["step"] == logs["updated"][-1]["step"]


def test_two_tapes_in_one_directory_raise(tmp_path, monkeypatch):
    """The sidecar and the --phi-labels table are both keyed by a tape's parent
    directory, so a second tape beside the first would silently inherit its
    environment seeds and its Phi' rows on identical (episode, t) keys."""
    episodes = {i: 900 + i for i in range(4)}
    base = write_tape(tmp_path / "d", episodes, 16, 5, seed=21)
    elsewhere = write_tape(tmp_path / "other", episodes, 16, 5, seed=22)
    second = tmp_path / "d" / "tape_b.pt"
    second.write_bytes(elsewhere.read_bytes())
    assert B.sha256_file(base) != B.sha256_file(second)
    monkeypatch.setattr(sys, "argv", cli(tmp_path, base, second, "twotapes"))
    with pytest.raises(ValueError, match="share the directory"):
        B.main()


def test_phi_target_variance_is_reported_next_to_the_std(tmp_path, monkeypatch):
    """Criterion (d) compares a holdout phi_mse against a VARIANCE.  Reporting only
    the standard deviation invites that comparison to be made in the wrong units."""
    checkpoint = run_builder(
        tmp_path, monkeypatch, "var",
        ["--phi-labels", str(tmp_path / "base"), str(tmp_path / "update")],
    )
    for split in ("train", "holdout"):
        row = checkpoint["metrics"]["shared_base_only"][split]
        assert row["phi_target_var"] == pytest.approx(row["phi_target_std"] ** 2, rel=1e-6)
        assert row["phi_target_var"] > 0.0
        assert row["phi_target_var"] != row["phi_target_std"]


def test_cross_role_seed_overlap_is_recorded_both_ways(tmp_path, monkeypatch):
    checkpoint = run_builder(tmp_path, monkeypatch, "roles", [])
    split = checkpoint["provenance"]["split"]
    assert split["cross_role_env_seed_overlap"] == []
    assert split["cross_role_train_holdout_env_seeds"] == []
    assert split["one_tape_per_directory_asserted"] is True


def test_v250_shaped_artifacts_build_without_any_outcome_field(tmp_path, monkeypatch):
    """The round's own layout: tape.pt + labels.pt + outcome.json, no summary.json,
    and the sidecar stripped of success/success_step entirely."""
    checkpoint = run_builder(
        tmp_path, monkeypatch, "v250",
        ["--phi-labels", str(tmp_path / "base"), str(tmp_path / "update")],
        sidecar="outcome", outcome_fields=False,
    )
    assert not (tmp_path / "base" / "summary.json").exists()
    inputs = checkpoint["provenance"]["inputs"]
    assert {record["records_key"] for record in inputs} == {"episodes"}
    assert all(record["outcome_fields_read"] is False for record in inputs)
    assert checkpoint["provenance"]["phi_target"]["outcome_fields_read"] is False
    assert checkpoint["provenance"]["phi_target"]["gamma"] is None
    metrics = checkpoint["metrics"]["shared_base_only"]
    assert metrics["train"]["phi_target_std"] > 0.0
    assert metrics["holdout"]["transitions"] > 0
    for split in checkpoint["provenance"]["split"].values():
        if isinstance(split, list):
            assert all(row["success"] is None for row in split)
