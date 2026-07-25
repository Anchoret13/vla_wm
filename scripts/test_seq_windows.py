#!/usr/bin/env python
"""P1 loader smoke test (the pre-registered P1 artifact, 2026-07-25.md).

Asserts, on the real artifacts (CPU only, no model load):
1. aligned window shapes from BOTH sources, printed:
   (h_t, a_block, h_next, labels, source-id);
2. no source episode crosses splits (cache + chain histories together);
3. action normalization round-trip: action_block_env ==
   action_block_norm * std_eps + mean (the runner's exact inverse);
4. terminal records captured before reset: terminal row exists, zero action
   block, success bit consistent with first_success_t;
5. chain-history dedup: one episode per source_trajectory_id;
6. a served window is shape-compatible with LCState.posterior/step
   (random-initialized LC state, CPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.seq_prefix_cache import CACHE_DIR  # noqa: E402
from lcwm.seq_windows import (  # noqa: E402
    SequentialEpisodeDataset,
    iter_windows,
)


def main() -> None:
    train = SequentialEpisodeDataset(CACHE_DIR, splits=("train",))
    dev = SequentialEpisodeDataset(CACHE_DIR, splits=("dev",))
    print(
        f"train: {len(train)} episodes / {train.transition_count()} "
        f"transitions; dev: {len(dev)} episodes / "
        f"{dev.transition_count()} transitions"
    )
    kinds = {e["source_kind"] for e in train.episodes}
    assert kinds == {"demo", "chain"}, f"expected both sources, got {kinds}"

    # -- 2. split integrity across the union ---------------------------------
    union = train.episodes + dev.episodes
    seen: dict[str, str] = {}
    for episode in union:
        source, split = episode["source_id"], episode["split"]
        assert seen.get(source, split) == split, (
            f"source {source} crosses splits"
        )
        seen[source] = split
    print(f"split integrity: {len(seen)} sources, no split crossing")

    # -- 5. chain dedup -------------------------------------------------------
    chain_ids = [
        e["source_id"] for e in union if e["source_kind"] == "chain"
    ]
    assert len(chain_ids) == len(set(chain_ids)), "chain dedup failed"
    print(f"chain histories: {len(chain_ids)} unique source episodes")

    # -- 1. aligned shapes from both sources ----------------------------------
    for kind in ("demo", "chain"):
        episode = next(
            e for e in train.episodes if e["source_kind"] == kind
        )
        window = next(iter_windows(episode, k=1))
        labels = (
            f"q_next {tuple(window['q_next'].shape)} "
            f"bits_next {tuple(window['bits_next'].shape)}"
            if "q_next" in window
            else "labels: absent (flagged)"
        )
        print(
            f"[{kind}] {window['source_id']} idx {window['index']}: "
            f"h_t {tuple(window['h_t'].shape)} {window['h_t'].dtype} | "
            f"a {tuple(window['actions_norm'].shape)} | "
            f"h_next {tuple(window['h_next'].shape)} | {labels}"
        )

    # -- 3. normalization round-trip on a cache episode ------------------------
    demo = next(e for e in train.episodes if e["source_kind"] == "demo")
    raw = torch.load(demo["path"], weights_only=False)
    reconstructed = (
        raw["action_block_norm"] * raw["action_std_eps"] + raw["action_mean"]
    )
    round_trip = float(
        (reconstructed - raw["action_block_env"]).abs().max()
    )
    print(f"normalization round-trip max|err| = {round_trip:.2e}")
    assert round_trip < 1e-5, "action normalization round-trip failed"

    # -- 4. terminal-record semantics -----------------------------------------
    checked = 0
    for episode in union:
        if episode["source_kind"] != "demo":
            continue
        raw = torch.load(episode["path"], weights_only=False)
        terminal = raw["terminal"]
        assert bool(terminal[-1]) and int(terminal.sum()) == 1, (
            "exactly the last record must be terminal"
        )
        assert float(raw["action_block_env"][-1].abs().max()) == 0.0, (
            "terminal record must carry a zero action block"
        )
        if raw["first_success_t"] is not None:
            assert bool(raw["success"][-1]), (
                f"{episode['source_id']}: success at t="
                f"{raw['first_success_t']} but terminal record success bit "
                "is False"
            )
        checked += 1
    print(f"terminal records verified on {checked} cache episodes")

    # -- 7. chain label merge: prefix-bounded, recovery tails unlabeled --------
    labeled_chain = merged_prefix = 0
    for episode in union:
        if episode["source_kind"] != "chain" or not episode["has_labels"]:
            continue
        labeled_chain += 1
        mask = episode["label_mask"]
        R = episode["prefix_hidden"].shape[0]
        prefix_len = int(mask.sum())
        assert bool(mask[:prefix_len].all()) and not bool(
            mask[prefix_len:].any()
        ), "label mask must be a contiguous prefix"
        merged_prefix += prefix_len
        assert episode["generating_policy"] == "pi05_full_prompt_frozen"
        if prefix_len < R:  # stall episode with recovery tail
            windows = list(iter_windows(episode, k=1))
            assert not windows[-1]["labeled"], (
                "recovery-tail window must be unlabeled"
            )
    assert labeled_chain > 0, "no chain episode received labels"
    print(
        f"chain label merge: {labeled_chain} episodes, "
        f"{merged_prefix} labeled prefix records; recovery tails unlabeled"
    )

    # -- 6. LCState shape compatibility (CPU, random init) ---------------------
    from lcwm.lc_flow import LCState

    lc = LCState()
    episode = train.episodes[0]
    window = next(iter_windows(episode, k=1))
    h_t = window["h_t"][None]
    z = lc.posterior(h_t, window["mask_t"][None])
    z_next = lc.step(
        z,
        window["actions_norm"][0][None],
        window["h_next"][None],
        window["mask_next"][None],
        action_mask=window["action_exec_mask"][0][None],
    )
    assert tuple(z.shape) == tuple(z_next.shape) == (1, 4, 384)
    print(f"LCState round: z {tuple(z.shape)} -> z_next {tuple(z_next.shape)}")
    print("P1 SEQ-WINDOW SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
