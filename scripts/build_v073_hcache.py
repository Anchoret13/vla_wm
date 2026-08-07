#!/usr/bin/env python
"""V7.3C — extend the prompt-conditioned h-cache to the merged
anchor universe.

v072B keys resolve through the existing 2026-08-03_v072_data_r1
hcache_index (identical (obs, prompt) function). New keys computed
here: full canonical histories + per-branch next-obs for the released
V7.2 teacher anchors, and full per-text histories + next-obs for the
V7.3B anchors (2 GoalSpecs x 3 texts, plus canonical-p0 second-stream
keys). Emits 2026-08-06_v073_lcwm_r1/hcache_index.json mapping every
key of the MERGED universe to its file. Resumable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v072_schedule as S72  # noqa: E402
from lcwm import v073_schedule as S  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V72DATA = RESULTS / "2026-08-03_v072_data_r1"
OUT = RESULTS / "2026-08-06_v073_lcwm_r1"
HDIR = OUT / "hcache"


def main() -> None:
    HDIR.mkdir(parents=True, exist_ok=True)
    anchors, tq = S.load_universe()
    lang_of = {}
    for task, rows in tq.items():
        for r in rows:
            lang_of[r["text_variant_id"]] = r["language"]
    visits = S.needed_visits(anchors, tq)
    print(f"[hc] {len(anchors)} anchors, {len(visits)} visits",
          flush=True)

    old_index = json.loads(
        (V72DATA / "hcache_index.json").read_text())
    index = dict(old_index)

    src_lru, shard_lru = {}, {}

    def src(universe, sid):
        key = (universe, sid)
        if key not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[key] = torch.load(
                S.sources_for(universe) / f"{sid}.pt",
                weights_only=False)
        return src_lru[key]

    def shard(universe, akey):
        key = (universe, akey)
        if key not in shard_lru:
            if len(shard_lru) > 2:
                shard_lru.clear()
            shard_lru[key] = torch.load(
                S.shards_for(universe) / f"{akey}.pt",
                weights_only=False)
        return shard_lru[key]

    def canon_p0(task):
        return S.canon_goal(task) + "_p0"

    needed = {}
    expanded = []
    for ak, g, tvv in visits:
        a = anchors[ak]
        if a["universe"] == "v072B":
            continue    # fully covered by the old index
        expanded.append((ak, tvv, True))
        cp0 = canon_p0(a["task"])
        if tvv != cp0:
            expanded.append((ak, cp0, False))
    for ak, tvv, with_next in expanded:
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        u = a["universe"]
        for dd in range(d + 1):
            needed[f"{tvv}::src::{sid}::d{dd}"] = \
                ("srcdec", (u, sid, dd))
        if with_next:
            akey = f"{sid}_d{d}"
            s = shard(u, akey) if u == "v073" else None
            if u == "v073":
                for tr in s["transitions"]:
                    if tr["branch_key"] == "u0_repeat":
                        continue
                    needed[f"{tvv}::next::v073::"
                           f"{tr['transition_id']}"] = \
                        ("next73", (sid, d, tr["branch_key"]))
            else:
                s2 = shard("v072T", akey)
                for tr in s2["transitions"]:
                    if tr["kind"] == "audit":
                        continue
                    needed[f"{tvv}::next::v072T::"
                           f"{tr['transition_id']}"] = \
                        ("next72t", (sid, d, tr["candidate_id"]))
    print(f"[hc] {len(needed)} candidate new keys", flush=True)

    def obs_of(kind, ref):
        if kind == "srcdec":
            u, sid, dd = ref
            return src(u, sid)["rows"][dd]["obs"]
        if kind == "next73":
            sid, d, bk = ref
            s = shard("v073", f"{sid}_d{d}")
            tr = next(x for x in s["transitions"]
                      if x["branch_key"] == bk)
            return tr["frames"][-1]
        sid, d, cid = ref
        s = shard("v072T", f"{sid}_d{d}")
        tr = next(x for x in s["transitions"]
                  if x["candidate_id"] == cid)
        return tr["frames"][-1]

    todo = []
    for key, (kind, ref) in sorted(needed.items()):
        if key in index:
            continue
        own = HDIR / (key.replace("::", "__") + ".pt")
        if own.exists():
            index[key] = str(own)
        else:
            todo.append((key, kind, ref, own))
    print(f"[hc] to compute: {len(todo)}", flush=True)

    if todo:
        from lcwm.chassis import Pi05Runner
        from lcwm.sampler import prefix_forward
        runner = Pi05Runner(suite_name="libero_10")
        with torch.no_grad():
            for i, (key, kind, ref, own) in enumerate(todo):
                tvv = key.split("::", 1)[0]
                b = runner._obs_to_policy_batch(
                    obs_of(kind, ref), lang_of[tvv])
                pfx = prefix_forward(runner.policy, b)
                tmp = own.with_suffix(".tmp")
                torch.save({"h": pfx.hidden[0].half().cpu(),
                            "mask": pfx.pad_masks[0].bool().cpu()},
                           tmp)
                tmp.replace(own)
                index[key] = str(own)
                if (i + 1) % 250 == 0:
                    print(f"  [hc] {i + 1}/{len(todo)}", flush=True)
    (OUT / "hcache_index.json").write_text(json.dumps(index,
                                                      indent=0))
    print(f"[hc] merged index: {len(index)} keys -> "
          f"{OUT / 'hcache_index.json'}", flush=True)


if __name__ == "__main__":
    main()
