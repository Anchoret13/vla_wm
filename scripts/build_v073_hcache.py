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
        # v072B visits are enumerated TOO: the merged schedule
        # rotates them onto (goal, text) pairs the v072-only
        # enumeration never produced (review critical: 372 missing
        # keys). Each key still resolves against the old index
        # first; only genuinely new ones are computed.
        expanded.append((ak, tvv, True))
        cp0 = canon_p0(a["task"])
        if tvv != cp0:
            expanded.append((ak, cp0, False))
    tr72 = {r["pt_id"]: r for r in S72.load_union()}

    def v72_shard(root_name, shard_name):
        key = ("v72raw", root_name, shard_name)
        if key not in shard_lru:
            if len(shard_lru) > 2:
                shard_lru.clear()
            raw_roots = {k: Path(v["path"]) for k, v in json.loads(
                (V72DATA / "run_manifest.json").read_text())
                ["raw_roots"].items()}
            shard_lru[key] = torch.load(
                raw_roots[root_name] / "shards" / shard_name,
                weights_only=False)
        return shard_lru[key]

    for ak, tvv, with_next in expanded:
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        u = a["universe"]
        if u == "v072B":
            for dd in range(d):
                needed[f"{tvv}::src::{sid}::d{dd}"] = \
                    ("srcdec72", (sid, dd))
            if a["origin"] == "v071_semwin":
                rec0 = tr72[a["rows"][0]]
                wid = rec0["raw_key"].split("|")[0]
                s = v72_shard(rec0["raw_root"], rec0["raw_shard"])
                cum = [0]
                for sg_ in s["segments"]:
                    cum.append(cum[-1] + sg_["actions"])
                for b in cum:
                    needed[f"{tvv}::swb::{wid}::b{b}"] = \
                        ("swb72", (rec0["raw_root"],
                                   rec0["raw_shard"], b))
            else:
                needed[f"{tvv}::src::{sid}::d{d}"] = \
                    ("srcdec72", (sid, d))
                if with_next:
                    for pt in a["rows"]:
                        rec = tr72[pt]
                        if rec["kind"] == "audit":
                            continue
                        needed[f"{tvv}::next::{pt}"] = \
                            ("next72b", pt)
            continue
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

    union72 = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                          / "union_manifest.json").read_text())
    src72_paths = {sid: b["path"] for sid, b in
                   union72["source_histories"].items()}
    for tag in ("selector_r1", "selector_r2"):
        for pth in (RESULTS / f"2026-08-03_v071_{tag}"
                    / "prospective_sources").glob("*.pt"):
            src72_paths[pth.stem] = str(pth)

    def src72(sid):
        key = ("72src", sid)
        if key not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[key] = torch.load(src72_paths[sid],
                                      weights_only=False)
        return src_lru[key]

    def obs_of(kind, ref):
        if kind == "srcdec72":
            sid, dd = ref
            return src72(sid)["rows"][dd]["obs"]
        if kind == "swb72":
            root_name, shard_name, b = ref
            return v72_shard(root_name, shard_name)["frames"][b]
        if kind == "next72b":
            rec = tr72[ref]
            s = v72_shard(rec["raw_root"], rec["raw_shard"])
            if rec["origin"] == "v071_semwin":
                wid, seg = rec["raw_key"].split("|")
                k = int(seg[3:])
                cum = [0]
                for sg_ in s["segments"]:
                    cum.append(cum[-1] + sg_["actions"])
                return s["frames"][cum[k + 1]]
            for tr in s["transitions"]:
                if rec["origin"].startswith("v071_sel"):
                    if f"{s['anchor']}_{tr['candidate_id']}" \
                            == rec["raw_key"]:
                        return tr["frames"][-1]
                elif tr["transition_id"] == rec["raw_key"]:
                    return tr["frames"][-1]
            raise KeyError(rec["raw_key"])
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

    def alias_valid(key, path):
        """Inherited aliases into CANONICAL-namespace caches
        (h_canonical / ext_* dirs) are only valid when the key's text
        is the canonical p0 of the SOURCE'S OWN task (cross-task
        canonical texts were mis-aliased by the v072 builder; caught
        by the V7.3C zero-inv-gradient assertion). The source task is
        read as the unique loho_t* substring of the key."""
        pth = str(path)
        if "/h_canonical/" not in pth and "/ext_" not in pth:
            return True
        tvid = key.split("::", 1)[0]
        task = next((tk for tk in S.TASKS if tk in key), None)
        if task is None:
            return True
        return tvid == S.canon_goal(task) + "_p0"

    invalid = [k for k in list(index)
               if k in needed and not alias_valid(k, index[k])]
    for k in invalid:
        del index[k]
    print(f"[hc] invalidated cross-task canonical aliases: "
          f"{len(invalid)}", flush=True)

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
