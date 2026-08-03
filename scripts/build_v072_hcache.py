#!/usr/bin/env python
"""V7.2B step 1 — build the prompt-conditioned PrefixVLM h-cache.

Enumerates EXACTLY the (observation, text) keys the frozen 25-epoch
schedule (lcwm/v072_schedule.py) will touch — full-history unrolls
from episode reset under the visit's query text, anchor observations,
and per-row next observations — then resolves each key against the
existing 2026-08-02 caches (same (obs, prompt) function) and computes
only the missing ones. Emits hcache_index.json mapping every v072 key
to its file; the trainer loads through this index only.

Never copies a feature computed under another prompt: aliases resolve
strictly to files whose prompt text is identical (canonical p0 == the
behavior prompt; q_{gid}_{pv} == the same goal/variant text used in
2c). Resumable; safe to re-run.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v072_schedule as S  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
DATA_R1 = RESULTS / "2026-08-03_v072_data_r1"
HDIR = DATA_R1 / "hcache"
CACHE_2A = RESULTS / "2026-08-02_v071_lcwm_r1" / "train" / "h_canonical"
CACHE_2C_EXT = RESULTS / "2026-08-02_v071_lcwm2c_r1" / "train" \
    / "ext_lc_full"
CACHE_2C_Q = RESULTS / "2026-08-02_v071_lcwm2c_r1" / "train" \
    / "query_lc_full"


def main() -> None:
    HDIR.mkdir(exist_ok=True)
    transitions = S.load_union()
    by_pt = {r["pt_id"]: r for r in transitions}
    queries = S.load_queries()
    lang_of = {}
    for task, rows in queries.items():
        for r in rows:
            lang_of[r["text_variant_id"]] = r["language"]
    anchors = S.build_anchors(transitions)
    visits = S.needed_visits(anchors, queries)
    print(f"[hcache] {len(visits)} scheduled visits", flush=True)

    # sources rows / shard obs loaders (lazy, LRU)
    union = json.loads((RESULTS / "2026-08-02_v071_union_f1"
                        / "union_manifest.json").read_text())
    src_paths = {sid: b["path"]
                 for sid, b in union["source_histories"].items()}
    for tag in ("selector_r1", "selector_r2"):
        root = RESULTS / f"2026-08-03_v071_{tag}"
        for p in (root / "prospective_sources").glob("*.pt"):
            src_paths[p.stem] = str(p)
    root_dirs = {r["raw_root"]: None for r in transitions}
    raw_roots = json.loads((DATA_R1 / "run_manifest.json")
                           .read_text())["raw_roots"]
    for k in root_dirs:
        root_dirs[k] = Path(raw_roots[k]["path"])
    src_lru, shard_lru = {}, {}

    def src(sid):
        if sid not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[sid] = torch.load(src_paths[sid],
                                      weights_only=False)
        return src_lru[sid]

    def shard(root_name, shard_name):
        key = (root_name, shard_name)
        if key not in shard_lru:
            if len(shard_lru) > 3:
                shard_lru.clear()
            shard_lru[key] = torch.load(
                root_dirs[root_name] / "shards" / shard_name,
                weights_only=False)
        return shard_lru[key]

    def obs_of(kind, ref):
        """Resolve an observation reference to the raw obs dict."""
        if kind == "srcdec":
            sid, dd = ref
            return src(sid)["rows"][dd]["obs"]
        if kind == "swb":
            rec, b = ref
            s = shard(rec["raw_root"], rec["raw_shard"])
            return s["frames"][b]
        if kind == "next":
            rec = ref
            s = shard(rec["raw_root"], rec["raw_shard"])
            if rec["origin"] == "v071_semwin":
                wid, seg = rec["raw_key"].split("|")
                k = int(seg[3:])
                cum = [0]
                for sg in s["segments"]:
                    cum.append(cum[-1] + sg["actions"])
                return s["frames"][cum[k + 1]]
            for tr in s["transitions"]:
                tid_key = tr.get("transition_id",
                                 tr.get("candidate_id"))
                if rec["origin"].startswith("v071_sel"):
                    if f"{s['anchor']}_{tr['candidate_id']}" \
                            == rec["raw_key"]:
                        return tr["frames"][-1]
                elif tr["transition_id"] == rec["raw_key"]:
                    return tr["frames"][-1]
            raise KeyError(rec["raw_key"])
        raise ValueError(kind)

    # ---- enumerate needed keys -------------------------------------
    # key = f"{tvid}::{obs_id}"; obs_id = "src::{sid}::d{dd}" |
    # "swb::{window}::b{b}" | "next::{pt_id}"
    needed = {}   # key -> (kind, ref)

    def canon_p0(task):
        return S.canonical_goal(task) + "_p0"

    semwin_shards = {}
    for ak, g, tv in visits:
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        for dd in range(d):
            needed[f"{tv}::src::{sid}::d{dd}"] = \
                ("srcdec", (sid, dd))
        if a["origin"] == "v071_semwin":
            rec0 = by_pt[a["rows"][0]]
            wid = rec0["raw_key"].split("|")[0]
            needed[f"{tv}::swb::{wid}::b0"] = \
                ("swb", (rec0, 0))
            s = semwin_shards.setdefault(
                rec0["raw_shard"],
                shard(rec0["raw_root"], rec0["raw_shard"]))
            cum = [0]
            for sg in s["segments"]:
                cum.append(cum[-1] + sg["actions"])
            for k in range(1, len(cum)):
                needed[f"{tv}::swb::{wid}::b{cum[k]}"] = \
                    ("swb", (rec0, cum[k]))
        else:
            needed[f"{tv}::src::{sid}::d{d}"] = ("srcdec", (sid, d))
            for pt in a["rows"]:
                rec = by_pt[pt]
                if rec["kind"] == "audit":
                    continue
                needed[f"{tv}::next::{pt}"] = ("next", rec)
    print(f"[hcache] {len(needed)} unique (text, obs) keys",
          flush=True)

    # ---- alias resolution ------------------------------------------
    index, todo = {}, []
    for key, (kind, ref) in sorted(needed.items()):
        tv = key.split("::", 1)[0]
        rest = key.split("::", 1)[1]
        alias = None
        task_of_tv = next((t for t, rows in queries.items()
                           if any(r["text_variant_id"] == tv
                                  for r in rows)), None)
        is_canon = task_of_tv and tv == canon_p0(task_of_tv)
        if kind == "srcdec":
            sid, dd = ref
            if is_canon:
                for dd_dir in (CACHE_2A, CACHE_2C_EXT):
                    p = dd_dir / f"{sid}__dec{dd}.pt"
                    if p.exists():
                        alias = p
                        break
            else:
                g, pv = tv.rsplit("_", 1)
                p = CACHE_2C_Q / f"q_{g}_{pv}__{sid}__dec{dd}.pt"
                if p.exists():
                    alias = p
        elif kind == "swb":
            wid_b = rest.split("::")
            wid, b = wid_b[1], wid_b[2]
            if is_canon:
                p = CACHE_2C_EXT / f"{wid}__{b}.pt"
                if p.exists():
                    alias = p
            elif b == "b0":
                g, pv = tv.rsplit("_", 1)
                p = CACHE_2C_Q / f"q_{g}_{pv}__{wid}__b0.pt"
                if p.exists():
                    alias = p
        elif kind == "next" and is_canon:
            pt = rest.split("::", 1)[1]
            if pt.startswith("u1::"):
                p = CACHE_2A / f"{pt[4:]}__post.pt"
                if p.exists():
                    alias = p
        own = HDIR / (key.replace("::", "__") + ".pt")
        if alias is not None:
            index[key] = str(alias)
        elif own.exists():
            index[key] = str(own)
        else:
            todo.append((key, kind, ref, own))
    print(f"[hcache] aliased/existing "
          f"{len(index)}, to compute {len(todo)}", flush=True)

    if todo:
        from lcwm.chassis import Pi05Runner
        from lcwm.sampler import prefix_forward
        runner = Pi05Runner(suite_name="libero_10")
        with torch.no_grad():
            for i, (key, kind, ref, own) in enumerate(todo):
                tv = key.split("::", 1)[0]
                obs = obs_of(kind, ref)
                b = runner._obs_to_policy_batch(obs, lang_of[tv])
                pfx = prefix_forward(runner.policy, b)
                tmp = own.with_suffix(".tmp")
                torch.save({"h": pfx.hidden[0].half().cpu(),
                            "mask": pfx.pad_masks[0].bool().cpu()},
                           tmp)
                tmp.replace(own)
                index[key] = str(own)
                if (i + 1) % 250 == 0:
                    print(f"  [hcache] {i + 1}/{len(todo)}",
                          flush=True)
    (DATA_R1 / "hcache_index.json").write_text(
        json.dumps(index, indent=0))
    print(f"[hcache] index complete: {len(index)} keys -> "
          f"{DATA_R1 / 'hcache_index.json'}", flush=True)


if __name__ == "__main__":
    main()
