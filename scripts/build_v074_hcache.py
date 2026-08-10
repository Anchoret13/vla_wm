#!/usr/bin/env python
"""V7.4C — extend the prompt-conditioned h-cache to the V7.4
universe (lcwm/v074_schedule.py: V7.3 merged universe + the
v073T_released teacher-assessment anchors + the V7.4B tranche,
balanced schedule).

Inherited keys resolve through the V7.3C merged hcache_index, but
ONLY where reuse is provably safe: the key scheme embeds
(text_variant, source_id/transition_id, position), so an old entry
is reused iff its kind resolves through the SAME frozen pre-V7.4
source paths the V7.3C builder used, the text-variant files are
byte-identical (sha-checked), no consumed file is newer than the old
index, and the cached file still exists. Keys into the v074 shards /
sources and the v073T assessment shards can never be inherited and
are always computed here. Emits <date>_v074_lcwm_r1/
hcache_index.json over the FULL v074 universe. Resumable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm import v072_schedule as S72  # noqa: E402
from lcwm import v073_schedule as S73  # noqa: E402
from lcwm import v074_schedule as S  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
V72DATA = RESULTS / "2026-08-03_v072_data_r1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"
RID = "v074_lcwm_r1"


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
                          ).stdout.strip()


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    out = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        out = p
    out = out or RESULTS / f"{run_date()}_{RID}"
    hdir = out / "hcache"
    hdir.mkdir(parents=True, exist_ok=True)

    old_root = None
    for p in sorted(RESULTS.glob("*_v073_lcwm_r1")):
        old_root = p
    if old_root is None:
        raise SystemExit("[hc] no *_v073_lcwm_r1 root — the V7.3C "
                         "merged index is required for reuse")
    old_index_path = old_root / "hcache_index.json"
    anchors, tq = S.load_universe()

    # V7.4B must be COMPLETE before caching: a partial shard set
    # would silently drop next-obs keys the trainer will assert on.
    missing = []
    for ak in sorted(anchors):
        a = anchors[ak]
        if a["universe"] != "v074":
            continue
        p = S.shards_for("v074") \
            / f"{a['source_id']}_d{a['decision']}.pt"
        if not p.exists():
            missing.append(p.name)
    if missing:
        raise SystemExit(
            f"[hc] v074B incomplete: {len(missing)} acq shards "
            f"missing (finish scripts/build_v074_data.py first), "
            f"e.g. {missing[:4]}")

    lang_of = {}
    for task, rows in tq.items():
        for r in rows:
            lang_of[r["text_variant_id"]] = r["language"]
    visits = S.needed_visits(anchors, tq)
    print(f"[hc] {len(anchors)} anchors, {len(visits)} visits",
          flush=True)
    miss_tv = sorted({tvv for _ak, _g, tvv in visits}
                     - set(lang_of))
    assert not miss_tv, f"text variants missing from queries: " \
        f"{miss_tv[:5]}"

    # ---- text identity: the tvv -> language map is provably the
    # one the V7.3C builder used only if the frozen text files are
    # byte-identical to what both data runs recorded ----------------
    tv_sha = sha256_of(SEMWIN / "text_variants.json")
    gq_sha = sha256_of(V72DATA / "goal_queries.jsonl")
    for tag, mp in (("v073", S73.V73 / "run_manifest.json"),
                    ("v074", S.v074_root()
                     / "run_manifest.json")):
        rec = json.loads(mp.read_text())["text_variants_sha256"]
        assert rec == tv_sha, f"{tag} text_variants sha drift"

    old_bytes = old_index_path.read_bytes()
    old_index_sha = hashlib.sha256(old_bytes).hexdigest()
    old_index = json.loads(old_bytes)
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
        # every visit key resolves against the old index first;
        # the canonical-p0 second stream (trainer language contract)
        # needs source-history keys only (with_next=False)
        expanded.append((ak, tvv, True))
        cp0 = canon_p0(a["task"])
        if tvv != cp0:
            expanded.append((ak, cp0, False))
    tr72 = {r["pt_id"]: r for r in S72.load_union()}
    raw_roots = {k: Path(v["path"]) for k, v in json.loads(
        (V72DATA / "run_manifest.json").read_text())
        ["raw_roots"].items()}

    def v72_shard(root_name, shard_name):
        key = ("v72raw", root_name, shard_name)
        if key not in shard_lru:
            if len(shard_lru) > 2:
                shard_lru.clear()
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
        if not with_next:
            continue
        akey = f"{sid}_d{d}"
        if u == "v073":
            s = shard("v073", akey)
            for tr in s["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                needed[f"{tvv}::next::v073::"
                       f"{tr['transition_id']}"] = \
                    ("next73", (sid, d, tr["branch_key"]))
        elif u == "v073T_released":
            s = shard("v073T_released", akey)
            for tr in s["transitions"]:
                needed[f"{tvv}::next::v073T_released::"
                       f"{tr['transition_id']}"] = \
                    ("next73T", (sid, d, tr["candidate_id"]))
        elif u == "v074":
            s = shard("v074", akey)
            for tr in s["transitions"]:
                if tr["branch_key"] == "u0_repeat":
                    continue
                needed[f"{tvv}::next::v074::"
                       f"{tr['transition_id']}"] = \
                    ("next74", (sid, d, tr["branch_key"]))
        else:
            s2 = shard("v072T", akey)
            for tr in s2["transitions"]:
                if tr["kind"] == "audit":
                    continue
                needed[f"{tvv}::next::v072T::"
                       f"{tr['transition_id']}"] = \
                    ("next72t", (sid, d, tr["candidate_id"]))
    print(f"[hc] {len(needed)} candidate keys", flush=True)

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
        if kind == "next73T":
            sid, d, cid = ref
            s = shard("v073T_released", f"{sid}_d{d}")
            tr = next(x for x in s["transitions"]
                      if x["candidate_id"] == cid)
            return tr["frames"][-1]
        if kind == "next74":
            sid, d, bk = ref
            s = shard("v074", f"{sid}_d{d}")
            tr = next(x for x in s["transitions"]
                      if x["branch_key"] == bk)
            return tr["frames"][-1]
        sid, d, cid = ref
        s = shard("v072T", f"{sid}_d{d}")
        tr = next(x for x in s["transitions"]
                  if x["candidate_id"] == cid)
        return tr["frames"][-1]

    # ---- reuse rule -------------------------------------------------
    # Staleness: reuse additionally requires that no file the V7.3C
    # builder read for that universe group is NEWER than the old
    # index (frozen-results discipline made mechanical; we cannot
    # hash multi-GB .pt sources per key).
    old_mtime = old_index_path.stat().st_mtime

    def newest(paths):
        m = 0.0
        for p in paths:
            p = Path(p)
            if p.exists():
                m = max(m, p.stat().st_mtime)
        return m

    grp_files = {
        "v072B": ([Path(v) for v in src72_paths.values()]
                  + [q for r in raw_roots.values()
                     for q in (r / "shards").glob("*.pt")]),
        "v072T": (list(S73.sources_for("v072T").glob("*.pt"))
                  + list(S73.shards_for("v072T").glob("*.pt"))),
        "v073": (list(S73.sources_for("v073").glob("*.pt"))
                 + list(S73.shards_for("v073").glob("*.pt"))),
    }
    stale = {g for g, fs in grp_files.items()
             if newest(fs) > old_mtime}
    if stale:
        print(f"[hc] STALE inherited groups (reuse refused, will "
              f"recompute): {sorted(stale)}", flush=True)

    def reuse_group(kind, ref):
        """Universe group an inherited key resolves through, or None
        when the key can never be inherited (v074 / v073T shards)."""
        if kind in ("srcdec72", "swb72", "next72b"):
            return "v072B"
        if kind == "next72t":
            return "v072T"
        if kind == "next73":
            return "v073"
        if kind == "srcdec":
            # released teacher sources ARE the frozen V7.3B dir
            return {"v072T": "v072T", "v073": "v073",
                    "v073T_released": "v073"}.get(ref[0])
        return None

    def alias_valid(key, path):
        """Inherited aliases into CANONICAL-namespace caches
        (h_canonical / ext_* dirs) are only valid when the key's
        text is the canonical p0 of the SOURCE'S OWN task (V7.3C
        finding, kept verbatim)."""
        pth = str(path)
        if "/h_canonical/" not in pth and "/ext_" not in pth:
            return True
        tvid = key.split("::", 1)[0]
        task = next((tk for tk in S.TASKS if tk in key), None)
        if task is None:
            return True
        return tvid == canon_p0(task)

    n_alias = n_stale = n_unsafe = 0
    for key, (kind, ref) in needed.items():
        if key not in index:
            continue
        g = reuse_group(kind, ref)
        if g is None:
            # v074/v073T-shard keys must be new — an old-index hit
            # here would mean a key-scheme collision
            raise AssertionError(f"non-inheritable key in old "
                                 f"index: {key}")
        if g in stale:
            n_stale += 1
            del index[key]
        elif not alias_valid(key, index[key]):
            n_alias += 1
            del index[key]
        elif not Path(index[key]).exists():
            n_unsafe += 1
            del index[key]
    print(f"[hc] reuse filter: alias={n_alias} stale={n_stale} "
          f"missing_file={n_unsafe}", flush=True)

    todo = []
    n_reused = 0
    inherited_paths = set()
    for key, (kind, ref) in sorted(needed.items()):
        if key in index:
            n_reused += 1
            inherited_paths.add(index[key])
            continue
        own = hdir / (key.replace("::", "__") + ".pt")
        if own.exists():
            index[key] = str(own)
        else:
            todo.append((key, kind, ref, own))
    print(f"[hc] reused: {n_reused}, to compute: {len(todo)}",
          flush=True)

    # Census of the inherited mapped .pt files at reuse-decision
    # time: byte size + mtime only — a full per-file sha of
    # thousands of cached files is deliberately not taken (the old
    # index is sha-bound above and the staleness rule bounds mtimes).
    inherited_census = {}
    for pth in sorted(inherited_paths):
        st = Path(pth).stat()
        inherited_census[pth] = {"bytes": st.st_size,
                                 "mtime": st.st_mtime}

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
    (out / "hcache_index.json").write_text(json.dumps(index,
                                                      indent=0))
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    (out / "hcache_manifest.json").write_text(json.dumps({
        "schema": "v074_hcache_manifest_v1", "run_schema": "v074",
        "run_id": RID, "git_sha": git_sha,
        "old_index": str(old_index_path),
        "old_index_sha256": old_index_sha,
        "reuse_rule": "old V7.3C entries reused only for keys "
                      "resolving through frozen pre-V7.4 paths "
                      "(v072B/v072T/v073 sources+shards) with "
                      "byte-identical text files (sha-checked), no "
                      "file newer than the old index, cached file "
                      "present, and the V7.3C cross-task canonical "
                      "alias rule; v074 / v073T_released shard keys "
                      "always computed fresh",
        "text_variants_sha256": tv_sha,
        "goal_queries_sha256": gq_sha,
        "counts": {"anchors": len(anchors), "visits": len(visits),
                   "needed": len(needed), "reused": n_reused,
                   "dropped_alias": n_alias,
                   "dropped_stale": n_stale,
                   "dropped_missing_file": n_unsafe,
                   "computed": len(todo),
                   "inherited_unique_files":
                       len(inherited_census),
                   "index_keys": len(index)},
        "inherited_file_census": inherited_census,
        "note": "index is a superset (carries all old keys); only "
                "keys enumerated by lcwm/v074_schedule."
                "needed_visits are validated by the reuse rule",
    }, indent=2))
    print(f"[hc] merged index: {len(index)} keys -> "
          f"{out / 'hcache_index.json'}", flush=True)


if __name__ == "__main__":
    main()
