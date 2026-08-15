#!/usr/bin/env python
"""V7.5C — prompt-conditioned h-cache over the surviving universe.

Unlike scripts/build_v074_hcache.py there is NO inheritance path. The
V7.3C cache files are deleted (only its index JSON survives, and every
path in it dangles), so every key is computed fresh here. That removes
the whole reuse/staleness/alias apparatus the v074 builder needed.

Universe: `lcwm/v075_schedule.py` -- the v074 acquisition tranche
alone. Keys, unchanged in scheme from V7.4:

    unroll     {tvv}::src::{sid}::d{dd}     for dd in 0..decision
    next-obs   {tvv}::next::v074::{transition_id}

Emitted for EVERY (anchor, text-variant) pair rather than only the
scheduled ones: the trainer's language second stream and its dev
readouts both unroll at the task's canonical p0 regardless of the
visit's own query, and a missing key surfaces as a mid-training
assertion hours in. The superset costs cache entries, not correctness.

`u0_repeat` transitions are skipped, matching the trainer's payload.

Output: <date>_v075_lcwm_r1/{hcache_index.json,hcache_manifest.json}
Resumable: an existing per-key .pt is adopted without recompute.
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

from lcwm import v075_schedule as S  # noqa: E402

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"
DEMO_DIR = Path("/home/stargazer/Desktop/vla_wm/datasets"
                "/libero_loho_public_v1/demo_rehearsal_v067")
RID = "v075_lcwm_r1"


def run_date() -> str:
    return subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}).stdout.strip()


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preserve_and_write(path: Path, payload: str, stage: str) -> dict:
    """Write `payload` to `path` without ever destroying a prior version.

    V7.5 lost the exact identity of its training input: this script was
    run once with `--stage lcwm` (3978 keys) and again with
    `--stage policy` (5825 keys), and the second run rewrote
    `hcache_index.json` in place. The training index is unrecoverable --
    the stage counts do not subtract cleanly, so it cannot be
    reconstructed by filtering the survivor. See the 2026-08-14 audit
    amendment in plan_and_progress/2026-08-09.md and action item 1
    ("preserve every new immutable input index before any cache
    expansion").

    An index is an IMMUTABLE INPUT once a run has trained against it.
    So: content-address every version under `index_archive/`, append an
    audit row to `index_lineage.jsonl`, and only then move the
    convenience pointer. Re-writing identical bytes is a no-op, and a
    content-addressed name that already exists is asserted byte-equal
    rather than rewritten.
    """
    new_sha = hashlib.sha256(payload.encode()).hexdigest()
    archive = path.parent / "index_archive"
    archive.mkdir(parents=True, exist_ok=True)
    row = {"file": path.name, "stage": stage, "sha256": new_sha,
           "bytes": len(payload.encode())}

    # 1. preserve whatever is already there, if it differs
    if path.exists():
        old = path.read_bytes()
        old_sha = hashlib.sha256(old).hexdigest()
        if old_sha == new_sha:
            row["action"] = "unchanged"
            return row
        kept = archive / f"{path.stem}__{old_sha[:12]}{path.suffix}"
        if not kept.exists():
            kept.write_bytes(old)
        row["superseded_sha256"] = old_sha
        row["superseded_preserved_at"] = str(kept.relative_to(path.parent))
        row["action"] = "superseded"
    else:
        row["action"] = "created"

    # 2. immutable content-addressed copy of the new version
    frozen = archive / f"{path.stem}__{stage}__{new_sha[:12]}{path.suffix}"
    if frozen.exists():
        assert sha256_of(frozen) == new_sha, \
            f"content-addressed collision at {frozen}"
    else:
        frozen.write_text(payload)
    row["frozen_at"] = str(frozen.relative_to(path.parent))

    # 3. only now move the pointer, and log it
    path.write_text(payload)
    with (path.parent / "index_lineage.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate and validate keys, compute none")
    ap.add_argument("--stage", choices=("lcwm", "policy"),
                    default="lcwm",
                    help="lcwm: the 33 acq anchors x 6 queries "
                         "(V7.5C training). policy: additionally the "
                         "9 teach anchors at canonical p0 and every "
                         "train demo-rehearsal row, which the Stage 3 "
                         "gate needs to build the complete "
                         "policy-state set.")
    args = ap.parse_args()
    out = None
    for p in sorted(RESULTS.glob(f"*_{RID}")):
        out = p
    out = out or RESULTS / f"{run_date()}_{RID}"
    hdir = out / "hcache"
    hdir.mkdir(parents=True, exist_ok=True)

    anchors, tq = S.load_universe()
    lang_of = {}
    for _task, rows in tq.items():
        for r in rows:
            lang_of[r["text_variant_id"]] = r["language"]

    # Text identity: the tvv -> language map must be the one the
    # acquisition run recorded, or the cached prompt is not the prompt
    # the data was collected under.
    tv_sha = sha256_of(SEMWIN / "text_variants.json")
    rec = json.loads(
        (S.v074_root() / "run_manifest.json").read_text())
    assert rec["text_variants_sha256"] == tv_sha, \
        "v074 text_variants sha drift — the prompt set moved under " \
        "the collected data"

    src_lru, shard_lru = {}, {}

    def src(sid):
        if sid not in src_lru:
            if len(src_lru) > 6:
                src_lru.clear()
            src_lru[sid] = torch.load(
                S.sources_for(S.UNIVERSE) / f"{sid}.pt",
                weights_only=False)
        return src_lru[sid]

    def shard(akey):
        if akey not in shard_lru:
            if len(shard_lru) > 3:
                shard_lru.clear()
            shard_lru[akey] = torch.load(
                S.shards_for(S.UNIVERSE) / f"{akey}.pt",
                weights_only=False)
        return shard_lru[akey]

    # ---- enumerate every key the trainer can ask for ----------------
    needed = {}
    for ak in sorted(anchors):
        a = anchors[ak]
        sid, d = a["source_id"], a["decision"]
        n_rows = len(src(sid)["rows"])
        assert d < n_rows, \
            f"{ak}: decision {d} beyond source rows {n_rows}"
        trs = [t["transition_id"] for t in shard(f"{sid}_d{d}")
               ["transitions"] if t["branch_key"] != "u0_repeat"]
        for _g, tvv in S.queries_for(a):
            for dd in range(d + 1):
                needed[f"{tvv}::src::{sid}::d{dd}"] = ("srcdec",
                                                       (sid, dd))
            for tid in trs:
                needed[f"{tvv}::next::{S.UNIVERSE}::{tid}"] = (
                    "next", (sid, d, tid))

    if args.stage == "policy":
        # teach anchors: canonical p0 only. They carry no
        # semantic_query_index rows, so they stay uncrossed — the same
        # per-anchor admission rule v073/v074 used.
        man = json.loads(
            (S.v074_root() / "anchor_manifest.json").read_text())
        n_teach = 0
        for a in man["anchors"]:
            if a["kind"] != "teach":
                continue
            sid, d = a["source_id"], a["decision"]
            cp0 = f"{S.canon_goal(a['task'])}_p0"
            assert (S.sources_for(S.UNIVERSE) / f"{sid}.pt").exists()
            for dd in range(d + 1):
                needed[f"{cp0}::src::{sid}::d{dd}"] = ("srcdec",
                                                       (sid, dd))
            n_teach += 1
        # Demo-rehearsal observations at the WM's OWN 10-action
        # decision stride (`ep["obs_10"]`), NOT the 50-stride `rows`.
        # This is the V7.3D defect: rows are 50-stride, so keying the
        # cache by row observation feeds the recurrence one frame per
        # 50 expert actions and drives the transition out of
        # distribution. The gate unrolls obs_10 with t <= 50*ri, so
        # the cache is keyed by t.
        from lcwm.v067_lineage import load_v067
        dman = json.loads((DEMO_DIR / "manifest.json").read_text())
        n_demo = 0
        for e in dman["episodes"]:
            ep = load_v067(DEMO_DIR / e["path"], "demo_rehearsal")
            if ep["split"] != "train":
                continue
            t_max = 50 * (len(ep["rows"]) - 1)
            for j, o in enumerate(ep["obs_10"]):
                if o["t"] > t_max:
                    continue
                needed[f"demo::{e['task_id']}::{e['demo']}"
                       f"::t{o['t']}"] = (
                    "demo", (str(DEMO_DIR / e["path"]), j))
                n_demo += 1
        print(f"[hc] policy stage: +{n_teach} teach anchors, "
              f"+{n_demo} demo rows", flush=True)

    miss_tv = sorted({k.split("::", 1)[0] for k in needed
                      if not k.startswith("demo::")}
                     - set(lang_of))
    assert not miss_tv, f"text variants missing from queries: {miss_tv}"

    demo_lru = {}

    def demo_ep(path):
        if path not in demo_lru:
            from lcwm.v067_lineage import load_v067
            if len(demo_lru) > 3:
                demo_lru.clear()
            demo_lru[path] = load_v067(Path(path), "demo_rehearsal")
        return demo_lru[path]

    def obs_of(kind, ref):
        if kind == "srcdec":
            sid, dd = ref
            return src(sid)["rows"][dd]["obs"]
        if kind == "demo":
            path, j = ref
            return demo_ep(path)["obs_10"][j]["obs"]
        sid, d, tid = ref
        s = shard(f"{sid}_d{d}")
        tr = next(x for x in s["transitions"]
                  if x["transition_id"] == tid)
        return tr["frames"][-1]

    index, todo = {}, []
    for key, (kind, ref) in sorted(needed.items()):
        own = hdir / (key.replace("::", "__") + ".pt")
        if own.exists():
            index[key] = str(own)
        else:
            todo.append((key, kind, ref, own))
    print(f"[hc] {len(anchors)} anchors, {len(needed)} keys "
          f"({len(index)} already cached, {len(todo)} to compute)",
          flush=True)

    if args.dry_run:
        kinds_d = {}
        for _k, (kind, _r) in needed.items():
            kinds_d[kind] = kinds_d.get(kind, 0) + 1
        print(f"[hc] DRY RUN — key kinds: {kinds_d}")
        print(f"[hc] sample: {sorted(needed)[0]}")
        print(f"[hc] obs resolves: "
              f"{type(obs_of(*needed[sorted(needed)[0]])).__name__}")
        return

    if todo:
        from lcwm.chassis import Pi05Runner
        from lcwm.sampler import prefix_forward
        runner = Pi05Runner(suite_name="libero_10")
        with torch.no_grad():
            for i, (key, kind, ref, own) in enumerate(todo):
                if kind == "demo":
                    lang = demo_ep(ref[0])["language"]
                else:
                    lang = lang_of[key.split("::", 1)[0]]
                b = runner._obs_to_policy_batch(obs_of(kind, ref),
                                                lang)
                pfx = prefix_forward(runner.policy, b)
                tmp = own.with_suffix(".tmp")
                torch.save({"h": pfx.hidden[0].half().cpu(),
                            "mask": pfx.pad_masks[0].bool().cpu()},
                           tmp)
                tmp.replace(own)
                index[key] = str(own)
                if (i + 1) % 250 == 0:
                    print(f"  [hc] {i + 1}/{len(todo)}", flush=True)

    assert set(index) == set(needed), \
        f"index/needed mismatch: {len(index)} vs {len(needed)}"
    idx_row = preserve_and_write(out / "hcache_index.json",
                                 json.dumps(index, indent=0), args.stage)

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    kinds = {}
    for _k, (kind, _r) in needed.items():
        kinds[kind] = kinds.get(kind, 0) + 1
    preserve_and_write(out / "hcache_manifest.json", json.dumps({
        "schema": "v075_hcache_manifest_v1", "run_schema": "v075",
        "run_id": RID, "git_sha": git_sha,
        "universe": S.UNIVERSE, "stage": args.stage,
        "inheritance": "NONE — the V7.3C cache files are deleted; "
                       "every key is computed fresh",
        "enumeration": "every (anchor, text-variant) pair, a superset "
                       "of the scheduled visits (the trainer's "
                       "language second stream and dev readouts "
                       "unroll at canonical p0 regardless of the "
                       "visit query)",
        "counts": {"anchors": len(anchors), "keys": len(needed),
                   "computed": len(todo), **kinds},
        "input_sha256": {
            "text_variants.json": tv_sha,
            "v074_anchor_manifest.json": sha256_of(
                S.v074_root() / "anchor_manifest.json")},
        "index_sha256": idx_row["sha256"],
        "index_write": idx_row["action"],
    }, indent=2), args.stage)
    print(f"[hc] wrote {out / 'hcache_index.json'} "
          f"({len(index)} keys, {idx_row['action']}, "
          f"sha {idx_row['sha256'][:12]})", flush=True)
    if idx_row["action"] == "superseded":
        print(f"[hc] PRIOR index preserved at "
              f"{idx_row['superseded_preserved_at']} — a run trained "
              f"against it still has an exact input identity",
              flush=True)


if __name__ == "__main__":
    main()
