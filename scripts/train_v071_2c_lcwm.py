#!/usr/bin/env python
"""V7.1.2c — full correction-aware LCWM (merge + semantic enablement).

Bindings registered in plan_and_progress/2026-08-02.md before launch.

Data: frozen V7.1.1F union (724 rows, physical losses exactly as 2a,
identities preserved) + the 22 frozen V7.1.2b semwin windows
decomposed into their recorded consecutive 10-action segments.

Semwin supervision (canonical stream, full history):
  chained open-loop prediction zt_{k+1} = T(zt_k, a_seg_k) with
  per-boundary latent posterior loss (EMA E(h_boundary)) — boundaries
  that cross milestones are where the posterior moves — plus
  per-segment eef loss and window-level obj loss.

Instruction-dependent heads (semwin rows ONLY — valid per-goal
before/after labels): the V0.6 heads DNext.flips_01/flips_10/
valid_bits are ENABLED (per-subgoal BCE masked to the goal's subgoal
count; no new parameters). Query state built under the (goal,
text_variant) prompt with REGISTERED K=8 history truncation; exactly
ONE query pair per anchor visit (round-robin) so crossed multiplicity
never inflates physical mass.

Variants (matched): lc_full, readout_baseline (constant prompt),
shuffled_lang (fixed SHA task permutation). Plus a closed-form
proprio-only ridge probe. Attribution only.

Selection: min dev (latent+eef+obj+sem) at checkpoint epochs; ONE
frozen checkpoint lc_full_selected.pt. Tensor-only run, no videos.
Output: results/libero_loho_public_v1/2026-08-02_v071_lcwm2c_r1/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lcwm.libero_paths import ensure_project_libero_config  # noqa: E402

ensure_project_libero_config()

from lcwm.v06_model import V06State, ema_update, make_ema  # noqa: E402
from lcwm.v071_loader import V071Loader  # noqa: E402
from scripts.train_v071_2a_lcwm import (Q_SCALE, OBJ_SCALE,  # noqa: E402
                                        huber, goal_objects)

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION = RESULTS / "2026-08-02_v071_union_f1"
SEMWIN = RESULTS / "2026-08-02_v071_semwin_r1"
INIT_CKPT = (RESULTS / "2026-08-02_v071_lcwm_r2" / "checkpoints"
             / "lc_main_selected.pt")
CACHE_2A = RESULTS / "2026-08-02_v071_lcwm_r1" / "train"
STAGE, RUN_ID = "lcwm2c", "r1"
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
TASK_OBJ_MM, TASK_OBJ_UPW, CKPT_EVERY = 0.002, 4.0, 5
K_HIST = 8
SEM_W = 1.0
TASKS = ["loho_t1_drawer", "loho_t2_basket3", "loho_t3_tray",
         "loho_t4_tray", "loho_t5_drawer_cabinet"]


def sha_seed(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(
        payload.encode()).digest()[:8], "big") & ((1 << 63) - 1)


class HCacheMulti:
    """LRU cache over an ordered list of dirs (first hit wins)."""

    def __init__(self, dirs):
        self.dirs = [Path(d) for d in dirs]
        self.lru = {}

    def get(self, key, device):
        if key not in self.lru:
            if len(self.lru) > 200:
                self.lru.clear()
            for dd in self.dirs:
                p = dd / f"{key}.pt"
                if p.exists():
                    self.lru[key] = torch.load(p, weights_only=False)
                    break
            else:
                raise KeyError(key)
        d = self.lru[key]
        return (d["h"][None].float().to(device),
                d["mask"][None].to(device))


def load_semwin_rows():
    """Lightweight semwin rows (frames dropped after boundary obs are
    captured for the cache job list)."""
    rows, cache_obs = [], {}
    for p in sorted((SEMWIN / "shards").glob("*.pt")):
        s = torch.load(p, weights_only=False)
        cum = [0]
        for seg in s["segments"]:
            cum.append(cum[-1] + seg["actions"])
        for k, b in enumerate(cum):
            cache_obs[f"{s['window_id']}__b{b}"] = s["frames"][b]
        rows.append({
            "window_id": s["window_id"], "task": s["task"],
            "split": s["split"], "source_id": s["source_id"],
            "decision": s["decision"], "tranche": s["tranche"],
            "branch": s["branch"],
            "goal_ids": s["goal_ids"],
            "canonical_goal": s["canonical_goal"],
            "cum": cum,
            "actions_env": np.asarray(s["actions_env"]),
            "eef_seq": [np.asarray(e) for e in s["eef_seq"]],
            "valid_seq": {g: [list(v) for v in vs]
                          for g, vs in s["valid_seq"].items()},
            "obj_before": np.asarray(s["obj_before"]),
            "obj_after": np.asarray(s["obj_after"]),
        })
    return rows, cache_obs


def seg_labels(row, gid):
    """Per-segment (v_end, flip01, flip10) label triples under gid."""
    vs = row["valid_seq"][gid]
    out = []
    for k in range(len(row["cum"]) - 1):
        v0 = np.asarray(vs[row["cum"][k]], dtype=np.float32)
        v1 = np.asarray(vs[row["cum"][k + 1]], dtype=np.float32)
        out.append((v1, (v1 > v0).astype(np.float32),
                    (v1 < v0).astype(np.float32)))
    return out


@torch.no_grad()
def build_ext_cache(runner, jobs, cache_dir):
    """jobs: key -> (obs, lang). Computes only missing keys."""
    from lcwm.sampler import prefix_forward
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in cache_dir.glob("*.pt")}
    todo = [(k, v) for k, v in sorted(jobs.items()) if k not in done]
    print(f"[cache:{cache_dir.name}] {len(jobs)} keys, "
          f"{len(todo)} to compute", flush=True)
    for i, (key, (obs, lang)) in enumerate(todo):
        batch = runner._obs_to_policy_batch(obs, lang)
        prefix = prefix_forward(runner.policy, batch)
        tmp = cache_dir / f"{key}.tmp"
        torch.save({"h": prefix.hidden[0].half().cpu(),
                    "mask": prefix.pad_masks[0].bool().cpu()}, tmp)
        tmp.replace(cache_dir / f"{key}.pt")
        if (i + 1) % 200 == 0:
            print(f"  [cache:{cache_dir.name}] {i + 1}/{len(todo)}",
                  flush=True)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from lcwm.probe_data import CONSTANT_PROMPT
    from lcwm.seq_prefix_cache import normalize_actions
    from lcwm.v067_lineage import sha256_file

    device = torch.device("cuda")
    torch.manual_seed(0)
    date = subprocess.run(["date", "+%F"], capture_output=True,
                          text=True, env={"TZ": "America/Chicago"}
                          ).stdout.strip()
    root = None
    for p in sorted(RESULTS.glob(f"*_v071_{STAGE}_{RUN_ID}")):
        root = p
    root = root or RESULTS / f"{date}_v071_{STAGE}_{RUN_ID}"
    for sub in ("checkpoints", "train", "analysis"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    ref = torch.load(Path("/home/stargazer/Desktop/vla_wm/datasets"
                          "/seq_prefix_cache_v1/task0_demo0.pt"),
                     weights_only=False)
    ref_mean, ref_std = ref["action_mean"], ref["action_std_eps"]

    loader = V071Loader(UNION)
    union = json.loads((UNION / "union_manifest.json").read_text())
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    tv = json.loads((SEMWIN / "text_variants.json").read_text())
    goal_texts = defaultdict(dict)
    for r in tv["rows"]:
        goal_texts[r["goal_id"]][r["text_variant_id"]] = r["language"]
    sources_rows = {}
    for sid, bind in union["source_histories"].items():
        sources_rows[sid] = torch.load(
            bind["path"], weights_only=False)["rows"]
    sem_rows, sem_cache_obs = load_semwin_rows()
    # fixed SHA task permutation for the shuffled-language variant
    g = torch.Generator().manual_seed(
        sha_seed("v071_lcwm2c_r1|shuffle_tasks"))
    while True:
        perm = torch.randperm(5, generator=g).tolist()
        if all(perm[i] != i for i in range(5)):
            break
    canon_lang = {t: goal_texts[
        goal_manifest["tasks"][t]["canonical_goal_spec_id"]][
        goal_manifest["tasks"][t]["canonical_goal_spec_id"] + "_p0"]
        for t in TASKS}
    shuffle_map = {canon_lang[TASKS[i]]: canon_lang[TASKS[perm[i]]]
                   for i in range(5)}
    task_of_sid = {sid: torch.load(
        union["source_histories"][sid]["path"],
        weights_only=False)["task"] for sid in sources_rows}

    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=REPO_ROOT).stdout.strip()
    manifest = {
        "schema": "v071_lcwm2c_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE, "run_date": date,
        "git_sha": git_sha,
        "init_checkpoint": {"path": str(INIT_CKPT),
                            "sha256": sha256_file(INIT_CKPT)},
        "union_manifest_sha256": sha256_file(
            UNION / "union_manifest.json"),
        "semwin_manifest_sha256": sha256_file(
            SEMWIN / "run_manifest.json"),
        "optimizer": {"lr": LR, "wd": WD, "epochs": EPOCHS,
                      "grad_norm": GRAD_NORM, "tbptt": TBPTT,
                      "ckpt_every": CKPT_EVERY},
        "semantic": {"heads": "DNext.flips_01/flips_10/valid_bits "
                              "(existing V0.6 params, enabled)",
                     "query_history_K": K_HIST, "sem_weight": SEM_W,
                     "one_query_per_visit": True},
        "task_shuffle_perm": perm,
        "variants": ["lc_full", "readout_baseline", "shuffled_lang"],
        "video": "tensor-only run; no simulator rollouts",
    }
    mp = root / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(manifest, indent=2))

    # ---- cache jobs -----------------------------------------------------
    def query_pairs(task):
        gids = list(goal_manifest["tasks"][task]["goal_specs"])
        return [(gid, pv) for gid in sorted(gids)
                for pv in ("p0", "p1", "p2")]

    def sem_stream_jobs(prompt_of, variant):
        """Canonical-stream extension jobs + query-stream jobs.
        prompt_of maps a canonical task language; queries use the goal
        text for lc_full but prompt_of(canonical) for baselines."""
        jobs_main, jobs_query = {}, {}
        for row in sem_rows:
            sid, d = row["source_id"], row["decision"]
            lang_c = prompt_of(
                canon_lang[task_of_sid[sid]])
            for dd in range(d):
                jobs_main[f"{sid}__dec{dd}"] = (
                    sources_rows[sid][dd]["obs"], lang_c)
            for b in row["cum"]:
                key = f"{row['window_id']}__b{b}"
                jobs_main[key] = (sem_cache_obs[key], lang_c)
            for gid, pv in query_pairs(row["task"]):
                if variant == "lc_full":
                    qlang = goal_texts[gid][f"{gid}_{pv}"]
                else:
                    qlang = lang_c
                qp = f"q_{gid}_{pv}"
                for dd in range(max(0, d - K_HIST), d):
                    jobs_query[f"{qp}__{sid}__dec{dd}"] = (
                        sources_rows[sid][dd]["obs"], qlang)
                jobs_query[f"{qp}__{row['window_id']}__b0"] = (
                    sem_cache_obs[f"{row['window_id']}__b0"], qlang)
        return jobs_main, jobs_query

    def union_jobs(prompt_of):
        jobs = {}
        for tid in (loader.ordered_ids("train")
                    + loader.ordered_ids("dev")):
            r = loader.by_id[tid]
            sid = r["source_id"]
            lang = prompt_of(canon_lang[task_of_sid[sid]])
            for dd in range(r["decision"] + 1):
                jobs[f"{sid}__dec{dd}"] = (
                    sources_rows[sid][dd]["obs"], lang)
            jobs[f"{tid}__post"] = (None, lang)  # filled below
        return jobs

    variants_cfg = {
        "lc_full": {"prompt_of": lambda s: s,
                    "base_dirs": [CACHE_2A / "h_canonical"]},
        "readout_baseline": {
            "prompt_of": lambda s: CONSTANT_PROMPT,
            "base_dirs": [CACHE_2A / "h_constant"]},
        "shuffled_lang": {
            "prompt_of": lambda s: shuffle_map.get(s, s),
            "base_dirs": []},
    }

    need_runner = False
    for vname, vc in variants_cfg.items():
        vc["ext_dir"] = root / "train" / f"ext_{vname}"
        vc["query_dir"] = root / "train" / f"query_{vname}"
        jm, jq = sem_stream_jobs(vc["prompt_of"], vname)
        vc["_jm"], vc["_jq"] = jm, jq
        have = set()
        for dd in (vc["base_dirs"] + [vc["ext_dir"], vc["query_dir"]]):
            if Path(dd).exists():
                have |= {p.stem for p in Path(dd).glob("*.pt")}
        if (set(jm) | set(jq)) - have:
            need_runner = True
        if vname == "shuffled_lang" and not (
                vc["ext_dir"] / "_union_done").exists():
            need_runner = True

    if need_runner:
        from lcwm.chassis import Pi05Runner
        runner = Pi05Runner(suite_name="libero_10")
        for vname, vc in variants_cfg.items():
            jm, jq = vc["_jm"], vc["_jq"]
            have = set()
            for dd in vc["base_dirs"]:
                if Path(dd).exists():
                    have |= {p.stem for p in Path(dd).glob("*.pt")}
            build_ext_cache(runner,
                            {k: v for k, v in jm.items()
                             if k not in have},
                            vc["ext_dir"])
            build_ext_cache(runner, jq, vc["query_dir"])
            if vname == "shuffled_lang" and not (
                    vc["ext_dir"] / "_union_done").exists():
                uj = union_jobs(vc["prompt_of"])
                for tid in list(uj):
                    if tid.endswith("__post"):
                        tr = loader.load_transition(tid[:-6])
                        uj[tid] = (tr["frames"][-1], uj[tid][1])
                build_ext_cache(runner, uj, vc["ext_dir"])
                (vc["ext_dir"] / "_union_done").write_text("1")
        del runner
        torch.cuda.empty_cache()

    # ---- proprio-only ridge probe (closed-form, physical only) ---------
    def proprio_probe():
        Xs, Ys = {"train": [], "dev": []}, {"train": [], "dev": []}
        for split in ("train", "dev"):
            for tid in loader.ordered_ids(split):
                tr = loader.load_transition(tid)
                if tr["steps"] == 0:
                    continue
                a = np.zeros((10, 7), dtype=np.float32)
                a[:tr["actions_env"].shape[0]] = tr["actions_env"]
                Xs[split].append(np.concatenate(
                    [np.asarray(tr["eef_seq"][0]), a.flatten()]))
                Ys[split].append(np.asarray(tr["eef_seq"][-1])
                                 - np.asarray(tr["eef_seq"][0]))
        Xtr = np.stack(Xs["train"])
        Ytr = np.stack(Ys["train"])
        Xtr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
        W = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1]),
                            Xtr.T @ Ytr)
        Xdv = np.stack(Xs["dev"])
        Xdv = np.concatenate([Xdv, np.ones((len(Xdv), 1))], axis=1)
        pred = torch.from_numpy(Xdv @ W).float()
        tgt = torch.from_numpy(np.stack(Ys["dev"])).float()
        return float(huber(pred, tgt, Q_SCALE.cpu()))

    probe_dev = proprio_probe()
    (root / "analysis" / "proprio_probe.json").write_text(json.dumps(
        {"dev_eef_huber": probe_dev,
         "note": "ridge on (eef_anchor, actions_env) union rows; "
                 "attribution only"}, indent=2))
    print(f"[probe] proprio-only dev eef huber: {probe_dev:.4f}",
          flush=True)

    q_scale = Q_SCALE.to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    sem_train = [r for r in sem_rows if r["split"] == "train"]
    sem_dev = [r for r in sem_rows if r["split"] == "dev"]

    def n_sub(gid):
        for t in goal_manifest["tasks"].values():
            if gid in t["goal_specs"]:
                return len(t["goal_specs"][gid]["ordered_subgoals"])
        raise KeyError(gid)

    # frozen semantic trivial reference: constant train-prior predictor
    def sem_prior_reference():
        pos = {"v": [], "f01": [], "f10": []}
        for row in sem_train:
            for gid in row["goal_ids"]:
                for (v1, f01, f10) in seg_labels(row, gid):
                    n = n_sub(gid)
                    pos["v"].append(v1[:n])
                    pos["f01"].append(f01[:n])
                    pos["f10"].append(f10[:n])
        priors = {k: float(np.concatenate(v).mean())
                  for k, v in pos.items()}
        eps = 1e-4
        logits = {k: float(np.log((p + eps) / (1 - p + eps)))
                  for k, p in priors.items()}
        losses = []
        for row in sem_dev:
            for gid in row["goal_ids"]:
                for (v1, f01, f10) in seg_labels(row, gid):
                    n = n_sub(gid)
                    for k, lab in (("v", v1), ("f01", f01),
                                   ("f10", f10)):
                        t_ = torch.from_numpy(lab[:n])
                        l_ = torch.full_like(t_, logits[k])
                        losses.append(float(bce(l_, t_)))
        return {"train_priors": priors,
                "dev_bce_prior_predictor": float(np.mean(losses))}

    sem_ref = sem_prior_reference()
    (root / "analysis" / "sem_reference.json").write_text(
        json.dumps(sem_ref, indent=2))
    print(f"[sem-ref] {sem_ref}", flush=True)

    def norm_chunk(a_env):
        ae = torch.from_numpy(np.asarray(a_env)).float()
        cn = normalize_actions(ae, ref_mean, ref_std)[None].to(device)
        am = (torch.arange(10, device=device)[None] < ae.shape[0])
        if cn.shape[1] < 10:
            cn = torch.cat([cn, torch.zeros(
                1, 10 - cn.shape[1], 7, device=device)], dim=1)
        return cn, am

    def train_variant(vname):
        vc = variants_cfg[vname]
        hc = HCacheMulti(vc["base_dirs"]
                         + [vc["ext_dir"], vc["query_dir"]])
        model = V06State().to(device)
        bundle = torch.load(INIT_CKPT, weights_only=False)
        model.load_state_dict(bundle["model"])
        ema = make_ema(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WD)

        # ---- union machinery (as 2a) --------------------------------
        def anchor_groups(split):
            groups = defaultdict(list)
            for tid in loader.ordered_ids(split):
                groups[loader.by_id[tid]["anchor"]].append(tid)
            return groups

        groups_train = anchor_groups("train")
        groups_dev = anchor_groups("dev")
        task_obj_flag = {}
        for split in ("train", "dev"):
            for anchor, tids in anchor_groups(split).items():
                trs = {t: loader.load_transition(t) for t in tids}
                u0 = next((trs[t] for t in tids
                           if loader.by_id[t]["branch_key"]
                           in ("u0", "0")), None)
                task = loader.by_id[tids[0]]["task"]
                objs = goal_objects(goal_manifest, task)
                names = loader.by_id[tids[0]]["body_order"]
                for t in tids:
                    flag = False
                    if u0 is not None:
                        ddv = np.abs(np.asarray(trs[t]["obj_after"])
                                     - np.asarray(u0["obj_after"]))
                        for bi, nm in enumerate(names):
                            if nm in objs and bi < ddv.shape[0] \
                                    and ddv[bi].max() >= TASK_OBJ_MM:
                                flag = True
                    task_obj_flag[t] = flag

        def union_anchor_losses(anchor, tids, train=True):
            r0 = loader.by_id[tids[0]]
            sid, d = r0["source_id"], r0["decision"]
            z = None
            for dd in range(d + 1):
                h, m = hc.get(f"{sid}__dec{dd}", device)
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = sources_rows[sid][dd - 1]
                    a = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
                if train and dd > 0 and dd % TBPTT == 0:
                    z = z.detach()
            losses, cats = [], []
            obj_before = np.asarray(sources_rows[sid][d]["obj_before"])
            active = [t for t in tids
                      if loader.load_transition(t)["steps"] > 0]
            raw_w = {t: (TASK_OBJ_UPW if task_obj_flag.get(t)
                         else 1.0) for t in active}
            wsum = sum(raw_w.values()) or 1.0
            norm_w = {t: raw_w[t] * len(active) / wsum
                      for t in active}
            for tid in active:
                tr = loader.load_transition(tid)
                a_norm, am = norm_chunk(tr["actions_env"])
                zt = model.predict(z, a_norm, action_mask=am)
                h_p, m_p = hc.get(f"{tid}__post", device)
                with torch.no_grad():
                    z_post = ema.initial_state(h_p, m_p)
                l_lat = torch.nn.functional.mse_loss(
                    zt, z_post.detach())
                out = model.d_next(zt)
                d_eef = (tr["eef_seq"][-1]
                         - tr["eef_seq"][0]).to(device)
                l_eef = huber(out["d_q"][0], d_eef, q_scale)
                d_obj = torch.from_numpy(
                    np.asarray(tr["obj_after"])
                    - obj_before).float().to(device)
                n_obj = d_obj.shape[0]
                w = norm_w[tid]
                l_obj = w * huber(out["d_obj"][0, :n_obj].flatten(),
                                  d_obj.flatten(), OBJ_SCALE)
                losses.append(l_lat + l_eef + l_obj)
                cats.append({"latent": float(l_lat),
                             "eef": float(l_eef),
                             "obj": float(l_obj) / w})
            return losses, cats

        def build_query_state(row, gid, pv, reset_hist=False):
            """Query-stream state at the window anchor (K=8 history
            unless reset_hist)."""
            sid, d = row["source_id"], row["decision"]
            qp = f"q_{gid}_{pv}"
            z = None
            decs = [] if reset_hist else \
                list(range(max(0, d - K_HIST), d))
            for dd in decs:
                h, m = hc.get(f"{qp}__{sid}__dec{dd}", device)
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = sources_rows[sid][dd - 1]
                    a = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
            h, m = hc.get(f"{qp}__{row['window_id']}__b0", device)
            if z is None:
                z = model.initial_state(h, m)
            else:
                prev = sources_rows[sid][d - 1]
                a = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model.step(z, a, h, m, action_mask=am)
            return z

        def sem_query_loss(row, gid, pv, train=True,
                           actions_override=None, reset_hist=False):
            """Query-stream chained prediction + semantic BCE."""
            z = build_query_state(row, gid, pv, reset_hist)
            labels = seg_labels(row, gid)
            n = n_sub(gid)
            acts = (actions_override if actions_override is not None
                    else row["actions_env"])
            total, per_seg_p01 = [], []
            state = z
            for k in range(len(row["cum"]) - 1):
                seg = acts[row["cum"][k]:row["cum"][k + 1]]
                a_norm, am = norm_chunk(seg)
                state = model.predict(state, a_norm, action_mask=am)
                out = model.d_next(state)
                v1, f01, f10 = labels[k]
                lv = bce(out["valid_bits"][0, :n],
                         torch.from_numpy(v1[:n]).to(device))
                l01 = bce(out["flips_01"][0, :n],
                          torch.from_numpy(f01[:n]).to(device))
                l10 = bce(out["flips_10"][0, :n],
                          torch.from_numpy(f10[:n]).to(device))
                total.append(lv + l01 + l10)
                per_seg_p01.append(torch.sigmoid(
                    out["flips_01"][0, :n]).detach().cpu())
            return torch.stack(total).mean(), per_seg_p01

        def sem_canonical_losses(row, train=True):
            """Canonical stream: full history + chained boundary
            latent + per-segment eef + window obj."""
            sid, d = row["source_id"], row["decision"]
            z = None
            for dd in range(d):
                h, m = hc.get(f"{sid}__dec{dd}", device)
                if z is None:
                    z = model.initial_state(h, m)
                else:
                    prev = sources_rows[sid][dd - 1]
                    a = prev["chunk_norm"][None, :10].float() \
                        .to(device)
                    am = (torch.arange(10, device=device)[None]
                          < prev["executed_len"])
                    z = model.step(z, a, h, m, action_mask=am)
                if train and dd > 0 and dd % TBPTT == 0:
                    z = z.detach()
            h, m = hc.get(f"{row['window_id']}__b0", device)
            if z is None:
                z = model.initial_state(h, m)
            else:
                prev = sources_rows[sid][d - 1]
                a = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model.step(z, a, h, m, action_mask=am)
            state, l_lat, l_eef = z, [], []
            for k in range(len(row["cum"]) - 1):
                b0, b1 = row["cum"][k], row["cum"][k + 1]
                a_norm, am = norm_chunk(row["actions_env"][b0:b1])
                state = model.predict(state, a_norm, action_mask=am)
                h_b, m_b = hc.get(f"{row['window_id']}__b{b1}",
                                  device)
                with torch.no_grad():
                    z_b = ema.initial_state(h_b, m_b)
                l_lat.append(torch.nn.functional.mse_loss(
                    state, z_b.detach()))
                d_eef = torch.from_numpy(
                    row["eef_seq"][b1]
                    - row["eef_seq"][b0]).float().to(device)
                l_eef.append(huber(
                    model.d_next(state)["d_q"][0], d_eef, q_scale))
            d_obj = torch.from_numpy(
                row["obj_after"] - row["obj_before"]).float() \
                .to(device)
            n_obj = d_obj.shape[0]
            l_obj = huber(
                model.d_next(state)["d_obj"][0, :n_obj].flatten(),
                d_obj.flatten(), OBJ_SCALE)
            return (torch.stack(l_lat).mean()
                    + torch.stack(l_eef).mean() + l_obj,
                    {"latent": float(torch.stack(l_lat).mean()),
                     "eef": float(torch.stack(l_eef).mean()),
                     "obj": float(l_obj)})

        # rr schedule over union anchors + semwin windows
        def rr_schedule(epoch):
            by_task = defaultdict(lambda: defaultdict(list))
            for anchor, tids in groups_train.items():
                r0 = loader.by_id[tids[0]]
                by_task[r0["task"]][r0["source_id"]].append(
                    ("union", anchor, tids))
            for wi, row in enumerate(sem_train):
                by_task[row["task"]][row["source_id"]].append(
                    ("semwin", row["window_id"], wi))
            queues = []
            for task in sorted(by_task):
                srcs = sorted(by_task[task])
                rot = srcs[epoch % len(srcs):] \
                    + srcs[:epoch % len(srcs)]
                merged, i = [], 0
                while any(i < len(by_task[task][s2]) for s2 in rot):
                    for s2 in rot:
                        if i < len(by_task[task][s2]):
                            merged.append(by_task[task][s2][i])
                    i += 1
                queues.append(merged)
            order, j = [], 0
            while any(j < len(q) for q in queues):
                for q in queues:
                    if j < len(q):
                        order.append(q[j])
                j += 1
            return order

        def dev_pass():
            dv = defaultdict(list)
            with torch.no_grad():
                for anchor, tids in groups_dev.items():
                    _, cats = union_anchor_losses(anchor, tids,
                                                  train=False)
                    for c in cats:
                        for k in ("latent", "eef", "obj"):
                            dv[k].append(c[k])
                for row in sem_dev:
                    _, cats = sem_canonical_losses(row, train=False)
                    for k in ("latent", "eef", "obj"):
                        dv[k].append(cats[k])
                    for gid, pv in query_pairs(row["task"]):
                        ls, _ = sem_query_loss(row, gid, pv,
                                               train=False)
                        dv["sem"].append(float(ls))
            obj_ = float(np.mean(dv["latent"]) + np.mean(dv["eef"])
                         + np.mean(dv["obj"])
                         + SEM_W * np.mean(dv["sem"]))
            return obj_, {k: float(np.mean(v))
                          for k, v in dv.items()}

        logs, ckpt_devs, start_epoch = [], {}, 0
        if args.resume:
            cks = sorted((root / "checkpoints").glob(
                f"{vname}_epoch*.pt"))
            if cks:
                st = torch.load(cks[-1], weights_only=False)
                model.load_state_dict(st["model"])
                ema.load_state_dict(st["ema"])
                optimizer.load_state_dict(st["optimizer"])
                start_epoch = st["epoch"] + 1
                ckpt_devs = {c.name: torch.load(
                    c, weights_only=False)["dev_objective"]
                    for c in cks}
                print(f"[{vname}] resumed at {start_epoch}",
                      flush=True)
        for epoch in range(start_epoch, EPOCHS):
            agg = defaultdict(list)
            for item in rr_schedule(epoch):
                optimizer.zero_grad(set_to_none=True)
                if item[0] == "union":
                    losses, cats = union_anchor_losses(
                        item[1], item[2], train=True)
                    if not losses:
                        continue
                    total = torch.stack(losses).mean()
                    for c in cats:
                        for k in ("latent", "eef", "obj"):
                            agg[k].append(c[k])
                else:
                    row = sem_train[item[2]]
                    phys, cats = sem_canonical_losses(row,
                                                      train=True)
                    pairs = query_pairs(row["task"])
                    gid, pv = pairs[(epoch + item[2]) % len(pairs)]
                    sem, _ = sem_query_loss(row, gid, pv, train=True)
                    total = phys + SEM_W * sem
                    for k in ("latent", "eef", "obj"):
                        agg[f"sw_{k}"].append(cats[k])
                    agg["sw_sem"].append(float(sem))
                assert torch.isfinite(total)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               GRAD_NORM)
                optimizer.step()
                ema_update(ema, model)
            means = {k: float(np.mean(v)) for k, v in agg.items()
                     if v}
            logs.append({"epoch": epoch, **means})
            print(f"[{vname} ep{epoch}] " + " ".join(
                f"{k}={v:.4f}" for k, v in means.items()),
                flush=True)
            if (epoch + 1) % CKPT_EVERY == 0 or (epoch + 1) == EPOCHS:
                dev_obj, dev_detail = dev_pass()
                name = f"{vname}_epoch{epoch:03d}.pt"
                torch.save({"model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "run_schema": "v071", "variant": vname,
                            "epoch": epoch,
                            "dev_objective": dev_obj},
                           root / "checkpoints" / name)
                ckpt_devs[name] = dev_obj
                logs[-1]["dev_objective"] = dev_obj
                logs[-1]["dev_detail"] = dev_detail
                print(f"  [{vname} dev ep{epoch}] {dev_obj:.4f} "
                      f"({dev_detail})", flush=True)
        best = min(ckpt_devs, key=lambda k: (ckpt_devs[k], k))
        sel = torch.load(root / "checkpoints" / best,
                         weights_only=False)
        torch.save(sel, root / "checkpoints"
                   / f"{vname}_selected.pt")

        # ---- held-out report (selected weights) ---------------------
        model.load_state_dict(sel["model"])
        ema.load_state_dict(sel["ema"])
        report = {}
        with torch.no_grad():
            inv_phys, disc, para, act_sens, hist = [], [], [], [], []
            for wi, row in enumerate(sem_dev):
                pairs = query_pairs(row["task"])
                p01 = {}
                for gid, pv in pairs:
                    ls, seg_p = sem_query_loss(row, gid, pv,
                                               train=False)
                    p01[(gid, pv)] = (float(ls), seg_p)
                gids = sorted({g for g, _ in pairs})
                # (b) goal-swap discrimination on flipped subgoals
                for k in range(len(row["cum"]) - 1):
                    for g1 in gids:
                        v1, f01, _ = seg_labels(row, g1)[k]
                        for si in np.nonzero(f01[:n_sub(g1)])[0]:
                            for g0 in gids:
                                if g0 == g1:
                                    continue
                                _, f01_0, _ = seg_labels(row, g0)[k]
                                if si < n_sub(g0) \
                                        and f01_0[si] == 0:
                                    a1 = float(p01[(g1, "p0")][1][k]
                                               [si])
                                    a0 = float(p01[(g0, "p0")][1][k]
                                               [si])
                                    disc.append(a1 > a0)
                # (c) paraphrase invariance
                for gid in gids:
                    for pv in ("p1", "p2"):
                        for k in range(len(row["cum"]) - 1):
                            para.append(float(
                                (p01[(gid, pv)][1][k]
                                 - p01[(gid, "p0")][1][k])
                                .abs().mean()))
                # (d) action sensitivity: rotated dev actions
                other = sem_dev[(wi + 1) % len(sem_dev)]
                na = min(len(row["actions_env"]),
                         len(other["actions_env"]))
                if na >= row["cum"][-1]:
                    aov = other["actions_env"][:row["cum"][-1]]
                    gid = row["canonical_goal"]
                    ls_true = p01[(gid, "p0")][0]
                    ls_rot, _ = sem_query_loss(
                        row, gid, "p0", train=False,
                        actions_override=aov)
                    act_sens.append(float(ls_rot) - float(ls_true))
                # (e) history contribution K=8 vs K=0
                gid = row["canonical_goal"]
                ls_k8 = p01[(gid, "p0")][0]
                ls_k0, _ = sem_query_loss(row, gid, "p0",
                                          train=False,
                                          reset_hist=True)
                hist.append(float(ls_k0) - float(ls_k8))
                # (a) physical invariance under goal swap (d_q diff,
                # same K=8 query streams as training)
                base_gid = row["canonical_goal"]
                dq = {}
                for gid in gids:
                    zq = build_query_state(row, gid, "p0")
                    a_norm, am = norm_chunk(
                        row["actions_env"][:row["cum"][1]])
                    zt = model.predict(zq, a_norm, action_mask=am)
                    dq[gid] = model.d_next(zt)["d_q"][0]
                for gid in gids:
                    if gid != base_gid:
                        inv_phys.append(float(
                            ((dq[gid] - dq[base_gid])
                             / q_scale).abs().mean()))
            report = {
                "a_phys_goal_swap_dq_scaled_absdiff":
                    float(np.mean(inv_phys)) if inv_phys else None,
                "b_goal_swap_discrimination_acc":
                    float(np.mean(disc)) if disc else None,
                "b_n_pairs": len(disc),
                "c_paraphrase_p01_absdiff":
                    float(np.mean(para)) if para else None,
                "d_action_sensitivity_bce_increase":
                    float(np.mean(act_sens)) if act_sens else None,
                "e_history_K0_minus_K8_bce":
                    float(np.mean(hist)) if hist else None,
            }
        (root / "analysis" / f"heldout_{vname}.json").write_text(
            json.dumps(report, indent=2))
        print(f"[{vname}] held-out report: {report}", flush=True)
        return {"logs": logs, "selected": best,
                "dev_objective": ckpt_devs[best],
                "heldout": report}

    results = {}
    for vname in ("lc_full", "readout_baseline", "shuffled_lang"):
        results[vname] = train_variant(vname)
        (root / "analysis" / "train_results.json").write_text(
            json.dumps(results, indent=2))
    print(json.dumps({v: {"selected": r["selected"],
                          "dev": r["dev_objective"]}
                      for v, r in results.items()}, indent=1),
          flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
