#!/usr/bin/env python
"""V7.1.2a — physical/latent LCWM bootstrap on the frozen V7.1.1F
union (first actual model-learning run of V7.1).

Model: V06State initialized from the registered predictive lineage
(v069_predictive/checkpoint_selected.pt; NEVER the N1 smoke
checkpoints). z_t^l = E(H_t) with the full-prompt pi0.5 prefix
carrying l; z_hat_{t+c} = T(z_t, u_{t:t+c-1}), c=10.

Losses (semantic heads DISABLED — current predicate-change labels are
all-negative and are not enabled or selected on):
  latent_post   mse(T(z_anchor, u_branch), EMA_E(h_posterior))
                — real posterior observations, per executed branch;
  phys_eef      huber d_q head vs measured eef delta (branch);
  phys_obj      huber d_obj head vs measured all-object delta, with
                task-object-effect rows upweighted x4 INSIDE their
                anchor (sibling-relative >= 2 mm vs the anchor u_0
                endpoint);
  recurrence    z unrolled from episode reset along the bound source
                history (genuine consecutive decisions only).

Matched baseline: identical data/capacity/sampler/schedule but E
consumes CONSTANT-prompt prefixes (task-agnostic/readout-only). It is
an attribution baseline; an offline tie cannot redirect the LC
mainline.

Selection (frozen): min dev objective = mean dev latent_post MSE
+ mean dev phys errors (eef + all-object), computed every 5 epochs;
task-object dev support is ABSENT (0 rows) and reported as such.
Tensor-only run: no simulator, no videos required.

Output: results/libero_loho_public_v1/<DATE>_v071_lcwm_r1/
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

RESULTS = REPO_ROOT / "results" / "libero_loho_public_v1"
UNION = RESULTS / "2026-08-02_v071_union_f1"
PRED_CKPT = RESULTS / "v069_predictive" / "checkpoint_selected.pt"
STAGE, RUN_ID = "lcwm", "r1"
Q_SCALE = torch.tensor([7.344e-3] * 3 + [5.451e-3] * 4
                       + [2.598e-4] * 2)
OBJ_SCALE = 1.175e-3
HUBER = 4.0
LR, WD, EPOCHS, GRAD_NORM, TBPTT = 3e-4, 1e-4, 25, 1.0, 16
TASK_OBJ_MM = 0.002
TASK_OBJ_UPW = 4.0
CKPT_EVERY = 5


def huber(p, t, scale):
    return torch.nn.functional.huber_loss(p / scale, t / scale,
                                          delta=HUBER)


def goal_objects(manifest, task):
    objs = set()
    for spec in manifest["tasks"][task]["goal_specs"].values():
        for sg in spec["ordered_subgoals"]:
            parts = sg.split()
            if parts[0] in ("place", "pick_up"):
                objs.add(parts[1])
    return objs


@torch.no_grad()
def build_h_cache(runner, loader, cache_dir, variant, prompt_of):
    """fp16 prefix hidden cache on disk, one file per unique obs key."""
    from lcwm.sampler import prefix_forward
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed = {}   # key -> (obs_getter args)
    src_cache = {}

    union = json.loads((UNION / "union_manifest.json").read_text())

    def src(sid):
        if sid not in src_cache:
            src_cache[sid] = torch.load(
                Path(union["source_histories"][sid]["path"]),
                weights_only=False)
        return src_cache[sid]

    for tid in (loader.ordered_ids("train")
                + loader.ordered_ids("dev")):
        r = loader.by_id[tid]
        sid = r["source_id"]
        for d in range(r["decision"] + 1):
            needed[f"{sid}__dec{d}"] = ("src", sid, d)
        needed[f"{tid}__post"] = ("post", tid, None)
    done = {p.stem for p in cache_dir.glob("*.pt")}
    todo = [(k, v) for k, v in sorted(needed.items())
            if k not in done]
    print(f"[h-cache:{variant}] {len(needed)} keys, "
          f"{len(todo)} to compute", flush=True)
    for i, (key, spec) in enumerate(todo):
        if spec[0] == "src":
            _, sid, d = spec
            row = src(sid)["rows"][d]
            obs = row["obs"]
            lang = prompt_of(src(sid)["language_canonical"])
        else:
            _, tid, _ = spec
            tr = loader.load_transition(tid)
            obs = tr["frames"][-1]
            lang = prompt_of(src(loader.by_id[tid]["source_id"])
                             ["language_canonical"])
        batch = runner._obs_to_policy_batch(obs, lang)
        prefix = prefix_forward(runner.policy, batch)
        tmp = cache_dir / f"{key}.tmp"
        torch.save({"h": prefix.hidden[0].half().cpu(),
                    "mask": prefix.pad_masks[0].bool().cpu()}, tmp)
        tmp.replace(cache_dir / f"{key}.pt")
        if (i + 1) % 200 == 0:
            print(f"  [h-cache:{variant}] {i + 1}/{len(todo)}",
                  flush=True)


class HCache:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        self.lru = {}

    def get(self, key, device):
        if key not in self.lru:
            if len(self.lru) > 200:
                self.lru.clear()
            d = torch.load(self.dir / f"{key}.pt",
                           weights_only=False)
            self.lru[key] = d
        d = self.lru[key]
        return (d["h"][None].float().to(device),
                d["mask"][None].to(device))


def train_variant(variant, cache_dir, loader, device, out_root,
                  goal_manifest, sources_rows):
    model = V06State().to(device)
    bundle = torch.load(PRED_CKPT, weights_only=False)
    assert bundle.get("run_schema") == "v069"
    model.load_state_dict(bundle["model"])
    ema = make_ema(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WD)
    hc = HCache(cache_dir)
    q_scale = Q_SCALE.to(device)

    # group unique transitions per anchor; mark task-object rows
    def anchor_groups(split):
        groups = defaultdict(list)
        for tid in loader.ordered_ids(split):
            groups[loader.by_id[tid]["anchor"]].append(tid)
        return groups

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
                    dd = np.abs(np.asarray(trs[t]["obj_after"])
                                - np.asarray(u0["obj_after"]))
                    for bi, nm in enumerate(names):
                        if nm in objs and bi < dd.shape[0] \
                                and dd[bi].max() >= TASK_OBJ_MM:
                            flag = True
                task_obj_flag[t] = flag
    n_to_train = sum(1 for t, f in task_obj_flag.items()
                     if f and loader.by_id[t]["split"] == "train")
    n_to_dev = sum(1 for t, f in task_obj_flag.items()
                   if f and loader.by_id[t]["split"] == "dev")
    print(f"[{variant}] task-object rows: {n_to_train} train / "
          f"{n_to_dev} dev (dev support absent="
          f"{n_to_dev == 0})", flush=True)

    def anchor_losses(anchor, tids, train=True):
        r0 = loader.by_id[tids[0]]
        sid, d = r0["source_id"], r0["decision"]
        z = None
        for dd in range(d + 1):
            h, m = hc.get(f"{sid}__dec{dd}", device)
            if z is None:
                z = model.initial_state(h, m)
            else:
                prev = sources_rows[sid][dd - 1]
                a = prev["chunk_norm"][None, :10].float().to(device)
                am = (torch.arange(10, device=device)[None]
                      < prev["executed_len"])
                z = model.step(z, a, h, m, action_mask=am)
            if train and dd > 0 and dd % TBPTT == 0:
                z = z.detach()
        losses, cats = [], []
        anchor_row = sources_rows[sid][d]
        obj_before = np.asarray(anchor_row["obj_before"])
        for tid in tids:
            tr = loader.load_transition(tid)
            if tr["steps"] == 0:
                continue
            ae = torch.from_numpy(tr["actions_env"]).float()
            from lcwm.seq_prefix_cache import normalize_actions
            a_norm = normalize_actions(
                ae, REF_MEAN, REF_STD)[None].to(device)
            am = (torch.arange(10, device=device)[None]
                  < tr["steps"])
            if a_norm.shape[1] < 10:
                pad = torch.zeros(1, 10 - a_norm.shape[1], 7,
                                  device=device)
                a_norm = torch.cat([a_norm, pad], dim=1)
            zt = model.predict(z, a_norm, action_mask=am)
            h_post, m_post = hc.get(f"{tid}__post", device)
            with torch.no_grad():
                z_post = ema.initial_state(h_post, m_post)
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
            w = TASK_OBJ_UPW if task_obj_flag.get(tid) else 1.0
            l_obj = w * huber(
                out["d_obj"][0, :n_obj].flatten(),
                d_obj.flatten(), OBJ_SCALE)
            losses.append(l_lat + l_eef + l_obj)
            cats.append({
                "latent": float(l_lat), "eef": float(l_eef),
                "obj": float(l_obj) / w,
                "task_obj": bool(task_obj_flag.get(tid)),
                "servo": r0 is not None and
                loader.by_id[tid]["provenance"]
                == "scripted_servo"})
        return losses, cats

    groups_train = anchor_groups("train")
    groups_dev = anchor_groups("dev")
    logs, ckpt_devs = [], {}
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        order = list(groups_train.items())
        rng.shuffle(order)
        agg = defaultdict(list)
        for anchor, tids in order:
            optimizer.zero_grad(set_to_none=True)
            losses, cats = anchor_losses(anchor, tids, train=True)
            if not losses:
                continue
            total = torch.stack(losses).mean()
            assert torch.isfinite(total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           GRAD_NORM)
            optimizer.step()
            ema_update(ema, model)
            for c in cats:
                agg["latent"].append(c["latent"])
                agg["eef"].append(c["eef"])
                (agg["obj_task"] if c["task_obj"]
                 else agg["obj_generic"]).append(c["obj"])
                (agg["servo"] if c["servo"]
                 else agg["non_servo"]).append(c["latent"])
        means = {k: float(np.mean(v)) for k, v in agg.items() if v}
        logs.append({"epoch": epoch, **means})
        print(f"[{variant} ep{epoch}] " + " ".join(
            f"{k}={v:.4f}" for k, v in means.items()), flush=True)
        if (epoch + 1) % CKPT_EVERY == 0 or (epoch + 1) == EPOCHS:
            with torch.no_grad():
                dv = defaultdict(list)
                for anchor, tids in groups_dev.items():
                    losses, cats = anchor_losses(anchor, tids,
                                                 train=False)
                    for c in cats:
                        dv["latent"].append(c["latent"])
                        dv["eef"].append(c["eef"])
                        dv["obj"].append(c["obj"])
                dev_obj = float(np.mean(dv["latent"])
                                + np.mean(dv["eef"])
                                + np.mean(dv["obj"]))
            name = f"{variant}_epoch{epoch:03d}.pt"
            torch.save({"model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "run_schema": "v071", "variant": variant,
                        "epoch": epoch, "dev_objective": dev_obj},
                       out_root / "checkpoints" / name)
            ckpt_devs[name] = dev_obj
            logs[-1]["dev_objective"] = dev_obj
            logs[-1]["dev_detail"] = {
                k: float(np.mean(v)) for k, v in dv.items()}
            print(f"  [{variant} dev ep{epoch}] {dev_obj:.4f} "
                  f"({logs[-1]['dev_detail']})", flush=True)
    best = min(ckpt_devs, key=lambda k: (ckpt_devs[k], k))
    sel = torch.load(out_root / "checkpoints" / best,
                     weights_only=False)
    torch.save(sel, out_root / "checkpoints"
               / f"{variant}_selected.pt")
    return {"logs": logs, "selected": best,
            "dev_objective": ckpt_devs[best],
            "task_obj_rows": {"train": n_to_train,
                              "dev": n_to_dev}}


REF_MEAN = REF_STD = None


def main() -> None:
    global REF_MEAN, REF_STD
    from lcwm.chassis import Pi05Runner
    from lcwm.probe_data import CONSTANT_PROMPT

    device = torch.device("cuda")
    torch.manual_seed(0)
    date = subprocess.run(["date", "+%F"], capture_output=True,
                          text=True,
                          env={"TZ": "America/Chicago"}
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
    REF_MEAN, REF_STD = ref["action_mean"], ref["action_std_eps"]

    loader = V071Loader(UNION)
    union = json.loads((UNION / "union_manifest.json").read_text())
    goal_manifest = json.loads(
        (RESULTS / "goal_spec_manifest_v067.json").read_text())
    sources_rows = {}
    for sid, bind in union["source_histories"].items():
        sources_rows[sid] = torch.load(
            bind["path"], weights_only=False)["rows"]

    from lcwm.v067_lineage import sha256_file
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
        cwd=REPO_ROOT).stdout.strip()
    manifest = {
        "schema": "v071_lcwm_manifest_v1", "run_schema": "v071",
        "run_id": RUN_ID, "stage": STAGE, "run_date": date,
        "git_sha": git_sha,
        "init_checkpoint": {"path": str(PRED_CKPT),
                            "sha256": sha256_file(PRED_CKPT)},
        "union_manifest_sha256": sha256_file(
            UNION / "union_manifest.json"),
        "optimizer": {"lr": LR, "wd": WD, "epochs": EPOCHS,
                      "grad_norm": GRAD_NORM, "tbptt": TBPTT,
                      "ckpt_every": CKPT_EVERY},
        "task_object_upweight": TASK_OBJ_UPW,
        "semantic_heads": "DISABLED (all-negative labels not "
                          "enabled or selected on)",
        "selection": "min dev latent+eef+obj at ckpt epochs; "
                     "task-object dev support ABSENT (0 rows)",
        "variants": ["lc_main (canonical prompts)",
                     "readout_baseline (constant prompt)"],
        "video": "tensor-only run; no simulator rollouts",
    }
    mp = root / "run_manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(manifest, indent=2))

    runner = Pi05Runner(suite_name="libero_10")
    build_h_cache(runner, loader, root / "train" / "h_canonical",
                  "canonical", lambda lang: lang)
    build_h_cache(runner, loader, root / "train" / "h_constant",
                  "constant", lambda lang: CONSTANT_PROMPT)
    del runner
    torch.cuda.empty_cache()

    results = {}
    results["lc_main"] = train_variant(
        "lc_main", root / "train" / "h_canonical", loader, device,
        root, goal_manifest, sources_rows)
    results["readout_baseline"] = train_variant(
        "readout_baseline", root / "train" / "h_constant", loader,
        device, root, goal_manifest, sources_rows)
    (root / "analysis" / "train_results.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps({v: {"selected": r["selected"],
                          "dev": r["dev_objective"]}
                      for v, r in results.items()}, indent=1),
          flush=True)
    print(f"-> {root}", flush=True)


if __name__ == "__main__":
    main()
