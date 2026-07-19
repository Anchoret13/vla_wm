"""Per-scene BDDL predicate-pool inventory (plan §2 GPU-free lane).

Walks site-packages/libero/libero/bddl_files/<suite>/*.bddl for all 5 suites
(libero_spatial, libero_object, libero_goal, libero_10, libero_90 = 130 tasks),
parses each problem file (preferring the installed `bddl` package's tokenizer,
falling back to a local s-expression scanner), and extracts per task:

  - objects   (name, category) and fixtures (name, category)
  - regions   (name, target, ranges present?)
  - init predicates  ((:init ...)  — predicate name + args)
  - goal predicates  ((:goal ...) — flattened through and/or/not, name + args)

Aggregates per suite AND per scene (the "predicate pool" a world-model verifier
must cover for that scene): predicate vocabulary with counts, object/fixture
category vocabulary, per-task goal-arity stats.

Run (CPU only):
    PYTHONPATH=/home/stargazer/Desktop/vla_wm/vla_wm \
    /home/stargazer/miniconda3/envs/vf0s/bin/python -m lcwm.bddl_inventory

Output: results/bddl_inventory.json
Note: the bddl tokenizer lowercases everything, so predicate/category names in
the output are lowercase (e.g. `turnon`, `on`) regardless of file spelling.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lcwm.libero_paths import ensure_project_libero_config

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
EXPECTED_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
LOGICAL_OPS = {"and", "or", "not"}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "bddl_inventory.json"


# --------------------------------------------------------------------------- #
# Tokenizing: prefer the installed bddl package's scanner, fall back to local. #
# --------------------------------------------------------------------------- #
def _local_scan_tokens(filename: str):
    """Minimal s-expression tokenizer mirroring bddl.parsing.scan_tokens."""
    raw = Path(filename).read_text()
    text = re.sub(r";.*$", "", raw, flags=re.MULTILINE).lower()
    stack, tokens = [], []
    for t in re.findall(r"[()]|[^\s()]+", text):
        if t == "(":
            stack.append(tokens)
            tokens = []
        elif t == ")":
            if not stack:
                raise ValueError("Missing open parenthesis")
            inner, tokens = tokens, stack.pop()
            tokens.append(inner)
        else:
            tokens.append(t)
    if stack:
        raise ValueError("Missing close parenthesis")
    if len(tokens) != 1:
        raise ValueError("Malformed expression")
    return tokens[0]


try:
    from bddl.parsing import scan_tokens as _scan_tokens

    TOKENIZER = "bddl.parsing.scan_tokens"

    def scan(filename: str):
        return _scan_tokens(filename=filename)

except ImportError:  # pragma: no cover - vf0s ships bddl
    TOKENIZER = "local s-expression fallback"
    scan = _local_scan_tokens


# --------------------------------------------------------------------------- #
# Per-file extraction                                                          #
# --------------------------------------------------------------------------- #
def _typed_list(group):
    """Parse `name+ - category` runs, accumulating (not overwriting) repeats."""
    out = []  # list of (name, category)
    pending = []
    items = list(group[1:])  # drop the ':objects' / ':fixtures' tag
    while items:
        tok = items.pop(0)
        if tok == "-":
            category = items.pop(0)
            out.extend((name, category) for name in pending)
            pending = []
        else:
            pending.append(tok)
    out.extend((name, "untyped") for name in pending)
    return out


def _parse_regions(group):
    regions = []
    for entry in group[1:]:
        name, target, n_ranges, has_yaw = entry[0], None, 0, False
        for attr in entry[1:]:
            if attr[0] == ":target":
                target = attr[1]
            elif attr[0] == ":ranges":
                n_ranges = len(attr[1])
            elif attr[0] == ":yaw_rotation":
                has_yaw = True
        regions.append(
            {"name": name, "target": target, "full_name": f"{target}_{name}",
             "n_ranges": n_ranges, "has_yaw": has_yaw}
        )
    return regions


def _flatten_predicates(expr, ops_seen: Counter):
    """Flatten a goal/init logical expression into leaf predicates."""
    leaves = []
    if not isinstance(expr, list) or not expr:
        raise ValueError(f"predicate expression is not a list: {expr!r}")
    head = expr[0]
    if head in LOGICAL_OPS:
        ops_seen[head] += 1
        for sub in expr[1:]:
            leaves.extend(_flatten_predicates(sub, ops_seen))
    else:
        if not all(isinstance(a, str) for a in expr):
            raise ValueError(f"non-atomic predicate leaf: {expr!r}")
        leaves.append({"pred": head, "args": expr[1:]})
    return leaves


def parse_bddl_file(path: Path) -> dict:
    tokens = scan(str(path))
    if not (isinstance(tokens, list) and tokens and tokens[0] == "define"):
        raise ValueError("file does not start with (define ...)")

    task = {
        "file": path.name,
        "problem_name": None,
        "language": None,
        "objects": [],
        "fixtures": [],
        "regions": [],
        "obj_of_interest": [],
        "init_predicates": [],
        "goal_predicates": [],
        "goal_ops": {},
        "unrecognized_sections": [],
    }
    for group in tokens[1:]:
        tag = group[0]
        if tag == "problem":
            task["problem_name"] = group[-1]
        elif tag == ":domain":
            task["domain"] = group[-1]
        elif tag == ":language":
            task["language"] = " ".join(group[1:])
        elif tag == ":objects":
            task["objects"] = [{"name": n, "category": c} for n, c in _typed_list(group)]
        elif tag == ":fixtures":
            task["fixtures"] = [{"name": n, "category": c} for n, c in _typed_list(group)]
        elif tag == ":regions":
            task["regions"] = _parse_regions(group)
        elif tag == ":obj_of_interest":
            task["obj_of_interest"] = list(group[1:])
        elif tag == ":init":
            ops = Counter()
            for pred in group[1:]:
                task["init_predicates"].extend(_flatten_predicates(pred, ops))
            if ops:
                task["init_ops"] = dict(ops)
        elif tag == ":goal":
            ops = Counter()
            for expr in group[1:]:
                task["goal_predicates"].extend(_flatten_predicates(expr, ops))
            task["goal_ops"] = dict(ops)
        elif tag in (":requirements", ":scene_properties"):
            pass
        else:
            task["unrecognized_sections"].append(tag)
    return task


def scene_of(task: dict) -> str:
    """Scene key: uppercase filename prefix (KITCHEN_SCENE3, ...) when present,
    else the problem name (single-scene suites like libero_object/goal/spatial)."""
    m = re.match(r"^([A-Z][A-Z0-9_]*?SCENE[0-9]*)_", task["file"])
    if m:
        return m.group(1)
    return task["problem_name"] or "unknown_scene"


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _vocab(counter_total: Counter, counter_tasks: Counter) -> dict:
    return {
        k: {"count": counter_total[k], "tasks": counter_tasks[k]}
        for k in sorted(counter_total)
    }


def aggregate(tasks: list[dict]) -> dict:
    goal_pred_total, goal_pred_tasks = Counter(), Counter()
    init_pred_total, init_pred_tasks = Counter(), Counter()
    obj_cat_total, obj_cat_tasks = Counter(), Counter()
    fix_cat_total, fix_cat_tasks = Counter(), Counter()
    pred_arities = defaultdict(set)
    goal_arities = []
    for t in tasks:
        for where, total_c, task_c in (
            ("goal_predicates", goal_pred_total, goal_pred_tasks),
            ("init_predicates", init_pred_total, init_pred_tasks),
        ):
            seen = set()
            for p in t[where]:
                total_c[p["pred"]] += 1
                seen.add(p["pred"])
                pred_arities[p["pred"]].add(len(p["args"]))
            for name in seen:
                task_c[name] += 1
        for where, total_c, task_c in (
            ("objects", obj_cat_total, obj_cat_tasks),
            ("fixtures", fix_cat_total, fix_cat_tasks),
        ):
            seen = set()
            for entry in t[where]:
                total_c[entry["category"]] += 1
                seen.add(entry["category"])
            for cat in seen:
                task_c[cat] += 1
        goal_arities.append(len(t["goal_predicates"]))

    return {
        "n_tasks": len(tasks),
        "goal_predicate_vocab": _vocab(goal_pred_total, goal_pred_tasks),
        "init_predicate_vocab": _vocab(init_pred_total, init_pred_tasks),
        "object_category_vocab": _vocab(obj_cat_total, obj_cat_tasks),
        "fixture_category_vocab": _vocab(fix_cat_total, fix_cat_tasks),
        "predicate_arities": {k: sorted(v) for k, v in sorted(pred_arities.items())},
        "goal_arity_stats": {
            "per_task": goal_arities,
            "min": min(goal_arities),
            "max": max(goal_arities),
            "mean": round(statistics.mean(goal_arities), 4),
            "histogram": dict(sorted(Counter(goal_arities).items())),
        },
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build_inventory() -> dict:
    ensure_project_libero_config()
    try:
        from libero.libero import get_libero_path

        bddl_root = Path(get_libero_path("bddl_files"))
    except Exception:  # fall back to locating the package directly
        from lcwm.libero_paths import _packaged_libero_root

        bddl_root = _packaged_libero_root() / "bddl_files"

    inventory = {
        "bddl_root": str(bddl_root),
        "tokenizer": TOKENIZER,
        "note": "tokenizer lowercases all symbols (e.g. Turnon -> turnon)",
        "suites": {},
        "errors": [],
    }
    all_tasks = []
    for suite in SUITES:
        suite_dir = bddl_root / suite
        files = sorted(suite_dir.glob("*.bddl"))
        tasks, scenes = [], defaultdict(list)
        for f in files:
            try:
                task = parse_bddl_file(f)
            except Exception as e:  # keep going; report at the end
                inventory["errors"].append({"suite": suite, "file": f.name, "error": str(e)})
                continue
            task["scene"] = scene_of(task)
            tasks.append(task)
            scenes[task["scene"]].append(task)
        all_tasks.extend(tasks)
        inventory["suites"][suite] = {
            "n_files": len(files),
            "expected_files": EXPECTED_COUNTS[suite],
            "aggregate": aggregate(tasks),
            "per_scene": {
                scene: aggregate(ts) for scene, ts in sorted(scenes.items())
            },
            "tasks": tasks,
        }
    inventory["global_aggregate"] = aggregate(all_tasks)
    inventory["n_tasks_total"] = len(all_tasks)
    return inventory


def main() -> int:
    inv = build_inventory()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(inv, indent=1, sort_keys=False))
    print(f"wrote {RESULTS_PATH}")
    for suite, s in inv["suites"].items():
        agg = s["aggregate"]
        print(
            f"{suite}: {agg['n_tasks']}/{s['expected_files']} tasks, "
            f"{len(s['per_scene'])} scenes, "
            f"goal preds {sorted(agg['goal_predicate_vocab'])}, "
            f"obj cats {len(agg['object_category_vocab'])}, "
            f"fixture cats {len(agg['fixture_category_vocab'])}, "
            f"goal arity {agg['goal_arity_stats']['min']}-{agg['goal_arity_stats']['max']} "
            f"(mean {agg['goal_arity_stats']['mean']})"
        )
    if inv["errors"]:
        print(f"ERRORS: {inv['errors']}")
        return 1
    if inv["n_tasks_total"] != sum(EXPECTED_COUNTS.values()):
        print(f"task count mismatch: {inv['n_tasks_total']} != {sum(EXPECTED_COUNTS.values())}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
