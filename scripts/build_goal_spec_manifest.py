#!/usr/bin/env python
"""V6.2 — freeze goal_spec_manifest.json before any collection.

Contents per scene family: canonical composite GoalSpec (language +
ordered automaton subgoals), TWO same-goal paraphrases, every registered
scene-valid DISTINCT composite GoalSpec, wording resolution against BDDL
instances, invalidation rules, and the staged-support recipe. Every
object/region token is validated against the BDDL text before writing.
Atomic/active-subgoal prompts are not distinct tasks and are not here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BDDL = REPO_ROOT / "bddl" / "libero_loho_public_v1"
OUT = REPO_ROOT / "results" / "libero_loho_public_v1"

T1 = "loho_t1_drawer"
T2 = "loho_t2_basket3"
T3 = "loho_t3_tray"
T4 = "loho_t4_tray"
T5 = "loho_t5_drawer_cabinet"

SPECS = {
    T1: {
        "scene_family": "kitchen_cabinet",
        "canonical": {
            "goal_spec_id": "t1_canonical",
            "language": ("put the butter at the front and the chocolate "
                         "pudding in the top drawer of the cabinet and "
                         "close it"),
            "ordered_subgoals": [
                "pick_up butter_1",
                "place butter_1 wooden_cabinet_1_top_region",
                "pick_up chocolate_pudding_1",
                "place chocolate_pudding_1 wooden_cabinet_1_top_region",
                "close wooden_cabinet_1_top_region"],
        },
        "paraphrases": [
            "place the front butter and the chocolate pudding into the "
            "top drawer of the cabinet and shut the drawer",
            "put the butter closer to the front, along with the chocolate "
            "pudding, into the cabinet's top drawer and close the drawer"],
        "distinct_goals": [
            {"goal_spec_id": "t1_back_butter",
             "language": ("put the butter at the back and the chocolate "
                          "pudding in the top drawer of the cabinet and "
                          "close it"),
             "ordered_subgoals": [
                 "pick_up butter_2",
                 "place butter_2 wooden_cabinet_1_top_region",
                 "pick_up chocolate_pudding_1",
                 "place chocolate_pudding_1 wooden_cabinet_1_top_region",
                 "close wooden_cabinet_1_top_region"]},
            {"goal_spec_id": "t5_canonical", "reference": T5},
        ],
        "wording_resolution": {
            "the butter at the front": "butter_1",
            "the butter at the back": "butter_2"},
        "staging": {"place_objects": ["butter_1"],
                    "target": "wooden_cabinet_1_top_region"},
    },
    T2: {
        "scene_family": "basket",
        "canonical": {
            "goal_spec_id": "t2_canonical",
            "language": ("put the alphabet soup and the butter and the "
                         "tomato sauce in the basket"),
            "ordered_subgoals": [
                "pick_up alphabet_soup_1",
                "place alphabet_soup_1 basket_1_contain_region",
                "pick_up butter_1",
                "place butter_1 basket_1_contain_region",
                "pick_up tomato_sauce_1",
                "place tomato_sauce_1 basket_1_contain_region"],
        },
        "paraphrases": [
            "place the alphabet soup, the butter, and the tomato sauce "
            "into the basket",
            "put the can of alphabet soup, the stick of butter, and the "
            "tomato sauce inside the basket"],
        "distinct_goals": [
            {"goal_spec_id": "t2_milk_ketchup_oj",
             "language": ("put the milk and the ketchup and the orange "
                          "juice in the basket"),
             "ordered_subgoals": [
                 "pick_up milk_1",
                 "place milk_1 basket_1_contain_region",
                 "pick_up ketchup_1",
                 "place ketchup_1 basket_1_contain_region",
                 "pick_up orange_juice_1",
                 "place orange_juice_1 basket_1_contain_region"]},
            {"goal_spec_id": "t2_cheese_butter_milk",
             "language": ("put the cream cheese and the butter and the "
                          "milk in the basket"),
             "ordered_subgoals": [
                 "pick_up cream_cheese_1",
                 "place cream_cheese_1 basket_1_contain_region",
                 "pick_up butter_1",
                 "place butter_1 basket_1_contain_region",
                 "pick_up milk_1",
                 "place milk_1 basket_1_contain_region"]},
        ],
        "wording_resolution": {},
        "staging": {"place_objects": ["alphabet_soup_1", "butter_1"],
                    "target": "basket_1_contain_region"},
    },
    T3: {
        "scene_family": "tray_scene3",
        "canonical": {
            "goal_spec_id": "t3_canonical",
            "language": ("put the alphabet soup and the cream cheese and "
                         "the butter in the tray"),
            "ordered_subgoals": [
                "pick_up alphabet_soup_1",
                "place alphabet_soup_1 wooden_tray_1_contain_region",
                "pick_up cream_cheese_1",
                "place cream_cheese_1 wooden_tray_1_contain_region",
                "pick_up butter_1",
                "place butter_1 wooden_tray_1_contain_region"],
        },
        "paraphrases": [
            "place the alphabet soup, the cream cheese, and the butter "
            "onto the tray",
            "put the soup can, the cream cheese box, and the butter in "
            "the wooden tray"],
        "distinct_goals": [
            {"goal_spec_id": "t3_sauce_ketchup_butter",
             "language": ("put the tomato sauce and the ketchup and the "
                          "butter in the tray"),
             "ordered_subgoals": [
                 "pick_up tomato_sauce_1",
                 "place tomato_sauce_1 wooden_tray_1_contain_region",
                 "pick_up ketchup_1",
                 "place ketchup_1 wooden_tray_1_contain_region",
                 "pick_up butter_1",
                 "place butter_1 wooden_tray_1_contain_region"]},
            {"goal_spec_id": "t3_soup_sauce_ketchup",
             "language": ("put the alphabet soup and the tomato sauce and "
                          "the ketchup in the tray"),
             "ordered_subgoals": [
                 "pick_up alphabet_soup_1",
                 "place alphabet_soup_1 wooden_tray_1_contain_region",
                 "pick_up tomato_sauce_1",
                 "place tomato_sauce_1 wooden_tray_1_contain_region",
                 "pick_up ketchup_1",
                 "place ketchup_1 wooden_tray_1_contain_region"]},
        ],
        "wording_resolution": {},
        "staging": {"place_objects": ["alphabet_soup_1", "cream_cheese_1"],
                    "target": "wooden_tray_1_contain_region"},
    },
    T4: {
        "scene_family": "tray_scene4",
        "canonical": {
            "goal_spec_id": "t4_canonical",
            "language": ("put the black bowl at the left and the salad "
                         "dressing and the chocolate pudding in the tray"),
            "ordered_subgoals": [
                "pick_up akita_black_bowl_1",
                "place akita_black_bowl_1 wooden_tray_1_contain_region",
                "pick_up new_salad_dressing_1",
                "place new_salad_dressing_1 wooden_tray_1_contain_region",
                "pick_up chocolate_pudding_1",
                "place chocolate_pudding_1 wooden_tray_1_contain_region"],
        },
        "paraphrases": [
            "place the left black bowl, the salad dressing, and the "
            "chocolate pudding into the tray",
            "put the black bowl on the left side, the bottle of salad "
            "dressing, and the chocolate pudding in the wooden tray"],
        "distinct_goals": [
            {"goal_spec_id": "t4_right_bowl",
             "language": ("put the black bowl at the right and the salad "
                          "dressing and the chocolate pudding in the "
                          "tray"),
             "ordered_subgoals": [
                 "pick_up akita_black_bowl_2",
                 "place akita_black_bowl_2 wooden_tray_1_contain_region",
                 "pick_up new_salad_dressing_1",
                 "place new_salad_dressing_1 wooden_tray_1_contain_region",
                 "pick_up chocolate_pudding_1",
                 "place chocolate_pudding_1 wooden_tray_1_contain_region"]},
        ],
        "wording_resolution": {
            "the black bowl at the left": "akita_black_bowl_1",
            "the black bowl at the right": "akita_black_bowl_2"},
        "staging": {"place_objects": ["akita_black_bowl_1",
                                      "new_salad_dressing_1"],
                    "target": "wooden_tray_1_contain_region"},
    },
    T5: {
        "scene_family": "kitchen_cabinet",
        "canonical": {
            "goal_spec_id": "t5_canonical",
            "language": ("put the butter at the back and the chocolate "
                         "pudding in the top drawer of the cabinet and "
                         "close it and put the black bowl on top of the "
                         "cabinet"),
            "ordered_subgoals": [
                "pick_up butter_2",
                "place butter_2 wooden_cabinet_1_top_region",
                "pick_up chocolate_pudding_1",
                "place chocolate_pudding_1 wooden_cabinet_1_top_region",
                "close wooden_cabinet_1_top_region",
                "pick_up akita_black_bowl_1",
                "place akita_black_bowl_1 wooden_cabinet_1_top_side"],
        },
        "paraphrases": [
            "place the back butter and the chocolate pudding into the top "
            "drawer of the cabinet, close the drawer, and set the black "
            "bowl on top of the cabinet",
            "put the butter closer to the back and the pudding in the "
            "cabinet's top drawer, shut it, then place the black bowl on "
            "the cabinet top"],
        "distinct_goals": [
            {"goal_spec_id": "t1_canonical", "reference": T1},
            {"goal_spec_id": "t5_front_butter",
             "language": ("put the butter at the front and the chocolate "
                          "pudding in the top drawer of the cabinet and "
                          "close it and put the black bowl on top of the "
                          "cabinet"),
             "ordered_subgoals": [
                 "pick_up butter_1",
                 "place butter_1 wooden_cabinet_1_top_region",
                 "pick_up chocolate_pudding_1",
                 "place chocolate_pudding_1 wooden_cabinet_1_top_region",
                 "close wooden_cabinet_1_top_region",
                 "pick_up akita_black_bowl_1",
                 "place akita_black_bowl_1 wooden_cabinet_1_top_side"]},
        ],
        "wording_resolution": {
            "the butter at the back": "butter_2",
            "the butter at the front": "butter_1"},
        "staging": {"place_objects": ["butter_2", "chocolate_pudding_1"],
                    "target": "wooden_cabinet_1_top_region"},
    },
}

INVALIDATION_RULES = {
    "place": "revocable — re-evaluated every check; knocked-out placement "
             "invalidates and counts as damage if unrecovered",
    "open": "revocable",
    "close": "revocable",
    "pick_up": "event milestone — sticky after correct release",
}


def validate(task: str, spec: dict) -> None:
    text = (BDDL / f"{task}.bddl").read_text()
    goals = [spec["canonical"]] + [g for g in spec["distinct_goals"]
                                   if "ordered_subgoals" in g]
    for goal in goals:
        for sg in goal["ordered_subgoals"]:
            parts = sg.split()
            obj = parts[1]
            assert obj.rsplit("_", 1)[0] in text or obj in text, \
                f"{task}: object {obj} not in BDDL"
            if parts[0] == "place":
                region = parts[2]
                suffix = region.split("_1_", 1)[1]
                assert suffix in text, \
                    f"{task}: region suffix {suffix} not in BDDL"
    for phrase, instance in spec["wording_resolution"].items():
        assert instance in text, f"{task}: {instance} not in BDDL"


def main() -> None:
    tasks = {}
    for task, spec in SPECS.items():
        validate(task, spec)
        # resolve cross-references
        resolved_distinct = []
        for g in spec["distinct_goals"]:
            if "reference" in g:
                ref = SPECS[g["reference"]]["canonical"]
                resolved_distinct.append({
                    "goal_spec_id": ref["goal_spec_id"],
                    "language": ref["language"],
                    "ordered_subgoals": ref["ordered_subgoals"],
                    "cross_reference": g["reference"]})
            else:
                resolved_distinct.append(g)
        entry = {
            "scene_family": spec["scene_family"],
            "canonical": spec["canonical"],
            "paraphrases": spec["paraphrases"],
            "paraphrase_hashes": [
                hashlib.sha256(p.encode()).hexdigest()[:16]
                for p in spec["paraphrases"]],
            "distinct_goals": resolved_distinct,
            "wording_resolution": spec["wording_resolution"],
            "staging": {**spec["staging"], "provenance": "staged",
                        "recipe": ("privileged pose-set of listed objects "
                                   "into target region center, 50 settle "
                                   "steps, staged predicates asserted "
                                   "true, then stock π0.5 rollout under "
                                   "the canonical full prompt")},
            "language_conditioning_claim": True,
        }
        tasks[task] = entry
    manifest = {
        "schema": "goal_spec_manifest_v1",
        "invalidation_rules": INVALIDATION_RULES,
        "note": ("atomic/active-subgoal prompts are not distinct tasks; "
                 "scene families all have >=1 registered scene-valid "
                 "distinct composite GoalSpec"),
        "tasks": tasks,
    }
    payload = json.dumps(manifest, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(
        payload.encode()).hexdigest()
    out = OUT / "goal_spec_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    n_goals = sum(1 + len(t["distinct_goals"]) for t in tasks.values())
    print(f"frozen {len(tasks)} scenes / {n_goals} goal specs -> {out}")


if __name__ == "__main__":
    main()
