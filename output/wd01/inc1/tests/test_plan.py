"""World-plan tests (shape, loadings, penetrations, incident split)."""
from __future__ import annotations

import pytest

from inc1 import constants as C
from inc1 import plan as P


def _plan():
    return P.build_plan(seeds=C.seed_block(), include_scenario_binding=False)


def test_plan_shape_and_distributions():
    plan = _plan()
    assert plan["schema"] == C.PLAN_SCHEMA
    assert len(plan["worlds"]) == 20
    assert len(plan["runs"]) == 20 * 8
    loadings = {}
    for world in plan["worlds"]:
        loadings.setdefault(world["loading_veh_h"], []).append(world)
    assert {k: len(v) for k, v in loadings.items()} == {
        2400: 7, 2600: 7, 2700: 6}
    incident = {"A": 0, "B": 0}
    for world in plan["worlds"]:
        incident[world["incident_class"]] += 1
        assert world["incident_s_down_m"] == C.INCIDENT_S_DOWN[
            world["incident_class"]]
    assert incident == {"A": 10, "B": 10}


def test_penetration_is_round_robin_within_each_loading():
    plan = _plan()
    for loading, count in zip(C.LOADINGS, C.WORLDS_PER_LOADING):
        rows = [w for w in plan["worlds"] if w["loading_veh_h"] == loading]
        rows.sort(key=lambda w: w["seed"])
        expected = [C.PENETRATIONS[i % len(C.PENETRATIONS)]
                    for i in range(count)]
        assert [w["penetration"] for w in rows] == expected


def test_penetration_counts_per_loading_after_addendum_3():
    """A3.1: {p10, p20, p40} round-robin within each loading."""
    plan = _plan()
    for loading, expected in ((2400, {0.10: 3, 0.20: 2, 0.40: 2}),
                              (2600, {0.10: 3, 0.20: 2, 0.40: 2}),
                              (2700, {0.10: 2, 0.20: 2, 0.40: 2})):
        rows = [w for w in plan["worlds"] if w["loading_veh_h"] == loading]
        counts = {}
        for world in rows:
            counts[world["penetration"]] = counts.get(
                world["penetration"], 0) + 1
        assert counts == expected, (loading, counts)
    total = {}
    for world in plan["worlds"]:
        total[world["penetration"]] = total.get(world["penetration"], 0) + 1
    assert total == {0.10: 8, 0.20: 6, 0.40: 6}
    assert set(total) == set(C.PENETRATIONS)
    assert 0.05 not in total


def test_incident_is_distributed_evenly_per_loading():
    plan = _plan()
    for loading in C.LOADINGS:
        rows = [w for w in plan["worlds"] if w["loading_veh_h"] == loading]
        classes = [w["incident_class"] for w in rows]
        assert abs(classes.count("A") - classes.count("B")) <= 1


def test_every_arm_restores_the_warmup_state():
    plan = _plan()
    for world in plan["worlds"]:
        arms = [b for b in world["branches"] if b["arm"] != "warmup"]
        assert sorted(b["arm"] for b in arms) == sorted(C.ARM_NAMES)
        for branch in arms:
            assert branch["mode"] == "restore_incident"
            assert branch["restores_state"]["from_run"] == world["warmup_run"]
            assert branch["duration_s"] == C.ARM_HORIZON_S


def test_write_plan_refuses_overwrite(tmp_path):
    plan = _plan()
    path = P.write_plan(plan, tmp_path)
    assert path.is_file()
    with pytest.raises(P.PlanError):
        P.write_plan(plan, tmp_path)


def test_budget_is_registered_total():
    plan = _plan()
    assert plan["budget"]["total_sim_s"] == C.BUDGET_TOTAL_S == 49920.0
    assert plan["budget"]["warmup_sim_s"] == 6000.0
    assert plan["budget"]["arms_sim_s"] == 42000.0
    assert plan["budget"]["harness_equivalence_smoke_sim_s"] == 1800.0


def test_committed_plan_matches_builder():
    import json

    assert C.PLAN_PATH.is_file(), "run the plan builder first"
    committed = json.loads(C.PLAN_PATH.read_text(encoding="utf-8"))
    rebuilt = P.build_plan(seeds=[w["seed"] for w in committed["worlds"]],
                           include_scenario_binding=False)
    assert [w["world"] for w in rebuilt["worlds"]] == [
        w["world"] for w in committed["worlds"]]
    assert committed["constants"]["merge_start_s"] == C.MERGE_START_S
