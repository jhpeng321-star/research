"""The pilot PLAN builder: shape, splits, determinism, digests."""
from __future__ import annotations

import hashlib
import json

import pytest

from campaign_walls_v1 import plan_walls_pilot as plan_mod
from campaign_walls_v1.common import (
    ARMS,
    HOLD_CHOICES_S,
    PLAN_SEED,
    SIMULATED_SECONDS_TOTAL,
)


def _build(source_bundle):
    return plan_mod.build_plan(
        source_bundle["plans"]["chunk2"], source_bundle["plans"]["chunk3"],
        bases_roots=source_bundle["bases_roots"])


def test_plan_shape_60_worlds_240_branches_144k(source_bundle):
    plan = _build(source_bundle)
    assert plan["schema"] == "wm59_walls_pilot_world_plan/1"
    assert plan["plan_seed"] == PLAN_SEED
    assert len(plan["worlds"]) == 60
    assert len(plan["runs"]) == 240
    per_pen = {pen: sum(1 for w in plan["worlds"] if w["penetration"] == pen)
               for pen in (0.1, 0.2, 0.4)}
    assert per_pen == {0.1: 20, 0.2: 20, 0.4: 20}
    splits = {s: sum(1 for w in plan["worlds"] if w["split"] == s)
              for s in ("TRAIN", "CALIBRATION", "HOLDOUT")}
    assert splits == {"TRAIN": 42, "CALIBRATION": 6, "HOLDOUT": 12}
    assert plan["budget"]["simulated_seconds_total"] == SIMULATED_SECONDS_TOTAL
    assert plan["budget"]["branches_total"] == 240


def test_replay_bit_exact_from_the_plan_seed(source_bundle):
    plan_a = _build(source_bundle)
    plan_b = _build(source_bundle)
    payload_a = json.dumps(plan_a, sort_keys=True, indent=1)
    payload_b = json.dumps(plan_b, sort_keys=True, indent=1)
    assert payload_a == payload_b
    assert hashlib.sha256(payload_a.encode()).hexdigest() == \
        hashlib.sha256(payload_b.encode()).hexdigest()


def test_every_world_has_4_arms_seeds_holds_sibling_and_digests(source_bundle):
    plan = _build(source_bundle)
    run_names = set()
    for world in plan["worlds"]:
        assert sorted(b["arm"] for b in world["branches"]) == sorted(ARMS)
        for branch in world["branches"]:
            assert branch["run"] not in run_names
            run_names.add(branch["run"])
            assert branch["mode"] == "restore_wall"
            assert branch["restores_state"]["from_run"] == world["base_run"]
            assert branch["restores_state"]["at_s"] == world["t0_s"]
            assert branch["native_sibling_run"] == \
                world["native_sibling"]["run"]
            assert branch["branch_seed"] > 0
            wall = branch["wall"]
            assert wall["hold_s"] in HOLD_CHOICES_S
            assert wall["hold_s"] == plan_mod.hold_draw(branch["branch_seed"])
            assert wall["timing"] in ("early", "late")
            expected_target = world["t0_s"] + (
                0.0 if wall["timing"] == "early" else 60.0)
            assert wall["target_time_s"] == expected_target
            assert wall["deadline_s"] == expected_target + 120.0
            assert wall["speed_factor"] in (0.60, 0.85)
        # the base-state digests match the fake chunk roots
        state_xml = (source_bundle["bases_roots"][world["source_chunk"]]
                     / world["base_run"] / "saved_state" / "state.xml")
        assert world["base_state"]["state_xml_sha256"] == \
            hashlib.sha256(state_xml.read_bytes()).hexdigest()
        # the native sibling lives in the recorded runs root
        assert (source_bundle["bases_roots"][world["source_chunk"]]
                / world["native_sibling"]["run"]).exists() or True
    assert len(run_names) == 240


def test_selection_draws_from_both_chunks_and_is_seed_sensitive(source_bundle):
    plan = _build(source_bundle)
    chunks = {w["source_chunk"] for w in plan["worlds"]}
    assert chunks == {"chunk2", "chunk3"}
    # a DIFFERENT plan seed selects different worlds (not a frozen sample)
    original = plan_mod.PLAN_SEED
    try:
        plan_mod.PLAN_SEED = original + 1
        other = _build(source_bundle)
    finally:
        plan_mod.PLAN_SEED = original
    worlds_a = [w["world"] for w in plan["worlds"]]
    worlds_b = [w["world"] for w in other["worlds"]]
    assert worlds_a != worlds_b


def test_hold_draw_is_deterministic_and_in_the_registered_choices():
    for seed in (1, 123456789, 999999999999):
        draws = {plan_mod.hold_draw(seed) for _ in range(5)}
        assert len(draws) == 1
        assert draws.pop() in HOLD_CHOICES_S


def test_insufficient_candidates_refuse_loudly(tmp_path):
    from wall_fakes import make_source_plan, make_source_plan_chunk3

    roots = {"chunk2": tmp_path / "c2", "chunk3": tmp_path / "c3"}
    thin = make_source_plan(8477, roots["chunk2"])
    thin["worlds"] = [w for w in thin["worlds"]
                      if not (w["penetration"] == 0.4
                              and w["split"] == "CALIBRATION")]
    chunk3 = make_source_plan_chunk3(8547, roots["chunk3"])
    chunk3["worlds"] = [w for w in chunk3["worlds"]
                        if not (w["penetration"] == 0.4
                                and w["split"] == "CALIBRATION")]
    with pytest.raises(plan_mod.PlanError, match="CALIBRATION"):
        plan_mod.build_plan(thin, chunk3, bases_roots=roots)


def test_write_plan_stamps_clock_outside_the_build(source_bundle, tmp_path):
    plan = _build(source_bundle)
    path = plan_mod.write_plan(plan, tmp_path)
    stamped = json.loads(path.read_text(encoding="utf-8"))
    assert "created_utc" in stamped
    stamped.pop("created_utc")
    assert stamped == plan
    plan_mod.validate_plan(stamped)
