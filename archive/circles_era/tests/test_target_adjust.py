"""Contracts for the repaired continuation target-adjustment interface (v2)."""
import json
from dataclasses import asdict

import pytest

from traffic_shepherd.continuation_dataset import build_dataset
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.target_adjust_mechanics import ADJUST_MPS, plan
from traffic_shepherd.target_adjust_wall import (REGISTERED_ADJUSTMENTS_MPS,
                                                 VERSION, ContinuationTargetProfile,
                                                 continuation_target, policy_factory)


def test_entry_profile_contract_is_unchanged():
    assert asdict(LifecycleProfile())["maximum_target_speed_mps"] == 24.0
    data = plan()
    assert data["profile"] == asdict(LifecycleProfile())
    assert data["profile"]["maximum_target_speed_mps"] == 24.0


def test_continuation_target_contract():
    assert VERSION == "hold_target_adjust/2"
    assert REGISTERED_ADJUSTMENTS_MPS == (-2.0, 2.0)
    assert asdict(ContinuationTargetProfile())["maximum_target_speed_mps"] == 40.0
    allowed = [33.0, 33.0]
    assert continuation_target(24.0, -2.0, allowed)[0] == 22.0
    assert continuation_target(24.0, 2.0, allowed)[0] == 26.0
    assert isinstance(continuation_target(24.0, 2.0, allowed)[1], ContinuationTargetProfile)


def test_out_of_contract_targets_rejected_before_allocation():
    with pytest.raises(ValueError):
        continuation_target(24.0, 3.0, [33.0, 33.0])  # not a registered adjustment
    with pytest.raises(ValueError):
        continuation_target(24.0, 2.0, [25.0, 33.0])  # above a member allowed speed
    entry = LifecycleProfile()
    with pytest.raises(ValueError):
        __import__("traffic_shepherd.formation", fromlist=["FormationProposal"])
        # the entry profile itself still rejects a 26 m/s proposal target
        from traffic_shepherd.formation import FormationProposal, WallMember
        FormationProposal(600.0, ("l0", "l1"),
                          (WallMember("a", "l0"), WallMember("b", "l1")),
                          26.0, entry)


def test_policy_rejects_unregistered_adjustments():
    profile = asdict(LifecycleProfile())
    base = dict(arm="wall", decision_time_s=600, policy_spec=dict(
        member_ids=["a", "b"], required_lanes=["x", "y"], profile=profile,
        target_adjust_branch=dict(branch_mode="absolute", adjust_mps=3.0,
                                  expected_hold_start_s=612., branch_time_s=617.)))
    with pytest.raises(ValueError):
        policy_factory(json.loads(json.dumps(base)), registration=None)


def test_mechanics_plan_asks_the_up_question():
    data = plan()
    assert data["cases"][0]["adjust_mps"] == ADJUST_MPS == 2.0
    assert data["maximum_simulated_seconds"] == 120
    assert data["automatic_retries"] == 0 and data["automatic_successors"] is False


def test_continuation_dataset_rebuilds_with_sign_checks():
    dataset = build_dataset()
    assert dataset["schema"] == "continuation_decision_dataset/2"
    assert dataset["independent_seed_groups"] == 2 and dataset["qualifying_actions"] == 0
    by_action = {(r["seed_group"], r["action"]["kind"]): r for r in dataset["rows"]
                 if r.get("status") == "delivered"}
    down4107 = by_action[(4107, "lower_22")]
    cd = down4107["cost_difference_treatment_minus_reference"]
    # authoritative signs: service, braking and wait improve; effort and ttc worsen
    assert cd["system_seconds_per_request"] == pytest.approx(-1.562544, abs=1e-5)
    assert cd["braking_seconds_per_request"] == pytest.approx(-0.053004, abs=1e-5)
    assert cd["effort_per_request"] == pytest.approx(0.949963, abs=1e-5)
    assert cd["ttc_seconds_per_request"] == pytest.approx(0.053357, abs=1e-5)
    assert cd["source_wait_seconds_per_request"] == pytest.approx(-0.107774, abs=1e-5)
    assert down4107["collisions_treatment"] == 0
    assert down4107["collisions_reference"] == 4
    assert down4107["original_label"] == "harmful"
    assert set(down4107["guardrail_failures"]) == {"effort_per_request", "ttc_seconds_per_request"}
    down4108 = by_action[(4108, "lower_22")]
    cd8 = down4108["cost_difference_treatment_minus_reference"]
    assert cd8["system_seconds_per_request"] == pytest.approx(9.456389, abs=1e-5)
    assert cd8["effort_per_request"] == pytest.approx(7.749620, abs=1e-5)
    assert cd8["ttc_seconds_per_request"] == pytest.approx(0.071155, abs=1e-5)
    assert cd8["source_wait_seconds_per_request"] == pytest.approx(-0.000383, abs=1e-5)
    assert down4108["original_label"] == "harmful"
    keep4107 = by_action[(4107, "keep_24")]
    assert keep4107["collisions_treatment"] == 4  # the keep-24 source has four, not zero
    for r in by_action.values():
        diff = r["cost_difference_treatment_minus_reference"]
        imp = r["improvement_negative_of_cost_difference"]
        for m in diff:
            assert imp[m] == pytest.approx(-diff[m], abs=1e-12)
    fail = [r for r in dataset["rows"] if r.get("status") == "implementation_failure"]
    assert len(fail) == 1 and fail[0]["seed_group"] == 4107 and fail[0]["action"]["kind"] == "raise_26"
    assert fail[0]["physical_outcome_label"] is None
    # transition metrics are post-decision only
    for r in by_action.values():
        if r["action"] == "lower_22":
            assert "post_decision_measurements" in r
            assert "transition" not in json.dumps(r.get("action"))


def test_continuation_target_accepts_mapping_and_rejects_low_allowed():
    # regression: dict input previously iterated vehicle-id keys (strings)
    adjusted, _ = continuation_target(24.0, 2.0, {"a": 33.0, "b": 31.0})
    assert adjusted == 26.0
    with pytest.raises(ValueError):
        continuation_target(24.0, 2.0, {"a": 25.0, "b": 33.0})
