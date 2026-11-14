"""Contracts for the registered uncapped target-speed rule and its evidence."""
import json

import pytest

from traffic_shepherd.target_speed_mechanics import EXPECTED_TARGET_MPS, plan
from traffic_shepherd.target_speed_wall import (REGISTERED_MAXIMUM_TARGET_MPS,
                                                TargetSpeedProfile, TargetSpeedWallPolicy,
                                                expected_uncapped_target,
                                                predecision_identity_keys)


def test_profile_changes_only_the_cap_and_version():
    from dataclasses import asdict
    from traffic_shepherd.gentle_wall import GentleProfile
    profile = TargetSpeedProfile()
    expected = asdict(GentleProfile())
    expected["version"] = "target_speed_wall/1"
    expected["maximum_target_speed_mps"] = REGISTERED_MAXIMUM_TARGET_MPS
    assert asdict(profile) == expected
    assert REGISTERED_MAXIMUM_TARGET_MPS == 40.0


def test_profile_rejects_any_other_change():
    with pytest.raises(ValueError):
        TargetSpeedProfile(hold_s=10.)
    with pytest.raises(ValueError):
        TargetSpeedProfile(target_speed_reduction_mps=3.)


def test_uncapped_target_law():
    assert expected_uncapped_target([27.66, 27.71], [33., 33.]) == pytest.approx(25.685)
    assert expected_uncapped_target([29.24, 29.63], [31., 31.]) == pytest.approx(27.435)
    # allowed-speed bound still applies under the raised registration cap
    assert expected_uncapped_target([29.24, 29.63], [24., 24.]) == 24.


def test_policy_requires_wall_arm_and_registered_profile():
    job = dict(arm="damping", decision_time_s=5,
               policy_spec=dict(member_ids=["a"], required_lanes=["l"],
                                profile=json.loads(json.dumps(
                                    __import__("dataclasses").asdict(TargetSpeedProfile())))))
    with pytest.raises(ValueError):
        TargetSpeedWallPolicy(job, registration=None)


def test_mechanics_plan_is_bounded_and_uncapped():
    data = plan()
    assert data["maximum_simulated_seconds"] == 120
    assert data["automatic_retries"] == 0 and data["automatic_successors"] is False
    assert data["expected_commanded_target_mps"] == EXPECTED_TARGET_MPS == 25.
    assert data["profile"]["maximum_target_speed_mps"] == REGISTERED_MAXIMUM_TARGET_MPS
    assert data["criteria"]["commanded_target_equals_uncapped_rule"] is True


def test_predecision_identity_keys_exclude_proposal():
    keys = predecision_identity_keys()
    assert "proposal_sha256" not in keys
    assert "predecision_command_prefix_sha256" in keys
