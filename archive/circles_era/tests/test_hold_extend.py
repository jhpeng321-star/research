"""Contracts for the registered HOLD extension (v1).

These tests exercise the analysis/policy implementation offline: wrong branch
ticks, duplicate or late extension requests, clock resets, profile swaps and
spec validation would each have caught a defect found in earlier blocks.
"""
from dataclasses import asdict

import pytest

from traffic_shepherd.hold_extend_wall import (EXTENSION_S, ExtensionProfile,
                                               HoldExtendPolicy,
                                               validate_extension_spec)
from traffic_shepherd.lifecycle_wall import LifecycleProfile


def _job(adjust_spec=None, ext_spec=None):
    profile = asdict(LifecycleProfile())
    if adjust_spec is None:
        adjust_spec = dict(branch_mode="absolute", adjust_mps=2.0,
                           expected_hold_start_s=612.0, branch_time_s=617.0)
    if ext_spec is None:
        ext_spec = dict(branch_mode="absolute", extension_s=10.0,
                        expected_hold_start_s=612.0, branch_time_s=632.0)
    return dict(arm="wall", decision_time_s=600, policy_spec=dict(
        member_ids=["a", "b"], required_lanes=["x", "y"], profile=profile,
        target_adjust_branch=adjust_spec, hold_extension=ext_spec))


def test_extension_profile_differs_only_in_hold_and_cap():
    profile = ExtensionProfile()
    data = asdict(profile)
    base = asdict(LifecycleProfile())
    assert data["hold_s"] == 30.0 and base["hold_s"] == 20.0
    assert data["maximum_target_speed_mps"] == 40.0
    for key in data:
        if key not in ("version", "hold_s", "maximum_target_speed_mps"):
            assert data[key] == base[key], key


def test_extension_spec_validation():
    checked = validate_extension_spec(dict(
        branch_mode="absolute", extension_s=10.0,
        expected_hold_start_s=612.0, branch_time_s=632.0),
        adjust_expected_hold_start_s=612.0)
    assert checked["branch_time_s"] == 632.0
    with pytest.raises(ValueError):
        validate_extension_spec(dict(branch_mode="absolute", extension_s=20.0,
                                     expected_hold_start_s=612.0, branch_time_s=632.0))
    with pytest.raises(ValueError):
        validate_extension_spec(dict(branch_mode="absolute", extension_s=10.0,
                                     expected_hold_start_s=612.0, branch_time_s=635.0),
                                adjust_expected_hold_start_s=612.0)
    with pytest.raises(ValueError):
        validate_extension_spec(dict(branch_mode="absolute", extension_s=10.0,
                                     expected_hold_start_s=618.0, branch_time_s=638.0),
                                adjust_expected_hold_start_s=612.0)
    rel = validate_extension_spec(dict(branch_mode="hold_relative",
                                       extension_s=10.0,
                                       branch_after_hold_start_s=20.0))
    assert rel["extension_s"] == 10.0
    with pytest.raises(ValueError):
        validate_extension_spec(dict(branch_mode="hold_relative", extension_s=10.0,
                                     branch_after_hold_start_s=25.0))


def test_policy_rejects_missing_or_invalid_extension_spec():
    with pytest.raises(KeyError):
        HoldExtendPolicy(dict(arm="wall", decision_time_s=600, policy_spec=dict(
            member_ids=["a"], required_lanes=["l"],
            profile=asdict(LifecycleProfile()))), registration=None)
    with pytest.raises(ValueError):
        HoldExtendPolicy(_job(ext_spec=dict(branch_mode="absolute", extension_s=5.0,
                                            expected_hold_start_s=612.0,
                                            branch_time_s=632.0)), registration=None)


def test_extension_constants_are_the_registered_increment():
    assert EXTENSION_S == 10.0
    assert ExtensionProfile().hold_s == 30.0
