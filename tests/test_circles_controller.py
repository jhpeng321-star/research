"""Frozen-weight replay and observation/actuator boundary regressions; no SUMO."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.config import CONFIG_PATH, load_config
from traffic_shepherd.damping import CirclesController


def frame(**changes):
    values = dict(vehicle_id="cav", timestamp_s=0., speed_mps=15., acceleration_mps2=0.,
        allowed_speed_mps=30., lane_index=0, distance_to_merge_m=50.,
        front=SimpleNamespace(gap_m=30., relative_speed_mps=0.), rear=None,
        front_coverage_m=100., rear_coverage_m=100., previous_command_delta=0.)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture(scope="module")
def controller():
    return CirclesController()


def test_model_and_license_identity():
    cfg = load_config()
    provenance = json.loads((CONFIG_PATH.parent / "provenance.json").read_text())
    assert hashlib.sha256(cfg.model_path.read_bytes()).hexdigest() == cfg.model_sha256 == provenance["model_sha256"]
    assert hashlib.sha256((CONFIG_PATH.parent / "LICENSE").read_bytes()).hexdigest() == provenance["license_sha256"]
    assert provenance["working_adopted"] and not provenance["scientifically_accepted"]


def test_saved_trajectories_reproduce_exactly(controller):
    fixture = json.loads((CONFIG_PATH.parent / "replay_samples.json").read_text())
    for row in fixture["samples"]:
        assert controller.infer(row["raw_si"]) == row["network_output_mps2"]


def test_no_model_substitution(tmp_path):
    altered = tmp_path / "wrong.onnx"
    altered.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        CirclesController(replace(load_config(), model_path=altered))


def test_singleton_only(controller):
    with pytest.raises(ValueError, match="one finite"):
        controller.infer([[15,15,30],[20,10,20]])


def test_only_three_raw_inputs_enter_network(controller, monkeypatch):
    calls = []
    monkeypatch.setattr(controller, "infer", lambda row: calls.append(row) or -2.)
    original = frame()
    a = controller.decide(original, now_s=0.)
    b = controller.decide(frame(acceleration_mps2=4., lane_index=2,
        distance_to_merge_m=-300., previous_command_delta=100., rear=SimpleNamespace(gap_m=1., relative_speed_mps=10.)), now_s=0.)
    assert calls == [(15.,15.,30.), (15.,15.,30.)]
    assert a == b
    assert a.command_acceleration_mps2 == -2.  # no tight jerk or rear projection
    assert a.target_speed_mps == 13.  # absolute acceleration, no nominal IDM addition


@pytest.mark.parametrize("changes,reason", [
    ({"front":None}, "missing_or_uncovered_leader"),
    ({"rear_coverage_m":0.}, "missing_or_uncovered_leader"),
    ({"timestamp_s":-.5}, "noncurrent_observation"),
    ({"front_coverage_m":101.}, "coverage_exceeds_contract"),
    ({"speed_mps":float('nan')}, "invalid_observation"),
])
def test_invalid_observation_fallback(controller, changes, reason):
    assert controller.decide(frame(**changes), now_s=0.).fallback_reason == reason


def test_graph_nonfinite_boundary_is_not_commanded(controller):
    decision = controller.decide(frame(speed_mps=0., front=SimpleNamespace(gap_m=0.,relative_speed_mps=0.)), now_s=0.)
    assert np.isnan(decision.network_output_mps2)
    assert decision.command_acceleration_mps2 is None
    assert decision.fallback_reason == "invalid_model_output"


def test_tested_direct_speed_bounds(controller, monkeypatch):
    monkeypatch.setattr(controller, "infer", lambda row: 1.5)
    decision = controller.decide(frame(speed_mps=29.8), now_s=0.)
    assert decision.command_acceleration_mps2 == pytest.approx(.2)
    assert decision.target_speed_mps == 30.
    assert controller.decide(frame(speed_mps=40.), now_s=0.).fallback_reason == "infeasible_speed_bounds"
