"""Preserved request-ledger accounting plus physical TTC/effort regression tests."""
from types import SimpleNamespace

import pytest

from traffic_shepherd.metrics import RequestLedger, braking_effort, closing_ttc, service_summary, ttc_exposure
from traffic_shepherd.scenarios import TripRequest


def test_ledger_counts_blocked_source_and_late_arrival():
    requests = [TripRequest("a", .25, "r", "normal_car", True),
                TripRequest("b", .75, "r", "normal_car", False)]
    ledger = RequestLedger(requests)
    assert ledger.advance(.5, ["a"], []) == .25
    assert ledger.advance(1, [], []) == .75
    assert ledger.advance(1.5, [], ["a"]) == 1
    assert ledger.advance(2, ["b"], []) == .5
    assert ledger.system_vehicle_seconds == 2.5 and ledger.source_vehicle_seconds == 1.5
    summary = service_summary(ledger, 2)
    assert summary["unfinished_requested"] == 1 and summary["system_seconds_per_requested_vehicle"] == 1.25
    with pytest.raises(ValueError, match="duplicate"):
        ledger.advance(2.5, ["b"], [])


@pytest.mark.parametrize("departed,arrived", [(["unknown"], []), (["a","a"], []), ([], ["a"])])
def test_ledger_rejects_invalid_lifecycle(departed, arrived):
    ledger = RequestLedger([TripRequest("a", 0, "r", "normal_car", True)])
    with pytest.raises(ValueError):
        ledger.advance(.5, departed, arrived)


def test_late_uninserted_request_is_in_service_cost():
    ledger = RequestLedger([TripRequest("a", .25, "r", "normal_car", False)])
    ledger.advance(1, [], [])
    assert ledger.system_vehicle_seconds == ledger.source_vehicle_seconds == .75
    assert service_summary(ledger, 1)["uninserted_due"] == 1


def test_severe_braking_boundary_effort_and_ttc_gap_convention():
    assert braking_effort([-3, -4, 2], .5) == dict(braking_vehicle_seconds=.5, acceleration_energy=14.5)
    assert closing_ttc(20, 20, 10) == 2
    assert closing_ttc(20, 10, 20) is None
    # SUMO leader gap excludes ego minGap: restoring it changes 2.9 to 3.15 s.
    api = SimpleNamespace(vehicle=SimpleNamespace(
        getLeader=lambda v, _: ("leader",29) if v == "ego" else None, getMinGap=lambda _: 2.5,
        getSpeed=lambda v: 20 if v == "ego" else 10))
    exposure, minimum = ttc_exposure(api, ["ego", "leader"], .5)
    assert exposure == 0 and minimum == 3.15
    assert closing_ttc(0, 10.0000001, 10) is None


@pytest.mark.parametrize("accels,dt", [([float("nan")], .5), ([1], 0)])
def test_invalid_effort_data_rejected(accels, dt):
    with pytest.raises(ValueError):
        braking_effort(accels, dt)
