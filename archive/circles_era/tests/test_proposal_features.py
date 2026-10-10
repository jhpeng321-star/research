"""Observation boundary and selection tests, not predictor evidence."""
from dataclasses import replace
import pytest
from traffic_shepherd.proposal_features import feature_packet, latest_received, select_proposal
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import SensorFrame


def history(now=10., speeds=(20., 20.), distances=(1800., 1700.)):
    return [(now, CAVReport(SensorFrame(vehicle_id=str(i), timestamp_s=now,
        speed_mps=speeds[i], acceleration_mps2=0., allowed_speed_mps=30., lane_index=i,
        lane_id=f"L{i}", position_m=2000.-distances[i], distance_to_merge_m=distances[i],
        front_coverage_m=100., rear_coverage_m=100.), now, "damping")) for i in range(2)]


@pytest.mark.parametrize("speed", [0., 5., 20., 30.])
def test_selection_any_speed_and_above_50m(speed):
    h = history(speeds=(speed, speed))
    proposal, assessment, accounting = select_proposal(h, now_s=10., required_lanes=("L0", "L1"))
    assert proposal and assessment.approved and assessment.initial_or_current_span_m == 100.
    assert not accounting["traffic_state_filter"]


def test_unreported_lane_is_unknown_not_empty():
    p, a, accounting = select_proposal(history()[:1], now_s=10., required_lanes=("L0", "L1"))
    assert p is a is None
    assert accounting["missing_reported_lanes"] == ["L1"] and not accounting["lane_occupancy_known"]


def test_future_and_conflicting_receipts_fail_closed():
    h = history()
    with pytest.raises(ValueError, match="future"):
        latest_received([(11., h[0][1])], now_s=10.)
    with pytest.raises(ValueError, match="conflicting"):
        latest_received(h + [(10., replace(h[0][1], owner="native"))], now_s=10.)


def test_exact_evidence_binding_and_no_experiment_metadata_in_features():
    h = history()
    proposal, _, _ = select_proposal(h, now_s=10., required_lanes=("L0", "L1"))
    row = feature_packet(proposal, h, now_s=10.)
    assert row["values"]["member_0_front_present"] == 0.
    assert row["values"]["member_0_front_coverage_m"] == 100.
    assert all(isinstance(v, (int, float)) for v in row["values"].values())
    assert not set(row["values"]) & {"seed", "penetration", "density", "health", "time", "vehicle_id", "loading", "benefit"}
    changed = [(t, replace(r, issued_acceleration_mps2=.1)) for t, r in h]
    assert feature_packet(proposal, changed, now_s=10.)["row_sha256"] != row["row_sha256"]


def test_stale_member_missing_flags_and_delay_age():
    h = history()
    proposal, _, _ = select_proposal(h, now_s=10., required_lanes=("L0", "L1"))
    row = feature_packet(proposal, h, now_s=12.)
    assert row["values"]["member_0_report_present"] == 0.
    delayed = [(10.5, r) for _, r in h]
    row = feature_packet(proposal, delayed, now_s=10.5)
    assert row["values"]["member_0_observation_age_s"] == .5
    assert row["values"]["member_0_receipt_age_s"] == 0.


def test_limited_search_reports_truncation_without_inventing_absence():
    h = history()
    h += [(t, replace(r, frame=replace(r.frame, vehicle_id=r.frame.vehicle_id+"copy"))) for t, r in h]
    _, _, accounting = select_proposal(h, now_s=10., required_lanes=("L0", "L1"), maximum_members_per_lane=1)
    assert not accounting["enumeration_complete"] and accounting["evaluated_combinations"] == 1
