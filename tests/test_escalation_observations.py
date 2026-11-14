"""Observation boundaries and spatial diagnostic tests; no physical efficacy claims."""
from dataclasses import replace
from copy import deepcopy

import pytest

from traffic_shepherd.escalation_observations import candidate_geometry, region_statistics, report_from_payload
from traffic_shepherd.reporting import CAVReport, RSUReportBuffer
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


def reports(span=50., now=10.):
    return [CAVReport(SensorFrame(vehicle_id=v,timestamp_s=now,speed_mps=20.,acceleration_mps2=0.,
                allowed_speed_mps=30.,lane_index=i,lane_id=f"L{i}",distance_to_merge_m=500.+span*i,
                front=NeighborMeasurement(25.,-1.),rear=NeighborMeasurement(30.,1.),
                front_coverage_m=100.,rear_coverage_m=100.),sent_at_s=now,owner="damping")
            for i,v in enumerate(("a","b"))]


def geometry(packets, **changes):
    args = dict(now_s=10.,required_lanes=["L0","L1"],recruitment_band_m=[200.,1900.],max_report_age_s=1.)
    return candidate_geometry(packets,**{**args,**changes})


@pytest.mark.parametrize("span", [10.,25.,49.,50.,51.,75.,100.,150.])
def test_geometry_capture_does_not_program_the_answer_to_the_50m_question(span):
    result = geometry(reports(span))
    assert result["candidates"][0]["initial_span_m"] == span
    assert result["physically_feasible_count"] is None


def test_missing_lane_report_is_unknown_not_empty_lane():
    result = geometry(reports()[:1])
    assert result["missing_reported_lanes"] == ["L1"]
    assert result["total_candidate_combinations"] == 0 and not result["lane_occupancy_known"]


def test_actual_receipt_controls_availability_and_old_packets_do_not_help():
    buffer = RSUReportBuffer(max_age_s=1.,history_s=20.)
    for report in reports():
        buffer.receive(report,received_at_s=11.)
    assert geometry(buffer.reports(now_s=10.))["total_candidate_combinations"] == 0
    assert geometry(buffer.reports(now_s=11.),now_s=11.)["total_candidate_combinations"] == 1
    assert geometry(buffer.reports(now_s=12.),now_s=12.)["total_candidate_combinations"] == 0


def test_report_schema_rejects_privileged_features_and_roundtrips():
    payload = reports()[0].payload()
    assert report_from_payload(payload) == reports()[0]
    for key in ("true_density", "true_penetration", "global_health", "driver_type"):
        altered = deepcopy(payload)
        altered["frame"][key] = 1.
        with pytest.raises(TypeError):
            report_from_payload(altered)
    with pytest.raises(TypeError):
        geometry([{"global_traffic":100}])


def test_unknown_local_space_remains_unknown():
    packets = reports()
    packets[0] = replace(packets[0], frame=replace(packets[0].frame,front=None,front_coverage_m=0.))
    candidate = geometry(packets)["candidates"][0]
    assert candidate["front_gap_m"][0] is None and candidate["front_coverage_m"][0] == 0.


def test_candidate_budget_reports_truncation_instead_of_proving_no_opportunity():
    packets = reports()
    packets += [replace(r, frame=replace(r.frame,vehicle_id=r.frame.vehicle_id+"2",distance_to_merge_m=800.)) for r in packets]
    result = geometry(packets,max_groups=1)
    assert result["total_candidate_combinations"] == 4
    assert len(result["candidates"]) == 1 and not result["enumeration_complete"]


def test_hidden_state_is_not_an_argument_and_packet_order_does_not_change_candidates():
    assert geometry(reports()) == geometry(list(reversed(reports())))
    # Full-state metadata cannot be passed through the report-only boundary.
    with pytest.raises(TypeError):
        geometry(reports(),global_density=100.)


def test_report_thinning_can_reduce_information_without_mandated_penetration_rule():
    assert geometry(reports())["total_candidate_combinations"] == 1
    assert geometry(reports()[:1])["total_candidate_combinations"] == 0
    assert "penetration" not in geometry(reports())


def test_measured_density_uses_lane_length_and_empty_road_has_no_mean_speed():
    result = region_statistics([10.,20.],summed_lane_length_m=400.)
    assert result["density_vehicles_per_km_lane"] == 5.
    assert result["mean_speed_mps"] == 15. and result["speed_std_mps"] == 5.
    assert region_statistics([],summed_lane_length_m=400.)["mean_speed_mps"] is None
