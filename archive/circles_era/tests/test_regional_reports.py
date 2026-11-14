"""Acquisition/causality regressions, not simulated traffic efficacy tests."""
from dataclasses import replace
import json

import pytest

from traffic_shepherd.regional_reports import (
    DEFAULT_REGIONS, RegionalReportBuffer, RegionalReportProfile, ReportRegion,
    iter_regional_summaries, received_reports, regional_report_features,
)
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


REGIONS = (ReportRegion("near", ("L0", "L1"), 0., 400.), ReportRegion("far", ("L0", "L1"), 400., 1000.))


def report(vehicle="a", time=10., speed=20., distance=200., lane="L0", front=None, rear=None, coverage=100.):
    return CAVReport(SensorFrame(vehicle_id=vehicle, timestamp_s=time, speed_mps=speed,
        acceleration_mps2=0., allowed_speed_mps=30., lane_index=0, lane_id=lane,
        distance_to_merge_m=distance, front=front, rear=rear, front_coverage_m=coverage,
        rear_coverage_m=coverage), sent_at_s=time, owner="damping")


def snapshot(rows, now=10., **kwargs):
    return regional_report_features(rows, now_s=now, regions=REGIONS, **kwargs)


def test_all_reporters_contribute_without_a_proposal_or_hidden_population():
    result = snapshot([(10., report()), (10., report("b", speed=10., lane="L1")),
                       (10., report("c", distance=500.))])
    near = result["regions"]["near"]
    assert near["fresh_reporters"] == 2 and near["mean_reporter_speed_mps"] == 15.
    assert near["slow_reporter_fraction"] == .5 and near["missing_reported_lanes"] == []
    assert result["regions"]["far"]["fresh_reporters"] == 1
    assert not near["total_traffic_count_known"] and not near["coverage_is_union"]
    with pytest.raises(TypeError):
        snapshot([], true_penetration=.2)


@pytest.mark.parametrize("receipt,now", [(11., 10.), (9., 10.)])
def test_future_receipt_and_receipt_before_observation_are_rejected(receipt, now):
    with pytest.raises(ValueError, match="future or invalid"):
        snapshot([(receipt, report())], now=now)


def test_late_and_out_of_order_observations_do_not_replace_newer_state():
    newer = report(time=10., speed=20.)
    delayed = report(time=8., speed=5.)
    result = snapshot([(10., newer), (10.5, delayed)], now=10.5)
    near = result["regions"]["near"]
    assert near["mean_reporter_speed_mps"] == 20.
    assert near["history"]["unique_observations"] == 2
    assert near["receipt_age_s"]["mean"] == .5
    assert result == snapshot([(10.5, delayed), (10., newer)], now=10.5)


def test_absent_stale_unassigned_and_missing_lane_have_distinct_meanings():
    result = snapshot([(7., report(time=7.)), (10., report("b", lane="unknown"))])
    near = result["regions"]["near"]
    assert near["fresh_reporters"] == 0 and near["stale_known_reporters"] == 1
    assert near["mean_reporter_speed_mps"] is None and near["slow_reporter_fraction"] is None
    assert near["missing_reported_lanes"] == ["L0", "L1"]
    assert not near["lane_occupancy_known"] and result["packet_audit"]["fresh_reporters_unassigned"] == 1
    assert snapshot([])["regions"]["near"]["history"]["unique_observations"] == 0


def test_latest_position_removes_reporter_from_former_region():
    result = snapshot([(9., report(time=9., distance=450.)), (10., report(distance=350.))])
    assert result["regions"]["near"]["fresh_reporters"] == 1
    assert result["regions"]["far"]["fresh_reporters"] == 0
    assert result["regions"]["far"]["history"]["distinct_reporters"] == 1


def test_duplicate_packets_do_not_inflate_counts_trends_or_evidence():
    r = report()
    one, repeated = snapshot([(10., r)], now=10.5), snapshot([(10.5, r), (10., r)], now=10.5)
    assert one["regions"] == repeated["regions"]
    assert one["evidence_sha256"] == repeated["evidence_sha256"]
    assert repeated["packet_audit"]["duplicate_packets"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        snapshot([(10., r), (10., replace(r, frame=replace(r.frame, speed_mps=2.)))])


def test_anonymous_overlapping_neighbors_remain_samples_not_unique_vehicles():
    same = NeighborMeasurement(20., -2.)
    result = snapshot([(10., report(front=same)), (10., report("b", front=same))])
    front = result["regions"]["near"]["front"]
    assert front["detection_samples"] == 2 and front["minimum_closing_ttc_s"] == 10.
    assert not front["unique_detected_vehicles_known"] and not front["detections_located_in_region"]
    assert "summed_coverage_m" not in front
    assert front["reporter_coverage_m"]["maximum"] == 100.


def test_front_and_rear_closing_sign_and_uncovered_empty_are_explicit():
    r = report(front=NeighborMeasurement(20., -2.), rear=NeighborMeasurement(30., 3.))
    near = snapshot([(10., r), (10., report("b", coverage=0.)), (10., report("c"))])["regions"]["near"]
    assert near["front"]["minimum_closing_ttc_s"] == near["rear"]["minimum_closing_ttc_s"] == 10.
    assert near["front"]["zero_coverage_samples"] == 1
    assert near["front"]["no_detection_with_positive_coverage"] == 1
    no_closure = snapshot([(10., report(front=NeighborMeasurement(20., 2.)))])["regions"]["near"]["front"]
    assert no_closure["minimum_closing_ttc_s"] is None


def test_trend_uses_observation_time_and_current_reporter_weight_not_packet_rate():
    rows = [(5., report(time=0., speed=30.)), (10., report(speed=20.)),
            (8., report("b", time=5., speed=10.)), (10., report("b", speed=20.))]
    near = snapshot(rows)["regions"]["near"]
    assert near["history"]["current_reporters_with_speed_trend"] == 2
    assert near["history"]["current_reporter_speed_trend_mps2"]["mean"] == .5
    # High sampling frequency for one CAV does not reweight reporter means.
    extra = rows+[(6., report(time=6., speed=24.))]
    assert snapshot(extra)["regions"]["near"]["history"]["current_reporter_speed_trend_mps2"]["mean"] == pytest.approx(.5)


def test_fixed_60s_history_uses_observation_age_not_recent_arrival():
    rows = [(100., report(time=39., speed=1.)), (100., report("b", time=40., speed=2.)),
            (100., report("c", time=100., speed=20.))]
    result = snapshot(rows, now=100.)
    assert result["definition"]["profile"]["history_s"] == 60.
    assert result["packet_audit"]["unique_packets_outside_history"] == 1
    assert result["regions"]["near"]["history"]["distinct_reporters"] == 2
    assert result["regions"]["near"]["history"]["current_reporters_with_speed_trend"] == 0


def test_reporter_regions_are_half_open_and_overlapping_partitions_rejected():
    result = snapshot([(10., report(distance=400.))])
    assert result["regions"]["near"]["fresh_reporters"] == 0
    assert result["regions"]["far"]["fresh_reporters"] == 1
    with pytest.raises(ValueError, match="overlapping"):
        regional_report_features([], now_s=10., regions=(REGIONS[0], ReportRegion("all", ("L0",))))
    missing = snapshot([(10., report(distance=None))])
    assert missing["packet_audit"]["fresh_reporters_unassigned"] == 1


def test_streaming_receipt_cutoff_late_history_and_backwards_queries(tmp_path):
    records = [dict(kind="rsu_received_report", received_at_s=11., **report().payload()),
               dict(kind="offline_state", true_density=100),
               dict(kind="rsu_received_report", received_at_s=12., **report(time=12., speed=25.).payload())]
    path = tmp_path/"telemetry.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    rows = list(iter_regional_summaries(path, query_times_s=(10., 11., 12.), regions=REGIONS))
    assert rows[0]["regions"]["near"]["fresh_reporters"] == 0
    assert rows[1]["regions"]["near"]["mean_reporter_speed_mps"] == 20.
    assert rows[2]["regions"]["near"]["mean_reporter_speed_mps"] == 25.
    assert rows[1]["regions"]["near"]["transport_delay_s"]["mean"] == 1.
    with pytest.raises(ValueError, match="snapshot"):
        list(iter_regional_summaries(path, query_times_s=(12., 11.), regions=REGIONS))


def test_streaming_rejects_privileged_report_fields_and_out_of_order_receipts(tmp_path):
    row = dict(kind="rsu_received_report", received_at_s=10., **report().payload())
    row["global_health"] = .5
    path = tmp_path/"bad.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(TypeError):
        list(received_reports(path))
    row.pop("global_health")
    older = dict(kind="rsu_received_report", received_at_s=9., **report(time=9.).payload())
    path.write_text(json.dumps(row)+"\n"+json.dumps(older), encoding="utf-8")
    with pytest.raises(ValueError, match="nonmonotone"):
        list(received_reports(path))


def test_buffer_does_not_allow_retroactive_receipt_or_snapshot():
    buffer = RegionalReportBuffer(regions=REGIONS)
    buffer.receive(report(), received_at_s=10.)
    assert buffer.snapshot(now_s=10.)["regions"]["near"]["fresh_reporters"] == 1
    with pytest.raises(ValueError):
        buffer.receive(report(time=9.), received_at_s=9.)
    with pytest.raises(ValueError):
        buffer.snapshot(now_s=9.)


def test_profile_validation_and_explicit_three_lane_bottleneck_map():
    assert len(next(r for r in DEFAULT_REGIONS if r.name == "bottleneck").lane_ids) == 3
    with pytest.raises(ValueError):
        RegionalReportProfile(history_s=0.)
    with pytest.raises(ValueError):
        RegionalReportProfile(max_age_s=61.)
    with pytest.raises(ValueError):
        ReportRegion("bad", ("L0",), 10., 0.)
