"""Messages are the complete RSU boundary: delay, stale data and missingness."""
from dataclasses import replace

import pytest

from traffic_shepherd.reporting import CAVReport, RSUReportBuffer
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


def frame(cav="cav", timestamp=1):
    return SensorFrame(cav, timestamp, 20, 0, 30, 0,
                       front=NeighborMeasurement(30, -5), front_coverage_m=100,
                       rear_coverage_m=80, lane_id="lane0", position_m=200)


def test_report_content_has_only_self_perception_and_command_telemetry():
    report = CAVReport(frame(), 1, "damping", -.5, -.5)
    payload = report.payload()
    assert payload["frame"]["front"] == dict(gap_m=30, relative_speed_mps=-5)
    assert payload["frame"]["front_coverage_m"] == 100
    assert not {"global_health", "native_outcome", "demand", "penetration", "driver_type"} & payload.keys()


def test_future_receipts_stale_frames_and_missing_reports_are_explicit():
    buffer = RSUReportBuffer(max_age_s=2, history_s=60)
    buffer.receive(CAVReport(frame(), 1, "native"), received_at_s=3)
    assert not buffer.features(now_s=2)["observation_available"]
    assert buffer.features(now_s=3)["maximum_observation_age_s"] == 2
    empty = buffer.features(now_s=4)
    assert empty["reporting_cavs"] == 0 and empty["mean_cav_speed_mps"] is None
    assert empty["total_traffic_count_known"] is False


def test_rsu_combines_only_received_cavs_without_claiming_union_or_hdv_census():
    buffer = RSUReportBuffer(max_age_s=2, history_s=60)
    buffer.receive(CAVReport(frame("a"), 1, "native"), received_at_s=1)
    buffer.receive(CAVReport(replace(frame("b"), speed_mps=10), 1, "wall"), received_at_s=1)
    features = buffer.features(now_s=1)
    assert features["reporting_cavs"] == 2 and features["mean_cav_speed_mps"] == 15
    assert features["cav_speed_variance"] == 25 and features["front_detections"] == 2
    assert features["coverage_is_union"] is False
    assert features["reporter_front_coverage_m"] == [100,100]


def test_delayed_old_packet_does_not_replace_newer_observation():
    buffer = RSUReportBuffer(max_age_s=3, history_s=60)
    buffer.receive(CAVReport(frame(timestamp=2), 2, "native"), received_at_s=2)
    buffer.receive(CAVReport(frame(timestamp=1), 1, "native"), received_at_s=3)
    assert buffer.reports(now_s=3)[0].frame.timestamp_s == 2


def test_duplicate_is_idempotent_conflicting_packet_rejected():
    buffer = RSUReportBuffer(max_age_s=2, history_s=60)
    report = CAVReport(frame(), 1, "native")
    buffer.receive(report, received_at_s=1)
    buffer.receive(report, received_at_s=2)
    assert len(buffer.reports(now_s=2)) == 1
    with pytest.raises(ValueError, match="conflicting"):
        buffer.receive(replace(report, owner="wall"), received_at_s=2)


def test_impossible_timestamps_and_invalid_command_rejected():
    with pytest.raises(ValueError):
        CAVReport(frame(), 0, "native")
    with pytest.raises(ValueError):
        CAVReport(frame(), 1, "damping", float("nan"))
    with pytest.raises(ValueError):
        RSUReportBuffer(max_age_s=2, history_s=60).receive(CAVReport(frame(), 1, "native"), received_at_s=0)
