"""Regional descriptions of acquired CAV reports; no traffic census or model.

This module has no simulator dependency. Region membership uses the reporter's
observed position, never an extrapolation. Anonymous front/rear detections stay
attached to their reporter's region: overlapping detections cannot be counted
as unique vehicles or combined into detector-union coverage. The 60 s regional
history is separate from the unchanged 20 s proposal callback history.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean, pvariance

from .escalation_observations import report_from_payload
from .reporting import CAVReport


SCHEMA = "acquired_regional_reports/1"


@dataclass(frozen=True)
class ReportRegion:
    """Static map lanes, optionally restricted to [minimum, maximum) distance."""

    name: str
    lane_ids: tuple[str, ...]
    minimum_distance_m: float | None = None
    maximum_distance_m: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "lane_ids", tuple(self.lane_ids))
        if not self.name or not self.lane_ids or len(set(self.lane_ids)) != len(self.lane_ids):
            raise ValueError("named region with distinct static map lanes required")
        if any(not isinstance(lane, str) or not lane for lane in self.lane_ids):
            raise ValueError("map lane names must be nonempty strings")
        lo, hi = self.minimum_distance_m, self.maximum_distance_m
        if any(x is not None and not isfinite(x) for x in (lo, hi)) or (
                lo is not None and hi is not None and lo >= hi):
            raise ValueError("finite increasing optional distance bounds required")

    def contains(self, frame):
        if frame.lane_id not in self.lane_ids:
            return False
        lo, hi = self.minimum_distance_m, self.maximum_distance_m
        if lo is None and hi is None:
            return True
        d = frame.distance_to_merge_m
        return d is not None and (lo is None or d >= lo) and (hi is None or d < hi)


@dataclass(frozen=True)
class RegionalReportProfile:
    history_s: float = 60.
    max_age_s: float = 1.
    slow_speed_mps: float = 15.
    minimum_trend_span_s: float = 5.

    def __post_init__(self):
        if any(not isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("finite nonnegative regional observation parameters required")
        if self.history_s <= 0 or self.max_age_s > self.history_s or self.minimum_trend_span_s > self.history_s:
            raise ValueError("age and trend support must fit the positive history window")


# Static lane inventory checked against the unchanged shared long-corridor map.
# The bottleneck has THREE lanes; observed reporters do not define lane count.
DEFAULT_REGIONS = (
    ReportRegion("approach_near", ("1191554636_0", "1191554636_1"), 0., 1322.835),
    ReportRegion("approach_far", ("1191554636_0", "1191554636_1"), 1322.835, None),
    ReportRegion("bottleneck", ("1487606179_0", "1487606179_1", "1487606179_2")),
    ReportRegion("downstream", ("1487606178_0", "1487606178_1")),
    ReportRegion("ramp", ("11553048#1_0",)),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=32768)
def _report_digest(report):
    # Immutable reports recur in successive windows; hashing each payload once
    # preserves evidence binding without repeatedly materializing large trees.
    return _digest(report.payload())


def _regions(regions):
    regions = tuple(regions)
    if not regions or any(not isinstance(r, ReportRegion) for r in regions):
        raise TypeError("explicit ReportRegion definitions required")
    if len({r.name for r in regions}) != len(regions):
        raise ValueError("region names must be distinct")
    for i, first in enumerate(regions):
        for second in regions[i+1:]:
            if not set(first.lane_ids) & set(second.lane_ids):
                continue
            lo = max(x for x in (first.minimum_distance_m, second.minimum_distance_m, -float("inf")) if x is not None)
            hi = min(x for x in (first.maximum_distance_m, second.maximum_distance_m, float("inf")) if x is not None)
            if lo < hi:
                raise ValueError("overlapping reporter regions would double count self reports")
    return regions


def _packets(history, now_s, profile):
    if not isfinite(now_s) or now_s < 0:
        raise ValueError("finite nonnegative snapshot time required")
    unique, duplicates = {}, 0
    for receipt, report in history:
        if not isinstance(report, CAVReport):
            raise TypeError("regional input must contain only acquired CAVReport packets")
        if not isfinite(receipt) or not 0 <= report.frame.timestamp_s <= report.sent_at_s <= receipt <= now_s:
            raise ValueError("future or invalid observation/send/receipt time")
        key = (report.frame.vehicle_id, report.frame.timestamp_s)
        old = unique.get(key)
        if old is not None:
            if old[1] != report:
                raise ValueError("conflicting packet for the same reporter observation")
            duplicates += 1
            receipt = min(receipt, old[0])
        unique[key] = (receipt, report)
    retained = sorted((row for row in unique.values() if now_s-row[1].frame.timestamp_s <= profile.history_s),
                      key=lambda row: (row[1].frame.vehicle_id, row[1].frame.timestamp_s, row[0]))
    return retained, dict(duplicate_packets=duplicates, unique_packets_outside_history=len(unique)-len(retained))


def _moments(values):
    return dict(mean=mean(values) if values else None, minimum=min(values) if values else None,
                maximum=max(values) if values else None)


def _neighbors(rows, side):
    detections = [getattr(r.frame, side) for _, r in rows if getattr(r.frame, side) is not None]
    sign = -1. if side == "front" else 1.
    closures = [max(0., sign*n.relative_speed_mps) for n in detections]
    ttcs = [n.gap_m/c for n, c in zip(detections, closures) if c > 0]
    coverage = [getattr(r.frame, side+"_coverage_m") for _, r in rows]
    return dict(detection_samples=len(detections), no_detection_samples=len(rows)-len(detections),
        zero_coverage_samples=sum(c == 0 for c in coverage),
        no_detection_with_positive_coverage=sum(getattr(r.frame, side) is None and
            getattr(r.frame, side+"_coverage_m") > 0 for _, r in rows),
        gap_m=_moments([n.gap_m for n in detections]), closing_speed_mps=_moments(closures),
        closing_detection_samples=sum(c > 0 for c in closures),
        minimum_closing_ttc_s=min(ttcs) if ttcs else None,
        reporter_coverage_m=_moments(coverage),
        unique_detected_vehicles_known=False, detections_located_in_region=False,
        interpretation="anonymous_same_lane_samples_attached_to_reporter_region")


def _slope(rows, minimum_span):
    times = [r.frame.timestamp_s for _, r in rows]
    if len(times) < 2 or max(times)-min(times) < minimum_span:
        return None
    centre, speed = mean(times), mean(r.frame.speed_mps for _, r in rows)
    denominator = sum((t-centre)**2 for t in times)
    if denominator == 0:
        return None
    return sum((r.frame.timestamp_s-centre)*(r.frame.speed_mps-speed) for _, r in rows)/denominator


def regional_report_features(history, *, now_s, regions=DEFAULT_REGIONS, profile=None):
    """Reconstruct a causal snapshot from already received packets only.

    Future receipts are rejected, not silently admitted. Current aggregates use
    one newest observed packet per reporter. History speed means weight each
    reporting CAV equally; trends use only currently fresh reporters with enough
    within-region observation-time support. They are not traffic-flow trends.
    """
    p = profile or RegionalReportProfile()
    if not isinstance(p, RegionalReportProfile):
        raise TypeError("RegionalReportProfile required")
    regions = _regions(regions)
    packets, audit = _packets(history, now_s, p)
    latest = {}
    for receipt, report in packets:
        latest[report.frame.vehicle_id] = (receipt, report)
    fresh = {v: row for v, row in latest.items() if now_s-row[1].frame.timestamp_s <= p.max_age_s}
    result = {}
    for region in regions:
        current = [row for row in fresh.values() if region.contains(row[1].frame)]
        known = [row for row in latest.values() if region.contains(row[1].frame)]
        past, by_reporter = [], {}
        for row in packets:
            if region.contains(row[1].frame):
                past.append(row)
                by_reporter.setdefault(row[1].frame.vehicle_id, []).append(row)
        speeds = [r.frame.speed_mps for _, r in current]
        slopes = [_slope(by_reporter[r.frame.vehicle_id], p.minimum_trend_span_s) for _, r in current]
        slopes = [v for v in slopes if v is not None]
        lanes = {r.frame.lane_id for _, r in current}
        reporter_means = [mean(r.frame.speed_mps for _, r in rows) for rows in by_reporter.values()]
        result[region.name] = dict(
            available=bool(current), fresh_reporters=len(current), stale_known_reporters=len(known)-len(current),
            observed_reporter_ids=[r.frame.vehicle_id for _, r in current],
            missing_reported_lanes=[lane for lane in region.lane_ids if lane not in lanes],
            mean_reporter_speed_mps=mean(speeds) if speeds else None,
            reporter_speed_std_mps=pvariance(speeds)**.5 if speeds else None,
            slow_reporter_fraction=sum(v < p.slow_speed_mps for v in speeds)/len(speeds) if speeds else None,
            reporter_acceleration_mps2=_moments([r.frame.acceleration_mps2 for _, r in current]),
            observation_age_s=_moments([now_s-r.frame.timestamp_s for _, r in current]),
            receipt_age_s=_moments([now_s-t for t, _ in current]),
            transport_delay_s=_moments([t-r.sent_at_s for t, r in current]),
            stale_observation_age_s=_moments([now_s-r.frame.timestamp_s for _, r in known
                                             if now_s-r.frame.timestamp_s > p.max_age_s]),
            front=_neighbors(current, "front"), rear=_neighbors(current, "rear"),
            history=dict(unique_observations=len(past), distinct_reporters=len(by_reporter),
                equally_weighted_reporter_mean_speed_mps=mean(reporter_means) if reporter_means else None,
                current_reporters_with_speed_trend=len(slopes),
                current_reporters_without_speed_trend=len(current)-len(slopes),
                current_reporter_speed_trend_mps2=_moments(slopes),
                observed_span_s=max((r.frame.timestamp_s for _, r in past), default=now_s)-
                    min((r.frame.timestamp_s for _, r in past), default=now_s),
                front=_neighbors(past, "front"), rear=_neighbors(past, "rear")),
            coverage_is_union=False, total_traffic_count_known=False, lane_occupancy_known=False,
            missing_report_interpretation="absence_of_fresh_report_not_empty_road_or_packet_loss_rate")
    assigned = {r.frame.vehicle_id for _, r in fresh.values() if any(region.contains(r.frame) for region in regions)}
    evidence = [dict(received_at_s=t, report_sha256=_report_digest(r)) for t, r in packets]
    definition = dict(schema=SCHEMA, regions=[asdict(r) for r in regions], profile=asdict(p))
    return dict(schema=SCHEMA, timestamp_s=now_s, definition=definition, definition_sha256=_digest(definition),
        evidence_sha256=_digest(evidence), regions=result,
        packet_audit=dict(audit, retained_unique_observations=len(packets), known_reporters=len(latest),
                          fresh_reporters=len(fresh), fresh_reporters_unassigned=len(fresh)-len(assigned)),
        claim="acquired_CAV_report_description_not_full_traffic_state_or_fitted_gate")


class RegionalReportBuffer:
    """Forward-only bounded stream; observation time governs history expiry."""

    def __init__(self, *, regions=DEFAULT_REGIONS, profile=None):
        self.regions = _regions(regions)
        self.profile = profile or RegionalReportProfile()
        self._rows, self._last_receipt, self._last_snapshot = [], -1., -1.

    def receive(self, report, *, received_at_s):
        if received_at_s < self._last_receipt or received_at_s < self._last_snapshot:
            raise ValueError("receipt stream must move forward from prior snapshots")
        _packets([(received_at_s, report)], received_at_s, self.profile)
        self._rows.append((received_at_s, report))
        self._last_receipt = received_at_s

    def snapshot(self, *, now_s):
        if now_s < max(self._last_receipt, self._last_snapshot):
            raise ValueError("snapshot cannot precede received evidence or previous snapshot")
        result = regional_report_features(self._rows, now_s=now_s, regions=self.regions, profile=self.profile)
        self._rows = [row for row in self._rows if now_s-row[1].frame.timestamp_s <= self.profile.history_s]
        self._last_snapshot = now_s
        return result


def received_reports(path):
    """Read only received-report JSONL rows; ignore command/offline channels."""
    previous = -1.
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") != "rsu_received_report":
                continue
            receipt = row.pop("received_at_s")
            row["kind"] = "cav_report"
            report = report_from_payload(row)
            if not isfinite(receipt) or receipt < previous or receipt < report.sent_at_s:
                raise ValueError("invalid or nonmonotone recorded receipt stream")
            previous = receipt
            yield receipt, report


def iter_regional_summaries(path, *, query_times_s, regions=DEFAULT_REGIONS, profile=None):
    """Streaming as-of reconstruction; packets received later never enter a row."""
    buffer = RegionalReportBuffer(regions=regions, profile=profile)
    packets = iter(received_reports(path))
    pending = next(packets, None)
    for now_s in query_times_s:
        if not isfinite(now_s) or now_s < 0:
            raise ValueError("finite nonnegative query time required")
        while pending is not None and pending[0] <= now_s:
            receipt, report = pending
            buffer.receive(report, received_at_s=receipt)
            pending = next(packets, None)
        yield buffer.snapshot(now_s=now_s)
