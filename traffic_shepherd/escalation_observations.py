"""CAV-report geometry capture and a separate offline traffic measurement channel."""
from itertools import product
from math import isfinite, prod
from statistics import mean, pvariance

from .reporting import CAVReport
from .sensing import NeighborMeasurement, SensorFrame


def report_from_payload(payload):
    """Reconstruct only the typed report schema; unexpected privileged fields fail."""
    data = dict(payload)
    if data.pop("kind", "cav_report") != "cav_report":
        raise ValueError("expected CAV report telemetry")
    frame = dict(data.pop("frame"))
    for key in ("front", "rear"):
        if frame.get(key) is not None:
            frame[key] = NeighborMeasurement(**frame[key])
    return CAVReport(frame=SensorFrame(**frame), **data)


def candidate_geometry(reports, *, now_s, required_lanes, recruitment_band_m,
                       max_report_age_s, max_groups=256):
    """Observe candidate separation WITHOUT declaring an arbitrary span feasible.

Input must be packets already received by the RSU. Read the buffer at now_s
before calling. No true density, penetration, vehicle type or global health is
accepted. Required lanes come from a static map, never from observed occupancy.
Truncated enumeration is explicitly incomplete and cannot prove no opportunity.
"""
    if not required_lanes or len(required_lanes) != len(set(required_lanes)):
        raise ValueError("explicit distinct required map lanes needed")
    lo, hi = recruitment_band_m
    if not all(isfinite(v) for v in (now_s, lo, hi, max_report_age_s)) or now_s < 0 or not 0 <= lo < hi or max_report_age_s < 0:
        raise ValueError("invalid observation clock, band or age")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("positive candidate budget required")
    seen, by_lane, stale = set(), {lane: [] for lane in required_lanes}, 0
    for report in reports:
        if not isinstance(report, CAVReport):
            raise TypeError("only CAVReport observations are accepted")
        f = report.frame
        if f.vehicle_id in seen:
            raise ValueError("one latest received packet per CAV required")
        seen.add(f.vehicle_id)
        if report.sent_at_s > now_s or not 0 <= now_s-f.timestamp_s <= max_report_age_s:
            stale += 1
            continue
        if f.lane_id in by_lane and f.distance_to_merge_m is not None and lo <= f.distance_to_merge_m <= hi:
            by_lane[f.lane_id].append(f)
    for rows in by_lane.values():
        rows.sort(key=lambda f: (f.distance_to_merge_m, f.vehicle_id))
    total = prod(len(rows) for rows in by_lane.values())
    groups = []
    for index, members in enumerate(product(*(by_lane[lane] for lane in required_lanes))):
        if index >= max_groups:
            break
        distances, speeds = [f.distance_to_merge_m for f in members], [f.speed_mps for f in members]
        groups.append(dict(
            member_ids=[f.vehicle_id for f in members], lanes=list(required_lanes),
            initial_span_m=max(distances)-min(distances), speed_spread_mps=max(speeds)-min(speeds),
            nearest_merge_distance_m=min(distances), mean_speed_mps=mean(speeds),
            maximum_report_age_s=max(now_s-f.timestamp_s for f in members),
            front_gap_m=[f.front.gap_m if f.front else None for f in members],
            rear_gap_m=[f.rear.gap_m if f.rear else None for f in members],
            front_relative_speed_mps=[f.front.relative_speed_mps if f.front else None for f in members],
            rear_relative_speed_mps=[f.rear.relative_speed_mps if f.rear else None for f in members],
            front_coverage_m=[f.front_coverage_m for f in members],
            rear_coverage_m=[f.rear_coverage_m for f in members],
        ))
    return dict(reporting_cavs=len(seen), rejected_stale_or_future=stale,
                missing_reported_lanes=[lane for lane, rows in by_lane.items() if not rows],
                total_candidate_combinations=total, enumerated_combinations=len(groups),
                enumeration_complete=total <= max_groups, candidates=groups,
                physically_feasible_count=None, lane_occupancy_known=False,
                interpretation="observed_geometry_not_a_formation_success_label")


def region_statistics(speeds, *, summed_lane_length_m):
    """Evaluation-only measured density; nominal demand is not called density."""
    if not isfinite(summed_lane_length_m) or summed_lane_length_m <= 0:
        raise ValueError("positive surveyed lane length needed")
    speeds = list(speeds)
    if any(not isfinite(v) or v < 0 for v in speeds):
        raise ValueError("invalid measured traffic speed")
    return dict(vehicles=len(speeds), density_vehicles_per_km_lane=1000*len(speeds)/summed_lane_length_m,
                mean_speed_mps=mean(speeds) if speeds else None,
                speed_std_mps=pvariance(speeds)**.5 if speeds else None,
                stopped_below_0p1_mps=sum(v < .1 for v in speeds),
                summed_lane_length_m=summed_lane_length_m)


class OfflineTrafficObserver:
    """Full simulation measurements written to a separate file, never to the RSU."""
    def __init__(self, api, config):
        self.api = api
        self.regions = {}
        for name, edge in config.edge_ids.items():
            lanes = [f"{edge}_{i}" for i in range(api.edge.getLaneNumber(edge))]
            self.regions[name] = (lanes, sum(float(api.lane.getLength(lane)) for lane in lanes))

    def sample(self, *, now_s):
        regions = {}
        for name, (lanes, length) in self.regions.items():
            ids = {v for lane in lanes for v in self.api.lane.getLastStepVehicleIDs(lane)}
            regions[name] = region_statistics([float(self.api.vehicle.getSpeed(v)) for v in sorted(ids)],
                                              summed_lane_length_m=length)
        return dict(timestamp_s=now_s, channel="offline_evaluation_only", regions=regions)
