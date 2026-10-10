"""CAV-origin observations for the RSU; no simulator-global traffic inputs.

An RSU can aggregate timestamped self states and local detections. Neighbor
detections have only reporter-local front/rear identity, so they are not
silently deduplicated into a census of HDVs. Missing messages do not mean an
empty road. No fitted predictor or gate performance is claimed here.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from statistics import mean, pvariance

from traffic_shepherd.sensing import SensorFrame


@dataclass(frozen=True)
class CAVReport:
    frame: SensorFrame
    sent_at_s: float
    owner: str
    network_output_mps2: float | None = None
    issued_acceleration_mps2: float | None = None
    fallback_reason: str | None = None

    def __post_init__(self):
        if not isfinite(self.sent_at_s) or self.sent_at_s < self.frame.timestamp_s:
            raise ValueError("report sent before observation or at invalid time")
        if self.owner not in {"native", "damping", "wall"}:
            raise ValueError("unknown command ownership")
        if any(v is not None and not isfinite(v) for v in
               (self.network_output_mps2, self.issued_acceleration_mps2)):
            raise ValueError("nonfinite command telemetry")

    def payload(self):
        return asdict(self)


class RSUReportBuffer:
    """Explicit message receipt and history; the only inputs are CAV reports."""

    def __init__(self, *, max_age_s, history_s):
        if not all(isfinite(v) and v >= 0 for v in (max_age_s, history_s)) or history_s < max_age_s:
            raise ValueError("invalid report age/history window")
        self.max_age_s, self.history_s = max_age_s, history_s
        self._reports = {}

    def receive(self, report: CAVReport, *, received_at_s):
        if not isfinite(received_at_s) or received_at_s < report.sent_at_s:
            raise ValueError("report received before it was sent")
        rows = self._reports.setdefault(report.frame.vehicle_id, [])
        if rows and received_at_s < rows[-1][0]:
            raise ValueError("receipt times must be monotone per reporter")
        # Equal observation timestamps must have one consistent packet identity.
        for _, previous in rows:
            if previous.frame.timestamp_s == report.frame.timestamp_s:
                if previous != report:
                    raise ValueError("conflicting report for the same observation")
                return
        rows.append((received_at_s, report))
        cutoff = received_at_s - self.history_s
        self._reports[report.frame.vehicle_id] = [r for r in rows if r[0] >= cutoff]

    def reports(self, *, now_s, fresh_only=True):
        if not isfinite(now_s):
            raise ValueError("invalid RSU time")
        latest = []
        for rows in self._reports.values():
            available = [r for received, r in rows if received <= now_s and now_s-r.frame.timestamp_s <= self.history_s]
            if available:
                report = max(available, key=lambda r: r.frame.timestamp_s)
                if not fresh_only or now_s-report.frame.timestamp_s <= self.max_age_s:
                    latest.append(report)
        return sorted(latest, key=lambda r: r.frame.vehicle_id)

    def features(self, *, now_s):
        rows = self.reports(now_s=now_s)
        speeds = [r.frame.speed_mps for r in rows]
        ages = [now_s-r.frame.timestamp_s for r in rows]
        return dict(reporting_cavs=len(rows), observation_available=bool(rows),
                    mean_cav_speed_mps=mean(speeds) if speeds else None,
                    cav_speed_variance=pvariance(speeds) if speeds else None,
                    maximum_observation_age_s=max(ages) if ages else None,
                    front_detections=sum(r.frame.front is not None for r in rows),
                    rear_detections=sum(r.frame.rear is not None for r in rows),
                    reporter_front_coverage_m=[r.frame.front_coverage_m for r in rows],
                    reporter_rear_coverage_m=[r.frame.rear_coverage_m for r in rows],
                    coverage_is_union=False, total_traffic_count_known=False)
