"""Evaluation-only physical outcomes; never controller/RSU observations.

RequestLedger is extracted unchanged from env/damping_rollout.py. Braking
vehicle-seconds and integral acceleration squared are physical proxies, not
fuel energy or a collision-risk guarantee. Lifecycle boundaries remain 0.5 s.
"""
from bisect import bisect_right
from math import isfinite
import numpy as np
from traffic_shepherd.scenarios import TripRequest


class RequestLedger:
    """Discrete SUMO-clock system-time accounting, including source waiting."""

    def __init__(self, requests: list[TripRequest]):
        self.requests = {r.vehicle_id: r for r in requests}
        if len(self.requests) != len(requests) or not requests:
            raise ValueError("requested population must be nonempty and unique")
        self.times = sorted(r.depart_s for r in requests)
        self.inserted: set[str] = set()
        self.arrived: set[str] = set()
        self.time = 0.0
        self.system_vehicle_seconds = 0.0
        self.source_vehicle_seconds = 0.0

    def due(self, now: float) -> int:
        return bisect_right(self.times, now)

    def advance(self, now: float, departed: list[str], arrived: list[str]) -> float:
        if now <= self.time:
            raise ValueError("ledger time must increase")
        for ids, seen in ((departed, self.inserted), (arrived, self.arrived)):
            if set(ids) - self.requests.keys() or set(ids) & seen or len(ids) != len(set(ids)):
                raise ValueError("unknown or duplicate lifecycle event")
        if set(arrived) - (self.inserted | set(departed)):
            raise ValueError("arrival before insertion")
        if any(self.requests[v].depart_s > now + 1e-6 for v in departed):
            raise ValueError("insertion before requested departure")
        dt = now - self.time
        # Requests inside this interval contribute only their remaining time.
        prior_due = self.due(self.time)
        new_times = self.times[prior_due : self.due(now)]
        requested_area = prior_due * dt + sum(now - value for value in new_times)
        # Arrival/insertion events are assigned to the reported tick boundary.
        # This convention is identical in every arm and has 0.5 s resolution.
        system = requested_area - len(self.arrived) * dt
        source = requested_area - len(self.inserted) * dt
        if min(system, source) < -1e-6:
            raise ValueError("negative service accounting")
        self.system_vehicle_seconds += max(0.0, system)
        self.source_vehicle_seconds += max(0.0, source)
        self.inserted.update(departed)
        self.arrived.update(arrived)
        self.time = now
        return max(0.0, system)


def braking_effort(accelerations, dt_s):
    """Count a < -3 m/s² and integrate a², matching the tested collector."""
    values = tuple(float(a) for a in accelerations)
    if not isfinite(dt_s) or dt_s <= 0 or not all(isfinite(a) for a in values):
        raise ValueError("invalid physical interval")
    array = np.asarray(values, dtype=float)
    return dict(braking_vehicle_seconds=float(np.sum(array < -3.0)) * dt_s,
                acceleration_energy=float(np.dot(array, array)) * dt_s)


def closing_ttc(gap_m, ego_speed_mps, leader_speed_mps):
    if not all(isfinite(x) for x in (gap_m, ego_speed_mps, leader_speed_mps)):
        raise ValueError("nonfinite TTC state")
    closing = ego_speed_mps - leader_speed_mps
    return max(0.0, gap_m) / closing if closing > 1e-6 else None


def ttc_exposure(api, vehicle_ids, dt_s):
    """Outcome-only 200 m leader query, physical gap restores SUMO minGap."""
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("invalid outcome interval")
    exposure, minimum = 0.0, None
    vehicle_ids = tuple(vehicle_ids)
    speeds = {v: float(api.vehicle.getSpeed(v)) for v in vehicle_ids}
    for vehicle in vehicle_ids:
        leader = api.vehicle.getLeader(vehicle, 200.0)
        if not leader or not leader[0] or leader[0] not in speeds:
            continue
        other, gap_without_min = leader
        ttc = closing_ttc(float(gap_without_min) + float(api.vehicle.getMinGap(vehicle)),
                          speeds[vehicle], speeds[other])
        if ttc is not None:
            minimum = ttc if minimum is None else min(minimum, ttc)
            exposure += dt_s if ttc < 3.0 else 0.0
    return exposure, minimum


def service_summary(ledger, now_s):
    population = len(ledger.requests)
    return dict(population=population, inserted=len(ledger.inserted), arrived=len(ledger.arrived),
                uninserted_due=ledger.due(now_s)-len(ledger.inserted),
                unfinished_requested=ledger.due(now_s)-len(ledger.arrived),
                system_seconds_per_requested_vehicle=ledger.system_vehicle_seconds/population,
                source_wait_vehicle_seconds=ledger.source_vehicle_seconds)
