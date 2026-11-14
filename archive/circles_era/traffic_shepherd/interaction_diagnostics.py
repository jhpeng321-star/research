"""Compressed, strictly offline half-second interaction and service evidence.

No policy/controller is imported and no simulator setter is available through
the shared read cache. Geometry follows the CAV sensor's same-lane semantics;
the route-aware 200 m leader used by the outcome TTC metric is kept separate.
Only one current/previous snapshot is retained in memory.
"""
from collections import defaultdict
import gzip
import hashlib
from itertools import groupby
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

from .metrics import RequestLedger, closing_ttc
from .sensor_batch import _ReadCache


SCHEMA = "offline_interaction_diagnostics/1"
FILENAME = "interactions.jsonl.gz"
DT_S = .5


def offline_read_cache(api):
    """One immutable physics instant; discard this facade before simulationStep.

    Existing physical metrics may share these reads with the observer. This
    cache is never supplied to sensing, CIRCLES, CommandOwnership or PolicyInput.
    """
    return SimpleNamespace(vehicle=_ReadCache(api.vehicle), lane=_ReadCache(api.lane))


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def same_lane_links(states):
    """Exact nearest bumper-gap candidate, including unequal lengths/tied positions.

    Front candidates have position >= ego position; rear candidates have
    strictly smaller position. The sensor breaks equal gap ties by vehicle ID.
    Negative gaps are preserved here as offline overlap evidence, never clipped
    into an apparently valid CAV observation.
    """
    lanes = defaultdict(list)
    for row in states:
        lanes[row["lane_id"]].append(row)
    links = {}
    for rows in lanes.values():
        groups = [list(group) for _, group in groupby(
            sorted(rows, key=lambda r: (r["lane_position_m"], r["vehicle_id"])),
            key=lambda r: r["lane_position_m"])]
        # Two best front candidates suffice to exclude ego while retaining the
        # minimum rear-bumper position from all equal/greater-position vehicles.
        fronts, suffix = {}, []
        for group in reversed(groups):
            suffix = sorted(suffix+group,
                key=lambda r: (r["lane_position_m"]-r["length_m"], r["vehicle_id"]))[:2]
            for row in group:
                fronts[row["vehicle_id"]] = next(
                    (r for r in suffix if r["vehicle_id"] != row["vehicle_id"]), None)
        rear = None
        for group in groups:
            for row in group:
                front = fronts[row["vehicle_id"]]
                links[row["vehicle_id"]] = dict(
                    same_lane_front_id=front["vehicle_id"] if front else None,
                    same_lane_front_gap_m=front["lane_position_m"]-front["length_m"]-row["lane_position_m"] if front else None,
                    same_lane_front_relative_speed_mps=front["speed_mps"]-row["speed_mps"] if front else None,
                    same_lane_rear_id=rear["vehicle_id"] if rear else None,
                    same_lane_rear_gap_m=row["lane_position_m"]-row["length_m"]-rear["lane_position_m"] if rear else None,
                    same_lane_rear_relative_speed_mps=rear["speed_mps"]-row["speed_mps"] if rear else None)
            rear = min(group, key=lambda r: r["vehicle_id"])
    return links


def capture_states(api, live_ids):
    """Read one measured instant; no source values are available to online code."""
    ids = sorted(live_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate live identity in offline snapshot")
    rows = []
    for vehicle in ids:
        lane = api.vehicle.getLaneID(vehicle)
        row = dict(vehicle_id=vehicle, edge_id=api.vehicle.getRoadID(vehicle), lane_id=lane,
            lane_position_m=float(api.vehicle.getLanePosition(vehicle)),
            lane_length_m=float(api.lane.getLength(lane)), length_m=float(api.vehicle.getLength(vehicle)),
            speed_mps=float(api.vehicle.getSpeed(vehicle)), acceleration_mps2=float(api.vehicle.getAcceleration(vehicle)))
        numbers = [row[k] for k in ("lane_position_m", "lane_length_m", "length_m", "speed_mps", "acceleration_mps2")]
        if not all(isfinite(v) for v in numbers) or min(row["length_m"], row["lane_length_m"], row["speed_mps"]) < 0:
            raise ValueError("invalid measured offline vehicle state")
        rows.append(row)
    links = same_lane_links(rows)
    speeds = {row["vehicle_id"]: row["speed_mps"] for row in rows}
    for row in rows:
        vehicle = row["vehicle_id"]
        row.update(links[vehicle])
        row["front_coverage_m"] = min(100., max(0., row["lane_length_m"]-row["lane_position_m"]))
        row["rear_coverage_m"] = min(100., max(0., row["lane_position_m"]-row["length_m"]))
        for side in ("front", "rear"):
            gap = row[f"same_lane_{side}_gap_m"]
            row[f"sensor_{side}_candidate_id"] = row[f"same_lane_{side}_id"] if (
                gap is not None and gap <= row[f"{side}_coverage_m"]) else None
        row["sensor_overlap_invalid"] = any(
            row[f"sensor_{side}_candidate_id"] is not None and row[f"same_lane_{side}_gap_m"] < 0
            for side in ("front", "rear"))
        leader = api.vehicle.getLeader(vehicle, 200.)
        row.update(ttc_leader_id=leader[0] if leader and leader[0] else None,
            ttc_leader_state_available=False, ttc_gap_m=None, ttc_closing_speed_mps=None,
            closing_ttc_s=None, low_ttc_vehicle_seconds=0.)
        if leader and leader[0] and leader[0] in speeds:
            gap = float(leader[1])+float(api.vehicle.getMinGap(vehicle))
            ttc = closing_ttc(gap, row["speed_mps"], speeds[leader[0]])
            row.update(ttc_leader_state_available=True, ttc_gap_m=gap,
                ttc_closing_speed_mps=row["speed_mps"]-speeds[leader[0]], closing_ttc_s=ttc,
                low_ttc_vehicle_seconds=DT_S if ttc is not None and ttc < 3. else 0.)
        row["braking_vehicle_seconds"] = DT_S if row["acceleration_mps2"] < -3. else 0.
        row["acceleration_effort"] = DT_S*row["acceleration_mps2"]**2
    return rows


class OfflineInteractionObserver:
    """Stream header + t=0 and every 0.5 s sample + completion footer.

    Integer-time samples follow the preceding physics step and precede the next
    control decision. Initial t=0 precedes every action. Contributions at t>0
    use end-of-interval measurements, exactly like the existing physical metrics.
    The final t=horizon snapshot precedes shutdown release writes.
    """
    def __init__(self, path, *, requests, horizon_s, edge_regions=None):
        if type(horizon_s) is not int or horizon_s <= 0:
            raise ValueError("positive registered integer horizon required")
        self.path, self.horizon_s = Path(path), horizon_s
        groups = defaultdict(list)
        for request in requests:
            groups[request.route].append(request)
        self.ledgers = {key: RequestLedger(value) for key, value in sorted(groups.items())}
        if not self.ledgers or len({r.vehicle_id for r in requests}) != len(requests):
            raise ValueError("nonempty unique exogenous requests required")
        self.request_ids = frozenset(r.vehicle_id for r in requests)
        self.edge_regions = dict(edge_regions or {})
        self.previous, self.last_time, self.samples, self.vehicle_rows = {}, None, 0, 0
        self._hash, self._raw_bytes, self._closed = hashlib.sha256(), 0, False
        self._raw = self.path.open("xb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", compresslevel=1, mtime=0, fileobj=self._raw)
        self._write(dict(kind="header", schema=SCHEMA, channel="offline_evaluation_only",
            physics_dt_s=DT_S, horizon_s=horizon_s, sensor_range_m=100., ttc_leader_range_m=200.,
            sample_timing="initial_then_post_physics_before_next_control_or_shutdown",
            geometry="same_lane_sensor_candidates_separate_from_route_aware_ttc_leader",
            service_regions="request_origin_route_not_current_vehicle_location",
            edge_regions=self.edge_regions,
            requests=[dict(vehicle_id=r.vehicle_id, requested_departure_s=r.depart_s,
                origin_route=r.route, is_cav=r.is_cav) for r in sorted(requests, key=lambda r: r.vehicle_id)]))

    def _write(self, row):
        raw = (json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n").encode("utf-8")
        self._gzip.write(raw)
        self._hash.update(raw)
        self._raw_bytes += len(raw)

    def sample(self, api, *, now_s, live_ids, departed_ids=(), arrived_ids=(),
               colliding_ids=(), new_colliding_ids=(), starting_teleports=0, starting_teleport_ids=None):
        if self._closed or not isfinite(now_s) or now_s != (0. if self.last_time is None else self.last_time+DT_S):
            raise ValueError("offline samples must cover the exact 0.5 s clock from zero")
        if now_s > self.horizon_s:
            raise ValueError("offline sample beyond registered horizon")
        if type(starting_teleports) is not int or starting_teleports < 0:
            raise ValueError("nonnegative integer starting teleport count required")
        for ids in (departed_ids, arrived_ids):
            if len(ids) != len(set(ids)) or set(ids)-self.request_ids:
                raise ValueError("unknown or duplicate lifecycle event")
        if not set(new_colliding_ids) <= set(colliding_ids):
            raise ValueError("new collision involvement absent from current collision IDs")
        if starting_teleport_ids is not None and (
                len(starting_teleport_ids) != starting_teleports
                or len(starting_teleport_ids) != len(set(starting_teleport_ids))):
            raise ValueError("teleport event identity/count mismatch")
        states = capture_states(api, live_ids)
        current = {row["vehicle_id"]: (row["edge_id"], row["lane_id"]) for row in states}
        transitions = []
        regions = {}
        for row in states:
            before = self.previous.get(row["vehicle_id"])
            after = current[row["vehicle_id"]]
            if before is not None and before != after:
                transitions.append(dict(vehicle_id=row["vehicle_id"], from_edge_id=before[0], from_lane_id=before[1],
                    to_edge_id=after[0], to_lane_id=after[1],
                    kind="within_edge_lane_change" if before[0] == after[0] else "edge_transition"))
            if now_s == 0:
                for key in ("braking_vehicle_seconds", "acceleration_effort", "low_ttc_vehicle_seconds"):
                    row[key] = 0.
            region = self.edge_regions.get(row["edge_id"], row["edge_id"])
            total = regions.setdefault(region, dict(vehicle_count=0, speed_sum_mps=0.,
                braking_vehicle_seconds=0., acceleration_effort=0., low_ttc_vehicle_seconds=0.))
            total["vehicle_count"] += 1
            total["speed_sum_mps"] += row["speed_mps"]
            for key in ("braking_vehicle_seconds", "acceleration_effort", "low_ttc_vehicle_seconds"):
                total[key] += row[key]
        service = {}
        for origin, ledger in self.ledgers.items():
            previous_system, previous_source = ledger.system_vehicle_seconds, ledger.source_vehicle_seconds
            if now_s > 0:
                ledger.advance(now_s, [v for v in departed_ids if v in ledger.requests],
                    [v for v in arrived_ids if v in ledger.requests])
            elif departed_ids or arrived_ids:
                raise ValueError("initial sample cannot advance lifecycle events")
            service[origin] = dict(requested_population=len(ledger.requests), requested_due=ledger.due(now_s),
                inserted=len(ledger.inserted), arrived=len(ledger.arrived),
                source_waiting=ledger.due(now_s)-len(ledger.inserted),
                requested_not_arrived=ledger.due(now_s)-len(ledger.arrived),
                system_vehicle_seconds=ledger.system_vehicle_seconds,
                source_wait_vehicle_seconds=ledger.source_vehicle_seconds,
                interval_system_vehicle_seconds=ledger.system_vehicle_seconds-previous_system,
                interval_source_wait_vehicle_seconds=ledger.source_vehicle_seconds-previous_source)
        self._write(dict(kind="sample", time_s=now_s, interval_start_s=max(0., now_s-DT_S),
            interval_dt_s=DT_S if now_s else 0., vehicles=states, regions=regions, service_by_origin=service,
            lane_transitions=transitions, first_observed_ids=sorted(current.keys()-self.previous.keys()),
            disappeared_ids=sorted(self.previous.keys()-current.keys()), departed_ids=sorted(departed_ids),
            arrived_ids=sorted(arrived_ids), colliding_ids=sorted(colliding_ids),
            new_collision_involvement_ids=sorted(new_colliding_ids), starting_teleports=starting_teleports,
            starting_teleport_ids=sorted(starting_teleport_ids) if starting_teleport_ids is not None else None,
            starting_teleport_ids_available=starting_teleport_ids is not None))
        self.previous, self.last_time = current, now_s
        self.samples += 1
        self.vehicle_rows += len(states)

    def close(self):
        if not self._closed:
            self._write(dict(kind="footer", complete=self.last_time == self.horizon_s,
                final_time_s=self.last_time, sample_count=self.samples, vehicle_rows=self.vehicle_rows))
            self._gzip.close()
            self._raw.close()
            self._closed = True
        return dict(enabled=True, schema=SCHEMA, file=FILENAME, compression="gzip_level_1_mtime_0",
            sha256=file_sha256(self.path), uncompressed_sha256=self._hash.hexdigest(),
            compressed_bytes=self.path.stat().st_size, uncompressed_bytes=self._raw_bytes,
            sample_count=self.samples, vehicle_rows=self.vehicle_rows, final_time_s=self.last_time,
            complete=self.last_time == self.horizon_s, online_input=False)
