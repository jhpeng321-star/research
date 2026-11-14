"""Offline evaluation: run SUMO and compute metrics from tripinfo / FCD XML.

- `run_sumo_evaluation`       — run SUMO subprocess to produce tripinfo + FCD
- `evaluate_travel_time`      — mean / P90 travel time (all, mainline, ramp)
- `evaluate_ttc`              — count Time-To-Collision events below a threshold
- `evaluate_hard_braking`     — count deceleration < -4.5 m/s^2 events from FCD
- `parse_sumo_stats`          — parse statistic-output (arrived, teleports, collisions)
- `evaluate_type_distribution`— driver-type counts/fractions from tripinfo
- `evaluate_run`              — unified per-run record shared by the All-HDV
                                pure-SUMO path and the TraCI policy path
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from config.constants import (
    DEFAULT_VEH_LENGTH,
    MAINLINE_EDGE,
    MERGE_DROPPED_LANE_INDEX,
    MERGE_EDGE,
    MERGE_SURVIVING_LANE_INDEX,
    RAMP_EDGE,
)
from env.traci_query import percentile

DEFAULT_TTC_THRESHOLD = 1.5  # s — stricter than before; catches risky close-following
# Deceleration below this (m/s^2) between consecutive FCD snapshots counts as
# one hard-braking event (paper metric definition).
HARD_BRAKING_THRESHOLD = -4.5


# ---------------------------------------------------------------------------
# SUMO runner
# ---------------------------------------------------------------------------
def run_sumo_evaluation(
    sumo_binary: str,
    sumo_config: Path,
    project_root: Path,
    tripinfo_output: Path,
    fcd_output: Path,
    fcd_period: int = 1,
    seed: Optional[int] = None,
    stats_output: Optional[Path] = None,
    collision_output: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Run SUMO with tripinfo and FCD outputs and return the output paths.

    Args:
        sumo_binary: SUMO binary name or full path (e.g. "sumo" or "sumo-gui").
        sumo_config: Path to the base SUMO configuration file.
        project_root: Directory where SUMO should run (cwd for the subprocess).
        tripinfo_output: Path for the tripinfo XML output.
        fcd_output: Path for the FCD XML output.
        fcd_period: FCD recording period in seconds (default 1).
        seed: Optional SUMO random seed (passed through as ``--seed``).
        stats_output: Optional path for the statistic-output XML (CLI override
            of the sumocfg value), used for collisions/teleports on the
            pure-SUMO path.
        collision_output: Optional path for the collision-output XML.

    Returns:
        (tripinfo_output, fcd_output) paths after SUMO finishes.
    """
    # SUMO uses --device.fcd.period to control FCD output frequency.
    cmd = [
        sumo_binary,
        "-c",
        str(sumo_config),
        "--tripinfo-output",
        str(tripinfo_output),
        "--fcd-output",
        str(fcd_output),
        "--device.fcd.period",
        str(fcd_period),
        "--no-step-log",
    ]
    if stats_output is not None:
        cmd += ["--statistic-output", str(stats_output)]
    if collision_output is not None:
        cmd += ["--collision-output", str(collision_output)]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    subprocess.run(cmd, cwd=str(project_root), check=True)
    return tripinfo_output, fcd_output


# ---------------------------------------------------------------------------
# Travel-time metrics (tripinfo XML)
# ---------------------------------------------------------------------------
def _edge_from_lane(lane_id: str) -> str:
    """Return the edge ID from a SUMO lane ID (edge_lane_index)."""
    return lane_id.rsplit("_", 1)[0]


def parse_tripinfo(tripinfo_path: Path) -> List[dict]:
    """Parse a SUMO tripinfo XML file and return a list of trip records."""
    tree = ET.parse(str(tripinfo_path))
    root = tree.getroot()
    trips = []
    for trip in root.findall("tripinfo"):
        trips.append(
            {
                "id": trip.get("id"),
                "depart": float(trip.get("depart", 0.0)),
                "arrival": float(trip.get("arrival", 0.0)),
                "duration": float(trip.get("duration", 0.0)),
                "departLane": trip.get("departLane"),
                "arrivalLane": trip.get("arrivalLane"),
                "routeLength": float(trip.get("routeLength", 0.0)),
                "timeLoss": float(trip.get("timeLoss", 0.0)),
                "vType": trip.get("vType"),
                "vaporized": trip.get("vaporized", ""),
            }
        )
    return trips


def evaluate_travel_time(tripinfo_path: Path) -> Dict[str, dict]:
    """Compute travel-time metrics split by origin (mainline lane vs ramp).

    Returns a dict with keys:
      - "all": overall stats
      - "mainline": stats for vehicles starting on the mainline edge
      - "ramp": stats for vehicles starting on the ramp edge

    Each stat dict contains:
      - "count": number of trips
      - "mean": mean travel time
      - "p90": 90th-percentile travel time
      - "trips": list of (vehicle_id, duration) tuples
    """
    trips = parse_tripinfo(tripinfo_path)

    mainline_durations = []
    ramp_durations = []
    all_durations = []
    mainline_trips = []
    ramp_trips = []
    all_trips = []

    for trip in trips:
        edge = _edge_from_lane(trip["departLane"])
        duration = trip["duration"]
        all_durations.append(duration)
        all_trips.append((trip["id"], duration))
        # Bottleneck scenario: all vehicles on upstream are "mainline"
        if edge == "upstream":
            mainline_durations.append(duration)
            mainline_trips.append((trip["id"], duration))
        elif edge == MAINLINE_EDGE:
            mainline_durations.append(duration)
            mainline_trips.append((trip["id"], duration))
        elif edge == RAMP_EDGE:
            ramp_durations.append(duration)
            ramp_trips.append((trip["id"], duration))

    def _stats(durations, trip_pairs):
        if not durations:
            return {"count": 0, "mean": float("nan"), "p90": float("nan"), "trips": []}
        return {
            "count": len(durations),
            "mean": sum(durations) / len(durations),
            "p90": percentile(durations, 90),
            "trips": trip_pairs,
        }

    return {
        "all": _stats(all_durations, all_trips),
        "mainline": _stats(mainline_durations, mainline_trips),
        "ramp": _stats(ramp_durations, ramp_trips),
    }


# ---------------------------------------------------------------------------
# TTC metrics (FCD XML)
# ---------------------------------------------------------------------------
def parse_fcd(fcd_path: Path) -> Dict[float, List[dict]]:
    """Parse a SUMO FCD XML file; returns dict of ``time -> list of vehicle states``.

    Uses line-by-line regex extraction so that duplicate attributes
    (a known SUMO FCD quirk) don't break parsing.
    """
    import re

    _TIME = re.compile(r'time="([^"]+)"')
    _ATTR = re.compile(r'\s+(id|x|y|speed|lane|pos)="([^"]+)"')

    timesteps: Dict[float, List[dict]] = {}
    current_time: Optional[float] = None

    with fcd_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("<timestep"):
                m = _TIME.search(line)
                if m:
                    current_time = float(m.group(1))
                    timesteps.setdefault(current_time, [])
            elif line.startswith("</timestep"):
                current_time = None
            elif line.startswith("<vehicle") and current_time is not None:
                attrs: dict = {}
                for m in _ATTR.finditer(line):
                    name, val = m.group(1), m.group(2)
                    # Keep only the *first* occurrence (ignore duplicates)
                    if name not in attrs:
                        attrs[name] = val
                if len(attrs) >= 6:  # must have id, x, y, speed, lane, pos
                    try:
                        timesteps[current_time].append(
                            {
                                "id": attrs["id"],
                                "x": float(attrs["x"]),
                                "y": float(attrs["y"]),
                                "speed": float(attrs["speed"]),
                                "lane": attrs["lane"],
                                "pos": float(attrs["pos"]),
                            }
                        )
                    except (KeyError, ValueError):
                        pass  # malformed line — skip
    return timesteps


def _compute_ttc(follower: dict, leader: dict) -> float:
    """Return TTC between a follower and a leader on the same lane.

    TTC is computed only if the follower is behind the leader and faster.
    Gap is bumper-to-bumper: ``pos`` is the front position on the lane, so
    the leader's vehicle length must be subtracted (this was the 2026-07
    "TTC always 0" bug — every gap was inflated by ~5 m).
    """
    # Same lane required (lane ID includes edge + index)
    if follower["lane"] != leader["lane"]:
        return float("inf")
    distance = leader["pos"] - follower["pos"] - DEFAULT_VEH_LENGTH
    speed_diff = follower["speed"] - leader["speed"]
    if speed_diff <= 0:
        return float("inf")
    if distance <= 0:
        return 0.0  # overlapped while closing — collision imminent
    return distance / speed_diff


def evaluate_ttc(fcd_path: Path, threshold: float = DEFAULT_TTC_THRESHOLD) -> dict:
    """Count TTC events below the threshold from a SUMO FCD file.

    A "TTC event" is counted once per (time, follower) pair where the
    Time-To-Collision to the nearest leading vehicle on the same lane is
    below the threshold.

    Returns a dict with:
      - "threshold": the TTC threshold used
      - "total_events": total count of TTC < threshold events
      - "events_by_timestep": dict of time -> count
    """
    timesteps = parse_fcd(fcd_path)
    total_events = 0
    events_by_timestep = {}

    for time, vehicles in sorted(timesteps.items()):
        # Group vehicles by lane and sort by longitudinal position.
        lane_groups = {}
        for veh in vehicles:
            lane_groups.setdefault(veh["lane"], []).append(veh)
        for lane in lane_groups:
            lane_groups[lane].sort(key=lambda v: v["pos"])

        step_count = 0
        for lane, vehs in lane_groups.items():
            # Vehicles are sorted by position ascending; vehs[i-1] is behind vehs[i].
            for i in range(1, len(vehs)):
                follower = vehs[i - 1]
                leader = vehs[i]
                ttc = _compute_ttc(follower, leader)
                if ttc < threshold:
                    step_count += 1
                    total_events += 1
        if step_count > 0:
            events_by_timestep[time] = step_count

    return {
        "threshold": threshold,
        "total_events": total_events,
        "events_by_timestep": events_by_timestep,
    }


def evaluate_side_ttc(
    fcd_path: Path,
    threshold: float = DEFAULT_TTC_THRESHOLD,
    merge_edge: str = MERGE_EDGE,
    dropped_lane: int = MERGE_DROPPED_LANE_INDEX,
    surviving_lane: int = MERGE_SURVIVING_LANE_INDEX,
) -> dict:
    """Count side-conflict TTC events at the merge lane-drop from a SUMO FCD file.

    A side conflict exists only where two vehicles must race for the SAME
    lane: on the merge edge (3 lanes -> 2 downstream), the dropped lane ends,
    so pairs on (dropped_lane, surviving_lane) are converging. Parallel pairs
    on continuing lanes are ignored. Conflict time = net longitudinal gap /
    relative speed; counted once per (time, pair) below ``threshold``
    (mirrors env.traci_query.side_ttc used by the reward).
    """
    timesteps = parse_fcd(fcd_path)
    total_events = 0
    events_by_timestep = {}

    for time, vehicles in sorted(timesteps.items()):
        merge_vehs = [v for v in vehicles if v["lane"].rsplit("_", 1)[0] == merge_edge]
        step_count = 0
        for i in range(len(merge_vehs)):
            for j in range(i + 1, len(merge_vehs)):
                a, b = merge_vehs[i], merge_vehs[j]
                la = int(a["lane"].rsplit("_", 1)[1])
                lb = int(b["lane"].rsplit("_", 1)[1])
                if {la, lb} != {dropped_lane, surviving_lane}:
                    continue
                d = abs(a["pos"] - b["pos"]) - DEFAULT_VEH_LENGTH
                if d > 20.0:
                    continue
                if a["pos"] < b["pos"]:
                    closing = a["speed"] - b["speed"]
                else:
                    closing = b["speed"] - a["speed"]
                if d <= 0.0:
                    if closing > 0.1:
                        step_count += 1
                    continue
                if closing <= 0.1:
                    continue
                if d / closing < threshold:
                    step_count += 1
        if step_count > 0:
            total_events += step_count
            events_by_timestep[time] = step_count

    return {
        "threshold": threshold,
        "total_events": total_events,
        "events_by_timestep": events_by_timestep,
    }


# ---------------------------------------------------------------------------
# Hard-braking metrics (FCD XML, streaming)
# ---------------------------------------------------------------------------
def evaluate_hard_braking(fcd_path: Path, threshold: float = HARD_BRAKING_THRESHOLD) -> dict:
    """Count hard-braking events from a SUMO FCD file.

    One event is counted per (vehicle, timestep) pair where the deceleration
    between two consecutive FCD snapshots of the same vehicle is below the
    threshold (default -4.5 m/s^2).

    Returns a dict with:
      - "threshold": the deceleration threshold used (m/s^2)
      - "total_events": total count of hard-braking events
      - "events_by_timestep": dict of time -> count
    """
    import re

    _TIME = re.compile(r'time="([^"]+)"')
    _ATTR = re.compile(r'\s+(id|speed)="([^"]+)"')

    total_events = 0
    events_by_timestep: Dict[float, int] = {}
    last_speed: Dict[str, Tuple[float, float]] = {}  # veh_id -> (time, speed)

    current_time: Optional[float] = None
    with fcd_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("<timestep"):
                m = _TIME.search(line)
                if m:
                    current_time = float(m.group(1))
            elif line.startswith("</timestep"):
                if current_time is not None:
                    # Drop vehicles that left the network
                    for vid in list(last_speed):
                        if last_speed[vid][0] != current_time:
                            del last_speed[vid]
                current_time = None
            elif line.startswith("<vehicle") and current_time is not None:
                attrs: dict = {}
                for m in _ATTR.finditer(line):
                    name = m.group(1)
                    if name not in attrs:
                        attrs[name] = m.group(2)
                if "id" not in attrs or "speed" not in attrs:
                    continue
                vid = attrs["id"]
                speed = float(attrs["speed"])
                prev = last_speed.get(vid)
                if prev is not None:
                    dt = current_time - prev[0]
                    if dt > 0 and (speed - prev[1]) / dt < threshold:
                        total_events += 1
                        events_by_timestep[current_time] = (
                            events_by_timestep.get(current_time, 0) + 1
                        )
                last_speed[vid] = (current_time, speed)

    return {
        "threshold": threshold,
        "total_events": total_events,
        "events_by_timestep": events_by_timestep,
    }


# ---------------------------------------------------------------------------
# statistic-output parsing (pure-SUMO safety / teleport counters)
# ---------------------------------------------------------------------------
def parse_sumo_stats(stats_path: Path) -> dict:
    """Parse a SUMO statistic-output XML file.

    Returns a dict with:
      - "arrived": vehicleTripStatistics count (completed trips)
      - "duration_s": simulation duration from the performance element
      - "teleports": teleports total
      - "collisions": safety collisions
      - "mean_speed" / "mean_time_loss" / "mean_duration": trip statistics
    """
    root = ET.parse(str(stats_path)).getroot()
    perf = root.find("performance")
    trip = root.find("vehicleTripStatistics")
    tele = root.find("teleports")
    safe = root.find("safety")

    def _float(elem, attr):
        return float(elem.get(attr)) if elem is not None and elem.get(attr) else None

    def _int(elem, attr):
        return int(elem.get(attr)) if elem is not None and elem.get(attr) else 0

    return {
        "arrived": _int(trip, "count"),
        "duration_s": _float(perf, "duration"),
        "teleports": _int(tele, "total"),
        "collisions": _int(safe, "collisions"),
        "mean_speed": _float(trip, "speed"),
        "mean_time_loss": _float(trip, "timeLoss"),
        "mean_duration": _float(trip, "duration"),
    }


# ---------------------------------------------------------------------------
# Driver-type distribution (tripinfo vType attribute)
# ---------------------------------------------------------------------------
def evaluate_type_distribution(tripinfo_path: Path) -> dict:
    """Count driver types (vType) over all trip records in a tripinfo file.

    The counts verify that the route file's embedded vTypeDistribution
    (~10/70/20 aggressive/normal/cautious) actually materialized in the run.
    """
    counts: Dict[str, int] = {}
    total = 0
    for trip in parse_tripinfo(tripinfo_path):
        vtype = trip["vType"] or "unknown"
        counts[vtype] = counts.get(vtype, 0) + 1
        total += 1
    fractions = {vtype: count / total for vtype, count in counts.items()} if total else {}
    return {"counts": counts, "fractions": fractions, "total": total}


# ---------------------------------------------------------------------------
# Unified per-run record (same metric definitions for All-HDV and policies)
# ---------------------------------------------------------------------------
def evaluate_run(
    tripinfo_path: Path,
    fcd_path: Path,
    *,
    stats_path: Optional[Path] = None,
    collisions: Optional[int] = None,
    teleports: Optional[int] = None,
    sim_seconds: Optional[float] = None,
    ttc_threshold: float = DEFAULT_TTC_THRESHOLD,
) -> dict:
    """Compute the full metric set for one simulation run.

    Args:
        tripinfo_path: tripinfo XML (travel time, arrived count, types).
        fcd_path: FCD XML (TTC + hard braking).
        stats_path: optional statistic-output XML — source of collisions,
            teleports and simulation duration on the pure-SUMO path.
        collisions: explicit collision count (TraCI path); wins over stats.
        teleports: explicit teleport count (TraCI path); wins over stats.
        sim_seconds: explicit simulated duration; wins over stats / fallback
            (last arrival time in the tripinfo).
        ttc_threshold: TTC event threshold (seconds).

    Returns a dict with travel_time, ttc, ttc_side, hard_braking sub-records plus
    arrived, sim_seconds, throughput_veh_per_hour, mean_speed, mean_time_loss,
    collisions, teleports, type_counts and type_fractions.
    """
    trips = parse_tripinfo(tripinfo_path)
    arrived_trips = [t for t in trips if not t["vaporized"]]
    arrived = len(arrived_trips)

    stats = parse_sumo_stats(stats_path) if stats_path is not None else None

    if sim_seconds is None:
        if stats is not None and stats["duration_s"]:
            sim_seconds = stats["duration_s"]
        else:
            sim_seconds = max((t["arrival"] for t in arrived_trips), default=0.0)
    sim_hours = sim_seconds / 3600.0 if sim_seconds else 0.0
    throughput = arrived / sim_hours if sim_hours > 0 else float("nan")

    speeds = [t["routeLength"] / t["duration"] for t in arrived_trips if t["duration"] > 0]
    mean_speed = sum(speeds) / len(speeds) if speeds else float("nan")
    time_losses = [t["timeLoss"] for t in arrived_trips]
    mean_time_loss = sum(time_losses) / len(time_losses) if time_losses else float("nan")

    if collisions is None and stats is not None:
        collisions = stats["collisions"]
    if teleports is None and stats is not None:
        teleports = stats["teleports"]

    type_dist = evaluate_type_distribution(tripinfo_path)

    return {
        "travel_time": evaluate_travel_time(tripinfo_path),
        "ttc": evaluate_ttc(fcd_path, threshold=ttc_threshold),
        "ttc_side": evaluate_side_ttc(fcd_path, threshold=ttc_threshold),
        "hard_braking": evaluate_hard_braking(fcd_path),
        "arrived": arrived,
        "sim_seconds": sim_seconds,
        "throughput_veh_per_hour": throughput,
        "mean_speed": mean_speed,
        "mean_time_loss": mean_time_loss,
        "collisions": collisions,
        "teleports": teleports,
        "type_counts": type_dist["counts"],
        "type_fractions": type_dist["fractions"],
    }
