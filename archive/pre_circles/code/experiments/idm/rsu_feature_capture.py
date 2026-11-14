"""Corrected RSU feature capture (Stage 1c-2 Part A) + shared 1c-2 helpers.

Fixes the p20 recording defect at its root: the generation capture read
``getattr(env, "active_cavs", set())`` but the environment only ever exposes
``env._active_cavs`` (env/core.py), so ``cav_count_band`` and
``cav_count_upstream`` were silently constant zero. ``ramp_queue_length``
used a 15 m/s threshold that never fired at decision times on this demand;
per the 1c-2 task it is redefined as the count of ramp-edge vehicles below
5 m/s (the 30 s ramp arrival count already exists as ``ramp_30s_entries``).

Also hosts the shared Stage 1c-2 helpers used by the treatment runner and
analyzers: the treatment-blind onset finder (first step whose trailing
10 s of D-branch merge-zone health is below 0.15) and the location x phase
safety-event aggregator (hard brakes = accel <= -4.5 m/s^2, the frozen
OutcomeDefinition threshold; low-TTC pairs from the same definition),
both computed from ReplayTrace snapshots so the runner stays schema-free.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH
from counterfactual.outcomes import OutcomeDefinition, _low_ttc_pairs

POST_MERGE_EDGE = "1487606178"
RAMP_EDGE_PREFIX = "11553048"
RAMP_QUEUE_SPEED_MPS = 5.0

# Location bands on the mainline approach, by distance to the band top
# (distance = PRE_MERGE_LANE_LENGTH - lane position).
BAND_LO_M = 200.0
BAND_HI_M = 400.0

ONSET_H_THRESHOLD = 0.15
ONSET_SUSTAIN_S = 10.0


def active_cav_ids(env: Any) -> set:
    """The active CAV set the way env.core actually exposes it.

    ``env.active_cavs`` does not exist (the p20 defect); the authoritative
    attribute is ``env._active_cavs``, with ``env._prev_obs_ids`` (the last
    observation's key set) as a fallback.
    """
    cav = getattr(env, "_active_cavs", None)
    if cav is None:
        cav = getattr(env, "_prev_obs_ids", set()) or set()
    return set(cav)


def _distance_m(lane_pos: float) -> float:
    return max(0.0, PRE_MERGE_LANE_LENGTH - float(lane_pos))


def _approach_band(distance: float) -> str:
    if distance > 800.0:
        return "upstream_gt800"
    if distance > 400.0:
        return "upstream_400_800"
    if distance >= BAND_HI_M:
        return "band_400_500"
    if distance >= BAND_LO_M:
        return "band_200_400"
    return "near_merge_lt200"


def step_features_fixed(env: Any) -> Dict[str, Any]:
    """The prep-4 per-step RSU record with the CAV counts and ramp queue fixed.

    Identical to ``run_sustained_wall_v1_stage1b_prep4_observation_map.
    _step_features`` except: (a) the CAV set is read from
    ``active_cav_ids(env)`` so the per-CAV counts are real; (b)
    ``ramp_queue_length`` counts ramp-edge vehicles below 5 m/s (the 15 m/s
    version never fired at decision times). Per-step ``ramp_entries`` feeds
    the unchanged 30 s ramp-arrival feature.
    """
    import traci

    live = {str(v) for v in traci.vehicle.getIDList()}
    rows: List[Tuple[str, float]] = []
    lane_counts: Dict[str, int] = {}
    ramp_queue = 0
    cav_band = 0
    cav_upstream = 0
    active_cavs = active_cav_ids(env)
    prev_upstream = getattr(env, "_prep4_prev_upstream", set())
    prev_ramp = getattr(env, "_prep4_prev_ramp", set())
    now_upstream: set = set()
    now_ramp: set = set()
    for vid in sorted(live):
        try:
            road = str(traci.vehicle.getRoadID(vid))
            lane = str(traci.vehicle.getLaneID(vid))
            lane_pos = float(traci.vehicle.getLanePosition(vid))
            speed = float(traci.vehicle.getSpeed(vid))
            is_cav = vid in active_cavs
        except Exception:
            continue
        if road == PRE_MERGE_EDGE:
            distance = _distance_m(lane_pos)
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            if distance > 400.0:
                rows.append(("upstream", speed))
                now_upstream.add(vid)
                if is_cav:
                    cav_upstream += 1
            elif distance >= 200.0:
                rows.append(("approach", speed))
                if is_cav:
                    cav_band += 1
            else:
                rows.append(("near_merge", speed))
        elif road == MERGE_EDGE or road.startswith(":"):
            rows.append(("merge_edge", speed))
        elif road == POST_MERGE_EDGE:
            rows.append(("post_merge", speed))
        elif road.startswith(RAMP_EDGE_PREFIX):
            rows.append(("ramp", speed))
            if speed < RAMP_QUEUE_SPEED_MPS:
                ramp_queue += 1
            now_ramp.add(vid)
    env._prep4_prev_upstream = now_upstream
    env._prep4_prev_ramp = now_ramp
    on_merge = set()
    for vid in sorted(live):
        try:
            if str(traci.vehicle.getRoadID(vid)) == MERGE_EDGE:
                on_merge.add(vid)
        except Exception:
            pass
    prev_on_merge = getattr(env, "_prep4_prev_merge", set())
    discharge = len(prev_on_merge - on_merge)
    env._prep4_prev_merge = on_merge

    from experiments.idm.run_sustained_wall_v1_stage1b_prep4_observation_map import (
        _region_metrics,
    )

    regions = _region_metrics(rows)
    mainline_counts = list(lane_counts.values())
    imbalance = (max(mainline_counts) - min(mainline_counts)) if mainline_counts else 0.0
    return {
        "regions": regions,
        "mainline_lane_counts": dict(lane_counts),
        "mainline_lane_imbalance": float(imbalance),
        "ramp_queue_length": float(ramp_queue),
        "cav_count_band": float(cav_band),
        "cav_count_upstream": float(cav_upstream),
        "upstream_entries": float(len(now_upstream - prev_upstream)),
        "ramp_entries": float(len(now_ramp - prev_ramp)),
        "merge_discharge": float(discharge),
    }


def per_lane_cav_distances(env: Any) -> Dict[str, list]:
    """Distances-to-band-top (m) of every active CAV on the approach, by lane.

    Distance = PRE_MERGE_LANE_LENGTH - lane position (0 at the downstream
    end of the pre-merge edge). Feeds the unclipped per-lane geometry.
    """
    import traci

    active = active_cav_ids(env)
    per_lane: Dict[str, list] = {}
    for vid in sorted(active):
        try:
            if str(traci.vehicle.getRoadID(vid)) != PRE_MERGE_EDGE:
                continue
            lane = str(traci.vehicle.getLaneID(vid))
            lane_pos = float(traci.vehicle.getLanePosition(vid))
        except Exception:
            continue
        per_lane.setdefault(lane, []).append(_distance_m(lane_pos))
    return per_lane


def first_onset_step(
    h_series: List[float],
    *,
    threshold: float = ONSET_H_THRESHOLD,
    sustain_s: float = ONSET_SUSTAIN_S,
    max_step: int = 900,
) -> Optional[int]:
    """Treatment-blind onset: the first step t <= max_step whose trailing
    ``sustain_s`` seconds of h are all below ``threshold``.

    ``h_series[k]`` is the health after step ``k`` (index == step). Returns
    the step that completes the first qualifying run, or None.
    """
    sustain = int(round(sustain_s))
    run = 0
    for t in range(1, len(h_series)):
        run = run + 1 if float(h_series[t]) < threshold else 0
        if run >= sustain and t <= max_step:
            return t
        if t > max_step:
            return None
    return None


def _vehicle_location(row: Tuple) -> str:
    """Location bucket for a snapshot vehicle row (capture_step layout)."""
    road = str(row[1])
    if road == PRE_MERGE_EDGE:
        return _approach_band(_distance_m(float(row[3])))
    if road == MERGE_EDGE or road.startswith(":"):
        return "merge_edge"
    if road == POST_MERGE_EDGE:
        return "post_merge"
    if road.startswith(RAMP_EDGE_PREFIX):
        return "ramp"
    return "other"


def _phase(step: int, engage_step: int, release_step: Optional[int]) -> str:
    if step <= engage_step:
        return "pre_engagement"
    if release_step is None or step < release_step:
        return "hold"
    if step <= release_step + 5:
        return "taper"
    return "post_release"


def safety_events_by_location(
    snapshots,
    *,
    engage_step: int,
    release_step: Optional[int],
    definition: Optional[OutcomeDefinition] = None,
) -> Dict[str, Any]:
    """Hard-brake vehicle-steps and low-TTC pair-steps by location x phase.

    Hard brake: accel <= OutcomeDefinition.hard_brake_threshold_mps2
    (-4.5), located at the braking vehicle. Low TTC: pairs from the frozen
    definition, located at the follower. Phases: pre_engagement (step <=
    engage), hold (up to the taper start), taper (5 s), post_release.
    """
    definition = definition or OutcomeDefinition()
    events: Dict[str, Any] = {}
    for snapshot in snapshots:
        phase = _phase(int(snapshot.relative_step), int(engage_step), release_step)
        low_ttc = _low_ttc_pairs(snapshot.vehicles, definition)
        follower_ids = {str(pair[0]) for pair in low_ttc}
        by_id = {str(row[0]): row for row in snapshot.vehicles}
        for row in snapshot.vehicles:
            if float(row[5]) <= definition.hard_brake_threshold_mps2:
                location = _vehicle_location(row)
                bucket = events.setdefault("hard_brake", {}).setdefault(
                    location, {}
                ).setdefault(phase, 0)
                events["hard_brake"][location][phase] = bucket + 1
        for follower_id in follower_ids:
            row = by_id.get(follower_id)
            if row is None:
                continue
            location = _vehicle_location(row)
            bucket = events.setdefault("low_ttc", {}).setdefault(
                location, {}
            ).setdefault(phase, 0)
            events["low_ttc"][location][phase] = bucket + 1
    return events
