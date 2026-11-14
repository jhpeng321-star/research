"""Shared bucketing helpers for the Stage 1b-prep artifact analysis.

Location and phase buckets are defined once here so Part A (Stage 1a
artifact analysis) and Part C (WHOLD arms) report identically-granular
splits.

Location buckets (from stored event fields road_id + lane_position_m):
  * pre_edge_upstream     mainline approach, distance-to-merge > 400 m
  * pre_edge_band         mainline approach, distance-to-merge in [200, 400]
  * pre_edge_near_merge   mainline approach, distance-to-merge < 200 m
  * merge_edge            on (or at the junction of) the merge edge
  * post_merge            downstream of the merge edge
  * ramp                  the ramp edge
  * other                 anything else (unexpected)

Phase buckets (from relative steps vs the wall release step, where release
is the taper-start step = regime end, matching Stage 1a windows):
  * during_wall   t_d + 1 <= step < release_step
  * release_60    release_step <= step <= release_step + 60
  * after_60      step > release_step + 60
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from config.constants import MERGE_EDGE, PRE_MERGE_EDGE, PRE_MERGE_LANE_LENGTH, RAMP_EDGE

POST_MERGE_EDGE = "1487606178"  # downstream edge of the frozen route

LOCATION_UPSTREAM = "pre_edge_upstream"
LOCATION_BAND = "pre_edge_band"
LOCATION_NEAR = "pre_edge_near_merge"
LOCATION_MERGE = "merge_edge"
LOCATION_POST = "post_merge"
LOCATION_RAMP = "ramp"
LOCATION_OTHER = "other"

LOCATION_ORDER = (
    LOCATION_UPSTREAM,
    LOCATION_BAND,
    LOCATION_NEAR,
    LOCATION_MERGE,
    LOCATION_POST,
    LOCATION_RAMP,
    LOCATION_OTHER,
)


def location_bucket(road_id: str, lane_position_m: Optional[float]) -> str:
    """Map a stored event (road_id, lane_position_m) to its location bucket."""
    road_id = str(road_id)
    if road_id == PRE_MERGE_EDGE:
        if lane_position_m is None:
            return LOCATION_OTHER
        distance = max(0.0, float(PRE_MERGE_LANE_LENGTH) - float(lane_position_m))
        if distance > 400.0:
            return LOCATION_UPSTREAM
        if distance >= 200.0:
            return LOCATION_BAND
        return LOCATION_NEAR
    if road_id == MERGE_EDGE or road_id.startswith(":"):
        return LOCATION_MERGE
    if road_id == POST_MERGE_EDGE:
        return LOCATION_POST
    if road_id == RAMP_EDGE or road_id.startswith("11553048"):
        return LOCATION_RAMP
    return LOCATION_OTHER


def location_from_road_only(road_id: str) -> str:
    """Coarser bucket available from step-level exposure marginals: only the
    road id is known (no lane position), so the approach-edge band split is
    not recoverable and the pre-edge is reported as one bucket."""
    road_id = str(road_id)
    if road_id == PRE_MERGE_EDGE:
        return "mainline_approach"
    if road_id == MERGE_EDGE or road_id.startswith(":"):
        return LOCATION_MERGE
    if road_id == POST_MERGE_EDGE:
        return LOCATION_POST
    if road_id == RAMP_EDGE or road_id.startswith("11553048"):
        return LOCATION_RAMP
    return LOCATION_OTHER


def phase_bucket(step: int, t_d: int, release_step: int) -> str:
    if step <= t_d:
        return "prefix"
    if step < release_step:
        return "during_wall"
    if step <= release_step + 60:
        return "release_60"
    return "after_60"


def series_in_window(
    series: Mapping[str, Any], metric_key: str, step_from: int, step_to: int
) -> float:
    """Sum of one per-step metric over steps in [from, to] (inclusive)."""
    values = series.get(metric_key)
    if values is None:
        return 0.0
    total = 0.0
    for index, step in enumerate(series["step"]):
        if step_from <= int(step) <= step_to and index < len(values):
            item = values[index]
            if item is not None:
                total += float(item)
    return total


def event_phase_bucket(event: Mapping[str, Any], release_step: int) -> str:
    step = int(event.get("relative_step") or 0)
    if step < release_step:
        return "during_wall"
    if step <= release_step + 60:
        return "release_60"
    return "after_60"
