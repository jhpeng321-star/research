"""Frozen definitions for the Stage 1a sustained-wall effect-existence pilot.

Cell: 2200 veh/h mainline / 400 veh/h ramp, CAV penetration 0.20, long IDM
scenario, seeds 6001--6030. Horizon 1080 s with the verified natural-clearance
terminal rule (docs/IDM_V5_7_1_GATE_A_TERMINAL_RECONCILIATION.md).

The wall definition follows the estimand reset (docs/ESTIMAND_RESET_2026-09-02.md):
a sustained health-gated moving wall with membership by band passage (legacy
actuator semantics), engaged at the first metastable landmark and released on
health recovery, maximum duration, or the demand-end cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from idm.contract import (
    CAV_PENETRATION,
    SCENARIO_ROOT,
    SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS,
    scenario_config,
)

# --- Cell ------------------------------------------------------------------
MAINLINE_DEMAND_VPH = 2200
RAMP_DEMAND_VPH = 400
CAV_PENETRATION = CAV_PENETRATION
SCENARIO_FILE = scenario_config(MAINLINE_DEMAND_VPH).name
SCENARIO_ROOT_PATH = SCENARIO_ROOT
SEEDS = SUSTAINED_WALL_V1_STAGE1A_DEVELOPMENT_SEEDS
STEP_LENGTH_S = 1.0  # controller step; SUMO tick is 0.5 s
HORIZON_S = 1080.0

# --- Decision landmark -----------------------------------------------------
LANDMARK_START_S = 200.0
LANDMARK_LAST_S = 700.0
LANDMARK_TRIGGER_H = 0.45
LANDMARK_DECAY_LOOKBACK_S = 30.0

# --- Wall regime -----------------------------------------------------------
WALL_PARAMS: Dict[str, float] = {
    "d_lo": 200.0,
    "d_hi": 400.0,
    "wall_speed": 10.0,
    "wall_tau": 3.0,
    "a_max_dec": 0.3,
}
RECOVERY_H = 0.45
RECOVERY_PERSIST_S = 20.0
MAX_WALL_DURATION_S = 180.0
WALL_CAP_S = 900.0
TAPER_DURATION_S = 5.0

# --- Arm P perturbation ----------------------------------------------------
NUDGE_SLOW_DOWN_MPS = 0.5
NUDGE_LANDMARK_DISTANCE_M = 300.0

# --- Collapse labels -------------------------------------------------------
HEALTH_COLLAPSE_H = 0.15
HEALTH_COLLAPSE_SUSTAIN_S = 60.0
HEALTH_COLLAPSE_AFTER_S = 100.0
PHYSICAL_QUEUE_SPEED_MPS = 15.0
PHYSICAL_QUEUE_COUNT = 5
PHYSICAL_MEAN_SPEED_MPS = 15.0
PHYSICAL_SUSTAIN_S = 60.0

# --- Outcome windows -------------------------------------------------------
HARD_BRAKE_ACCEL_MPS2 = -4.5
LOW_TTC_THRESHOLD_S = 1.5
RELEASE_TAIL_WINDOW_S = 60.0


@dataclass(frozen=True)
class CollapseStatistics:
    incidence: bool
    first_onset_step: Optional[int]
    total_duration_s: float
    longest_run_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def find_landmark(
    h_series: Sequence[float],
    *,
    start_s: float = LANDMARK_START_S,
    last_s: float = LANDMARK_LAST_S,
    trigger_h: float = LANDMARK_TRIGGER_H,
    lookback_s: float = LANDMARK_DECAY_LOOKBACK_S,
) -> Optional[int]:
    """First controller step t in [start, last] with h(t) <= trigger_h and
    h(t) - h(t - lookback) <= 0. None when no such step exists.

    ``h_series`` is indexed by relative controller step (each step = 1 s).
    """
    start = int(round(start_s))
    last = int(round(last_s))
    lookback = int(round(lookback_s))
    for step in range(max(start, lookback), last + 1):
        if step >= len(h_series) or step - lookback < 0:
            continue
        h_now = float(h_series[step])
        h_prev = float(h_series[step - lookback])
        if h_now <= float(trigger_h) and (h_now - h_prev) <= 0.0:
            return step
    return None


def _runs_where(predicate: Sequence[bool], after_step: int) -> list[tuple[int, int]]:
    """Consecutive True runs among steps >= after_step: (start, length) pairs.

    Runs are only recorded when they fully start at or after ``after_step``.
    """
    runs: list[tuple[int, int]] = []
    run_start: Optional[int] = None
    for index in range(len(predicate)):
        if index < after_step:
            continue
        if not predicate[index]:
            if run_start is not None:
                runs.append((run_start, index - run_start))
                run_start = None
            continue
        if run_start is None:
            run_start = index
    if run_start is not None:
        runs.append((run_start, len(predicate) - run_start))
    return runs


def health_collapse_statistics(
    h_series: Sequence[float],
    *,
    collapse_h: float = HEALTH_COLLAPSE_H,
    sustain_s: float = HEALTH_COLLAPSE_SUSTAIN_S,
    after_s: float = HEALTH_COLLAPSE_AFTER_S,
) -> CollapseStatistics:
    """Legacy health collapse label: h < collapse_h for >= sustain_s
    consecutive seconds after t = after_s.

    First onset is the start of the first qualifying run; total duration
    counts every h < collapse_h step after ``after_s`` (all runs, qualifying
    or not); longest run is the maximum run length (steps).
    """
    sustain = int(round(sustain_s))
    after = int(round(after_s))
    below = [float(value) < collapse_h for value in h_series]
    runs = _runs_where(below, after)
    qualifying = [run for run in runs if run[1] >= sustain]
    total = sum(length for _start, length in runs)
    longest = max((length for _start, length in runs), default=0)
    return CollapseStatistics(
        incidence=bool(qualifying),
        first_onset_step=(qualifying[0][0] if qualifying else None),
        total_duration_s=float(total),
        longest_run_s=float(longest),
    )


def physical_collapse_statistics(
    step_metrics: Sequence[Mapping[str, Any]],
    *,
    queue_speed_mps: float = PHYSICAL_QUEUE_SPEED_MPS,
    queue_count: int = PHYSICAL_QUEUE_COUNT,
    mean_speed_mps: float = PHYSICAL_MEAN_SPEED_MPS,
    sustain_s: float = PHYSICAL_SUSTAIN_S,
) -> CollapseStatistics:
    """v5.5 physical label: >= queue_count premerge vehicles below
    queue_speed_mps AND premerge mean speed <= mean_speed_mps for
    sustain_s consecutive steps (no after-t offset for this label, matching
    ``counterfactual.outcomes._sustained_congestion``).
    """
    sustain = int(round(sustain_s))
    congested = []
    for metric in step_metrics:
        mean = float(metric.get("pre_merge_mean_speed_mps") or float("inf"))
        n = int(metric.get("queue_count") or 0)
        congested.append(n >= queue_count and mean <= mean_speed_mps)
    runs = _runs_where(congested, 0)
    qualifying = [run for run in runs if run[1] >= sustain]
    total = sum(length for _start, length in runs)
    longest = max((length for _start, length in runs), default=0)
    return CollapseStatistics(
        incidence=bool(qualifying),
        first_onset_step=(qualifying[0][0] if qualifying else None),
        total_duration_s=float(total),
        longest_run_s=float(longest),
    )


def contract_manifest() -> Dict[str, Any]:
    return {
        "schema": "sustained_wall_v1_stage1a/1",
        "cell": {
            "scenario": SCENARIO_FILE,
            "scenario_root": str(SCENARIO_ROOT_PATH),
            "mainline_demand_vph": MAINLINE_DEMAND_VPH,
            "ramp_demand_vph": RAMP_DEMAND_VPH,
            "cav_penetration": CAV_PENETRATION,
            "seeds": list(SEEDS),
            "seed_role": "sustained_wall_v1_stage1a_development",
            "horizon_s": HORIZON_S,
        },
        "decision_landmark": {
            "start_s": LANDMARK_START_S,
            "last_s": LANDMARK_LAST_S,
            "trigger_h": LANDMARK_TRIGGER_H,
            "decay_lookback_s": LANDMARK_DECAY_LOOKBACK_S,
            "source": "env.core.TrafficShepherdMultiAgentEnv.metastable_health_detail()[0]",
            "no_landmark_action": "skip Arm W and Arm P; record seed",
        },
        "wall": dict(
            WALL_PARAMS,
            recovery_h=RECOVERY_H,
            recovery_persist_s=RECOVERY_PERSIST_S,
            max_wall_duration_s=MAX_WALL_DURATION_S,
            wall_cap_s=WALL_CAP_S,
            taper_duration_s=TAPER_DURATION_S,
        ),
        "nudge": {
            "slow_down_mps": NUDGE_SLOW_DOWN_MPS,
            "landmark_distance_m": NUDGE_LANDMARK_DISTANCE_M,
        },
        "collapse_labels": {
            "health": {
                "label": "h < 0.15 for >= 60 consecutive seconds after t = 100 s",
                "h_threshold": HEALTH_COLLAPSE_H,
                "sustain_s": HEALTH_COLLAPSE_SUSTAIN_S,
                "after_s": HEALTH_COLLAPSE_AFTER_S,
            },
            "physical": {
                "label": ">= 5 premerge vehicles < 15 m/s and premerge mean <= 15 m/s for 60 s",
                "queue_speed_mps": PHYSICAL_QUEUE_SPEED_MPS,
                "queue_count": PHYSICAL_QUEUE_COUNT,
                "mean_speed_mps": PHYSICAL_MEAN_SPEED_MPS,
                "sustain_s": PHYSICAL_SUSTAIN_S,
            },
        },
        "outcome_windows": {
            "hard_brake_accel_mps2": HARD_BRAKE_ACCEL_MPS2,
            "low_ttc_threshold_s": LOW_TTC_THRESHOLD_S,
            "release_tail_window_s": RELEASE_TAIL_WINDOW_S,
        },
    }
