"""Measured-interaction grouping (PL01_PLAN decision loop step 2).

Interaction edges come from MEASURED trajectory deviation between the step-1
predictions (current plan with CAV i -> action a, others held) and the
current plan, on ``trajectories_tracked`` (contract section 4.3). For CAV j:

- deviation(m, k) = max over t of the |position| (and |speed|) difference,
  NaN-aware: both NaN at t -> no deviation; exactly one NaN at t (one plan
  already exited the vehicle) -> deviation +inf (a real effect on j);
- the per-cell max-over-time deviation is summarized by its MEDIAN over ALL
  M*K cells (paired cells, consistent with the agreement/margin rule).

w_ij = max over i's step-1 alternatives of j's deviation; w = max(w_ij, w_ji).
An edge exists if the position deviation exceeds ``interaction_pos_m`` OR the
speed deviation exceeds ``interaction_speed_mps``. Groups are connected
components; components above ``group_cap`` shed their weakest edge (smallest
threshold ratio, ties by sorted ids) until every component fits.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from .config import PlannerConfig
from .interface import Prediction

# trajectory columns: (t_s, s_m, lane_id, speed_mps)
_POS_COL, _SPEED_COL = 1, 3


@dataclass(frozen=True)
class InteractionEdge:
    """One measured interaction edge between two CAVs (ids sorted)."""

    a: str
    b: str
    pos_dev_m: float
    speed_dev_mps: float
    weight: float  # dimensionless: max(pos/thr_pos, speed/thr_speed)


def trajectory_deviation(plan_arr: np.ndarray,
                         ref_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware per-cell deviation of one vehicle: (position, speed) [M, K].

    ``plan_arr``/``ref_arr`` are float [M, K, T, 4] (columns per contract
    section 4.3). At each t: both finite -> |diff|; both NaN -> 0; exactly
    one finite -> +inf. All-time max then reduces over t.
    """
    plan_arr = np.asarray(plan_arr, dtype=float)
    ref_arr = np.asarray(ref_arr, dtype=float)
    deviations = []
    for col in (_POS_COL, _SPEED_COL):
        a = plan_arr[..., col]
        b = ref_arr[..., col]
        fin_a = np.isfinite(a)
        fin_b = np.isfinite(b)
        both = fin_a & fin_b
        one_sided = fin_a ^ fin_b
        diff = np.where(both, np.abs(a - b), 0.0)
        diff = np.where(one_sided, np.inf, diff)
        deviations.append(diff.max(axis=-1))
    return deviations[0], deviations[1]


def influences(
    alternatives: Mapping[str, Mapping[str, Prediction]],
    reference: Prediction,
    cav_ids: Sequence[str],
) -> Tuple[Tuple[str, str, float, float], ...]:
    """Symmetric pairwise deviations: (lo, hi, pos_dev, speed_dev), lo < hi.

    ``alternatives[cav][action]`` are the step-1 predictions. w_ij = max over
    i's alternatives of the median-over-cells max-over-time deviation of j;
    the returned pair value is max over both directions.
    """
    ids = sorted(cav_ids)
    out = []
    for i, lo in enumerate(ids):
        for hi in ids[i + 1:]:
            pos = speed = 0.0
            for source, target in ((lo, hi), (hi, lo)):
                for prediction in alternatives[source].values():
                    tracked = prediction.trajectories_tracked[target]
                    d_pos, d_speed = trajectory_deviation(
                        tracked, reference.trajectories_tracked[target])
                    pos = max(pos, float(np.median(d_pos)))
                    speed = max(speed, float(np.median(d_speed)))
            out.append((lo, hi, pos, speed))
    return tuple(out)


def build_edges(
    cav_ids: Sequence[str],
    pairwise: Sequence[Tuple[str, str, float, float]],
    config: PlannerConfig,
) -> Tuple[InteractionEdge, ...]:
    """Edges where position or speed deviation EXCEEDS its threshold."""
    nodes = set(cav_ids)
    edges = []
    for lo, hi, pos, speed in pairwise:
        if lo not in nodes or hi not in nodes:
            continue
        if pos > config.interaction_pos_m or speed > config.interaction_speed_mps:
            weight = max(pos / config.interaction_pos_m,
                         speed / config.interaction_speed_mps)
            edges.append(InteractionEdge(lo, hi, pos, speed, weight))
    return tuple(sorted(edges, key=lambda e: (e.a, e.b)))


def components(cav_ids: Sequence[str],
               edges: Sequence[InteractionEdge]) -> Tuple[Tuple[str, ...], ...]:
    """Connected components: each a sorted id tuple; list sorted for determinism."""
    adjacency = {cav: set() for cav in cav_ids}
    for edge in edges:
        adjacency.setdefault(edge.a, set()).add(edge.b)
        adjacency.setdefault(edge.b, set()).add(edge.a)
    seen = set()
    out = []
    for cav in sorted(adjacency):
        if cav in seen:
            continue
        stack, comp = [cav], set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adjacency[node] - comp)
        seen |= comp
        out.append(tuple(sorted(comp)))
    return tuple(sorted(out))


def cap_split(cav_ids: Sequence[str],
              edges: Sequence[InteractionEdge],
              cap: int) -> Tuple[Tuple[str, ...], ...]:
    """Components after repeatedly removing the weakest edge of any oversized
    component (deterministic: smallest threshold ratio, ties by sorted ids,
    smallest component id tuple processed first)."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    remaining = list(edges)
    while True:
        comps = components(cav_ids, remaining)
        oversized = sorted(comp for comp in comps if len(comp) > cap)
        if not oversized:
            return tuple(comps)
        target = set(oversized[0])
        inside = [e for e in remaining if e.a in target and e.b in target]
        weakest = min(inside, key=lambda e: (e.weight, e.a, e.b))
        remaining.remove(weakest)
