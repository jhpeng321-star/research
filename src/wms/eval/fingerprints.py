"""Failure fingerprints: *why* a rollout's HDV error is what it is.

A fingerprint separates the mechanisms that produce a given ``|ds|`` so that two
checkpoints with the same headline error can still be told apart:

* **residual size** - how much the gated residual moves the physics acceleration
  (applied correction, |mean|, p95, signed mean, gate) over live learned-HDV steps;
* **lane-change rate** per HDV step, predicted vs recorded;
* **wrong-lane vs wrong-speed split** of the horizon ``|ds|``: error mass carried by
  vehicles in the wrong lane versus vehicles in the right lane;
* **signed speed bias** (pred - recorded, m/s; > 0 = model too fast) at the horizon and
  averaged over the window;
* **jam vs free-flow slices**, chosen from the t0 snapshot only (median HDV speed below
  :data:`JAM_SPEED_MPS`) so the slice is observable at decision time.

:func:`best_vs_last` and :func:`stage_table` score a series of checkpoints (per stage or
per epoch) against the same states to show whether the last checkpoint is the best one.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .metrics import frame_at, present_by_id
from .normalize import DT_S, initial_frame, recorded_for
from .report import blowup_rate, horizon_summary
from .score import Predictor, ScoreSpec, score_states
from .stats import quantile
from .trace import trace_steps

JAM_SPEED_MPS = 8.0
SPEED_SERIES_STEP_S = 10.0
SLICES = ("all", "jam", "free_flow")


def state_slice(record: Mapping, jam_speed_mps: float = JAM_SPEED_MPS) -> str:
    """``jam`` iff the median t0 HDV speed is below ``jam_speed_mps``, else ``free_flow``."""
    speeds = [float(v["v"]) for v in initial_frame(dict(record))["vehicles"]
              if v.get("present", True) and not v.get("is_cav") and int(v.get("lane", -1)) >= 0]
    if not speeds:
        return "free_flow"
    return "jam" if statistics.median(speeds) < jam_speed_mps else "free_flow"


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
def residual_stats(steps: Sequence[Mapping[str, np.ndarray]]) -> dict:
    """Residual magnitude statistics over live learned-HDV slots of traced steps."""
    corrections, physics, accels, gates = [], [], [], []
    for step in steps:
        live = np.asarray(step["present"], dtype=bool) & ~np.asarray(step["is_cav"], dtype=bool)
        if not live.any():
            continue
        corrections.append(step["correction"][live])
        physics.append(step["physics"][live])
        accels.append(step["accel"][live])
        gates.append(float(np.mean(step["gate"])))
    if not corrections:
        return {"n_hdv_steps": 0, "mean_abs_correction": None, "p95_abs_correction": None,
                "mean_correction": None, "mean_abs_physics": None, "mean_abs_accel": None,
                "mean_gate": None}
    corr, phys, acc = (np.concatenate(v) for v in (corrections, physics, accels))
    return {"n_hdv_steps": int(corr.size),
            "mean_abs_correction": float(np.abs(corr).mean()),
            "p95_abs_correction": float(np.percentile(np.abs(corr), 95)),
            "mean_correction": float(corr.mean()),
            "mean_abs_physics": float(np.abs(phys).mean()),
            "mean_abs_accel": float(np.abs(acc).mean()),
            "mean_gate": float(np.mean(gates))}


def lane_change_counts(frames: Sequence[Mapping]) -> dict:
    """Lane changes and eligible steps of HDVs present (on a modelled lane) in consecutive frames."""
    changes = steps = 0
    previous: dict[str, int] = {}
    for frame in frames:
        current = {vid: int(row.get("lane", -1)) for vid, row in present_by_id(dict(frame)).items()
                   if not row.get("is_cav") and int(row.get("lane", -1)) >= 0}
        for vid, lane in current.items():
            if vid in previous:
                steps += 1
                changes += int(previous[vid] != lane)
        previous = current
    return {"changes": changes, "steps": steps}


def _matched_hdv(pred_frame: Mapping | None, rec_frame: Mapping | None) -> list[tuple[dict, dict]]:
    if pred_frame is None or rec_frame is None:
        return []
    p, r = present_by_id(dict(pred_frame)), present_by_id(dict(rec_frame))
    return [(p[vid], r[vid]) for vid in sorted(set(p) & set(r)) if not r[vid].get("is_cav")]


def horizon_split(pred_frames: Sequence[Mapping], rec_frames: Sequence[Mapping], t0_s: float,
                  horizon_s: float) -> dict:
    """Wrong-lane vs wrong-speed split of the HDV ``|ds|`` at one horizon."""
    pairs = _matched_hdv(frame_at(list(pred_frames), t0_s + horizon_s),
                         frame_at(list(rec_frames), t0_s + horizon_s))
    wrong_lane = [abs(float(p["s"]) - float(r["s"])) for p, r in pairs
                  if p.get("lane") != r.get("lane")]
    right_lane = [abs(float(p["s"]) - float(r["s"])) for p, r in pairs
                  if p.get("lane") == r.get("lane")]
    total = sum(wrong_lane) + sum(right_lane)
    return {"n": len(pairs), "n_wrong_lane": len(wrong_lane),
            "wrong_lane_share": (sum(wrong_lane) / total) if total else None,
            "wrong_speed_share": (sum(right_lane) / total) if total else None,
            "ds_median_wrong_lane": quantile(wrong_lane, 0.5),
            "ds_median_right_lane": quantile(right_lane, 0.5)}


def speed_bias(pred_frames: Sequence[Mapping], rec_frames: Sequence[Mapping], t0_s: float,
               horizon_s: float) -> dict:
    """Signed mean speed error (pred - recorded) at the horizon, over the window and per 10 s."""
    def mean_error(t: float) -> tuple[float, int] | None:
        pairs = _matched_hdv(frame_at(list(pred_frames), t), frame_at(list(rec_frames), t))
        if not pairs:
            return None
        return sum(float(p["v"]) - float(r["v"]) for p, r in pairs), len(pairs)

    n_ticks = int(round(horizon_s / DT_S))
    total = count = 0
    for tick in range(1, n_ticks + 1):
        got = mean_error(t0_s + tick * DT_S)
        if got:
            total, count = total + got[0], count + got[1]
    at_horizon = mean_error(t0_s + horizon_s)
    series = {}
    t = SPEED_SERIES_STEP_S
    while t <= horizon_s + 1e-9:
        got = mean_error(t0_s + t)
        if got:
            series[f"{t:g}"] = got[0] / got[1]
        t += SPEED_SERIES_STEP_S
    return {"at_horizon": (at_horizon[0] / at_horizon[1]) if at_horizon else None,
            "window_mean": (total / count) if count else None,
            "series": series}


# --------------------------------------------------------------------------- #
# per state and aggregate
# --------------------------------------------------------------------------- #
def state_fingerprint(record: dict, predictor: Any, arm: str, *, horizon_s: float = 60.0,
                      seed: int = 0, jam_speed_mps: float = JAM_SPEED_MPS) -> dict:
    """Roll out one arm of ``record`` with step tracing and compute its fingerprint.

    ``predictor`` is a :class:`~wms.eval.predictor.WmsPredictor` (any object with
    ``model`` and ``predict_states`` works).
    """
    t0_s = float(record["t0_s"])
    controllers = {**record.get("cav_controllers", {}), "arm": arm}
    with trace_steps(predictor.model) as steps:
        predicted = predictor.predict_states(record, controllers, horizon_s, seed)
    recorded = recorded_for(record, arm)
    window_rec = [f for f in recorded if t0_s - 1e-9 <= float(f["time_s"]) <= t0_s + horizon_s + 1e-9]
    return {
        "world": str(record.get("world")), "t0_s": t0_s, "key": record.get("key"), "arm": arm,
        "slice": state_slice(record, jam_speed_mps),
        "residual": residual_stats(steps),
        "lane_changes": {"predicted": lane_change_counts(predicted),
                         "recorded": lane_change_counts(window_rec)},
        "split": horizon_split(predicted, recorded, t0_s, horizon_s),
        "speed_bias": speed_bias(predicted, recorded, t0_s, horizon_s),
    }


def _weighted_rate(blocks: Iterable[Mapping]) -> float | None:
    changes = sum(b["changes"] for b in blocks)
    steps = sum(b["steps"] for b in blocks)
    return (changes / steps) if steps else None


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    return (sum(xs) / len(xs)) if xs else None


def aggregate(fingerprints: Sequence[Mapping]) -> dict:
    """Pool per-state fingerprints into one summary for a set of states."""
    res = [f["residual"] for f in fingerprints if f["residual"]["n_hdv_steps"]]
    weights = [r["n_hdv_steps"] for r in res]

    def pooled(field: str) -> float | None:
        if not res or not sum(weights):
            return None
        return sum(r[field] * w for r, w in zip(res, weights)) / sum(weights)

    pred_rate = _weighted_rate(f["lane_changes"]["predicted"] for f in fingerprints)
    rec_rate = _weighted_rate(f["lane_changes"]["recorded"] for f in fingerprints)
    split_n = [f["split"] for f in fingerprints if f["split"]["n"]]
    return {
        "n_states": len(fingerprints),
        "residual": {"mean_abs_correction": pooled("mean_abs_correction"),
                     "mean_correction": pooled("mean_correction"),
                     "mean_abs_physics": pooled("mean_abs_physics"),
                     "mean_gate": _mean(r["mean_gate"] for r in res),
                     "p95_abs_correction_max": max((r["p95_abs_correction"] for r in res),
                                                   default=None)},
        "lane_change_rate": {"predicted": pred_rate, "recorded": rec_rate,
                             "ratio": (pred_rate / rec_rate) if pred_rate is not None
                             and rec_rate else None},
        "split": {"wrong_lane_share": _mean(s["wrong_lane_share"] for s in split_n),
                  "wrong_speed_share": _mean(s["wrong_speed_share"] for s in split_n),
                  "n_wrong_lane": sum(s["n_wrong_lane"] for s in split_n),
                  "n_matched": sum(s["n"] for s in split_n)},
        "speed_bias": {"at_horizon": _mean(f["speed_bias"]["at_horizon"] for f in fingerprints),
                       "window_mean": _mean(f["speed_bias"]["window_mean"] for f in fingerprints)},
    }


def sliced_fingerprint(fingerprints: Sequence[Mapping]) -> dict[str, dict]:
    """:func:`aggregate` for all states and for the jam / free-flow slices."""
    out = {"all": aggregate(fingerprints)}
    for name in SLICES[1:]:
        out[name] = aggregate([f for f in fingerprints if f["slice"] == name])
    return out


# --------------------------------------------------------------------------- #
# checkpoint series
# --------------------------------------------------------------------------- #
def stage_table(results_by_checkpoint: Mapping[str, Sequence[dict]], arm: str,
                horizons: Sequence[float] = (10.0, 30.0, 60.0)) -> dict[str, dict]:
    """Median HDV ``|ds|`` per horizon, and blow-up rate, for every checkpoint of a series."""
    return {name: {"ds_median": {f"{h:g}": horizon_summary(states, arm, h)["median"]
                                 for h in horizons},
                   "blowup": blowup_rate(states, arm)["rate"],
                   "n_states": len(states)}
            for name, states in results_by_checkpoint.items()}


def best_vs_last(results_by_checkpoint: Mapping[str, Sequence[dict]], arm: str,
                 horizon_s: float = 60.0) -> dict:
    """Which checkpoint is best at ``horizon_s`` and how much worse the last one is.

    The mapping order is the training order; the last entry is the "last" checkpoint.
    """
    medians = {name: horizon_summary(states, arm, horizon_s)["median"]
               for name, states in results_by_checkpoint.items()}
    scored = {name: m for name, m in medians.items() if m is not None}
    if not scored:
        return {"horizon_s": horizon_s, "medians": medians, "best": None, "last": None,
                "last_minus_best_m": None, "last_is_best": None}
    best = min(scored, key=scored.__getitem__)
    last = list(results_by_checkpoint)[-1]
    gap = None if last not in scored else scored[last] - scored[best]
    return {"horizon_s": horizon_s, "medians": medians, "best": best, "last": last,
            "last_minus_best_m": gap, "last_is_best": best == last}


def score_checkpoint_series(factories: Mapping[str, Callable[[], Predictor]],
                            records: Sequence[dict], spec: ScoreSpec = ScoreSpec(), *,
                            workers: int = 1) -> dict[str, list[dict]]:
    """Score every checkpoint of a series on the same records (training order preserved).

    ``factories`` maps a checkpoint name to a picklable zero-argument factory returning a
    ``predict_states`` callable, e.g. ``lambda: WmsPredictor.from_checkpoint(p).predict_states``
    wrapped as a top-level function.
    """
    return {name: score_states(records, factory, spec, workers=workers)
            for name, factory in factories.items()}
