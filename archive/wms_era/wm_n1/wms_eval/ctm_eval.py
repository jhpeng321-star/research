"""Evaluate the standalone CTM on the shifted C1 v2 sets and the unshifted sets.

For every state the CTM is initialised from the recorded t0 snapshot (hard cell
counts), driven by the t0 entrance counters, and rolled to the recorded horizon;
its per-section 30 s-bin traffic metrics are compared with the recorded arm using
the identical ``wms_eval.metrics`` conventions the per-car models use.
Per-car comparison reuses the existing ``states__<model>__<set>.json`` files
(no rollout is re-run), so the CTM and the per-car models are scored on the same
bins, sections, metrics and recorded reference.  All reads are streamed one
window/one file at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from . import ctm
from .c1v2 import iter_windows
from .metrics import traffic_metrics

SECTIONS = ("upstream", "merge", "downstream")
METRICS = ("mean_speed", "density", "stopped_vehicles", "merge_queue_length",
           "merge_flow", "total_delay")
BLOWUP_DS_M = 100.0
BLOWUP_HORIZON = "120.0"


def _baseline_arm(record: dict) -> str:
    recorded = record.get("recorded")
    keys = list(recorded.keys()) if isinstance(recorded, dict) else []
    if "native" in keys:
        return "native"
    own = str(record.get("arm", ""))
    if own in keys:
        return own
    return keys[0] if keys else "native"


def arm_for(record: dict, *, unseen: bool) -> str:
    recorded = record.get("recorded")
    if unseen and isinstance(recorded, dict) and "pi" in recorded:
        return "pi"
    return _baseline_arm(record)


def init_counts(record: dict, layout: ctm.CellLayout) -> np.ndarray:
    vehicles = record["initial_state"]["vehicles"]
    s = np.array([float(v.get("s", 0.0)) for v in vehicles])
    lane = np.array([int(v.get("lane", -1)) for v in vehicles])
    present = np.array([bool(v.get("present", True)) for v in vehicles])
    return ctm.hard_bin_counts(layout, s, lane, present=present)


def evaluate_record(record: dict, params: ctm.CTMParams, layout: ctm.CellLayout, *,
                    unseen: bool, horizon_cap: float, bin_s: float) -> Optional[dict]:
    arm = arm_for(record, unseen=unseen)
    recorded = record.get("recorded")
    if not isinstance(recorded, dict) or arm not in recorded:
        return None
    frames = recorded[arm]
    if not frames:
        return None
    v0 = float(record.get("v0_mps", 27.78))
    t0_s = float(record.get("t0_s", frames[0].get("time_s", 0.0)))
    horizon_steps = record.get("recorded_horizon_steps", {}).get(arm)
    if horizon_steps is None:
        horizon_steps = int(round(horizon_cap / params.dt_s))
    steps = int(min(int(round(horizon_cap / params.dt_s)), int(horizon_steps)))
    counter = record.get("counter")
    if counter is None:
        counter = np.zeros((2, 30))
    counts0 = init_counts(record, layout)
    run = ctm.run_numpy(counts0, layout, params, counter, steps=steps, t0_s=t0_s)
    ctm_traffic = ctm.ctm_traffic_metrics(run, bin_s=bin_s, free_speed_mps=v0)
    ground = traffic_metrics(frames, t0_s=t0_s, bin_s=bin_s, free_speed_mps=v0)
    errors: dict[str, dict[str, dict[str, float]]] = {}
    for section in SECTIONS:
        errors[section] = {}
        for metric in METRICS:
            bins: dict[str, float] = {}
            for key in sorted(set(ctm_traffic) & set(ground), key=lambda k: int(k)):
                pred = ctm_traffic[key]["sections"][section].get(metric)
                rec = ground[key]["sections"][section].get(metric)
                if pred is None or rec is None:
                    continue
                bins[key] = abs(float(pred) - float(rec))
            errors[section][metric] = bins
    return {"world": record.get("world"), "t0_s": t0_s, "arm": arm,
            "n_steps": steps, "ctm": errors,
            "abs": _merge_abs(ctm_traffic, ground)}


def _merge_abs(ctm_traffic: dict, ground: dict) -> dict:
    """Absolute merge-section per-bin metrics for CTM and recorded truth."""
    out = {"ctm": {}, "gt": {}}
    for key in sorted(set(ctm_traffic) & set(ground), key=lambda k: int(k)):
        out["ctm"][key] = {m: ctm_traffic[key]["sections"]["merge"].get(m)
                           for m in ("mean_speed", "merge_queue_length", "total_delay")}
        out["gt"][key] = {m: ground[key]["sections"]["merge"].get(m)
                          for m in ("mean_speed", "merge_queue_length", "total_delay")}
    return out


def evaluate_set(path: str | Path, params: ctm.CTMParams, layout: ctm.CellLayout, *,
                 unseen: bool, horizon_cap: float, bin_s: float = 30.0,
                 limit: Optional[int] = None) -> list[dict]:
    out = []
    for record in iter_windows(path, limit=limit):
        try:
            result = evaluate_record(record, params, layout, unseen=unseen,
                                     horizon_cap=horizon_cap, bin_s=bin_s)
        except Exception as exc:  # one bad window must not abort the set
            out.append({"world": record.get("world"), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result is not None:
            out.append(result)
    return out


# --------------------------------------------------------------------------- #
# per-car comparison (existing per-state result files)
# --------------------------------------------------------------------------- #
def load_per_car(path: str | Path) -> list[dict]:
    """Per-state results in file order (one entry per evaluated episode).

    The ``seen`` sets carry ~6 policy episodes per world at the same ``t0``;
    keying by ``world@t0`` would silently collapse them, so the full ordered list
    is returned instead.  ``wms_eval.c1v2`` writes results in input-record order,
    so entry ``i`` corresponds to the ``i``-th state of the matching state set.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("states", []))


def per_car_error(state: dict, section: str, metric: str, bin_idx: int) -> Optional[float]:
    block = state.get("arms", {})
    # prefer the pi arm on unseen, else native; callers pass the scored arm
    for arm in ("pi", "native"):
        entry = block.get(arm)
        if not entry or entry.get("error"):
            continue
        value = (entry.get("traffic", {}).get(section, {})
                 .get(metric, {}).get(str(bin_idx)))
        if value is not None:
            return float(value)
    return None


def per_car_error_arm(state: dict, arm: str, section: str, metric: str,
                      bin_idx: int) -> Optional[float]:
    entry = state.get("arms", {}).get(arm)
    if not entry or entry.get("error"):
        return None
    value = entry.get("traffic", {}).get(section, {}).get(metric, {}).get(str(bin_idx))
    return None if value is None else float(value)


def per_car_car_field(state: dict, arm: str, horizon: str, stratum: str,
                      field: str) -> Optional[float]:
    entry = state.get("arms", {}).get(arm)
    if not entry or entry.get("error"):
        return None
    return entry.get("car", {}).get(horizon, {}).get(stratum, {}).get(field)


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(clean)) if clean else None


def median_of_errors(states: Sequence[dict], section: str, metric: str,
                     bin_idx: int) -> Optional[float]:
    return median(state.get("ctm", {}).get(section, {}).get(metric, {}).get(str(bin_idx))
                  for state in states if "error" not in state)


__all__ = ["SECTIONS", "METRICS", "BLOWUP_DS_M", "BLOWUP_HORIZON", "arm_for",
           "init_counts", "evaluate_record", "evaluate_set", "load_per_car",
           "per_car_error", "per_car_error_arm", "per_car_car_field", "median",
           "median_of_errors"]
