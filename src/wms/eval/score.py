"""Per-state scoring: roll out every recorded arm and reduce to compact JSON-safe errors.

A *state result* is::

    {"world", "t0_s", "key", "run", "source_arm", "source_set", "own_arm", "pack", "paired",
     "native_pair_status", "wall_exposure_status", "wall_phase_at_t0", "wall_engaged",
     "assembled_rel_s", "t0_shift_s",
     "arms": {arm: {"error": None | str,
                    "car":   {"<h>": {"all"|"hdv"|"cav": {"ds_med","ds_p90","dv_med","dv_p90","match"}}},
                    "delay": {"predicted": {"<h>": {section: total_delay}}, "recorded": {...}}}}}

All errors are HDV/CAV identity-matched ``|pred - recorded|`` at fixed horizons; the delay
blocks feed the wall-vs-native ranking.  CAVs are teacher-forced by the predictor.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .metrics import SECTIONS, car_metrics, traffic_metrics
from .normalize import recorded_for

Predictor = Callable[[dict, Mapping[str, Any], float, int], list]

CAR_HORIZONS = (10.0, 30.0, 60.0, 120.0)
DELAY_HORIZONS = (60.0, 120.0)
DEFAULT_FREE_SPEED_MPS = 13.89
STRATA = ("all", "hdv", "cav")
_META_FIELDS = ("run", "source_arm", "source_set", "pack", "paired", "native_pair_status",
                "wall_exposure_status", "wall_phase_at_t0", "wall_engaged", "t0_shift_s")


@dataclass(frozen=True)
class ScoreSpec:
    """What one scoring pass computes."""

    horizons_s: tuple[float, ...] = CAR_HORIZONS
    delay_horizons_s: tuple[float, ...] = DELAY_HORIZONS
    predict_horizon_s: float = 120.0
    seed: int = 0


def car_compact(car: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict]:
    """Reduce :func:`car_metrics` output to the median/p90 scalars per stratum."""
    out: dict[str, dict] = {}
    for horizon, strata in car.items():
        block: dict[str, dict] = {}
        for stratum in STRATA:
            group = strata.get(stratum, {})
            ds = group.get("abs_delta_s_m") or {}
            dv = group.get("abs_delta_v_mps") or {}
            block[stratum] = {
                "ds_med": ds.get("median"), "ds_p90": ds.get("p90"),
                "dv_med": dv.get("median"), "dv_p90": dv.get("p90"),
                "match": group.get("matched_fraction"),
            }
        out[str(horizon)] = block
    return out


def section_delays(frames: Sequence[dict], t0_s: float, v0: float,
                   horizons: Sequence[float]) -> dict[str, dict[str, float]]:
    """Cumulative normalised delay per section over ``[t0, t0+h)`` for each horizon."""
    out: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        selected = [f for f in frames if t0_s <= float(f["time_s"]) < t0_s + horizon]
        if not selected:
            continue
        bins = traffic_metrics(selected, t0_s=t0_s, bin_s=horizon, free_speed_mps=v0)
        block: dict[str, float] = {}
        for section in SECTIONS:
            values = [bins[b]["sections"][section].get("total_delay") for b in bins]
            values = [v for v in values if v is not None and math.isfinite(float(v))]
            if values:
                block[section] = float(sum(values))
        out[str(horizon)] = block
    return out


def evaluate_state(record: dict, predictor: Predictor, spec: ScoreSpec = ScoreSpec()) -> dict:
    """Roll out and score every recorded arm of one state; one failing arm never aborts the rest."""
    t0_s = float(record["t0_s"])
    v0 = float(record.get("v0_mps", DEFAULT_FREE_SPEED_MPS))
    controllers = record.get("cav_controllers", {})
    arms: dict[str, dict] = {}
    for arm in sorted((record.get("recorded") or {}).keys()):
        try:
            predicted = predictor(record, {**controllers, "arm": arm},
                                  spec.predict_horizon_s, spec.seed)
            recorded = recorded_for(record, arm)
            car = car_compact(car_metrics(predicted, recorded, t0_s=t0_s,
                                          horizons=tuple(spec.horizons_s)))
            delay = {"predicted": section_delays(predicted, t0_s, v0, spec.delay_horizons_s),
                     "recorded": section_delays(recorded, t0_s, v0, spec.delay_horizons_s)}
            arms[arm] = {"error": None, "car": car, "delay": delay}
        except Exception as exc:  # noqa: BLE001 - recorded per arm, never swallowed silently
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    timeline = record.get("wall_timeline") or {}
    result = {"world": str(record.get("world")), "t0_s": t0_s, "key": record.get("key"),
              "own_arm": record.get("arm"), "assembled_rel_s": timeline.get("assembled_rel_s"),
              "arms": arms}
    result.update({name: record.get(name) for name in _META_FIELDS})
    return result


# --------------------------------------------------------------------------- #
# bounded parallel driver
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _init_worker(factory: Callable[[], Predictor], spec: ScoreSpec) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    _WORKER["predictor"] = factory()
    _WORKER["spec"] = spec


def _work(payload: tuple[int, dict]) -> tuple[int, dict]:
    index, record = payload
    return index, evaluate_state(record, _WORKER["predictor"], _WORKER["spec"])


def score_states(records: Iterable[dict], predictor_factory: Callable[[], Predictor],
                 spec: ScoreSpec = ScoreSpec(), *, workers: int = 1,
                 inflight: int = 4) -> list[dict]:
    """Score streamed records, preserving input order.

    ``predictor_factory`` builds one predictor per worker process (it must be picklable
    when ``workers > 1``).  At most ``inflight`` records are in flight, so memory stays
    bounded however large the state file is.
    """
    iterator: Iterator[tuple[int, dict]] = enumerate(records)
    if workers <= 1:
        predictor = predictor_factory()
        return [evaluate_state(record, predictor, spec) for _, record in iterator]
    results: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(predictor_factory, spec)) as pool:
        pending: dict[Any, int] = {}
        exhausted = False

        def submit_next() -> None:
            nonlocal exhausted
            if exhausted:
                return
            try:
                index, record = next(iterator)
            except StopIteration:
                exhausted = True
                return
            pending[pool.submit(_work, (index, record))] = index

        for _ in range(max(1, inflight)):
            submit_next()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index, result = future.result()
                results[index] = result
                del pending[future]
                submit_next()
    return [results[i] for i in sorted(results)]
