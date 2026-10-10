"""Reference predictors and the predictor loader.

A predictor is any callable ``predict(state, controllers, horizon_s, seed) -> frames``.
Besides a loaded world-model checkpoint (:class:`~wms.eval.predictor.WmsPredictor`) the
package offers the trivial ``keep_speed`` reference (constant speed, no lane changes) and
its CAV-teacher-forced variant, so baselines see the identical CAV motion as the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from .errors import DataError, EvalError
from .normalize import DT_S, initial_frame, recorded_for
from .predictor import RolloutFlags, WmsPredictor
from .rollout_inputs import nearest_frame

Predict = Callable[[dict, Mapping, float, int], list]
MODEL_SPECS = ("keep_speed", "keep_speed_tf", "wms")


def keep_speed(state: dict, _controllers: Mapping, horizon_s: float, _seed: int) -> list[dict]:
    """Every vehicle keeps its t0 speed and lane (constant-velocity reference)."""
    initial = initial_frame(state)
    dt = float(state.get("dt_s", DT_S))
    out = []
    for step in range(int(round(float(horizon_s) / dt)) + 1):
        t = initial["time_s"] + step * dt
        rows = [dict(item, s=item["s"] + item["v"] * step * dt, time_s=t)
                for item in initial["vehicles"]]
        out.append({"time_s": t, "vehicles": rows})
    return out


def teacher_force_cavs(frames: list[dict], state: dict, arm: str) -> list[dict]:
    """Replay the recorded-arm CAV trajectory on top of a baseline rollout.

    A CAV on a junction lane (``lane < 0``) or missing from the recorded step is dropped,
    and drops are permanent (cumulative mask), matching the world-model teacher.
    """
    if not frames:
        return frames
    init = initial_frame(state)
    cav_ids = [str(v["id"]) for v in init["vehicles"]
               if v.get("present", True) and v.get("is_cav") and int(v.get("lane", -1)) >= 0]
    if not cav_ids:
        return frames
    try:
        recorded = recorded_for(state, arm)
    except DataError:
        return frames
    t0 = float(frames[0]["time_s"])
    by_round = {int(round((float(f["time_s"]) - t0) / DT_S)): f for f in recorded}
    initial = {str(v["id"]): v for v in init["vehicles"]}
    live = {vid: True for vid in cav_ids}
    out = []
    for index, frame in enumerate(frames):
        cav_rows = []
        if index == 0:
            for vid in cav_ids:
                row = initial.get(vid, {})
                cav_rows.append({"id": vid, "s": float(row.get("s", 0.0)),
                                 "lane": int(row.get("lane", -1)), "v": float(row.get("v", 0.0)),
                                 "present": True, "is_cav": True})
        else:
            source = nearest_frame(by_round, index)
            rows = {str(r["id"]): r for r in (source["vehicles"] if source else [])
                    if r.get("present", True)}
            for vid in cav_ids:
                row = rows.get(vid)
                if row is None or int(row.get("lane", -1)) < 0:
                    live[vid] = False
                    continue
                if live[vid]:
                    cav_rows.append({"id": vid, "s": float(row["s"]), "lane": int(row["lane"]),
                                     "v": float(row["v"]), "present": True, "is_cav": True})
        vehicles = [dict(row) for row in frame["vehicles"] if not row.get("is_cav")]
        vehicles.extend(cav_rows)
        out.append({"time_s": frame["time_s"], "vehicles": vehicles})
    return out


def keep_speed_teacher_forced(state: dict, controllers: Mapping, horizon_s: float,
                              seed: int) -> list[dict]:
    arm = str((controllers or {}).get("arm", "native"))
    return teacher_force_cavs(keep_speed(state, controllers, horizon_s, seed), state, arm)


def load_predictor(spec: str, checkpoint: str | Path | None = None,
                   flags: RolloutFlags = RolloutFlags()) -> Predict:
    """Resolve a model spec to its ``predict_states`` callable."""
    if spec == "keep_speed":
        return keep_speed
    if spec == "keep_speed_tf":
        return keep_speed_teacher_forced
    if spec == "wms":
        if checkpoint is None:
            raise EvalError("model 'wms' requires a checkpoint path")
        return WmsPredictor.from_checkpoint(checkpoint, flags).predict_states
    raise EvalError(f"unknown model {spec!r}; choose from {', '.join(MODEL_SPECS)}")
