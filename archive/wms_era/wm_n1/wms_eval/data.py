"""Input and output normalization for the WM-S evaluation contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[4]


class DataError(ValueError):
    """The states list does not satisfy the local evaluation schema."""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def vehicle(row: dict, *, default_id: str | None = None) -> dict:
    """Map common physics-log and compact state spellings to one schema."""
    vid = row.get("id", row.get("vehicle_id", default_id))
    if vid is None:
        raise DataError("vehicle is missing id")
    present = bool(row.get("present", True))
    return {
        "id": str(vid),
        "s": _number(row.get("s", row.get("s_down_m", row.get("lane_position_m")))),
        "lane": row.get("lane", row.get("lane_phys", row.get("lane_id", -1))),
        "v": _number(row.get("v", row.get("v_mps", row.get("speed_mps")))),
        "present": present,
        "is_cav": bool(row.get("is_cav", row.get("cav", False))),
    }


def physics_frame(raw: dict, cav_ids: set[str] | None = None) -> dict:
    """Convert one frozen physics row to the public s_down frame."""
    cav_ids = {str(x) for x in (cav_ids or ())}
    try:
        from features.geometry import is_junction_lane, s_down_of
    except ImportError:  # pragma: no cover - only relevant to direct imports
        def is_junction_lane(lane):
            return str(lane).startswith(":")

        def s_down_of(_lane, position):
            return -1, float(position)
    live = {str(x) for x in raw.get("live_ids", raw.get("vehicles", {}).keys())}
    rows = []
    for vid, raw_vehicle in (raw.get("vehicles") or {}).items():
        lane_id = raw_vehicle.get("lane_id")
        present = str(vid) in live
        if lane_id and not is_junction_lane(lane_id):
            try:
                lane, s = s_down_of(lane_id, raw_vehicle.get("lane_position_m", 0.0))
            except (TypeError, ValueError, KeyError):
                lane, s = -1, float(raw_vehicle.get("lane_position_m", 0.0))
        else:
            lane, s = -1, 0.0
        rows.append({
            "id": str(vid), "s": float(s), "lane": int(lane),
            "v": _number(raw_vehicle.get("speed_mps")), "present": present,
            "is_cav": str(vid) in cav_ids,
        })
    return {"time_s": _number(raw.get("time_s")), "vehicles": rows}


def frame(raw: Any, default_time: float = 0.0) -> dict:
    if not isinstance(raw, dict):
        raise DataError("trajectory frame must be an object")
    raw_vehicles = raw.get("vehicles", raw.get("state", []))
    if isinstance(raw_vehicles, dict):
        rows = [dict(value, id=key) for key, value in raw_vehicles.items()]
    elif isinstance(raw_vehicles, list):
        rows = raw_vehicles
    else:
        raise DataError("frame vehicles must be a list or object")
    return {
        "time_s": _number(raw.get("time_s", raw.get("time", default_time)), default_time),
        "vehicles": [vehicle(row) for row in rows],
    }


def frames(raw: Any, *, t0_s: float = 0.0, dt_s: float = 0.5) -> list[dict]:
    """Normalize frames, accepting a direct frame list or ``{'frames': ...}``."""
    if isinstance(raw, dict) and "frames" in raw:
        raw = raw["frames"]
    if not isinstance(raw, list):
        raise DataError("trajectory must be a list of frames")
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, list):
            item = {"time_s": t0_s + i * dt_s, "vehicles": item}
        out.append(frame(item, t0_s + i * dt_s))
    return out


def initial_frame(record: dict) -> dict:
    state = record.get("state", record.get("initial_state", record))
    if isinstance(state, dict) and "vehicles" in state:
        return frame(state, _number(record.get("t0_s")))
    if isinstance(record.get("state_row"), dict):
        return physics_frame(record["state_row"], set(record.get("cav_ids", ())))
    if isinstance(state, dict) and "frame" in state:
        return frame(state["frame"], _number(record.get("t0_s")))
    raise DataError("state record needs state.vehicles or initial_state.vehicles")


def recorded_for(record: dict, arm: str) -> list[dict]:
    recorded = record.get("recorded", record.get("recorded_trajectories", {}))
    if isinstance(recorded, dict) and arm in recorded:
        return frames(recorded[arm], t0_s=_number(record.get("t0_s")))
    paths = record.get("recorded_paths", {})
    path = paths.get(arm) if isinstance(paths, dict) else None
    if path:
        path = WORKSPACE / str(path) if not Path(path).is_absolute() else Path(path)
        try:
            Path(os.path.abspath(str(path))).relative_to(WORKSPACE)
        except ValueError as exc:
            raise DataError(f"recorded trajectory path is outside workspace: {path}") from exc
        out = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(physics_frame(json.loads(line), set(record.get("cav_ids", ()))))
        return out
    if arm == "native" and isinstance(recorded, list):
        return frames(recorded, t0_s=_number(record.get("t0_s")))
    raise DataError(f"recorded trajectory for arm {arm!r} is missing")


def load_states(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("states", raw.get("records", raw.get("windows")))
    if not isinstance(raw, list) or not raw:
        raise DataError("states file must contain a non-empty JSON list")
    if not all(isinstance(item, dict) for item in raw):
        raise DataError("every state entry must be an object")
    return raw


def iter_states(path: Path):
    """Yield state records one at a time.

    ``.jsonl``/``.ndjson`` files are read line-by-line so a large RUN2 TRAIN
    evaluation set is never held in RAM; any other suffix falls back to the
    in-memory :func:`load_states` reader.
    """
    suffix = str(path).lower()
    if suffix.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    yield from load_states(path)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
