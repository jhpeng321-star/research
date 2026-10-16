"""Input normalisation and streaming readers for evaluation state records.

A *frame* is ``{"time_s": float, "vehicles": [{"id", "s", "lane", "v", "present",
"is_cav"}, ...]}``; a *state record* carries an initial frame and the recorded
future frames per arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .errors import DataError

DT_S = 0.5
_STATE_ARRAY_KEYS = ('"windows"', '"states"', '"records"')
_CHUNK_BYTES = 1 << 20


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def vehicle(row: dict) -> dict:
    """Map the common spellings of one vehicle row to the public schema."""
    vid = row.get("id", row.get("vehicle_id"))
    if vid is None:
        raise DataError("vehicle is missing id")
    return {
        "id": str(vid),
        "s": _number(row.get("s", row.get("s_down_m", row.get("lane_position_m")))),
        "lane": row.get("lane", row.get("lane_phys", row.get("lane_id", -1))),
        "v": _number(row.get("v", row.get("v_mps", row.get("speed_mps")))),
        "present": bool(row.get("present", True)),
        "is_cav": bool(row.get("is_cav", row.get("cav", False))),
    }


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


def frames(raw: Any, *, t0_s: float = 0.0, dt_s: float = DT_S) -> list[dict]:
    """Normalise a frame list or ``{"frames": [...]}``; bare vehicle lists get times."""
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
    if isinstance(state, dict) and "frame" in state:
        return frame(state["frame"], _number(record.get("t0_s")))
    raise DataError("state record needs state.vehicles or initial_state.vehicles")


def recorded_for(record: dict, arm: str) -> list[dict]:
    """Recorded future frames of ``arm`` (a bare list is the ``native`` arm)."""
    recorded = record.get("recorded", record.get("recorded_trajectories", {}))
    if isinstance(recorded, dict) and arm in recorded:
        return frames(recorded[arm], t0_s=_number(record.get("t0_s")))
    if arm == "native" and isinstance(recorded, list):
        return frames(recorded, t0_s=_number(record.get("t0_s")))
    raise DataError(f"recorded trajectory for arm {arm!r} is missing")


def load_states(path: Path) -> list[dict]:
    """Read a whole state file (small files only; use :func:`iter_windows` otherwise)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("states", raw.get("records", raw.get("windows")))
    if not isinstance(raw, list) or not raw:
        raise DataError("states file must contain a non-empty JSON list")
    if not all(isinstance(item, dict) for item in raw):
        raise DataError("every state entry must be an object")
    return raw


def iter_windows(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    """Yield the top-level records of a ``{"windows"|"states"|"records": [...]}`` file.

    The file is decoded as latin-1 per binary chunk (the JSON is ASCII), so header
    byte offsets stay valid for ``seek`` and at most one record plus one chunk is
    resident, however large the file is.
    """
    path = Path(path)
    decoder = json.JSONDecoder()
    with path.open("rb") as fh:
        head = fh.read(1 << 16).decode("latin-1")
        key_at = next((head.find(key) for key in _STATE_ARRAY_KEYS if key in head), -1)
        if key_at == -1:
            raise DataError(f"{path.name}: no states array key found")
        bracket = head.find("[", key_at)
        if bracket == -1:
            raise DataError(f"{path.name}: states key is not followed by an array")
        fh.seek(bracket + 1)
        buf = fh.read(_CHUNK_BYTES).decode("latin-1")
        pos = 0
        emitted = 0
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                chunk = fh.read(_CHUNK_BYTES)
                if not chunk:
                    return
                buf = buf[pos:] + chunk.decode("latin-1")
                pos = 0
            if buf[pos] == "]":
                return
            try:
                record, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = fh.read(_CHUNK_BYTES)
                if not chunk:
                    raise DataError(f"{path.name}: truncated states array") from None
                buf = buf[pos:] + chunk.decode("latin-1")
                pos = 0
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            pos = end
