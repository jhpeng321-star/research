"""Bind generator switch schedules to indexed-pack slots by vehicle identity.

The pack's t0 identity row and ``vehicle_map.json`` establish stable IDs. Dense
truth's per-tick slot map is then used for current columns; dense arrays stay
memory-mapped and are indexed one tick at a time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from output.wd01.wm_n1.wms.action_timeline import ActionTimeline, CONTROLLERS


class IndexedSwitchError(ValueError):
    """The schedule cannot be joined to every required packed identity."""


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise IndexedSwitchError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise IndexedSwitchError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def bind_indexed_switch_timeline(*, schedule_path: str | Path,
                                 suffix_path: str | Path | None,
                                 identities_path: str | Path,
                                 vehicle_map_path: str | Path,
                                 dense_truth_dir: str | Path,
                                 world: str, arm: str, identity_row: int,
                                 dense_t0_tick: int = 0) -> ActionTimeline:
    """Resolve switch and issued-command records onto dense tick-local slots.

    ``identity_row`` is the row aligned with the evaluated t0 window in
    ``identities.npy``. ``dense_t0_tick`` is its local 0.5 s tick in the dense
    run. Schedule/suffix rows accept ``step`` or integer-grid ``timestamp_s``.
    Every scheduled vehicle must be a CAV in t0 identities and in
    ``vehicle_map``; every suffix command must resolve to a present dense slot.
    """
    schedule_path = Path(schedule_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("schema") not in ("wms_switch_schedule/1", "wms_action_timeline/1"):
        raise IndexedSwitchError("unsupported switch schedule schema")
    n_steps = int(schedule.get("control_steps", 0))
    control_dt = float(schedule.get("control_dt_s", 1.0))
    t0_s = float(schedule.get("t0_s", 0.0))
    if n_steps < 1 or not np.isfinite(control_dt) or control_dt <= 0:
        raise IndexedSwitchError("schedule requires positive control_steps and control_dt_s")
    roster = [str(vid) for vid in schedule.get("vehicle_ids", [])]
    if not roster or len(set(roster)) != len(roster):
        raise IndexedSwitchError("schedule vehicle_ids must be non-empty and unique")

    identities = np.load(Path(identities_path), mmap_mode="r", allow_pickle=False)
    if identities.ndim != 2 or identities.dtype.kind not in "iu" \
            or not 0 <= int(identity_row) < identities.shape[0]:
        raise IndexedSwitchError("identity sidecar row/shape is invalid")
    vehicle_map = json.loads(Path(vehicle_map_path).read_text(encoding="utf-8"))
    if not isinstance(vehicle_map, list):
        raise IndexedSwitchError("vehicle_map.json must contain an identity list")
    by_id: dict[str, list[int]] = {}
    for canonical, key in enumerate(vehicle_map):
        if not isinstance(key, (list, tuple)) or len(key) != 3:
            raise IndexedSwitchError("vehicle_map entries must be (chunk, world, vehicle_id)")
        chunk, mapped_world, vehicle_id = map(str, key)
        if mapped_world == str(world):
            by_id.setdefault(vehicle_id, []).append(canonical)
    id_to_pack_slot = {}
    row_ids = np.asarray(identities[int(identity_row)])
    for vid in roster:
        candidates = by_id.get(vid, [])
        if len(candidates) != 1:
            raise IndexedSwitchError(f"t0 identity {vid!r} is missing or ambiguous in vehicle_map")
        slots = np.flatnonzero(row_ids == candidates[0])
        if slots.size != 1:
            raise IndexedSwitchError(f"t0 identity {vid!r} is unmatched in identities.npy")
        id_to_pack_slot[vid] = int(slots[0])

    initial = {str(item["vehicle_id"]): str(item["controller"])
               for item in schedule.get("initial", [])}
    if set(initial) - set(roster):
        raise IndexedSwitchError("initial schedule references an unknown vehicle")
    controller = np.full((n_steps, len(roster)), CONTROLLERS.index("native"),
                         dtype=np.int64)
    accel = np.zeros((n_steps, len(roster)), dtype=np.float32)
    target = np.zeros_like(accel)
    supplied = np.zeros((n_steps, len(roster)), dtype=bool)
    roster_col = {vid: i for i, vid in enumerate(roster)}
    for vid, label in initial.items():
        if label not in CONTROLLERS:
            raise IndexedSwitchError(f"unsupported initial controller {label!r}")
        controller[:, roster_col[vid]] = CONTROLLERS.index(label)

    def step_for(row: dict) -> int:
        if "step" in row:
            return int(row["step"])
        timestamp = float(row["timestamp_s"])
        step = int(round((timestamp - t0_s) / control_dt))
        if abs(t0_s + step * control_dt - timestamp) > 1e-6:
            raise IndexedSwitchError("switch/command timestamp is off the control grid")
        return step

    for event in schedule.get("switches", []):
        vid = str(event.get("vehicle_id", ""))
        if vid not in roster_col:
            raise IndexedSwitchError(f"switch references unmatched identity {vid!r}")
        step = step_for(event)
        label = str(event.get("controller", ""))
        if not 0 <= step < n_steps or label not in CONTROLLERS:
            raise IndexedSwitchError("switch step/controller is invalid")
        controller[step:, roster_col[vid]] = CONTROLLERS.index(label)

    if suffix_path is not None:
        for row in _read_jsonl(Path(suffix_path)):
            vid = str(row.get("vehicle_id", ""))
            if vid not in roster_col:
                raise IndexedSwitchError(f"suffix command references unmatched identity {vid!r}")
            step = step_for(row)
            col = roster_col[vid]
            if not 0 <= step < n_steps or controller[step, col] != CONTROLLERS.index("direct"):
                raise IndexedSwitchError("suffix command does not align with a direct schedule action")
            if supplied[step, col]:
                raise IndexedSwitchError("duplicate suffix command")
            accel[step, col] = float(row["issued_acceleration_mps2"])
            target[step, col] = float(row["issued_target_speed_mps"])
            supplied[step, col] = True
    if np.any((controller == CONTROLLERS.index("direct")) & ~supplied):
        raise IndexedSwitchError("every direct schedule action requires a suffix command")

    dense_root = Path(dense_truth_dir)
    dense = json.loads((dense_root / "DENSE_TRUTH_RECEIPT.json").read_text(encoding="utf-8"))
    dense_run = next((row for row in _read_jsonl(
        dense_root / dense.get("runs_file", "runs.jsonl"))
        if row.get("world") == str(world) and row.get("arm") == str(arm)), None)
    if dense_run is None:
        raise IndexedSwitchError(f"dense truth has no run {(world, arm)}")
    run_root = dense_root / dense_run["dir"]
    slot_vehicle = np.load(run_root / "slot_vehicle.npy", mmap_mode="r", allow_pickle=False)
    dense_ids = json.loads((run_root / "vehicle_ids.json").read_text(encoding="utf-8"))
    out = np.full((n_steps, identities.shape[1]), CONTROLLERS.index("native"),
                  dtype=np.int64)
    out_accel = np.zeros(out.shape, dtype=np.float32)
    out_target = np.zeros(out.shape, dtype=np.float32)
    for step in range(n_steps):
        tick = int(dense_t0_tick) + int(round(step * control_dt / 0.5))
        if not 0 <= tick < slot_vehicle.shape[0]:
            raise IndexedSwitchError(f"dense tick {tick} is outside the run")
        current = {}
        for slot, vehicle_row in enumerate(slot_vehicle[tick]):
            vehicle_row = int(vehicle_row)
            if vehicle_row >= 0:
                vid = str(dense_ids[vehicle_row])
                if vid in current:
                    raise IndexedSwitchError(f"dense tick {tick} duplicates vehicle {vid!r}")
                current[vid] = slot
        for vid, pack_slot in id_to_pack_slot.items():
            # The t0 sidecar is an identity assertion; dense slots may change.
            if step == 0 and pack_slot >= identities.shape[1]:
                raise IndexedSwitchError(f"pack slot for {vid!r} is outside identities.npy")
            if vid not in current:
                continue  # departed vehicle has no slot at this tick
            # Dense slots are tick-local.  Resolve the identity back to its t0
            # pack column rather than leaking the dense slot index into the WM
            # timeline.
            pack_slot = id_to_pack_slot[vid]
            col = roster_col[vid]
            out[step, pack_slot] = controller[step, col]
            out_accel[step, pack_slot] = accel[step, col]
            out_target[step, pack_slot] = target[step, col]
    # Keep the t0 pack join explicit and fail closed if its slot disagrees with
    # the dense identity at t0; later ticks intentionally use dense slot maps.
    for vid, pack_slot in id_to_pack_slot.items():
        dense_t0 = next((slot for slot, value in enumerate(slot_vehicle[int(dense_t0_tick)])
                         if int(value) >= 0 and str(dense_ids[int(value)]) == vid), None)
        if dense_t0 is None:
            raise IndexedSwitchError(f"t0 identity {vid!r} has no dense-truth slot")
        if str(vehicle_map[int(row_ids[pack_slot])][2]) != vid:
            raise IndexedSwitchError(f"t0 pack slot does not resolve to {vid!r}")
    return ActionTimeline(out, out_accel, out_target, control_dt)
