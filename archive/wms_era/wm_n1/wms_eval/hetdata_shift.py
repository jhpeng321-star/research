"""t0-shifted state materialiser for the WM-S2 heterogeneous-driver packs.

The hetdata packs (``pack_hetdata_val`` / ``pack_hetdata_test``, box-side) store
one window per run.  Each manifest item's ``tick`` is the FIRST NON-EMPTY tick of
the window, not ``tick_start + 1``, so the pre-t0 ticks of the stored snapshot do
not contain the ``HISTORY_H`` (10-step / 5 s) history the WM-S2 driver encoder was
trained on.  This module builds the SAME kind of shifted state set the C1 v2
materialiser builds (``output/wd01/wm_n1/wms_eval/materialize.py``, the
``shift_steps`` / ``--shift-seconds`` path), but for the hetdata packs:

* for every manifest item it picks a later tick inside the SAME stored window --
  the first BUSY tick at/after ``--shift-to-t0-s`` seconds into the window (plus a
  small margin), never before ``--min-t0-s`` and never so late that
  ``tick_end - horizon_steps`` is crossed;
* it records the real ``--history-steps`` pre-t0 window ending at that tick,
  bounded by ``tick_start`` so it never crosses the run/window start;
* it writes the identical ``{"schema": "wms_driver_states/1", "states": [...]}``
  payload :func:`wms_eval.driver.materialise_pack_states` writes, from the same
  memmapped arrays, so the existing ``python -m wms_eval driver driver`` scorer
  consumes it unchanged.

The t0 is a *new* decision time: the state's ``tick`` is the shifted pack row,
``item_position`` still names the original manifest item (so the driver bundle
truth, keyed by item position and slot, is looked up exactly as before), and the
slot -> vehicle continuity is what the pack's ``target_mask`` already encodes.

Geometry (why 120 s): the window starts near the run origin, so a 120 s offset +
10-step (5 s) margin lands at a BUSY, fully-warmed-up t0 with a real 10-step
history behind it and room for the horizon ahead.  The shift is bounded above by
``tick_end - horizon_steps - 1`` and below by ``history_steps`` (and
``--min-t0-s``); an item with no fitting busy tick is skipped, never silently
placed at a wrong time.

Memory: every array is opened ``mmap_mode='r'`` and one item is processed at a
time (selection stores only integer ticks), so RSS stays far below the 6 GB
ceiling even for the full VAL/TEST packs.  This module touches no CAL / HOLDOUT /
sealed data and never modifies the pack.  It reuses
:func:`wms_eval.driver._read_pack_arrays` for the pack read and
:data:`wms_eval.driver.DEFAULT_HISTORY` / ``DT_S`` for the geometry -- it does not
re-implement the encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:  # normal ``python -m wms_eval.hetdata_shift``
    from .driver import (DEFAULT_HISTORY, DT_S, DriverScoreError,
                         _read_pack_arrays)
except ImportError:  # direct ``python .../wms_eval/hetdata_shift.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wms_eval.driver import (DEFAULT_HISTORY, DT_S, DriverScoreError,
                                 _read_pack_arrays)

DEFAULT_SHIFT_TO_T0_S = 120.0
DEFAULT_MARGIN_STEPS = 10
DEFAULT_HORIZON_STEPS = 0
DEFAULT_MIN_VEHICLES = 1
SCHEMA = "wms_driver_states/1"


def _seconds_to_steps(seconds: float, *, ceil: bool = False) -> int:
    value = float(seconds) / DT_S
    return int(np.ceil(value - 1e-9)) if ceil else int(round(value))


def _run_bounds(index: dict) -> dict:
    """``(world, arm) -> (tick_start, tick_end)`` for every packed run."""
    bounds = {}
    for rec in index.get("runs") or []:
        key = (rec.get("world"), rec.get("arm"))
        bounds[key] = (int(rec.get("tick_start", 0)), int(rec.get("tick_end", 0)))
    return bounds


def select_shift_tick(present_counts, tick_start: int, tick_end: int, *,
                      shift_to_t0_s: float, min_t0_s: float, margin_steps: int,
                      history_steps: int, horizon_steps: int,
                      min_vehicles: int = DEFAULT_MIN_VEHICLES):
    """First BUSY tick at/after the requested t0 inside ``[tick_start, tick_end)``.

    Returns ``(tick, reason)``; ``tick`` is ``None`` with a machine-readable
    ``reason`` when the window is too short or no busy tick fits.  ``present``
    is the per-tick slot-occupancy count of the pack ``veh_mask``.
    """
    target_steps = max(_seconds_to_steps(float(shift_to_t0_s)),
                       _seconds_to_steps(float(min_t0_s), ceil=True))
    start = tick_start + target_steps + max(0, int(margin_steps))
    # A real ``history_steps`` history needs ticks ``new_tick-H+1 .. new_tick``;
    # the horizon needs ``new_tick .. new_tick+horizon``.
    start = max(start, tick_start + int(history_steps) - 1)
    upper = int(tick_end) - 1 - max(0, int(horizon_steps))
    if start > upper:
        return None, (f"window too short: need tick >= {start} for t0>="
                      f"{shift_to_t0_s:.3f}s + margin, but tick_end-horizon={upper}")
    for tick in range(int(start), int(upper) + 1):
        if int(present_counts[tick]) >= int(min_vehicles):
            return tick, "ok"
    return None, (f"no busy tick (>= {min_vehicles} vehicles) in "
                  f"[{start}, {upper}]")


def _state_record(item: dict, position: int, *, tick_start: int, tick_end: int,
                  new_tick: int, kin, mask, target, history_steps: int) -> dict:
    """One shifted record, byte-compatible with ``materialise_pack_states``."""
    row = np.asarray(kin[new_tick], dtype=np.float32)
    present = np.asarray(mask[new_tick]).astype(bool)
    k = row.shape[0]
    lo = max(int(tick_start), int(new_tick) - int(history_steps) + 1)
    hist_kin = np.asarray(kin[lo:new_tick + 1], dtype=np.float32)
    hist_mask = np.asarray(mask[lo:new_tick + 1]).astype(bool)
    hist_target = (np.asarray(target[lo:new_tick], dtype=bool)
                   if new_tick > lo else np.zeros((0, k), dtype=bool))
    vehicles = []
    for slot in range(k):
        lane = int(np.argmax(row[slot, 1:4])) if row[slot, 1:4].sum() > 0 else -1
        vehicles.append({"id": f"slot{slot}", "s": float(row[slot, 0]),
                         "lane": lane, "v": float(row[slot, 4]),
                         "present": bool(present[slot]), "is_cav": False})
    item_tick = int(item.get("tick", new_tick))
    shift_steps = int(new_tick) - item_tick
    t0_s = float(item.get("t0_s", 0.0)) + shift_steps * DT_S
    return {
        "world": item.get("world"), "arm": item.get("arm"),
        "t0_s": t0_s, "tick": int(new_tick), "tick_start": int(tick_start),
        "tick_end": int(tick_end), "item_position": int(position),
        "t0_pack_tick": int(new_tick), "t0_window_s": (new_tick - tick_start) * DT_S,
        "shift_steps": int(shift_steps), "shift_s": shift_steps * DT_S,
        "history_steps": int(hist_kin.shape[0]),
        "split": item.get("split"), "policy": item.get("policy"),
        "state": {"time_s": t0_s, "vehicles": vehicles},
        "veh_kin": row.tolist(), "veh_mask": present.tolist(),
        "history": {"kin": hist_kin.tolist(), "mask": hist_mask.tolist(),
                    "target_mask": hist_target.tolist()},
    }


def materialise_shifted_states(pack_dir, out_path, *,
                               shift_to_t0_s: float = DEFAULT_SHIFT_TO_T0_S,
                               min_t0_s: float = 0.0,
                               margin_steps: int = DEFAULT_MARGIN_STEPS,
                               history_steps: int = DEFAULT_HISTORY,
                               horizon_steps: int = DEFAULT_HORIZON_STEPS,
                               min_vehicles: int = DEFAULT_MIN_VEHICLES,
                               max_items=None) -> dict:
    """Write the shifted state set for one hetdata pack.

    ``shift_to_t0_s`` is the requested t0 offset (seconds) from the window's
    ``tick_start``; ``min_t0_s`` is a hard lower bound (an item is skipped if its
    chosen t0 would fall below it).  ``horizon_steps`` only reserves room for a
    future window (the driver encoder itself needs no horizon), so the shift is
    bounded by ``tick_end - horizon_steps``.  Returns a JSON-able summary.
    """
    pack_dir = Path(pack_dir)
    out_path = Path(out_path)
    manifest, index, kin, mask, target = _read_pack_arrays(pack_dir)
    bounds = _run_bounds(index)
    items = manifest.get("items") or []
    if max_items is not None:
        items = items[: int(max_items)]
    history_steps = int(history_steps)

    present_counts = mask.astype(bool).sum(axis=1)
    selected: list[tuple[int, dict, int, int, int]] = []
    skipped: list[dict] = []
    for position, item in enumerate(items):
        key = (item.get("world"), item.get("arm"))
        tick_start, tick_end = bounds.get(key, (0, len(present_counts)))
        tick, reason = select_shift_tick(
            present_counts, tick_start, tick_end,
            shift_to_t0_s=shift_to_t0_s, min_t0_s=min_t0_s,
            margin_steps=margin_steps, history_steps=history_steps,
            horizon_steps=horizon_steps, min_vehicles=min_vehicles)
        if tick is None:
            skipped.append({"item_position": position, "world": item.get("world"),
                            "arm": item.get("arm"), "reason": reason})
            continue
        selected.append((position, item, tick, tick_start, tick_end))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write('{"schema":"%s","pack":%s,"history_steps":%d,"n_states":%d,'
                 '"shift_to_t0_s":%s,"min_t0_s":%s,"margin_steps":%d,'
                 '"horizon_steps":%d,"states":[' % (
                     SCHEMA, json.dumps(str(pack_dir)), history_steps,
                     len(selected), json.dumps(float(shift_to_t0_s)),
                     json.dumps(float(min_t0_s)), int(margin_steps),
                     int(horizon_steps)))
        for i, (position, item, tick, tick_start, tick_end) in enumerate(selected):
            record = _state_record(item, position, tick_start=tick_start,
                                   tick_end=tick_end, new_tick=tick, kin=kin,
                                   mask=mask, target=target,
                                   history_steps=history_steps)
            if i:
                fh.write(",")
            fh.write(json.dumps(record, separators=(",", ":")))
        fh.write("]}")
    return {"pack": str(pack_dir), "out": str(out_path),
            "n_states": len(selected), "n_requested": len(items),
            "n_skipped": len(skipped), "skipped": skipped}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wms_eval.hetdata_shift",
        description="Build t0-shifted WM-S2 driver state sets from a hetdata "
                    "pack (real pre-t0 history inside the same stored window).")
    p.add_argument("--pack", required=True,
                   help="hetdata pack dir (manifest.json + INDEX.json + arrays)")
    p.add_argument("--out", required=True, help="output wms_driver_states/1 JSON")
    p.add_argument("--shift-to-t0-s", type=float, default=DEFAULT_SHIFT_TO_T0_S,
                   help="requested t0 offset in seconds from tick_start "
                        "(default 120.0)")
    p.add_argument("--min-t0-s", type=float, default=0.0,
                   help="hard lower bound on the t0 offset; an item whose chosen "
                        "t0 would fall below it is skipped (default 0.0)")
    p.add_argument("--margin-steps", type=int, default=DEFAULT_MARGIN_STEPS,
                   help="extra 0.5 s ticks added after the requested t0 "
                        "(default 10)")
    p.add_argument("--history-steps", type=int, default=DEFAULT_HISTORY,
                   help="pre-t0 history width in 0.5 s ticks (default 10)")
    p.add_argument("--horizon-steps", type=int, default=DEFAULT_HORIZON_STEPS,
                   help="future ticks to reserve: new_tick <= tick_end-1-horizon "
                        "(default 0 for driver recovery)")
    p.add_argument("--min-vehicles", type=int, default=DEFAULT_MIN_VEHICLES,
                   help="minimum occupied slots for a tick to count as BUSY "
                        "(default 1)")
    p.add_argument("--max-items", type=int, default=None,
                   help="process at most N manifest items (debug/smoke)")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = materialise_shifted_states(
            args.pack, args.out, shift_to_t0_s=args.shift_to_t0_s,
            min_t0_s=args.min_t0_s, margin_steps=args.margin_steps,
            history_steps=args.history_steps, horizon_steps=args.horizon_steps,
            min_vehicles=args.min_vehicles, max_items=args.max_items)
    except (DriverScoreError, OSError, ValueError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
