"""Fail-closed A11-style gate for the bounded WM-S 400-step smoke.

Usage: python -m wms.check400 train_log.jsonl [--expected 400] [--unroll 20]
Checks: record count, every loss term finite, valid targets present, gate g(k)
monotone non-decreasing with g(0) exactly 0 at every logged step, unroll length.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Encoder driver-parameter value bounds (hard clamps in wms.driver_encoder).
DRIVER_PARAM_BOUNDS = {
    "v0": (5.0, 60.0), "time_headway": (0.2, 4.0), "min_gap": (0.5, 10.0),
    "accel": (0.3, 5.0), "decel": (0.5, 10.0), "politeness": (0.0, 1.0),
    "mobil_threshold": (0.0, 2.0), "merge_gap": (2.0, 30.0),
    "merge_urgency": (0.1, 2.0),
}


def _check_driver_params(rows: list[dict]) -> None:
    """WM-S2 smoke gate: encoder outputs finite and inside the value bounds."""
    for row in rows:
        step = row.get("step")
        stats = row.get("driver_param_stats")
        if not stats:
            raise SystemExit(f"GATE FAIL driver_param_stats missing at step {step}")
        if int(row.get("valid_hdv_slots_t0", 0)) < 1:
            raise SystemExit(f"GATE FAIL no valid HDV t0 slots at step {step}")
        for name, bounds in DRIVER_PARAM_BOUNDS.items():
            entry = stats.get(name)
            if entry is None:
                raise SystemExit(f"GATE FAIL driver param {name} missing at step {step}")
            lo, hi = bounds
            for key in ("min", "median", "max"):
                value = float(entry[key])
                if not math.isfinite(value):
                    raise SystemExit(
                        f"GATE FAIL nonfinite driver param {name}.{key} at step {step}")
                if value < lo - 1.0e-6 or value > hi + 1.0e-6:
                    raise SystemExit(
                        f"GATE FAIL driver param {name}.{key}={value} outside "
                        f"[{lo}, {hi}] at step {step}")


def check(path: Path, expected: int = 400, unroll: int = 20,
          driver_params: bool = False) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != expected:
        raise SystemExit(f"GATE FAIL records {len(rows)} != {expected}")
    numeric = ("loss", "position", "speed", "lane", "gate_g0", "gate_g_last", "seconds")
    for row in rows:
        step = row.get("step")
        if row.get("finite") is not True:
            raise SystemExit(f"GATE FAIL finite flag at step {step}")
        if row.get("gate_monotone") is not True:
            raise SystemExit(f"GATE FAIL gate monotonicity at step {step}")
        if row.get("unroll_steps") != unroll:
            raise SystemExit(f"GATE FAIL unroll_steps {row.get('unroll_steps')} != {unroll}")
        if any(not math.isfinite(float(row[name])) for name in numeric):
            raise SystemExit(f"GATE FAIL nonfinite term at step {step}")
        if int(row.get("valid_targets", 0)) < 1:
            raise SystemExit(f"GATE FAIL empty HDV target mask at step {step}")
        curve = [float(x) for x in row["gate_curve"]]
        if curve[0] != 0.0 or abs(float(row["gate_g0"])) > 0.0:
            raise SystemExit(f"GATE FAIL g(0) != 0 at step {step}")
        if any(b < a for a, b in zip(curve, curve[1:])) or curve[-1] < 0.0:
            raise SystemExit(f"GATE FAIL g(k) not monotone/non-negative at step {step}")
    if driver_params:
        _check_driver_params(rows)
    result = {"status": "passed", "records": len(rows), "finite_terms": True,
              "driver_params_checked": bool(driver_params),
              "gate_g0": 0.0, "gate_curve_last": rows[-1]["gate_curve"],
              "gate_monotone": True, "unroll_steps": unroll,
              "loss_first10": sum(r["loss"] for r in rows[:10]) / 10,
              "loss_last10": sum(r["loss"] for r in rows[-10:]) / 10,
              "mean_step_seconds": sum(r["seconds"] for r in rows) / len(rows)}
    print("records %d" % len(rows))
    print("nonfinite 0 missing 0")
    print("gate g0 0 monotone yes unroll %d" % unroll)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected", type=int, default=400)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--driver-params", action="store_true",
                        help="also gate the driver-parameter encoder outputs (WM-S2)")
    args = parser.parse_args()
    check(args.log, args.expected, args.unroll, driver_params=args.driver_params)
