"""WM-S2 multi-stage smoke gate: every curriculum stage, every loss term.

Unlike :mod:`wms.check400` (one fixed unroll and an exact record count), this
gate accepts a single smoke log whose records span several curriculum stages
(``stage_s``) and fails closed unless:

* every requested curriculum stage produced exactly ``steps_per_stage`` records;
* every numeric log field is finite, the gate is monotone with ``g(0) == 0``,
  and every record has a non-empty HDV target mask (the frozen WM-S contract);
* the driver-parameter encoder outputs stay inside their hard bounds
  (:data:`wms.check400.DRIVER_PARAM_BOUNDS`, WM-S2 ``driver_params`` mode);
* each run-2 loss/metric family is present (traffic loss, entrant count,
  projection, conservation).

Usage::

    python -m wms.gate_s2 smoke_stages/train_log.jsonl \
        --stages 10,30,60,120,300 --steps-per-stage 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import check400
from .brakefix import required_log_terms

#: Run-2/S2 loss/metric name fragments that must appear somewhere in the log.
REQUIRED_TERMS = ("traffic", "entrant", "projection", "conservation")
#: Braking-fix loss terms (wms/brakefix.py); require only the ENABLED ones.
BRAKEFIX_TERMS = ("asym_speed", "decel_weight", "propagation")


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stage_key(row: dict) -> float:
    return round(float(row.get("stage_s", -1.0)), 6)


def check(path: Path, *, stages, steps_per_stage: int,
          required_terms=REQUIRED_TERMS, driver_params: bool = True,
          config_path: Path | None = None) -> dict:
    required_terms = tuple(dict.fromkeys(
        (*required_terms, *required_log_terms(path, config_path))))
    rows = [json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit("GATE FAIL empty multi-stage smoke log")
    wanted = [round(float(s), 6) for s in stages]
    by_stage: dict[float, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(_stage_key(row), []).append(row)
    missing = [s for s in wanted if s not in by_stage]
    if missing:
        raise SystemExit(f"GATE FAIL curriculum stages missing from log: {missing}")
    for stage, stage_rows in by_stage.items():
        if len(stage_rows) != int(steps_per_stage):
            raise SystemExit(
                f"GATE FAIL stage {stage:g}: {len(stage_rows)} records != "
                f"{int(steps_per_stage)}")
    seen_keys: set[str] = set()
    for row in rows:
        step = row.get("step")
        for term in set(required_terms) & set(BRAKEFIX_TERMS):
            if not _numeric(row.get(term)):
                raise SystemExit(f"GATE FAIL required term {term!r} at step {step}")
        if row.get("finite") is not True:
            raise SystemExit(f"GATE FAIL finite flag at step {step}")
        if row.get("gate_monotone") is not True:
            raise SystemExit(f"GATE FAIL gate monotonicity at step {step}")
        if abs(float(row.get("gate_g0", 1.0))) > 0.0:
            raise SystemExit(f"GATE FAIL g(0) != 0 at step {step}")
        if int(row.get("valid_targets", 0)) < 1:
            raise SystemExit(f"GATE FAIL empty HDV target mask at step {step}")
        curve = [float(x) for x in row["gate_curve"]]
        if curve[0] != 0.0 or any(b < a for a, b in zip(curve, curve[1:])):
            raise SystemExit(f"GATE FAIL g(k) not monotone/non-negative at step {step}")
        for key, value in row.items():
            if not _numeric(value):
                continue
            seen_keys.add(key)
            if not math.isfinite(float(value)):
                raise SystemExit(f"GATE FAIL nonfinite numeric field {key} "
                                 f"at step {step}")
    lowered = {key.lower() for key in seen_keys}
    for term in required_terms:
        if not any(term in key for key in lowered):
            raise SystemExit(f"GATE FAIL no {term!r} term in the smoke log")
    if driver_params:
        check400._check_driver_params(rows)
    by_stage_sorted = sorted(by_stage)
    result = {
        "status": "passed", "records": len(rows),
        "stages": [int(s) for s in by_stage_sorted],
        "steps_per_stage": int(steps_per_stage),
        "required_terms": sorted(required_terms),
        "driver_params_checked": bool(driver_params),
    }
    print("records %d" % len(rows))
    print("stages %s" % ",".join(str(int(s)) for s in by_stage_sorted))
    print("nonfinite 0 missing 0")
    if driver_params:
        print("driver params finite in-bounds")
    print(json.dumps(result, sort_keys=True))
    return result


def _split_stages(value: str) -> list[str]:
    return [x for x in str(value or "").replace(" ", ",").split(",") if x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WM-S2 multi-stage smoke gate")
    parser.add_argument("log", type=Path)
    parser.add_argument("--stages", default="10,30,60,120,300",
                        help="comma list of curriculum stage unroll lengths (s)")
    parser.add_argument("--steps-per-stage", type=int, default=4)
    parser.add_argument("--no-driver-params", action="store_true",
                        help="skip the driver-parameter encoder bounds check")
    parser.add_argument("--no-required-terms", action="store_true",
                        help="do not require the run-2 loss/metric families "
                             "(for a history-only driver_params chain that runs "
                             "the C1 struct flags unchanged)")
    parser.add_argument("--require-extra-terms", default="",
                        help="comma list of additional loss-term name fragments that "
                             "must appear in the log; use the braking-fix terms "
                             "(asym_speed,decel_weight,propagation) for the enabled "
                             "braking-fix flags, even with --no-required-terms")
    parser.add_argument("--train-config", type=Path, default=None,
                        help="config/receipt override; enabled braking terms are automatic")
    args = parser.parse_args()
    check(args.log, stages=_split_stages(args.stages),
          steps_per_stage=int(args.steps_per_stage),
          required_terms=tuple(() if args.no_required_terms else REQUIRED_TERMS)
          + tuple(_split_stages(args.require_extra_terms)),
          driver_params=not args.no_driver_params, config_path=args.train_config)
