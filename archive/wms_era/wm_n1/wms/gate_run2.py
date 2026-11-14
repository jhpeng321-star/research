"""Run-2 smoke gate: check400 base contract + traffic-loss finite + entrants > 0.

The run-2 trainer (entries-v2 + traffic-level losses) adds log record fields on
top of the frozen WM-S contract.  This gate fails closed unless, in addition to
the ``check400`` contract (400 records, all base terms finite, monotone gate),
every numeric log field is finite and the training actually produced entrants.

Usage::

    python -m wms.gate_run2 smoke400/train_log.jsonl --expected 400

Field names are discovered from the log by name (keys containing ``traffic`` and
``entrant``); explicit ``--traffic-term`` / ``--entrant-term`` may pin them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import check400


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check(path: Path, expected: int = 400,
          traffic_terms: list[str] | None = None,
          entrant_terms: list[str] | None = None,
          require_projection: bool = False,
          require_conservation: bool = False) -> dict:
    base = check400.check(path, expected=expected)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen_keys: set[str] = set()
    found_traffic: set[str] = set()
    found_entrant: set[str] = set()
    found_projection: set[str] = set()
    found_conservation: set[str] = set()
    entrant_total = 0.0
    for row in rows:
        step = row.get("step")
        for key, value in row.items():
            if not _numeric(value):
                continue
            seen_keys.add(key)
            if not math.isfinite(float(value)):
                raise SystemExit(f"GATE FAIL nonfinite numeric field {key} "
                                 f"at step {step}")
            low = key.lower()
            if "traffic" in low:
                found_traffic.add(key)
            if "entrant" in low:
                found_entrant.add(key)
                entrant_total += float(value)
            if "projection" in low:
                found_projection.add(key)
            if "conservation" in low:
                found_conservation.add(key)
    if traffic_terms:
        missing = sorted(set(traffic_terms) - seen_keys)
        if missing:
            raise SystemExit(f"GATE FAIL traffic terms absent from log: {missing}")
        found_traffic |= set(traffic_terms)
    if not found_traffic:
        raise SystemExit("GATE FAIL no traffic-loss term found in the run-2 log")
    if entrant_terms:
        missing = sorted(set(entrant_terms) - seen_keys)
        if missing:
            raise SystemExit(f"GATE FAIL entrant terms absent from log: {missing}")
        found_entrant |= set(entrant_terms)
    if not found_entrant:
        raise SystemExit("GATE FAIL no entrant count found in the run-2 log")
    if entrant_total <= 0.0:
        raise SystemExit(f"GATE FAIL zero entrants over {len(rows)} records")
    if require_projection and not found_projection:
        raise SystemExit("GATE FAIL no projection metric found in the run-2 log")
    if require_conservation and not found_conservation:
        raise SystemExit("GATE FAIL no conservation metric found in the run-2 log")
    result = {**base, "run2_gate": "passed",
              "traffic_terms": sorted(found_traffic),
              "entrant_terms": sorted(found_entrant),
              "projection_terms": sorted(found_projection),
              "conservation_terms": sorted(found_conservation),
              "entrant_total": entrant_total}
    print("records %d" % len(rows))
    print("nonfinite 0 missing 0")
    print("traffic terms %s" % ",".join(sorted(found_traffic)))
    print("entrant terms %s total %g" % (",".join(sorted(found_entrant)), entrant_total))
    print("projection terms %s" % ",".join(sorted(found_projection)))
    print("conservation terms %s" % ",".join(sorted(found_conservation)))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected", type=int, default=400)
    parser.add_argument("--traffic-term", action="append", default=None)
    parser.add_argument("--entrant-term", action="append", default=None)
    parser.add_argument("--require-projection", action="store_true")
    parser.add_argument("--require-conservation", action="store_true")
    args = parser.parse_args()
    check(args.log, args.expected, args.traffic_term, args.entrant_term,
          require_projection=args.require_projection,
          require_conservation=args.require_conservation)
