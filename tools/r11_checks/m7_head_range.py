"""R11 M7 (R11_REVIEW C1): head-saturation rule on EVERY step of the round-11 run log.

Reject round 11 if any step with courtesy rows has follower_courtesy_head_min <= accel_min - accel_max, head_max >=
accel_max - accel_min, or head_mean outside (accel_min, accel_max) (bounds = the run config's physics plant clamp;
the same rule as the launch gate, wms.train.round11_gate.head_range_errors).  Also checks that the run completed all
steps.  Usage: m7_head_range.py RUN_DIR CONFIG OUT.json [EXPECTED_STEPS=90]
"""
import json
import sys
from pathlib import Path

from wms.train.round9_gate import read_rows
from wms.train.round11_gate import head_range_errors, physics_accel_bounds


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run, config, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    expected = int(argv[3]) if len(argv) > 3 else 90
    rows = read_rows(run / "train_log.jsonl")
    bounds = physics_accel_bounds(config)
    errors = head_range_errors(rows, bounds)
    steps = len(rows)
    verdict = "REJECT-ROUND-11" if errors else ("INCOMPLETE" if steps < expected else "PASS")
    heads = [(r.get("follower_courtesy_head_min"), r.get("follower_courtesy_head_max"),
              r.get("follower_courtesy_head_mean")) for r in rows]
    out.write_text(json.dumps({"verdict": verdict, "steps": steps, "expected_steps": expected, "bounds": bounds,
                               "errors": errors, "head_min_max_mean_per_step": heads}, indent=1), encoding="utf-8")
    print(f"M7 head range: {verdict} ({steps}/{expected} steps, {len(errors)} violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
