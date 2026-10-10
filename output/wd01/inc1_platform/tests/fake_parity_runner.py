"""FAKE parity runner for the inc1_platform tests (ZERO SUMO).

Stands in for the box-side runner that will load the saved t0 state into
the scenario copy, run 60 s and write physics_steps.jsonl.  Here it
"simulates" a byte-identical platform by writing the recorded reference
rows, so the comparison must PASS with zero deltas.
"""
from __future__ import annotations

import shutil
from pathlib import Path

CALLS: list = []


def fake_runner(scenario_dir, state_xml, duration_s, out_jsonl):
    scenario_dir = Path(scenario_dir)
    out_jsonl = Path(out_jsonl)
    reference = scenario_dir.parent / "physics_steps.jsonl"
    CALLS.append({"scenario_dir": str(scenario_dir),
                  "state_xml": str(state_xml),
                  "duration_s": float(duration_s),
                  "out_jsonl": str(out_jsonl)})
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference, out_jsonl)
    return out_jsonl
