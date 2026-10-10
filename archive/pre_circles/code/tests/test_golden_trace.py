"""Golden-trace equivalence test for the TraCI snapshot refactor.

Replays the fixed deterministic episode (dense, seed 42, all
non_intervention actions, 100 steps) on the current code and compares every
per-CAV observation vector and reward against the golden file recorded on
the pre-refactor baseline (tests/golden/golden_trace_dense.json).

Regenerate the golden file ONLY from a known-good baseline:
  conda run -n rl_driving python -u tools/golden_trace.py dump \
      --out tests/golden/golden_trace_dense.json
"""

import json
from pathlib import Path

from tools.golden_trace import compare_traces, run_trace

GOLDEN_PATH = Path(__file__).parent / "golden" / "golden_trace_dense.json"


def test_golden_trace_equivalence():
    reference = json.loads(GOLDEN_PATH.read_text())
    candidate = run_trace()
    mismatches = compare_traces(reference, candidate)
    assert not mismatches, "trace diverged from golden:\n" + "\n".join(mismatches)


if __name__ == "__main__":
    test_golden_trace_equivalence()
    print("ok")
