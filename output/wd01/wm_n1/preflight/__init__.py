"""WM-N1 Idea A pre-flight gates (piece P11).

Owner-run gates G1-G6 (see ``PREFLIGHT_GATES_IDEA_A.md``).  Invoked from
``output/wd01/wm_n1`` as::

    python -m preflight <zero-residual|ctrl-parity|label-leak|overfit|e2e-smoke|heldout-dryrun|all>

Each gate prints PASS/FAIL and writes a JSON report under
``preflight/reports/``; ``all`` also writes ``PREFLIGHT_SUMMARY.json``.
Zero SUMO, CPU only, TRAIN worlds only.

The gates call into the other Idea A pieces through the frozen
``INTERFACES_V2`` names.  Those pieces are built in parallel, so a missing
piece is reported as an explicit FAIL ("piece missing: ...") rather than a
crash; every gate's logic is unit-tested with fakes.
"""
from __future__ import annotations

__all__ = ["GATE_ORDER", "GATES"]
__version__ = "0.1.0"
