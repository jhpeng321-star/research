"""WD-01-PB-V04B: Part B direct-difference tree model v0.4b on J4 wall pairs.

Implements REGISTRATION_PARTB_V04B.md (LF-sha256 e00cd98e..., registered
FINAL 2026-10-01) as a NEW package under output/wd01/partB_v04b/.  The
v0.4a/v0.4 data packages (partB_v0, partB_v0data) are imported READ-ONLY:
every shared metric primitive (CRPS-from-quantiles on the K=40 midpoint
grid, world-bootstrap CI, TRAIN references, cell-mean baseline) and every
reconstruction primitive (actions_v04, wall_tokens_v04,
deployable_features, v0lib) is reused, never copied.

The model predicts the ARM DIFFERENCE dY = Y(wall arm) - Y(native
sibling) DIRECTLY per tau (one quantile-GBM per tau x sensing variant;
no level head, no paired predictions).  The action input is the PLANNED
SCHEDULE (19 x 30 s segments + wall hold/speed/cap), so "damp 60 s then
native" != "damp 570 s" (registration section 12).

Entry points (python -m partB_v04b <cmd>, run from output/wd01):
  dry-run   TRAIN role-play on <= 4 J4 TRAIN worlds (no CAL row)
  fit       full TRAIN build + tuning + fit + TRAIN-OOB metrics
  cal-read  THE ONE CAL read, guarded by OWNER_ACK + read-once marker
  selfcheck  print registration binding + split integrity, read-only

Zero SUMO.  CPU only.  No D: paths (WD_FORBID_D=1).  HOLDOUT never
(J4 contains none); chunk-5 never; dispatcher_rows / insertions.jsonl /
outcome blocks never.
"""
from __future__ import annotations

from . import _paths  # noqa: F401  (sys.path shims, import side effect)

__version__ = "0.4b"
