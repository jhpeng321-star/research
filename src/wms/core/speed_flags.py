"""S-SPEED2 opt-in speed-up switches (all default OFF).

Each switch is a process environment variable read at call time, so a switch can be set
for one run (``WMS_SPEED_COMPACT_SLOTS=1 python -m wms train ...``) without any config,
checkpoint or state_dict change. Boolean switches require exactly ``"1"``; the audit
period defaults to 1 (off). Set flags before starting a run and keep them fixed through
rollout/backward, including checkpoint recomputation. Evidence: ``SPEED2.md``.

* ``compact_slots`` (``WMS_SPEED_COMPACT_SLOTS``): the HDV residual attention network runs on
  the present slots only (gather -> network -> scatter).  Absent slots never enter a present
  slot's attention (neighbourhood mask), so outputs equal the dense ones up to float
  reassociation (GEMM / softmax row lengths change). Training with nonzero dropout
  falls back to the dense path to preserve random masks and RNG consumption.
* ``pivot_diag_off`` (``WMS_SPEED_PIVOT_DIAG_OFF``): rollouts skip the merge-pivot hybrid
  first-passage quadrature that only produces the ``logS`` / ``event_mass`` / ``cdf``
  diagnostics (returned as NaN).  The trajectory, every other carry, the RNG stream and the
  training losses (the merge-pivot NLL integrates its own recorded history) are bit-identical.
* ``batch_replay`` (``WMS_SPEED_BATCH_REPLAY``): the merge-pivot loss's frozen recorded-geometry
  replays (the window scenes and the ragged pre-window prefixes, both no-grad) run as ONE
  batched replay instead of one batch-1 replay per row; prefixes are padded with absent
  ticks/slots. Equal up to float reassociation of batched kernels. Active training
  dropout and incompatible scene inputs fall back to per-row replay.
* ``fast_cav_drive`` (``WMS_SPEED_FAST_CAV_DRIVE``): the timeline-controller callback reads its
  per-slot scalars from batched host copies instead of one device sync per slot. Bit-identical.
* ``fast_long_history`` (``WMS_SPEED_FAST_LONG_HISTORY``): the long driver-history features
  are computed for all samples at once (same elementwise maths, first-index leader).
  Bit-identical.
* ``fast_merge_lane`` (``WMS_SPEED_FAST_MERGE_LANE``): the bridge's forced 6 m ramp-merge
  scan (``native_ramp_merge = "gap6"``) stops after the last merge candidate instead of
  visiting all K slots; the skipped iterations are exact no-ops.  Bit-identical.
* ``grad_audit_every`` (``WMS_SPEED_GRAD_AUDIT_EVERY=N``, integer >= 2): with
  ``actor_gradient_audit`` on, the per-component actor gradient norms (one extra full
  backward through the imagination per component) are logged on updates 1, 1+N, ... only.
  The update itself never reads them: policy / critic weights are bit-identical; only the
  ``actor_grad_norms`` diagnostic is thinned.
* ``graph_cvar`` (``WMS_SPEED_GRAPH_CVAR``, S-POLSPEED): the detached 32-step CVaR bisection
  of the team reward replays one captured CUDA graph per input geometry instead of ~360
  eager launches. Same kernels, shapes and inputs: bit-identical. CPU/grad paths unchanged.
* ``async_spill`` (``WMS_SPEED_ASYNC_SPILL``, S-POLSPEED): LONGMEM saved-tensor spill copies
  device->pinned host with ``non_blocking=True`` (no host sync per saved tensor). Same bytes.
"""
from __future__ import annotations

import os

#: switch name -> environment variable.
FLAGS = {
    "compact_slots": "WMS_SPEED_COMPACT_SLOTS",
    "pivot_diag_off": "WMS_SPEED_PIVOT_DIAG_OFF",
    "batch_replay": "WMS_SPEED_BATCH_REPLAY",
    "fast_cav_drive": "WMS_SPEED_FAST_CAV_DRIVE",
    "fast_long_history": "WMS_SPEED_FAST_LONG_HISTORY",
    "fast_merge_lane": "WMS_SPEED_FAST_MERGE_LANE",
    "grad_audit_every": "WMS_SPEED_GRAD_AUDIT_EVERY",
    "graph_cvar": "WMS_SPEED_GRAPH_CVAR",
    "async_spill": "WMS_SPEED_ASYNC_SPILL",
}


def enabled(name: str) -> bool:
    """True iff the switch's environment variable is exactly ``"1"``."""
    return os.environ.get(FLAGS[name], "") == "1"


def every(name: str) -> int:
    """Integer period switch: the variable's value if it is an integer >= 1, else 1 (off)."""
    raw = os.environ.get(FLAGS[name], "").strip()
    try:
        value = int(raw) if raw.isdigit() else 1
    except ValueError:
        return 1
    return max(value, 1)


def active() -> dict:
    """Current value of every switch (for run receipts / logs)."""
    return {name: (every(name) if name == "grad_audit_every" else enabled(name))
            for name in FLAGS}
