"""WM-N1 featurizer + memmap writer (design WM_N1_DESIGN.md sections 1, 3, 4).

Deployment-honest, zero-SUMO featurization of the campaign physics logs into
per-run memmap shards, plus a resumable multi-worker CLI.  See FORMAT.md for
the frozen tensor layout.

Read-only imports: ``output/wd01/partB_v0data/actions_v04.py`` and
``wall_tokens_v04.py``.  This package never opens the per-tick dispatcher
action log or the realised-insertion log.
"""
from __future__ import annotations

__all__ = ["constants", "geometry", "actions_adapter", "counters",
           "featurizer", "outcomes", "splits", "writer"]
