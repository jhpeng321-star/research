"""C4 SUMO side: per-CAV direct-acceleration environment, baselines, recorder, collector.

Implements ``wms.control.contract.CavEnv`` on the preserved merge network
(docs/WM_C4_DESIGN.md section 2.7, notes in SUMO_NOTES.md).  ``CommandOwnership`` stays
the sole longitudinal writer; HDVs are never commanded.

Layering: the environment, conversion, recorder and collector never import ``wms``.
Every ``wms`` dependency (wall executor, trainer history featurisers, contract constants)
is confined to :mod:`traffic_shepherd.c4.wms_glue` (the one sanctioned sim -> wms edge,
listed in ``tests/unit/test_import_graph.py``) and is imported lazily from there.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["ROOT"]
