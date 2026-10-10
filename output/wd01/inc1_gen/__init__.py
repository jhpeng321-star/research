"""Shared scripted R5/R6 multi-CAV response generators (INC1 addendum 2).

ONE shared module used by BOTH call sites: the SUMO arm (through
``CommandOwnership``, longitudinal commands only) and the WM rollout as the
in-the-loop CAV mover.  Numpy-only: no traci, no torch.  Zero SUMO; nothing
reads an outcome, a held-out split, or a live simulator.

* :mod:`inc1_gen.states`     -- the shared per-vehicle/obstacle record
* :mod:`inc1_gen.generators` -- R5 harmonisation, R6 gap opening (A2.2/A2.3)
* :mod:`inc1_gen.adapters`   -- traci-like and tensor-like call-site shims
  plus the A2.6 two-call-site identity harness
* :mod:`inc1_gen.manifest`   -- the ``INC1_GENERATORS.json`` sha gate (A2.6)

R1/R2 stay where they are (``output/wd01/inc1/schedules.py`` SUMO-side,
``output/wd01/wm_n1/inc1/incident_ctrl.py`` WM-side); neither existing pair
can be imported here UNCHANGED through this numpy-only interface (see
``INC1_GEN.md``).
"""
from __future__ import annotations

from .adapters import (run_two_site_identity, states_equal,
                       states_from_traci_like, states_from_wm_arrays,
                       steps_equal)
from .generators import (R5Harmonisation, R6GapOpening, GeneratorStep,
                         clip_command, headway_accel, make_generator,
                         staggered_target)
from .states import ObstacleRecord, VehState

__all__ = [
    "GeneratorStep",
    "ObstacleRecord",
    "R5Harmonisation",
    "R6GapOpening",
    "VehState",
    "clip_command",
    "headway_accel",
    "make_generator",
    "run_two_site_identity",
    "staggered_target",
    "states_equal",
    "states_from_traci_like",
    "states_from_wm_arrays",
    "steps_equal",
]
