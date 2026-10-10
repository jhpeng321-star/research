"""Controller vocabulary shared by the black-box variants and the action timeline.

The vocabulary itself lives in :mod:`wms.core.constants` (single source of truth); this
module re-exports it and adds the model-side id maps.  ``none`` is the HDV / no-controller
slot; ``other`` collects open-loop training policies (trajectory replay, fixed or switching
arms) so they are never mislabelled as native.  The structured WM-S ignores the controller
embedding entirely.
"""
from __future__ import annotations

from wms.core.constants import CONTROLLER_VOCAB, N_CONTROLLERS, controller_index

__all__ = ["CONTROLLER_VOCAB", "DIRECT_ID", "N_CONTROLLERS", "TIMELINE_TO_VOCAB",
           "controller_index"]

DIRECT_ID = CONTROLLER_VOCAB.index("direct")

#: Action-timeline ids (native, damping, PI, direct) -> vocabulary index.
TIMELINE_TO_VOCAB = (1, 3, 2, 6)
