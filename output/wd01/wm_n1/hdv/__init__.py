"""P2 ``hdv``: HDV residual network (attention over neighbour kinematics)."""
from __future__ import annotations

import importlib as _importlib

from .config import CAV_FLAG_FIELD, KIN_F, KIN_FIELDS, HdvConfig
from .net import HdvResidualNet, input_schema, lane_logit_delta_to_logits

# Integration wiring: the pre-flight G3 entry points live in a submodule named
# ``hdv.input_schema``.  Load it eagerly so ``importlib.import_module`` returns
# the cached module, then re-bind the public name to the frozen net function so
# ``from hdv import input_schema`` keeps returning the function.
_input_schema_module = _importlib.import_module(__name__ + ".input_schema")
from .net import input_schema  # noqa: E402,F811  (re-bind to the function)

__all__ = ["CAV_FLAG_FIELD", "KIN_F", "KIN_FIELDS", "HdvConfig",
           "HdvResidualNet", "input_schema", "lane_logit_delta_to_logits"]
