"""WM-N1 model package (design ``WM_N1_DESIGN.md`` sections 2-3).

Public API:

* :class:`model.net.WmN1Net`, :class:`model.net.WmN1Config`,
  :func:`model.net.parameter_count`;
* :mod:`model.losses`;
* :func:`model.rollout.rollout`, :func:`model.rollout.project_step`;
* :class:`model.ensemble.WmN1Ensemble`;
* :class:`model.paired_effect.PairedEffectHead` (optional, default-off).

Inputs are exactly the frozen featurizer tensors (``features/FORMAT.md``);
the package never imports featurizer code at runtime.
"""
from __future__ import annotations

__all__ = [
    "constants", "net", "losses", "rollout", "ensemble", "paired_effect",
    "WmN1Net", "WmN1Config", "WmN1Ensemble", "parameter_count", "rollout",
    "PairedEffectHead",
]

from .net import WmN1Config, WmN1Net, parameter_count
from .ensemble import WmN1Ensemble
from .rollout import rollout
from .paired_effect import PairedEffectHead
