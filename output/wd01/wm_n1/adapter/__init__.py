"""WM-N1 eval adapter package.

Piece ``ADAPTER``.  Bridges the trained ensemble to two consumers: the
``evalh`` harness (via the ``Predictor`` protocol) and the ``combine`` stacking
layer (via a neural OOF/CAL export).

Modules:

* :mod:`adapter.predictor` -- ``WmN1Predictor`` (evalh ``Predictor``);
* :mod:`adapter.export` -- ``export_cal`` / ``export_oof`` (combine row tables);
* :mod:`adapter.checkpoint` -- plain ``state_dict`` + JSON ensemble checkpoint
  I/O (``weights_only`` explicit);
* :mod:`adapter.quantiles` -- 19 -> 7 quantile-curve interpolation
  (INTERFACES Mismatch 1).
"""
from __future__ import annotations

__all__ = ["predictor", "export", "checkpoint", "quantiles"]

from . import checkpoint, export, predictor, quantiles
