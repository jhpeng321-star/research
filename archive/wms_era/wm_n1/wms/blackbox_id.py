"""C1 black-box *controller-id* variant: evaluation adapter (``wms/bb``).

The learned model class is ``wms.model.WmSModel(black_box=True)``: human drivers
come from the physics base plus a learned residual, while CAVs are predicted by
the same network conditioned on a controller-id embedding (no controller code is
ever executed).  This module is the factory the ``wms_eval`` black-box loader
imports (``wms.blackbox_id.build``), keeping the harness's clean hook namespace.

The returned adapter reuses :class:`wms_eval.wms_plugin.WmsCheckpointModel` in
``blackbox_id`` mode: it loads the ``wm_s/1`` container with the trained physics
and CAV-prediction contract, and passes the arm's controller id
(``native``/``pi``) to the rollout instead of teacher-forcing CAVs.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def build(checkpoint: str, flags: Any = None,
          physics: Optional[Mapping[str, Any]] = None) -> Any:
    """Return the C1 evaluation adapter for a black-box-id checkpoint."""
    from wms_eval.wms_plugin import WmsCheckpointModel

    return WmsCheckpointModel(checkpoint, flags, physics=physics,
                              model_kind="blackbox_id")


__all__ = ["build"]
