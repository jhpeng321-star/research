"""R13E shared secure-gap law (GAPACC ``gs``), used by the insertion veto and the yield pass rule.

``need = T_back * v_back + relu(v_back**2 / (2 b_back) - v_front**2 / (2 b_front))``

This is exactly the GAPACC ``scripts/factworker.py::install_gate(use_T=True)`` requirement:
learned per-driver time headway ``T`` and comfortable deceleration ``b`` of the actual
back/front vehicles, NO standstill (s0) term, no multiplier, no padding.  A side passes when
``need`` is non-finite or ``gap >= need`` (GAPACC semantics, including its acceptance of a
non-finite requirement).  No constants are introduced.
"""
from __future__ import annotations

import torch
from torch import Tensor


def secure_gap_need(v_back: Tensor, v_front: Tensor, t_back: Tensor,
                    b_back: Tensor, b_front: Tensor) -> Tensor:
    """GAPACC ``gs`` required bumper gap of the back vehicle behind the front vehicle."""
    return t_back * v_back + torch.relu(v_back.square() / (2.0 * b_back)
                                        - v_front.square() / (2.0 * b_front))


def secure_gap_ok(gap: Tensor, need: Tensor) -> Tensor:
    """GAPACC acceptance of one side: ``~isfinite(need) | (gap >= need)``."""
    return ~torch.isfinite(need) | (gap >= need)
