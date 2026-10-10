"""Accessors for the WM-S state mapping (scalar fields or the packed ``veh_kin``)."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M, KIN_F  # noqa: F401  (single source of truth)

_SCALAR_FIELDS = ("s_m", "v_mps", "a_mps2", "lane")


def state_kin(state: Mapping[str, Tensor]) -> Tensor:
    """Packed kinematics ``[B, K, 7]`` of a state mapping.

    Scalar fields are authoritative after a rollout step; ``veh_kin`` is only the
    initial packed view and is used when the scalar fields are missing.
    """
    if not all(key in state for key in _SCALAR_FIELDS) and "veh_kin" in state:
        return state["veh_kin"]
    lane = state["lane"].long()
    one_hot = F.one_hot(lane.clamp_min(0), num_classes=3).to(state["s_m"].dtype)
    one_hot = one_hot * (lane >= 0).unsqueeze(-1).to(one_hot.dtype)
    length = state.get("length_m", torch.full_like(state["s_m"], DEFAULT_LENGTH_M))
    return torch.cat((state["s_m"].unsqueeze(-1), one_hot,
                      state["v_mps"].unsqueeze(-1),
                      state["a_mps2"].unsqueeze(-1),
                      length.unsqueeze(-1)), dim=-1)


def state_present(state: Mapping[str, Tensor], kin: Tensor) -> Tensor:
    """Boolean occupancy ``[B, K]``; all slots are present when unspecified."""
    return state.get("present", torch.ones(kin.shape[:2], dtype=torch.bool,
                                           device=kin.device)).bool()
