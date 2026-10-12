"""Local sensing and CAV reports, in the physical units of the C4 contract."""
from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import (ACCEL_MIN_MPS2, CONTROL_DT_S, MAX_REPORTS, MERGE_S0_M,
                       MERGE_S1_M, N_LANES, OWN_FEATURES, REPORT_FEATURES,
                       REPORT_RANGE_M, SENSOR_RANGE_M, SPEED_LIMIT_MPS, Observation)

# Observation tensors remain in physical units (including the 100 m sentinel).
# All network feature scaling is defined here, shared by actor/RSU/critic.
FEATURE_NORMALIZATION = {
    "v_mps": SPEED_LIMIT_MPS, "dv_mps": SPEED_LIMIT_MPS,
    "a_mps2": abs(ACCEL_MIN_MPS2), "lane_0": 1.0, "lane_1": 1.0, "lane_2": 1.0,
    "s_to_merge_end_m": REPORT_RANGE_M, "s_m": REPORT_RANGE_M,
    "front_gap_m": SENSOR_RANGE_M, "front_dv_mps": SPEED_LIMIT_MPS,
    "front_present": 1.0, "rear_gap_m": SENSOR_RANGE_M,
    "rear_dv_mps": SPEED_LIMIT_MPS, "rear_present": 1.0, "in_merge_zone": 1.0,
    "ds_m": REPORT_RANGE_M, "dlane": float(N_LANES - 1), "age_s": CONTROL_DT_S,
    "length_m": DEFAULT_LENGTH_M, "is_cav": 1.0,
    "cmd_accel_mps2": abs(ACCEL_MIN_MPS2), "lane_intent": 1.0,
}


def normalize_features(values: Tensor, features: tuple[str, ...]) -> Tensor:
    """Scale features without clipping or detaching their gradients."""
    return values / values.new_tensor([FEATURE_NORMALIZATION[name] for name in features])


class CommandMemory:
    """Per-slot last commanded acceleration and lane intent that CAVs report to the RSU.

    ``augment(state)`` adds ``cmd_accel_mps2`` / ``lane_intent`` [B,K] (zeros for slots that are
    new, absent or not yet commanded); ``record(state, accel, intent)`` stores this tick's
    commands.  One memory per episode (deployment: shared by every executed candidate)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.accel = self.intent = self.present = None

    def augment(self, state: Mapping[str, Tensor]) -> dict:
        present = state["present"].bool()
        if self.accel is None or self.accel.shape != present.shape:
            zeros = torch.zeros(present.shape, dtype=state["s_m"].dtype, device=present.device)
            return {**state, "cmd_accel_mps2": zeros, "lane_intent": zeros.clone()}
        keep = present & self.present
        return {**state, "cmd_accel_mps2": torch.where(keep, self.accel, torch.zeros_like(self.accel)),
                "lane_intent": torch.where(keep, self.intent, torch.zeros_like(self.intent))}

    def record(self, state: Mapping[str, Tensor], accel: Tensor, intent: Tensor) -> None:
        present = state["present"].bool()
        accel = torch.nan_to_num(torch.as_tensor(accel, dtype=state["s_m"].dtype,
                                                 device=present.device), nan=0.0)
        self.accel = torch.where(present, accel, torch.zeros_like(accel)).detach()
        self.intent = torch.where(present, torch.as_tensor(intent, device=present.device).to(
            self.accel.dtype), torch.zeros_like(self.accel)).detach()
        self.present = present


def build_observations(state: Mapping[str, Tensor], report_age_s: Optional[Tensor] = None
                       ) -> Observation:
    """Build batched local observations; no privileged population features.

    s denotes the front bumper. Sensing range uses longitudinal |ds|; reported
    gaps are bumper gaps (floored at zero for overlaps). Both front and rear dv
    mean neighbour speed minus ego speed. Reports are sorted by |ds|, ties by
    slot index. Age can be scalar, [B,K] source ages, or [B,K,K] recipient/source
    ages. Only presence, CAV identity and geometry affect discrete membership.
    Optional ``state["cmd_accel_mps2"]`` / ``state["lane_intent"]`` [B,K] (the reporting CAV's
    previous command, see :class:`CommandMemory`) fill the last two report features (else 0).
    """
    present = state["present"].bool()
    lane = state["lane"].long()
    valid = present & (lane >= 0) & (lane < N_LANES)
    cav = valid & state["is_cav"].bool()
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(s))
    a = torch.where(valid, state["a_mps2"], torch.zeros_like(s))
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    length = torch.where(valid, length, torch.zeros_like(s))
    b, k = s.shape
    onehot = F.one_hot(lane.clamp(0, N_LANES - 1), N_LANES).to(s.dtype)
    ds = s[:, None, :] - s[:, :, None]
    same = (valid[:, :, None] & valid[:, None, :]
            & (lane[:, :, None] == lane[:, None, :])
            & ~torch.eye(k, dtype=torch.bool, device=s.device)[None])

    def neighbour(ahead: bool) -> tuple[Tensor, Tensor, Tensor]:
        eligible = same & (ds.abs() <= SENSOR_RANGE_M) & ((ds >= 0) if ahead else (ds <= 0))
        distance = torch.where(eligible, ds.abs(), torch.full_like(ds, float("inf")))
        if k == 0:
            return s, s, valid
        distance, index = distance.min(-1)
        has = torch.isfinite(distance)
        bumper_length = length.gather(-1, index) if ahead else length
        gap = torch.where(has, (distance - bumper_length).clamp_min(0),
                          torch.full_like(s, SENSOR_RANGE_M))
        dv = torch.where(has, v.gather(-1, index) - v, torch.zeros_like(v))
        return gap, dv, has

    front_gap, front_dv, front_has = neighbour(True)
    rear_gap, rear_dv, rear_has = neighbour(False)
    zone = (s >= MERGE_S0_M) & (s <= MERGE_S1_M)
    own = torch.cat((v[..., None], a[..., None], onehot,
                     (MERGE_S1_M - s)[..., None], front_gap[..., None],
                     front_dv[..., None], front_has.to(s.dtype)[..., None],
                     rear_gap[..., None], rear_dv[..., None], rear_has.to(s.dtype)[..., None],
                     zone.to(s.dtype)[..., None]), -1)
    own = torch.where(cav[..., None], own, torch.zeros_like(own))
    eligible = (cav[:, :, None] & cav[:, None, :] & (ds.abs() <= REPORT_RANGE_M)
                & ~torch.eye(k, dtype=torch.bool, device=s.device)[None])
    ranked = torch.where(eligible, ds.abs(), torch.full_like(ds, float("inf")))
    n = min(k, MAX_REPORTS)
    index = ranked.argsort(dim=-1, stable=True)[..., :n]
    mask = eligible.gather(-1, index)

    def source(values: Tensor) -> Tensor:
        return values[:, None, :].expand(b, k, k).gather(-1, index)

    age = torch.zeros_like(s) if report_age_s is None else torch.as_tensor(
        report_age_s, device=s.device, dtype=s.dtype)
    if age.ndim == 3:
        if age.shape != (b, k, k):
            raise ValueError("recipient/source report ages must be [B,K,K]")
        report_age = age.gather(-1, index)
    else:
        report_age = source(torch.broadcast_to(age, (b, k)))
    command = state.get("cmd_accel_mps2")
    intent = state.get("lane_intent")
    command = torch.zeros_like(s) if command is None else torch.where(cav, command.to(s.dtype),
                                                                      torch.zeros_like(s))
    intent = torch.zeros_like(s) if intent is None else torch.where(cav, intent.to(s.dtype),
                                                                    torch.zeros_like(s))
    report_lane = source(lane)
    report_onehot = F.one_hot(report_lane.clamp(0, N_LANES - 1), N_LANES).to(s.dtype)
    reports = torch.cat((ds.gather(-1, index)[..., None],
                         (source(v) - v[..., None])[..., None],
                         (report_lane - lane[..., None]).to(s.dtype)[..., None],
                         report_onehot, source(v)[..., None], source(front_gap)[..., None],
                         source(front_dv)[..., None], source(front_has).to(s.dtype)[..., None],
                         report_age[..., None], source(command)[..., None],
                         source(intent)[..., None]), -1)
    reports = torch.where(mask[..., None], reports, torch.zeros_like(reports))
    reports = F.pad(reports, (0, 0, 0, MAX_REPORTS - n))
    mask = F.pad(mask, (0, MAX_REPORTS - n), value=False)
    assert own.shape[-1] == len(OWN_FEATURES) and reports.shape[-1] == len(REPORT_FEATURES)
    return Observation(own, reports, mask, cav)
