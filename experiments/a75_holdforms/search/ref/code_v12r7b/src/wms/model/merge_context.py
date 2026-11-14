"""Stateful merge context features (pure torch, no learned parameters).

This module is the *prep* for the next round: it computes raw physical context for
ramp vehicles (lane 2) and their lane-1 followers.  It does not change the model and
has no learned weights.  The only constants are geometry (merge lane / target lane /
merge-zone extent) and the neighbourhood association range, all shared with
:mod:`wms.model.merge_core`.

Frame and units follow :mod:`wms.train.losses.base` (``UnrollTargets.states``):
``s`` is the front-bumper position in metres (merge zone ``[0, 75.44]``, upstream
negative), ``v`` m/s, ``a`` m/s^2, ``length`` metres.  Lane indices: ramp = 2,
target = 1.

Every feature is attached to a slot.  Ramp-slot features are zero (and their valid
mask false where one exists) for non-ramp slots.  Zipper timers are world-level and
are broadcast to every present slot.

No behavioural thresholds or rules are encoded here; all quantities are raw and the
consumer (the offline fitter or a learned gate) decides what to do with them.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

#: Lane indices / merge geometry (same convention as ``merge_core``).
RAMP_LANE = 2
TARGET_LANE = 1
MERGE_ZONE_START_M = 0.0
MERGE_ZONE_END_M = 75.44
#: Neighbour association range for leader/follower pairs (geometry constant of the
#: recorded merge scenes; the nearest neighbour beyond it is not part of the gap).
MAX_NEIGHBOUR_GAP_M = 150.0
#: Follower acceleration EMA time constants (seconds).
EMA_TAU_S = (1.0, 3.0)

#: Ordered context feature names.  Units are documented in ``PREP_A.md``.
CONTEXT_FEATURES: tuple[str, ...] = (
    "n_ramp_ahead",
    "ramp_lead_gap_m",
    "ramp_lead_dv_mps",
    "dist_to_lane_end_m",
    "wait_s",
    "t_since_ramp_insertion_s",
    "t_since_lane1_passage_s",
    "last_service_was_ramp",
    "foll_speed_mps",
    "foll_accel_now",
    "foll_accel_ema1_s",
    "foll_accel_ema3_s",
    "foll_dv_mps",
    "n_ramp_same_gap",
)

#: Features whose value can be missing; a matching ``<name>_valid`` bool mask is emitted.
VALID_MASKED: tuple[str, ...] = (
    "ramp_lead_gap_m",
    "ramp_lead_dv_mps",
    "wait_s",
    "t_since_ramp_insertion_s",
    "t_since_lane1_passage_s",
    "last_service_was_ramp",
    "foll_speed_mps",
    "foll_accel_now",
    "foll_accel_ema1_s",
    "foll_accel_ema3_s",
    "foll_dv_mps",
)


class MergeContextTracker:
    """Stateful per-step context for ramp cars and their lane-1 followers.

    One call to :meth:`step` advances one decision step.  All inputs are ``[B, K]``;
    all outputs are ``[B, K]`` (valid masks are ``bool``).  The state is plain
    tensors and can be detached (:meth:`detach_`) at TBPTT chunk boundaries.
    """

    def __init__(self, batch: int, k: int, *, dt: float = 0.5,
                 device=None, dtype=None) -> None:
        if int(batch) < 0 or int(k) < 0:
            raise ValueError("batch and k must be non-negative")
        if not (math.isfinite(float(dt)) and float(dt) > 0.0):
            raise ValueError("dt must be finite and positive")
        self.batch = int(batch)
        self.k = int(k)
        self.dt = float(dt)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")

        fd = self.dtype
        b, kk = self.batch, self.k
        self.prev_present = torch.zeros(b, kk, dtype=torch.bool, device=self.device)
        self.prev_lane = torch.full((b, kk), -1, dtype=torch.long, device=self.device)
        self.prev_s = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.in_zone = torch.zeros(b, kk, dtype=torch.bool, device=self.device)
        self.wait_s = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.foll_index = torch.full((b, kk), -1, dtype=torch.long, device=self.device)
        self.has_foll = torch.zeros(b, kk, dtype=torch.bool, device=self.device)
        self.foll_ema1 = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.foll_ema3 = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.t_since_ramp = torch.zeros(b, dtype=fd, device=self.device)
        self.t_since_lane1 = torch.zeros(b, dtype=fd, device=self.device)
        self.has_ramp_insertion = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.has_lane1_passage = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.last_service_ramp = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.has_service = torch.zeros(b, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ state
    def reset_slots(self, new_car_mask: Tensor) -> None:
        """Reset all per-car state of reused / identity-changed slots (``[B, K]``).

        World-level zipper timers are kept: they describe the merge area, not a car.
        After a reset the slot counts as absent, so its next appearance is a new car.
        """
        mask = torch.as_tensor(new_car_mask).to(self.device, dtype=torch.bool)
        if mask.shape != self.prev_present.shape:
            raise ValueError("new_car_mask must have shape [batch, k]")
        zero = torch.zeros_like(self.wait_s)
        self.prev_present = self.prev_present & ~mask
        self.prev_lane = torch.where(mask, torch.full_like(self.prev_lane, -1), self.prev_lane)
        self.prev_s = torch.where(mask, zero, self.prev_s)
        self.in_zone = self.in_zone & ~mask
        self.wait_s = torch.where(mask, zero, self.wait_s)
        self.foll_index = torch.where(mask, torch.full_like(self.foll_index, -1),
                                      self.foll_index)
        self.has_foll = self.has_foll & ~mask
        self.foll_ema1 = torch.where(mask, zero, self.foll_ema1)
        self.foll_ema3 = torch.where(mask, zero, self.foll_ema3)

    def state_dict(self) -> dict[str, Tensor]:
        """Detached copy of the whole tracker state."""
        return {name: getattr(self, name).detach().clone()
                for name in ("prev_present", "prev_lane", "prev_s", "in_zone", "wait_s",
                             "foll_index", "has_foll", "foll_ema1", "foll_ema3",
                             "t_since_ramp", "t_since_lane1", "has_ramp_insertion",
                             "has_lane1_passage", "last_service_ramp", "has_service")}

    def load_state_dict(self, d: dict) -> None:
        """Restore a state produced by :meth:`state_dict` (device/dtype adapted)."""
        for name in ("prev_present", "prev_lane", "prev_s", "in_zone", "wait_s",
                     "foll_index", "has_foll", "foll_ema1", "foll_ema3",
                     "t_since_ramp", "t_since_lane1", "has_ramp_insertion",
                     "has_lane1_passage", "last_service_ramp", "has_service"):
            if name not in d:
                raise KeyError(f"missing tracker state key: {name}")
            current = getattr(self, name)
            value = torch.as_tensor(d[name]).to(device=self.device, dtype=current.dtype)
            if value.shape != current.shape:
                raise ValueError(f"state {name} has shape {tuple(value.shape)}, "
                                 f"expected {tuple(current.shape)}")
            setattr(self, name, value.clone())

    def detach_(self) -> None:
        """Detach every state tensor in place (TBPTT chunk boundary)."""
        for name in ("prev_s", "wait_s", "foll_ema1", "foll_ema3",
                     "t_since_ramp", "t_since_lane1"):
            setattr(self, name, getattr(self, name).detach())

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move floating state (and adapt dtype) before a step on new tensors."""
        if device == self.device and dtype == self.dtype:
            return
        self.device = device
        self.dtype = dtype
        for name in ("prev_s", "wait_s", "foll_ema1", "foll_ema3",
                     "t_since_ramp", "t_since_lane1"):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        for name in ("prev_present", "prev_lane", "in_zone", "foll_index", "has_foll",
                     "has_ramp_insertion", "has_lane1_passage", "last_service_ramp",
                     "has_service"):
            setattr(self, name, getattr(self, name).to(device=device))

    # ------------------------------------------------------------------ step
    def step(self, s: Tensor, lane: Tensor, v: Tensor, a: Tensor, present: Tensor,
             is_cav: Tensor, length: Tensor) -> dict[str, Tensor]:
        """Advance one step and return every :data:`CONTEXT_FEATURES` value + valid masks.

        ``lane`` is the integer lane index (``-1`` / absent when unknown); ``is_cav`` is
        accepted for API symmetry because CAVs count as ordinary neighbours.
        """
        s_in = torch.as_tensor(s)
        self._to(s_in.device, s_in.dtype if s_in.is_floating_point() else self.dtype)
        dev, fd = self.device, self.dtype
        if s_in.shape != (self.batch, self.k):
            raise ValueError(f"step inputs must have shape [{self.batch}, {self.k}]")
        present = torch.as_tensor(present).to(device=dev, dtype=torch.bool)
        lane = torch.as_tensor(lane).to(device=dev, dtype=torch.long)
        s = torch.as_tensor(s).to(device=dev, dtype=fd)
        v = torch.as_tensor(v).to(device=dev, dtype=fd)
        a = torch.as_tensor(a).to(device=dev, dtype=fd)
        length = torch.as_tensor(length).to(device=dev, dtype=fd)
        _ = torch.as_tensor(is_cav)  # CAVs are ordinary neighbours; nothing to filter
        if self.batch == 0 or self.k == 0:
            return self._empty_outputs(dev, fd)

        s = torch.where(present, s, torch.zeros_like(s))
        v = torch.where(present, v, torch.zeros_like(v))
        a = torch.where(present, a, torch.zeros_like(a))
        length = torch.where(present, length, torch.zeros_like(length))
        lane = torch.where(present, lane, torch.full_like(lane, -1))
        ramp = present & (lane == RAMP_LANE)
        target = present & (lane == TARGET_LANE)

        # Pairwise forward / backward distances in metres (front bumpers).
        fwd = s.unsqueeze(-2) - s.unsqueeze(-1)        # [B, K, K]: s_j - s_i
        back = -fwd                                    # s_i - s_j
        target_j = target.unsqueeze(-2)
        ramp_j = ramp.unsqueeze(-2)

        # --- ramp queue ---------------------------------------------------------
        n_ramp_ahead = (ramp_j & (fwd > 0.0)).sum(-1).to(fd)
        n_ramp_ahead = torch.where(ramp, n_ramp_ahead, torch.zeros_like(n_ramp_ahead))

        lead_mask = ramp_j & (fwd > 0.0) & (fwd <= MAX_NEIGHBOUR_GAP_M)
        lead_gap_raw = torch.where(lead_mask, fwd - length.unsqueeze(-2),
                                   torch.full_like(fwd, float("inf")))
        lead_gap, lead_idx = lead_gap_raw.min(-1)
        has_lead = torch.isfinite(lead_gap) & ramp
        lead_gap = torch.where(has_lead, lead_gap.clamp_min(0.0), torch.zeros_like(lead_gap))
        lead_idx = torch.where(has_lead, lead_idx, torch.full_like(lead_idx, -1))
        safe_lead = lead_idx.clamp(0, max(self.k - 1, 0))
        lead_speed = v.gather(-1, safe_lead)
        ramp_lead_dv = torch.where(has_lead, lead_speed - v, torch.zeros_like(v))

        # --- lane-1 leader / follower pair (also used by the zipper features) ---
        l1_mask = target_j & (fwd > 0.0) & (fwd <= MAX_NEIGHBOUR_GAP_M)
        l1_gap_raw = torch.where(l1_mask, fwd - length.unsqueeze(-2),
                                 torch.full_like(fwd, float("inf")))
        _, l1_idx = l1_gap_raw.min(-1)
        has_l1 = torch.isfinite(l1_gap_raw.min(-1).values) & ramp
        l1_idx = torch.where(has_l1, l1_idx, torch.full_like(l1_idx, -1))

        foll_mask = target_j & (back >= 0.0) & (back <= MAX_NEIGHBOUR_GAP_M)
        foll_dist = torch.where(foll_mask, back, torch.full_like(back, -float("inf")))
        _, foll_idx = foll_dist.max(-1)
        has_f = torch.isfinite(foll_dist.max(-1).values) & ramp
        foll_idx = torch.where(has_f, foll_idx, torch.full_like(foll_idx, -1))
        safe_foll = foll_idx.clamp(0, max(self.k - 1, 0))
        foll_speed = v.gather(-1, safe_foll)
        foll_accel = a.gather(-1, safe_foll)

        # --- n_ramp_same_gap ----------------------------------------------------
        lead_key = torch.where(has_l1, l1_idx, torch.full_like(l1_idx, -1))
        foll_key = torch.where(has_f, safe_foll, torch.full_like(safe_foll, -1))
        same_pair = ((lead_key.unsqueeze(-1) == lead_key.unsqueeze(-2))
                     & (foll_key.unsqueeze(-1) == foll_key.unsqueeze(-2)))
        eye = torch.eye(self.k, dtype=torch.bool, device=dev).unsqueeze(0)
        other_ramp = ramp_j & ~eye
        has_pair = has_l1 | has_f
        n_ramp_same_gap = (same_pair & other_ramp).sum(-1).to(fd)
        n_ramp_same_gap = torch.where(ramp & has_pair, n_ramp_same_gap,
                                      torch.zeros_like(n_ramp_same_gap))

        # --- per-car zone / follower state -------------------------------------
        in_zone = ramp & (s >= MERGE_ZONE_START_M) & (s <= MERGE_ZONE_END_M)
        wait = torch.where(in_zone,
                           torch.where(self.in_zone, self.wait_s + self.dt,
                                       torch.zeros_like(self.wait_s)),
                           torch.zeros_like(self.wait_s))
        dist_end = torch.where(ramp, torch.full_like(s, MERGE_ZONE_END_M) - s,
                               torch.zeros_like(s))

        same_follower = self.has_foll & has_f & (self.foll_index == foll_idx)
        decay1 = 1.0 - math.exp(-self.dt / EMA_TAU_S[0])
        decay3 = 1.0 - math.exp(-self.dt / EMA_TAU_S[1])
        ema1 = torch.where(same_follower, self.foll_ema1 + (foll_accel - self.foll_ema1) * decay1,
                           foll_accel)
        ema3 = torch.where(same_follower, self.foll_ema3 + (foll_accel - self.foll_ema3) * decay3,
                           foll_accel)
        ema1 = torch.where(has_f, ema1, torch.zeros_like(ema1))
        ema3 = torch.where(has_f, ema3, torch.zeros_like(ema3))
        foll_dv = torch.where(has_f, foll_speed - v, torch.zeros_like(v))

        # --- zipper events at the merge area -----------------------------------
        insertion = (present & self.prev_present & (self.prev_lane == RAMP_LANE)
                     & (lane == TARGET_LANE))
        passage = (present & self.prev_present & (self.prev_lane == TARGET_LANE)
                   & (lane == TARGET_LANE) & (self.prev_s < MERGE_ZONE_END_M)
                   & (s >= MERGE_ZONE_END_M))
        ins_any = insertion.any(-1)
        pass_any = passage.any(-1)
        t_ramp = torch.where(ins_any, torch.zeros_like(self.t_since_ramp),
                             self.t_since_ramp + self.dt)
        t_lane1 = torch.where(pass_any, torch.zeros_like(self.t_since_lane1),
                              self.t_since_lane1 + self.dt)
        has_ins = self.has_ramp_insertion | ins_any
        has_pass = self.has_lane1_passage | pass_any
        service = torch.where(ins_any, torch.ones_like(self.last_service_ramp),
                              torch.where(pass_any, torch.zeros_like(self.last_service_ramp),
                                          self.last_service_ramp))
        has_service = self.has_service | ins_any | pass_any

        # --- outputs ------------------------------------------------------------
        zeros = torch.zeros_like(v)
        foll_valid = has_f
        zipper_valid = present & has_ins.unsqueeze(-1)
        pass_valid = present & has_pass.unsqueeze(-1)
        service_valid = present & has_service.unsqueeze(-1)
        out = {
            "n_ramp_ahead": n_ramp_ahead,
            "ramp_lead_gap_m": lead_gap,
            "ramp_lead_dv_mps": ramp_lead_dv,
            "dist_to_lane_end_m": dist_end,
            "wait_s": wait,
            "t_since_ramp_insertion_s": torch.where(zipper_valid, t_ramp.unsqueeze(-1), zeros),
            "t_since_lane1_passage_s": torch.where(pass_valid, t_lane1.unsqueeze(-1), zeros),
            "last_service_was_ramp": torch.where(
                service_valid, service.to(fd).unsqueeze(-1), zeros),
            "foll_speed_mps": torch.where(foll_valid, foll_speed, zeros),
            "foll_accel_now": torch.where(foll_valid, foll_accel, zeros),
            "foll_accel_ema1_s": ema1,
            "foll_accel_ema3_s": ema3,
            "foll_dv_mps": foll_dv,
            "n_ramp_same_gap": n_ramp_same_gap,
            "ramp_lead_gap_m_valid": has_lead,
            "ramp_lead_dv_mps_valid": has_lead,
            "wait_s_valid": in_zone,
            "t_since_ramp_insertion_s_valid": zipper_valid,
            "t_since_lane1_passage_s_valid": pass_valid,
            "last_service_was_ramp_valid": service_valid,
            "foll_speed_mps_valid": foll_valid,
            "foll_accel_now_valid": foll_valid,
            "foll_accel_ema1_s_valid": foll_valid,
            "foll_accel_ema3_s_valid": foll_valid,
            "foll_dv_mps_valid": foll_valid,
        }

        # --- commit state -------------------------------------------------------
        self.prev_present = present
        self.prev_lane = lane
        self.prev_s = s
        self.in_zone = in_zone
        self.wait_s = wait
        self.foll_index = torch.where(has_f, safe_foll, torch.full_like(safe_foll, -1))
        self.has_foll = has_f
        self.foll_ema1 = ema1
        self.foll_ema3 = ema3
        self.t_since_ramp = t_ramp
        self.t_since_lane1 = t_lane1
        self.has_ramp_insertion = has_ins
        self.has_lane1_passage = has_pass
        self.last_service_ramp = service
        self.has_service = has_service
        return out

    def _empty_outputs(self, dev: torch.device, fd: torch.dtype) -> dict[str, Tensor]:
        shape = (self.batch, self.k)
        out = {name: torch.zeros(shape, dtype=fd, device=dev) for name in CONTEXT_FEATURES}
        out.update({f"{name}_valid": torch.zeros(shape, dtype=torch.bool, device=dev)
                    for name in VALID_MASKED})
        return out


def recorded_context(veh_kin: Tensor, present: Tensor, is_cav: Tensor, *,
                     dt: float = 0.5) -> dict[str, Tensor]:
    """Run :class:`MergeContextTracker` over a recorded sequence.

    ``veh_kin`` is ``[B, N, K, 7]`` (``s``, lane one-hot x3, ``v``, ``a``, ``length``
    as in ``UnrollTargets.states``); ``present`` / ``is_cav`` are ``[B, N, K]``.
    Returns ``{name: [B, N, K]}`` including the valid masks.  The result is identical
    to stepping one fresh tracker over the same sequence.
    """
    kin = torch.as_tensor(veh_kin)
    present = torch.as_tensor(present)
    is_cav = torch.as_tensor(is_cav)
    if kin.ndim != 4 or kin.shape[-1] != 7:
        raise ValueError("veh_kin must have shape [B, N, K, 7]")
    if present.shape != kin.shape[:3] or is_cav.shape != kin.shape[:3]:
        raise ValueError("present / is_cav must have shape [B, N, K]")
    b, n, k = kin.shape[:3]
    dtype = kin.dtype if kin.is_floating_point() else None
    tracker = MergeContextTracker(b, k, dt=dt, device=kin.device, dtype=dtype)
    onehot = kin[..., 1:4]
    lanes = onehot.argmax(-1)
    lanes = torch.where(onehot.sum(-1) > 0, lanes, torch.full_like(lanes, -1))
    pieces: dict[str, list[Tensor]] = {}
    for t in range(n):
        out = tracker.step(kin[:, t, :, 0], lanes[:, t], kin[:, t, :, 4], kin[:, t, :, 5],
                           present[:, t], is_cav[:, t], kin[:, t, :, 6])
        for name, value in out.items():
            pieces.setdefault(name, []).append(value)
    result = {name: torch.stack(values, 1) for name, values in pieces.items()}
    if n == 0:
        fd = dtype or torch.get_default_dtype()
        result = {name: torch.zeros(b, 0, k, dtype=fd, device=kin.device)
                  for name in CONTEXT_FEATURES}
        result.update({f"{name}_valid": torch.zeros(b, 0, k, dtype=torch.bool,
                                                    device=kin.device)
                       for name in VALID_MASKED})
    return result


__all__ = ["CONTEXT_FEATURES", "VALID_MASKED", "RAMP_LANE", "TARGET_LANE",
           "MERGE_ZONE_START_M", "MERGE_ZONE_END_M", "MAX_NEIGHBOUR_GAP_M", "EMA_TAU_S",
           "MergeContextTracker", "recorded_context"]
