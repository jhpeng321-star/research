"""Stateful per-vehicle leader-change features (R9K, pure torch, no parameters).

The supplied brief reports excess WM braking after lane-1 insertions. This module
exposes leader-identity history and geometry without prescribing an acceleration
response. A measured missing-history effect is required before enabling it. The
per-slot tracker is exposed the same way as
:mod:`wms.model.merge_context` (``MergeContextTracker``), for the coordinator to
append to the residual-net inputs behind ``model.leader_change_features`` (default
false; NO wms.py edit is made in this branch - R9_K.md documents the exact hooks).

Every feature is attached to a slot ``[B, K]`` and computed from PAST observations
only (the previous and current step), so it is causal at every decision.  No
behavioural threshold or reaction rule is encoded; the consumer decides.

Change types (priority order, first match wins):

* ``own_lane_change`` - the slot itself changed lane at this step, so the new
  leader is simply the leader in the new lane;
* ``cut_in`` - the NEW leader changed into this slot's lane at this step (came
  from an adjacent lane between me and my previous leader);
* ``leader_exit`` - the PREVIOUS leader left the lane / vanished this step (the
  next car ahead becomes the leader);
* unflagged re-association - none of the above (e.g. the previous leader left the
  association range): the episode clock runs but no ``changed_*`` flag is set.

Frame and units follow ``UnrollTargets.states``: ``s`` front-bumper metres, ``v``
m/s, ``a`` m/s^2, ``length`` metres; lanes 0/1 mainline, 2 ramp (``-1`` absent).
Leader association: nearest same-lane vehicle ahead within
:data:`MAX_NEIGHBOUR_GAP_M` (the recorded-scene geometry constant shared with
:mod:`wms.model.merge_core` / :mod:`wms.model.merge_context`).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

#: Lane indices (same convention as ``merge_core`` / ``merge_context``).
RAMP_LANE = 2
TARGET_LANE = 1
#: Same-lane leader association range (geometry constant, not behavioural).
MAX_NEIGHBOUR_GAP_M = 150.0
#: Episode clock cap (seconds); purely a feature-saturation bound, not a rule.
T_SINCE_CHANGE_MAX_S = 30.0

#: Ordered leader-change feature names (units in brackets).
LEADER_CHANGE_FEATURES: tuple[str, ...] = (
    "has_leader",             # 1 when a same-lane leader exists now
    "t_since_change_s",       # [s] since the leader identity last changed
    "gap_now_m",              # [m] bumper gap to the current leader
    "closing_now_mps",        # [m/s] v_leader - v_own now
    "gap_at_change_m",        # [m] leader gap recorded at the change step
    "closing_at_change_mps",  # [m/s] closing speed recorded at the change step
    "prev_leader_gap_m",      # [m] last observed gap to the displaced leader
    "changed_cut_in",         # 1 while the current episode is an adjacent-lane cut-in
    "changed_own_lane",       # 1 while the episode followed the slot's own lane change
    "changed_leader_exit",    # 1 while the episode followed the leader leaving
)

#: Features that can be missing; a matching ``<name>_valid`` bool mask is emitted.
VALID_MASKED: tuple[str, ...] = (
    "t_since_change_s",
    "gap_at_change_m",
    "closing_at_change_mps",
    "prev_leader_gap_m",
)

LEADER_CHANGE_INPUTS = LEADER_CHANGE_FEATURES + tuple(f"{name}_valid" for name in VALID_MASKED)

#: Change-type ids of :attr:`LeaderChangeTracker.CHANGE_TYPES`.
CHANGE_CUT_IN = 1
CHANGE_OWN_LANE = 2
CHANGE_LEADER_EXIT = 3

#: Documented normalisation of :func:`feature_matrix` (module keeps raw values in
#: the per-name outputs; the matrix divides by these scales like the courtesy
#: channels normalise by 20 m/s etc.).  Learnable adapters sit AFTER this scale.
LEADER_CHANGE_SCALES: dict[str, float] = {
    "t_since_change_s": 10.0,
    "gap_now_m": 50.0,
    "closing_now_mps": 10.0,
    "gap_at_change_m": 50.0,
    "closing_at_change_mps": 10.0,
    "prev_leader_gap_m": 50.0,
}


def leader_geometry(s: Tensor, lane: Tensor, v: Tensor, present: Tensor,
                    length: Tensor) -> dict[str, Tensor]:
    """Same-lane leader index / gap / speed of every slot (one step, no state).

    Returns ``index`` (-1 when absent), ``gap`` (m, 0 when absent), ``speed``
    (m/s, 0 when absent) and ``has`` (bool), all ``[B, K]``.  Absent slots never
    lead and never have a leader.
    """
    present = present.bool()
    safe_lane = torch.where(present, lane, torch.full_like(lane, -1))
    # fwd[b, i, j] = s_j - s_i (front bumper, metres).
    fwd = s.unsqueeze(-2) - s.unsqueeze(-1)
    lead_ok = (present.unsqueeze(-2)                    # candidate j present
               & (safe_lane.unsqueeze(-2) == safe_lane.unsqueeze(-1))
               & (safe_lane.unsqueeze(-1) >= 0)         # i on a real lane
               & (fwd > 0.0) & (fwd - length.unsqueeze(-2) <= float(MAX_NEIGHBOUR_GAP_M)))
    gap_raw = torch.where(lead_ok, fwd - length.unsqueeze(-2),
                          torch.full_like(fwd, float("inf")))
    # Association is by front position, not bumper gap (lengths may differ).
    _, index = torch.where(lead_ok, fwd, torch.full_like(fwd, float("inf"))).min(-1)
    gap = gap_raw.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    has = torch.isfinite(gap) & present
    gap = torch.where(has, gap.clamp_min(0.0), torch.zeros_like(gap))
    index = torch.where(has, index, torch.full_like(index, -1))
    safe_index = index.clamp(0, max(int(s.shape[-1]) - 1, 0))
    speed = torch.where(has, v.gather(-1, safe_index), torch.zeros_like(v))
    return {"index": index, "gap": gap, "speed": speed, "has": has}


def feature_matrix(out: dict[str, Tensor]) -> Tensor:
    """Normalised ``[B, K, 14]`` values then :data:`VALID_MASKED` validity bits.

    Binary flags (``has_leader``, ``changed_*``) pass through as 0/1; physical
    values are divided by :data:`LEADER_CHANGE_SCALES`; missing values (valid
    mask false) are already zero in ``out``.
    """
    columns = []
    for name in LEADER_CHANGE_FEATURES:
        value = out[name].float()
        if name in VALID_MASKED:
            value = torch.where(out[f"{name}_valid"], value, torch.zeros_like(value))
        if name in LEADER_CHANGE_SCALES:
            value = value / float(LEADER_CHANGE_SCALES[name])
        columns.append(value)
    # Unknown history and a genuine zero-age/zero-gap change must be distinguishable.
    columns.extend(out[f"{name}_valid"].float() for name in VALID_MASKED)
    return torch.stack(columns, dim=-1)


class LeaderChangeTracker:
    """Per-slot leader-identity change history, one :meth:`step` per decision step.

    All inputs are ``[B, K]``; ``step`` returns every
    :data:`LEADER_CHANGE_FEATURES` value plus the ``<name>_valid`` masks of
    :data:`VALID_MASKED`.  The state is plain tensors: :meth:`state_dict` /
    :meth:`load_state_dict` carry it across truncated-BPTT chunks (with
    :meth:`detach_` at boundaries) and :meth:`reset_slots` clears reused slots
    (slot reuse / new entrants start with no leader history).
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

        fd, b, kk = self.dtype, self.batch, self.k
        self.prev_present = torch.zeros(b, kk, dtype=torch.bool, device=self.device)
        self.prev_lane = torch.full((b, kk), -1, dtype=torch.long, device=self.device)
        self.leader_index = torch.full((b, kk), -1, dtype=torch.long, device=self.device)
        self.prev_gap = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.t_since = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.gap_at = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.closing_at = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.prev_leader_gap = torch.zeros(b, kk, dtype=fd, device=self.device)
        self.change_type = torch.zeros(b, kk, dtype=torch.long, device=self.device)
        self.has_change = torch.zeros(b, kk, dtype=torch.bool, device=self.device)
        self.prev_gap_valid = torch.zeros(b, kk, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ state
    _STATE_NAMES = ("prev_present", "prev_lane", "leader_index", "prev_gap",
                    "t_since", "gap_at", "closing_at", "prev_leader_gap",
                    "change_type", "has_change", "prev_gap_valid")

    def reset_slots(self, new_car_mask: Tensor) -> None:
        """Clear all per-car state of reused / identity-changed slots ``[B, K]``."""
        mask = torch.as_tensor(new_car_mask).to(self.device, dtype=torch.bool)
        if mask.shape != self.prev_present.shape:
            raise ValueError("new_car_mask must have shape [batch, k]")
        # An immediately replaced leader is a different identity even if its slot
        # index stays the same. Invalidate dependent history too.
        safe = self.leader_index.clamp_min(0)
        dependent = (self.leader_index >= 0) & mask.gather(-1, safe) if self.k else mask
        affected = mask | dependent
        self.prev_present = self.prev_present & ~mask
        self.prev_lane = torch.where(mask, torch.full_like(self.prev_lane, -1),
                                     self.prev_lane)
        self.leader_index = torch.where(affected, torch.full_like(self.leader_index, -1),
                                        self.leader_index)
        zero = torch.zeros_like(self.prev_gap)
        self.prev_gap = torch.where(mask, zero, self.prev_gap)
        self.t_since = torch.where(mask, zero, self.t_since)
        self.gap_at = torch.where(mask, zero, self.gap_at)
        self.closing_at = torch.where(mask, zero, self.closing_at)
        self.prev_leader_gap = torch.where(mask, zero, self.prev_leader_gap)
        self.change_type = torch.where(mask, torch.zeros_like(self.change_type),
                                       self.change_type)
        self.has_change = self.has_change & ~affected
        self.prev_gap_valid = self.prev_gap_valid & ~affected

    def state_dict(self) -> dict[str, Tensor]:
        """Detached copy of the whole tracker state."""
        return {name: getattr(self, name).detach().clone()
                for name in self._STATE_NAMES}

    def load_state_dict(self, d: dict) -> None:
        """Restore a state produced by :meth:`state_dict` (device/dtype adapted)."""
        for name in self._STATE_NAMES:
            if name not in d:
                raise KeyError(f"missing tracker state key: {name}")
            current = getattr(self, name)
            value = torch.as_tensor(d[name]).to(device=self.device, dtype=current.dtype)
            if value.shape != current.shape:
                raise ValueError(f"state {name} has shape {tuple(value.shape)}, "
                                 f"expected {tuple(current.shape)}")
            setattr(self, name, value.clone())

    def detach_(self) -> None:
        """Detach every floating state tensor in place (TBPTT chunk boundary)."""
        for name in ("prev_gap", "t_since", "gap_at", "closing_at", "prev_leader_gap"):
            setattr(self, name, getattr(self, name).detach())

    def _to(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move state before a step on tensors of another device/dtype."""
        if device == self.device and dtype == self.dtype:
            return
        self.device = device
        self.dtype = dtype
        for name in ("prev_gap", "t_since", "gap_at", "closing_at", "prev_leader_gap"):
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        for name in ("prev_present", "prev_lane", "leader_index", "change_type",
                      "has_change", "prev_gap_valid"):
            setattr(self, name, getattr(self, name).to(device=device))

    # ------------------------------------------------------------------ step
    def step(self, s: Tensor, lane: Tensor, v: Tensor, present: Tensor,
             length: Tensor) -> dict[str, Tensor]:
        """Advance one step; returns every feature value plus valid masks.

        ``lane`` is the integer lane index (``-1`` / absent when unknown);
        ``is_cav`` is deliberately NOT an input: CAVs are ordinary neighbours.
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
        length = torch.as_tensor(length).to(device=dev, dtype=fd)
        if any(value.shape != (self.batch, self.k) for value in (present, lane, v, length)):
            raise ValueError("all step inputs must have shape [batch,k]")
        if self.batch == 0 or self.k == 0:
            return self._empty_outputs(dev, fd)

        present = present & (lane >= 0)
        s = torch.where(present, s, torch.zeros_like(s))
        v = torch.where(present, v, torch.zeros_like(v))
        length = torch.where(present, length, torch.zeros_like(length))
        lane = torch.where(present, lane, torch.full_like(lane, -1))

        geo = leader_geometry(s, lane, v, present, length)
        index, gap, lead_speed, has = geo["index"], geo["gap"], geo["speed"], geo["has"]
        closing = torch.where(has, lead_speed - v, torch.zeros_like(v))

        cont = present & self.prev_present           # observed at both ends
        old_index = self.leader_index
        # A leader identity change on a continuous observation of this slot.
        own_lane_change = cont & (lane != self.prev_lane)
        changed = cont & ((index != old_index) | own_lane_change)

        safe_new = index.clamp(0, max(self.k - 1, 0))
        new_came = torch.zeros_like(changed)
        if changed.any():
            # The new leader switched INTO this slot's lane at this step (cut-in).
            new_lane_now = lane.gather(-1, safe_new)
            new_lane_prev = self.prev_lane.gather(-1, safe_new)
            new_came = (changed & has & self.prev_present.gather(-1, safe_new)
                        & (new_lane_now == lane) & (new_lane_prev >= 0)
                        & (new_lane_prev != lane))
        safe_old = old_index.clamp(0, max(self.k - 1, 0))
        old_gone = torch.zeros_like(changed)
        if changed.any():
            # The previous leader left this slot's lane / vanished at this step.
            old_lane_now = torch.where(old_index >= 0, lane.gather(-1, safe_old),
                                       torch.full_like(lane, -1))
            old_lane_prev = torch.where(old_index >= 0,
                                        self.prev_lane.gather(-1, safe_old),
                                        torch.full_like(lane, -1))
            old_present = torch.where(old_index >= 0, present.gather(-1, safe_old),
                                      torch.zeros_like(present))
            old_gone = changed & (old_index >= 0) & (
                ~old_present | (old_lane_now != old_lane_prev))

        change_type = torch.zeros_like(self.change_type)
        change_type = torch.where(own_lane_change & changed,
                                  torch.full_like(change_type, CHANGE_OWN_LANE),
                                  change_type)
        change_type = torch.where(
            (change_type == 0) & new_came, torch.full_like(change_type, CHANGE_CUT_IN),
            change_type)
        change_type = torch.where(
            (change_type == 0) & old_gone & ~new_came,
            torch.full_like(change_type, CHANGE_LEADER_EXIT), change_type)
        # Every observed re-association has known age, even if its cause is unknown.
        episode = changed

        # Geometry recorded at the change step (current values when a leader
        # exists; the last carried gap when the change removed the leader).
        gap_now_or_old = torch.where(has, gap,
                                     torch.where(old_index >= 0, self.prev_gap, gap))
        gap_at = torch.where(changed, gap_now_or_old, self.gap_at)
        closing_at = torch.where(changed, closing, self.closing_at)
        prev_leader_gap = torch.where(changed,
                                      torch.where(old_index >= 0, self.prev_gap,
                                                  torch.zeros_like(self.prev_gap)),
                                      self.prev_leader_gap)
        t_since = torch.where(changed, torch.zeros_like(self.t_since),
                              self.t_since + self.dt)
        t_since = torch.clamp(t_since, max=float(T_SINCE_CHANGE_MAX_S))
        # An unobserved gap (absent slot or first sight) restarts the episode
        # clock: the next observation is a new car as far as history goes.
        t_since = torch.where(cont, t_since, torch.zeros_like(t_since))
        has_episode = cont & (self.has_change | episode)
        gap_at = torch.where(has_episode, gap_at, torch.zeros_like(gap_at))
        closing_at = torch.where(has_episode, closing_at, torch.zeros_like(closing_at))
        prev_leader_gap = torch.where(has_episode, prev_leader_gap,
                                      torch.zeros_like(prev_leader_gap))
        t_since = torch.where(has_episode, t_since, torch.zeros_like(t_since))
        change_type = torch.where(changed, change_type, self.change_type)
        change_type = torch.where(has_episode, change_type,
                                   torch.zeros_like(change_type))
        prev_gap_valid = torch.where(changed, old_index >= 0, self.prev_gap_valid) & has_episode

        out = {
            "has_leader": has.to(fd),
            "t_since_change_s": t_since,
            "gap_now_m": gap,
            "closing_now_mps": closing,
            "gap_at_change_m": gap_at,
            "closing_at_change_mps": closing_at,
            "prev_leader_gap_m": prev_leader_gap,
            "changed_cut_in": (change_type == CHANGE_CUT_IN).to(fd),
            "changed_own_lane": (change_type == CHANGE_OWN_LANE).to(fd),
            "changed_leader_exit": (change_type == CHANGE_LEADER_EXIT).to(fd),
            "t_since_change_s_valid": has_episode,
            "gap_at_change_m_valid": has_episode & has,
            "closing_at_change_mps_valid": has_episode & has,
            "prev_leader_gap_m_valid": prev_gap_valid,
        }
        out = {name: torch.where(present, value, torch.zeros_like(value))
               for name, value in out.items()}

        # --- commit state ------------------------------------------------------
        self.prev_present = present
        self.prev_lane = lane
        self.leader_index = index
        self.prev_gap = gap
        self.t_since = t_since
        self.gap_at = gap_at
        self.closing_at = closing_at
        self.prev_leader_gap = prev_leader_gap
        self.change_type = change_type
        self.has_change = has_episode
        self.prev_gap_valid = prev_gap_valid
        return out

    def _empty_outputs(self, dev: torch.device, fd: torch.dtype) -> dict[str, Tensor]:
        shape = (self.batch, self.k)
        out = {name: torch.zeros(shape, dtype=fd, device=dev)
               for name in LEADER_CHANGE_FEATURES}
        out.update({f"{name}_valid": torch.zeros(shape, dtype=torch.bool, device=dev)
                    for name in VALID_MASKED})
        return out


def recorded_leader_change(veh_kin: Tensor, present: Tensor, *,
                            dt: float = 0.5, reset: Tensor | None = None,
                            tracker: LeaderChangeTracker | None = None) -> dict[str, Tensor]:
    """Run :class:`LeaderChangeTracker` over a recorded sequence.

    ``veh_kin`` is ``[B, N, K, 7]`` (``s``, lane one-hot x3, ``v``, ``a``,
    ``length`` as in ``UnrollTargets.states``); ``present`` is ``[B, N, K]``.
    Returns ``{name: [B, N, K]}`` including the valid masks - identical to
    stepping the same tracker over the sequence. With no tracker, window-start
    history is unknown. Pass a tracker across chunks, without re-playing the last
    row of the preceding chunk. ``reset`` marks explicit identity replacements.
    """
    kin = torch.as_tensor(veh_kin)
    present = torch.as_tensor(present)
    if kin.ndim != 4 or kin.shape[-1] != 7:
        raise ValueError("veh_kin must have shape [B, N, K, 7]")
    if present.shape != kin.shape[:3]:
        raise ValueError("present must have shape [B, N, K]")
    b, n, k = kin.shape[:3]
    dtype = kin.dtype if kin.is_floating_point() else None
    if tracker is None:
        tracker = LeaderChangeTracker(b, k, dt=dt, device=kin.device, dtype=dtype)
    elif (tracker.batch, tracker.k, tracker.dt) != (b, k, float(dt)):
        raise ValueError("tracker shape/dt does not match recorded sequence")
    if reset is not None and reset.shape != present.shape:
        raise ValueError("reset must have shape [B,N,K]")
    onehot = kin[..., 1:4]
    lanes = onehot.argmax(-1)
    lanes = torch.where(onehot.sum(-1) > 0, lanes, torch.full_like(lanes, -1))
    pieces: dict[str, list[Tensor]] = {}
    for t in range(n):
        if reset is not None:
            tracker.reset_slots(reset[:, t])
        out = tracker.step(kin[:, t, :, 0], lanes[:, t], kin[:, t, :, 4],
                           present[:, t], kin[:, t, :, 6])
        for name, value in out.items():
            pieces.setdefault(name, []).append(value)
    result = {name: torch.stack(values, 1) for name, values in pieces.items()}
    if n == 0:
        fd = dtype or torch.get_default_dtype()
        result = {name: torch.zeros(b, 0, k, dtype=fd, device=kin.device)
                  for name in LEADER_CHANGE_FEATURES}
        result.update({f"{name}_valid": torch.zeros(b, 0, k, dtype=torch.bool,
                                                    device=kin.device)
                       for name in VALID_MASKED})
    return result


__all__ = ["LEADER_CHANGE_FEATURES", "VALID_MASKED", "LEADER_CHANGE_SCALES",
           "RAMP_LANE", "TARGET_LANE", "MAX_NEIGHBOUR_GAP_M", "T_SINCE_CHANGE_MAX_S",
           "CHANGE_CUT_IN", "CHANGE_OWN_LANE", "CHANGE_LEADER_EXIT",
           "leader_geometry", "feature_matrix", "LeaderChangeTracker",
            "recorded_leader_change", "LEADER_CHANGE_INPUTS"]
