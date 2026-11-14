"""Fixed-length per-vehicle motion-history tensors (WM-S T3, P2 input).

The learned correction reads, for every vehicle alive at ``t_now``, a fixed
window of that vehicle's own recent motion: ``[K, H, F]`` with ``H = 10``
(the last 5 s at 0.5 s) and ``F = 7`` fields

    s_down_m, v_mps, a_mps2, lane, gap_m, lead_dv_mps, present

``gap_m`` is the same-lane bumper gap to the leader (clipped to
:data:`HISTORY_GAP_MAX_M`; ``present=0`` marks "no observation"); ``lead_dv_mps``
is the ego-minus-leader closing speed in the IDM/rollout convention
(``rollout2.baseline`` uses ``dv = v_ego - v_lead``).

Two builders share :func:`build_history`:

* :func:`pack_history_tensor` -- offline hook over a ``train.dataset``
  ``MemmapCorpus`` pack (the same ``veh_step``/``veh_mask`` arrays the residual
  reads), using only ticks ``<= t_now``;
* :class:`RollingHistoryBuffer` -- online hook fed the per-step rollout state,
  so the same tensor can be produced at inference.

Leakage: every builder indexes only ticks at or before ``t_now``; future rows
are never touched (see ``driver/tests/test_history.py``).
"""
from __future__ import annotations

import numpy as np

DT_S = 0.5
HISTORY_H = 10
HISTORY_FIELDS = ("s_down_m", "v_mps", "a_mps2", "lane", "gap_m",
                  "lead_dv_mps", "present")
HISTORY_F = len(HISTORY_FIELDS)
NOMINAL_LENGTH_M = 5.0
HISTORY_GAP_MAX_M = 1000.0


def _kin_from_veh_step(veh_step):
    """``veh_step [..., VEH_F]`` -> ``(s, lane, v, a)`` (lane in {-1,0,1,2})."""
    veh = np.asarray(veh_step, dtype=np.float64)
    s = veh[..., 0]
    onehot = veh[..., 1:4]
    lane = np.where(onehot.sum(axis=-1) > 0, onehot.argmax(axis=-1), -1)
    return s, lane.astype(np.int64), veh[..., 4], veh[..., 5]


def tick_gap_lead_dv(s, lane, v, present, length=NOMINAL_LENGTH_M):
    """Same-lane bumper gap + closing speed for one tick.

    Returns ``(gap [K], lead_dv [K])``: ``gap = s_lead - s_ego - length_lead``
    (``inf`` when no leader), ``lead_dv = v_ego - v_lead`` (0 when no leader).
    """
    s = np.asarray(s, dtype=np.float64)
    lane = np.asarray(lane)
    v = np.asarray(v, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    length = np.asarray(length, dtype=np.float64)
    if length.ndim == 0:
        length = np.full(s.shape, float(length))
    k = s.shape[0]
    gap = np.full(k, np.inf, dtype=np.float64)
    lead_dv = np.zeros(k, dtype=np.float64)
    for li in np.unique(lane[present]):
        if int(li) < 0:
            continue
        rows = np.where(present & (lane == li))[0]
        order = rows[np.argsort(s[rows], kind="stable")]
        for j in range(len(order) - 1):
            ego = int(order[j])
            lead = int(order[j + 1])
            gap[ego] = s[lead] - s[ego] - length[lead]
            lead_dv[ego] = v[ego] - v[lead]
    return gap, lead_dv


def build_history(s_seq, lane_seq, v_seq, a_seq, present_seq, slot_ids_seq, *,
                  H: int = HISTORY_H, length=NOMINAL_LENGTH_M,
                  gap_max_m: float = HISTORY_GAP_MAX_M) -> np.ndarray:
    """Assemble ``[K, H, F]`` histories for the vehicles at the last tick.

    ``*_seq`` are ``[T, K]`` oldest-first (last row is ``t_now``); ``slot_ids_seq
    [T, K]`` is the per-tick identity (``-1`` = empty) so a reused slot never
    mixes two vehicles.  Output row ``k`` follows the identity at ``[t_now, k]``
    through the window; ticks before the window or without that identity are
    zero with ``present = 0``.
    """
    s_seq = np.asarray(s_seq, dtype=np.float64)
    v_seq = np.asarray(v_seq, dtype=np.float64)
    a_seq = np.asarray(a_seq, dtype=np.float64)
    lane_seq = np.asarray(lane_seq, dtype=np.int64)
    present_seq = np.asarray(present_seq, dtype=bool)
    slot_ids_seq = np.asarray(slot_ids_seq, dtype=np.int64)
    t, k = v_seq.shape
    out = np.zeros((k, int(H), HISTORY_F), dtype=np.float64)
    if t == 0 or k == 0:
        return out
    t_now = t - 1
    present_now = present_seq[t_now]
    ids_now = slot_ids_seq[t_now]

    id_to_slot = []
    for ti in range(t):
        mapping = {}
        pres = present_seq[ti]
        ids = slot_ids_seq[ti]
        for slot in range(k):
            if pres[slot] and ids[slot] >= 0:
                mapping[int(ids[slot])] = slot
        id_to_slot.append(mapping)

    for t_out in range(int(H)):
        ti = t - int(H) + t_out
        if ti < 0:
            continue
        gap, lead_dv = tick_gap_lead_dv(
            s_seq[ti], lane_seq[ti], v_seq[ti], present_seq[ti], length)
        gap = np.minimum(gap, float(gap_max_m))
        mapping = id_to_slot[ti]
        for slot in range(k):
            if not (present_now[slot] and ids_now[slot] >= 0):
                continue
            src = mapping.get(int(ids_now[slot]))
            if src is None:
                continue
            out[slot, t_out, 0] = s_seq[ti, src]
            out[slot, t_out, 1] = v_seq[ti, src]
            out[slot, t_out, 2] = a_seq[ti, src]
            out[slot, t_out, 3] = float(lane_seq[ti, src])
            out[slot, t_out, 4] = gap[src]
            out[slot, t_out, 5] = lead_dv[src]
            out[slot, t_out, 6] = 1.0
    return out


def history_from_veh_step(veh_step_seq, mask_seq, slot_ids_seq, *,
                          H: int = HISTORY_H) -> np.ndarray:
    """``[T,K,VEH_F]`` veh_step + ``[T,K]`` mask/identity -> ``[K,H,F]``."""
    veh_step_seq = np.asarray(veh_step_seq)
    s, lane, v, a = _kin_from_veh_step(veh_step_seq)
    return build_history(s, lane, v, a, mask_seq, slot_ids_seq, H=H)


def pack_history_tensor(dataset, position: int, *, H: int = HISTORY_H,
                        slot_ids=None, length=NOMINAL_LENGTH_M) -> np.ndarray:
    """Dataset hook: ``[K, H, F]`` history for ``MemmapCorpus`` item ``position``.

    Reads the packed ``veh_step``/``veh_mask`` at ticks ``[tick-H+1 .. tick]``
    only.  ``slot_ids`` (``[N_ticks, K]``) should be supplied from the TRAIN
    identity sidecar so reused slots are not mixed; when omitted the slot index
    is treated as the identity across the window (documented fallback for packs
    without an identity table).
    """
    order = dataset._order[position]
    item = dataset._items[order]
    tick = int(item["tick"])
    veh = dataset._array("veh_step")
    mask = dataset._array("veh_mask")
    lo = max(0, tick - int(H) + 1)
    idx = np.arange(lo, tick + 1)
    veh_seq = np.asarray(veh[idx])
    mask_seq = np.asarray(mask[idx]).astype(bool)
    k = int(veh_seq.shape[1])
    if slot_ids is None:
        base = np.tile(np.arange(k, dtype=np.int64), (len(idx), 1))
        ids = np.where(mask_seq, base, -1)
    else:
        ids = np.asarray(slot_ids)[idx].astype(np.int64)
        ids = np.where(mask_seq, ids, -1)
    return history_from_veh_step(veh_seq, mask_seq, ids, H=H)


class RollingHistoryBuffer:
    """Inference hook: push the rollout state each step, read ``[K, H, F]``."""

    def __init__(self, num_slots: int, *, H: int = HISTORY_H, dt_s: float = DT_S,
                 length=NOMINAL_LENGTH_M):
        self.num_slots = int(num_slots)
        self.H = int(H)
        self.dt_s = float(dt_s)
        self.length = length
        self._rows: list = []
        self._ids: list = []

    def reset(self) -> None:
        self._rows = []
        self._ids = []

    @property
    def n_filled(self) -> int:
        return len(self._rows)

    def push(self, s, v, a, lane, present, slot_id) -> None:
        """Append one observed tick (arrays ``[K]``; ``slot_id`` int ``[K]``)."""
        s = np.asarray(s, dtype=np.float64)
        if s.shape != (self.num_slots,):
            raise ValueError(
                f"state arrays must be [{self.num_slots}], got {s.shape}")
        self._rows.append((
            np.array(s, dtype=np.float64, copy=True),
            np.asarray(lane, dtype=np.int64).copy(),
            np.asarray(v, dtype=np.float64).copy(),
            np.asarray(a, dtype=np.float64).copy(),
            np.asarray(present, dtype=bool).copy()))
        self._ids.append(np.asarray(slot_id, dtype=np.int64).copy())
        if len(self._rows) > self.H:
            self._rows.pop(0)
            self._ids.pop(0)

    def tensor(self) -> np.ndarray:
        """``[K, H, F]`` over the buffered ticks (zero-padded at the front)."""
        if not self._rows:
            return np.zeros((self.num_slots, self.H, HISTORY_F), dtype=np.float64)
        s = np.stack([r[0] for r in self._rows])
        lane = np.stack([r[1] for r in self._rows])
        v = np.stack([r[2] for r in self._rows])
        a = np.stack([r[3] for r in self._rows])
        present = np.stack([r[4] for r in self._rows])
        ids = np.stack(self._ids)
        full = build_history(s, lane, v, a, present, ids, H=self.H,
                             length=self.length)
        # left-pad to exactly H when fewer ticks were pushed
        pad = self.H - len(self._rows)
        if pad > 0:
            full = np.concatenate(
                [np.zeros((self.num_slots, pad, HISTORY_F)), full], axis=1)
        return full
