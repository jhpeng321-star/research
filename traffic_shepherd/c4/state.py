"""SUMO vehicle table -> WM-S model-frame slot state, rolling history, RSU detections.

Every live vehicle on the network (all lanes, the full ~2.6 km upstream mainline and the
whole ramp) is converted, so the RSU zones of ``wms.control.contract`` are covered.
Encoding = the WM-S packs (``veh_kin`` = s_down, lane one-hot x3, v, a, nominal length
5 m; persistent slot pool; junction rows have no lane one-hot and carry the last s).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import (NOMINAL_LENGTH_M, SlotPool, is_junction_lane, lane_and_s)

DT_S = 0.5
# RSU zones (wms.control.contract; duplicated as plain floats so this module has no torch
# import cost at SUMO-worker start; a unit test asserts equality with the contract).
RSU_MAINLINE_S_M = (-800.0, 225.44)
RSU_RAMP_S_M = (-378.66, 75.44)
RSU_CELL_M = 50.0


@dataclass
class Frame:
    """One 0.5 s physics snapshot in slot layout (all arrays ``[K]``)."""

    time_s: float
    s: np.ndarray            # float32 s_down (junction: carried)
    lane: np.ndarray         # int8 0/1/2, -1 absent or junction
    lane_carry: np.ndarray   # int8 last modelled lane (junction rows keep it), -1 absent
    v: np.ndarray            # float32
    a: np.ndarray            # float32
    length_true: np.ndarray  # float32 SUMO vType length (pack uses nominal 5 m)
    present: np.ndarray      # bool slot occupied (incl. junction transit)
    is_cav: np.ndarray       # bool
    junction: np.ndarray     # bool on a junction internal lane
    vid_index: np.ndarray    # int32 index into the episode vehicle registry, -1 empty
    dropped: tuple = ()      # vehicle ids without a slot this step (pool full)

    @property
    def k(self) -> int:
        return int(self.s.shape[0])

    def kin(self) -> np.ndarray:
        """``[K, 7]`` pack ``veh_kin`` row: s, lane one-hot (0 on junction), v, a, 5 m."""
        out = np.zeros((self.k, 7), np.float32)
        out[:, 0] = np.where(self.present, self.s, 0.0)
        for lane in range(3):
            out[:, 1 + lane] = (self.lane == lane)
        out[:, 4] = np.where(self.present, self.v, 0.0)
        out[:, 5] = np.where(self.present, self.a, 0.0)
        out[:, 6] = np.where(self.present, NOMINAL_LENGTH_M, 0.0)
        return out


class FrameBuilder:
    """Stable slot assignment + per-vehicle junction carry over one episode."""

    def __init__(self, k: int, cav_ids):
        self.k = int(k)
        self.pool = SlotPool(self.k)
        self.cav_ids = frozenset(str(v) for v in cav_ids)
        self.vehicle_ids: list[str] = []
        self._index: dict[str, int] = {}
        self._carry: dict[str, tuple[int, float]] = {}

    def index_of(self, vid: str) -> int:
        i = self._index.get(vid)
        if i is None:
            i = self._index[vid] = len(self.vehicle_ids)
            self.vehicle_ids.append(vid)
        return i

    def build(self, time_s: float, table: dict) -> Frame:
        k = self.k
        s = np.zeros(k, np.float32)
        lane = np.full(k, -1, np.int8)
        lane_carry = np.full(k, -1, np.int8)
        v = np.zeros(k, np.float32)
        a = np.zeros(k, np.float32)
        length = np.zeros(k, np.float32)
        present = np.zeros(k, bool)
        is_cav = np.zeros(k, bool)
        junction = np.zeros(k, bool)
        vid_index = np.full(k, -1, np.int32)
        slots, dropped = self.pool.update(sorted(table))
        live = set(table)
        self._carry = {vid: c for vid, c in self._carry.items() if vid in live}
        for vid, slot in slots.items():
            veh = table[vid]
            if is_junction_lane(veh["lane_id"]):
                last_lane, last_s = self._carry.get(vid, (-1, 0.0))
                s[slot], lane_carry[slot], junction[slot] = last_s, last_lane, True
            else:
                ln, sd = lane_and_s(veh["lane_id"], veh["lane_position_m"])
                self._carry[vid] = (ln, sd)
                s[slot], lane[slot], lane_carry[slot] = sd, ln, ln
            v[slot] = veh["speed_mps"]
            a[slot] = veh["acceleration_mps2"]
            length[slot] = veh.get("length_m", NOMINAL_LENGTH_M)
            present[slot] = True
            is_cav[slot] = vid in self.cav_ids
            vid_index[slot] = self.index_of(vid)
        return Frame(float(time_s), s, lane, lane_carry, v, a, length, present, is_cav,
                     junction, vid_index, tuple(dropped))

    def slot_vehicle_ids(self, frame: Frame) -> list:
        return [self.vehicle_ids[i] if i >= 0 else None for i in frame.vid_index.tolist()]


def frame_to_state(frame: Frame, *, device="cpu") -> dict:
    """Contract state dict (batch 1) for ``wms.control``: s_m, v_mps, a_mps2, lane, present,
    is_cav, length_m (nominal, as the packs), junction."""
    import torch

    def t(x, dtype):
        return torch.as_tensor(np.asarray(x)[None], dtype=dtype, device=device)

    return {"s_m": t(np.where(frame.present, frame.s, 0.0), torch.float32),
            "v_mps": t(np.where(frame.present, frame.v, 0.0), torch.float32),
            "a_mps2": t(np.where(frame.present, frame.a, 0.0), torch.float32),
            "lane": t(frame.lane.astype(np.int64), torch.long),
            "present": t(frame.present, torch.bool),
            "is_cav": t(frame.is_cav & frame.present, torch.bool),
            "length_m": t(np.full(frame.k, NOMINAL_LENGTH_M, np.float32), torch.float32),
            "junction": t(frame.junction, torch.bool)}


# ----------------------------------------------------------------------- RSU zones
def in_rsu_zone(lane, s) -> np.ndarray:
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=np.float64)
    main = ((lane == 0) | (lane == 1)) & (s >= RSU_MAINLINE_S_M[0]) & (s < RSU_MAINLINE_S_M[1])
    ramp = (lane == 2) & (s >= RSU_RAMP_S_M[0]) & (s < RSU_RAMP_S_M[1])
    return main | ramp


def rsu_cell_index(lane: int, s: float):
    """Cell index in the contract order (lane 0 cells, lane 1 cells, lane 2 cells) or None."""
    n_main = math.ceil((RSU_MAINLINE_S_M[1] - RSU_MAINLINE_S_M[0]) / RSU_CELL_M)
    if lane in (0, 1) and RSU_MAINLINE_S_M[0] <= s < RSU_MAINLINE_S_M[1]:
        return lane * n_main + int((s - RSU_MAINLINE_S_M[0]) // RSU_CELL_M)
    if lane == 2 and RSU_RAMP_S_M[0] <= s < RSU_RAMP_S_M[1]:
        return 2 * n_main + int((s - RSU_RAMP_S_M[0]) // RSU_CELL_M)
    return None


def n_rsu_cells() -> int:
    n_main = math.ceil((RSU_MAINLINE_S_M[1] - RSU_MAINLINE_S_M[0]) / RSU_CELL_M)
    return 2 * n_main + math.ceil((RSU_RAMP_S_M[1] - RSU_RAMP_S_M[0]) / RSU_CELL_M)


def rsu_cell_counts(lanes, ss) -> np.ndarray:
    counts = np.zeros(n_rsu_cells(), np.int64)
    for lane, s in zip(np.asarray(lanes).tolist(), np.asarray(ss, np.float64).tolist()):
        cell = rsu_cell_index(int(lane), float(s))
        if cell is not None:
            counts[cell] += 1
    return counts


def rsu_detections(frame: Frame, vehicle_ids: list) -> list[dict]:
    """Every vehicle inside the RSU zones (junction transits use their carried lane)."""
    lane_eff = np.where(frame.junction, frame.lane_carry, frame.lane)
    zone = frame.present & in_rsu_zone(lane_eff, frame.s)
    out = []
    for slot in np.flatnonzero(zone).tolist():
        out.append({"vehicle_id": vehicle_ids[int(frame.vid_index[slot])], "slot": slot,
                    "timestamp_s": frame.time_s, "s_m": float(frame.s[slot]),
                    "lane": int(lane_eff[slot]), "v_mps": float(frame.v[slot]),
                    "a_mps2": float(frame.a[slot]), "is_cav": bool(frame.is_cav[slot]),
                    "length_m": float(frame.length_true[slot]),
                    "junction": bool(frame.junction[slot])})
    return out


class ReferenceFrame:
    """Independent SUMO-derived (lane, s) from live lane lengths (parity reference).

    Uses only ``api.lane.getLength`` and the adapter edge ids: s = position relative to
    the start of the merge edge along the vehicle's own lane chain.  Lane indices map
    left-most mainline lane -> 0, next -> 1, the merge edge's rightmost lane and the ramp
    -> 2.  Junction internal lanes return ``None``.
    """

    def __init__(self, api, config):
        e = config.edge_ids
        self.pre, self.merge, self.down, self.ramp = (e["pre_merge"], e["merge"],
                                                      e["downstream"], e["ramp"])
        L = api.lane.getLength
        self.n = {edge: api.edge.getLaneNumber(edge)
                  for edge in (self.pre, self.merge, self.down, self.ramp)}
        self.lengths = {f"{edge}_{i}": float(L(f"{edge}_{i}"))
                        for edge, n in self.n.items() for i in range(n)}

    def lane_and_s(self, lane_id: str, pos: float):
        if lane_id.startswith(":"):
            return None
        edge, _, idx = lane_id.rpartition("_")
        idx = int(idx)
        n = self.n[edge]
        if edge == self.ramp:
            return 2, pos - self.lengths[lane_id]
        if edge == self.merge:
            lane = 2 if idx == 0 else n - 1 - idx
            return lane, pos
        lane = n - 1 - idx                       # left-most = 0
        if edge == self.pre:
            return lane, pos - self.lengths[lane_id]
        if edge == self.down:
            merge_lane = f"{self.merge}_{self.n[self.merge] - 1 - lane}"
            return lane, self.lengths[merge_lane] + pos
        raise KeyError(lane_id)

    def cell_counts(self, table: dict) -> np.ndarray:
        lanes, ss = [], []
        for veh in table.values():
            r = self.lane_and_s(veh["lane_id"], veh["lane_position_m"])
            if r is not None:
                lanes.append(r[0])
                ss.append(r[1])
        return rsu_cell_counts(lanes, ss)


def frame_cell_counts(frame: Frame) -> np.ndarray:
    """Per-cell counts of non-junction vehicles of the converted slot state."""
    keep = frame.present & ~frame.junction
    return rsu_cell_counts(frame.lane[keep], frame.s[keep])


# -------------------------------------------------------------------- history buffer
class HistoryBuffer:
    """Rolling native-tick history of the slot state (pre-t0 driver history source)."""

    def __init__(self, max_ticks: int = 241):
        self.max_ticks = int(max_ticks)
        self._kin: deque = deque(maxlen=self.max_ticks)
        self._mask: deque = deque(maxlen=self.max_ticks)
        self._vid: deque = deque(maxlen=self.max_ticks)

    def clear(self) -> None:
        self._kin.clear()
        self._mask.clear()
        self._vid.clear()

    def append(self, frame: Frame) -> None:
        self._kin.append(frame.kin())
        self._mask.append(frame.present.copy())
        self._vid.append(frame.vid_index.copy())

    def __len__(self) -> int:
        return len(self._kin)

    def raw(self, n_ticks=None) -> dict:
        n = len(self) if n_ticks is None else min(int(n_ticks), len(self))
        if n < 1:
            raise ValueError("history buffer is empty")
        kin = np.stack(list(self._kin)[-n:])
        mask = np.stack(list(self._mask)[-n:])
        vid = np.stack(list(self._vid)[-n:])
        target = (vid[:-1] == vid[1:]) & (vid[:-1] >= 0)
        return {"kin": kin, "mask": mask, "target_mask": target, "vid_index": vid}

    def history(self, history_s: float | None = None, history_stride_s: float = 1.0) -> dict:
        """Slot-indexed pre-t0 history in the ``wms.eval`` state schema (``state["history"]``).

        ``history_s=None`` returns the model's fixed 10-tick rolling window.
        """
        ticks = 10 if history_s is None else int(math.floor(float(history_s) / DT_S + 1e-8)) + 1
        raw = self.raw(ticks)
        t = raw["kin"].shape[0]
        return {"kin": raw["kin"], "mask": raw["mask"], "target_mask": raw["target_mask"],
                "history_s": float(history_s if history_s is not None else (ticks - 1) * DT_S),
                "history_stride_s": float(history_stride_s), "dt_s": DT_S,
                "available_span_s": (t - 1) * DT_S, "run_start_limited": t < ticks,
                "ids": None}

    def driver_history(self, history_s: float | None = None, history_stride_s: float = 1.0,
                       history_len: int = 10):
        """``(driver_history [1,K,H,F], mask [1,K,H])`` for ``WmSModel.rollout``.

        ``history_s`` set: the long-history featuriser of the trainer
        (``wms.data.history.long_history_features``, F = 9).  Otherwise the rolling
        ``history_step_features`` window of ``history_len`` ticks (F = 6) with the
        trainer's backward same-vehicle continuity.
        """
        from .wms_glue import driver_history_tensors

        if history_s is not None:
            hist = self.history(history_s, history_stride_s)
            return driver_history_tensors(hist, history_s=history_s,
                                          history_stride_s=history_stride_s)
        return driver_history_tensors(self.raw(history_len), history_s=None,
                                      history_len=history_len)
