"""RSU-observable live state for the closed-loop A11 predictor (WM-N1-REPL-CL Part C).

Builds the A11 t0 input channels (``veh_kin [K,7]``, ``veh_mask [K]``,
``is_cav [K]``, ``slot_id [K]`` and the A6 entrance counter ``[2,30]``) from
the LIVE physics rows the closed-loop runner writes, by REUSING the featurizer
code -- never a parallel re-implementation:

* the per-vehicle rows go through :func:`features.featurizer.parse_run` on a
  one-row physics file with ``t_origin = T - 0.5 s`` (so the row at the
  decision time ``T`` is featurizer tick 1, exactly like the first physics row
  of a branch arm / ``snapshot_at_t0`` pack, whose snapshot is the first
  present tick);
* the ``veh_step`` columns are filled with the SAME assignments as
  :func:`features.featurizer.build_shard` (s_down + lane one-hot + v + a +
  nominal length + junction + present + CAV flag), rounded through float16
  like the pack, then :func:`features.idea_a.kinematics_view`;
* the entrance counter is :meth:`features.counters.DepartureLog.counter_at`
  over every departure at or before ``T`` (base pre-t0 history read with
  :func:`features.featurizer.parse_departures`, plus the closed-loop rows'
  ``departed_ids`` through the featurizer's ``_entrance_for``).

Documented approximation: training snapshots are the first branch physics row
(``t0 + 0.5 s``, i.e. after one 0.5 s substep under the arm's own action); the
live decision must be taken BEFORE the tick it applies to, so the live
snapshot is the row AT ``T`` (0.5 s earlier). Junction carry-forward starts
fresh at the snapshot exactly as on the training snapshot tick.

Only RSU-observable quantities enter: per-vehicle kinematics/lane of tracked
vehicles, the CAV roster flag and entrance (departure) counters. No hidden
driver types, no future demand, no outcome labels.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from features import constants as C
from features.counters import DepartureLog
from features.featurizer import _entrance_for, parse_run
from features.idea_a import kinematics_view

K_MAX = C.K_MAX_V3          # A11 trained on the v3 (K = 192) format


class LiveStateError(ValueError):
    pass


@dataclass
class LiveState:
    """One RSU-observable decision-time state (featurizer-equivalent)."""

    time_s: float
    veh_kin: np.ndarray            # float32 [K, 7]
    veh_mask: np.ndarray           # bool [K]
    is_cav: np.ndarray             # bool [K]
    slot_id: np.ndarray            # int64 [K] (-1 empty)
    counter: np.ndarray            # float32 [2, 30]
    slot_vid: list                 # vehicle id per slot (None = empty)
    n_dropped: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def rollout_mask(self) -> np.ndarray:
        """bool [K]: slots the A11 rollout engine can move -- present AND on
        a lane. Junction-lane rows (lane one-hot empty, featurizer s_down 0 at
        a snapshot tick) make rollout2 produce NaN speeds (no lane / segment
        limit), which poisons every plan of the decision; they are left out
        of the ROLLOUT world (documented approximation). The A6 state token
        still sees the featurized state unchanged."""
        on_lane = np.asarray(self.veh_kin, np.float32)[:, 1:4].sum(axis=1) > 0
        return np.asarray(self.veh_mask, bool) & on_lane

    @property
    def cav_slots(self) -> dict:
        """{cav_id: slot} for present CAV slots."""
        return {self.slot_vid[s]: int(s)
                for s in np.flatnonzero(self.veh_mask & self.is_cav)}

    @property
    def vehicle_slots(self) -> dict:
        return {self.slot_vid[s]: int(s) for s in np.flatnonzero(self.veh_mask)}


def base_departures_upto(base_run_dir, t0_s: float) -> list:
    """[(t, entrance)] of the base log's departures with t <= t0.

    Identical to ``features.featurizer.parse_departures`` (same row fields,
    same ``_entrance_for``) restricted to ``t <= t0``, which is exactly what
    ``build_shard`` keeps from the base history; it stops reading at the
    first row past t0 and also reads ``physics_steps.jsonl.gz``. A missing
    base log REFUSES (parse_departures would silently return no history and
    zero the A6 counter).
    """
    import gzip

    run = Path(base_run_dir) / "run"
    plain, gz = run / "physics_steps.jsonl", run / "physics_steps.jsonl.gz"
    if plain.is_file():
        fh = open(plain, encoding="utf-8")
    elif gz.is_file():
        fh = gzip.open(gz, "rt", encoding="utf-8")
    else:
        raise LiveStateError(f"base physics log missing under {run} (counter history)")
    out = []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = float(r["time_s"])
            if t > float(t0_s) + 1e-9:
                break
            vehs = r.get("vehicles") or {}
            for vid in (str(x) for x in r.get("departed_ids", ())):
                vv = vehs.get(vid)
                out.append((t, int(_entrance_for(vid, vv.get("road_id") if vv else None))))
    return out


class DepartureHistory:
    """Departures (time, entrance) <= the decision time, base + live rows."""

    def __init__(self, base_run_dir: Optional[Path], t0_s: float):
        self.t0_s = float(t0_s)
        self.times: list = []
        self.entrances: list = []
        if base_run_dir is not None:
            for t, e in base_departures_upto(base_run_dir, self.t0_s):
                self.times.append(float(t))
                self.entrances.append(int(e))

    def add_row(self, row: dict) -> None:
        t = float(row["time_s"])
        if t <= self.t0_s + 1e-9:
            return  # pre-t0 history comes from the base log only
        vehs = row.get("vehicles") or {}
        for vid in (str(x) for x in row.get("departed_ids", ())):
            vv = vehs.get(vid)
            self.times.append(t)
            self.entrances.append(int(_entrance_for(vid, vv.get("road_id") if vv else None)))

    def counter_at(self, t_s: float) -> np.ndarray:
        log = DepartureLog(0.0)
        for t, e in zip(self.times, self.entrances):
            if t <= float(t_s) + 1e-9:
                log.add(t, e)
        return log.counter_at(float(t_s))


def live_state_from_row(row: dict, departures: DepartureHistory, *,
                        k_max: int = K_MAX, tmp_dir=None) -> LiveState:
    """Featurize ONE physics row (the row at the decision time) -> LiveState."""
    t = float(row["time_s"])
    t_origin = t - C.DT_S
    fd, path = tempfile.mkstemp(suffix=".jsonl", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            clean = dict(row)
            clean["departed_ids"] = []   # counters come from DepartureHistory
            fh.write(json.dumps(clean) + "\n")
        parsed = parse_run(None, t_origin, k_max=int(k_max), physics_path=path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if parsed["n_ticks"] != 2:
        raise LiveStateError(f"expected the row at tick 1, got n_ticks={parsed['n_ticks']}")
    K = int(k_max)
    sel = parsed["rec_tick"] == 1
    sl = parsed["rec_slot"][sel]
    vi = parsed["rec_vididx"][sel]
    lane = parsed["rec_lane"][sel]
    sdown_raw = parsed["rec_sdown"][sel]
    jj = parsed["rec_junc"][sel]
    vv = parsed["rec_v"][sel]
    aa = parsed["rec_a"][sel]
    # build_shard junction carry-forward: no history at the first tick -> 0
    sdown = np.where((jj == 0) & (lane >= 0), sdown_raw, 0.0)
    veh = np.zeros((K, C.VEH_F), np.float16)
    mask = np.zeros(K, np.uint8)
    veh[sl, C.S_DOWN] = sdown.astype(np.float16)
    for cls, base in ((0, C.LANE_LEFT), (1, C.LANE_RIGHT), (2, C.LANE_ACCEL)):
        m = lane == cls
        veh[sl[m], base] = 1.0
    veh[sl, C.V_IDX] = vv.astype(np.float16)
    veh[sl, C.A_IDX] = aa.astype(np.float16)
    veh[sl, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
    veh[sl, C.JUNCTION_IDX] = jj.astype(np.float16)
    veh[sl, C.PRESENT_IDX] = 1.0
    mask[sl] = 1
    slot_vid_all = parsed["slot_vid"]
    slot_vid: list = [None] * K
    for s, v in zip(sl, vi):
        slot_vid[int(s)] = slot_vid_all[int(v)]
    cavs = parsed["all_cav_ids"]
    is_cav = np.zeros(K, bool)
    for s in sl:
        if slot_vid[int(s)] in cavs:
            is_cav[int(s)] = True
            veh[int(s), C.IS_CAV_IDX] = 1.0
    slot_index = np.asarray(parsed["slot_index"][1], np.int64)
    kin = np.asarray(kinematics_view(veh), np.float32)
    counter = departures.counter_at(t).astype(np.float16).astype(np.float32)
    return LiveState(
        time_s=t, veh_kin=kin, veh_mask=mask.astype(bool), is_cav=is_cav,
        slot_id=slot_index, counter=counter, slot_vid=slot_vid,
        n_dropped=int(parsed["n_dropped_concurrent"]),
        meta={"t_origin": t_origin, "n_present": int(mask.sum()),
              "n_cav_present": int(is_cav.sum()),
              "snapshot_convention": "row_at_T (training: T+0.5)"})
