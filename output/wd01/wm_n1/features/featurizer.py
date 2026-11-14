"""Per-run WM-N1 featurization (design sections 1, 3, 4).

One streaming pass over ``<arm>/run/physics_steps.jsonl`` builds:

* ``veh_step``  float16 [N_ticks, K, 16] + ``veh_mask`` (brief item 1);
* ``slot_index`` int32   [N_ticks, K] slot -> index into ``slot_vid``;
* ``plan``      float16 [N_dec, K, 19, 8] + ``decision_t`` (item 2);
* ``wall``      float16 [N_ticks, 4, 11] + ``wall_mask`` (item 3);
* ``counter``   float16 [N_ticks, 2, 30] (item 4);
* step targets  ``accel_next``/``target_mask``/``lane_next``/``arrivals`` (5);
* ``outcomes.parquet`` reused read-only from v0.4a labels (item 6).

K (the slot-pool size) is a FORMAT parameter
(:meth:`features.constants.resolve_k_max`): v2 keeps ``K_max = 128`` and stays
byte-reproducible; the v3 format (amendment 5) carries ``K_max = 192``.  Slots
are a persistent pool of ``K_max`` concurrent slots with reuse: a
continuing vehicle keeps its slot tick to tick; a slot freed by a leaving
vehicle is reused (the per-tick ``slot_index`` table is the authoritative
slot -> vehicle-id map).  Only ``lane_id``, ``lane_position_m``,
``speed_mps``, ``acceleration_mps2``, ``road_id`` and the tick lists
``departed_ids`` / ``cav_ids`` are read; ``type_id``, ``speed_mode``, world
strings, seed and t0 are never featurised, and CAV actions come only from the
v0.4a action adapter.
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np

from . import constants as C
from . import geometry as G
from .actions_adapter import PlannedActionTimeline, WallFeatureSource
from .counters import DepartureLog, entrance_of_edge
from .geometry import ENTRANCE_MAINLINE, ENTRANCE_RAMP


class FeaturizeError(ValueError):
    """A physics/run contract violation during featurization."""


def _entrance_for(vid: str, road) -> int:
    try:
        return entrance_of_edge(road) if road else (
            ENTRANCE_RAMP if vid.startswith("ramp.") else ENTRANCE_MAINLINE)
    except KeyError:
        return (ENTRANCE_RAMP if vid.startswith("ramp.")
                else ENTRANCE_MAINLINE)


def parse_departures(run_dir) -> DepartureLog:
    """Departure log over ALL rows of a run (used for the shared pre-restore
    counter history of a branch arm)."""
    physics = Path(run_dir) / "run" / "physics_steps.jsonl"
    dep = DepartureLog(0.0)
    if not physics.is_file():
        return dep
    with open(physics, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = float(r["time_s"])
            vehs = r.get("vehicles") or {}
            for vid in (str(x) for x in r.get("departed_ids", ())):
                vv = vehs.get(vid)
                dep.add(t, _entrance_for(vid, vv.get("road_id") if vv
                                         else None))
    return dep


def parse_run(run_dir, t_origin: float, *, k_max: int = C.K_MAX,
              physics_path=None) -> dict:
    # ``physics_path`` (additive WM-N1-DSPEED): an explicit physics log path
    # (optionally gzip-compressed) instead of the frozen
    # ``<run_dir>/run/physics_steps.jsonl`` layout.  ``None`` preserves the
    # frozen behaviour byte-for-byte.
    if physics_path is not None:
        physics = Path(physics_path)
    else:
        physics = Path(run_dir) / "run" / "physics_steps.jsonl"
    if not physics.is_file():
        raise FeaturizeError(f"missing {physics}")

    def _open_physics(path):
        if str(path).endswith(".gz"):
            import gzip
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, encoding="utf-8")

    vid_index: dict = {}
    vids: list = []
    free = list(range(int(k_max)))
    heapq.heapify(free)
    current: dict = {}
    recs = []                       # (tick, slot, vididx, lane, sdown, v, a, junc)
    dep = DepartureLog(t_origin)
    cav_ids: set = set()
    last_t = None
    n_dropped_concurrent = 0

    with _open_physics(physics) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = float(r["time_s"])
            vehs = r.get("vehicles") or {}
            for vid in (str(x) for x in r.get("departed_ids", ())):
                vv = vehs.get(vid)
                dep.add(t, _entrance_for(vid, vv.get("road_id") if vv
                                         else None))
            if t <= t_origin + 1e-9:
                continue
            last_t = t
            k = int(round((t - t_origin) / C.DT_S))
            if abs(t_origin + k * C.DT_S - t) > 1e-6:
                raise FeaturizeError(
                    f"{physics}: off-grid tick t={t} (t_origin={t_origin})")
            cav_ids.update(str(x) for x in r.get("cav_ids", ()))
            for vid in [v for v in current if v not in vehs]:
                heapq.heappush(free, current.pop(vid))
            for vid, vv in vehs.items():
                vid = str(vid)
                slot = current.get(vid)
                if slot is None:
                    if not free:
                        n_dropped_concurrent += 1    # >K_max concurrent: drop
                        continue
                    slot = current[vid] = heapq.heappop(free)
                vi = vid_index.get(vid)
                if vi is None:
                    vi = vid_index[vid] = len(vids)
                    vids.append(vid)
                lane_id = vv.get("lane_id")
                if lane_id is None or G.is_junction_lane(lane_id):
                    lane_phys, sdown, junc = -1, np.nan, 1
                else:
                    lane_phys, sdown = G.s_down_of(
                        lane_id, float(vv["lane_position_m"]))
                    junc = 0
                recs.append((k, slot, vi, lane_phys, sdown,
                             float(vv["speed_mps"]),
                             float(vv.get("acceleration_mps2", 0.0) or 0.0),
                             junc))

    if last_t is None:
        raise FeaturizeError(f"{physics}: no rows after t_origin={t_origin}")
    n_ticks = int(round((last_t - t_origin) / C.DT_S)) + 1

    rec = np.asarray(recs, np.float64) if recs else np.zeros((0, 8))
    tick = rec[:, 0].astype(np.int64)
    slot = rec[:, 1].astype(np.int64)
    vididx = rec[:, 2].astype(np.int64)
    lane = rec[:, 3].astype(np.int8)
    sdown = np.where(np.isnan(rec[:, 4]), 0.0, rec[:, 4])
    junc = np.where(rec[:, 7] > 0, 1, 0).astype(np.int8)
    varr = rec[:, 5].astype(np.float32)
    aarr = rec[:, 6].astype(np.float32)

    slot_index = np.full((n_ticks, int(k_max)), -1, np.int32)
    slot_index[tick, slot] = vididx
    return dict(
        physics=str(physics), n_ticks=n_ticks, t_origin=float(t_origin),
        data_end=float(last_t), slot_vid=list(vids), n_vehicles=len(vids),
        rec_tick=tick, rec_slot=slot, rec_vididx=vididx, rec_lane=lane,
        rec_sdown=sdown, rec_v=varr, rec_a=aarr, rec_junc=junc,
        slot_index=slot_index, departures=dep,
        n_dropped_concurrent=int(n_dropped_concurrent),
        all_cav_ids={str(x) for x in cav_ids})


def _lane_change_class(src: int, dst: int) -> int:
    """{stay,left,right} by lane_phys (accel 2 -> main is the merge/right)."""
    if src < 0 or dst < 0:
        return C.LANE_NONE
    if src == dst:
        return C.LANE_STAY
    if src == 2:
        return C.LANE_RIGHT_CLS
    if dst == 2:
        return C.LANE_LEFT_CLS
    return C.LANE_LEFT_CLS if dst < src else C.LANE_RIGHT_CLS


def _wall_row(w: dict) -> np.ndarray:
    row = np.zeros(C.WALL_F, np.float32)
    if w["phase"] in C.WALL_PHASES:
        row[C.WALL_PHASES.index(w["phase"])] = 1.0
    row[5] = float(w.get("speed_mps", 0.0)) / 40.0
    row[6] = float(w.get("time_in_phase_s", 0.0)) / 600.0
    row[7] = float(w.get("planned_hold_remaining_s", 0.0)) / 600.0
    row[8] = float(w.get("n_members", 0.0)) / 4.0
    row[9] = float(w.get("mean_s_down_m", 0.0)) / 1000.0
    row[10] = float(w.get("lane_span", 0.0)) / 2.0
    return row.astype(np.float16)


def build_shard(chunk: str, world: str, arm: str, run_dir, *, split: str,
                plan_row: dict, outcomes_root=None,
                base_history_run_dir=None, batch=None,
                policy: str | None = None, format: str | None = None,
                k_max: int | None = None, physics_path=None,
                snapshot_at_t0: bool = False) -> dict:
    """Build one run shard.

    ``arm`` is the exact run-dir arm suffix (``fixed1``, not the ``fixed``
    plan arm).  ``batch`` is the inventory batch name and keys the shard
    directory first component (cross-batch runs can share world + arm
    suffix); it defaults to ``chunk``.  ``policy`` is the plan-level policy
    label carried as metadata only.  ``format`` / ``k_max`` select the slot
    width (v2 = 128 default and byte-reproducible, v3 = 192, amendment 5);
    v3 shards self-describe via ``meta.json`` ``format``/``k_max`` keys,
    v2 shard bytes are unchanged.

    ``snapshot_at_t0`` (default OFF, additive): a full-run ``base`` /
    ``native_base`` physics log spans the whole episode, so the historical
    behaviour featurizes it from origin 0 and its decision-time snapshot does
    NOT line up with the item's ``t0`` labels.  With this flag ON the snapshot
    origin is the item's ``t0`` -- the first physics row is ``t0 + 0.5 s``,
    exactly like a branch arm -- while the pre-``t0`` departure history needed
    by the entrance-counter channel is still read from the same log (it is
    accumulated for every row before the origin cut).  Branch arms are
    unaffected by the flag; leaving it False preserves the frozen bytes.
    """
    fmt, K = C.resolve_k_max(format, k_max)
    t0 = float(plan_row["t0_s"])
    base_like = arm in ("base", "native_base")
    t_origin = 0.0 if (base_like and not snapshot_at_t0) else t0

    parsed = parse_run(run_dir, t_origin, k_max=K, physics_path=physics_path)
    # a branch arm's physics file starts at t0; the shared pre-restore
    # departure history for the counter channel comes from the base arm.
    if not base_like and base_history_run_dir is not None:
        base_dep = parse_departures(base_history_run_dir)
        for t, e in zip(*base_dep.arrays()):
            if t <= t0 + 1e-9:
                parsed["departures"].add(float(t), int(e))
    n = parsed["n_ticks"]

    tk = parsed["rec_tick"]
    sl = parsed["rec_slot"]
    vi = parsed["rec_vididx"]
    lane = parsed["rec_lane"]
    sdown_raw = parsed["rec_sdown"]
    vv = parsed["rec_v"]
    aa = parsed["rec_a"]
    jj = parsed["rec_junc"]
    order = np.lexsort((tk, sl))
    tk, sl, vi, lane, sdown_raw, vv, aa, jj = (x[order] for x in
        (tk, sl, vi, lane, sdown_raw, vv, aa, jj))

    # s_down carry-forward for junction rows (per slot)
    carry = np.zeros(K, np.float64)
    has = np.zeros(K, bool)
    sdown = sdown_raw.copy()
    for idx in range(len(sl)):
        s = sl[idx]
        if jj[idx] == 0 and lane[idx] >= 0:
            carry[s] = sdown_raw[idx]
            has[s] = True
        elif has[s]:
            sdown[idx] = carry[s]
        else:
            sdown[idx] = 0.0

    veh = np.zeros((n, K, C.VEH_F), np.float16)
    mask = np.zeros((n, K), np.uint8)
    lane_d = np.full((n, K), -1, np.int8)
    sdown_d = np.zeros((n, K), np.float64)
    vididx_d = np.full((n, K), -1, np.int64)
    a_d = np.zeros((n, K), np.float32)

    veh[tk, sl, C.S_DOWN] = sdown.astype(np.float16)
    for cls, base in ((0, C.LANE_LEFT), (1, C.LANE_RIGHT), (2, C.LANE_ACCEL)):
        sel = lane == cls
        veh[tk[sel], sl[sel], base] = 1.0
    veh[tk, sl, C.V_IDX] = vv.astype(np.float16)
    veh[tk, sl, C.A_IDX] = aa.astype(np.float16)
    veh[tk, sl, C.LENGTH_IDX] = C.NOMINAL_LENGTH_M
    veh[tk, sl, C.JUNCTION_IDX] = jj.astype(np.float16)
    veh[tk, sl, C.PRESENT_IDX] = 1.0
    mask[tk, sl] = 1
    lane_d[tk, sl] = lane
    sdown_d[tk, sl] = sdown
    a_d[tk, sl] = aa
    vididx_d[tk, sl] = vi

    slot_vid = parsed["slot_vid"]
    # is_cav is a per-vehicle property: mark via slot_index -> vid membership
    cav_vididx = {i for i, v in enumerate(slot_vid)
                  if v in parsed["all_cav_ids"]}
    is_cav_present = np.isin(vididx_d, list(cav_vididx))
    is_cav_present &= mask.astype(bool)
    veh[is_cav_present, C.IS_CAV_IDX] = 1.0

    # ---- walls + actions ----
    timeline = PlannedActionTimeline(run_dir)
    walls = WallFeatureSource.from_run_dir(run_dir)
    wall = np.zeros((n, C.MAX_WALLS, C.WALL_F), np.float16)
    wall_mask = np.zeros((n, C.MAX_WALLS), np.uint8)
    wall_ids: list = []
    slots_by_tick = [np.flatnonzero(mask[k]) for k in range(n)]

    for k in range(n):
        present = slots_by_tick[k]
        if not len(present):
            continue
        t = t_origin + k * C.DT_S
        positions = {slot_vid[vididx_d[k, s]]:
                     (float(sdown_d[k, s]), int(lane_d[k, s]))
                     for s in present}
        roster = {slot_vid[vididx_d[k, s]] for s in present
                  if is_cav_present[k, s]}
        states = walls.walls_at(t, positions=positions, cav_roster=roster)
        for wi, w in enumerate(states[:C.MAX_WALLS]):
            wall[k, wi] = _wall_row(w)
            wall_mask[k, wi] = 1
            wid = str(w["wall_id"])
            if wid not in wall_ids:
                wall_ids.append(wid)
        member = set()
        for w in states:
            if w["phase"] in ("requested", "assembling", "holding"):
                for s in present:
                    if slot_vid[vididx_d[k, s]] in {str(m)
                                                    for m in w["members"]}:
                        member.add(int(s))
        for s in present:
            vid = slot_vid[vididx_d[k, s]]
            a = (C.ACTIONS.index("wall") if int(s) in member
                 else timeline.action_slot(vid, t))
            veh[k, s, C.ACT_NATIVE + a] = 1.0
            if int(s) in member:
                veh[k, s, C.WALL_MEMBER_IDX] = 1.0

    # ---- step targets (per vehicle, not per slot) ----
    accel_next = np.zeros((n, K), np.float16)
    target_mask = np.zeros((n, K), np.uint8)
    lane_next = np.full((n, K), C.LANE_NONE, np.int8)
    if n > 1:
        same = (vididx_d[:-1] == vididx_d[1:]) & (vididx_d[:-1] >= 0)
        target_mask[:-1] = same.astype(np.uint8)
        accel_next[:-1] = np.where(same, a_d[1:], 0.0).astype(np.float16)
        for s in range(K):
            for k in range(n - 1):
                if same[k, s]:
                    lane_next[k, s] = _lane_change_class(
                        int(lane_d[k, s]), int(lane_d[k + 1, s]))

    counter = np.zeros((n, C.N_ENTRANCES, C.COUNTER_BINS), np.float16)
    for k in range(n):
        counter[k] = parsed["departures"].counter_at(t_origin + k * C.DT_S)
    arrivals = parsed["departures"].arrivals_by_tick(n)

    # ---- plan tensor (designated CAVs, aligned to the decision tick) ----
    decision_t = t0 + C.DECIDE_EVERY_S * np.arange(C.N_DECISION,
                                                   dtype=np.float64)
    plan = np.zeros((C.N_DECISION, K, C.N_SEGMENTS, C.PLAN_F), np.float16)
    for di, t_dec in enumerate(decision_t):
        k_dec = int(round((float(t_dec) - t_origin) / C.DT_S))
        k_dec = min(max(k_dec, 0), n - 1)
        if not mask[k_dec].any():
            k_dec = _nearest_present_tick(mask, k_dec)
        if k_dec < 0:
            continue
        for s in np.flatnonzero(mask[k_dec] & is_cav_present[k_dec]):
            vid = slot_vid[vididx_d[k_dec, s]]
            for seg in range(C.N_SEGMENTS):
                seg_t = float(t_dec) + C.SEGMENT_S * seg
                a = timeline.action_at(vid, seg_t)
                plan[di, s, seg, C.PLAN_ACTION_SLOTS + C.ACTIONS.index(a)] = 1.0
                st = _wall_of(walls, seg_t, vid)
                if st is not None:
                    wid = str(st["wall_id"])
                    if wid not in wall_ids:
                        wall_ids.append(wid)
                    plan[di, s, seg, C.PLAN_WALL_ID] = float(
                        wall_ids.index(wid))
                    plan[di, s, seg, C.PLAN_WALL_SPEED] = (
                        float(st.get("speed_mps", 0.0)) / 40.0)
                    plan[di, s, seg, C.PLAN_WALL_HOLD] = (
                        float(st.get("planned_hold_remaining_s", 0.0)) / 600.0)

    outcomes = _outcomes_table(chunk, world, arm, decision_t, outcomes_root,
                               batch=batch, policy=policy, split=split,
                               t0_s=t0)

    meta = dict(
        chunk=chunk, batch=batch or chunk, world=world, arm=arm, split=split,
        policy=policy, t0_s=t0,
        t_origin=t_origin, n_ticks=int(n), data_end=parsed["data_end"],
        n_vehicles=parsed["n_vehicles"],
        n_dropped_slots=0,
        n_dropped_slot_steps=parsed["n_dropped_concurrent"],
        n_slots=int(mask.any(axis=0).sum()),
        slot_vid=slot_vid, wall_ids=wall_ids,
        decision_t=[float(x) for x in decision_t],
        input_physics=parsed["physics"])
    if fmt != C.DEFAULT_FORMAT:
        # v3 (and any future non-default format) self-describes; the v2
        # meta.json bytes stay exactly as the frozen writer wrote them.
        meta["format"] = fmt
        meta["k_max"] = int(K)
    if snapshot_at_t0:
        # additive marker; absent (frozen bytes) when the option is OFF.
        meta["snapshot_at_t0"] = True
    return dict(veh_step=veh, veh_mask=mask, slot_index=parsed["slot_index"],
                slot_vid=slot_vid, plan=plan, decision_t=decision_t,
                wall=wall, wall_mask=wall_mask, counter=counter,
                accel_next=accel_next, target_mask=target_mask,
                lane_next=lane_next, arrivals=arrivals,
                outcomes=outcomes, meta=meta)


def _wall_of(walls, t, vid):
    for w in walls.walls_at(t):
        if vid in {str(m) for m in w["members"]}:
            return w
    return None


def _nearest_present_tick(mask, k0: int) -> int:
    """Smallest tick >= k0 with any present vehicle, else the largest < k0."""
    present = mask.any(axis=1)
    ahead = np.flatnonzero(present[k0:])
    if len(ahead):
        return int(k0 + ahead[0])
    before = np.flatnonzero(present[:k0])
    return int(before[-1]) if len(before) else -1


def _outcomes_table(chunk, world, arm, decision_t, outcomes_root, *,
                    batch=None, policy=None, split=None, t0_s=None):
    """Reuse the v0.4a label rows for this item's RESOLVED source.

    The pack ``arm`` is resolved to the v0.4a source suffix via
    :func:`features.outcomes.label_source` (the campaign chunk-2/3
    ``__native``/``__pi`` switching runs live under ``swnative``/``swpi``).
    A ``None`` source or an absent label file yields the historical
    zero-label empty frame; a present file that fails the hard identity check
    raises :class:`features.outcomes.OutcomeSourceError` (never a silent
    fall-back).
    """
    from . import outcomes
    resolved = outcomes.label_source(
        batch if batch is not None else chunk, chunk, arm, policy)
    if resolved is None:
        import pandas as pd
        return pd.DataFrame()
    if not outcomes.available(chunk, world, resolved, outcomes_root):
        import pandas as pd
        return pd.DataFrame()
    item = dict(world=world, chunk=chunk, arm=arm,
                batch=batch if batch is not None else chunk, policy=policy,
                split=split, t0_s=t0_s)
    return outcomes.load_outcome_rows(chunk, world, resolved, decision_t,
                                      C.TAUS, root=outcomes_root, item=item)
