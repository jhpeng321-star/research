"""P9 featurizer additions for Idea A (no edits to the frozen writer).

New per-tick channels (appended; the existing v1 tensors are unchanged):

=================  =========  ========================  ===============================
name               dtype      shape                      meaning
=================  =========  ========================  ===============================
veh_kin            float16    [N_ticks,K,KIN_F=7]     kinematics-only view (no CAV flag,
                                                        no action one-hot)
hdv_base           float16    [N_ticks,K,6]           baseline per-HDV predictions:
                                                        idm_accel, mobil_gain_left,
                                                        mobil_gain_right, merge_urgency,
                                                        lc_accept_left, lc_accept_right
cav_plan_ch        float16    [N_ticks,K,8]           controller planned-trajectory
                                                        channels (tgt speed/40,
                                                        a_cmd/3, wall speed/40,
                                                        wall id/4, hold/600, lane tgt
                                                        one-hot 3)
event_tok          float16    [N_ticks,8,8]             P6 event tokens
event_mask         uint8      [N_ticks,8]               P6 event mask
=================  =========  ========================  ===============================

``K`` is the format slot width (v2 = 128, v3 = 192; amendment 5) and equals
the ``veh_mask`` axis 1 of the shard the channels are built from.

Leakage rule: every channel value at tick ``t`` is computed from recorded
data with timestamps ``<= t`` only (CAV speed/accel at ``t``, wall events
``<= t`` per the frozen v1 wall convention); ``accel_next``/``lane_next``
style targets may look ahead but are not touched here.  HDV baseline
predictions replay the frozen Part A v2g nominal rules
(``partA_v2g.engine``, ``params_nominal.json`` LF-sha256
:data:`NOMINAL_PARAMS_LF_SHA256`, never modified) on the reconstructed
scene state at ``t``.

The amendment-2 road-end relabel flag for J4 is a run-level ``meta.json`` key
(:data:`ROAD_END_RELABEL_FLAG`); the relabel itself only rewrites the
``member_left_network`` end reason and never touches run receipts.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

# veh_step indices reused (features/constants.py:16).
S_DOWN, LANE_LEFT, LANE_RIGHT, LANE_ACCEL = 0, 1, 2, 3
V_IDX, A_IDX, LENGTH_IDX, IS_CAV_IDX = 4, 5, 6, 7
ACT_FIRST, ACT_LAST = 8, 12
WALL_MEMBER_IDX = 13
JUNCTION_IDX = 14
PRESENT_IDX = 15

# kinematics-only columns, in HdvConfig.KIN_FIELDS order.
KIN_SOURCE_INDICES = (S_DOWN, LANE_LEFT, LANE_RIGHT, LANE_ACCEL, V_IDX, A_IDX,
                      LENGTH_IDX)
KIN_F = len(KIN_SOURCE_INDICES)

HDV_BASE_F = 6
CAV_PLAN_CH_F = 8
EVENT_TOK_F = 8
EVENT_MAX = 8

IDEA_A_CHANNELS = ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                   "event_mask")

# channel normalisation scales (INTERFACES_V2 tensors table)
CAV_TGT_SPEED_SCALE = 40.0        # tgt_speed_mps / 40
CAV_ACCEL_SCALE = 3.0             # a_cmd_mps2 / 3
CAV_WALL_SPEED_SCALE = 40.0       # wall_speed_mps / 40 (token is pre-scaled)
CAV_WALL_ID_SCALE = 4.0           # wall id ordinal / 4, -1 = none
CAV_HOLD_SCALE = 600.0            # planned hold remaining s / 600

HDV_BASE_FIELDS = ("idm_accel", "mobil_gain_left", "mobil_gain_right",
                   "merge_urgency", "lc_accept_left", "lc_accept_right")
CAV_PLAN_FIELDS = ("tgt_speed_over_40", "a_cmd_over_3", "wall_speed_over_40",
                   "wall_id_over_4", "hold_over_600", "lane_target_left",
                   "lane_target_right", "lane_target_accel")

# amendment-2 meta.json key (J4 road-end relabel convention).
ROAD_END_RELABEL_FLAG = "amendment2_road_end_relabel"
# the relabelled end reason (amendment 2: disclosure, J4 v2.1 wall data).
ROAD_END_REASON_SOURCE = "member_left_network"
ROAD_END_REASON_RELABELLED = "road_end"
REASON_FIELDS = ("end_reason", "abort_reason", "reason")

# frozen nominal Part A v2g parameter file provenance (LF-normalised sha256;
# the featurizer refuses to run against a modified parameter set).
NOMINAL_PARAMS_LF_SHA256 = (
    "079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7")

# physical-lane offsets (features/geometry.py HERE_OFFSETS; s = s_down + off)
LANE_OFFSETS_M = (2645.67, 2644.22, 378.66)


def _engine():
    """Import the frozen Part A v2g engine (read-only mirror)."""
    wd01 = Path(__file__).resolve().parents[2]           # output/wd01
    if str(wd01) not in sys.path:
        sys.path.insert(0, str(wd01))
    from partA_v2g import engine as eng

    return eng


_ENGINE_MEMO: list = []


def _engine_cached():
    """Memoized :func:`_engine` for the per-tick fast path.

    The engine module object never changes across ticks, so the fast path
    skips the per-call ``sys.path``/import-machinery probe (cProfile: the
    pathlib resolve chain cost ~1.5 s per 1200-tick run).  The slow path is
    untouched and still re-resolves.
    """
    if not _ENGINE_MEMO:
        _ENGINE_MEMO.append(_engine())
    return _ENGINE_MEMO[0]


def lf_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def load_nominal_params():
    """Frozen ``partA_v2g/params_nominal.json``, provenance-checked.

    Raises :class:`RuntimeError` when the file's LF-sha256 drifts from
    :data:`NOMINAL_PARAMS_LF_SHA256` (the featurizer baseline must stay on
    the registered nominal parameter set).
    """
    eng = _engine()
    raw = Path(eng.PARAMS_PATH).read_bytes()
    got = lf_sha256_bytes(raw)
    if got != NOMINAL_PARAMS_LF_SHA256:
        raise RuntimeError(
            "partA_v2g/params_nominal.json drifted from the frozen "
            f"LF-sha256: got {got}, expected {NOMINAL_PARAMS_LF_SHA256}")
    return eng.load_params()


# ------------------------------------------------------------------ views --

def kinematics_view(veh_step):
    """``[..., 16]`` -> ``[..., KIN_F]`` kinematics only.

    Drops the CAV flag (index 7) and every action one-hot (indices 8-12); the
    headline HDV net must never see them (also drops the wall-member,
    junction and present columns, which are strategy/state bookkeeping).
    Accepts torch tensors and numpy arrays.
    """
    if isinstance(veh_step, Tensor):
        idx = torch.as_tensor(KIN_SOURCE_INDICES, dtype=torch.long,
                              device=veh_step.device)
        return veh_step.index_select(-1, idx)
    veh = np.asarray(veh_step)
    return veh[..., list(KIN_SOURCE_INDICES)]


def baseline_predictions(idm_accel_mps2, mobil_gain_left, mobil_gain_right,
                         merge_urgency, lc_accept_left,
                         lc_accept_right):
    """Stack the six per-HDV baseline channels ``[..., 6]``.

    Channel order is :data:`HDV_BASE_FIELDS`; boolean acceptance inputs are
    coerced to 0/1 floats.  Torch tensors stay torch, numpy stays numpy.
    """
    accept = []
    for x in (lc_accept_left, lc_accept_right):
        if isinstance(x, Tensor):
            accept.append(x.to(dtype=torch.float32))
        else:
            accept.append(np.asarray(x, dtype=np.float32))
    if isinstance(idm_accel_mps2, Tensor):
        return torch.stack([
            torch.as_tensor(idm_accel_mps2, dtype=torch.float32),
            torch.as_tensor(mobil_gain_left, dtype=torch.float32),
            torch.as_tensor(mobil_gain_right, dtype=torch.float32),
            torch.as_tensor(merge_urgency, dtype=torch.float32),
            accept[0], accept[1]], dim=-1)
    return np.stack([
        np.asarray(idm_accel_mps2, dtype=np.float32),
        np.asarray(mobil_gain_left, dtype=np.float32),
        np.asarray(mobil_gain_right, dtype=np.float32),
        np.asarray(merge_urgency, dtype=np.float32),
        accept[0], accept[1]], axis=-1)


def cav_plan_channels(target_speed_mps, accel_cmd_mps2, wall_speed_mps,
                      wall_id, hold_s, lane_target):
    """Encode the controller planned-trajectory channels ``[..., 8]``.

    Layout (:data:`CAV_PLAN_FIELDS`): ``tgt_speed/40, a_cmd/3,
    wall_speed/40, wall_id/4, hold/600, lane_target`` as a 3-wide one-hot.
    ``wall_id`` uses the frozen convention ``-1 = none`` (stored as
    ``-1/4 = -0.25``).  ``lane_target`` is either an integer lane index
    ``[..., ]`` (0/1/2, -1 none) or an already one-hot/probability
    ``[..., 3]`` tensor.  Torch tensors stay torch, numpy stays numpy.
    """
    torch_out = isinstance(target_speed_mps, Tensor) or \
        isinstance(lane_target, Tensor)

    def f32(x):
        if torch_out:
            return torch.as_tensor(_unwrap(x), dtype=torch.float32)
        return np.asarray(x, dtype=np.float32)

    if torch_out:
        tgt = torch.as_tensor(np.asarray(target_speed_mps, np.float64)
                              if not isinstance(target_speed_mps, Tensor)
                              else target_speed_mps, dtype=torch.float32)
        one_hot = None
        lt = lane_target
        if not isinstance(lt, Tensor):
            lt = torch.as_tensor(np.asarray(lt))
        if lt.ndim >= 1 and lt.shape[-1] == 3:
            one_hot = lt.to(dtype=torch.float32)
        else:
            idx = lt.to(dtype=torch.long)
            one_hot = torch.zeros(*idx.shape, 3, dtype=torch.float32,
                                  device=idx.device)
            ok = (idx >= 0) & (idx < 3)
            if bool(ok.any()):
                one_hot[ok.nonzero(as_tuple=True) +
                        (idx[ok],)] = 1.0
        return torch.cat([
            (tgt / CAV_TGT_SPEED_SCALE).to(torch.float32).unsqueeze(-1),
            f32(accel_cmd_mps2).unsqueeze(-1) / CAV_ACCEL_SCALE,
            f32(wall_speed_mps).unsqueeze(-1) / CAV_WALL_SPEED_SCALE,
            f32(wall_id).unsqueeze(-1) / CAV_WALL_ID_SCALE,
            f32(hold_s).unsqueeze(-1) / CAV_HOLD_SCALE,
            one_hot], dim=-1)

    tgt = np.asarray(target_speed_mps, dtype=np.float32)
    lt = np.asarray(lane_target)
    if lt.ndim >= 1 and lt.shape[-1] == 3:
        one_hot = lt.astype(np.float32)
    else:
        idx = lt.astype(np.int64)
        one_hot = np.zeros(idx.shape + (3,), np.float32)
        ok = (idx >= 0) & (idx < 3)
        one_hot[ok, idx[ok]] = 1.0
    return np.concatenate([
        (tgt / CAV_TGT_SPEED_SCALE)[..., None],
        (np.asarray(accel_cmd_mps2, np.float32) / CAV_ACCEL_SCALE)[..., None],
        (np.asarray(wall_speed_mps, np.float32) /
         CAV_WALL_SPEED_SCALE)[..., None],
        (np.asarray(wall_id, np.float32) / CAV_WALL_ID_SCALE)[..., None],
        (np.asarray(hold_s, np.float32) / CAV_HOLD_SCALE)[..., None],
        one_hot], axis=-1).astype(np.float32)


def _unwrap(x):
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


# ------------------------------------------------- amendment-2 relabel -----

def apply_road_end_relabel(records, *, enabled: bool = False) -> dict:
    """Apply the amendment-2 J4 road-end relabel to wall tokens/labels.

    ``records`` is a dict with any of the keys ``wall_tokens`` (list of
    token/episode dicts), ``labels`` (pandas DataFrame or list of row
    dicts) and ``meta`` (corpus metadata dict); a bare DataFrame is treated
    as ``labels``.  When ``enabled`` is true, every end-reason field
    (:data:`REASON_FIELDS`) equal to ``member_left_network`` is rewritten to
    ``road_end`` and the returned ``meta`` carries
    :data:`ROAD_END_RELABEL_FLAG` ``= True`` plus
    ``road_end_relabelled_n``.  The input records are copied, never
    mutated -- the J4 run receipts keep their original labels.  With
    ``enabled=False`` the copy is returned unchanged and unflagged.
    """
    import pandas as pd

    if isinstance(records, pd.DataFrame):
        records = {"labels": records}
    out = dict(records or {})
    meta = dict(out.get("meta") or {})
    n = 0
    if enabled:
        tokens = out.get("wall_tokens")
        if isinstance(tokens, (list, tuple)):
            new_tokens = []
            for tok in tokens:
                tok = dict(tok) if isinstance(tok, dict) else tok
                if isinstance(tok, dict):
                    for field in REASON_FIELDS:
                        if str(tok.get(field, "")).strip() == \
                                ROAD_END_REASON_SOURCE:
                            tok[field] = ROAD_END_REASON_RELABELLED
                            n += 1
                new_tokens.append(tok)
            out["wall_tokens"] = new_tokens
        labels = out.get("labels")
        if isinstance(labels, pd.DataFrame):
            labels = labels.copy()
            for field in REASON_FIELDS:
                if field in labels.columns:
                    m = labels[field].astype(str).str.strip() == \
                        ROAD_END_REASON_SOURCE
                    n += int(m.sum())
                    if bool(m.any()):
                        labels.loc[m, field] = ROAD_END_REASON_RELABELLED
            out["labels"] = labels
        elif isinstance(labels, (list, tuple)):
            new_rows = []
            for r in labels:
                r = dict(r) if isinstance(r, dict) else r
                if isinstance(r, dict):
                    for field in REASON_FIELDS:
                        if str(r.get(field, "")).strip() == \
                                ROAD_END_REASON_SOURCE:
                            r[field] = ROAD_END_REASON_RELABELLED
                            n += 1
                new_rows.append(r)
            out["labels"] = new_rows
        meta[ROAD_END_RELABEL_FLAG] = True
        meta["road_end_relabelled_n"] = int(n)
        meta["road_end_relabel_note"] = (
            f"{ROAD_END_REASON_SOURCE} -> {ROAD_END_REASON_RELABELLED} "
            "(REGISTRATION_WM_N1_AMENDMENT_2_J4_DISCLOSURE.md; run receipts "
            "keep their original labels)")
    out["meta"] = meta
    return out


# ------------------------------------------------------------- byte spec ---

@dataclass
class IdeaAChannelSpecs:
    """Byte layout additions for the writer (frozen writer.py is untouched)."""

    kin_f: int = KIN_F
    hdv_base_f: int = HDV_BASE_F
    cav_plan_ch_f: int = CAV_PLAN_CH_F
    event_tok_f: int = EVENT_TOK_F
    event_max: int = EVENT_MAX

    def array_shapes(self, n_ticks: int, n_slots: int) -> dict:
        """``{channel: shape}`` of the five Idea A arrays for one run."""
        k = int(n_slots)
        return {
            "veh_kin": (int(n_ticks), k, self.kin_f),
            "hdv_base": (int(n_ticks), k, self.hdv_base_f),
            "cav_plan_ch": (int(n_ticks), k, self.cav_plan_ch_f),
            "event_tok": (int(n_ticks), self.event_max, self.event_tok_f),
            "event_mask": (int(n_ticks), self.event_max),
        }


def _npy_exact_bytes(shape, dtype) -> int:
    """Exact on-disk size of the ``.npy`` file numpy writes for ``shape``.

    Uses numpy's own v1.0 header writer (the same code path as
    ``np.lib.format.open_memmap``), so header + payload are exact.
    """
    import io

    import numpy.lib.format as fmt

    fp = io.BytesIO()
    fmt.write_array_header_1_0(fp, dict(
        descr=fmt.dtype_to_descr(np.dtype(dtype)), fortran_order=False,
        shape=tuple(int(s) for s in shape)))
    payload = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    return int(fp.tell()) + payload


def expected_bytes_idea_a(n_ticks: int, n_slots: int, *,
                          base_bytes: int = 0,
                          specs: Optional[IdeaAChannelSpecs] = None) -> int:
    """Exact extra on-disk bytes of the five Idea A ``.npy`` files.

    ``base_bytes`` is returned unchanged on top (v1 estimate convention).
    The sidecar ``IDEA_A_DONE.json`` receipt is not counted (the v1
    estimate likewise excludes JSON bookkeeping); ``write_idea_a_channels``
    reports its exact size separately in the written record.
    """
    specs = specs or IdeaAChannelSpecs()
    dtypes = {"veh_kin": np.float16, "hdv_base": np.float16,
              "cav_plan_ch": np.float16, "event_tok": np.float16,
              "event_mask": np.uint8}
    total = 0
    for name, shape in specs.array_shapes(n_ticks, n_slots).items():
        total += _npy_exact_bytes(shape, dtypes[name])
    return int(base_bytes) + int(total)


# ---------------------------------------------- Part A v2g scene mirror ----

class NominalParamsError(ValueError):
    """The parameter set is not the plain nominal one (guards/variants on)."""


def _check_nominal(params: dict) -> None:
    mob = params.get("mobil", {})
    for key in ("cooldown_s", "lc_min_speed_mps", "ego_gain_floor_mps2"):
        if float(mob.get(key, 0.0)) != 0.0:
            raise NominalParamsError(
                f"mobil.{key} != 0: the featurizer baseline replays only "
                "the nominal textbook rule set")
    if params.get("lc_variants"):
        raise NominalParamsError(
            "lc_variants present: the featurizer baseline replays only the "
            "nominal rule set")
    if int(params["idm"]["delta"]) != 4:
        raise NominalParamsError("non-nominal idm.delta")


_GAP_MAX = 250.0            # engine.GAP_MAX (free-road sentinel)
_EPS = 1e-9


def _seg_limit1(li: int, s: float, params: dict) -> float:
    starts = params["geometry"]["lane_segment_starts_m"][li]
    lims = params["geometry"]["lane_segment_limits_mps"][li]
    idx = 0
    for k in range(len(starts) - 1, -1, -1):
        if s >= starts[k]:
            idx = k
            break
    return float(lims[idx])


def _idm1(v: float, gap: float, dv: float, v0: float, ip: dict) -> float:
    """Scalar IDM, math-identical to partA_v2g.engine._idm1."""
    import math

    a = float(ip["a_max_mps2"])
    b = float(ip["b_comf_mps2"])
    t = float(ip["T_s"])
    s0 = float(ip["s0_m"])
    delta = int(ip["delta"])
    gap = gap if gap > 1e-6 else 1e-6
    v0 = v0 if v0 > 1e-6 else 1e-6
    s_star = s0 + max(0.0, v * t + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)


def scene_baseline_channels(lane, s_m, v_mps, length_m, present, *,
                            params=None, fast=False):
    """One-step Part A v2g nominal baseline for every vehicle in a scene.

    ``lane`` int [K] (0/1/2 physical, -1 junction/empty), ``s_m`` physical
    lane coordinates [K], ``v_mps`` [K], ``length_m`` [K], ``present`` bool
    [K].  Returns float64 [K] arrays: the six :data:`HDV_BASE_FIELDS`
    channels plus the one-step prediction extras ``lane_next`` (int, the
    post-decision lane), ``s_next``, ``v_next`` and ``merged`` (bool).

    The decisions mirror ``partA_v2g.engine.step`` exactly: mandatory merge
    (accel -> main_right, front-most first, safe-gap tests then MOBIL gain
    vs the urgency-relaxed threshold), then discretionary MOBIL 1->0 and
    0->1 (front-most first, keep-right bias), then own-lane IDM with the
    accel-lane end obstacle, clipped to ``a_clip`` and integrated ballistically
    (``v' = max(0, v + a*dt); s' = s + v'*dt``).  ``merge_urgency`` is the
    rule value ``u`` on the pre-decision accel lane.  ``mobil_gain_*`` is
    the incentive gain of the candidate change (0 when the vehicle is not a
    candidate or is refused before the gain, i.e. below the lead gap
    floor / new-follower b-safe); ``lc_accept_*`` is the discretionary
    acceptance (merges are encoded in ``lane_next``/``merged`` only).
    Junction-lane and absent slots stay all zero.

    ``fast=True`` enables the FEAT1 profiling speed-up (byte-identical
    outputs): the per-lane sorted rosters are cached and rebuilt only after
    an accepted lane change mutates ``(r_lane, r_s)`` -- a reused cache is
    by construction the same list a fresh ``_lane_sorted()`` returns, and
    the engine module is memoized.  The default ``fast=False`` path is the
    untouched reference implementation.
    """
    eng = _engine_cached() if fast else _engine()
    params = params if params is not None else load_nominal_params()
    _check_nominal(params)

    lane = np.asarray(lane, dtype=np.int64)
    s = np.asarray(s_m, dtype=np.float64)
    v = np.asarray(v_mps, dtype=np.float64)
    length = np.asarray(length_m, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    k = lane.shape[0]

    out = {name: np.zeros(k, np.float64) for name in HDV_BASE_FIELDS}
    out["lane_next"] = np.full(k, -1, np.int64)
    out["s_next"] = np.zeros(k, np.float64)
    out["v_next"] = np.zeros(k, np.float64)
    out["merged"] = np.zeros(k, bool)
    if not present.any():
        return out

    mp = params["merge"]
    ip = params["idm"]
    mob = params["mobil"]
    a_clip = tuple(params["dynamics"]["a_clip_mps2"])
    off = tuple(float(x) for x in params["geometry"]["lane_offsets_m"])
    s1z = float(mp["zone_s1_m"])
    zone_s0 = float(mp["zone_s0_m"])
    dt = float(params["dt_s"])
    pol = float(mob["politeness_p"])
    th = float(mob["accel_threshold_mps2"])
    bias = float(mob["keep_right_bias_mps2"])
    b_safe = float(mob["b_safe_mps2"])
    gap_floor = float(mp["gap_floor_m"])
    s0i = float(ip["s0_m"])
    ti = float(ip["T_s"])

    rows = [int(r) for r in np.flatnonzero(present)]
    r_lane = {r: int(lane[r]) for r in rows}
    r_s = {r: float(s[r]) for r in rows}
    r_v = {r: float(v[r]) for r in rows}
    r_len = {r: float(length[r]) for r in rows}

    def _aclip1(a: float) -> float:
        lo, hi = a_clip
        return hi if a > hi else (lo if a < lo else a)

    def _lane_sorted():
        by_lane = {}
        for li in (0, 1, 2):
            rs = [r for r in rows if r_lane[r] == li]
            rs.sort(key=lambda q: r_s[q])
            by_lane[li] = rs
        return by_lane

    # fast path: cache the per-lane sorted rosters; within one tick they can
    # only change when an ACCEPTED lane change mutates (r_lane, r_s), and the
    # two acceptance sites below mark the cache dirty.  A reused cache is
    # therefore the identical list a fresh _lane_sorted() would return.
    _lc_cache: list = []
    _lc_dirty = [True]

    def _ls():
        if fast and not _lc_dirty[0]:
            return _lc_cache[0]
        by = _lane_sorted()
        if fast:
            _lc_cache.clear()
            _lc_cache.append(by)
            _lc_dirty[0] = False
        return by

    def _own_leader_follower(r, by_lane):
        rs = by_lane[r_lane[r]]
        if r not in rs:
            return -1, -1
        j = rs.index(r)
        return (int(rs[j + 1]) if j + 1 < len(rs) else -1,
                int(rs[j - 1]) if j >= 1 else -1)

    def _d_sorted(li):
        rs = [r for r in rows if r_lane[r] == li]
        rs.sort(key=lambda q: off[li] - r_s[q])
        return [float(off[li] - r_s[q]) for q in rs], rs

    def _d_neighbors(d_sorted, ds_rows, d_i):
        import bisect

        j = bisect.bisect_left(d_sorted, d_i)
        return (ds_rows[j - 1] if j >= 1 else -1,
                ds_rows[j] if j < len(ds_rows) else -1)

    def _a_stay(r, by_lane):
        lo, _fo = _own_leader_follower(r, by_lane)
        li = r_lane[r]
        v0 = _seg_limit1(li, r_s[r], params)
        if lo >= 0:
            gap = r_s[lo] - r_len[lo] - r_s[r]
            dv = r_v[r] - r_v[lo]
        else:
            gap, dv = _GAP_MAX, 0.0
        a = _idm1(r_v[r], gap, dv, v0, ip)
        if li == 2:
            gap_end = max(s1z - r_s[r], 0.3)
            a = min(a, _idm1(r_v[r], gap_end, r_v[r], v0, ip))
        return _aclip1(a)

    # -- merge urgency on the pre-decision accel lane -----------------------
    for r in rows:
        if r_lane[r] == 2:
            out["merge_urgency"][r] = float(np.clip(
                1.0 - (s1z - r_s[r]) / float(mp["zone_len_m"]), 0.0, 1.0))

    # -- mandatory merge (accel -> main_right), front-most first ------------
    cand2 = [r for r in rows if r_lane[r] == 2 and zone_s0 <= r_s[r] <= s1z]
    cand2.sort(key=lambda q: -r_s[q])
    d_sorted1, rows1 = _d_sorted(1)
    for r in cand2:
        u = min(max(1.0 - (s1z - r_s[r]) / float(mp["zone_len_m"]), 0.0), 1.0)
        bsafe = float(mp["b_safe_base_mps2"]) + u * (
            float(mp["b_safe_end_mps2"]) - float(mp["b_safe_base_mps2"]))
        d_i = off[2] - r_s[r]
        lrow, frow = _d_neighbors(d_sorted1, rows1, d_i)
        s_ego_t = r_s[r] + off[1] - off[2]
        v0_t = _seg_limit1(1, s_ego_t, params)
        if lrow >= 0:
            gap_lead = r_s[lrow] - r_len[lrow] - s_ego_t
            d_gap_lead = r_s[lrow] - s_ego_t
            v_lead = r_v[lrow]
        else:
            gap_lead, d_gap_lead, v_lead = _GAP_MAX, _GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = s_ego_t - r_len[r] - r_s[frow]
            d_gap_foll = s_ego_t - r_s[frow]
            v_foll = r_v[frow]
        else:
            gap_foll, d_gap_foll, v_foll = _GAP_MAX, _GAP_MAX, 0.0
        # -- the four sequential safe-gap tests (engine _merge_decision) ----
        if lrow >= 0 and d_gap_lead < s0i + max(0.0, r_v[r] * ti):
            continue
        if frow >= 0 and d_gap_foll < s0i:
            continue
        if lrow >= 0:
            b_req_lead = (r_v[r] ** 2 - v_lead ** 2) / \
                (2.0 * max(d_gap_lead - s0i, gap_floor))
            if b_req_lead > bsafe:
                continue
        if frow >= 0:
            b_req_foll = (v_foll ** 2 - r_v[r] ** 2) / \
                (2.0 * max(d_gap_foll - s0i, gap_floor))
            if b_req_foll > bsafe:
                continue
        a_ego_new = _aclip1(_idm1(r_v[r], gap_lead, r_v[r] - v_lead, v0_t,
                                  ip))
        d_new_foll = 0.0
        if frow >= 0:
            v0_f = _seg_limit1(1, r_s[frow], params)
            a_f_after = _aclip1(_idm1(v_foll, gap_foll, v_foll - r_v[r],
                                      v0_f, ip))
            lo, _fo = _own_leader_follower(frow, _ls())
            if lo >= 0:
                cg = r_s[lo] - r_len[lo] - r_s[frow]
                cv = r_v[lo]
            else:
                cg, cv = _GAP_MAX, 0.0
            a_f_before = _aclip1(_idm1(v_foll, cg, v_foll - cv, v0_f, ip))
            d_new_foll = a_f_after - a_f_before
        d_old_foll = 0.0
        by_lane = _ls()
        lo2, fo2 = _own_leader_follower(r, by_lane)
        if fo2 >= 0:
            v_of = r_v[fo2]
            gap_of = r_s[r] - r_len[r] - r_s[fo2]
            v0_o = _seg_limit1(2, r_s[fo2], params)
            a_of_before = _aclip1(_idm1(v_of, gap_of, v_of - r_v[r], v0_o,
                                        ip))
            if lo2 >= 0:
                gap_ol = r_s[lo2] - r_len[lo2] - r_s[fo2]
                v_ol = r_v[lo2]
            else:
                gap_ol, v_ol = _GAP_MAX, 0.0
            a_of_after = _aclip1(_idm1(v_of, gap_ol, v_of - v_ol, v0_o, ip))
            d_old_foll = a_of_after - a_of_before
        gain = (a_ego_new - _a_stay(r, by_lane)) + \
            pol * (d_new_foll + d_old_foll)
        thr = th * (1.0 - u)
        if gain >= thr:
            out["merged"][r] = True
            r_lane[r] = 1
            r_s[r] = r_s[r] + off[1] - off[2]
            _lc_dirty[0] = True
            d_sorted1, rows1 = _d_sorted(1)

    # -- discretionary MOBIL (1->0 left, then 0->1 right), front-most first -
    for src, tgt, to_right, gain_key, acc_key in (
            (1, 0, False, "mobil_gain_left", "lc_accept_left"),
            (0, 1, True, "mobil_gain_right", "lc_accept_right")):
        cand = [r for r in rows if r_lane[r] == src]
        cand.sort(key=lambda q: -r_s[q])
        d_sorted_t, rows_t = _d_sorted(tgt)
        for r in cand:
            li = r_lane[r]
            d_i = off[li] - r_s[r]
            lrow, frow = _d_neighbors(d_sorted_t, rows_t, d_i)
            s_ego_t = r_s[r] + off[tgt] - off[li]
            v0_t = _seg_limit1(tgt, s_ego_t, params)
            if lrow >= 0:
                gap_lead = r_s[lrow] - r_len[lrow] - s_ego_t
                v_lead = r_v[lrow]
            else:
                gap_lead, v_lead = _GAP_MAX, 0.0
            if frow >= 0:
                gap_foll = s_ego_t - r_len[r] - r_s[frow]
                v_foll = r_v[frow]
            else:
                gap_foll, v_foll = _GAP_MAX, 0.0
            if gap_lead < gap_floor:
                continue                    # refused before the gain
            a_ego_new = _aclip1(_idm1(r_v[r], gap_lead, r_v[r] - v_lead,
                                      v0_t, ip))
            a_ego_old = _a_stay(r, _ls())
            a_new_after = a_new_before = 0.0
            if frow >= 0:
                v0_f = _seg_limit1(tgt, r_s[frow], params)
                a_new_after = _aclip1(_idm1(v_foll, gap_foll,
                                            v_foll - r_v[r], v0_f, ip))
                if a_new_after < -b_safe:
                    continue                # refused before the gain
                lo, _fo = _own_leader_follower(frow, _ls())
                if lo >= 0:
                    cg = r_s[lo] - r_len[lo] - r_s[frow]
                    cv = r_v[lo]
                else:
                    cg, cv = _GAP_MAX, 0.0
                a_new_before = _aclip1(_idm1(v_foll, cg, v_foll - cv, v0_f,
                                             ip))
            lo_s, fo_s = _own_leader_follower(r, _ls())
            a_old_after = a_old_before = 0.0
            if fo_s >= 0:
                v_of = r_v[fo_s]
                gap_of = r_s[r] - r_len[r] - r_s[fo_s]
                v0_o = _seg_limit1(li, r_s[fo_s], params)
                a_old_before = _aclip1(_idm1(v_of, gap_of, v_of - r_v[r],
                                             v0_o, ip))
                if lo_s >= 0:
                    gap_ol = r_s[lo_s] - r_len[lo_s] - r_s[fo_s]
                    v_ol = r_v[lo_s]
                else:
                    gap_ol, v_ol = _GAP_MAX, 0.0
                a_old_after = _aclip1(_idm1(v_of, gap_ol, v_of - v_ol, v0_o,
                                            ip))
            gain = (a_ego_new - a_ego_old) + pol * (
                (a_new_after - a_new_before) + (a_old_after - a_old_before))
            out[gain_key][r] = gain
            eff = th - bias if to_right else th + bias
            if gain >= eff:
                out[acc_key][r] = 1.0
                r_lane[r] = tgt
                r_s[r] = r_s[r] + off[tgt] - off[li]
                _lc_dirty[0] = True
                d_sorted_t, rows_t = _d_sorted(tgt)

    # -- IDM dynamics (post-decision own-lane leaders) ----------------------
    flat = [r for li in (0, 1, 2) for r in _ls()[li]]
    lane_arr = np.array([r_lane[r] for r in flat], dtype=np.int64)
    s_arr = np.array([r_s[r] for r in flat])
    v_arr = np.array([r_v[r] for r in flat])
    len_arr = np.array([r_len[r] for r in flat])
    leader = np.full(len(flat), -1, np.int64)
    gap = np.full(len(flat), _GAP_MAX)
    lv = np.zeros(len(flat))
    for li in (0, 1, 2):
        idx = np.flatnonzero(lane_arr == li)
        for j in range(len(idx) - 1):
            f, l_ = int(idx[j]), int(idx[j + 1])
            leader[f] = l_
            gap[f] = s_arr[l_] - len_arr[l_] - s_arr[f]
            lv[f] = v_arr[l_]
    v0 = eng.segment_limits(lane_arr, s_arr, params)
    dv = np.where(leader >= 0, v_arr - lv, 0.0)
    gap_eff = np.where(leader >= 0, gap, _GAP_MAX)
    acc = eng.idm_accel(v_arr, gap_eff, dv, v0, ip)
    on2 = lane_arr == 2
    if on2.any():
        gap_end = np.maximum(s1z - s_arr, 0.3)
        acc_obs = eng.idm_accel(v_arr, gap_end, v_arr.copy(), v0, ip)
        acc = np.where(on2, np.minimum(acc, acc_obs), acc)
    acc = np.clip(acc, a_clip[0], a_clip[1])

    for j, r in enumerate(flat):
        out["idm_accel"][r] = float(acc[j])
        out["lane_next"][r] = int(r_lane[r])
        v_new = max(0.0, r_v[r] + float(acc[j]) * dt)
        out["v_next"][r] = v_new
        out["s_next"][r] = r_s[r] + v_new * dt
    return out


def idea_a_channels_from_shard(shard: dict, *, params=None,
                               fast: bool = False) -> dict:
    """Build the five Idea A arrays from one ``features.featurizer`` shard.

    ``fast=True`` passes the byte-identical fast baseline flag down to
    :func:`scene_baseline_channels` (see its docstring).

    Returns ``{channel: np.ndarray}`` plus ``relabel`` info:
    ``{"n_aborted_tokens": int}`` -- the wall-token rows whose frozen phase
    is ``aborted`` and which the amendment-2 road-end relabel convention
    covers on J4 wall branches (member_left_network in-hold aborts).
    """
    veh = np.asarray(shard["veh_step"])
    mask = np.asarray(shard["veh_mask"])
    wall = np.asarray(shard["wall"])
    wall_mask = np.asarray(shard["wall_mask"])
    n, k = mask.shape
    if veh.shape[:2] != (n, k) or wall.shape[0] != n or \
            wall_mask.shape != (n, wall.shape[1]):
        raise ValueError(f"unexpected shard shape veh={veh.shape} "
                         f"mask={mask.shape} wall={wall.shape}")

    veh_kin = np.ascontiguousarray(kinematics_view(veh)).astype(np.float16)
    hdv_base = np.zeros((n, k, HDV_BASE_F), np.float16)
    cav_plan = np.zeros((n, k, CAV_PLAN_CH_F), np.float16)
    event_tok = np.zeros((n, EVENT_MAX, EVENT_TOK_F), np.float16)
    event_mask = np.zeros((n, EVENT_MAX), np.uint8)

    params = params if params is not None else load_nominal_params()
    lane_oh = veh[..., LANE_LEFT:LANE_ACCEL + 1].astype(np.float32)
    has_lane = lane_oh.sum(axis=-1) > 0
    lane = np.where(has_lane, lane_oh.argmax(axis=-1), -1).astype(np.int64)
    is_cav = veh[..., IS_CAV_IDX] > 0
    is_member = veh[..., WALL_MEMBER_IDX] > 0
    s_m = veh[..., S_DOWN].astype(np.float64) + \
        np.asarray(LANE_OFFSETS_M, np.float64)[np.clip(lane, 0, 2)]
    v_mps = veh[..., V_IDX].astype(np.float64)
    length_m = veh[..., LENGTH_IDX].astype(np.float64)
    present = mask.astype(bool)

    active_phases = (0, 1, 2)        # requested/assembling/holding
    wall_episode = _wall_episode_ordinals(wall_mask.astype(bool))

    for tick in range(n):
        pres = present[tick]
        if not pres.any():
            continue
        sc = scene_baseline_channels(
            lane[tick], s_m[tick], v_mps[tick], length_m[tick], pres,
            params=params, fast=fast)
        base = np.stack([sc[name] for name in HDV_BASE_FIELDS], axis=-1)
        hdv_base[tick][pres] = base[pres].astype(np.float16)
        hdv_base[tick][is_cav[tick]] = 0.0        # per-HDV: CAV rows zero

        # active wall row for this tick (first active-phase token)
        wrow = -1
        for r in range(wall.shape[1]):
            if wall_mask[tick, r] and \
                    int(np.argmax(wall[tick, r, :5])) in active_phases:
                wrow = r
                break
        cav = is_cav[tick] & pres
        cav_plan[tick, cav, 0] = (v_mps[tick][cav] /
                                  CAV_TGT_SPEED_SCALE).astype(np.float16)
        cav_plan[tick, cav, 1] = (veh[tick, cav, A_IDX].astype(np.float64) /
                                  CAV_ACCEL_SCALE).astype(np.float16)
        member = cav & is_member[tick]
        if wrow >= 0:
            cav_plan[tick, member, 2] = wall[tick, wrow, 5]  # speed/40
            cav_plan[tick, member, 4] = wall[tick, wrow, 7]  # hold/600
            if member.any():
                ids = wall_episode[tick, wrow]
                cav_plan[tick, member, 3] = np.float16(
                    ids / CAV_WALL_ID_SCALE)
        cav_plan[tick, cav & ~member, 3] = np.float16(
            -1.0 / CAV_WALL_ID_SCALE)                      # no wall: -1/4
        lt = np.where(lane[tick] >= 0, lane[tick], -1)
        for li in range(3):
            cav_plan[tick, cav, 5 + li] = (lt[cav] == li).astype(np.float16)

    n_aborted = int(((wall.astype(np.float32)[..., 4] > 0)
                     & (wall_mask.astype(np.int32) > 0)).sum())
    return dict(veh_kin=veh_kin, hdv_base=hdv_base, cav_plan_ch=cav_plan,
                event_tok=event_tok, event_mask=event_mask,
                relabel={"n_aborted_tokens": n_aborted})


def _wall_episode_ordinals(wall_mask: np.ndarray) -> np.ndarray:
    """Per-row episode ordinals from the ``[n, R]`` wall mask.

    A row's mask 0->1 transition starts an episode; ordinals count episodes
    in (first tick, row) order, which equals the v1 ``meta.wall_ids``
    first-appearance order.  Returns int64 ``[n, R]`` with -1 = no token.
    Row continuity across a mid-life row swap (only possible with
    overlapping walls) is a documented approximation.
    """
    n, r_max = wall_mask.shape
    ep = np.full((n, r_max), -1, np.int64)
    ordinal = 0
    prev = np.zeros(r_max, bool)
    for tick in range(n):
        cur = wall_mask[tick]
        started = cur & ~prev
        if started.any():
            for r in np.flatnonzero(started):
                ep[tick, r] = ordinal
                ordinal += 1
        cont = cur & prev
        if cont.any():
            for r in np.flatnonzero(cont):
                ep[tick, r] = ep[tick - 1, r]
        prev = cur
    return ep
