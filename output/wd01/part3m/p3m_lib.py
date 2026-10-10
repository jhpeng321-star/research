"""WD-01 part 3m shared library: TTC<3 s pair extraction IDENTICAL to the
frozen window_metrics definition, event attribution (post-merge /
post-lane-change / pure car-following), segment bins, and the inherited
p3l split / spread-calibration math.

TTC definition parity (part 3m TASK A requirement): on BOTH sides the
exposure unit is a same-physical-lane CONSECUTIVE follower->leader pair
(next larger mapped-lane s) with gap > 0, closing speed > 0 and
gap/closing < 3 s; each flagged pair-tick contributes dt = 0.5 veh s.
The PRED side counts every predicted vehicle; the TRUTH side counts the
MATCHED vehicle set (predicted vids present in the truth tick), exactly
as engine.window_metrics does.  diag_p3m.py asserts per window that this
module reproduces window_metrics' pred_/true_ttc3_exposure exactly.

ZERO simulation: no SUMO/TraCI import anywhere in part 3m; the only
dynamics is the part-3 rollout engine (pure torch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "part3k"), str(HERE.parent / "part3l")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decompose_p3k as D  # noqa: E402  (segments, Tee, glue helpers)
import p3lib as P3  # noqa: E402
import p3l_lib as L3  # noqa: E402  (splits_3l + spread math, inherited)

DT = 0.5
TTC_THR = 3.0
POST_WINDOW = 10               # 5 s in 0.5 s steps (event attribution)
# event classes (pair-level; priority merge > discretionary LC > pure CF)
EV_MERGE = "post_merge"
EV_LC = "post_lc"
EV_CF = "pure_cf"
TTC_BANDS = ("[0,1)", "[1,1.5)", "[1.5,2)", "[2,3)")
CELL_KINDS = ("lane", "seg", "ev", "ev_lane", "who", "vt", "ttc",
              "dawdle")


# ------------------------------------------------------------ TTC pairs --

def ttc_pairs(lane, s, v, thr=TTC_THR):
    """(follower_row, leader_row, gap, ttc) for every same-lane
    CONSECUTIVE pair with gap > 0, closing > 0 and ttc < thr.

    Ordering identical to engine.ttc_stats: per lane, vehicles sorted
    front (largest s) first, pairs are consecutive in that order, i.e.
    the leader is the next larger s -- the same leader selection
    window_metrics applies to BOTH sides (junction rows carry the mapped
    lane on both sides, so the selection is identical across junctions
    and lane boundaries)."""
    lane = np.asarray(lane)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    out = []
    for li in (0, 1, 2):
        ids = np.where(lane == li)[0]
        if len(ids) < 2:
            continue
        o = ids[np.argsort(-s[ids], kind="stable")]  # front first
        for j in range(len(o) - 1):
            l_, f = int(o[j]), int(o[j + 1])
            gap = s[l_] - s[f]
            closing = v[f] - v[l_]
            if gap > 0 and closing > 0:
                t = gap / closing
                if t < thr:
                    out.append((f, l_, float(gap), float(t)))
    return out


def pair_exposure_s(lane, s, v, thr=TTC_THR):
    """Vehicle-seconds of TTC<thr exposure in one 0.5 s snapshot
    (== len(ttc_pairs) * DT; the window_metrics unit)."""
    return len(ttc_pairs(lane, s, v, thr)) * DT


def band_idx(t):
    """TTC sub-band: [0,1) -> 0, [1,1.5) -> 1, [1.5,2) -> 2, [2,3) -> 3."""
    return 0 if t < 1.0 else 1 if t < 1.5 else 2 if t < 2.0 else 3


# ------------------------------------------------------ event attribution --

def pred_event_ticks(snaps, K):
    """Per-vid lane-change step indices from pred snapshots:
    {vid: (merge_js, lc_js)}; merge = 2->1 (the M2 mandatory merge),
    lc = any other change (0<->1 discretionary).  j indexes steps 1..K
    (snapshot j is AFTER step j)."""
    events = {}
    prev = {}
    sn0 = snaps[0]
    for r, vid in enumerate(sn0["vid"]):
        prev[int(vid)] = int(sn0["lane"][r])
    for j in range(1, K + 1):
        sn = snaps[j]
        for r, vid in enumerate(sn["vid"]):
            vid = int(vid)
            l = int(sn["lane"][r])
            p = prev.get(vid)
            if p is not None and p != l:
                m, lc = events.get(vid, ((), ()))
                if (p, l) == (2, 1):
                    events[vid] = (m + (j,), lc)
                else:
                    events[vid] = (m, lc + (j,))
            prev[vid] = l
    return events


def truth_event_ticks(ra, k0, K):
    """Per-vid lane-change ABSOLUTE ticks from the declared p2C events:
    ({vid: merge_abs_ticks}, {vid: lc_abs_ticks}); merge = lc_from 2 ->
    lc_to 1.  ABSOLUTE ticks so callers compare against k0 + j on either
    side."""
    m, lc = {}, {}
    for v, t, f, tto in zip(ra.lc_vid, ra.lc_tick, ra.lc_from, ra.lc_to):
        if not (k0 < t <= k0 + K):
            continue
        d = m if (int(f), int(tto)) == (2, 1) else lc
        d.setdefault(int(v), []).append(int(t))
    return ({v: tuple(ts) for v, ts in m.items()},
            {v: tuple(ts) for v, ts in lc.items()})


def post_event(events_m, abs_tick, window=POST_WINDOW):
    """True if a MERGE of this vehicle happened within the previous
    `window` steps (the change tick itself included)."""
    for t in events_m:
        if 0 <= abs_tick - t <= window:
            return True
    return False


def post_event_lc(events_l, abs_tick, window=POST_WINDOW):
    """True if a discretionary lane change happened within the previous
    `window` steps (the change tick itself included)."""
    for t in events_l:
        if 0 <= abs_tick - t <= window:
            return True
    return False


def pair_event_class(m_f, l_f, m_l, l_l, abs_tick):
    """(event_class, who): pair-level class with priority merge >
    discretionary lc > pure car-following; `who` names which side of the
    pair merged (foll / lead / both / '-')."""
    mf, ml = post_event(m_f, abs_tick), post_event(m_l, abs_tick)
    if mf and ml:
        return EV_MERGE, "both"
    if mf:
        return EV_MERGE, "foll"
    if ml:
        return EV_MERGE, "lead"
    if post_event_lc(l_f, abs_tick) or post_event_lc(l_l, abs_tick):
        return EV_LC, "-"
    return EV_CF, "-"


def seg_of_d(d):
    """Segment bin (p3j/p3k convention): downstream d<0, merge_edge
    0-75 m, rsu_window 75-800 m, upstream >= 800 m."""
    return D.seg_bin(d)


# ------------------------------------------------------ window accounting --

def truth_arrays_for_tick(ra, abs_tick, sn_vids):
    """(lane, s, v, typ) arrays for the MATCHED set at one tick:
    predicted vids present in the truth tick (window_metrics' `keep`
    convention), in snapshot row order; unmatched rows get lane -1."""
    ids = ra.tick_rows(abs_tick)
    tv = {int(v): (int(l), float(ss), float(vv), int(tp))
          for v, l, ss, vv, tp in zip(ra.vid[ids], ra.lane_m[ids],
                                      ra.s[ids], ra.v[ids], ra.typ[ids])}
    lane, s, v, typ = [], [], [], []
    for vid in sn_vids:
        row = tv.get(int(vid))
        if row is None:
            lane.append(-1)
            s.append(np.nan)
            v.append(np.nan)
            typ.append(-1)
            continue
        lane.append(row[0])
        s.append(row[1])
        v.append(row[2])
        typ.append(row[3])
    return (np.asarray(lane), np.asarray(s, dtype=np.float64),
            np.asarray(v, dtype=np.float64), np.asarray(typ))


def window_cells(r, ra, K, k0=None, dt=DT, thr=TTC_THR):
    """Diagnosis accounting for ONE rollout vs truth over steps 1..K.

    Returns (cells, totals).  cells maps cell-kind -> cell key -> dict
    with six counters:
      pa  pred_all_s   all predicted vehicles (the frozen metric's PRED
                       convention)
      pm  pred_matched_s  MATCHED vehicles, PREDICTED dynamics
      tr  truth_s      MATCHED vehicles, TRUE dynamics (the frozen
                       metric's TRUTH convention)
      na/nm/nt         pair-tick counts of the three views
    The pm and tr views share the identical vehicle set and row order
    (matched, snapshot row order) and differ ONLY in the state arrays
    (predicted vs true), so pm - tr isolates BEHAVIOUR and
    pa - pm isolates vehicle-SET composition.  kinds: lane, seg, ev,
    ev_lane, who, vt, ttc, dawdle.  totals pa/tr equal
    window_metrics' pred_ttc3_exposure / true_ttc3_exposure (asserted
    by the caller).  vtype: pred views = the rollout's SAMPLED vType,
    truth view = the supervision label ra.typ."""
    if k0 is None:
        k0 = r.k0
    snaps = r.snapshots
    pev = pred_event_ticks(snaps, K)
    # PER-SIDE event maps: the pred side is attributed with the events
    # the PREDICTED vehicles actually made (observed from the
    # snapshots); the truth side with the DECLARED truth events.  A
    # shared vid -> events map would mislabel pred vehicles whose truth
    # twin already changed lanes (e.g. still queued on the accel lane
    # while the truth twin merged) -- that contamination is exactly
    # what part 3m TASK A must not produce.
    ev_m_p = {v: tuple(k0 + j for j in js[0]) for v, js in pev.items()
              if js[0]}
    ev_l_p = {v: tuple(k0 + j for j in js[1]) for v, js in pev.items()
              if js[1]}
    ev_m_t, ev_l_t = truth_event_ticks(ra, k0, K)
    dw = float(getattr(r, "dawdle_scale", 1.0))
    cells = {k: {} for k in CELL_KINDS}

    def put(kind, key, col, s_inc, n_inc):
        c = cells[kind].setdefault(key, dict(pa=0.0, pm=0.0, tr=0.0,
                                             na=0, nm=0, nt=0))
        c[col] += s_inc
        c[{"pa": "na", "pm": "nm", "tr": "nt"}[col]] += n_inc

    t3p = t3t = t3m = 0
    for j in range(1, K + 1):
        sn = snaps[j]
        lane_p, s_p, v_p = sn["lane"], sn["s"], sn["v"]
        tl_, ts_, tv_, typ_ = truth_arrays_for_tick(ra, k0 + j, sn["vid"])
        keep = tl_ >= 0
        vmap = r.vtype_map              # vid -> sampled vType (part 3m)

        def vt_of(row_vids):
            return np.fromiter((vmap.get(int(v), -1) for v in row_vids),
                               dtype=np.int64, count=len(row_vids))

        views = (
            # side, lane, s, v, vids, vtypes, cell col
            ("pred", lane_p, s_p, v_p, sn["vid"], vt_of(sn["vid"]), "pa"),
            ("pm", lane_p[keep], s_p[keep], v_p[keep],
             sn["vid"][keep], vt_of(sn["vid"][keep]), "pm"),
            ("truth", tl_[keep], ts_[keep], tv_[keep],
             sn["vid"][keep], typ_[keep], "tr"),
        )
        for side, ln, ss, vv, vids, vts, col in views:
            ev_m = ev_m_p if side != "truth" else ev_m_t
            ev_l = ev_l_p if side != "truth" else ev_l_t
            for f, l_, gap, t in ttc_pairs(ln, ss, vv, thr):
                vid_f, vid_l = int(vids[f]), int(vids[l_])
                ev, who = pair_event_class(
                    ev_m.get(vid_f, ()), ev_l.get(vid_f, ()),
                    ev_m.get(vid_l, ()), ev_l.get(vid_l, ()), k0 + j)
                seg = seg_of_d(P3.OFFSETS[int(ln[f])] - ss[f])
                vt = int(vts[f]) if vts is not None else -1
                put("lane", (int(ln[f]),), col, dt, 1)
                put("seg", (seg,), col, dt, 1)
                put("ev", (ev,), col, dt, 1)
                put("ev_lane", (ev, int(ln[f])), col, dt, 1)
                put("who", (ev, who), col, dt, 1)
                put("vt", (vt,), col, dt, 1)
                put("ttc", (TTC_BANDS[band_idx(t)],), col, dt, 1)
                put("dawdle", (dw,), col, dt, 1)
                if side == "pred":
                    t3p += 1
                elif side == "pm":
                    t3m += 1
                else:
                    t3t += 1
    totals = dict(pred_s=t3p * dt, pred_matched_s=t3m * dt,
                  truth_s=t3t * dt, pred_n=t3p, pred_matched_n=t3m,
                  truth_n=t3t)
    return cells, totals


def truth_ttc3_exposure(ra, k0, K, dt=DT, thr=TTC_THR):
    """Truth TTC<thr vehicle-seconds over steps 1..K computed over ALL
    truth rows at each tick.  This is the fixed FIT TARGET of part 3m
    TASK B: unlike the gate metric's truth side (restricted to the
    matched vehicle set) it is a property of the window alone, so fit
    stages can compare candidates against one number per window."""
    total = 0.0
    for j in range(1, K + 1):
        ids = ra.tick_rows(k0 + j)
        n = len(ttc_pairs(ra.lane_m[ids], ra.s[ids], ra.v[ids], thr))
        total += n * dt
    return total


def merge_cells(target, cells, weight=1.0):
    """Accumulate one window's cells into `target` (per-kind dicts;
    vehicle-seconds scale by weight, counts are additive)."""
    for kind, cmap in cells.items():
        tgt = target.setdefault(kind, {})
        for key, c in cmap.items():
            acc = tgt.setdefault(key, dict(pa=0.0, pm=0.0, tr=0.0,
                                           na=0, nm=0, nt=0))
            acc["pa"] += weight * c["pa"]
            acc["pm"] += weight * c["pm"]
            acc["tr"] += weight * c["tr"]
            acc["na"] += c["na"]
            acc["nm"] += c["nm"]
            acc["nt"] += c["nt"]


def cells_to_json(cells):
    """Make a cells dict JSON-serialisable: tuple keys become
    '|'-joined strings (json.dumps rejects tuple keys even with
    default=str -- caught by the part 3m TASK A run)."""
    return {kind: {"|".join(map(str, key)): c
                   for key, c in cmap.items()}
            for kind, cmap in cells.items()}


def excess_table(cells, key_sort="pa-tr"):
    """[(excess_s, pred_all_s, pred_matched_s, truth_s, n_pred_all,
    n_pred_matched, n_truth, key)] sorted by the requested excess
    (default pa - tr, the frozen metric's bias) descending -- the
    top-contributor table.  pm - tr isolates behaviour, pa - pm
    vehicle-set composition."""
    col = {"pa-tr": ("pa", "tr"), "pm-tr": ("pm", "tr"),
           "pa-pm": ("pa", "pm")}[key_sort]
    rows = []
    for key, c in cells.items():
        rows.append((round(c[col[0]] - c[col[1]], 3),
                     round(c["pa"], 3), round(c["pm"], 3),
                     round(c["tr"], 3), c["na"], c["nm"], c["nt"], key))
    rows.sort(key=lambda r: -r[0])
    return rows
