"""Observable moment enrichment from WM-D native prefix tapes (schema wzqual_native_tape/1).

A tape is the SUMO-identical native prefix of a labelled world x cell (WM-D, identity-gated 74/74
against the labelled natives), every vehicle every 0.5 s. At a moment t0 the proposer may use only
RSU-observable content, so enrichment keeps:

  cavs      CAV self-states at t0 (must equal the moment's all_cavs_t0: identity check)
  vehicles  CAV-DETECTED neighbours (N1 RSU source 'cav_reports': |dx| <= 100 m and |dWM lane| <= 1
            of at least one CAV); vehicles no CAV detects are dropped (only their COUNT is reported
            as coverage bookkeeping, never passed to the WM)
  counter   RSU entrance counters [2, 30]: insertions (first appearance on the road) per 10 s bin
            before t0, bin 0 newest; entrance 0 = through lanes (SUMO up 1/2), 1 = closing lane (up 0)

Never read: vType / driver parameters (not in the tape), scheduled departures (depart_s), the
source backlog, anything after t0. Lengths are nominal (5 m).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry as G

SCHEMA = 'wzqual_native_tape/1'
REPORT_RANGE_M = 100.0            # N1 RsuDescriptor.report_range_m (= sensor range)
REPORT_LANE_REACH = 1             # N1 RsuDescriptor.report_lane_reach
N_BINS, BIN_S = 30, 10.0          # wms.model.entries.N_COUNTER_BINS / COUNTER_BIN_S
IDENTITY_TOL_M = 0.05


def tape_path(tapes_root, m):
    return Path(tapes_root) / 'runs' / m['cell'] / f"s{m['seed']}" / 'prefix' / 'TAPE.npz'


def load_tape(path):
    path = G.check_path(path)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta_json']))
    if meta.get('schema') != SCHEMA:
        raise ValueError(f'{path}: not a {SCHEMA} tape')
    G.check_seed(meta['seed'], where=f'({path})')
    keep = ('time_s', 'step', 'vid', 'edge', 'lane', 'x', 'v', 'ids', 'is_cav', 'edges')
    return dict({k: z[k] for k in keep}, meta=meta)


def first_appearance(tape):
    """(entry time, entrance) per vehicle index from the first row of each vehicle."""
    vid = tape['vid']
    n = len(tape['ids'])
    first = np.full(n, -1, np.int64)
    order = np.argsort(tape['step'], kind='stable')
    seen = np.zeros(n, bool)
    for r in order:                                   # rows are step-ordered already; stable fallback
        i = vid[r]
        if not seen[i]:
            seen[i] = True
            first[i] = r
    t = np.where(first >= 0, tape['time_s'][tape['step'][np.maximum(first, 0)]], np.nan)
    edges = [str(e) for e in tape['edges']]
    ent = np.full(n, -1, np.int64)
    for i in np.flatnonzero(first >= 0):
        r = first[i]
        ent[i] = 1 if (edges[tape['edge'][r]] == 'up' and int(tape['lane'][r]) == 0) else 0
    return t, ent


def counter_at(tape, t0, entries=None):
    t, ent = entries if entries is not None else first_appearance(tape)
    c = np.zeros((2, N_BINS))
    for e in (0, 1):
        dt = t0 - t[(ent == e) & np.isfinite(t)]
        dt = dt[(dt >= 0) & (dt < N_BINS * BIN_S)]
        b = np.floor(dt / BIN_S + 1e-9).astype(int)       # entry at t0 - 0.5 -> bin 0
        np.add.at(c[e], np.clip(b, 0, N_BINS - 1), 1)
    return c.tolist()


def frame_at(tape, t0):
    k = np.flatnonzero(np.abs(tape['time_s'] - float(t0)) < 1e-6)
    if len(k) != 1:
        raise KeyError(f'tape has no step at t0={t0}')
    rows = np.flatnonzero(tape['step'] == k[0])
    edges = [str(e) for e in tape['edges']]
    out = []
    for r in rows:
        i = int(tape['vid'][r])
        out.append(dict(id=str(tape['ids'][i]), edge=edges[tape['edge'][r]], lane=int(tape['lane'][r]),
                        x=float(tape['x'][r]), v=float(tape['v'][r]), is_cav=bool(tape['is_cav'][i])))
    return out


def observable(frame, report_range=REPORT_RANGE_M, lane_reach=REPORT_LANE_REACH):
    """(cavs, detected non-CAV vehicles, n_undetected) under the N1 cav_reports rule."""
    cavs = [r for r in frame if r['is_cav']]
    cl = [(float(c['x']), G.wm_lane(c['edge'], c['lane'])) for c in cavs]
    det, hidden = [], 0
    for r in frame:
        if r['is_cav']:
            continue
        wl = G.wm_lane(r['edge'], r['lane'])
        seen = wl is not None and any(abs(float(r['x']) - cx) <= report_range and cln is not None
                                      and abs(wl - cln) <= lane_reach for cx, cln in cl)
        if seen:
            det.append(dict(id=r['id'], edge=r['edge'], lane=r['lane'], x=round(r['x'], 3), v=round(r['v'], 4),
                            is_cav=False))
        else:
            hidden += 1
    return cavs, det, hidden


def enrich(moments_path, tapes_root, out_path):
    """Moments + observed{vehicles, counter, coverage, identity}; refuses a CAV identity mismatch."""
    moments = json.loads(G.check_path(moments_path).read_text())
    cache, out, report = {}, [], dict(moments=0, enriched=0, missing_tape=0, identity_fail=[])
    for m in moments:
        if not m.get('valid'):
            out.append(m)
            continue
        report['moments'] += 1
        G.check_seed(m['seed'], where=f"(moment {m.get('sid')})")
        tp = tape_path(tapes_root, m)
        if not tp.exists():
            report['missing_tape'] += 1
            out.append(m)
            continue
        key = str(tp)
        if key not in cache:
            tape = load_tape(tp)
            cache.clear()
            cache[key] = (tape, first_appearance(tape))
        tape, entries = cache[key]
        if int(tape['meta']['seed']) != int(m['seed']) or tape['meta']['cell'] != m['cell']:
            raise ValueError(f"{m['sid']}: tape {tp} belongs to another world-cell")
        frame = frame_at(tape, m['t0'])
        cavs, det, hidden = observable(frame)
        ref = {c['id']: c for c in m['all_cavs_t0']}
        got = {c['id']: c for c in cavs}
        dev = max([abs(float(got[i]['x']) - float(ref[i]['x'])) for i in ref if i in got] or [0.0])
        if set(ref) != set(got) or dev > IDENTITY_TOL_M:
            report['identity_fail'].append(dict(sid=m['sid'], missing=sorted(set(ref) ^ set(got))[:5], max_dx=dev))
            out.append(m)
            continue
        m = dict(m)
        m['observed'] = dict(vehicles=det, counter=counter_at(tape, float(m['t0']), entries),
                             coverage=dict(cavs=len(cavs), detected=len(det), undetected=hidden,
                                           rule='N1 cav_reports |dx|<=100 m, |dWM lane|<=1'),
                             tape=dict(path=str(tp), seed=int(tape['meta']['seed']), cell=tape['meta']['cell']),
                             identity_max_dx_m=round(dev, 4))
        out.append(m)
        report['enriched'] += 1
    out_path = G.check_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x') as fh:
        fh.write(json.dumps(out))
    return report
