"""Prefix-only restart algebra shared by the real-tape and imagined-state adapter.

No learned model or simulator imports. Vehicle identities are alignment/CRN keys,
never value features. Source initialization is kept outside the observation path.
"""
import numpy as np

from .core import finite
from .reconstruct import LENGTH, OBS_KEYS, SPEED_LIMIT, boundary_events, flow_at, project


def prefix(tape, t0):
    tick = int(round(2*t0))
    out = {k: np.asarray(tape[k])[:tick+1] for k in OBS_KEYS}
    if len(out['time_s']) != tick+1 or not np.array_equal(out['time_s'], np.arange(tick+1)*.5):
        raise ValueError('missing/unordered observation prefix')
    # Access names only for identities already seen; no future registry entries.
    seen = np.unique(out['vid_index'][out['present'].astype(bool)])
    if (seen < 0).any():
        raise ValueError('unknown observed identity')
    out['names'] = {int(i): str(tape['vehicle_ids'][i]) for i in seen}
    return out


def current_names(p):
    return [p['names'][int(i)] for i in p['vid_index'][-1, p['present'][-1].astype(bool)]]


def align_history(p, columns):
    """Native 121 frames ending at t0, including exited historical neighbours.

    First columns are a common current-identity union across arms. Extra columns
    hold historical-only neighbours so their leader evidence survives featurizing;
    callers slice to the first len(columns) AFTER featurizing.
    """
    lo = len(p['time_s'])-121
    if lo < 0 or len(set(columns)) != len(columns):
        raise ValueError('need sixty seconds and unique identity columns')
    hist_ids = np.unique(p['vid_index'][lo:][p['present'][lo:].astype(bool)])
    extra = sorted({p['names'][int(i)] for i in hist_ids}-set(columns))
    names = [*columns, *extra]
    index = {name: j for j, name in enumerate(names)}
    kin = np.zeros((121, len(names), 7), np.float32)
    mask = np.zeros((121, len(names)), bool)
    cav = np.zeros((121, len(names)), bool)
    lane = np.full((121, len(names)), -1, np.int64)
    for h, tick in enumerate(range(lo, len(p['time_s']))):
        for j in np.flatnonzero(p['present'][tick]):
            col = index[p['names'][int(p['vid_index'][tick, j])]]
            ln = int(p['lane'][tick, j])
            if ln not in (-1, 0, 1, 2):
                raise ValueError('invalid recorded lane')
            kin[h, col, 0] = p['s_m'][tick, j]
            if ln >= 0:
                kin[h, col, 1+ln] = 1.
            kin[h, col, 4:7] = [p['v_mps'][tick, j], p['a_mps2'][tick, j], LENGTH]
            lane[h, col] = ln
            mask[h, col] = True
            cav[h, col] = p['is_cav'][tick, j]
    finite(kin[mask])
    return dict(kin=kin, mask=mask, target_mask=mask[:-1] & mask[1:],
                lane=lane, cav=cav, names=names)


def entry_records(p):
    """Recorded boundary detector events, including local entry-speed evidence."""
    boundary_events(p)  # independent boundary/identity audit, before any statistics
    seen, records = set(), []
    for tick, time in enumerate(p['time_s']):
        for j in np.flatnonzero(p['present'][tick]):
            identity = int(p['vid_index'][tick, j])
            if identity in seen:
                continue
            seen.add(identity)
            ln = int(p['lane'][tick, j])
            pos = float(p['s_m'][tick, j])
            idx = np.flatnonzero(p['present'][tick] & (p['lane'][tick] == ln))
            idx = idx[np.argsort(p['s_m'][tick, idx])]
            ahead = idx[p['s_m'][tick, idx] > pos]
            gap = float(p['s_m'][tick, ahead[0]]-pos-LENGTH) if len(ahead) else None
            records.append(dict(time=float(time), lane=ln, s_m=pos,
                                v_mps=float(p['v_mps'][tick, j]),
                                is_cav=bool(p['is_cav'][tick, j]),
                                local_speed=max(float(p['v_mps'][tick, idx[:3]].mean()), 1.),
                                gap=gap if gap is not None and gap > 0 else None))
    return records


def causal_queue_prior(records, warmup_queue, t0):
    """WM-only prior: warmup observed queue + past rate - actual boundary flow.

    Late queues are NOT reconstructed from seeded request schedules or outcomes.
    Use the existing recorded t300 queue/rate initialization, propagated in mean
    through the observed prefix. This is neither a V input nor a real label.
    """
    q = finite(warmup_queue, (2,))
    times = np.array([r['time'] for r in records])
    entrance = np.array([int(r['lane'] == 2) for r in records])
    inserted300 = np.array([np.sum((times <= 300) & (entrance == e)) for e in (0, 1)])
    inserted = np.array([np.sum((times <= t0) & (entrance == e)) for e in (0, 1)])
    rate = (inserted300+q)/300.
    backlog = np.maximum(0., q+rate*(t0-300)-(inserted-inserted300))
    # Entry process uses integer vehicles, with fixed nearest rounding (no tuning).
    shares = np.array([np.mean([r['is_cav'] for r in records if int(r['lane'] == 2) == e])
                       if np.any(entrance == e) else 0. for e in (0, 1)])
    return np.r_[np.rint(backlog), rate, shares].astype(np.float32)


def road_rectangle(before, after, dt=.5):
    """Observer's transition-union/end-speed convention, including births/exits."""
    pre, post = np.asarray(before['present'], bool), np.asarray(after['present'], bool)
    speed = np.where(post, after['v_mps'], before['v_mps'])
    return dt*np.where(pre | post, np.maximum(0., 1.-speed/SPEED_LIMIT), 0.).sum(-1)


def imagined_observation(frame, events, time_s):
    """One observation operator; events include real prefix + imagined births."""
    kin = np.column_stack([frame[k] for k in ('s_m', 'lane', 'v_mps', 'a_mps2')])
    return project(kin, frame['present'], frame['is_cav'], flow_at(events, time_s), time_s)
