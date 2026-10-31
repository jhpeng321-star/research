"""Prospectively fixed worlds, budget, observable group and option definitions."""
import hashlib
import json
from pathlib import Path
import re

CELLS = tuple((p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700))
HIGH = (7, 8, 10, 11)
TIMES = (300, 360, 420, 480, 540)
DATA_CAP = 1_489_200
LIVE_CAP = 100_000
CAP = DATA_CAP + LIVE_CAP
FORBIDDEN = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype|assignment', re.I)


def checked(path):
    p = Path(path)
    if FORBIDDEN.search(str(p)) or FORBIDDEN.search(str(p.resolve())):
        raise ValueError('restricted path')
    return p


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def worlds():
    out = []
    for cell, (p, d) in enumerate(CELLS):
        for r in range(12 if cell in HIGH else 4):
            out.append(dict(seed=981000+len(out), cell=cell, repeat=r, penetration=p, demand=d))
    assert len(out) == 80
    return out


def dev_worlds():
    return [dict(seed=981080+c, cell=c, repeat=0, penetration=p, demand=d)
            for c, (p, d) in enumerate(CELLS)]


def guard(world):
    if world not in worlds()+dev_worlds():
        raise ValueError('not an exact registered world/cell')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'], world['demand'], 'exposed')
    check_case(world['seed'], world['demand'])


def install_seed_scope():
    """Narrow process-local extension of BOTH guards; frozen runtime files unchanged."""
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(w['seed'], w['demand']) for w in worlds()+dev_worlds()}
    original_s, original_c = s.require_development_case, c.check_case
    def base(seed, loading, role):
        if type(seed) is int and type(loading) is int and role == 'exposed' and (seed, loading) in allowed:
            return
        return original_s(seed, loading, role)
    def c4(seed, demand_vph):
        if type(seed) is int and type(demand_vph) is int and (seed, demand_vph) in allowed:
            return base(seed, demand_vph, 'exposed')
        return original_c(seed, demand_vph)
    s.require_development_case, c.check_case = base, c4


def group(rows):
    """k=2 nearest sensed upstream mainline CAVs, solely geometric selection."""
    cavs = [r for r in rows if r['is_cav'] and r['lane'] in (0, 1) and -100 <= r['s_m'] <= 0]
    return [r['vid'] for r in sorted(cavs, key=lambda r:(abs(r['s_m']), r['lane'], r['vid']))[:2]]


def option(option_id, rows, setting, members=None):
    selected = group(rows)
    if option_id == 'O0':
        return dict(id='O0', members=[])
    family = dict(O1='L2', O2='L3', O3='L4', O4='L1')[option_id]
    # Grid setting is part of the context/action, not a hidden state feature.
    level, window = setting // 2, (30, 60)[setting % 2]
    magnitude = ((3., 6.) if family in ('L2', 'L3') else (2., 3.))[level]
    active = len(selected) == 2
    if option_id == 'O2':
        byid = {r['vid']:r for r in rows}
        active = active and len({byid[v]['lane'] for v in selected}) == 2
    if option_id == 'O3':
        byid = {r['vid']:r for r in rows}
        active = active and all(byid[v]['lane'] == 1 for v in selected)
        active = active and any(r['lane'] == 2 and -100 <= r['s_m'] <= 75.44 for r in rows)
    return dict(id=option_id, family=family, magnitude=magnitude, window=window,
                members=(selected if members is None else members) if active else [],
                pair=selected, available=active, setting=setting)


def observable(rows, t, setting):
    """No world seed, demand, penetration, types, driver values or source truth."""
    out = dict(time_s=float(t), setting=float(setting))
    for lane in range(3):
        sensed = [r for r in rows if r['lane'] == lane and -100 <= r['s_m'] <= 75.44]
        speeds = [r['v_mps'] for r in sensed]
        out.update({f'l{lane}_count':float(len(sensed)),
                    f'l{lane}_cavs':float(sum(r['is_cav'] for r in sensed)),
                    f'l{lane}_speed':sum(speeds)/len(speeds) if speeds else 0.,
                    f'l{lane}_queue':float(sum(v < 5 for v in speeds)),
                    f'l{lane}_density':len(sensed)/.17544})
    for i, vid in enumerate(group(rows)):
        own = next(r for r in rows if r['vid'] == vid)
        front = [r['s_m']-own['s_m']-5 for r in rows if r['lane'] == own['lane'] and r['s_m'] > own['s_m']]
        back = [own['s_m']-r['s_m']-5 for r in rows if r['lane'] == own['lane'] and r['s_m'] < own['s_m']]
        out.update({f'g{i}_s':own['s_m'], f'g{i}_v':own['v_mps'], f'g{i}_lane':float(own['lane']),
                    f'g{i}_gap':max(0., min(front)) if front else 1000.,
                    f'g{i}_backgap':max(0., min(back)) if back else 1000.})
    for i in range(2):
        for key in ('s', 'v', 'lane', 'gap', 'backgap'):
            out.setdefault(f'g{i}_{key}', -1.)
    out['group_size'] = float(len(group(rows)))
    out['ramp_present'] = float(out['l2_count'] > 0)
    return out


def budget():
    return dict(screen=80*1140, repeat=12*900, options=60*4*sum(t+600 for t in TIMES),
                solo=8*4*sum(t+600 for t in TIMES), live=LIVE_CAP)
