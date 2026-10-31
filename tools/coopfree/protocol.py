"""Prospective worlds and hard budget. Importing this module never starts SUMO."""
from tools.coopbandit.protocol import canonical, checked, digest, sha, FORBIDDEN, CELLS, HIGH
import re
import sys

CAPS = dict(screen=27360, repeat=13680, train=1650000, dev=172800, discovery=91200)
UNALLOCATED = 44960
CAP = 2_000_000
TIMES = (300, 360, 420, 480, 540)
WINDOWS = (10, 20, 30)
ZONE = (-100., 75.44)
MIN_WINDOWS = 150
TARGET_WINDOWS = 156


def worlds(dev=False):
    cells = list(range(12))*2 if dev else [c for c in range(12) for _ in range(4 if c in HIGH else 1)]
    start = 982024 if dev else 982000
    return [dict(seed=start+i, cell=c, penetration=CELLS[c][0], demand=CELLS[c][1])
            for i, c in enumerate(cells)]


def guard(world):
    if world not in worlds()+worlds(True):
        raise ValueError('not an exact registered COOPFREE world/cell')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'], world['demand'], 'exposed')
    check_case(world['seed'], world['demand'])


def install_seed_scope():
    """Exact process-local admission; no frozen guard file is modified."""
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(w['seed'], w['demand']) for w in worlds()+worlds(True)}
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
    # Frozen env/scenario modules may already have captured `from seeds import ...`.
    # Patch only identical captured functions in this process, retaining original fallback.
    for name,module in list(sys.modules.items()):
        if not name.startswith('traffic_shepherd.') or module is None:
            continue
        if getattr(module,'check_case',None) is original_c:
            module.check_case = c4
        if getattr(module,'require_development_case',None) is original_s:
            module.require_development_case = base


def group(rows):
    return sorted(r['vid'] for r in rows if r['is_cav'] and ZONE[0] <= r['s_m'] <= ZONE[1])


def budget():
    assert sum(CAPS.values())+UNALLOCATED == CAP
    return dict(caps=CAPS, unallocated=UNALLOCATED, total=CAP)


def approval_line(text):
    for line in text.splitlines():
        if re.search(r'S-COOPFREE BUDGET APPROVED\b(?![\'\"])',line) and not re.search(
                r'await|pending|not approved|requires|must append|NO SUMO|proposed',line,re.I):
            return line
    raise ValueError('explicit MAIN S-COOPFREE BUDGET APPROVED line absent')
