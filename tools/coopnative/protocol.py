"""Registered admission, exact world roles, and conservative total-physics budget."""
import re
import sys
from tools.coopbandit.protocol import canonical, checked, digest, sha, FORBIDDEN, CELLS

CAP = 4_664_400
CAPS = dict(train=4_147_200, checks=288_000, dev=115_200, discovery=96_000, mechanics=1200)
WINDOW = 10
UPDATES = 144
CHECK_EVERY = 8
REGISTRY = 'E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md'


def worlds(role='train'):
    if role not in ('train', 'check', 'dev'):
        raise ValueError('unknown world role')
    if role == 'dev':
        return [dict(seed=984000+i, cell=11-i//2, repeat=i%2,
                     penetration=CELLS[11-i//2][0], demand=CELLS[11-i//2][1]) for i in range(24)]
    start, repeats = (986000, 6) if role == 'train' else (986072, 1)
    return [dict(seed=start+i, cell=i//repeats, repeat=i%repeats,
                 penetration=CELLS[i//repeats][0], demand=CELLS[i//repeats][1])
            for i in range(12*repeats)]


def batch_worlds(update):
    offset = ((update-1) % 3)*2
    return [w for w in worlds() if w['repeat'] in (offset, offset+1)]


def guard(world, role):
    if world not in worlds(role):
        raise ValueError('exact registered world and role required')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'], world['demand'], 'exposed')
    check_case(world['seed'], world['demand'])


def install_seed_scope():
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(w['seed'], w['demand']) for role in ('train', 'check', 'dev') for w in worlds(role)}
    original_s, original_c = s.require_development_case, c.check_case
    def base(seed, loading, role):
        if type(seed) is int and type(loading) is int and role == 'exposed' and (seed, loading) in allowed:
            return
        return original_s(seed, loading, role)
    def case(seed, demand_vph):
        if type(seed) is int and type(demand_vph) is int and (seed, demand_vph) in allowed:
            return base(seed, demand_vph, 'exposed')
        return original_c(seed, demand_vph)
    s.require_development_case, c.check_case = base, case
    for name, module in list(sys.modules.items()):
        if name.startswith('traffic_shepherd.') and module is not None:
            if getattr(module, 'check_case', None) is original_c:
                module.check_case = case
            if getattr(module, 'require_development_case', None) is original_s:
                module.require_development_case = base


def approval_line(text):
    for line in text.splitlines():
        if re.search(r'\bMAIN\b.*\bS-COOPNATIVE BUDGET APPROVED\b', line) and not re.search(
                r'await|pending|NO SUMO|until|not approved|must append|[\'"]S-COOPNATIVE', line, re.I):
            return line
    raise ValueError('explicit MAIN approval absent; no physics permitted')


def registry_conflicts(text):
    """Read only the registry; ranges and isolated IDs, ignoring hexadecimal hashes."""
    seeds = {w['seed'] for role in ('train', 'check') for w in worlds(role)}
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if 'S-COOPNATIVE DESIGN READY' in line:
            continue
        for a, b in re.findall(r'(?<![\da-fA-F.])(9\d{5})\s*(?:[-–]|\.\.)\s*(\d{3,6})(?![\da-fA-F])', line):
            if any(int(a) <= s <= int(a[:6-len(b)]+b) for s in seeds):
                hits.append(number)
        for value in re.findall(r'(?<![\da-fA-F.])(?:s|w)?(9\d{5})(?![\da-fA-F])', line):
            if int(value) in seeds:
                hits.append(number)
    return sorted(set(hits))


def team_delay(interval):
    # Identical reducer and units to LEVERPROBE / DISTILL comparator scoring.
    return interval['speed_deficit_vehicle_seconds'] + interval['source_wait_vehicle_seconds']


def budget():
    return dict(total=CAP, phases=CAPS, unallocated=CAP-sum(CAPS.values()),
                primary_training_ceiling=4_664_400, comparator_total=6_996_600)
