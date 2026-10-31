"""Exact prospective panel and conservative full-rerun accounting."""
from tools.coopbandit.protocol import CELLS, checked, canonical, digest, sha
from .planner import TIMES, HORIZON

CAP = 800_000
REGISTRATION = 'E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md'


def worlds():
    return [dict(seed=983100+c,cell=c,repeat=0,penetration=p,demand=d)
            for c,(p,d) in enumerate(CELLS)]


def extension_worlds():
    return [dict(seed=983200+12*r+c,cell=c,repeat=r+1,penetration=p,demand=d)
            for r in range(2) for c,(p,d) in enumerate(CELLS)]


def extension_budget():
    # Disjoint host ceilings, not a forecast or a refund of the original preregistration.
    return dict(cpuQ=193000,gpuA=193000,gpuR=193020)


def h300_worlds():
    # Fresh never-fit block. Higher-share cells first, two worlds per selected cell.
    cells = [(p,d) for p in (.4,.2,.1) for d in (2700,2400,1900)]
    confirmation = [dict(seed=983300+i,cell=CELLS.index((p,d)),repeat=r,
                         penetration=p,demand=d,cohort='confirmation18')
                    for i,(p,d,r) in enumerate((p,d,r) for p,d in cells for r in range(2))]
    high = [(.4,2700),(.4,2400),(.2,2700),(.2,2400),(.4,2700),(.4,2400)]
    wall = [dict(seed=983318+i,cell=CELLS.index((p,d)),repeat=int(i>=4),
                 penetration=p,demand=d,cohort='wall6') for i,(p,d) in enumerate(high)]
    return confirmation+wall


def h300_budget():
    return dict(cpuQ=78240,gpuA=78240,gpuR=78240)


def budget():
    return dict(episodes=12*2*1200, lookahead=12*6*sum(t+HORIZON for t in TIMES),
                member_alone=12*2*(max(TIMES)+HORIZON), mechanics=1200+2*(300+HORIZON))


def approval(text):
    lines = [s for s in text.splitlines() if 'S-COOPPLAN BUDGET APPROVED' in s]
    if not lines or not any('800,000' in s and not any(x in s.lower() for x in
                           ('awaiting','not approved','pending')) for s in lines):
        raise PermissionError('MAIN must append S-COOPPLAN BUDGET APPROVED with the 800,000 cap')
    return lines[-1]


def guard(world):
    if world not in worlds()+extension_worlds()+h300_worlds():
        raise ValueError('not an exact preregistered fresh development world/cell')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'],world['demand'],'exposed')
    check_case(world['seed'],world['demand'])


def install_seed_scope():
    # Same narrow two-guard admission used by COOPBANDIT; no runtime-file edits.
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    allowed = {(w['seed'],w['demand']) for w in worlds()+extension_worlds()+h300_worlds()}
    original_s, original_c = s.require_development_case, c.check_case
    def base(seed,loading,role):
        if type(seed) is int and type(loading) is int and role == 'exposed' and (seed,loading) in allowed:
            return
        return original_s(seed,loading,role)
    def c4(seed,demand_vph):
        if type(seed) is int and type(demand_vph) is int and (seed,demand_vph) in allowed:
            return base(seed,demand_vph,'exposed')
        return original_c(seed,demand_vph)
    s.require_development_case, c.check_case = base,c4
