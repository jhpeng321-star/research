"""Frozen world schedule, budgets, path and allocation guards."""
import hashlib
from pathlib import Path
import re

CONTROLLERS = ('native', 'ctrl0', 'mappo', 'pi', 'circles')
CELLS = tuple((p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700))
HIGH = tuple(c for c in CELLS if c[0] in (.2, .4) and c[1] in (2400, 2700))
CAP = 500_000
STEP0_CAP = 60_000
BUDGET = dict(step0_canary=480, step0_prefix=3600, step0_branches=54000,
              step0_repeat=1080, wm=115200, bb=115200, oracle=79200,
              random=21600, **{c:21600 for c in CONTROLLERS})
FORBIDDEN = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|'
                       r'hetdata_truth|vehicle_params|driver_params|vtype|assignment', re.I)


def checked(path):
    p = Path(path)
    if FORBIDDEN.search(str(p)) or FORBIDDEN.search(str(p.resolve())):
        raise ValueError('restricted path')
    return p


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def worlds(stage):
    cells = CELLS if stage == 'step0' else CELLS + HIGH * 3
    start = 969100 if stage == 'step0' else 969112
    if stage not in ('step0', 'main'):
        raise ValueError(stage)
    return [dict(seed=start+i, penetration=p, demand=d) for i, (p, d) in enumerate(cells)]


def guard(world):
    seed = world['seed']
    if not 969100 <= seed <= 969135 or world not in worlds('step0') + worlds('main'):
        raise ValueError('unregistered world')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(seed, world['demand'], 'exposed')
    check_case(seed, world['demand'])


if __name__ == '__main__':
    import json
    print(json.dumps(dict(design_sha256=sha('docs/paper_results/C1SWITCH_DESIGN.md'),
                          per_arm_sim_s=BUDGET, planned_sim_s=sum(BUDGET.values()),
                          worlds=worlds('step0')+worlds('main')), indent=2))
