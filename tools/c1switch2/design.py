"""Single source of truth for the prospective protocol and full-rerun costs."""
import random

from tools.c1switch.design import CELLS, HIGH, sha

PROTOCOL = 'S-C1SWITCH2/v2'
BASE_COMMIT = '1121117f'
CONTROLLERS = ('native', 'ctrl0', 'mappo', 'pi', 'circles', 'wall')
CPU_ARMS = CONTROLLERS + ('random',)
MODEL_ARMS = ('wm', 'bb', 'wm_phys')
RL_ARMS = ('fleet_ppo', 'cav_mappo')
PICKERS = ('wm', 'bb') + RL_ARMS
ARMS = PICKERS + CONTROLLERS + ('random',)
TIMES = tuple(range(300, 900, 20))
HORIZON = 60
CADENCE = 20
K = 8
CAP = 600_000
REGISTRATION = 'E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md'
DESIGN = 'docs/paper_results/C1SWITCH2_DESIGN.md'
# Hard allocation ceilings, including service loading, validation and idle time.
# Concurrent use of two devices counts as two GPU-seconds per elapsed second.
GPU_SECONDS = dict(wm=57600, bb=10800, wm_phys=57600,
                   fleet_ppo=0, cav_mappo=0, random=0, canary=0,
                   **{c: 0 for c in CONTROLLERS})


def worlds():
    return [dict(seed=969136+i, penetration=p, demand=d)
            for i, (p, d) in enumerate(CELLS+HIGH)]


def guard(world):
    if world not in worlds()+[canary_world()] or type(world['seed']) is not int:
        raise ValueError('unregistered S-C1SWITCH2 world')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'], world['demand'], 'exposed')
    check_case(world['seed'], world['demand'])


def canary_world():
    return dict(seed=969152, penetration=.4, demand=2700)


def measurement_indices(seed):
    if seed not in {w['seed'] for w in worlds()}:
        raise ValueError('unregistered measurement world')
    return sorted(random.Random(f'S-C1SWITCH2:{seed}').sample(range(30), 1))


def assigned(seed):
    index = next(i for i, w in enumerate(worlds()) if w['seed'] == seed)
    return 'gpuR' if index % 4 == 3 else 'cpuQ'


def local_budget(host, include_phys=False):
    if host not in ('cpuQ', 'gpuR'):
        raise ValueError('unregistered CPU host')
    selected = [w for w in worlds() if assigned(w['seed']) == host]
    branch = sum(TIMES[j]+HORIZON for w in selected for j in measurement_indices(w['seed']))
    arms = ARMS+(('wm_phys',) if include_phys else ())
    result = {a: 1200*len(selected)+(6*branch if a in PICKERS+('wm_phys',) else 0)
              for a in arms}
    result['cav_mappo'] += branch
    result['canary'] = 4680 if host == 'cpuQ' else 0
    return result


def budget(include_phys=False):
    branches = sum(len(CONTROLLERS)*(TIMES[j]+HORIZON)
                   for w in worlds() for j in measurement_indices(w['seed']))
    arms = ARMS + (('wm_phys',) if include_phys else ())
    result = {a: len(worlds())*1200 + (branches if a in PICKERS+('wm_phys',) else 0)
              for a in arms}
    # A per-CAV joint choice is outside the six fleet alternatives. Its extra
    # held-joint-action branch is needed for a truthful signed oracle gap.
    result['cav_mappo'] += branches//len(CONTROLLERS)
    result['canary'] = 4680  # 2x900 scripted episodes + 2x360 native + 6x360 branches.
    if sum(result.values()) > CAP:
        raise ValueError('full-rerun plan exceeds target cap')
    return result


def manifest(include_phys=False):
    return dict(protocol=PROTOCOL, inherited_commit=BASE_COMMIT, library=CONTROLLERS,
                cadence_s=CADENCE, horizon_s=HORIZON, K=K, worlds=worlds(),
                measured={str(w['seed']): measurement_indices(w['seed']) for w in worlds()},
                sim_s=budget(include_phys), total_sim_s=sum(budget(include_phys).values()),
                gpu_seconds={a: GPU_SECONDS[a] for a in budget(include_phys)},
                design_sha256=sha(DESIGN))


if __name__ == '__main__':
    import json
    print(json.dumps(manifest(include_phys=True), indent=2))
