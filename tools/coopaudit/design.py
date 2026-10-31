"""Exact fresh TRAIN cases, prospective analysis and conservative budget admission."""
import hashlib
import json
import re
from pathlib import Path

CAP = 500_000
SEEDS = set(range(969650, 969678)) | {969699}
CONTROLLERS = ('mappo', 'pi', 'circles', 'ctrl0', 'wall')
MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
SOURCE_SHA = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
RESTRICTED = re.compile('sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype|driver_params|vehicle_params|requests\\.rou|assignment', re.I)


def checked(path):
    p = Path(path)
    if RESTRICTED.search(str(p)) or RESTRICTED.search(str(p.resolve())):
        raise ValueError('restricted path')
    return p


def sha(path):
    return hashlib.sha256(checked(path).read_bytes()).hexdigest()


def worlds():
    cells = ((.2, 2400), (.2, 2700), (.4, 2400), (.4, 2700))
    result = [dict(seed=969650 + 4*r+c, penetration=p, demand=d, cell=c)
              for r in range(6) for c, (p, d) in enumerate(cells)]
    result += [dict(seed=969674+c, penetration=p, demand=d, cell=4+c)
               for c, (p, d) in enumerate(((.05, 2400), (.1, 2700), (.2, 1900), (.4, 1900)))]
    return result


def canary():
    return dict(seed=969699, penetration=.4, demand=2700, cell=3)


def guard(seed, demand):
    cases = {(w['seed'], w['demand']) for w in worlds()+[canary()]}
    if type(seed) is not int or type(demand) is not int or (seed, demand) not in cases:
        raise ValueError('not an exact registered audit case')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(seed, demand, 'exposed')
    check_case(seed, demand)


def select_group(rows, size='all', lo=-150., hi=75.44):
    """Only identity, CAV eligibility and RSU-observable geometry; never outcomes/types."""
    eligible = [r for r in rows if r['is_cav'] and not r['junction']
                and r['lane'] in (0, 1, 2) and lo <= r['s'] <= hi]
    eligible.sort(key=lambda r: (abs(r['s']), r['vid']))
    chosen = eligible if size == 'all' else eligible[:int(size)]
    return [r['vid'] for r in chosen] if len(chosen) >= 2 else []


def interaction(baseline, group, solos):
    if len(solos) < 2:
        raise ValueError('interaction requires at least two exact solo branches')
    gains = [baseline - x for x in solos]
    dg = baseline - group
    return dict(D_GROUP=dg, sum_D_ONLY=sum(gains), C_G=dg-sum(gains), D_ONLY=gains)


def branch_charge(n, horizon=600):
    return (2 + len(CONTROLLERS)*(n+1))*horizon


def document():
    return dict(id='S-COOPMEASURE', cap_sim_s=CAP, worlds=worlds(), canary=canary(),
                controllers=CONTROLLERS, t0=300, clocks=[300, 600],
                group_size='WAIT_SUMOCF', window_s='WAIT_SUMOCF',
                group_zone_m=[-150., 75.44], bootstrap_draws=5000, bootstrap_seed=20261031,
                baseline='native IDM, no TraCI acceleration/lane command; native release after window',
                mapppo_sha256=MAPPO_SHA, runtime_manifest_sha256=SOURCE_SHA)


if __name__ == '__main__':
    print(json.dumps(document(), indent=1))
