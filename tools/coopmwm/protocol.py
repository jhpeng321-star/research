"""Owner-bound exact protocol; unused COOPNATIVE worlds reassigned prospectively."""
import re
import sys
from tools.coopnative.protocol import (worlds as prior_worlds,
    checked, canonical, digest, sha, FORBIDDEN, REGISTRY)
from tools.coopbandit.protocol import CELLS

CAP=6_000_000
CAPS=dict(train=4_147_200,checks=532_800,dev=230_400,mechanics=2400,
          calibration=240_000,discovery=240_000,final_test=172_800,replication_test=172_800)
UPDATES=24
TRAINING_SEEDS=(20261101,20261102,20261103)
TRAIN_STARTS=(986000,987000,988000)
FINAL_START=989600
CHECK_EVERY=4
HORIZONS=(60,120,300,600)


def worlds(role='train',training_seed=0):
    if role in ('check','dev'):return prior_worlds(role)
    if role=='train':start,repeats=TRAIN_STARTS[training_seed],6
    elif role=='final_test':start,repeats=FINAL_START,3
    else:raise ValueError('unknown world role')
    return [dict(seed=start+i,cell=i//repeats,repeat=i%repeats,
        penetration=CELLS[i//repeats][0],demand=CELLS[i//repeats][1]) for i in range(12*repeats)]


def batch_worlds(update,training_seed=0):
    offset=((update-1)%3)*2
    return [w for w in worlds('train',training_seed) if w['repeat'] in (offset,offset+1)]


def guard(world,role):
    allowed=([w for i in range(3) for w in worlds('train',i)] if role=='train' else worlds(role))
    if world not in allowed:raise ValueError('exact registered world and role required')
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    require_development_case(world['seed'],world['demand'],'exposed')
    check_case(world['seed'],world['demand'])


def install_seed_scope():
    from traffic_shepherd import seeds as s
    from traffic_shepherd.c4 import seeds as c
    ws=[w for i in range(3) for w in worlds('train',i)]+[w for r in ('check','dev','final_test') for w in worlds(r)]
    allowed={(w['seed'],w['demand']) for w in ws}
    old_s,old_c=s.require_development_case,c.check_case
    def base(seed,loading,role):
        if type(seed) is int and type(loading) is int and role=='exposed' and (seed,loading) in allowed:return
        return old_s(seed,loading,role)
    def case(seed,demand_vph):
        if type(seed) is int and type(demand_vph) is int and (seed,demand_vph) in allowed:return base(seed,demand_vph,'exposed')
        return old_c(seed,demand_vph)
    s.require_development_case,c.check_case=base,case
    for name,module in list(sys.modules.items()):
        if name.startswith('traffic_shepherd.') and module is not None:
            if getattr(module,'check_case',None) is old_c:module.check_case=case
            if getattr(module,'require_development_case',None) is old_s:module.require_development_case=base


def authority(text):
    lines=text.splitlines()
    a=next((x for x in lines if '16:22 MAIN:' in x and 'S-COOPMWM' in x and '6,000,000' in x),None)
    b=next((x for x in lines if '16:25 MAIN:' in x and 'SPEED MODE' in x and 'FULL-v4' in x),None)
    c=next((x for x in lines if '16:38 MAIN:' in x and 'S-COOPMWM' in x and '600' in x),None)
    d=next((x for x in lines if '16:46 MAIN:' in x and 'FINAL TEST PANEL' in x and '172,800' in x),None)
    e=next((x for x in lines if '17:07 MAIN:' in x and 'S-COOPMWM' in x and '172,800' in x),None)
    if not a or not b or not c or not d or not e:
        raise ValueError('owner MAPPO re-scope and speed amendment required')
    return [a,b,c,d,e]


def registry_conflicts(text,seeds=None):
    if seeds is None:
        seeds={w['seed'] for i in range(3) for w in worlds('train',i)}|{w['seed'] for r in ('check','final_test') for w in worlds(r)}
    hits=[]
    for n,line in enumerate(text.splitlines(),1):
        if 'S-COOPMWM' in line or 'S-COOPNATIVE DESIGN READY' in line:continue
        for a,b in re.findall(r'(?<![\da-fA-F.])(9\d{5})\s*(?:[-–]|\.\.)\s*(\d{3,6})(?![\da-fA-F])',line):
            if any(int(a)<=s<=int(a[:6-len(b)]+b) for s in seeds):hits.append(n)
        if any(int(v) in seeds for v in re.findall(r'(?<![\da-fA-F.])(?:s|w)?(9\d{5})(?![\da-fA-F])',line)):hits.append(n)
    return sorted(set(hits))


def budget():
    return dict(total=CAP,phases=CAPS,unallocated=CAP-sum(CAPS.values()),
                training_seeds=list(TRAINING_SEEDS),updates_per_arm_per_seed=UPDATES,
                episodes_per_update=24,sim_s_per_episode=1200,
                train_controlled_sim_s=3_110_400,new_dev_mappo_sim_s=0,
                final_test_precharged_sim_s=172800,final_checkpoint_rule='seed0 update24; no selection',
                final_panel_sha256=digest(worlds('final_test')))
