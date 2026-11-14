"""Paired world bootstrap, descriptive learning diagnostics, existing detectors."""
import numpy as np
from .protocol import worlds


def paired(records, comparator, role='dev'):
    expected = {w['seed'] for w in worlds(role)}
    arms = {}
    for r in records:
        key = (r['world']['seed'],r['arm'])
        if r['world'] not in worlds(role):
            raise ValueError('world/cell mismatch')
        if key in arms:
            raise ValueError('duplicate world-arm')
        arms[key] = r
    if {s for s,a in arms if a == 'learned'} != expected or {s for s,a in arms if a == comparator} != expected:
        raise ValueError('complete registered paired panel required')
    seeds = sorted(expected)
    indices = np.random.default_rng(20261101).integers(0,len(seeds),(10000,len(seeds)))
    result = {}
    for horizon in ('600','900'):
        for metric in ('team_delay_vs','collisions','teleports','ttc_below_3_vs','hard_brake'):
            delta = np.array([arms[s,'learned']['metrics'][horizon][metric]-arms[s,comparator]['metrics'][horizon][metric] for s in seeds])
            result[horizon+'_'+metric] = dict(mean=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
    return dict(worlds=len(seeds),learned_minus=comparator,metrics=result,
                bootstrap='10000 whole-world paired resamples; RNG20261101; negative delay is better')


def behaviour(record, trace_path):
    from tools.coopfree.analysis import tape
    from tools.coopbehav.detectors import detect
    samples = [s['sample'] for s in record['steps']]
    speed = [a[0] for s in samples for a in s['action']]
    gap = [a[1] for s in samples for a in s['action']]
    # Continuous summaries, with original sign/geometry detectors; no learned-move cutoffs.
    observed=tape(trace_path,record['world']['seed'])
    velocities=[v.v for row in observed.rows for v in row.values() if v.cav]
    headways=[]
    for row in observed.rows:
        for vid,v in row.items():
            if not v.cav or v.v<=0:
                continue
            front=[w for j,w in row.items() if j!=vid and w.lane==v.lane and w.x>v.x]
            if front:
                leader=min(front,key=lambda w:w.x)
                headways.append((leader.x-leader.length-v.x)/v.v)
    return dict(observed_speed_variance=float(np.var(velocities)) if velocities else None,
                observed_timegap_variance=float(np.var(headways)) if headways else None,
                speed_residual_variance=float(np.var(speed)) if speed else 0.,
                loggap_residual_variance=float(np.var(gap)) if gap else 0.,
                group_slowdown_fraction=float(np.mean([np.mean([a[0] for a in s['action']]) < 0 for s in samples if s['action']])) if speed else 0.,
                detectors=detect(observed),
                note='observational candidates; exact C_G and native-release support required for attribution')


def exact_credit(baseline,joint,solos):
    if not solos:
        raise ValueError('every group member needs its exact solo')
    dg = baseline-joint
    di = [baseline-v for v in solos]
    return dict(D_G=dg,D_i=di,C_G=dg-sum(di))
