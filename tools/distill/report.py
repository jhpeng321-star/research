"""Independent full-panel live QA and registered world-bootstrap estimates."""
import argparse
from collections import Counter
import json

import numpy as np

from tools.coopbandit.protocol import canonical, sha
from tools.coopplan.predictor import ObservableState
from tools.coopplan.planner import TIMES
from .data import read, save, LOCAL
from .live import panel, budget, WORLD_CREDIT, HORIZON, admit_event
from .portable import Student

OUT = LOCAL/'docs/paper_results/distill_live'
FIT = LOCAL/'docs/paper_results/distill_v1'


def ci(values):
    x = np.asarray(values,float)
    draws = np.random.default_rng(20261101).integers(0,len(x),(10000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist(),worlds=len(x))


def credit_ci(world_credits):
    sums = np.array([sum(c['C_G'] for c in rows) for rows in world_credits],float)
    counts = np.array([len(rows) for rows in world_credits])
    if not counts.sum():
        return None
    draws = np.random.default_rng(20261101).integers(0,len(sums),(10000,len(sums)))
    denominators = counts[draws].sum(1)
    good = denominators > 0
    boot = sums[draws].sum(1)[good]/denominators[good]
    return dict(mean=float(sums.sum()/counts.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                events=int(counts.sum()),positive=sum(c['C_G'] > 0 for rows in world_credits for c in rows),
                active_worlds=int((counts>0).sum()),zero_event_draws=int((~good).sum()),
                resampling='all24worlds including zero-action worlds; event-weighted mean')


def build(receipt, policy):
    if (not receipt['complete'] or receipt['stopped'] or receipt['incomplete'] or
            receipt['worlds'] != 24 or not receipt['mechanics']['pass_']):
        raise ValueError('complete24paired panel and mechanics required')
    records = sorted(receipt['records'],key=lambda r:r['world']['seed'])
    if [r['world'] for r in records] != panel():
        raise ValueError('exact registered panel order/cells')
    if receipt['actual'] != receipt['charged'] or receipt['charged'] > budget()['total']:
        raise ValueError('conservative full-rerun accounting')
    if len({r['id'] for r in receipt['charges']}) != len(receipt['charges']):
        raise ValueError('duplicate attempts')
    charges = {r['id']:r for r in receipt['charges']}
    closes = {r['id']:r for r in receipt['closed']}
    if set(charges) != set(closes):
        raise ValueError('unclosed or uncharged jobs')
    if any(r['sim_s'] != closes[k]['actual_sim_s'] or r['sim_s'] != closes[k]['charged_sim_s']
           for k,r in charges.items()):
        raise ValueError('per-attempt actual/charge mismatch')
    rows,credits,expected_branches = [],[],set()
    safety = {a:{k:0. for k in ('collisions','teleports','ttc_below_3_vs','hard_brake')}
              for a in ('mappo','student')}
    actions = Counter();proposals = Counter();pruned = 0;requests = overrides = 0
    zero_pass = 0
    for r in records:
        w = r['world'];seed = w['seed'];b,s = r['baseline'],r['student']
        if b['world'] != w or s['world'] != w or not s['warmup_pass']:
            raise ValueError('paired world/warmup mismatch')
        if [e['time_s'] for e in s['history']] != list(TIMES):
            raise ValueError('decision cadence')
        if s['actual_sim_s'] != 1200 or b['actual_sim_s'] != 1200:
            raise ValueError('full episodes required')
        for arm,data in (('mappo',b),('student',s)):
            for k in safety[arm]:safety[arm][k] += data['clocks']['900'][k]
        model = Student(policy);spent = 0
        event_credits = {c['time_s']:c for c in s['credits']}
        if len(event_credits) != len(s['credits']):
            raise ValueError('duplicate event credits')
        for e in s['history']:
            t = e['time_s'];key = f's{seed}_t{t}'
            recomputed = model.step(ObservableState(t,tuple(e['initial_rows']),key))
            cost = admit_event(recomputed,spent)
            if (canonical(e['proposed_action']) != canonical(recomputed['proposed_action']) or
                    canonical(e['executed_action']) != canonical(recomputed['executed_action']) or
                    e['budget_pruned'] != recomputed['budget_pruned']):
                raise ValueError('frozen observable student/action/quota mismatch')
            np.testing.assert_allclose(e['decision']['predicted_mean_D'],
                recomputed['decision']['predicted_mean_D'],atol=1e-8,rtol=1e-10)
            name = e['executed_action']['id']
            proposals[e['proposed_action']['id']] += 1;actions[name] += 1
            pruned += e['budget_pruned']
            if name == 'O0':
                if t in event_credits:
                    raise ValueError('credit for non-executed move')
                continue
            c = event_credits[t]
            if c['action'] != e['executed_action'] or c['horizon_s'] != HORIZON:
                raise ValueError('credit action/horizon')
            ids = [key+'_O0',key+'_'+name,key+'_'+name+'_solo0',key+'_'+name+'_solo1']
            branches = [receipt['branch_results'][ident] for ident in ids]
            for ident,branch in zip(ids,branches):
                if not branch['prefix_pass'] or branch['actual_sim_s'] != t+HORIZON:
                    raise ValueError('credit exact-prefix/length mismatch')
                if charges[ident]['kind'] != 'credit':
                    raise ValueError('branch charge kind')
            j = [branch['metrics']['team_delay_vs'] for branch in branches]
            dg = j[0]-j[1];ds = [j[0]-v for v in j[2:]]
            if c['D_G'] != dg or c['member_alone_D'] != ds or c['C_G'] != dg-sum(ds):
                raise ValueError('exact all-event CG arithmetic')
            reservation = receipt['reservations'][key]
            if reservation != dict(cost=cost,world=w,time_s=t,action=name):
                raise ValueError('reservation binding')
            expected_branches.update(ids)
            credits.append(dict(world_id=seed,**c));spent += cost
        if spent != s['credit_sim_s'] or spent > WORLD_CREDIT:
            raise ValueError('world credit quota')
        if not s['credits']:
            if (not s['all_O0_bitwise_pass'] or canonical(b['final_prefix']) != canonical(s['final_prefix'])
                    or b['clocks'] != s['clocks']):
                raise ValueError('all-O0 not bitwise identical')
            zero_pass += 1
        row = dict(world_id=seed,cell=w['cell'],penetration=w['penetration'],demand=w['demand'])
        for clock in ('600','900'):
            row[f'saving_J{clock}'] = b['clocks'][clock]['team_delay_vs']-s['clocks'][clock]['team_delay_vs']
            row[f'MAPPO_J{clock}'] = b['clocks'][clock]['team_delay_vs']
            row[f'student_J{clock}'] = s['clocks'][clock]['team_delay_vs']
        row.update(executed_moves=len(s['credits']),credit_sim_s=spent)
        rows.append(row);requests += s['requests'];overrides += s['overrides']
    if expected_branches != set(receipt['branch_results']):
        raise ValueError('missing/unexpected credit branches')
    expected_jobs = expected_branches | {'mechanics_all_O0'} | {
        f's{w["seed"]}_{arm}' for w in panel() for arm in ('mappo','student')}
    if set(charges) != expected_jobs:
        raise ValueError('unexpected job accounting')
    if len(receipt['reservations']) != len(credits):
        raise ValueError('outstanding event reservations')
    credit_charge = sum(r['sim_s'] for r in charges.values() if r['kind'] == 'credit')
    if credit_charge != sum(r['credit_sim_s'] for r in rows):
        raise ValueError('event-credit charge reconciliation')
    summary = dict(worlds=24,decisions=288,savings={clock:ci([r[f'saving_J{clock}'] for r in rows])
        for clock in ('600','900')},executed_actions=dict(actions),proposed_actions=dict(proposals),
        budget_pruned=pruned,credit=credit_ci([r['student']['credits'] for r in records]),
        safety=safety,requests=requests,overrides=overrides,all_O0_bitwise_worlds=zero_pass,
        exact_credit_prefixes=len(expected_branches),actual=receipt['actual'],charged=receipt['charged'],
        prospective_upper=budget()['total'],cap=200000,
        bootstrap=dict(resamples=10000,seed=20261101,unit='whole_world'),
        limitations='resource-limited frozen EXAM-only D600 student; event CG is H120, not episode efficacy')
    return summary,rows,credits


def insights():
    importance = read(FIT/'FEATURE_IMPORTANCE.json')
    triggers = read(FIT/'OBSERVABLE_TRIGGERS.json')
    for option in ('O1','O2','O3','O4'):
        top = sorted([r for r in importance if r['option'] == option and 'sensitivity' in r['method']],
                     key=lambda r:r['importance'],reverse=True)[:3]
        print(json.dumps(dict(option=option,top_features=top)))
    print(json.dumps([r for r in triggers if r['model'] == 'nested']))


def main():
    p = argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--insights',action='store_true')
    a = p.parse_args()
    if a.insights:insights();return
    receipt = read(OUT/'RECEIPTS.json')
    if receipt['binding'] != read(OUT/'STAGED.json')['binding']:
        raise ValueError('frozen staging identity')
    summary,rows,credits = build(receipt,read(OUT/'POLICY.json'))
    qa = dict(pass_=True,worlds=24,all_event_twins=True,complete_accounting=True,
              frozen_policy_actions=True,receipt_sha256=sha(OUT/'RECEIPTS.json'))
    for name,value in (('SUMMARY.json',summary),('WORLD_METRICS.json',rows),('CREDITS.json',credits),('QA.json',qa)):
        if a.verify:
            if read(OUT/name) != value:raise ValueError('report recomputation mismatch: '+name)
        else:save(OUT/name,value)
    print(json.dumps(summary))


if __name__ == '__main__':main()
