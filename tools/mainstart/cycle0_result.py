"""Reproducible cycle-0 paired analysis; existing observables only, never starts SUMO."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tools/mainstart/mainfix'
METRICS = ('total_delay', 'onroad_delay', 'source_wait', 'throughput', 'hard_brakes',
           'ttc_exposure', 'collisions', 'teleports', 'population', 'horizon')
LABELS = ('Total delay incl. entrance waiting (veh-s)', 'On-road delay (veh-s)',
          'Entrance waiting (veh-s)', 'Throughput (arrivals)', 'Hard-brake exposure (veh-s)',
          'TTC<3s exposure (veh-s)', 'Collision involvements', 'Starting teleports')
DRAWS, SEED = 5000, 20261026


def bootstrap(worlds):
    """Resample paired whole worlds; retain both decision starts and every metric together."""
    a = np.asarray([w['metrics'] for w in worlds], dtype=float)  # world, arm(B,P), metric
    n = len(a)
    sums = a.sum(axis=0)
    rng = np.random.default_rng(SEED)
    boot = a[rng.integers(n, size=(DRAWS,n))].sum(axis=1)
    delta = sums[1] - sums[0]
    bd = boot[:,1] - boot[:,0]
    interval = np.quantile(bd,[.025,.975],axis=0).T if n>1 else [None]*len(METRICS)
    rows=[]
    for i,name in enumerate(METRICS[:8]):
        ci = list(map(float,interval[i])) if interval[i] is not None else None
        rows.append({'metric':name,'native':float(sums[0,i]),'policy':float(sums[1,i]),
                     'cost_difference':float(delta[i]),'cost_difference_ci95':ci,
                     'gain':float(-delta[i]),'gain_ci95':[-ci[1],-ci[0]] if ci else None})
    assert sums[0,8] == sums[1,8]
    normalization={}
    for name,col,den,scale in (('delay_per_requested',0,8,1),('road_per_requested',1,8,1),
                               ('source_per_requested',2,8,1),('arrivals_per_hour',3,9,3600)):
        d=float(delta[col]/sums[0,den]*scale)
        b=bd[:,col]/boot[:,0,den]*scale
        ci=list(map(float,np.quantile(b,[.025,.975]))) if n>1 else None
        normalization[name]={'native':float(sums[0,col]/sums[0,den]*scale),
                             'policy':float(sums[1,col]/sums[1,den]*scale),
                             'difference':d,'gain':-d,'ci95':ci,'gain_ci95':[-ci[1],-ci[0]] if ci else None}
    return {'worlds':n,'starts':sum(w['starts'] for w in worlds),'metrics':rows,
            'normalization':normalization,'population_sum':float(sums[0,8]),'horizon_sum':float(sums[0,9]),
            'equal_world_mean_delay_difference':float((a[:,1,0]-a[:,0,0]).mean()),
            'equal_world_mean_per_requested_difference':float(np.mean((a[:,1,0]-a[:,0,0])/a[:,0,8]))}


def num(x):
    return f'{x:+,.3f}'


def ci_text(ci):
    return '['+', '.join(num(x) for x in ci)+']' if ci is not None else 'not estimable (one world)'


def credit_terms(point):
    j=point.get('sumo')
    if not j or not point.get('group'):
        return None
    assert not point['collisions'] and not point['teleports']
    group=point['group']
    assert all(f'ONLY:{v}' in j for v in group) and 'GROUP' in j and 'B' in j
    di={v:j['B']-j[f'ONLY:{v}'] for v in group}
    dg=j['B']-j['GROUP']
    cg=dg-sum(di.values())
    return {'seed':point['seed'],'t_s':point['t_s'],'horizon_s':point['horizon_s'],
            'group':group,'D_i_gain':di,'D_G_gain':dg,'C_G_gain':cg,
            'D_i_cost_difference':{v:-d for v,d in di.items()},'D_G_cost_difference':-dg,
            'C_G_cost_interaction':-cg,'arms':j,
            'flips':{v:{'group_minus_i_gain':j['B']-j[f'FLIP:{v}'],
                        'marginal_gain':j[f'FLIP:{v}']-j['GROUP']} for v in point.get('flips',[])}}


def main():
    inputs=json.loads((OUT/'CYCLE0_INPUTS.json').read_text())
    safety=json.loads((OUT/'CYCLE0_SAFETY.json').read_text())
    assert inputs['returns_sha256']==safety['returns_sha256']==inputs['conversion']['returns_sha256']
    brakes={(r['pair_key'],r['kind']):r['braking_vehicle_seconds'] for r in safety['rows']}
    grouped=defaultdict(dict)
    for row in inputs['rows']:
        e,s=row['entry'],row['summary']
        assert e['split_role']=='train' and not 964000<=int(e['seed'])<=964199
        assert not s['dropped_vehicle_steps']
        key=e['pair_key']
        kind=e['branch']['kind']
        assert kind not in grouped[key]
        ledger=s['ledger']
        assert np.isclose(ledger['J_vehicle_seconds'],ledger['speed_deficit_vehicle_seconds']+ledger['source_wait_vehicle_seconds'])
        vec=[ledger['J_vehicle_seconds'],ledger['speed_deficit_vehicle_seconds'],ledger['source_wait_vehicle_seconds'],
             ledger['arrivals'],brakes[key,kind],ledger['ttc_below_3_vehicle_seconds'],s['collisions'],s['teleports'],
             s['requested_population'],s['horizon_s']]
        grouped[key][kind]={'metrics':vec,'entry':e,'summary':s}
    assert len(grouped)==26
    worlds={}
    points=[]
    for key,arms in sorted(grouped.items()):
        assert set(arms)=={'B','POLICY'}
        b,p=arms['B'],arms['POLICY']
        e=b['entry']
        assert b['summary']['requested_population']==p['summary']['requested_population']
        assert b['summary']['start_entry_backlog']==p['summary']['start_entry_backlog']
        w=worlds.setdefault(int(e['seed']),{'seed':int(e['seed']),'demand':e['demand_vph'],
                   'penetration':e['penetration'],'starts':0,'metrics':np.zeros((2,len(METRICS)))})
        w['starts']+=1
        w['metrics']+=np.array([b['metrics'],p['metrics']])
        points.append({'pair_key':key,'seed':int(e['seed']),'warmup_s':b['summary']['warmup_s'],
                       'native':dict(zip(METRICS,b['metrics'])),'policy':dict(zip(METRICS,p['metrics']))})
    worlds=list(worlds.values())
    assert len(worlds)==13 and all(w['starts']==2 for w in worlds)
    summary=bootstrap(worlds)
    # Independent direct reductions and collector receipts cross-check the paired grouping.
    for col,key in ((0,'J_vehicle_seconds'),(1,'speed_deficit_vehicle_seconds'),
                    (2,'source_wait_vehicle_seconds'),(3,'arrivals'),(5,'ttc_below_3_vehicle_seconds')):
        totals={arm:math.fsum(row['summary']['ledger'][key] for row in inputs['rows']
                              if row['entry']['branch']['kind']==arm) for arm in ('B','POLICY')}
        assert np.isclose(totals['POLICY']-totals['B'],summary['metrics'][col]['cost_difference'],rtol=1e-12,atol=1e-8)
    for world in inputs['collection']['worlds']:
        w=next(w for w in worlds if w['seed']==int(world['case']['seed']))
        measured=(w['metrics'][1,0]-w['metrics'][0,0])/w['starts']
        assert np.isclose(measured,world['collect']['sumo_mean_policy_minus_native_J'],rtol=1e-12,atol=1e-8)
    # Independent numerical cross-check against exported current-policy critic targets.
    observed={r['pair_key']:r['policy']['total_delay'] for r in points}
    assert np.allclose([-x for x in safety['returns']['returns']],
                       [observed[e['pair_key']] for e in inputs['returns_receipt']['entries']],rtol=0,atol=0)
    byworld=[{**{k:v for k,v in w.items() if k!='metrics'},**bootstrap([w])} for w in worlds]
    strata={key:{str(v):bootstrap([w for w in worlds if w[key]==v]) for v in sorted({w[key] for w in worlds})}
            for key in ('penetration','demand')}
    credits=[credit_terms(p) for c in inputs['credits'] for p in c['rows']]
    credit_values=[p for p in credits if p is not None]
    credit_summary={}
    missing_points=[{'seed':p['seed'],'t_s':p['t_s']} for c in inputs['credits'] for p in c['rows']
                    if not p.get('sumo')]
    if credit_values:
        cw=defaultdict(list)
        for p in credit_values:
            cw[p['seed']].append(p)
        for key in ('D_i_gain','D_G_gain','C_G_gain'):
            values=[]
            for seed,ps in sorted(cw.items()):
                values.append(np.mean([np.mean(list(p[key].values())) if key=='D_i_gain' else p[key] for p in ps]))
            a=np.array(values)
            rng=np.random.default_rng(SEED)
            b=a[rng.integers(len(a),size=(DRAWS,len(a)))].mean(1)
            credit_summary[key]={'worlds':len(a),'mean':float(a.mean()),
                                  'ci95':list(map(float,np.quantile(b,[.025,.975]))) if len(a)>1 else None}
    result={'method':{'draws':DRAWS,'rng_seed':SEED,'cluster':'paired traffic world; both starts retained',
                      'cost_difference':'treatment-reference','gain':'native-policy'},
            'pooled':summary,'worlds':byworld,'strata':strata,'points':points,
            'credit_points':credit_values,'credit_points_total':len(credits),'credit_points_missing':credits.count(None),
            'credit_summary':credit_summary,
            'missing_credit_points':missing_points,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (OUT/'CYCLE0_INPUTS.json',OUT/'CYCLE0_SAFETY.json')}}
    (OUT/'CYCLE0_RESULT.json').write_text(json.dumps(result,indent=1)+'\n')
    lines=['# Cycle 0 real-SUMO result — round-16 coop-ON policy',
           '', '## Headline', '']
    total=summary['metrics'][0]
    per=summary['normalization']['delay_per_requested']
    wins=sum(w['metrics'][0]['cost_difference']<0 for w in byworld)
    pointwins=sum(p['policy']['total_delay']<p['native']['total_delay'] for p in points)
    lines += [f"**Policy − native total delay: {num(total['cost_difference'])} veh-s pooled across 26 × 300 s paired continuations / 13 worlds; 95% world-bootstrap CI {ci_text(total['cost_difference_ci95'])}.**",
              f"Gain (native − policy): **{num(total['gain'])} veh-s**, CI {ci_text(total['gain_ci95'])}. Policy improves **{wins}/13 worlds; {pointwins}/26 starts**.",
              f"Per requested-vehicle exposure: cost difference **{num(per['difference'])} s/request**, CI {ci_text(per['ci95'])}; gain **{num(per['gain'])} s/request**, CI {ci_text(per['gain_ci95'])}.",
              '', '## Cohort, signs and estimand', '',
              '- Existing cycle-0 collection only: cpuP seven worlds, cpuQ six; seeds968129–968141; CAV shares5%,10%,20%,40%; demands1900,2100,2400,2700 veh/h. Two saved native-state decision times360s/660s per world; each branch lasts300s. These are training-world results, not final evaluation.',
              '- The two policy branches each restart from their saved native state. Pooling them does not establish performance of one continuously controlled600s episode. Prehistory/native base costs are excluded from branch metrics.',
              '- Cost difference = treatment − reference, here POLICY − native B: negative improves delay, braking, TTC and incidents. Gain = native − policy: positive improves those costs. For throughput, more arrivals are better: a positive POLICY−B difference is an improvement; native−policy is only the signed complement, not a throughput benefit.',
              '- Total delay J = authoritative on-road speed-deficit veh-s + observed entrance-waiting veh-s, including post-start entrants and queued requests. This is finite-horizon delay, not completed-trip travel time. On-road uses the collector’s fixed speed limit; no behavioral threshold is fitted.',
              f"- Normalized delay divides pooled J by {summary['population_sum']:,.0f} requested-vehicle exposures. `requested_population` counts requests due by each branch endpoint, including requests before branch start, and each world occurs at two endpoints; it is not a count of unique newly requested vehicles. Raw veh-s is the primary estimand.",
              '- Hard braking uses the existing collector metric a<−3m/s² at0.5s, reconstructed from observed acceleration/presence arrays after each branch start through its endpoint. It measures vehicle-seconds of exposure, not unique braking episodes. Road-delay reconstruction matches every authoritative summary at rtol1e−6/atol1e−3, verifying population/time coverage; no drops in52/52 branches.',
              '- TTC<3s vehicle-seconds, arrivals, collision involvements and starting teleports come directly from each branch’s authoritative ledger. No TTC reconstruction from lane geometry or driver/type files.',
              '', '## Pooled metrics and paired world-cluster uncertainty', '',
              '5,000 percentile bootstrap draws, NumPy default_rng seed20261026: sample13 whole worlds with replacement; retain both starts, both arms, numerators and denominators together. The pooled-total CI is for a13-world/26-start cohort total. Normalized ratios are recomputed in every draw. No vehicle/time-step pseudo-replication.', '',
              '| Metric | Native B | POLICY | Cost difference POLICY−B [95% CI] | Gain B−POLICY [95% CI] |',
              '|---|---:|---:|---|---|']
    for label,row in zip(LABELS,summary['metrics']):
        lines.append(f"| {label} | {row['native']:,.3f} | {row['policy']:,.3f} | {num(row['cost_difference'])} {ci_text(row['cost_difference_ci95'])} | {num(row['gain'])} {ci_text(row['gain_ci95'])} |")
    rate=summary['normalization']['arrivals_per_hour']
    lines += ['',f"Mean throughput over pooled branch time: native {rate['native']:,.2f}, policy {rate['policy']:,.2f} arrivals/hour; POLICY−B {num(rate['difference'])} {ci_text(rate['ci95'])}; B−POLICY {num(rate['gain'])} {ci_text(rate['gain_ci95'])}.",
              'Zero collision/teleport bootstrap bounds reflect zero observed incidents in this cohort; they do not bound incident risk in a different cohort.',
              f"Equal-world sensitivity: mean total difference {num(summary['equal_world_mean_delay_difference'])} veh-s/world (two starts); mean of within-world normalized differences {num(summary['equal_world_mean_per_requested_difference'])} s/request. Each world contributes two starts, so equal-world and equal-start raw-cost weighting agree.",
              '', '## Per world', '',
              'All entries below are POLICY−B; gain B−POLICY is the exact negative. Each row pools both300s starts. Uncertainty is estimated across worlds above; a single world cannot support a world-cluster CI.', '',
              '| Seed / box | CAV share | Demand (veh/h) | Δ total delay (veh-s) | Gain total (veh-s) | Δ on-road | Δ entrance | Δ arrivals | Δ hard-brake veh-s | Δ TTC veh-s | Δ collision / teleport |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    boxes={int(c['world']['seed']):c['world']['box'] for c in inputs['credits']}
    for w in byworld:
        m=w['metrics']
        d=[v['cost_difference'] for v in m]
        lines.append(f"| {w['seed']} / {boxes[w['seed']]} | {100*w['penetration']:g}% | {w['demand']} | {num(d[0])} | {num(-d[0])} | {num(d[1])} | {num(d[2])} | {d[3]:+g} | {num(d[4])} | {num(d[5])} | {d[6]:+g} / {d[7]:+g} |")
    lines += ['', '## CAV-share and demand strata', '',
              'Strata are marginal observed subsets, not a balanced factorial grid or causal share/demand effects. The2700veh/h stratum has one world; its world-cluster CI is not estimable. Other strata have only3–4 worlds; bootstrap bounds are exploratory.', '',
              '| Stratum | Worlds / starts | Δ total delay (veh-s) [95% CI] | Gain total [95% CI] | Δ delay (s/request) [95% CI] | Δ arrivals | Δ hard-brake veh-s | Δ TTC veh-s |',
              '|---|---:|---|---|---|---:|---:|---:|']
    for key,groups in strata.items():
        for value,s in groups.items():
            name=f'CAV {100*float(value):g}%' if key=='penetration' else f'Demand {value}veh/h'
            m=s['metrics']
            x=s['normalization']['delay_per_requested']
            lines.append(f"| {name} | {s['worlds']} / {s['starts']} | {num(m[0]['cost_difference'])} {ci_text(m[0]['cost_difference_ci95'])} | {num(m[0]['gain'])} {ci_text(m[0]['gain_ci95'])} | {num(x['difference'])} {ci_text(x['ci95'])} | {m[3]['cost_difference']:+g} | {num(m[4]['cost_difference'])} | {num(m[5]['cost_difference'])} |")
    lines += ['', '## Real-SUMO cooperation credit: ONLY / GROUP / FLIP', '',
              f"Recorded credit points: **{len(credits)}**; points with actual real-SUMO credit arms: **{len(credit_values)}**; unavailable: **{credits.count(None)}**. These have their own30s horizon, not the300s policy/native comparison. FLIP count is0 by registered configuration.",
              'Definitions in gain units: D_i=J(B)−J(ONLY:i), D_G=J(B)−J(GROUP), C_G=D_G−Σ_iD_i; positive C_G means extra group benefit beyond summed solo gains. Cost differences are −D_i/−D_G, and the cost-interaction counterpart is −C_G. Do not substitute WM-imagined credit or ALL−B for uncollected ONLY/GROUP arms.', '']
    if credit_values:
        lines.append(f"Conditional sample: {len(credit_values)} points from {len({p['seed'] for p in credit_values})} worlds; 5,000 whole-world bootstrap draws, seed20261026. Ineligible points are not imputed as zero. Missing points: "+', '.join(f"{p['seed']}@{p['t_s']}" for p in missing_points)+'.')
        for key,info in credit_summary.items():
            ci=info['ci95']
            cost_ci=[-ci[1],-ci[0]] if ci else None
            lines.append(f"- {key}: equal-world mean {num(info['mean'])} veh-s, 95% CI {ci_text(ci)}; cost-sign counterpart {num(-info['mean'])} veh-s, 95% CI {ci_text(cost_ci)}. Average members within point for D_i, points within world, then worlds.")
        lines.append('Positive C_G indicates less harm together than the sum of solo harms here; mean D_i and D_G remain negative gains versus native. It does not establish a beneficial group intervention.')
        lines += ['', '| Seed @ time | Group size | D_i gain (member:veh-s) | D_G gain | C_G gain | D_G cost difference | C_G cost interaction |',
                  '|---|---:|---|---:|---:|---:|---:|']
        for p in credit_values:
            vals='; '.join(f'{v}:{num(d)}' for v,d in p['D_i_gain'].items())
            lines.append(f"| {p['seed']}@{p['t_s']} | {len(p['group'])} | {vals} | {num(p['D_G_gain'])} | {num(p['C_G_gain'])} | {num(p['D_G_cost_difference'])} | {num(p['C_G_cost_interaction'])} |")
    else:
        lines += ['No eligible real ONLY/GROUP arms were collected: recorded `group=null`, `sumo=null`; D_i, D_G and C_G are **unavailable**, not zero. The configured real-credit option alone is not evidence of measured cooperation.']
    lines += ['', '## Provenance and validation', '',
              '- Accepted/training WM: round16 FINAL SHA256 `5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2`.',
              '- Policy: cycle0 `policy/c00.a2/policy_latest.pt`, SHA256 `fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d`;150 updates, frozen update10 gate PASS. These training gates do not imply SUMO improvement.',
              f"- Current-policy returns `/root/main14_loop/returns/returns_c00.a1.npz`, SHA256 `{inputs['returns_sha256']}`;26/26 returns exactly equal negative authoritative POLICY J.52 summaries/26 complete B–POLICY pair groups,13 unique worlds, matched branch-start backlog and requested populations.",
              '- Pairs `/root/main14_loop/data/pairs_c00.a1`; collector artifacts under `/root/main14_loop/worlds/c00/w<seed>.a1/artifact`; ledger `/root/main14_loop/cycle_ledger.jsonl`. Hash-checked78 summary/credit/calibration files against per-world `HASHES.json`; all208 named numeric array files hashed. Source JSON preserves per-summary paths/hashes, all real-credit records, collection receipts and ledger snapshot.',
              f"- Pairs manifest SHA256 `{safety['manifest_sha256']}`. Local input SHA256 `{result['source_sha256']['CYCLE0_INPUTS.json']}`; safety/array-validation SHA256 `{result['source_sha256']['CYCLE0_SAFETY.json']}`.",
              '- Reproduce: `python tools/mainstart/cycle0_result.py`; machine-readable values/stratum metrics/point-level arm costs are in `tools/mainstart/mainfix/CYCLE0_RESULT.json`. No new SUMO, evaluation seeds, restricted paths, driver truth or assignment files were used.',
              '- Existing accounting:87,399 charged /28,353 actual sim-s; prior58,266 charged startup reservations retained; total cap210,000. Completed world `pair_sim_seconds` already includes base and extra branches; do not add base `simulated_seconds` again.',
              '', '## S-MAINFIX operational status', '',
              '**WAIT_OWNER.** The driver recorded its5400s WM timeout at2026-10-26 18:59:03UTC+8; driver93301/trainer94892 exited under driver handling. Owner decisions were appended verbatim to the shared registration before changes. The local staging invocation then hit Windows WinError206 before SSH creation. Counting the timeout plus this invocation failure, the second non-scientific failure stop rule was applied; no source/config/state changes, retry or overlap launch occurred. New receipt: `restart/MAINFIX_WAIT_OWNER.json`; original `restart/WAIT_OWNER.json` retained.',
              'The live timeout remains1.5h, WM speed flags remain disabled, overlap remainsOFF. Source integration exists for SPEED2 and the earlier FASTLOOP schedule, but that schedule joins GPU policy before WM-FT; concurrent WM/policy scheduling needs an owner-authorized, tested extension and exact owned-PID guard before activation. Explicit owner direction is required to repair staging and continue.']
    (ROOT/'CYCLE0_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'pooled':summary,'worlds_better':wins,'starts_better':pointwins,
                      'credit_points':len(credit_values),'credit_summary':credit_summary},indent=1))


if __name__=='__main__':
    main()
