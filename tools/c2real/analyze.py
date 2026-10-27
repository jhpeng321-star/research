"""Read-only observable-tape analysis; no simulator or model imports."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1')
import numpy as np
import pandas as pd

ROOT = Path('/root/autodl-tmp/c4/restart20261026')
HOME = ROOT/'c2real'
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
INPUTS = {}
ROWS = []
NBOOT = 5000
SEED = 20261027


def safe(path):
    p = Path(path)
    assert not BAD.search(str(p)) and p.resolve().is_relative_to(ROOT.resolve()), p
    return p


def read(path):
    p = safe(path)
    b = p.read_bytes()
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return b


def doc(path):
    return json.loads(read(path))


def array(path):
    read(path)
    return np.load(path, allow_pickle=False)


def ci(draws):
    return list(map(float, np.quantile(draws, [.025, .975])))


def estimate(label, values, **meta):
    v = np.asarray(values, float)
    ix = np.random.default_rng(SEED).integers(0, len(v), (NBOOT, len(v)))
    result = dict(level='estimate', metric=label, value=float(v.mean()),
                  ci_low=ci(v[ix].mean(1))[0], ci_high=ci(v[ix].mean(1))[1],
                  n_worlds=len(v), **meta)
    ROWS.append(result)
    return result


def difference(label, first, second, **meta):
    a, b = np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(SEED)
    draws = a[rng.integers(0,len(a),(NBOOT,len(a)))].mean(1)
    draws -= b[rng.integers(0,len(b),(NBOOT,len(b)))].mean(1)
    lo, hi = ci(draws)
    result = dict(level='contrast', metric=label, value=float(a.mean()-b.mean()),
                  ci_low=lo, ci_high=hi, n_worlds=len(a)+len(b), pairing='unpaired_across_cycles', **meta)
    ROWS.append(result)
    return result


def census():
    result = {}
    for c in range(3):
        j = doc(ROOT/f'arm/loop/worlds_c{c:02}.json')
        result[c] = dict(registered_worlds=len(j['cases']), collected_worlds=len(j['worlds']),
                         missing=[w['seed'] for w in j['cases'] if str(w['seed']) not in j['worlds']],
                         checkpoint=j['policy_sha256'], worlds=j['worlds'])
    # c00's a1..a4 are conversion retries, not independent simulations or samples.
    retry = []
    for a in range(1,5):
        d = doc(ROOT/f'arm/loop/returns/returns_c00.a{a}.receipt.json')
        retry.append([(e['pair_key'], e['sha256'],e['J']) for e in d['entries']])
    assert all(x == retry[-1] for x in retry)
    (HOME/'CENSUS.json').write_text(json.dumps(dict(cycles=result,
        c00_conversion_retries_identical=True, selected_tags=['c00.a4','c01.a1','c02.a1']), indent=2))
    return result


def returns(c, tag):
    base = ROOT/'arm/loop/returns'
    receipt = doc(base/f'returns_{tag}.receipt.json')
    native = doc(base/f'native_returns_{tag}.receipt.json')
    pairs = doc(base/f'pairing_{tag}.json')
    pol, nat = array(base/f'returns_{tag}.npz'), array(base/f'native_returns_{tag}.npz')
    assert set(pol['seeds']) == set(nat['seeds'])
    assert np.all(pol['horizon_s'] == 300) and np.all(nat['horizon_s'] == 300)
    assert float(pol['gamma']) == float(nat['gamma']) == 1
    assert 'source' in str(pol['accounting']).lower() or 'entrance' in str(pol['accounting']).lower()
    assert hashlib.sha256(read(base/f'returns_{tag}.npz')).hexdigest() == pairs['policy_sha256']
    assert hashlib.sha256(read(base/f'native_returns_{tag}.npz')).hexdigest() == pairs['native_sha256']
    records = []
    for i, (p,n,binding) in enumerate(zip(receipt['entries'],native['entries'],pairs['pairs'],strict=True)):
        key = p['pair_key']
        assert n['pair_key'] == binding['pair_key'] == key and binding['same_saved_start']
        assert binding['horizon_s'] == 300
        assert not 964000 <= binding['seed'] <= 964199
        for name in pol.files:
            if name.startswith('state__'):
                assert np.array_equal(pol[name][i],nat[name][i]), (key,name)
        ps, ns = doc(p['summary']),doc(n['summary'])
        assert INPUTS[str(Path(p['summary']).relative_to(ROOT))] == p['sha256'] == binding['policy_summary_sha256']
        assert INPUTS[str(Path(n['summary']).relative_to(ROOT))] == n['sha256'] == binding['native_summary_sha256']
        assert abs(pol['returns'][i]+p['J']) < 1e-8 and abs(nat['returns'][i]+n['J']) < 1e-8
        assert ps['horizon_s'] == ns['horizon_s'] == 300
        for summary in (ps,ns):
            led = summary['ledger']
            assert abs(led['speed_deficit_vehicle_seconds']+led['source_wait_vehicle_seconds']-led['J_vehicle_seconds'])<1e-7
        m = re.fullmatch(r'w(\d+)m(\d+)p(\d+)t300@t(\d+)',key)
        seed, demand, share, start = map(int,m.groups())
        record = dict(level='start',metric='policy_minus_native_J300',cycle=c,world=seed,
            demand=demand,share=share/100,start_s=start,pair_key=key,horizon_s=300,
            policy_J=p['J'],native_J=n['J'],value=p['J']-n['J'],
            road_delta=ps['ledger']['speed_deficit_vehicle_seconds']-ns['ledger']['speed_deficit_vehicle_seconds'],
            source_delta=ps['ledger']['source_wait_vehicle_seconds']-ns['ledger']['source_wait_vehicle_seconds'],
            policy_checkpoint=receipt['policy_checkpoint'],observable_start_sha256=binding['observable_start_sha256'])
        ROWS.append(record)
        records.append(record)
    return records


def behaviour(c,tag):
    spec = importlib.util.spec_from_file_location('frozen_coopbehav',HOME/'coopbehav.py')
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    th = doc(HOME/'thresholds.json')
    assert INPUTS['c2real/thresholds.json'].startswith('9119cd77')
    read(HOME/'coopbehav.py')
    base = ROOT/'arm/loop/data'/f'pairs_{tag}'
    manifest = doc(base/'C4_EPISODES_MANIFEST.json')
    out = []
    for e in manifest['episodes']:
        arm = e['branch']['kind']
        assert arm in ('B','POLICY')
        seed = e['seed']
        assert not 964000<=seed<=964199
        p = safe(base/e['dir'])
        t,kin,present,cav,vid = [array(p/(n+'.npy')) for n in ('time_s','veh_kin','veh_mask','is_cav','vid')]
        assert t[0] == e['branch']['branch_t_s']-60 and t[-1] == e['branch']['branch_t_s']+300
        assert np.allclose(np.diff(t),.5)
        onehz = (np.abs(t-np.round(t))<1e-6)[:,None] & present.astype(bool)
        ti,ki = np.nonzero(onehz)
        lane = np.where(kin[ti,ki,1:4].sum(1)>0,kin[ti,ki,1:4].argmax(1),-1)
        frame = pd.DataFrame(dict(t=t[ti].astype(int),vid=vid[ti,ki],s=kin[ti,ki,0],
            v=kin[ti,ki,4],a=kin[ti,ki,5],lane=lane,cav=cav[ti,ki].astype(bool)))
        assert not frame.duplicated(['t','vid']).any()
        frame = cb.features(frame)
        start = e['branch']['branch_t_s']
        result = cb.detect(frame,th,(start,start+300))
        counts = {b:sum(ev['type']==b for ev in result['events']) for b in ('metering','multi_slow','vacate')}
        counts['side_by_side'] = sum(ev['type']=='multi_slow' and ev['subtype']=='side_by_side' for ev in result['events'])
        for b,count in counts.items():
            row = dict(level='behaviour_start',cycle=c,world=seed,share=e['penetration'],demand=e['demand_vph'],
                pair_key=e['pair_key'],arm='policy' if arm=='POLICY' else 'native',metric=b,
                count=count,exposure_cav_h=result['exposure_cav_s']/3600)
            ROWS.append(row)
            out.append(row)
        print(json.dumps(dict(progress='detector',cycle=c,world=seed,arm=arm,start=start)),flush=True)
    return out


def rate_summary(rows,c,metric):
    worlds = sorted({r['world'] for r in rows})
    a = np.zeros((len(worlds),2,2))
    for r in rows:
        if r['metric']==metric:
            a[worlds.index(r['world']),int(r['arm']=='policy')] += [r['count'],r['exposure_cav_h']]
    ix = np.random.default_rng(SEED).integers(0,len(a),(NBOOT,len(a)))
    sums = a.sum(0)
    draws=a[ix].sum(1)
    assert np.all(sums[:,1]>0) and np.all(draws[:,:,1]>0)
    rate=sums[:,0]/sums[:,1]
    bd=draws[:,:,0]/draws[:,:,1]
    for label,val,bs in [('native',rate[0],bd[:,0]),('policy',rate[1],bd[:,1]),
                          ('policy_minus_native',rate[1]-rate[0],bd[:,1]-bd[:,0])]:
        lo,hi=ci(bs)
        ROWS.append(dict(level='behaviour_estimate',cycle=c,metric=metric,arm=label,
            value=float(val),ci_low=lo,ci_high=hi,n_worlds=len(a),units='events/CAV-hour',
            count_native=float(sums[0,0]),count_policy=float(sums[1,0]),
            exposure_native=float(sums[0,1]),exposure_policy=float(sums[1,1])))
    return a,rate[1]-rate[0]


def existing_predictions(c,worlds):
    out = []
    for seed,w in worlds.items():
        root = safe(w['artifact'])
        for record in map(json.loads,read(root/'SUMOCREDIT_POINTS.jsonl').decode().splitlines()):
            if not record.get('imagined') or not record.get('sumo'):
                continue
            pred,real=record['imagined'],record['sumo']
            if not {'ALL','B'} <= pred.keys() or not {'ALL','B'} <= real.keys():
                continue
            assert record['horizon_s'] in (30,300),record['horizon_s']
            p=pred['ALL']-pred['B']
            r=real['ALL']-real['B']
            row=dict(level='wm_start',metric='real_minus_predicted_dJ',cycle=c,world=int(seed),
                start_s=record['t_s'],horizon_s=record['horizon_s'],predicted_dJ=p,realised_dJ=r,
                value=r-p,protocol='existing_group_eligible_ALL_B')
            ROWS.append(row)
            out.append(row)
    return out


def prediction_summary(rows,c,h):
    rows=[r for r in rows if r['horizon_s']==h]
    if not rows:
        return dict(cycle=c,available_starts=0)
    ws=sorted({r['world'] for r in rows})
    bias=estimate(f'WM{h}_real_minus_predicted_bias',[np.mean([r['value'] for r in rows if r['world']==w]) for w in ws],cycle=c,horizon_s=h)
    p=np.array([r['predicted_dJ'] for r in rows])
    real=np.array([r['realised_dJ'] for r in rows])
    corr=float(np.corrcoef(p,real)[0,1]) if np.std(p)*np.std(real)>0 else None
    rank=float(np.corrcoef(pd.Series(p).rank().to_numpy(),pd.Series(real).rank().to_numpy())[0,1])
    grouped=[np.array([[r['predicted_dJ'],r['realised_dJ']] for r in rows if r['world']==w]) for w in ws]
    rng=np.random.default_rng(SEED)
    pearson_draws=[]
    rank_draws=[]
    for indices in rng.integers(0,len(ws),(NBOOT,len(ws))):
        x=np.concatenate([grouped[i] for i in indices])
        if np.std(x[:,0])*np.std(x[:,1])>0:
            pearson_draws.append(float(np.corrcoef(x.T)[0,1]))
            rank_draws.append(float(np.corrcoef(pd.Series(x[:,0]).rank(),pd.Series(x[:,1]).rank())[0,1]))
    for metric,value,draws in [('pearson',corr,pearson_draws),('spearman',rank,rank_draws)]:
        lo,hi=ci(draws)
        ROWS.append(dict(level='wm_correlation',metric=metric,cycle=c,horizon_s=h,value=value,
                         ci_low=lo,ci_high=hi,n_worlds=len(ws),n_starts=len(rows),valid_draws=len(draws)))
    return dict(cycle=c,available_starts=len(rows),bias=bias,pearson=corr,spearman=rank,
                predicted_mean=float(p.mean()),realised_mean=float(real.mean()),horizon_s=h,
                pearson_ci=ci(pearson_draws),spearman_ci=ci(rank_draws))


def main():
    census_rows=census()
    results={'bootstrap':dict(draws=NBOOT,seed=SEED,cluster='world',interval='pointwise percentile95')}
    all_real=[]
    rates={}
    pred=[]
    for c,tag in enumerate(['c00.a4','c01.a1','c02.a1']):
        rs=returns(c,tag)
        all_real.extend(rs)
        bs=behaviour(c,tag)
        for b in ('metering','multi_slow','side_by_side','vacate'):
            rates[c,b]=rate_summary(bs,c,b)
        if c in (1,2):
            predictions=existing_predictions(c,census_rows[c]['worlds'])
            for row in predictions:
                if row['horizon_s']==300:
                    paired=next(r for r in rs if r['world']==row['world'] and r['start_s']==row['start_s'])
                    assert abs(paired['value']-row['realised_dJ'])<1e-6
            pred.extend(prediction_summary(predictions,c,h) for h in (30,300))
    real=pd.DataFrame(all_real)
    world=real.groupby(['cycle','world','demand','share'],as_index=False)[['value','road_delta','source_delta']].mean()
    assert (real.groupby(['cycle','world']).size()==2).all()
    for r in world.to_dict('records'):
        ROWS.append(dict(level='world',metric='mean_start_policy_minus_native_J300',**r))
        ROWS.append(dict(level='world',metric='two_start_sum_policy_minus_native_J300',
                         **{**r,'value':2*r['value'],'road_delta':2*r['road_delta'],'source_delta':2*r['source_delta']}))
    overall=[]
    for c,g in world.groupby('cycle'):
        overall.append(estimate('mean_start_policy_minus_native_J300',g.value.to_numpy(),cycle=int(c),horizon_s=300))
        estimate('two_start_sum_policy_minus_native_J300',2*g.value.to_numpy(),cycle=int(c),horizon_s=300)
        for col in ('share','demand'):
            for val,cell in g.groupby(col):
                estimate('mean_start_policy_minus_native_J300',cell.value.to_numpy(),cycle=int(c),**{col:float(val)},stratum=col)
        for (share,demand),cell in g.groupby(['share','demand']):
            estimate('mean_start_policy_minus_native_J300',cell.value.to_numpy(),cycle=int(c),share=float(share),demand=int(demand),stratum='share_x_demand')
    # Pool means within worlds, never promote the two starts to independent units.
    results['pooled']=estimate('pooled_mean_start_policy_minus_native_J300',world.value.to_numpy(),cycle='all')
    estimate('pooled_complete_cycles_mean_start_policy_minus_native_J300',world[world.cycle>0].value.to_numpy(),cycle='1_and_2')
    results['cycles']=overall
    for col in ('share','demand'):
        for val,g in world.groupby(col):
            estimate('pooled_mean_start_policy_minus_native_J300',g.value.to_numpy(),cycle='all',**{col:float(val)},stratum=col)
    results['cycle2_minus_cycle1']=difference('cycle2_minus_cycle1_mean_start_J300',world[world.cycle==2].value.to_numpy(),world[world.cycle==1].value.to_numpy())
    for col in ('share','demand'):
        for val,g in world.groupby(col):
            a=g[g.cycle==2].value.to_numpy()
            b=g[g.cycle==1].value.to_numpy()
            if len(a) and len(b):
                difference('cycle2_minus_cycle1_mean_start_J300',a,b,stratum=col,**{col:float(val)})
    results['metering'] = []
    for metric in ('metering','multi_slow','side_by_side','vacate'):
        a,va=rates[2,metric]
        b,vb=rates[1,metric]
        rng=np.random.default_rng(SEED)
        def effect(x):
            s=x.sum(1)
            return s[:,1,0]/s[:,1,1]-s[:,0,0]/s[:,0,1]
        draw=effect(a[rng.integers(0,len(a),(NBOOT,len(a)))] )-effect(b[rng.integers(0,len(b),(NBOOT,len(b)))])
        lo,hi=ci(draw)
        row=dict(level='behaviour_contrast',metric=metric,value=float(va-vb),ci_low=lo,ci_high=hi,
            units='events/CAV-hour',contrast='cycle2_minus_cycle1_policy_native_effect',pairing='unpaired_across_cycles')
        ROWS.append(row)
        results['metering'].append(row)
        # Raw policy/native changes are secondary to the baseline-adjusted contrast.
        for arm,index in [('native',0),('policy',1)]:
            rng=np.random.default_rng(SEED)
            sa=a[rng.integers(0,len(a),(NBOOT,len(a)))].sum(1)
            sb=b[rng.integers(0,len(b),(NBOOT,len(b)))].sum(1)
            draws=sa[:,index,0]/sa[:,index,1]-sb[:,index,0]/sb[:,index,1]
            aa=a.sum(0)[index]
            bb=b.sum(0)[index]
            lo,hi=ci(draws)
            ROWS.append(dict(level='behaviour_raw_contrast',metric=metric,arm=arm,
                value=float(aa[0]/aa[1]-bb[0]/bb[1]),ci_low=lo,ci_high=hi,
                units='events/CAV-hour',contrast='cycle2_minus_cycle1',pairing='unpaired_across_cycles'))
    results['wm_existing']=pred
    for h in (30,300):
        one=[r for r in ROWS if r.get('level')=='wm_start' and r.get('cycle')==1 and r.get('horizon_s')==h]
        two=[r for r in ROWS if r.get('level')=='wm_start' and r.get('cycle')==2 and r.get('horizon_s')==h]
        def world_bias(rows):
            return [np.mean([r['value'] for r in rows if r['world']==w]) for w in sorted({r['world'] for r in rows})]
        difference(f'WM{h}_cycle2_minus_cycle1_optimism',world_bias(two),world_bias(one),horizon_s=h)
    results['wm60']={'status':'no existing horizon-matched60s ALL/B predictions located; 10s selection panel is different starts',
                     'note':'30/300s credit observations are eligibility-selected subsets of the paired starts'}
    ROWS.append(dict(level='unavailable',metric='WM60_real_minus_predicted_bias',horizon_s=60,
                     reason='No collected60s ALL/B predictions; do not extrapolate30/300s or the different10s selection panel'))
    results['census']={c:{k:v for k,v in d.items() if k!='worlds'} for c,d in census_rows.items()}
    (HOME/'RESULTS.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    (HOME/'INPUTS.json').write_text(json.dumps(INPUTS,indent=2))
    fields=sorted(set().union(*(r.keys() for r in ROWS)))
    with (HOME/'C2REAL.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields)
        w.writeheader()
        w.writerows(ROWS)
    print(json.dumps(dict(done=True,start_pairs=len(all_real),worlds=len(world))),flush=True)


if __name__=='__main__':
    main()
