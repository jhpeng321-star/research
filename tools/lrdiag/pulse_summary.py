"""Qualification-wave amplification, kept separate from every verdict statistic."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from summarize import write_csv,interval


def main(root,out):
    files=sorted((root/'pulse_results').glob('s*_k*.json'))
    assert len(files)==96,len(files)
    rows=[];signals=[]
    for f in files:
        d=json.loads(f.read_text());assert d['scope']=='supplemental_non_verdict'
        rows.extend(d['rows']);signals.extend(d['signals'])
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'pulse_amplitude_draws.csv',rows);write_csv(out/'pulse_signal_draws.csv',signals)
    chains=defaultdict(list)
    for r in rows:chains[r['seed'],r['draw'],r['pulse'],r['source']].append(r)
    gains=[]
    for (seed,k,pulse,src),rs in sorted(chains.items()):
        good=[r for r in rs if r['amplitude_dv_mps']>0]
        rank=np.array([r['rank'] for r in good]);distance=np.array([r['distance_m'] for r in good])
        amp=np.array([r['amplitude_dv_mps'] for r in good]);y=np.log(amp)
        per_vehicle=float(np.exp(np.polyfit(rank,y,1)[0])) if len(good)>=3 and np.std(rank)>0 else None
        per_100m=float(np.exp(100*np.polyfit(distance,y,1)[0])) if len(good)>=3 and np.std(distance)>0 else None
        gains.append({'scope':'supplemental_non_verdict','seed':seed,'draw':k,'pulse':pulse,'source':src,
            'target_lane':rs[0]['target_lane'],'n_platoon':len(rs),'n_positive_amplitude':len(good),
            'n_zero_amplitude':sum(r['amplitude_dv_mps']==0 for r in rs),
            'log_amplitude_gain_per_vehicle':per_vehicle,'log_amplitude_gain_per_100m':per_100m,
            'interpretation':'finite 30 s initial platoon; positive amplitudes only for log fit; zero count retained'})
    write_csv(out/'pulse_gain_draws.csv',gains)
    gain_groups=defaultdict(list)
    for r in gains:gain_groups[r['seed'],r['pulse'],r['source']].append(r)
    gain_worlds=[]
    gain_keys=('log_amplitude_gain_per_vehicle','log_amplitude_gain_per_100m')
    for (seed,pulse,src),rs in sorted(gain_groups.items()):
        assert len(rs)==8
        row={'scope':'supplemental_non_verdict','seed':seed,'pulse':pulse,'source':src,
             'target_lane':rs[0]['target_lane'],'K':8,'n_platoon':rs[0]['n_platoon']}
        for key in gain_keys:
            vals=[r[key] for r in rs if r[key] is not None and np.isfinite(r[key])]
            row[key]=float(np.median(vals)) if vals else None
            row[key+'_valid_draws']=len(vals)
        row['median_zero_amplitudes']=float(np.median([r['n_zero_amplitude'] for r in rs]))
        gain_worlds.append(row)
    write_csv(out/'pulse_gain_worlds.csv',gain_worlds)
    gain_summary=[]
    for pulse in ('pulse_m','pulse_p'):
        for key in gain_keys:
            pairs=[]
            for seed in range(969824,969836):
                rs=[r for r in gain_worlds if r['seed']==seed and r['pulse'].startswith(pulse) and r['target_lane']<2]
                real={r['pulse']:r[key] for r in rs if r['source']=='real' and r[key] is not None}
                wm={r['pulse']:r[key] for r in rs if r['source']=='wm' and r[key] is not None}
                common=sorted(set(real)&set(wm))
                if common:pairs.append([seed,np.median([real[k] for k in common]),np.median([wm[k] for k in common]),len(common)])
            if not pairs:continue
            a=np.array(pairs,float);rng=np.random.default_rng(20261028)
            diff=a[:,2]-a[:,1];boot=np.median(diff[rng.integers(0,len(a),(2000,len(a)))],axis=1)
            lo,hi=interval(boot)
            gain_summary.append({'scope':'supplemental_non_verdict','pulse':pulse,'metric':key,
                'n_worlds':len(a),'n_matched_targets':int(a[:,3].sum()),'real_median_world_gain':np.median(a[:,1]),
                'wm_median_world_gain':np.median(a[:,2]),'median_paired_gain_difference':np.median(diff),
                'difference_lo95':lo,'difference_hi95':hi,
                'caveat':'initial mainline targets; median over K then targets/world; log fits use positive amplitudes only; '
                         'zero counts/missing fit support retained; NOT validation verdict evidence'})
    write_csv(out/'pulse_gain_summary.csv',gain_summary)
    grouped=defaultdict(list)
    for r in rows:grouped[r['seed'],r['pulse'],r['source'],r['rank']].append(r)
    means=[]
    for (seed,pulse,src,rank),rs in sorted(grouped.items()):
        assert len(rs)==8
        y=[r['amplitude_vs_target'] for r in rs if r['amplitude_vs_target'] is not None]
        means.append({'scope':'supplemental_non_verdict','seed':seed,'pulse':pulse,'source':src,'rank':rank,
            'target_lane':rs[0]['target_lane'],'distance_m':rs[0]['distance_m'],
            'mean_amplitude_dv_mps':float(np.mean([r['amplitude_dv_mps'] for r in rs])),
            'mean_amplitude_vs_target':float(np.mean(y)) if y else None,
            'mean_fraction_common_same_lane':float(np.mean([r['fraction_common_same_lane'] for r in rs]))})
    write_csv(out/'pulse_amplitude_worlds.csv',means)
    coverage={'scope':'supplemental_non_verdict','n_worlds':12,'world_seeds':[969824,969835],
              'K':8,'horizon_s':30,'n_draws':len(files),'WM':'WM2','validation_overlap':0,
              'caveat':'These worlds are outside E3 validation. They cannot establish the verdict. '
                       'Finite-time nonlinear pulse gains, not asymptotic linear string-stability eigenvalues.'}
    (out/'PULSE_COVERAGE.json').write_text(json.dumps(coverage,indent=1))
    print('PASS pulse qualification:',len(rows),'amplitudes,',len(gains),'chain gains')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.root,a.out)
