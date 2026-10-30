"""Complete-panel aggregation; missing training sigma never inherits legacy sigma."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.gated300.cct1 import horizon_pairs, horizon_result as cct1_result
from tools.gated300.cct2 import horizon_result as cct2_result, pool

OLD_SIGMA={'60':9.5,'90':25.38,'120':41.22,'180':42.11,'300':43.91}
METRICS=('J','road','source','upstream','downstream','t0_HDV','queue','discharge')
EXPECTED={'validation':36,'pairs300':48}


def stats(pred,real):
    p,r=np.asarray(pred,float),np.asarray(real,float)
    e=p-r
    return dict(n=len(p),sign=float((np.sign(p[r!=0])==np.sign(r[r!=0])).mean())
                if (r!=0).any() else None,mae=float(abs(e).mean()),bias=float(e.mean()),
                correlation=float(np.corrcoef(p,r)[0,1]) if p.std()>0 and r.std()>0 else None,
                error_sd=float(e.std()),mean_predicted=float(p.mean()),mean_real=float(r.mean()))


def boot_sign(rows,pred,real,select):
    worlds=sorted({r['world'] for r in rows})
    counts=[]
    for w in worlds:
        ix=np.array([r['world']==w for r in rows]) & select & (real!=0)
        counts.append([int(((np.sign(pred)==np.sign(real)) & ix).sum()),int(ix.sum())])
    c=np.asarray(counts)
    draws=c[np.random.default_rng(0).integers(0,len(c),(2000,len(c)))].sum(1)
    good=draws[:,1]>0
    return np.quantile(draws[good,0]/draws[good,1],[.025,.975]).tolist() if good.any() else [None,None]


def calls(rows,draws,real,sigma):
    median=np.quantile(draws,.5,axis=1)
    gain=(median < -2*sigma) & (np.quantile(-draws,.1,axis=1)>0)
    harm=(median > 2*sigma) & (np.quantile(draws,.1,axis=1)>0)
    confident=gain|harm
    signable=real!=0
    correct=np.sign(median)==np.sign(real)
    agreement=lambda mask:float(correct[mask & signable].mean()) if (mask & signable).any() else None
    wrong=confident & signable & ~correct
    wrongsize=float(abs(real[wrong]).mean()) if wrong.any() else 0.
    ga=agreement(gain);ca=agreement(confident)
    return dict(sigma=sigma,threshold=2*sigma,gain=int(gain.sum()),harm=int(harm.sum()),
        gain_sign=ga,harm_sign=agreement(harm),confident_sign=ca,
        gain_ci95=boot_sign(rows,median,real,gain),coverage=float(confident.mean()),
        wrong_mean_abs_real=wrongsize,
        CCT1_style_pass=bool(ca is not None and ca>=.8 and confident.mean()>=.25 and wrongsize<=2*sigma),
        CCT2_style_pass=bool(int(gain.sum())>=20 and ga is not None and ga>=.8))


def legacy(evidence):
    docs=[json.loads((evidence/f'WM2mix_{p}.json').read_text()) for p in ('validation','pairs300')]
    result={}
    for panel,doc in zip(('validation','pairs300'),docs):
        result[panel]={}
        for h in OLD_SIGMA:
            pairs=[v for r in doc['rows'] for v in r['dJ'][h]]
            a=np.asarray(pairs)
            result[panel][h]=stats(a[:,0],a[:,1])
    return result


def legacy_calls(evidence):
    docs=[json.loads((evidence/f'WM2mix_{p}.json').read_text()) for p in EXPECTED]
    panels={panel:{h:dict(CCT1=cct1_result(horizon_pairs(doc['rows'],h),sigma),
                         CCT2=cct2_result(horizon_pairs(doc['rows'],h),sigma))
                   for h,sigma in OLD_SIGMA.items()}
            for panel,doc in zip(EXPECTED,docs)}
    pooled={h:cct2_result(pool(docs,h),sigma) for h,sigma in OLD_SIGMA.items()}
    return dict(panels=panels,pooled_CCT2=pooled,
                H_G=max([0]+[int(h) for h in ('120','180','300') if pooled[h]['verdict']=='PASS']))


def outcome_equivalent_noise(real):
    """Measured spread of recorded equivalent controls, not null-perturbation SD."""
    rows=real['validation']
    values={}
    for h in OLD_SIGMA:
        d=[]
        for key in sorted({r['pair_key'] for r in rows}):
            a={r['arm']:r['values'][h]['J'] for r in rows if r['pair_key']==key}
            d.append(a['ours_c1']-a['coopoff_c1'])
        values[h]=dict(n=len(d),mean=float(np.mean(d)),sigma=float(np.std(d,ddof=1)),
                       two_sigma=float(2*np.std(d,ddof=1)),max_abs=float(np.max(np.abs(d))))
    return dict(contrast='recorded ours_c1 minus coopoff_c1, validation only',horizons=values,
                independent_null_perturbations=False,
                interpretation='Equivalent-control check only; does not identify training-objective sensitivity sigma.')


def rows_for(panel,real,files,legacy_real=None):
    all_draws=[r for p in files for r in json.loads(p.read_text())['rows']]
    assert len(all_draws)==len(real)*16, 'incomplete or duplicated K8/mix rows'
    assert len({(r['pair_key'],r['draw'],r['mix'],r['arm']) for r in all_draws})==len(all_draws)
    lookup={(r['pair_key'],r['arm']):r for r in real}
    keys=sorted({r['pair_key'] for r in real})
    assert len(keys)==EXPECTED[panel]
    assert {r['pair_key'] for r in all_draws}==set(keys), 'incomplete panel keys'
    out=[]
    for key in keys:
        rr=[r for r in real if r['pair_key']==key]
        native=next(r for r in rr if r['native'])
        for mix in (False,True):
            for arm in rr:
                if arm['native']:
                    continue
                wm=[r for r in all_draws if r['pair_key']==key and r['mix']==mix and r['arm']==arm['arm']]
                ref=[r for r in all_draws if r['pair_key']==key and r['mix']==mix and r['arm']=='native']
                assert len(wm)==len(ref)==8 and {r['draw'] for r in wm}=={r['draw'] for r in ref}==set(range(8))
                wm.sort(key=lambda r:r['draw']);ref.sort(key=lambda r:r['draw'])
                horizons={}
                for h,values in arm['values'].items():
                    d=np.array([[r['values'][h][m]-b['values'][h][m] for m in METRICS]
                                for r,b in zip(wm,ref)])
                    realdiff=np.array([values[m]-native['values'][h][m] for m in METRICS])
                    if legacy_real is not None:
                        realdiff[5]=legacy_real[key,arm['arm']]['values'][h]-legacy_real[key,'native']['values'][h]
                    # Use the same order-statistic weights for every component:
                    # component medians alone would not sum to median total J.
                    order=np.argsort(d[:,0],kind='stable')
                    joint=(d[order[3]]+d[order[4]])/2
                    assert abs(joint[0]-joint[2]-joint[3]-joint[4])<.02
                    horizons[h]=dict(draws=d.tolist(),real=realdiff.tolist(),joint_median=joint.tolist(),
                        absolute_predicted=np.mean([[r['values'][h][m] for m in METRICS] for r in wm],axis=0).tolist(),
                        absolute_real=[values[m] for m in METRICS])
                out.append(dict(pair_key=key,world=arm['world'],arm=arm['arm'],mix=mix,
                    demand=arm['demand'],penetration=arm['penetration'],horizons=horizons))
    return out,all_draws


def aggregate(rows):
    result={}
    for mode in (False,True):
        rs=[r for r in rows if r['mix']==mode]
        byh={}
        for h in OLD_SIGMA:
            keep=[r for r in rs if h in r['horizons']]
            if not keep:
                continue
            d=np.array([r['horizons'][h]['draws'] for r in keep])
            real=np.array([r['horizons'][h]['real'] for r in keep])
            mean=d.mean(1)
            med=np.array([r['horizons'][h]['joint_median'] for r in keep])
            e=med-real
            mse=float(np.square(e[:,0]).mean())
            comps={m:dict(bias=float(e[:,j].mean()),mae=float(abs(e[:,j]).mean()),
                          mse_covariance_share=float((e[:,j]*e[:,0]).mean()/mse) if mse else None)
                   for j,m in enumerate(METRICS) if m in ('source','upstream','downstream')}
            for m in comps:
                j=METRICS.index(m)
                comps[m]['J_mae_after_perfect_component_correction']=float(abs(e[:,0]-e[:,j]).mean())
            q=stats(med[:,2],real[:,2]);ql=stats(med[:,6],real[:,6])
            abs_p=np.array([r['horizons'][h]['absolute_predicted'] for r in keep])
            abs_r=np.array([r['horizons'][h]['absolute_real'] for r in keep])
            # Descriptive conditional association, not a causal decomposition.
            design=np.column_stack((np.ones(len(e)),e[:,2],e[:,7]))
            fitted=design@np.linalg.lstsq(design,e[:,0],rcond=None)[0]
            denom=np.square(e[:,0]-e[:,0].mean()).sum()
            def r2(columns):
                x=np.column_stack([np.ones(len(e)), *columns])
                fit=x@np.linalg.lstsq(x,e[:,0],rcond=None)[0]
                return float(1-np.square(e[:,0]-fit).sum()/denom) if denom else None
            association=dict(joint_source_discharge_error_R2=float(1-np.square(e[:,0]-fitted).sum()/denom)
                              if denom else None,
                              source_error_R2=r2([e[:,2]]),
                              discharge_error_R2=r2([e[:,7]]),
                              incremental_discharge_R2_given_source=(r2([e[:,2],e[:,7]])-r2([e[:,2]])) if denom else None,
                              road_error_discharge_error_correlation=stats(e[:,1],e[:,7])['correlation'],
                             causal=False)
            byh[h]=dict(J_median=stats(med[:,0],real[:,0]),J_mean=stats(mean[:,0],real[:,0]),
                legacy_population_matched_K8=stats(np.quantile(d[:,:,5],.5,axis=1),real[:,5]),
                source_change=q,queue_change=ql,
                source_absolute=stats(abs_p[:,2],abs_r[:,2]),queue_absolute=stats(abs_p[:,6],abs_r[:,6]),
                discharge_change_veh=stats(med[:,7],real[:,7]),decomposition=comps,
                discharge_association=association,
                CCT1_zero_scale_sensitivity=cct1_result(
                    {w:[[float(med[i,0]),float(real[i,0])] for i,r in enumerate(keep) if r['world']==w]
                     for w in sorted({r['world'] for r in keep})},0.),
                training_sigma=None,training_CCT1=None,training_CCT2=None,
                conditional_zero_replicate_sigma=calls(keep,d[:,:,0],real[:,0],0.),
                conditional_legacy_scale_sensitivity=calls(keep,d[:,:,0],real[:,0],OLD_SIGMA[h]))
        result['ON' if mode else 'OFF']=byh
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence',type=Path,default=Path(__file__).parent/'evidence')
    ap.add_argument('--out',type=Path,default=Path(__file__).parent/'evidence/SUMMARY.json')
    a=ap.parse_args()
    real=json.loads((a.evidence/'REAL.json').read_text())['panels']
    canonical=json.loads((a.evidence/'LEGACY_REAL.json').read_text())
    result=dict(schema='wmrequal_summary/2',panels={},metrics=list(METRICS),
        registered_panels=list(EXPECTED),skipped_panels=['calibration','panel120'],
        legacy_K1=legacy(a.evidence),legacy_K1_calls=legacy_calls(a.evidence),
        H_G_training=None,operational_H_G_training=0,policy_authorized=False,
        sigma_training=None,replicate_sigma=0.,
        equivalent_control_noise=outcome_equivalent_noise(real),
        blockers=['pairs300 initial per-source backlog is not identifiable from truncated counters',
                  'no null perturbation/replicate arms on these recorded verdict panels; training-objective perturbation sigma unmeasured'],
        rows=[])
    for panel in EXPECTED:
        folder=a.evidence/panel
        if not (folder/'DONE.json').is_file():
            blocker=a.evidence/'PAIRS300_BLOCKER.json'
            if panel!='pairs300' or not blocker.is_file():
                raise ValueError('panel not complete: '+panel)
            audit=json.loads(blocker.read_text())
            assert audit['episodes']==144 and audit['missing_per_source_Q0']==144
            result['panels'][panel]=dict(status='BLOCKED',scored=False,
                reason=audit['conclusion'],recorded_episodes=len(real[panel]),
                recorded_delta_J={h:dict(n=96,mean=float(np.mean([
                    r['values'][h]['J']-next(n for n in real[panel]
                        if n['pair_key']==r['pair_key'] and n['native'])['values'][h]['J']
                    for r in real[panel] if not r['native']]))) for h in OLD_SIGMA})
            continue
        lookup={(r['pair_key'],r['arm']):r for r in canonical[panel]}
        files=sorted(folder.glob('batch_*.json'))
        assert len(files)==EXPECTED[panel]
        receipt_hashes=set()
        real_hash=hashlib.sha256((a.evidence/'REAL.json').read_bytes()).hexdigest()
        for file in files:
            receipt=json.loads(file.read_text())['receipt']
            assert receipt['K']==8 and receipt['steps']==600 and receipt['physical_exits'] and receipt['queue_state']
            assert receipt['real_sha256']==real_hash
            assert receipt['gpu_fraction']<=.9
            assert receipt['wm_sha256']=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
            receipt_hashes.add((receipt['source_sha256'],receipt['config_sha256'],receipt['wm_sha256']))
        assert len(receipt_hashes)==1,'mixed runtime receipts'
        rows,draws=rows_for(panel,real[panel],files,lookup)
        result['panels'][panel]=aggregate(rows)
        result['panels'][panel]['provenance']=dict(batches=len(files),rows=len(draws),
            source_config_wm_sha256=list(next(iter(receipt_hashes))),real_sha256=real_hash)
        # Cell summaries retain all horizons, both settings, absolute and paired source/queue metrics.
        cells={}
        for demand,penetration in sorted({(r['demand'],r['penetration']) for r in rows}):
            cells[f'{demand}/{penetration}']=aggregate([r for r in rows if r['demand']==demand and r['penetration']==penetration])
        result['panels'][panel]['cells']=cells
        result['rows']+=rows
    if all('ON' in result['panels'][p] for p in EXPECTED):
        result['pooled']=aggregate(result['rows'])
    else:
        result['pooled']=None
    result['input_sha256']={str(p.relative_to(a.evidence)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in a.evidence.rglob('*.json') if p!=a.out and p.name!='SUMMARY.json'}
    a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps(dict(H_G_training=result['H_G_training'],sigma_training=None,
        panels={p:({h:v['J_median'] for h,v in r['ON'].items()} if 'ON' in r else r)
                for p,r in result['panels'].items()}),indent=1))


if __name__=='__main__':
    main()
