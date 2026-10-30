"""Complete-panel aggregation; missing training sigma never inherits legacy sigma."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

OLD_SIGMA={'60':9.5,'90':25.38,'120':41.22,'180':42.11,'300':43.91}
METRICS=('J','road','source','upstream','downstream','t0_HDV','queue','discharge')
EXPECTED={'validation':36,'calibration':36,'panel120':60}


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
        CCT1_style_pass=ca is not None and ca>=.8 and confident.mean()>=.25 and wrongsize<=2*sigma,
        CCT2_style_pass=int(gain.sum())>=20 and ga is not None and ga>=.8)


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


def rows_for(panel,real,files,legacy_real=None):
    all_draws=[r for p in files for r in json.loads(p.read_text())['rows']]
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
                assert len(wm)==len(ref)==8 and {r['draw'] for r in wm}==set(range(8))
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
            association=dict(joint_source_discharge_error_R2=float(1-np.square(e[:,0]-fitted).sum()/denom)
                             if denom else None,
                             road_error_discharge_error_correlation=stats(e[:,1],e[:,7])['correlation'],
                             causal=False)
            byh[h]=dict(J_median=stats(med[:,0],real[:,0]),J_mean=stats(mean[:,0],real[:,0]),
                legacy_population_matched_K8=stats(np.quantile(d[:,:,5],.5,axis=1),real[:,5]),
                source_change=q,queue_change=ql,
                source_absolute=stats(abs_p[:,2],abs_r[:,2]),queue_absolute=stats(abs_p[:,6],abs_r[:,6]),
                discharge_change_veh=stats(med[:,7],real[:,7]),decomposition=comps,
                discharge_association=association,
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
    result=dict(schema='wmrequal_summary/1',panels={},metrics=list(METRICS),
        legacy_K1=legacy(a.evidence),H_G_training=None,policy_authorized=False,
        sigma_training=None,replicate_sigma=0.,
        blockers=['pairs300 initial per-source backlog is not identifiable from truncated counters',
                  'no null perturbation/replicate arms on these recorded verdict panels; training-objective perturbation sigma unmeasured'],
        rows=[])
    for panel in EXPECTED:
        folder=a.evidence/panel
        if not (folder/'DONE.json').is_file():
            raise ValueError('panel not complete: '+panel)
        lookup={(r['pair_key'],r['arm']):r for r in canonical[panel]}
        rows,draws=rows_for(panel,real[panel],sorted(folder.glob('batch_*.json')),lookup)
        result['panels'][panel]=aggregate(rows)
        # Cell summaries retain all horizons, both settings, absolute and paired source/queue metrics.
        cells={}
        for demand,penetration in sorted({(r['demand'],r['penetration']) for r in rows}):
            cells[f'{demand}/{penetration}']=aggregate([r for r in rows if r['demand']==demand and r['penetration']==penetration])
        result['panels'][panel]['cells']=cells
        result['rows']+=rows
    result['input_sha256']={str(p.relative_to(a.evidence)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in a.evidence.rglob('*.json') if p!=a.out and p.name!='SUMMARY.json'}
    a.out.write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256')},indent=1))


if __name__=='__main__':
    main()
