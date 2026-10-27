"""Collect task-only CPU evidence, reproduce registered statistics and paired residuals."""
import hashlib
import json

import numpy as np

from evidence.registered_gate import bootstrap as registered_bootstrap
from evidence.registered_gate import pearson, sign_agreement as sign
from operate import ROOT, remote

EVIDENCE = ROOT/'tools/longhzn/evidence'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bootstrap(groups, statistic, seed):
    point, ci = registered_bootstrap(groups,statistic,seed=seed)
    return {'value':point, 'ci95':ci}


def error_stat(rows, kind):
    error = np.asarray(rows)[:,0] - np.asarray(rows)[:,1]
    bias = float(error.mean())
    mse = float(np.mean(error**2))
    return {'bias':bias, 'sd':float(error.std(ddof=0)), 'rmse':mse**.5,
            'bias_fraction_mse':bias*bias/mse, 'mae':float(np.abs(error).mean())}[kind]


def collect(host):
    source = '''import hashlib,json,pathlib
r=pathlib.Path.home()/'longhzn'
names=('ARTIFACTS.json','STAGED.json','LAUNCH.json','WORKERS.json','DONE.json','PANEL_VERIFIED.json','original_gate.json','threshold.json')
out={n:json.loads((r/n).read_text()) for n in names}
out['rows']=[json.loads(p.read_text()) for p in sorted((r/'rows').glob('*.json'))]
out['row_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((r/'rows').glob('*.json'))}
out['own_live']=[]
for rec in [out['LAUNCH.json'],*out['WORKERS.json']]:
 p=pathlib.Path(f"/proc/{rec['pid']}/stat")
 if p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==rec['start_ticks']: out['own_live'].append(rec['pid'])
print(json.dumps(out))
'''
    return json.loads(remote(host,source,echo=False))


def main():
    assert digest((EVIDENCE/'registered_gate.py').read_bytes()) == 'e9aac75ffd3a96fad9bedf0dcd6f38995bcd5eb9e7fa11dd6a353091f7a69970'
    boxes = {host:collect(host) for host in ('cpuP','cpuQ')}
    left,right = boxes.values()
    assert left['ARTIFACTS.json'] == right['ARTIFACTS.json']
    assert left['PANEL_VERIFIED.json']['sha256_by_file'] == right['PANEL_VERIFIED.json']['sha256_by_file']
    assert all(b['DONE.json']['all_ok'] and not b['own_live'] for b in boxes.values())
    rows = [row for b in boxes.values() for row in b['rows']]
    assert len(rows) == 288
    original = left['original_gate.json']
    original_r16 = json.loads((EVIDENCE/'original_r16_gate.json').read_text())
    assert original_r16['wm_sha256'] == left['ARTIFACTS.json']['r16.pt']
    assert original_r16['threshold_source']['sha256'] == left['ARTIFACTS.json']['threshold.json']
    assert [r['pair_key'] for r in original_r16['rows']] == [r['pair_key'] for r in original['rows']]
    groups = {name: sorted((r for r in rows if r['model']==name),key=lambda r:r['group']) for name in ('wm0','wm1','r16')}
    for name, value in groups.items():
        assert [r['group'] for r in value] == list(range(96))
        assert [r['pair_key'] for r in value] == [r['pair_key'] for r in original['rows']]
        assert all(r['seed']==140014+7919*r['group'] and r['steps']==600 and r['arms']==2 for r in value)
    results = {'threshold':left['threshold.json']['threshold'],'models':{},'changes':{},'verification':{}}
    for name,value in groups.items():
        results['models'][name] = {}
        for h in ('60','90','120','180','300'):
            panel = [r['dJ'][h] for r in value]
            array = np.asarray(panel)
            assert array.shape == (96,2,2) and np.isfinite(array).all()
            assert np.array_equal(array[:,:,1], np.asarray([r['dJ'][h] for r in groups['wm1']])[:,:,1])
            entry = {'sign':bootstrap(panel,sign,int(h)), 'pearson':bootstrap(panel,pearson,int(h)+1),
                     'n_pairs':192,'n_nonzero_sign':int((array[:,:,1]!=0).sum()),'n_groups':96}
            entry['verdict'] = 'PASS' if entry['sign']['value']>=results['threshold'] else 'FAIL'
            if h in ('120','300'):
                entry['residual'] = {k:bootstrap(panel,lambda x,k=k:error_stat(x,k),int(h))
                                     for k in ('bias','sd','rmse','bias_fraction_mse','mae')}
                bias,sd,rmse = [entry['residual'][k]['value'] for k in ('bias','sd','rmse')]
                assert np.isclose(rmse**2,bias**2+sd**2,rtol=1e-12)
            results['models'][name][h] = entry
    for h in ('60','90','120','180','300'):
        results['changes'][h] = {}
        for old,new in (('wm0','wm1'),('r16','wm0'),('r16','wm1')):
            panel = [[(a[0],b[0],a[1]) for a,b in zip(ra['dJ'][h],rb['dJ'][h])]
                     for ra,rb in zip(groups[old],groups[new])]
            def delta_sign(rows):
                a = np.asarray(rows)
                a = a[a[:,2]!=0]
                return float((np.sign(a[:,1])==np.sign(a[:,2])).mean()-(np.sign(a[:,0])==np.sign(a[:,2])).mean())
            def delta_pearson(rows):
                a = np.asarray(rows)
                return float(np.corrcoef(a[:,1],a[:,2])[0,1]-np.corrcoef(a[:,0],a[:,2])[0,1])
            results['changes'][h][old+'->'+new] = {'sign':bootstrap(panel,delta_sign,int(h)),
                                                   'pearson':bootstrap(panel,delta_pearson,int(h)+1)}
    old60 = np.asarray([r['dJ']['60'] for r in original['rows']])
    new60 = np.asarray([r['dJ']['60'] for r in groups['wm1']])
    assert results['models']['wm1']['60']['sign']['value'] == original['horizons']['60']['sign_agreement']
    assert results['models']['wm1']['60']['sign']['ci95'] == original['horizons']['60']['sign_ci95']
    assert results['models']['r16']['60']['sign']['value'] == original_r16['horizons']['60']['sign_agreement']
    assert results['models']['r16']['60']['sign']['ci95'] == original_r16['horizons']['60']['sign_ci95']
    results['verification'] = {'all_288_rows_finite_complete':True,'pair_reference_exact_across_models':True,
                               'panel_bytes_equal_across_boxes':True,'mse_decomposition_pass':True,
                               'wm1_original60_max_abs_delta_veh_s':float(np.abs(new60-old60).max()),
                               'wm1_original60_sign_and_ci_exact':True,
                               'r16_original60_sign_and_ci_exact':True,
                               'worlds':len({r['pair_key'].split('@')[0] for r in groups['wm1']}),
                               'bootstrap':'registered paired-group percentile,2000 draws,seeds h/sign and h+1/Pearson; SD ddof0',
                               'zero_new_sumo_training':True}
    EVIDENCE.mkdir(exist_ok=True)
    (EVIDENCE/'rows.json').write_text(json.dumps(groups,indent=1)+'\n',encoding='utf-8')
    receipts = {}
    for host,box in boxes.items():
        receipts[host] = {k:v for k,v in box.items() if k not in ('rows','original_gate.json','threshold.json','PANEL_VERIFIED.json')}
        receipts[host]['panel_verified'] = {k:v for k,v in box['PANEL_VERIFIED.json'].items() if k != 'sha256_by_file'}
        receipts[host]['panel_file_map_sha256'] = digest(json.dumps(box['PANEL_VERIFIED.json']['sha256_by_file'],sort_keys=True).encode())
    (EVIDENCE/'receipts.json').write_text(json.dumps(receipts,indent=1)+'\n',encoding='utf-8')
    results['evidence_sha256'] = {n:digest((EVIDENCE/n).read_bytes()) for n in ('rows.json','receipts.json')}
    (EVIDENCE/'results.json').write_text(json.dumps(results,indent=1)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=1))


if __name__ == '__main__':
    main()
