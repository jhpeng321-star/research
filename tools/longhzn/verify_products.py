"""Offline numerical and publication checks; no inference or simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from evidence.registered_gate import pearson, sign_agreement

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'tools/longhzn/evidence'
result = json.loads((EVIDENCE/'results.json').read_text())
rows = json.loads((EVIDENCE/'rows.json').read_text())
receipts = json.loads((EVIDENCE/'receipts.json').read_text())
report = (ROOT/'LONGHZN.md').read_text(encoding='utf-8')
assert len(report.split()) <= 300
for name, sha in result['evidence_sha256'].items():
    assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() == sha
assert result['threshold'] == 0.868421052631579
assert all(box['DONE.json']['all_ok'] and not box['own_live'] for box in receipts.values())
assert all(len(box['WORKERS.json'])==16 for box in receipts.values())
for model in ('wm0','wm1','r16'):
    assert len(rows[model]) == 96
    for h in ('60','90','120','180','300'):
        entry = result['models'][model][h]
        pairs = [p for row in rows[model] for p in row['dJ'][h]]
        array = np.asarray(pairs)
        assert array.shape == (192,2) and np.isfinite(array).all()
        assert int((array[:,1]!=0).sum()) == entry['n_nonzero_sign'] == 190
        assert sign_agreement(pairs) == entry['sign']['value']
        assert pearson(pairs) == entry['pearson']['value']
        assert (entry['verdict']=='PASS') == (int(h)<=90)
        if h in ('120','300'):
            e = array[:,0]-array[:,1]
            bias = float(e.mean())
            sd = float(e.std(ddof=0))
            mse = float(np.mean(e**2))
            assert bias == entry['residual']['bias']['value']
            assert sd == entry['residual']['sd']['value']
            assert np.isclose(mse,bias*bias+sd*sd,rtol=1e-12)
            assert np.isclose(entry['residual']['bias_fraction_mse']['value'],bias*bias/mse,rtol=1e-12)
        rendered = f"{entry['sign']['value']:.3f} [{entry['sign']['ci95'][0]:.3f},{entry['sign']['ci95'][1]:.3f}]; {entry['pearson']['value']:.3f}"
        rendered = rendered.replace('0.','.')
        line = next(line for line in report.splitlines() if line.startswith(f'- **{h} s:**'))
        assert rendered in line, (model,h,rendered)
proof = {'pass':True,'report_words':len(report.split()),'model_horizons_verified':15,
         'residual_decompositions_verified':6,'all_workers_exited':True,
         'report_sha256':hashlib.sha256((ROOT/'LONGHZN.md').read_bytes()).hexdigest()}
(EVIDENCE/'VERIFIED.json').write_text(json.dumps(proof,indent=1)+'\n')
print(json.dumps(proof))
