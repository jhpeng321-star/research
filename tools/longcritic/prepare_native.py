"""Pair the existing 300 s native B ledgers with current-policy start states; no SUMO."""
import json

from remote import ROOT, REMOTE, ssh

r = ssh('gpuR', '''import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
import pathlib,json,numpy as np,io,base64,hashlib
root=pathlib.Path('/root/main14_loop')
path=root/'returns/returns_c00.a1.npz'
receipt=json.loads(path.with_suffix('.receipt.json').read_text())
manifest=json.loads((root/'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json').read_text())
branches={e['pair_key']:e for e in manifest['episodes'] if e.get('branch',{}).get('kind')=='B'}
with np.load(path,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
rows=[r for r in receipt['entries'] if 'pair_key' in r]
assert len(rows)==len(payload['returns'])
targets=[]; proof=[]
for i,row in enumerate(rows):
 p=pathlib.Path(branches[row['pair_key']]['source'])/'summary.json'
 assert p.is_relative_to(root) and not any(x in str(p).lower() for x in ['sealed','holdout','heldout','cal_split','chunk5','j4b','inc1'])
 raw=p.read_bytes(); d=json.loads(raw)
 assert not (d.get('collisions',0) or d.get('teleports',0) or d.get('dropped_vehicle_steps',0))
 assert d['horizon_s']==payload['horizon_s'][i]==300
 assert d['seed']==payload['seeds'][i] and d['requested_population']==payload['requested_population'][i]
 assert np.array_equal(np.asarray(d['start_entry_backlog'],np.float32),payload['state__entry_backlog'][i])
 ledger=d['ledger']; targets.append(-float(ledger['speed_deficit_vehicle_seconds']+ledger['source_wait_vehicle_seconds']))
 proof.append({'pair_key':row['pair_key'],'B_summary_sha256':hashlib.sha256(raw).hexdigest()})
payload['returns']=np.asarray(targets,np.float64); payload['source_kind']=np.array('native'); payload.pop('policy_checkpoint',None)
stream=io.BytesIO(); np.savez_compressed(stream,**payload)
print(json.dumps({'data':base64.b64encode(stream.getvalue()).decode(),'proof':proof,'source_policy_returns_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
''')
if r.returncode:
    raise RuntimeError(r.stderr)
d = json.loads(r.stdout)
raw = d.pop('data')
r = ssh('gpuA', "import pathlib,base64,json\n"
        f"root=pathlib.Path('{REMOTE}'); p=root/'paired_native_returns.npz'; assert not p.exists()\n"
        f"p.write_bytes(base64.b64decode({raw!r}))\n"
        f"(root/'PAIRED_NATIVE_PROVENANCE.json').write_text({json.dumps(d)!r})\nprint('26 paired native B targets staged')\n")
print(r.stdout, r.stderr)
if r.returncode:
    raise RuntimeError('paired native staging failed')
(ROOT/'tools/longcritic/PAIRED_NATIVE_PROVENANCE.json').write_text(json.dumps(d, indent=2), encoding='utf-8')
