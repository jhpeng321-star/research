"""Source-bound, inference-only warm qualification; never dispatch SUMO."""
import argparse
import json
from . import ops
from .protocol import checked,sha
from .storage import read,write

ROOT='/root/coopmwm20261101_warmqual1'


def stage(host):
    files=ops.sources()
    files['FROZEN_COSTS.json']=checked(ops.EVIDENCE/'CPU_coopmwm20261101_cpu36.json').read_text(encoding='utf-8')
    # Fetch the already source-bound cost, not this receipt's JSON schema.
    files['FROZEN_COSTS.json']=ops.ssh.remote('gpuA',"""import pathlib
print(pathlib.Path('/root/coopmwm20261101_wmspeed2/FROZEN_COSTS.json').read_text())
""",timeout=20)
    files['FROZEN_MOTION_SOURCE.py']=checked(ops.EVIDENCE/'warm_api/tools__rbint__motion_posterior.py').read_text(encoding='utf-8')
    records=[read(ops.EVIDENCE/f'ACTION_PROBE_INPUT_{i:02}.json') for i in (0,1)]
    files['E2_REQUESTS.json']=json.dumps(records)
    value=json.loads(ops.stage(host,ROOT,files))
    write(ops.EVIDENCE/f'WARMQUAL_STAGE_{host}.json',value)
    print(json.dumps(dict(host=host,root=ROOT,files=len(value['hashes']))))


def fixtures():
    """Open only two explicit archived observable logs; no scenario extraction."""
    value=json.loads(ops.ssh.remote('gpuA',f'''import pathlib,json,tarfile,gzip,hashlib,sys
r=pathlib.Path({ROOT!r});assert r.is_dir()
sys.path.insert(0,str(r/'pytools'))
from tools.coopmwm.warm_history import ObservableHistory,FIELDS,arrays
requests=json.loads((r/'E2_REQUESTS.json').read_text());out=[];receipts=[]
archive=pathlib.Path('/root/autodl-tmp/archive_cpuQ_20261101/coopmwm20261101_v2.tar')
assert archive.stat().st_size==3119974400
with tarfile.open(archive,'r:') as tar:
 for record in requests:
  seed=record['world_id'];now=record['time_s'];assert seed in (986000,986007)
  member=f'coopmwm20261101_v2/jobs/s0_u001_B_{{seed}}/observable.jsonl.gz'
  info=tar.getmember(member);assert info.isfile() and not info.issym()
  history=ObservableHistory();selected=0
  with gzip.GzipFile(fileobj=tar.extractfile(info)) as stream:
   for line in stream:
    frame=json.loads(line);t=frame['time_s']
    if t>now:break
    if t<now-60:continue
    rows=[{{k:v[k] for k in FIELDS}} for v in frame['vehicles']]
    history.push(t,rows);selected+=1
  record['history']=history.snapshot(now);raw=arrays(record)
  assert selected==61
  receipts.append(dict(world_id=seed,time_s=now,member=member,frames=selected,
    start_s=record['history'][0]['time_s'],end_s=record['history'][-1]['time_s'],
    current_slots=raw['current_slots'],valid_samples=int(raw['mask'].sum()),
    observable_only=True,terminal_roster_exact=True))
  out.append(record)
target=r/'E2_INPUTS.json';text=json.dumps(out,sort_keys=True)
with target.open('x') as f:f.write(text)
print(json.dumps(dict(root=str(r),sha256=hashlib.sha256(text.encode()).hexdigest(),
 fixtures=receipts,new_sumo_sim_s=0)))
''',timeout=45))
    write(ops.EVIDENCE/'WARMQUAL_E2_FIXTURES.json',value)
    print(json.dumps(value))


def transfer():
    text=ops.ssh.remote('gpuA',f"import pathlib;print(pathlib.Path({ROOT!r},'E2_INPUTS.json').read_text())",timeout=30).strip()
    value=json.loads(ops.stage('gpuR',ROOT,{'E2_INPUTS.json':text}))
    assert value['hashes']['E2_INPUTS.json']==read(ops.EVIDENCE/'WARMQUAL_E2_FIXTURES.json')['sha256']
    write(ops.EVIDENCE/'WARMQUAL_E2_TRANSFER.json',value)
    print(json.dumps(dict(host='gpuR',sha256=value['hashes']['E2_INPUTS.json'])))


def cpu():
    out=ops.ssh.remote('gpuA',f'''import os,sys,unittest,json,time
from unittest.mock import patch
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',COOPMWM_MAPPO='/root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt',COOPMWM_MOTION_SOURCE={ROOT!r}+'/FROZEN_MOTION_SOURCE.py')
sys.path[:0]=[{ROOT!r}+'/pytools','/root/rbeval20261031/phys_code_22cc32f65a80/src','/root/rbeval20261031/phys_code_22cc32f65a80']
import torch;torch.set_num_threads(1);begin=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ['tools.coopmwm.test_warm_cpu','tools.coopmwm.test_speed2_cpu']])
with patch('subprocess.Popen',side_effect=AssertionError('physics forbidden')):
 result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
print(json.dumps(dict(pass_=result.wasSuccessful(),tests=result.testsRun,skips=len(result.skipped),wall_s=time.monotonic()-begin,root={ROOT!r},new_sumo_sim_s=0)))
''',timeout=60)
    print(out);value=json.loads(out.strip().splitlines()[-1])
    value['source_manifest_sha256']=sha(ops.EVIDENCE/'WARMQUAL_STAGE_gpuA.json')
    write(ops.EVIDENCE/'WARMQUAL_CPU_GPU_RUNTIME.json',value)
    assert value['pass_'] and not value['skips']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','fixtures','transfer','cpu'))
    p.add_argument('--host',choices=('gpuA','gpuR'),default='gpuA');a=p.parse_args()
    if a.mode=='stage':stage(a.host)
    else:globals()[a.mode]()
