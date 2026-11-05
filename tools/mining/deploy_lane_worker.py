"""Retire the idle own coordinator gracefully, stage one lane-only successor."""
import base64
import json
from pathlib import Path
import zlib
import sys
from .remote import run


def main():
    generation = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    assert generation in (2, 3)
    names = ('__init__.py', 'collect_contexts.py', 'wm_contract.py', 'wm_prescreen.py', 'wm_adapter.py',
             'wm_lane_adapter.py', 'wm_lane_schedule.py', 'wm_lane_worker.py')
    files = {n: base64.b64encode((Path(__file__).parent / n).read_bytes()).decode() for n in names}
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    code = r'''
import base64,hashlib,json,os,subprocess,time,zlib
from pathlib import Path
parent=Path('/root/mining20261106');prior=parent/'wm_worker01';root=parent/('wm_lane%02d'%GENERATION)
assert parent.is_dir() and not root.exists()
if GENERATION==3:
 failed=parent/'wm_lane02';failed_owner=json.loads((failed/'DISPATCH.json').read_bytes())
 assert not Path('/proc/'+str(failed_owner['pid'])).exists()
 assert (failed/'FAULT.json').is_file() and not (failed/'MODEL_READY.json').exists()
q=json.loads((prior/'FRESH_QUALIFICATION.json').read_bytes());assert q['pass_'] and q['pid']==3230
owner=json.loads((prior/'MINING_WM_OWNER.json').read_bytes())
assert owner['pid']==3230 and owner['start_ticks']=='928158286'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']
proc=Path('/proc/3230')
if proc.exists():
 assert (proc/'stat').read_text().split()[21]==owner['start_ticks']
 assert b'tools.mining.wm_coordinator' in (proc/'cmdline').read_bytes()
 assert not list((prior/'queries').glob('*.json')), 'do not interrupt actual scoring'
 if not (prior/'MINING_STOP.json').exists():
  with (prior/'MINING_STOP.json').open('x') as f:json.dump(dict(reason='accepted G5 lane scorer successor; keep completed eight gates',unix=time.time()),f)
 for _ in range(20):
  if (prior/'MINING_WM_EXIT.json').is_file() and not proc.exists():break
  time.sleep(1)
assert (prior/'MINING_WM_EXIT.json').is_file() and not proc.exists()
root.mkdir();source=root/'pytools';(source/'tools/mining').mkdir(parents=True)
files=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name,data in files.items():(source/'tools/mining'/name).write_bytes(base64.b64decode(data))
(source/'tools/__init__.py').write_text('__path__.extend('+repr([str(prior/'pytools/tools'),'/root/coopmwm20261102_deadlineA2/pytools/tools'])+')\n')
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
(root/'LANE_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True))
(root/'AUTHORITY.json').write_text(json.dumps(dict(original_eight_gates_pid=3230,
 new_mode='unoptimized reference bootstrap + K8/H600 exact lane-OFF comparison before any candidate',
 g5_commit='b903c6c2',K=8,H=600,onsets=[0,10],holds=[20,60],depths=[.3,1.],
 no_sumo=True,rank_only=True,source_manifest=manifest,unix=time.time()),sort_keys=True))
env=dict(os.environ,WMTEACH_BACKEND='FULL-v4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
cmd=['/root/miniconda3/bin/python','-B','-u','-m','tools.mining.wm_lane_worker',str(root)]
preflight=subprocess.run([cmd[0],'-B','-c','from tools.wmteach.runtime import bootstrap; from tools.mining.wm_lane_adapter import compile_bridge; print("IMPORT_CLOSURE_PASS")'],env=env,capture_output=True,text=True,timeout=60)
(root/'IMPORT_QUALIFICATION.json').write_text(json.dumps(dict(returncode=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr)))
assert preflight.returncode==0,preflight.stderr
with (root/'WORKER.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],argv=cmd,root=str(root),unix=time.time())
(root/'DISPATCH.json').write_text(json.dumps(record,sort_keys=True))
print(json.dumps(record),flush=True)
'''
    result = run('gpuA', 'PAYLOAD=' + repr(payload) + '\nGENERATION=' + str(generation) + '\n' + code,
                 '/root/miniconda3/bin/python -B -', timeout=90)
    path = Path('mining_receipts/LANE_WORKER_DISPATCH_%02d.json' % (generation - 1))
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    raise SystemExit(result['returncode'])


if __name__ == '__main__':
    main()
