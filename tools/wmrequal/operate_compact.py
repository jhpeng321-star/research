"""Retain completed dense batch, stop exact owned child, probe compact K8."""
import base64
from pathlib import Path
from remote import remote
raw=base64.b64encode((Path(__file__).parent/'score_compact.py').read_bytes()).decode()
remote(f'''import base64,json,pathlib,subprocess,sys,os,signal,time
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
assert (p/'fast/validation/batch_000.json').is_file()
assert not (p/'tools/wmrequal/score_compact.py').exists()
(p/'tools/wmrequal/score_compact.py').write_bytes(base64.b64decode({raw!r}))
owned=json.loads((p/'FAST_CHILDREN.jsonl').read_text().splitlines()[-1])
q=pathlib.Path(f'/proc/{{owned["pid"]}}/stat')
assert q.read_text().rsplit(')',1)[1].split()[19]==owned['start_ticks']
assert str(p/'tools/wmrequal/score_fast.py') in pathlib.Path(f'/proc/{{owned["pid"]}}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
os.kill(owned['pid'],signal.SIGTERM)
(p/'OWNED_DENSE_STOP.json').write_text(json.dumps(dict(time=time.time(),child=owned,completed_batches=len(list((p/'fast/validation').glob('batch_*.json'))))))
time.sleep(3)
env=dict(os.environ,PYTHONPATH=f'{{p}}:{{p}}/tools/wmrequal:{{p}}/src:/root/autodl-tmp/c4/restart20261026/runtime_longint',
 WMREQUAL_COMPACT_PROBE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
 WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2',
 PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
with (p/'compact_probe.log').open('x') as log:
 child=subprocess.Popen([sys.executable,'-B',str(p/'tools/wmrequal/score_compact.py'),'--panel','validation'],env=env,cwd=p,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 r=dict(pid=child.pid,start_ticks=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')',1)[1].split()[19],time=time.time(),gpu_fraction=.85,compact=True)
 (p/'COMPACT_PROBE_LAUNCH.json').write_text(json.dumps(r,indent=1));print(json.dumps(r))
''')
