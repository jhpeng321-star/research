"""Stage additive fast adapter, validate serial/grouped parity, retire owned slow child."""
import base64
import json
from pathlib import Path
from remote import remote

root=Path(__file__).resolve().parents[2]
payload={name:base64.b64encode((root/'tools/wmrequal'/name).read_bytes()).decode()
         for name in ('grouped_feedback.py','score_fast.py')}
encoded=base64.b64encode(json.dumps(payload).encode()).decode()
remote(f'''import base64,json,pathlib,subprocess,os,sys,time,signal
import numpy as np
p=pathlib.Path('/root/autodl-tmp/c4/wmrequal20261030')
for name,raw in json.loads(base64.b64decode({encoded!r})).items():
 q=p/'tools/wmrequal'/name
 assert not q.exists(),'preserve staged adapter'
 q.write_bytes(base64.b64decode(raw))
# Only this task's recorded child; exact incarnation and argv must both match.
launch=json.loads((p/'LAUNCH.json').read_text())
children=[json.loads(s) for s in (p/'OWNED_CHILDREN.jsonl').read_text().splitlines()]
stopped=[]
for r in children:
 q=pathlib.Path(f'/proc/{{r["pid"]}}/stat')
 if q.is_file():
  assert q.read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']
  cmd=pathlib.Path(f'/proc/{{r["pid"]}}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
  assert str(p/'tools/wmrequal/score.py') in cmd
  os.kill(r['pid'],signal.SIGTERM);stopped.append(r)
(p/'OWNED_SLOW_STOP.json').write_text(json.dumps(dict(time=time.time(),reason='replace per-row feedback dispatch after numerical parity check',children=stopped)))
time.sleep(3)
env=dict(os.environ,PYTHONPATH=f'{{p}}:{{p}}/tools/wmrequal:{{p}}/src:/root/autodl-tmp/c4/restart20261026/runtime_longint',
 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WMREQUAL_PARITY='1',
 WMS_SPEED_PIVOT_DIAG_OFF='1',WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
 WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2')
for serial in (True,False):
 e={{**env,'WMREQUAL_SERIAL_PARITY':'1' if serial else '0'}}
 r=subprocess.run([sys.executable,'-B',str(p/'tools/wmrequal/score_fast.py'),'--panel','validation','--smoke'],env=e,capture_output=True,text=True,timeout=180)
 (p/('parity_serial.log' if serial else 'parity_grouped.log')).write_text(r.stdout+r.stderr)
 assert r.returncode==0,r.stderr
a=json.loads((p/'parity_serial/smoke/batch_000.json').read_text())['rows']
b=json.loads((p/'parity_grouped/smoke/batch_000.json').read_text())['rows']
assert len(a)==len(b)==8
maximum=0.
for x,y in zip(a,b):
 assert (x['pair_key'],x['draw'],x['mix'],x['arm'])==(y['pair_key'],y['draw'],y['mix'],y['arm'])
 assert x['mix_assigned']==y['mix_assigned']
 for h in x['values']:
  for k in x['values'][h]:
   delta=abs(x['values'][h][k]-y['values'][h][k]);maximum=max(maximum,delta)
   assert delta<.01,(k,delta)
doc=dict(status='PASS',rows=len(a),horizon=10,max_abs_difference=maximum,no_optimizer=True)
(p/'FEEDBACK_PARITY.json').write_text(json.dumps(doc,indent=1));print(json.dumps(doc))
''',timeout=480)
