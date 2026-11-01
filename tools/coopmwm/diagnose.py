"""Owner 18:34 action-response investigation; no physical dispatch."""
import argparse
import json
from . import ops, deploy
from .storage import write


def hold():
    value=json.loads(ops.ssh.remote('cpuQ',f'''import os,signal,pathlib,json,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'TRAIN_DISPATCH_1742.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert (p/'stat').read_text().split()[21]==d['start_ticks']
assert 'tools.coopmwm.runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
receipt=r/'HOLD_1834.json'
if receipt.exists():
 v=json.loads(receipt.read_text());assert (p/'stat').read_text().split()[2]=='T'
else:
 os.kill(d['pid'],signal.SIGSTOP)
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
 assert not incomplete,'supervisor held; inspect outstanding owned jobs before proceeding'
 assert not (r/'SAMPLING_1755.json').exists(),'inspect any post-u1 outcomes'
 v=dict(owner=d,unix=time.time(),reason='OWNER18:34 WM action-response wiring investigation',
  arms_held=['B','R'],shared_supervisor_paused=True,A_update=1,B_update=1,
  charged_sim_s=sum(x['sim_s'] for x in charges),closed_jobs=len(charges),new_sumo_sim_s=0)
 with receipt.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''',timeout=25))
    write(ops.EVIDENCE/'HOLD_1834.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['hold']);a=p.parse_args()
    hold()
