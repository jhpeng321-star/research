"""Read-only comparison of TRAIN lesson and original bank grading windows."""
import json
import time
from . import ops
from .ls_collect_ops import ROOT, COLLECTION
from .ls_validity_ops import remote
from .storage import write


def run():
    value=json.loads(remote('cpuQ',f'''from pathlib import Path
import sys,json,time
sys.path.insert(0,'/home/ubuntu/coopmwm20261103_replaystartcpu5/pytools')
from tools.coopmwm.replay_bank import bank_decisions
r=Path({ROOT!r})
rows={{x['verification_id']:x for x in bank_decisions(r,'B',sources=[{COLLECTION!r}])}}
lesson=json.loads((r/'LESSON_CHECK_u004_B.json').read_text())
out=[]
for e in lesson['entries']:
 original=rows[e['verification_id']]['window'];new=e['new_policy_branch']['window']
 out.append(dict(id=e['verification_id'],original=original,new=new,
  same_window=all(original[k]==new[k] for k in ('start_s','end_s'))))
print(json.dumps(dict(unix=time.time(),entries=out,new_sim_s=0,read_only=True)))''',timeout=40))
    path=ops.EVIDENCE/f'LESSON_WINDOW_PROBE_{time.time_ns()}.json'
    write(path,value)
    print(json.dumps(dict(receipt=str(path),**value)))


if __name__=='__main__':run()
