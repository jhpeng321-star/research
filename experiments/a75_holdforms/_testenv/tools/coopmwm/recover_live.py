"""Receipt-only recovery; no simulator dispatch and no process control."""
import json
import time
from . import ops
from .storage import write


def snapshot():
    source = '''import pathlib,json,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_accel1')
d=json.loads((r/'A_WARM_CONTINUATION_DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or s[21]==d['start_ticks']
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'exited',
 log=(r/'A_WARM_CONTINUATION.log').read_text(),stop=json.loads((r/'STOP.json').read_text()),
 continuation_claim=(r/'A_WARM_CONTINUATION.claim').exists(),
 charged_sim_s=811640+sum(c['sim_s'] for c in charges),
 updates=sorted(x.name for x in r.glob('A_UPDATE_*.json')),new_sumo_sim_s=0)))
'''
    v = json.loads(ops.ssh.remote('cpuQ', source, timeout=25))
    write(ops.EVIDENCE/f'RECOVERY_LIVE_{int(v["unix"])}.json', v)
    print(json.dumps(v))


if __name__ == '__main__':
    snapshot()
