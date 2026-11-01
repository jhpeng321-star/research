"""Read only the owned warm-hold receipts and process identities."""
import json
import time
from . import ops
from .storage import write


def inspect_a():
    source = '''import pathlib,json,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_accel1');assert r.is_dir()
names=['A_WARM_CONTINUATION_DISPATCH.json','A_WARM2_DISPATCH.json','A_WARM3_DISPATCH.json',
 'A_WARM_CONTINUATION.log','A_WARM2.log','A_WARM3.log','A_WARM_FAILED.json','A_WARM_PAUSED.json']
out={n:(r/n).read_text() if (r/n).is_file() else None for n in names}
out['updates']={p.name:json.loads(p.read_text()) for p in r.glob('A_UPDATE_*.json')}
out['processes']={}
for n in names:
 if not n.endswith('DISPATCH.json') or not out[n]:continue
 d=json.loads(out[n]);p=pathlib.Path('/proc')/str(d['pid'])
 st=(p/'stat').read_text().split() if (p/'stat').exists() else None
 if st:assert st[21]==d['start_ticks'] and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
 out['processes'][n]=dict(pid=d['pid'],state=st[2] if st else 'exited')
out['unix']=time.time();out['new_sumo_sim_s']=0
print(json.dumps(out))
'''
    value = json.loads(ops.ssh.remote('cpuQ', source, timeout=25))
    write(ops.EVIDENCE/f'RESUME_A_{int(time.time())}.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    inspect_a()
