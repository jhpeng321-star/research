"""Owned recovery for the owner19:34 acceleration-space restart."""
import argparse
import json
from . import ops, deploy
from .storage import write


def hold():
    value = json.loads(ops.ssh.remote('cpuQ', f'''import pathlib,json,os,signal,time
r=pathlib.Path({deploy.ROOT!r});assert r.is_dir()
d=json.loads((r/'A_DISPATCH_1854.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
receipt=r/'HOLD_ACCEL_1934.json'
if receipt.exists():
 v=json.loads(receipt.read_text())
 if (p/'stat').exists():assert (p/'stat').read_text().split()[2] in ('T','Z')
else:
 alive=(p/'stat').exists() and (p/'stat').read_text().split()[2]!='Z'
 if alive:
  assert (p/'stat').read_text().split()[21]==d['start_ticks']
  assert 'tools.coopmwm.volume_runner' in (p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  os.kill(d['pid'],signal.SIGSTOP)
 charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
 incomplete=[x['id'] for x in charges if not all((r/'jobs'/x['id']/f).exists() for f in ('RESULT.json','CLOSE.json'))]
 v=dict(owner=d,unix=time.time(),reason='OWNER19:34: all arms restart from MAPPO in acceleration space; superseded runs and charges retained',
  arms_held=['A','B','R'],supervisor_was_alive=alive,incomplete_jobs=incomplete,
  charged_sim_s=sum(x['sim_s'] for x in charges),jobs=len(charges),new_sumo_sim_s=0,
  existing_updates=sorted(x.name for x in r.glob('S*_UPDATE_*.json')))
 with receipt.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
''', timeout=25))
    write(ops.EVIDENCE/'HOLD_ACCEL_1934.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('hold',))
    parser.parse_args()
    hold()
