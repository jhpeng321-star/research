"""Retire only INT's two superseded GPU panels after all reserved K8 results.

Operational throughput priority, never an efficacy-selected model stop. Retain
every completed forecast and original source/claim. No inference is launched.
"""
import argparse
import inspect
from pathlib import Path

from tools.rbint.remote import ssh

SOURCES = {'WM-PHYS-FULL-v1': '4015b6412cdfa2a94168b7095a59287d0fddcd3b',
           'WM-PHYS-FULL-v3-gpu': 'f378ac6033079bdd87c0059627e27e394a1d7cbe'}


def validate_owned(spec, workers, root, runtime, source):
    assert spec['owner'] == 'S-RB-INT' and spec['variant'] == 'FULL'
    assert spec['runtime'] == runtime and spec['stage']['commit'] == source
    assert len(workers) == 3
    outputs = []
    for worker in workers:
        argv = worker['argv']
        assert runtime+'/tools/rbint/evaluate_FULL.py' in argv
        out = argv[argv.index('--out')+1]
        assert out in [root+f'/shard{i}' for i in range(3)]
        outputs.append(out)
        assert worker['pid'] > 1 and worker['boot_id'] and worker['start_ticks']
    assert len(set(outputs)) == 3 and len({w['pid'] for w in workers}) == 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=tuple(SOURCES), required=True)
    args = parser.parse_args()
    source = SOURCES[args.name]
    root = '/root/rbeval20261031/int/'+args.name
    runtime = '/root/rbeval20261031/phys_code_'+source[:12]
    code = '''
import ctypes, hashlib, json, os, platform, signal, time
from pathlib import Path
'''+inspect.getsource(validate_owned)+f'''
root=Path({root!r});runtime={runtime!r};source={source!r}
assert root.is_dir()
spec=json.loads((root/'SPEC.json').read_text())
workers=json.loads((root/'PANEL_PROCESS.json').read_text())['workers']
validate_owned(spec,workers,str(root),runtime,source)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):
 with p.open('x') as stream: json.dump(data,stream,indent=1)
def alive(r):
 try:
  f=Path('/proc/'+str(r['pid'])+'/stat').read_text().split()
  return f[2]!='Z' and f[21]==r['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['boot_id']
 except (FileNotFoundError,ProcessLookupError): return False
verified=[]
for seed in range(969432,969440):
 shard=root/f'shard{{(seed-969420)%3}}'
 identity=sha(shard/'IDENTITY.json')
 for k in range(8):
  path=shard/f'w{{seed}}_k{{k}}.json';task=json.loads(path.read_text())
  assert task['seed']==seed and task['future']==k and task['identity_sha256']==identity
  assert len(task['episodes'])==5
  verified.append(dict(path=str(path),sha256=sha(path)))
request=dict(owner='S-RB-INT',unix=time.time(),source=source,
 reason='superseded engineering source; reserved8 complete; prioritize latest FULL-v4 throughput',
 selection='source lineage and completion only; no efficacy-selected parameters',
 workers=workers,reserved_K8=verified,automatic_retry=False,completed_forecasts_retained=True)
save(root/'SUPERSEDED_STOP_REQUEST.json',request)
for i in range(3):
 marker=root/f'shard{{i}}/STOP'
 if not marker.exists(): marker.touch(exist_ok=False)
end=time.monotonic()+20
while any(alive(r) for r in workers) and time.monotonic()<end: time.sleep(.5)
signalled=[]
assert platform.machine()=='x86_64'
libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
for r in workers:
 if not alive(r): continue
 fd=libc.syscall(434,r['pid'],0)  # pidfd_open on x86_64
 if fd<0: raise OSError(ctypes.get_errno(),'own pidfd_open')
 try:
  if alive(r):
   result=libc.syscall(424,fd,signal.SIGTERM,0,0)  # pidfd_send_signal
   if result<0: raise OSError(ctypes.get_errno(),'own pidfd_send_signal')
   signalled.append(r['pid'])
 finally: os.close(fd)
end=time.monotonic()+5
while any(alive(r) for r in workers) and time.monotonic()<end: time.sleep(.25)
result=dict(owner='S-RB-INT',unix=time.time(),root=str(root),source=source,signalled_exact_own=signalled,
 alive=[r['pid'] for r in workers if alive(r)],tasks=[len(list((root/f'shard{{i}}').glob('w*_k*.json'))) for i in range(3)],
 reserved_worlds=8,completed_forecasts_retained=True,scientific_verdict=None)
save(root/'SUPERSEDED_STOPPED.json',result)
print(json.dumps(result))
'''
    print(ssh('gpuA', code, timeout=60))


if __name__ == '__main__':
    main()
