"""Pause only the identity-bound INT dispatch supervisor; leave its children running.

The frozen live supervisors have no hot slot/reorder API. A drain-only pause
enforces zero NEW admission, conservatively within the owner's 2/1 ceiling,
without restarting a dispatcher or interrupting any evaluate process.
"""
import argparse
import json
from pathlib import Path

from tools.rb2int.final_ops import REMOTE_ROOT
from tools.rbint.remote import ssh

EXPECTED = {
    'gpuA': dict(pid=473053, start_ticks='889857083',
                 boot_id='a81555d7-176e-4ed8-bc7d-896fc89cc914'),
    'gpuR': dict(pid=924719, start_ticks='4214495421',
                 boot_id='d19e6898-efd0-4288-a901-d63b21f11163'),
}
CONTROL = '/root/autodl-tmp/c4/rb2int20261101/phys_code_d0df3d66e80a'


def remote_code(host):
    expected = EXPECTED[host]
    return f'''
import ctypes,hashlib,json,os,platform,signal,subprocess,time
from datetime import datetime
from pathlib import Path
root=Path({REMOTE_ROOT!r}); expected={expected!r}; control=Path({CONTROL!r})
def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def incarnation(pid):
 fields=Path('/proc/'+str(pid)+'/stat').read_text().split()
 return dict(pid=pid,start_ticks=fields[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def state(pid): return Path('/proc/'+str(pid)+'/stat').read_text().split()[2]
def save(path,value):
 with path.open('x') as stream:
  json.dump(value,stream,indent=1);stream.flush();os.fsync(stream.fileno())
def pidfd_open(pid):
 if hasattr(os,'pidfd_open'): return os.pidfd_open(pid)
 assert platform.machine()=='x86_64', 'pidfd syscall fallback is x86_64 only'
 fd=ctypes.CDLL(None,use_errno=True).syscall(434,pid,0)
 if fd<0: raise OSError(ctypes.get_errno(),'own supervisor pidfd_open')
 return fd
def pidfd_stop(fd):
 if hasattr(signal,'pidfd_send_signal'): return signal.pidfd_send_signal(fd,signal.SIGSTOP)
 assert platform.machine()=='x86_64'
 result=ctypes.CDLL(None,use_errno=True).syscall(424,fd,signal.SIGSTOP,0,0)
 if result<0: raise OSError(ctypes.get_errno(),'own supervisor pidfd_send_signal')
process=load(root/'SUPERVISOR_PROCESS.json')
assert {{k:process[k] for k in expected}}==expected
assert process['host']=={host!r} and process['root']==str(root)
assert incarnation(expected['pid'])==expected
cmdline=Path('/proc/'+str(expected['pid'])+'/cmdline').read_bytes().split(b'\\0')
assert str(control/'tools/rb2int/final_worker.py').encode() in cmdline
assert str(root).encode() in cmdline
assert sha(control/'tools/rb2int/final_worker.py')==load(root/'LAUNCH_ONCE.json')['source_sha256']
assert not (root/'OWNER_1735_DRAIN_INTENT.json').exists(), 'preserve prior admission intervention'
assert not (root/'PANEL_COMPLETE.json').exists() and not (root/'PANEL_TERMINAL_INCOMPLETE.json').exists()
before=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
fd=pidfd_open(expected['pid'])
try:
 assert incarnation(expected['pid'])==expected and state(expected['pid']) not in ('T','t','Z')
 save(root/'OWNER_1735_DRAIN_INTENT.json',dict(owner='S-RB2-INT',process=expected,
  time=datetime.now().astimezone().isoformat(),signal='SIGSTOP via exact pidfd',
  target='supervisor only; no process group or evaluate child',new_admission_ceiling=0,
  requested_ceiling={2 if host == 'gpuA' else 1},reason='owner17:35 GPU priority; no hot slots API',gpu_before=before))
 pidfd_stop(fd)
 for _ in range(100):
  if state(expected['pid'])=='T': break
  time.sleep(.02)
 assert incarnation(expected['pid'])==expected and state(expected['pid'])=='T'
finally: os.close(fd)
children=[]
for path in sorted((root/'jobs').glob('*/PROCESS.json')):
 child=load(path)
 try:
  if incarnation(child['pid'])=={{k:child[k] for k in expected}}:
   children.append(dict(job=path.parent.name,process={{k:child[k] for k in expected}},state=state(child['pid']),signal_sent=False))
 except FileNotFoundError: pass
result=dict(host={host!r},time=datetime.now().astimezone().isoformat(),process=expected,
 supervisor_state=state(expected['pid']),new_admission_ceiling=0,requested_ceiling={2 if host == 'gpuA' else 1},
 mode='drain-only; exact2/1 live cap unavailable',evaluate_children=children,
 intent_sha256=sha(root/'OWNER_1735_DRAIN_INTENT.json'),gpu_before=before,
 source_recipe_plan_changed=False,redispatched_tasks=0,automatic_resume=False,
 next_action='explicit capped supervisor continuation required; do not SIGCONT at frozen4/3 slots')
save(root/'OWNER_1735_DRAIN_APPLIED.json',result)
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=tuple(EXPECTED), required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    out = Path(args.receipt)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('exclusive receipt in an existing directory required')
    result = json.loads(ssh(args.host, remote_code(args.host), timeout=90))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
