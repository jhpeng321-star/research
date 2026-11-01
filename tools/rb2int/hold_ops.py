"""Owner17:55: suspend exact INT admission owners and standbys, not evaluations."""
import argparse
import json
from pathlib import Path

from tools.rb2int.drain_ops import EXPECTED, remote_code
from tools.rbint.remote import ssh


PROCESSES = {
    'gpuA': (513448, '890664618', 514854, '890706569'),
    'gpuR': (948148, '4215302457', 949247, '4215344299'),
}


def hold_code(host):
    # Reuse only the reviewed identity/pidfd helpers, never the old intervention.
    helpers = remote_code(host).split("process=load(root/'SUPERVISOR_PROCESS.json')")[0]
    current, current_start, late, late_start = PROCESSES[host]
    targets = [
        ('owner_late_control', late, late_start),
        ('owner1735_control', current, current_start),
    ]
    return helpers + f'''
assert not (root/'OWNER_1755_HOLD_INTENT.json').exists(), 'preserve prior hold'
before=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
bound=[]
for name,pid,start in {targets!r}:
 process=load(root/name/'PROCESS.json')
 identity=dict(pid=pid,start_ticks=start,boot_id={EXPECTED[host]['boot_id']!r})
 assert {{k:process[k] for k in identity}}==identity
 assert process['host']=={host!r}
 assert incarnation(pid)==identity
 cmdline=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\\0')
 assert str(root/name/'capped_worker.py').encode() in cmdline
 assert str(root).encode() in cmdline
 assert sha(root/name/'capped_worker.py')==process['source_sha256']
 fd=pidfd_open(pid)
 bound.append((name,identity,fd))
save(root/'OWNER_1755_HOLD_INTENT.json',dict(time=datetime.now().astimezone().isoformat(),
 owner='S-RB2-INT',processes=[identity for _,identity,_ in bound],new_gpu_admissions=0,
 resume_authority="MAIN appends S-RB2-INT RESUME",gpu_before=before))
try:
 for name,identity,fd in bound:
  assert incarnation(identity['pid'])==identity
  if state(identity['pid']) not in ('T','t'):
   assert state(identity['pid'])!='Z'
   pidfd_stop(fd)
  for _ in range(100):
   if state(identity['pid'])=='T': break
   time.sleep(.02)
  assert incarnation(identity['pid'])==identity and state(identity['pid'])=='T'
finally:
 for _,_,fd in bound: os.close(fd)
children=[]
for path in sorted((root/'jobs').glob('*/PROCESS.json')):
 child=load(path)
 try:
  identity={{k:child[k] for k in ('pid','start_ticks','boot_id')}}
  if incarnation(child['pid'])==identity and state(child['pid'])!='Z':
   children.append(dict(job=path.parent.name,process=identity,state=state(child['pid']),signal_sent=False))
 except FileNotFoundError: pass
result=dict(host={host!r},time=datetime.now().astimezone().isoformat(),
 paused=[dict(stage=name,process=identity,state=state(identity['pid'])) for name,identity,_ in bound],
 new_gpu_admissions=0,automatic_resume=False,evaluate_children=children,
 evaluate_signals=0,peer_signals=0,redispatched_tasks=0,source_recipe_plan_changed=False,
 gpu_before=before,resume_authority="MAIN appends S-RB2-INT RESUME")
save(root/'OWNER_1755_HOLD_APPLIED.json',result)
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=tuple(PROCESSES), required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    out = Path(args.receipt)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('exclusive receipt in existing parent required')
    result = json.loads(ssh(args.host, hold_code(args.host), timeout=90))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
