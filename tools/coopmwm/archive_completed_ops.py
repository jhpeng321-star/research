"""One-use, receipt-adopting launcher for MAIN14:50's completed tape archive."""
import argparse
import json
from . import ops
from .ls_validity_ops import remote
from .protocol import checked, sha
from .storage import write
from .archive_completed import CONTROL


def launch():
    p = checked(ops.LOCAL/'tools/coopmwm/archive_completed.py')
    text = p.read_text(encoding='utf-8'); h = sha(p)
    code = f'''from pathlib import Path
import json,subprocess,hashlib,os,time,fcntl
r=Path({CONTROL.as_posix()!r});assert r.is_absolute() and r.parent.is_dir();r.mkdir(exist_ok=True)
lock=(r/'PRIVILEGED_DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=r/'PRIVILEGED_DISPATCH.json'
if d.exists():print(d.read_text());raise SystemExit
p=r/'archive_completed.py';text={text!r}
if p.exists():assert p.read_text()==text
else:p.write_text(text)
assert hashlib.sha256(p.read_bytes()).hexdigest()=={h!r}
with (r/'ARCHIVE_PRIVILEGED.log').open('x') as log:
 child=subprocess.Popen(['sudo','-n','/home/ubuntu/miniconda3/bin/python','-B','-u',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=child.pid,start_ticks=Path('/proc',str(child.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source_sha256={h!r},control_root=str(r),authority='MAIN2026-11-03 14:50',sim_s=0)
with d.open('x') as f:json.dump(v,f)
print(json.dumps(v))'''
    value = json.loads(remote('cpuQ', code, timeout=35))
    target = ops.EVIDENCE/'ARCHIVE_MOVE_INTERIM008_DISPATCH_1450_V3.json'
    if not target.exists(): write(target, value)
    print(json.dumps(value))


def status():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,time
r=Path({CONTROL.as_posix()!r});d=json.loads((r/'PRIVILEGED_DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
alive=False
if p.exists():
 st=(p/'stat').read_text().split();alive=st[21]==d['start_ticks'] and st[2]!='Z'
receipt=r/'ARCHIVE_MOVE_interim008_observables.json'
print(json.dumps(dict(unix=time.time(),owner=d,alive=alive,receipt=json.loads(receipt.read_text()) if receipt.exists() else None,log=(r/'ARCHIVE_PRIVILEGED.log').read_text()[-6000:])))''', timeout=35))
    # Snapshot names are immutable; canonical result is copied only on completion.
    write(ops.EVIDENCE/f'ARCHIVE_MOVE_INTERIM008_STATUS_{__import__("time").time_ns()}.json', value)
    if value['receipt']:
        target = ops.EVIDENCE/'ARCHIVE_MOVE_interim008_observables.json'
        if not target.exists(): write(target, value['receipt'])
    print(json.dumps(value))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('launch', 'status'))
    a = p.parse_args(); globals()[a.mode]()
