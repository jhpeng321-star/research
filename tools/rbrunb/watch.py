"""Ten-minute INT/EVAL polling and atomic RUNB ownership; existing Python only."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SHARED = Path('E:/agent-work/rb-shared/evalclaims')
HANDOFFS = [Path(f'E:/agent-work/c4-rb-{x}/HANDOFF.md') for x in ('int', 'eval')]
EVAL = Path('E:/agent-work/c4-rb-eval/tools/rbeval')
REMOTE = '/root/rbeval20261031/runb'
OWNER = 'S-RB-RUNB'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def ssh(code):
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'gpuR', '/root/miniconda3/bin/python -B -'],
                       input=code, text=True, capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stdout+'\n'+p.stderr)
    return p.stdout


def ready_sections(text):
    sections = []
    ready_level = None
    for line in text.splitlines():
        heading = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading:
            level, title = len(heading[1]), heading[2]
            if re.match(r'^READY FOR EVAL\b', title, re.I):
                ready_level = level
                sections.append('')
                continue
            if ready_level is not None and level <= ready_level:
                ready_level = None
        if ready_level is not None:
            sections[-1] += line+'\n'
    return '\n'.join(sections)


def manifests(texts):
    section = '\n'.join(ready_sections(t) for t in texts)
    if not section.strip():
        return [], 'No READY FOR EVAL section'
    paths = [SHARED/'READY.json'] if 'READY.json' in section else []
    paths += [Path(x) for x in re.findall(r'[A-Za-z]:[/\\][^\s`"<>\[\]()]+\.json', section)]
    records = []
    seen = set()
    for path in paths:
        # Coordination manifests only; never follow data-file links.
        normal = str(path).replace('\\', '/').lower()
        if not normal.startswith((str(SHARED).replace('\\', '/').lower()+'/',
                                  'e:/agent-work/c4-rb-int/', 'e:/agent-work/c4-rb-eval/')):
            continue
        if any(x in normal for x in ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b',
                                    'inc1', 'sealed', 'truth', 'assignment', 'vehicle_params', 'driver_params')):
            continue
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict):
            continue
        variants = payload.get('variants', [payload] if 'variant' in payload else [])
        for record in variants:
            name = record.get('variant', '')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name):
                raise ValueError('Invalid variant name')
            if name not in seen:
                records.append(record)
                seen.add(name)
    return records, 'Ordered ready manifests found' if records else 'READY needs published ordered manifest'


def eligible(records):
    result = []
    for i, r in enumerate(records):
        name = r['variant']
        claim, done = SHARED/f'{name}.claim', SHARED/f'{name}.done'
        if done.exists() or (SHARED/f'{name}.failed.json').exists():
            continue
        if claim.exists():
            # Resume only claims actually owned by this runner.
            try:
                value = json.loads(claim.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            if value.get('owner') == OWNER:
                result.insert(0, r)
            continue
        # RUNA gets the first variant even if RUNB polls first.
        if i == 0:
            continue
        if r.get('host', 'gpuR') == 'gpuR':
            result.append(r)
    return result


def validate(record):
    from supervise import path_ok
    assert record.get('host') == 'gpuR', 'INT must stage the candidate on gpuR'
    assert record.get('model_name') not in (None, '', 'wm2')
    for key in ('runtime_root', 'wm', 'wm_config', 'smoke_receipt'):
        value = record[key]
        assert isinstance(value, str) and value.startswith('/root/')
        path_ok(value)
    assert re.fullmatch(r'[a-fA-F0-9]{64}', record['smoke_sha256'])
    if record.get('entrypoint'):
        path_ok(record['entrypoint'])
    else:
        assert re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', record['loader_module'])
        assert re.fullmatch(r'[A-Za-z_]\w*', record.get('loader_function', 'load_candidate'))
    if record.get('switches'):
        path_ok(record['switches'])


def stage(state):
    files = [HERE/'supervise.py', *[EVAL/n for n in
             ('run_wm.py', 'score.py', 'core.py', 'references.json', 'real_observables.json', 'PREPARED.json')]]
    hashes = {f.name: sha(f) for f in files}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    code = f'{REMOTE}/code_{version}'
    receipt = state/f'STAGED_{version}.json'
    if receipt.exists():
        return code
    ssh(f"""
from pathlib import Path
import hashlib
root=Path({REMOTE!r}); assert root.parent.is_dir(); root.mkdir(exist_ok=True)
code=Path({code!r}); code.mkdir(exist_ok=True)
for name,sha in {hashes!r}.items():
 p=code/name
 if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'Preserve existing code'
print('RUNB staging admitted')
""")
    subprocess.run(['scp', '-o', 'BatchMode=yes', *map(str, files), f'gpuR:{code}/'],
                   check=True, capture_output=True, text=True, timeout=180)
    ssh(f"""
from pathlib import Path
import ast,hashlib
code=Path({code!r})
for name,sha in {hashes!r}.items():
 p=code/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'Stage changed during transfer'
 if p.suffix=='.py': ast.parse(p.read_text())
print('RUNB sources verified without inference')
""")
    save_new(receipt, dict(code=code, hashes=hashes, unix=time.time()))
    return code


def launch_or_status(record, state, deadline):
    validate(record)
    name = record['variant']
    claim = SHARED/f'{name}.claim'
    if not claim.exists():
        try:
            save_new(claim, dict(owner=OWNER, host='gpuR', variant=name,
                                claimed_at=datetime.now().astimezone().isoformat(), manifest=record))
        except FileExistsError:
            return dict(state='CLAIM_LOST', variant=name)
    claimed = json.loads(claim.read_text(encoding='utf-8'))
    assert claimed['owner'] == OWNER
    assert claimed['manifest'] == record, 'Claimed candidate changed; preserve original'
    code = stage(state)
    root = f'{REMOTE}/{name}'
    # The remote boot/start-tick check plus flock permits recovery of a lost SSH
    # response without a second inference launch. Any old DISPATCH is preserved.
    response = ssh(f"""
from pathlib import Path
import fcntl,json,os,subprocess,time
root=Path({root!r}); assert root.parent.is_dir(); root.mkdir(exist_ok=True)
lock=(root/'LAUNCH.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX)
for name in ('FINALIZED.json','FAILED.json','DEADLINE.json'):
 if (root/name).exists():
  print(json.dumps(dict(state=name[:-5],receipt=json.loads((root/name).read_text())))); raise SystemExit
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for f in root.glob('SUPERVISOR_*.json'):
 old=json.loads(f.read_text()); proc=Path('/proc')/str(old['pid'])
 try: alive=old['boot_id']==boot and (proc/'stat').read_text().split()[21]==old['start_ticks']
 except (FileNotFoundError,ProcessLookupError): alive=False
 if alive:
  print(json.dumps(dict(state='LIVE',root=str(root),tasks=sum(len(list(d.glob('w*_k*.json'))) for d in root.glob('shard*') if d.is_dir())))); raise SystemExit
if (root/'DISPATCH.json').exists() or (root/'LAUNCH.json').exists():
 print(json.dumps(dict(state='RECONCILE_REQUIRED',root=str(root)))); raise SystemExit
if time.time()>={deadline!r}:
 print(json.dumps(dict(state='ADMISSION_CLOSED',root=str(root)))); raise SystemExit
m=root/'MANIFEST.json'
with m.open('x') as f: json.dump({record!r},f,indent=2)
argv=['nice','-n','19','/root/miniconda3/bin/python','-B','-u',{code!r}+'/supervise.py','--manifest',str(m),'--root',str(root),'--code',{code!r},'--deadline',str({deadline!r})]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'supervisor.log').open('x') as log:
 child=subprocess.Popen(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
with (root/'LAUNCH.json').open('x') as f: json.dump(dict(pid=child.pid,argv=argv,unix=time.time()),f)
print(json.dumps(dict(state='STARTED',pid=child.pid,root=str(root))))
""")
    result = json.loads(response)
    result['variant'] = name
    if result['state'] in ('FAILED', 'DEADLINE', 'RECONCILE_REQUIRED'):
        failed = SHARED/f'{name}.failed.json'
        if not failed.exists():
            save_new(failed, dict(owner=OWNER, variant=name, remote_output=root,
                                 result=result, unix=time.time(), automatic_retry=False))
    if result['state'] == 'FINALIZED':
        r = result['receipt']
        assert r['owner'] == OWNER and r['tasks'] == 160 and r['worlds'] == 20
        target = state/name
        target.mkdir(exist_ok=True)
        subprocess.run(['scp', '-r', '-o', 'BatchMode=yes', f'gpuR:{root}/reports',
                        f'gpuR:{root}/FINALIZED.json', str(target)],
                       check=True, capture_output=True, text=True, timeout=600)
        fetched = json.loads((target/'FINALIZED.json').read_text(encoding='utf-8'))
        assert fetched == r
        report = target/'reports'/'complete'
        assert all(sha(report/n) == h for n, h in r['report_hashes'].items())
        gate = r['gate']
        note = dict(owner=OWNER, variant=name, remote_output=root, remote_report=r['report'],
                    local_report=str(report), tasks=160, worlds=20,
                    gate={k: gate[k] for k in ('verdict', 'pilot_only') if k in gate},
                    key_numbers=r['key_numbers'], finalized_receipt=str(target/'FINALIZED.json'))
        done = SHARED/f'{name}.done'
        if not done.exists():
            save_new(done, note)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once', action='store_true')
    p.add_argument('--stage-only', action='store_true', help='Stage and verify code, without claiming or inference')
    p.add_argument('--interval', type=int, default=600)
    p.add_argument('--deadline', help='Explicit timezone-aware owner supervision deadline')
    p.add_argument('--freeze-at', help='Close intake of new claims at this owner freeze time')
    a = p.parse_args()
    assert a.interval >= 30
    state = WORK/'runs'/'rbrunb'
    assert WORK.is_dir() and SHARED.is_dir()
    state.mkdir(parents=True, exist_ok=True)
    if a.stage_only:
        print(stage(state))
        return
    # Windows byte-range lock is released on exit; stale files are preserved.
    import msvcrt
    lock = (state/'WATCH.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    now = datetime.now().astimezone()
    end = datetime.fromisoformat(a.deadline) if a.deadline else now.replace(hour=7, minute=0, second=0, microsecond=0)
    assert end.tzinfo is not None, 'Deadline must include timezone'
    if not a.deadline and now >= end:
        end += timedelta(days=1)
    deadline = end.timestamp()
    freeze = datetime.fromisoformat(a.freeze_at) if a.freeze_at else end
    assert freeze.tzinfo is not None and now < freeze <= end
    save_new(state/f'WATCH_{os.getpid()}_{time.time_ns()}.json',
             dict(owner=OWNER, pid=os.getpid(), interval=a.interval, deadline=end.isoformat(),
                  freeze_at=freeze.isoformat(),
                  started=now.isoformat(), script_sha256=sha(__file__)))
    while time.time() < deadline+1800:
        tick = time.monotonic()
        event = dict(time=datetime.now().astimezone().isoformat(), owner=OWNER, handoffs=[])
        try:
            texts = []
            for path in HANDOFFS:
                text = path.read_text(encoding='utf-8-sig') if path.exists() else ''
                texts.append(text)
                h = hashlib.sha256(text.encode()).hexdigest()
                event['handoffs'].append(dict(path=str(path), sha256=h, present=bool(text)))
                snapshot = state/f'{path.parent.name}_{h}.md'
                if not snapshot.exists():
                    with snapshot.open('x', encoding='utf-8') as f:
                        f.write(text)
            records, reason = manifests(texts)
            event.update(state=reason, ready=[r['variant'] for r in records])
            # Continue supervising our immutable claims even if INT moves a ready
            # entry out of the current handoff after dispatch.
            published = {r['variant'] for r in records}
            for claim in sorted(SHARED.glob('*.claim')):
                try:
                    old = json.loads(claim.read_text(encoding='utf-8'))
                except (ValueError, OSError):
                    continue
                if old.get('owner') == OWNER and old['variant'] not in published:
                    records.append(old['manifest'])
            candidates = eligible(records)
            if time.time() >= freeze.timestamp():
                candidates = [r for r in candidates if (SHARED/(r['variant']+'.claim')).exists()]
            if candidates:
                event['evaluation'] = launch_or_status(candidates[0], state, deadline)
        except Exception as e:
            event['error'] = repr(e)
        save_new(state/f'POLL_{time.time_ns()}.json', event)
        print(json.dumps(event), flush=True)
        # Status is operational, not a scientific completion receipt.
        (state/'STATUS.json').write_text(json.dumps(event, indent=2), encoding='utf-8')
        if a.once:
            return
        next_deadline = deadline if time.time() < deadline else deadline+1800
        interval = a.interval if time.time() < deadline else 60
        time.sleep(max(0, min(interval-(time.monotonic()-tick), next_deadline-time.time())))
    save_new(state/f'DEADLINE_{time.time_ns()}.json', dict(owner=OWNER, unix=time.time()))


if __name__ == '__main__':
    main()
