"""Ten-minute local coordinator. Durable claims, immutable staging, no retries."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
SHARED = Path('E:/agent-work/rb-shared/evalclaims')
EVAL = Path('E:/agent-work/c4-rb-eval/tools/rbeval')
HANDOFFS = [Path(f'E:/agent-work/c4-rb-{name}/HANDOFF.md') for name in ('int', 'eval')]
STATE = WORK/'runs/rbruna'
DEADLINE = datetime.fromisoformat('2026-11-01T11:30:00+08:00').timestamp()
FREEZE = datetime.fromisoformat('2026-11-01T09:30:00+08:00').timestamp()
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ControlMaster=no', '-o', 'ControlPath=none', 'gpuA']
FILES = ('run_wm.py', 'core.py', 'score.py', 'references.json', 'real_observables.json')


def save_new(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def ssh(code):
    result = subprocess.run(SSH + ['/root/miniconda3/bin/python -B -'], input=code,
                            text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout.strip()


def discover(text):
    """Only JSON links explicitly inside a READY FOR EVAL heading section."""
    ready = False
    level = 0
    found = []
    for line in text.splitlines():
        heading = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading:
            n, title = len(heading[1]), heading[2]
            if 'READY FOR EVAL' in title.upper():
                ready, level = True, n
            elif n <= level:
                ready = False
        elif line.strip().upper().rstrip(':') == 'READY FOR EVAL':
            ready, level = True, 6
        if ready:
            found += re.findall(r'E:[/\\]agent-work[/\\]rb-shared[/\\]evalclaims[/\\][\w.-]+\.runa\.json', line)
    return list(dict.fromkeys(found))


def validate(spec, path):
    variant = spec['variant']
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', variant)
    assert variant.lower() != 'wm2' and spec['host'] == 'gpuA'
    assert path.name == variant+'.runa.json'
    for key in ('runtime_root', 'wm', 'wm_config', 'loader_module', 'loader_function',
                'identity_sources', 'sources', 'smoke_receipt', 'smoke_sha256'):
        assert spec[key], key
    assert re.fullmatch(r'[\w.]+', spec['loader_module'])
    assert re.fullmatch(r'\w+', spec['loader_function'])


def claim(variant, spec):
    try:
        save_new(SHARED/(variant+'.claim'), dict(owner='S-RB-RUNA', host='gpuA',
            claimed=datetime.now().astimezone().isoformat(), spec=spec))
        return True
    except FileExistsError:
        return False


def read_claim(path):
    text = path.read_text(encoding='utf-8')
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else dict(owner=str(value))
    except json.JSONDecodeError:
        return dict(owner=text.strip())


def dispatch(spec):
    variant = spec['variant']
    root = '/root/rbeval20261031/runa/'+variant
    # Check every remote artifact and resource before claiming; no scientific execution.
    source = (HERE/'remote.py').read_text(encoding='utf-8')
    audit = ssh(source.split("if __name__ == '__main__':")[0] +
                '\nprint(json.dumps(preflight('+repr(spec)+')))\n')
    if not claim(variant, spec):
        return
    local = STATE/variant
    local.mkdir()
    save_new(local/'spec.json', spec)
    save_new(local/'PREFLIGHT.json', json.loads(audit))
    # NEW run root only. Remote parent existence checked before creating children.
    ssh(f"from pathlib import Path\np=Path({root!r})\nassert p.parent.parent.parent.is_dir()\n"
        "p.parent.parent.mkdir(exist_ok=True)\np.parent.mkdir(exist_ok=True)\np.mkdir()\n(p/'code').mkdir()\n")
    wrapper = ('import json\nfrom pathlib import Path\nfrom remote import candidate\n'
               "candidate(json.loads((Path(__file__).resolve().parents[1]/'spec.json').read_text()))\n")
    # Generated files stay inside own run receipts; source files are never overwritten.
    save_new(local/'STAGED.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [HERE/'remote.py', *[EVAL/n for n in FILES]]})
    ssh(f"from pathlib import Path\np=Path({root!r})\n"
        f"with (p/'code/candidate.py').open('x') as f: f.write({wrapper!r})\n"
        f"with (p/'spec.json').open('x') as f: f.write({json.dumps(spec)!r})\n")
    argv = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
            str(HERE/'remote.py'), *[str(EVAL/n) for n in FILES], f'gpuA:{root}/code/']
    subprocess.run(argv, check=True, capture_output=True, text=True, timeout=180)
    code = (f"import hashlib,json,subprocess\nfrom pathlib import Path\np=Path({root!r})\n"
            f"expected={json.loads((local/'STAGED.json').read_text())!r}\n"
            "for old,sha in expected.items():\n"
            " f=p/'code'/Path(old.replace('\\\\','/')).name\n"
            " assert hashlib.sha256(f.read_bytes()).hexdigest()==sha, str(f)\n"
            f"subprocess.run([{spec.get('python', '/root/miniconda3/bin/python')!r},'-B',str(p/'code/remote.py'),'launch','--root',str(p)],check=True)\n")
    receipt = ssh(code)
    save_new(local/'DISPATCH.json', dict(remote=root, receipt=receipt))
    print('DISPATCHED', variant, root, flush=True)


def collect():
    for path in SHARED.glob('*.claim'):
        own = read_claim(path)
        if own.get('owner') != 'S-RB-RUNA':
            continue
        variant = path.stem
        if (SHARED/(variant+'.done')).exists() or (SHARED/(variant+'.failed.json')).exists():
            continue
        root = '/root/rbeval20261031/runa/'+variant
        text = ssh(f"import json\nfrom pathlib import Path\np=Path({root!r})\n"
            "for n in ('RESULT.json','FAILED.json'):\n"
            " f=p/n\n"
            " if f.exists(): print(json.dumps(dict(marker=n,data=json.loads(f.read_text()))));break\n"
            "else:\n"
            " if not (p/'LAUNCH.lock').exists(): print(json.dumps(dict(marker='FAILED.json',data=dict(error='staging interrupted before launch; no inference started'))))\n")
        if not text:
            continue
        result = json.loads(text)
        if result['marker'] == 'FAILED.json':
            save_new(SHARED/(variant+'.failed.json'), result)
            print('FAILED', variant, result['data']['error'], flush=True)
        else:
            metrics = ssh(f"import csv,json\nfrom pathlib import Path\np=Path({root!r})/'reports/complete'\n"
                "with (p/'distribution_summary.csv').open() as f:\n"
                " rows=[r for r in csv.DictReader(f) if r['level']=='arm' and r['arm']=='native' and r['metric']=='team_delay_vs']\n"
                "keys=('split','horizon_s','worlds','wm_crps','cell_crps','error','wm_minus_cell_low','wm_minus_cell_high')\n"
                "print('Native team delay: '+json.dumps([{k:r.get(k) for k in keys} for r in rows]))\n"
                "with (p/'rates.csv').open() as f:\n"
                " rows=[r for r in csv.DictReader(f) if r['arm']=='native' and r['region']=='upstream']\n"
                "for split in ('development12','reserved8'):\n"
                " for h in ('120','300','600'):\n"
                "  rs=[r for r in rows if r['split']==split and r['horizon_s']==h and r['wm_rate'] and r['real_rate']]\n"
                "  n=sum(int(r['worlds']) for r in rs)\n"
                "  if n: print(json.dumps(dict(split=split,horizon=h,n=n,wm_rate=sum(float(r['wm_rate'])*int(r['worlds']) for r in rs)/n,real_rate=sum(float(r['real_rate'])*int(r['worlds']) for r in rs)/n)))\n")
            local = STATE/variant
            if (local/'RESULT.json').exists():
                assert json.loads((local/'RESULT.json').read_text()) == result['data']
            else:
                save_new(local/'RESULT.json', result['data'])
            with (SHARED/(variant+'.done')).open('x', encoding='utf-8') as f:
                f.write(f"S-RB-RUNA complete: gpuA:{root}\n160tasks/20worlds/K8/fivearms\n"
                        f"Reports: {root}/reports/complete; RESULT.json holds hashes and gate.\n"
                        f"Peak own GPU MiB: {result['data']['peak_own_gpu_mib']}\n"
                        f"Gate: {json.dumps(result['data']['gate'])}\n"+metrics)
            print('COMPLETE', variant, flush=True)


def tick():
    texts = {}
    for path in HANDOFFS:
        texts[str(path)] = path.read_text(encoding='utf-8') if path.exists() else ''
    stamp = datetime.now().astimezone().isoformat()
    save_new(STATE/f'POLL_{time.time_ns()}.json', dict(time=stamp, handoffs=texts))
    collect()
    if time.time() >= FREEZE:
        return
    candidates = list(dict.fromkeys(p for text in texts.values() for p in discover(text)))
    print(stamp, 'READY specs:', candidates, flush=True)
    # One active variant at a time: aggregate allocator budget is never duplicated.
    for path in SHARED.glob('*.claim'):
        data = read_claim(path)
        if data.get('owner') == 'S-RB-RUNA' and not any(
                (SHARED/(path.stem+suffix)).exists() for suffix in ('.done', '.failed.json')):
            return
    for value in candidates:
        path = Path(value)
        if not path.exists():
            continue
        spec = json.loads(path.read_text(encoding='utf-8'))
        validate(spec, path)
        if (SHARED/(spec['variant']+'.claim')).exists():
            continue
        dispatch(spec)
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--continuation-of', type=int)
    args = parser.parse_args()
    assert STATE.parent.is_dir() and SHARED.is_dir()
    STATE.mkdir(exist_ok=True)
    import msvcrt
    lock = (STATE/'WATCH.lock').open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    save_new(STATE/f'WATCH_{os.getpid()}.json', dict(pid=os.getpid(),
        started=datetime.now().astimezone().isoformat(), deadline=DEADLINE,
        freeze=FREEZE, poll_seconds=600, continuation_of=args.continuation_of,
        authority='owner02:08 / INT_NOON_SCHEDULE_REQUEST.md / registration5736'))
    while True:
        begin = time.monotonic()
        try:
            tick()
        except Exception as error:
            print('POLL ERROR', repr(error), flush=True)
        if args.once or time.time() >= DEADLINE:
            break
        time.sleep(min(max(1, 600-(time.monotonic()-begin)), max(0, DEADLINE-time.time())))


if __name__ == '__main__':
    main()
