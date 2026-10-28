"""Live SSH streams: append registration before each allocation; no polling loop.

S-EXECPARITY 00:20 registered SUMO check, separate cap 7,200 sim-s:
canary (s977000 always0 == stored S-CTRLARM CTRL0) then production
(s977001/977002/977003 alwaysx). At most four charged infrastructure retries
exist in the design; none is armed here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
REG = Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
EVIDENCE = ROOT / 'evidence_execparity'
REMOTE = '/home/ubuntu/execparity/always0020'
CAP = 7200
MARKER = 'S-EXECPARITY ALLOC '
LOCK = threading.Lock()


def append(line):
    with REG.open('ab', buffering=0) as f:
        f.write(('\n' + line + '\n').encode('utf-8'))
        os.fsync(f.fileno())


def charge(host, job):
    with LOCK:
        text = REG.read_text(encoding='utf-8')
        rows = [json.loads(s.split(MARKER, 1)[1]) for s in text.splitlines() if MARKER in s]
        assert all(r['id'] != job['id'] for r in rows), 'already charged; no retry'
        total = sum(r['charged_sim_s'] for r in rows) + 900
        assert total <= CAP, 'CAP: stop instead of allocating'
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = dict(id=job['id'], host=host, seed=job['seed'], arm=job['arm'],
                     registered_before_start_utc=now, warmup_s=300, horizon_s=600,
                     charged_sim_s=900, cumulative_charged_sim_s=total,
                     cap_sim_s=CAP, runtime='b3c031ad', decision='00:20')
        append('- ' + MARKER + json.dumps(entry, separators=(',', ':')))
        return dict(authorize=job['id'], charged_sim_s=900,
                    registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest())


def run_host(host, mode, only=None):
    # A single connected stream per host. No background launcher or remote poll;
    # the remote worker blocks for a durable local registration ACK for each job.
    names = ['tools/execparity/alwayscmd_adapter.py', 'tools/execparity/sumo_worker.py']
    bundle = {n: (ROOT / n).read_text(encoding='utf-8') for n in names}
    bootstrap = f'''
from pathlib import Path
import os,sys
r=Path({REMOTE!r}); assert r.parent.parent.is_dir(); r.parent.mkdir(exist_ok=True); r.mkdir(exist_ok=True)
bundle={bundle!r}
for name,text in bundle.items():
 p=r/name; p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): assert p.read_text()==text,'staged source differs'
 else:
  with p.open('x') as f: f.write(text)
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 C4_POLICY_STOCHASTIC='0',C4_POLICY_ACCEL_NOISE='0',PYTHONHASHSEED='0',
 C4_ALLOWED_ROOTS='/home/ubuntu/e3final:/home/ubuntu/execparity',
 SUMO_HOME='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo',
 PATH='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo/bin:/home/ubuntu/c4/venv/bin:/home/ubuntu/miniconda3/bin:/usr/bin:/bin')
sys.argv=['sumo_worker.py',{mode!r},{host!r}]+([{only!r}] if {only!r} else [])
source=r/'tools/execparity/sumo_worker.py'
exec(compile(source.read_text(),str(source),'exec'),dict(__name__='__main__',__file__=str(source)))
'''
    command = ('env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
               'nice -n 19 /home/ubuntu/c4/venv/bin/python -B -u -c ' +
               shlex.quote("import sys;exec(sys.stdin.readline())"))
    log_path = EVIDENCE / f'{host}_{mode}_SSH.log'
    events_path = EVIDENCE / f'{host}_{mode}_EVENTS.jsonl'
    assert not log_path.exists() and not events_path.exists()
    with log_path.open('x', encoding='utf-8') as err, events_path.open('x', encoding='utf-8') as events:
        p = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                              '-o', 'ServerAliveInterval=30', host, command],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                             text=True, encoding='utf-8', creationflags=0x08000000)
        p.stdin.write('exec(' + repr(bootstrap) + ')\n')
        p.stdin.flush()
        failed = False
        for line in p.stdout:
            if not line.startswith('@@EXECPARITY '):
                err.write(line)
                err.flush()
                continue
            doc = json.loads(line[len('@@EXECPARITY '):])
            events.write(json.dumps(doc) + '\n')
            events.flush()
            os.fsync(events.fileno())
            kind = doc['kind']
            if kind == 'allocation_request':
                ack = charge(host, doc['job'])
                p.stdin.write(json.dumps(ack) + '\n')
                p.stdin.flush()
            elif kind == 'started':
                with LOCK:
                    append('- S-EXECPARITY START ' + json.dumps(doc, separators=(',', ':')))
                print(f"{host}: START {doc['job']['id']} (900 sim-s charged)", flush=True)
            elif kind == 'complete':
                path = EVIDENCE / (doc['result']['job']['id'] + '.json')
                with path.open('x', encoding='utf-8') as f:
                    json.dump(doc, f, indent=1)
                    f.write('\n')
                print(f"{host}: COMPLETE {doc['result']['job']['id']} "
                      f"actual={doc['result']['actual_sim_s']} identity={doc['identity']}", flush=True)
            elif kind == 'failed':
                failed = True
                print(json.dumps(doc), flush=True)
            else:
                print(json.dumps({k: v for k, v in doc.items()
                                  if k not in ('jobs', 'input_hashes')}), flush=True)
        p.stdin.close()
        rc = p.wait()
        if rc or failed:
            raise RuntimeError(f'{host}/{mode} stopped; rc={rc}; see retained logs; never auto-retry')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('canary', 'production', 'resume'))
    parser.add_argument('--host', choices=('cpuP', 'cpuQ'))
    parser.add_argument('--only', help='resume a single registered job id after an uncharged interruption')
    args = parser.parse_args()
    assert EVIDENCE.is_dir() and REG.is_file()
    if args.mode == 'canary':
        run_host('cpuP', 'canary')
    elif args.mode == 'resume':
        assert args.host and args.only, 'resume needs --host and --only'
        run_host(args.host, 'production', only=args.only)
    else:
        canary = json.loads((EVIDENCE / 's977000_always0.json').read_text())
        assert canary['identity']['bit_identical']
        started = canary['result']['started']['start_unix']
        time.sleep(max(0, 120 - (time.time() - started)))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_host, host, 'production') for host in ('cpuP', 'cpuQ')]
            for f in futures:
                f.result()


if __name__ == '__main__':
    main()
