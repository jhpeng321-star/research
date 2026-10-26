"""One renewed cycle-1 collection, serial startup proofs, then existing numeric phases."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent
ARM = RELEASE/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]

# Only own job metadata, base numerical outputs, and exact collector descendants.
# Scenario files, SUMO saved XML and assignment files are never read.
PROBE = r'''
import hashlib,json,pathlib,subprocess
import numpy as np
work=pathlib.Path(WORK); out=work/'collect'
rows=[]
for line in subprocess.check_output(['ps','--no-headers','-eo','pid,ppid,args'],text=True).splitlines():
 f=line.strip().split(None,2)
 if len(f)==3: rows.append((int(f[0]),int(f[1]),f[2].split()))
collectors=[pid for pid,ppid,args in rows if 'traffic_shepherd.c4.collect' in args and '--out' in args and args[args.index('--out')+1]==str(out)]
owned=set(collectors)
while True:
 more={pid for pid,ppid,args in rows if ppid in owned}
 if more<=owned: break
 owned|=more
sumo=[{'pid':pid,'ppid':ppid,'start_ticks':pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]}
 for pid,ppid,args in rows if pid in owned and args and pathlib.Path(args[0]).name in ('sumo','sumo-gui') and any(a.startswith(str(work)+'/') for a in args)]
proofs=[]
for p in (out/'episodes').glob('*/*/summary.json'):
 d=json.loads(p.read_text()); ep=p.with_name('episode.npz')
 if not ep.is_file(): continue
 with np.load(ep,allow_pickle=False) as z: ts=z['time_s']
 if len(ts)>1 and np.isfinite(ts).all() and np.all(np.diff(ts)>0) and ts[-1]>ts[0] and d.get('sim_seconds',0)>0:
  proofs.append({'summary':str(p),'summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sim_seconds':d['sim_seconds'],'frames':len(ts),'time_range_s':[float(ts[0]),float(ts[-1])]})
print(json.dumps({'work':str(work),'collectors':collectors,'sumo':sumo,'outputs':proofs,'verified':bool(sumo and proofs)}))
'''


def probe(cfg, item, box, attempt):
    import loop14 as loop
    work = (f"{cfg['collect']['work_root']}/w{item['seed']}/{cfg['loop_id']}/c01"
            + (f'.a{attempt}' if attempt > 1 else ''))
    h = loop.host(cfg, box)
    result = subprocess.run([*h['ssh'], h['python']+' -B -'], input='WORK='+repr(work)+'\n'+PROBE,
                            text=True, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout)


def wait_canary(loop, cfg, item, box, attempt, future):
    deadline = time.monotonic()+1200
    while True:
        proof = probe(cfg, item, box, attempt)
        if proof['verified']:
            return dict(t=loop.now(), box=box, seed=item['seed'], attempt=attempt, **proof)
        if future.done():
            # A completed world is also valid, but its numerical artifact must verify.
            world = future.result()  # A pre-SUMO failure propagates; never retry.
            if not loop.fast_module().artifact_valid(world['artifact']) or not proof['outputs']:
                raise RuntimeError('canary finished without verified SUMO numerical outputs')
            return dict(t=loop.now(), box=box, seed=item['seed'], attempt=attempt,
                        completed_world=world['artifact'], **{**proof, 'verified': True})
        if time.monotonic()>=deadline:
            raise RuntimeError('canary SUMO output verification timed out; no further dispatch')
        time.sleep(5)


def drive():
    import loop14 as loop
    import loop14_resume as resume
    from tools.loopresume import continue_c1 as previous
    cfg = loop.load_config(ARM/'CONFIG.json')
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            claim = RELEASE/'C1COLLECT_ATTEMPT.json'
            with claim.open('x') as stream:
                json.dump({'t': loop.now(), 'pid': os.getpid(), 'authority': 'S-C1COLLECT ONE attempt'}, stream)
            code = loop.run_phase(cfg, 'collect')
            if code != 0:
                state = loop.load_state(cfg)
                if state.get('stopped'):
                    raise loop.Refused(state['stopped'])
                raise resume.OperationalPaused(f'single collection attempt exit {code}; main decision required')
            for name in ('convert',):
                code = resume.phase_many(loop, cfg, name, ARM)
                if code not in (0, 4):
                    raise resume.OperationalPaused(f'{name} exit {code}')
            previous.reconcile_wm(cfg)
            previous.next_boundary(cfg)
            for name in ('wm_ft', 'check'):
                while True:
                    code = resume.phase_many(loop, cfg, name, ARM)
                    if code != 5:
                        break
                    time.sleep(30)
                if code not in (0, 4):
                    raise resume.OperationalPaused(f'{name} exit {code}')
            loop.write_atomic(RELEASE/'C1COLLECT_COMPLETE.json', {'t': loop.now(), 'charged': loop.load_state(cfg)['sim_s_charged']})
        except Exception as error:
            name = 'WAIT_OWNER.json' if loop.load_state(cfg).get('stopped') else 'OPERATIONAL_PAUSED.json'
            loop.write_atomic(ARM/name, {'t': loop.now(), 'task': 'S-C1COLLECT', 'error': repr(error)})
            previous.event('S-C1COLLECT_stopped', receipt=name, error=repr(error))
            raise


if __name__ == '__main__':
    drive()
