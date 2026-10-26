"""Exactly one S-C1COLLECT3 dispatch attempt, followed by convert -> wm_ft -> check."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent
ARM = RELEASE/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]


def drive():
    import loop14 as loop
    import loop14_resume as resume
    from tools.loopresume import continue_c1 as previous
    cfg = loop.load_config(ARM/'CONFIG.json')
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with (RELEASE/'C1COLLECT3_ATTEMPT.json').open('x') as stream:
                json.dump({'t': loop.now(), 'pid': os.getpid(), 'authority': 'main 2026-10-27 06:00; ONE attempt'}, stream)
            # retry() arms and executes immediately; call it only inside this bound driver.
            phase = loop.load_state(cfg)['cycles'][-1]['phases']['collect']
            code = loop.retry(cfg, 'collect') if phase['status'] == 'failed' else loop.run_phase(cfg, 'collect')
            if code:
                if loop.load_state(cfg).get('stopped'):
                    raise loop.Refused(loop.load_state(cfg)['stopped'])
                raise resume.OperationalPaused(f'single collection attempt exit {code}; no automatic retry')
            code = resume.phase_many(loop, cfg, 'convert', ARM)
            if code not in (0, 4):
                raise resume.OperationalPaused(f'convert exit {code}')
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
            loop.write_atomic(RELEASE/'C1COLLECT3_COMPLETE.json',
                              {'t': loop.now(), 'charged': loop.load_state(cfg)['sim_s_charged']})
        except Exception as error:
            name = 'WAIT_OWNER.json' if loop.load_state(cfg).get('stopped') else 'OPERATIONAL_PAUSED.json'
            loop.write_atomic(ARM/name, {'t': loop.now(), 'task': 'S-C1COLLECT3', 'error': repr(error)})
            previous.event('S-C1COLLECT3_stopped', receipt=name, error=repr(error))
            raise


if __name__ == '__main__':
    drive()
