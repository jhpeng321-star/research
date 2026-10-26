"""Bound C1 remainder -> convert -> WM -> unchanged check; one operational retry."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]


def phase_once(loop, resume, cfg, name):
    """At most one durable operational recovery across this continuation."""
    marker = RELEASE/'C1COLLECT4_RETRY.json'
    while True:
        state = loop.load_state(cfg)
        ph = state['cycles'][-1]['phases'][name]
        code = loop.retry(cfg, name) if ph['status'] == 'failed' else loop.run_phase(cfg, name)
        if code == 5:
            time.sleep(30)
            continue
        if code in (0, 4):
            return code
        state = loop.load_state(cfg)
        ph = state['cycles'][-1]['phases'][name]
        text = ph.get('error') if code == 3 else ph.get('refused')
        if code != 3 or resume.scientific(text, state) or marker.exists():
            raise resume.OperationalPaused(f'{name} exit {code}; no eligible operational retry: {text}')
        doc = {'t': loop.now(), 'phase': name, 'failure': ph, 'charged': state['sim_s_charged'],
               'authority': 'S-C1COLLECT4 main 2026-10-27 06:20', 'retry_limit': 1,
               'repair': 'official fresh attempt; reread dependencies, retain verified outputs and all charges'}
        # Exclusive creation makes the retry budget durable across driver restarts.
        with marker.open('x') as stream:
            json.dump(doc, stream, indent=2)
        loop.append(ARM/'loop/cycle_ledger.jsonl', {'event': 'S-C1COLLECT4_operational_retry', **doc})


def drive():
    import loop14 as loop
    import loop14_resume as resume
    from tools.loopresume import continue_c1 as previous
    cfg = loop.load_config(ARM/'CONFIG.json')
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            binding = loop.read_json(RELEASE/'C1COLLECT4_BINDINGS.json')
            assert loop.sha256(ARM/'CONFIG.json') == binding['config_sha256']
            assert loop.sha256(cfg['registration']) == binding['registration_sha256']
            for name, sha in binding['source'].items():
                assert loop.sha256(ROOT/name) == sha
            with (RELEASE/'C1COLLECT4_ATTEMPT.json').open('x') as stream:
                json.dump({'t': loop.now(), 'pid': os.getpid(), 'remaining_worlds': list(range(968161, 968170))}, stream)
            phase_once(loop, resume, cfg, 'collect')
            phase_once(loop, resume, cfg, 'convert')
            previous.reconcile_wm(cfg)
            previous.next_boundary(cfg)
            for name in ('wm_ft', 'check'):
                phase_once(loop, resume, cfg, name)
            loop.write_atomic(RELEASE/'C1COLLECT4_COMPLETE.json',
                              {'t': loop.now(), 'charged': loop.load_state(cfg)['sim_s_charged']})
        except Exception as error:
            state = loop.load_state(cfg)
            name = 'WAIT_OWNER.json' if state.get('stopped') else 'OPERATIONAL_PAUSED.json'
            loop.write_atomic(ARM/name, {'t': loop.now(), 'task': 'S-C1COLLECT4', 'error': repr(error)})
            previous.event('S-C1COLLECT4_stopped', receipt=name, error=repr(error))
            raise


if __name__ == '__main__':
    drive()
