"""Owner-authorized operational continuation; no scientific gate is relaxed."""
from __future__ import annotations

import copy
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_children = {}
_lock = threading.Lock()
_retry_lock = threading.Lock()


def scientific(text, state=None):
    return bool((state or {}).get('stopped')) or any(s in str(text).lower() for s in
        ('nan', 'nonfinite', 'finite false', 'native-relative', 'gate', 'safety', 'incident',
         'scientific', 'second failure', 'second_failure', 'budget', 'exhausted', 'registration',
         'hash', 'provenance', 'restricted', 'phase order'))


def claim_retry(loop, arm, receipt):
    """One operational repair for the entire authorized continuation, including threads."""
    with _retry_lock:
        path = arm/'OPERATIONAL_RETRY.json'
        if path.exists():
            raise loop.Refused('second failure: authorized operational retry already consumed')
        loop.write_atomic(path, receipt)


def child(proc, token):
    with _lock:
        _children[proc.pid] = (proc, token)


def release(proc):
    with _lock:
        _children.pop(proc.pid, None)


def owns(pid, token):
    with _lock:
        item = _children.get(pid)
        return bool(item and item[0].poll() is None and item[1] == token and token is not None)


def phase(loop, cfg, name, arm):
    """One durable repair/retry per phase/cycle, including across driver restarts."""
    root = Path(cfg['loop_root'])
    state = loop.load_state(cfg)
    k = state['cycles'][-1]['cycle']
    record = arm / f'RETRY_c{k:02d}_{name}.json'
    previous = loop.read_json(record) if record.exists() else None
    failed = state['cycles'][-1]['phases'][name]['status'] == 'failed'
    if failed and previous:
        raise loop.Refused('second failure / previously consumed operational retry')
    code = 3 if failed else loop.run_phase(cfg, name)
    if code in (0, 4, 5):
        return code
    state = loop.load_state(cfg)
    ph = state['cycles'][-1]['phases'][name]
    # Scientific refusals/NaNs must never become operational retries. Refused codes
    # include accounting, safety and provenance gates and therefore fail closed.
    text = str(ph.get('error', ph.get('refused', ''))).lower()
    if scientific(text, state) or previous:
        raise loop.Refused(f'{name}: scientific refusal or second failure: {text}')
    # A fresh attempt uses existing official recovery: unique output, peer replay,
    # immutable verified successes, retained charges. No source/config auto-editing.
    receipt = {'t': loop.now(), 'cycle': k, 'phase': name, 'failure': copy.deepcopy(ph),
               'repair': 'official retry: fresh attempt directory; collection resumes verified worlds on peer',
               'status': 'retry_armed', 'charges_before': state['sim_s_charged']}
    claim_retry(loop, arm, receipt)
    loop.write_atomic(record, receipt)
    loop.append(root / 'cycle_ledger.jsonl', {'event': 'operational_fix_retry', **receipt})
    code = (loop.retry(cfg, name) if ph['status'] == 'failed' else loop.run_phase(cfg, name))
    receipt.update(status='recovered' if code in (0, 4) else 'second_failure', exit=code,
                   charges_after=loop.load_state(cfg)['sim_s_charged'])
    loop.write_atomic(record, receipt)
    if code not in (0, 4, 5):
        raise loop.Refused(f'{name}: second failure exit {code}')
    return code


def drive(loop, waiter, cfg, arm):
    """Catch one operational setup error as well as the phase-specific retries."""
    marker = arm/'DRIVER_RETRY.json'
    while True:
        try:
            result = _drive(loop, waiter, cfg, arm)
            if marker.exists():
                loop.write_atomic(marker, {**loop.read_json(marker), 'status':'recovered'})
            return result
        except Exception as error:
            if scientific(error) or marker.exists():
                raise
            receipt = {'t':loop.now(),'error':str(error),'status':'retry_armed',
                       'repair':'re-read pinned state and dependencies; retry owned setup once'}
            claim_retry(loop, arm, receipt)
            loop.write_atomic(marker, receipt)


def _drive(loop, waiter, cfg, arm):
    """Single state writer; immutable policy snapshot overlaps WM-FT and CPU check."""
    fast = loop.fast_module()
    root = Path(cfg['loop_root'])
    future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            state = loop.load_state(cfg)
            if state.get('stopped') or (arm / 'STOP_ARM').exists() or (root / 'STOP_DRIVER').exists():
                if future is not None:
                    future.result()
                return
            k = state['cycles'][-1]['cycle']
            name = loop._phase(state)['name']
            if name is None:
                return
            if name == 'policy':
                waiter.prepare_horizon(cfg, state, arm)
            if (loop.fast_enabled(cfg, 'overlap_policy') and name == 'wm_ft' and future is None
                    and k + 1 < loop.check_registration(cfg)['budget']['max_cycles']
                    and not (root / f'overlap_c{k+1:02d}.json').exists()):
                waiter.prepare_horizon(cfg, state, arm)
                frozen = copy.deepcopy(cfg)
                args = frozen['policy'].get('extra_args', [])
                for flag in ('--horizon-gate', '--gate-panel'):
                    if flag in args:
                        i = args.index(flag) + 1
                        dst = root / f'overlap_c{k+1:02d}_{flag[2:]}.json'
                        source = Path(args[i]).read_bytes()
                        if dst.exists() and dst.read_bytes() != source:
                            raise loop.Refused('overlap frozen dependency changed')
                        dst.write_bytes(source)
                        args[i] = str(dst)
                snapshot = copy.deepcopy(state)
                converted = state['cycles'][-1]['phases']['convert'].get('receipt', {})
                if converted.get('returns'):
                    snapshot['real_returns'] = converted['returns']
                future = pool.submit(overlap_retry, loop, fast, frozen, snapshot, root, arm)
            # Join BEFORE check can promote/advance state; wm_ft ran concurrently.
            if name == 'check' and future is not None:
                future.result()
                future = None
            code = phase(loop, cfg, name, arm)
            if code == 5:
                time.sleep(120)
            elif name == 'check':
                after = loop.load_state(cfg)
                completed = next(c for c in after['cycles'] if c['cycle'] == k)
                waiter.retire_pair_cache(cfg, completed, arm)


def overlap_retry(loop, fast, cfg, state, root, arm):
    k = state['cycles'][-1]['cycle'] + 1
    marker = arm / f'RETRY_overlap_c{k:02d}.json'
    attempt = 2 if marker.exists() else 1
    while True:
        cfg.setdefault('resumeprep', {})['overlap_attempt'] = attempt
        try:
            result = fast.overlap(loop, cfg, state, root)
            if marker.exists():
                doc = loop.read_json(marker)
                loop.write_atomic(marker, {**doc, 'status': 'recovered'})
            return result
        except loop.GpuBusy:
            time.sleep(2)
        except Exception as error:
            if attempt == 2 or scientific(error):
                raise loop.Refused(f'overlap scientific refusal or second failure: {error}') from error
            receipt = {'t': loop.now(), 'error': str(error), 'status': 'retry_armed',
                       'repair': 'new owned overlap attempt directory; same immutable dependencies'}
            claim_retry(loop, arm, receipt)
            loop.write_atomic(marker, receipt)
            attempt = 2
