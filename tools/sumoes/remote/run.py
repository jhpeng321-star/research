"""S-SUMOES-RUN driver entry on cpuQ (owned, nice 19). Registered design 05b84910.

    python run.py baseline            # canary (CTRL0 vs ours@0 on s969536) + 7 panel worlds
    python run.py request             # write screens/request_uXX.json for the next update
    python run.py updates --screener file --max 1  # consume screens/screen_uXX.json, run it

All SUMO goes through SumoRuntimeEvaluator (charge-before-dispatch, durable ledger,
no retries, STOP on any failure). Receipts: OWN/state.json + OWN/RUN_LOG.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

OWN = Path('/home/ubuntu/sumoes20261031')
DESIGN_SHA = '05b84910472db97619147b12df3bc2a28f82125ec2080ae333deb252d80105c7'
sys.path[:0] = [str(OWN / 'pytools')]

from tools.sumoes.config import SumoEsConfig  # noqa: E402
from tools.sumoes.driver import SumoEsDriver  # noqa: E402
from tools.sumoes.sumo_eval import SumoRuntimeEvaluator  # noqa: E402


def log(kind, **data):
    row = dict(kind=kind, unix=time.time(), **data)
    with (OWN / 'RUN_LOG.jsonl').open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    print('@@RUN ' + json.dumps(row, allow_nan=False), flush=True)


def screener_for(name, config):
    if name == 'file':
        from tools.sumoes.file_screen import FileScreener
        return FileScreener(OWN / 'screens', keep=config.pairs_per_update)
    raise SystemExit(f'screener {name!r} is not registered (only the WM 60 s screen via file)')


def stop_reason(receipts):
    """Screen no-step is a rejected opportunity, and breaks no losing streak."""
    updates = [r for r in receipts if r.get('update', -1) >= 0]
    if len(updates) >= 4 and all(r.get('accepted') is not True for r in updates[-4:]):
        return 'four_consecutive_rejected_updates_including_screen_no_step'
    return None


def check_ready(driver, state, evaluator):
    assert not (OWN / 'STOP').exists(), 'STOP exists; no retry'
    reason = stop_reason(driver.receipts)
    if reason:
        log('scientific_stop', reason=reason, update_count=state.update_count,
            budget=driver.ledger.to_dict())
        raise SystemExit(0)
    assert state.update_count < driver.config.max_updates, 'registered updates exhausted'
    # Never resume from a stale state over an already charged/unfinished dispatch.
    assert sum(r['sim_s'] for r in evaluator.ledger_rows()) == driver.ledger.charged_sim_s
    if state.update_count == 10 and not state.reject_sigma_halved:
        first_ten = [r for r in driver.receipts if 0 <= r.get('update', -1) <= 9]
        if sum(r.get('accepted') is not True for r in first_ten) > 2 * len(first_ten) / 3:
            state.sigma_target_residual_rms_mps2 *= .5
            state.reject_sigma_halved = True
            driver.save(state)
            log('registered_sigma_halving', target=state.sigma_target_residual_rms_mps2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'request', 'updates'))
    parser.add_argument('--screener', default='file')
    parser.add_argument('--max', type=int, default=1)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    import fcntl
    lock = (OWN / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    design_path = OWN / 'design.json'
    assert hashlib.sha256(design_path.read_bytes()).hexdigest() == DESIGN_SHA
    design = json.loads(design_path.read_text())
    config = SumoEsConfig.from_dict(design['config'])
    assert config.enabled and config.cap_sim_s == 650_000
    evaluator = SumoRuntimeEvaluator(OWN, cap_sim_s=config.cap_sim_s, workers=args.workers,
                                     design=design)
    state_path = OWN / 'state.json'
    if args.mode == 'baseline':
        assert not state_path.exists(), 'baseline already exists; never rerun'
        driver = SumoEsDriver(config=config, evaluator=evaluator, screener=None,
                              canary_arm_check=True, state_path=state_path)
        state = driver.initial_state()
        log('baseline_start', pid=os.getpid())
        receipt = driver.baseline(state)
        driver.receipts.append(receipt)
        driver.save(state)
        log('baseline_done', panel_J=receipt['panel_J'], panel_mean=receipt['panel_mean'],
            sensitivity=receipt['sensitivity'], budget=receipt['budget'])
        return 0
    if args.mode == 'request':
        from tools.sumoes.file_screen import request_payload, write_request
        driver, state = SumoEsDriver.resume(state_path, config=config, evaluator=evaluator,
                                             screener=None)
        check_ready(driver, state, evaluator)
        update = state.update_count
        sigma = state.sigma(driver.sensitivity)
        directions = state.directions(update, config.wm_screen_directions)
        digest = write_request(OWN / 'screens', request_payload(
            update, state.theta, sigma, driver.sensitivity, directions))
        log('screen_request', update=update, sigma=sigma, sensitivity=driver.sensitivity,
            request_sha256=digest)
        return 0
    driver, state = SumoEsDriver.resume(state_path, config=config, evaluator=evaluator,
                                        screener=screener_for(args.screener, config))
    for _ in range(args.max):
        if state.update_count >= config.max_updates:
            break
        check_ready(driver, state, evaluator)
        begin = time.monotonic()
        log('update_start', update=state.update_count, pid=os.getpid())
        receipt = driver.run_update(state)
        receipt['wall_s'] = time.monotonic() - begin
        driver.save(state)
        log('update_done', wall_s=receipt['wall_s'], **{k: receipt.get(k) for k in (
            'update', 'sigma', 'sensitivity', 'pair_deltas', 'grad_norm', 'accepted',
            'step_rule', 'step_norm', 'no_step', 'panel_mean_before', 'panel_mean_after', 'budget')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
