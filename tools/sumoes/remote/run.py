"""S-SUMOES-RUN driver entry on cpuQ (owned, nice 19). Registered design 05b84910.

    python run.py baseline            # canary (CTRL0 vs ours@0 on s969536) + 7 panel worlds
    python run.py updates --screener wm --max N   # resume state.json, run N more updates

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
    if name == 'wm':
        from tools.sumoes.remote.wm_screen import WmScreener
        return WmScreener(config)
    raise SystemExit(f'screener {name!r} is not registered (only the WM 60 s screen is)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'updates'))
    parser.add_argument('--screener', default='wm')
    parser.add_argument('--max', type=int, default=1)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
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
    driver, state = SumoEsDriver.resume(state_path, config=config, evaluator=evaluator,
                                        screener=screener_for(args.screener, config))
    for _ in range(args.max):
        if state.update_count >= config.max_updates:
            break
        log('update_start', update=state.update_count, pid=os.getpid())
        receipt = driver.run_update(state)
        log('update_done', **{k: receipt.get(k) for k in (
            'update', 'sigma', 'sensitivity', 'pair_deltas', 'grad_norm', 'accepted',
            'no_step', 'panel_mean_before', 'panel_mean_after', 'budget')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
