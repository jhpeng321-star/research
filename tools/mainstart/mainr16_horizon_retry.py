"""Single owner-authorized deployment-repair retry; preserves the failed horizon receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parent
R = A / 'runtime'
ROOT = Path('/root/main14_loop')
SHA = '5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2'
sys.path[:0] = [str(R / 'scripts/c4'), str(R), str(R / 'src')]
import loop14 as loop


def main():
    cfg = loop.read_json(A / 'CONFIG.json')
    target = A / f'horizon_{SHA}.json'
    prior = A / f'horizon_{SHA}.deployment_failure.json'
    assert not prior.exists(), 'exactly one retry allowed'
    assert loop.read_json(target).get('measurement_error')
    target.rename(prior)
    output = A / f'horizon_{SHA}.retry1.json'
    argv = [sys.executable, str(R / 'tools/coopw/horizon_gate.py'),
            '--wm-ckpt', cfg['wm']['initial_wm'], '--wm-config', cfg['wm']['initial_wm_config'],
            '--pairs', cfg['wm']['env']['WMS_DATA_PAIRS300'], '--groups', '96',
            '--horizons', '10', '30', '60', '--threshold',
            str(R / 'configs/coopw_horizon_gate_13E_b2.json'), '--out', str(output)]
    env = {**os.environ, 'PYTHONPATH': f'{R / "src"}:{R}', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'C4_ALLOWED_ROOTS': ':'.join([
               *cfg['policy']['allowed_roots'], cfg['loop_root'], str(A)])}
    with (A / 'horizon_retry1.log').open('xb') as log:
        code = subprocess.run(argv, cwd=R, env=env, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, timeout=7200).returncode
    if code:
        raise RuntimeError(f'horizon deployment repair retry failed, exit {code}')
    doc = loop.read_json(output)
    assert doc['wm_sha256'] == SHA and doc['groups'] == 96 and not doc.get('measurement_error')
    loop.write_atomic(target, doc)
    loop.write_atomic(A / 'horizon_gate_current.json', doc)
    receipt = {'status': 'PASS', 't': loop.now(), 'wm_sha256': SHA, 'groups': doc['groups'],
               'horizons': doc['horizons'], 'threshold': doc['threshold'],
               'archived_failure': str(prior), 'retry_output': str(output), 'new_sumo_sim_s': 0}
    loop.write_atomic(A / 'HORIZON_REPAIRED.json', receipt)
    loop.append(ROOT / 'cycle_ledger.jsonl', {'event': 'horizon_deployment_repair', **receipt})
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        receipt = {'t': loop.now(), 'error': f'{type(error).__name__}: {error}',
                   'status': 'WAIT_OWNER', 'reason': 'second non-scientific failure',
                   'log': str(A / 'horizon_retry1.log')}
        loop.write_atomic(A / 'WAIT_OWNER.json', receipt)
        (A / 'STOP_ARM').touch()
        (ROOT / 'STOP_DRIVER').touch()
        (ROOT / 'policy/c00.a2/STOP').touch()
        raise
