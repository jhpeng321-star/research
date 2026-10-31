"""Owned CPU-only, create-once addendum dispatch on cpuQ; no automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/ubuntu/rbfit20261101_addendum')
OLD = Path('/home/ubuntu/rbfit20261031')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'lane', 'calibrate', 'status'))
    args = parser.parse_args()
    if args.action == 'prepare':
        if not ROOT.parent.is_dir() or not OLD.is_dir():
            raise ValueError('Expected existing cpuQ parents missing')
        ROOT.mkdir(exist_ok=False)
        print(json.dumps({'root': str(ROOT), 'created': True}))
        return
    if args.action == 'status':
        for name in ('lane_v2', 'shrinkage_v1'):
            receipt = ROOT / (name + '_process.json')
            if receipt.exists():
                r = json.loads(receipt.read_text())
                proc = Path('/proc') / str(r['pid'])
                print(json.dumps({'name': name, 'pid': r['pid'], 'exists': proc.exists(),
                    'cmdline': (proc / 'cmdline').read_bytes().decode().replace('\0', ' ') if proc.exists() else '',
                    'validation': (ROOT/name/'validation.json').exists()}))
        return
    if os.getloadavg()[0] > min(20, os.cpu_count() - 2):
        raise RuntimeError('Spare-load admission failed; no dispatch')
    name = 'lane_v2' if args.action == 'lane' else 'shrinkage_v1'
    receipt = ROOT / (name + '_process.json')
    output = ROOT / name
    if receipt.exists() or output.exists():
        raise RuntimeError('Prior dispatch/output exists; preserve and inspect it')
    if args.action == 'lane':
        required = (OLD/'fit_v1/idm_population.json', OLD/'fit_v1/idm_posteriors.json',
                    OLD/'pooled_full_v5/idm_pooled.json')
        command = ['tools/rbfit/lane_fit.py', '--root', '/home/ubuntu/wmdata20261031/views',
            '--idm', str(OLD/'fit_v1'), '--idm-pooled', str(OLD/'pooled_full_v5/idm_pooled.json'),
            '--out', str(output), '--workers', '2', '--sobol-power', '6']
    else:
        required = (OLD/'fit_mixture_v4/idm_population.json',
                    ROOT/'tools/rbfit/calibrate_shrinkage.py')
        command = ['tools/rbfit/calibrate_shrinkage.py', '--root', '/home/ubuntu/wmdata20261031/views',
            '--fit', str(OLD/'fit_mixture_v4'), '--out', str(output)]
    if not all(p.is_file() for p in required):
        raise ValueError('Required frozen inputs missing before dispatch')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=str(ROOT/'src'))
    command = ['nice', '-n', '19', sys.executable, *command]
    source_paths = [ROOT/'src/wms/physics'/name for name in (
        'lc2013.py', 'lc2013_geometry.py', 'config.py', 'mobil.py', 'neighbours.py',
        'batched.py', 'motion_fit.py')]
    source_paths += [ROOT/'tools/rbfit'/name for name in (
        'idm_fit.py', 'lane_fit.py', 'calibrate_shrinkage.py', 'dispatch_addendum.py')
        if (ROOT/'tools/rbfit'/name).is_file()]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_paths}
    # Exclusive receipt claims dispatch before Popen. An interrupted claim requires
    # human reconciliation, never a second launch from this command.
    with receipt.open('x') as stream:
        json.dump({'state': 'claimed', 'argv': command, 'source_sha256': manifest}, stream)
    with (ROOT/(name+'.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    data = {'state': 'dispatched', 'pid': process.pid, 'argv': command,
            'timestamp_unix': time.time(), 'load': os.getloadavg(),
            'source_sha256': manifest, 'gpu': False, 'automatic_retry': False}
    receipt.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
