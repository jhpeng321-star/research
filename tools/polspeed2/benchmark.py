"""Owned sequential gpuA benchmark; STOP stops own runs/workers cooperatively."""
from pathlib import Path
import argparse
import json
import os
import struct
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4/polspeed2_20261028')
LOOP = Path('/root/autodl-tmp/c4/restart20261026/arm/loop')
RESOLVED = LOOP / 'c4prep/resolved_A_MAIN1500_8e75184392b3'
RT = BASE / 'runtime'
FGA = ('GRAPH_CVAR ASYNC_SPILL PIVOT_DIAG_OFF FAST_MERGE_LANE '
       'FAST_CAV_DRIVE FAST_LONG_HISTORY').split()
VARIANTS = {
    'nine_N2': (2, 8, '.245', False),
    'six_N2': (2, 8, '.20', True),
    'six_N4': (4, 2, '.10', True),
    'six_N8': (8, 1, '.05', True),
    'six_serial': (1, 8, '.25', True),
}


def gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                    '--format=csv,noheader,nounits'], text=True)
    return [int(x.strip()) for x in text.strip().split(',')]


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in {'t', 'wall_time_s', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'}:
                yield from flatten(item, prefix + key + '.')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from flatten(item, prefix + str(i) + '.')
    else:
        yield prefix.rstrip('.'), value


def compare(reference, actual):
    a, b = dict(flatten(reference)), dict(flatten(actual))
    diffs = []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        equal = (struct.pack('!d', x) == struct.pack('!d', y)
                 if type(x) is float and type(y) is float else x == y)
        if not equal:
            numeric = type(x) in (int, float) and type(y) in (int, float)
            diffs.append(dict(key=key, reference=x, actual=y,
                              absolute=abs(x-y) if numeric else None,
                              relative=abs(x-y)/max(abs(x), abs(y), 1e-30) if numeric else None))
    return dict(compared_leaves=len(a.keys() & b.keys()), bitwise_identical=not diffs,
                n_diff=len(diffs), differences=diffs)


def update_row(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return next(row for row in rows if row.get('update') == 181)


def run(name):
    n, segment, fraction, six = VARIANTS[name]
    used, total = gpu()
    if used > total * .05:
        raise RuntimeError(f'gpuA has {used} MiB in use; refusing benchmark without re-budgeting')
    out = BASE / 'out' / name
    out.mkdir(parents=True, exist_ok=False)
    config = BASE / f'{name}.toml'
    lines = [f'extends = "{RESOLVED}/POLICY.toml"', '[trainer]',
             f'imagination_checkpoint_segment = {segment}']
    if six:
        lines += ['[counterfactual]', 'curriculum_updates = [0,1]', 'n_individual = 0',
                  'cover_group_members = false', 'w_individual = [0.0,0.0]',
                  'w_coop = [0.0,0.0]', 'w_group = [18.8,18.8]',
                  'w_all = [1.0,1.0]', 'w_team = [10.3,10.3]']
    config.write_text('\n'.join(lines) + '\n')
    env = dict(os.environ, PYTHONPATH=f'{RT}/src:{RT}', C4_ALLOWED_ROOTS='/root/autodl-tmp/c4',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', C4_PROBES_DEVICE='cpu',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', C4_GPU_MEM_FRACTION='.10' if n > 1 else '.25',
        WMS_DATA_V5_NATIVE_Q='/root/autodl-tmp/c4/restart20261026_inputs/data/v5_native_q',
        WMS_DATA_GENRAND_PLAIN_Q='/root/autodl-tmp/c4/restart20261026_inputs/data/genrand_plain_q',
        WMS_SPEED_GRAD_AUDIT_EVERY='25', WMS_SPEED_FUTURE_WORKERS=str(n),
        WMS_SPEED_PARALLEL_FORWARD='1' if n > 1 else '0',
        WMS_SPEED_FUTURE_MEMORY_FRACTION=fraction, WMS_SPEED_FORWARD_MEMORY_FRACTION='.04',
        WMS_WMSPEED_DEDUPE_MOBIL='1', WMS_WMSPEED_MOBIL_LAZY_SAFE='1')
    env.update({f'WMS_SPEED_{flag}': '1' for flag in FGA})
    cmd = [sys.executable, '-B', '-m', 'wms.control.policy14_cli', '--config', str(config),
        '--wm-ckpt', str(LOOP/'wm/c03.a2/wm_s.pt'), '--wm-config', str(RT/'configs/c4_wm_loop14_r16_ft.toml'),
        '--real-returns', str(LOOP/'returns/returns_c03.a1.npz'),
        '--longcritic-native-returns', str(LOOP/'returns/native_returns_c03.a1.npz'),
        '--paircrit-receipt', str(LOOP/'returns/pairing_c03.a1.json'),
        '--paircrit-policy-sha256', '0978aa7288cf9e36a0b61949b1e236b3fe48a0d220dada64f67a4797f969dc5e',
        '--gate-panel', '/root/autodl-tmp/c4/restart20261026/arm/gate_panel.json',
        '--horizon-gate', str(RESOLVED/'HORIZON_AUTHORITY.json'),
        '--looplong-boundary', str(RESOLVED/'BOUNDARY.json'), '--native-init', '--phase-relative-signal-gate',
        '--real-credit', str(LOOP/'sumocredit/REAL_CREDIT_c03.a1.json'), '--critic-warmup-updates', '0',
        '--resume', str(LOOP/'policy/c01.a3/policy_latest.pt'), '--device', 'cuda', '--seed', '965140',
        '--max-updates', '150', '--output-dir', str(out), '--compact', '--prep-dry-run-updates', '1']
    receipt = dict(name=name, argv=cmd, env={k:v for k,v in env.items() if k.startswith(('WMS_', 'C4_', 'OMP_', 'MKL_', 'PYTORCH_'))},
                   initial_gpu_mib=used, gpu_total_mib=total, started=time.time())
    (out/'LAUNCH.json').write_text(json.dumps(receipt, indent=2))
    with (out/'console.log').open('w') as log, (out/'gpu_memory.jsonl').open('w') as memory:
        process = subprocess.Popen(cmd, env=env, cwd=BASE, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        (out/'PID').write_text(str(process.pid))
        peak = used
        while process.poll() is None:
            used, total = gpu()
            peak = max(used, peak)
            memory.write(json.dumps(dict(t=time.time(), used_mib=used, total_mib=total))+'\n')
            memory.flush()
            if (BASE/'STOP').exists() or used > total * .6:
                (out/'STOP').touch()
                for pool in (out/'future_workers').glob('*'):
                    (pool/'STOP').touch()
                if used > total * .6:
                    (BASE/'STOP').touch()
            time.sleep(2)
    result = dict(name=name, returncode=process.returncode, elapsed_s=time.time()-receipt['started'],
                  peak_total_gpu_mib=peak, peak_total_gpu_fraction=peak/total)
    if (out/'updates.jsonl').exists():
        row = update_row(out/'updates.jsonl')
        reference = update_row(RESOLVED/'DRY_RUN/A/updates.jsonl')
        result.update(s_per_update=row['wall_time_s'], projected_150_hours=row['wall_time_s']*150/3600,
                      parity=compare(reference, row))
        if (BASE/'out/six_serial/updates.jsonl').exists() and six:
            result['six_serial_parity'] = compare(update_row(BASE/'out/six_serial/updates.jsonl'), row)
    (out/'RESULT.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variants', nargs='+', choices=VARIANTS)
    args = parser.parse_args()
    for name in args.variants:
        if (BASE/'STOP').exists():
            break
        run(name)
