"""S-LONGINT: WM2 horizon gate = the unchanged registered tools/coopw/horizon_gate.py,
sharded by group on gpuA CPUs (CUDA hidden, one thread/worker, no SUMO/training).

Each group keeps its global index and CRN seed SEED0+7919*g; the combiner applies
the module's own bootstrap/sign/Pearson functions in global group order and writes
the same coopw_horizon_gate/1 document the sequential script would write.

    python -B tools/longint/gate_shard.py launch  OUTDIR WM_CKPT WM_CONFIG PAIRS THRESHOLD SHARDS
    python -B tools/longint/gate_shard.py worker  OUTDIR SHARD SHARDS
    python -B tools/longint/gate_shard.py combine OUTDIR
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2]
HORIZONS = [10.0, 30.0, 60.0, 90.0, 120.0, 180.0, 300.0]
GROUPS, PANEL_SEED = 96, 20261026


def load_gate():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['COOPW_ROOT'] = str(CODE)
    spec = importlib.util.spec_from_file_location('registered_gate', CODE / 'tools/coopw/horizon_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    import torch
    torch.set_num_threads(1)
    return gate


def worker(out, shard, shards):
    os.nice(10)
    spec = json.loads((out / 'SPEC.json').read_text())
    gate = load_gate()
    acc = gate._accept()
    acc.check_roots([spec['pairs']])
    cfg, settings, model = acc.load_arm(spec['wm_config'], spec['wm_ckpt'])
    steps = int(round(HORIZONS[-1] / gate.DT_S))
    ds = acc.build_sets(cfg, [spec['pairs']], steps)[0]
    panel = gate.panel_groups(ds, GROUPS, PANEL_SEED)
    (out / 'rows').mkdir(exist_ok=True)
    for g, (key, group) in enumerate(panel):
        target = out / 'rows' / f'{g:03d}.json'
        if g % shards != shard or target.exists():
            continue
        start = time.time()
        runs = {}
        for name, index in [('B', group['B'])] + [(f'SEED{j}', i) for j, i in enumerate(group['SEED'])]:
            r = acc.free_rollout(model, settings, ds[index], steps, gate.SEED0 + 7919 * g)
            sel = r['t0_present'] & ~r['is_cav']
            runs[name] = {h: (acc.hdv_delay(r['speed'], r['mask'], sel, int(round(h / gate.DT_S))),
                              acc.hdv_delay(r['rec_speed'], r['rec_mask'], sel, int(round(h / gate.DT_S))))
                          for h in HORIZONS}
        dj = {str(int(h)): [(runs[a][h][0] - runs['B'][h][0], runs[a][h][1] - runs['B'][h][1])
                            for a in runs if a != 'B'] for h in HORIZONS}
        row = {'group': g, 'pair_key': key, 'arms': len(runs) - 1, 'dJ': dj, 'n_panel': len(panel),
               'elapsed_s': time.time() - start}
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(row))
        tmp.replace(target)
        print(json.dumps({'group': g, 'seconds': row['elapsed_s']}), flush=True)


def combine(out):
    spec = json.loads((out / 'SPEC.json').read_text())
    gate = load_gate()
    rows = [json.loads(p.read_text()) for p in sorted((out / 'rows').glob('*.json'))]
    n = rows[0]['n_panel']
    assert [r['group'] for r in rows] == list(range(n)), 'missing groups'
    per_h = {h: [[tuple(x) for x in r['dJ'][str(int(h))]] for r in rows] for h in HORIZONS}
    results = {}
    for h in HORIZONS:
        sign, sign_ci = gate.bootstrap(per_h[h], gate.sign_agreement, seed=int(h))
        corr, corr_ci = gate.bootstrap(per_h[h], gate.pearson, seed=int(h) + 1)
        results[str(int(h))] = {'sign_agreement': sign, 'sign_ci95': sign_ci, 'pearson': corr,
                                'pearson_ci95': corr_ci, 'n_pairs': sum(len(p) for p in per_h[h]),
                                'n_groups': len(per_h[h])}
    reg = json.loads(Path(spec['threshold']).read_text(encoding='utf-8'))
    threshold = reg['threshold']
    source = {'kind': 'registered_file', 'file': spec['threshold'], 'sha256': gate.sha256(spec['threshold']),
              **reg.get('threshold_source', {})}
    for h in HORIZONS[1:]:
        entry = results[str(int(h))]
        entry['verdict'] = ('PASS' if entry['sign_agreement'] is not None and threshold is not None
                            and entry['sign_agreement'] >= threshold else 'FAIL')
    results[str(int(HORIZONS[0]))]['verdict'] = 'BASE'
    doc = {'schema': gate.SCHEMA, 'wm_sha256': gate.sha256(spec['wm_ckpt']),
           'wm_config_sha256': gate.sha256(spec['wm_config']), 'pairs_root': str(spec['pairs']),
           'panel_seed': PANEL_SEED, 'groups': n,
           'rule': ('horizon h PASS iff WM sign agreement of dJ(arm - B) t0-HDV delay at h >= '
                    'threshold; threshold = lower 95% group-bootstrap bound of the sign agreement '
                    'at the base horizon of the registering WM'),
           'threshold': threshold, 'threshold_source': source, 'horizons': results,
           'rows': [{'pair_key': r['pair_key'], 'arms': r['arms'], 'dJ': r['dJ']} for r in rows],
           'execution': {'task': 'S-LONGINT', 'sharded_registered_source_sha256':
                         gate.sha256(CODE / 'tools/coopw/horizon_gate.py'), 'shards': spec['shards']}}
    target = out / 'HORIZON_GATE_WM2.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k != 'rows'}, indent=1))


def launch(out, wm_ckpt, wm_config, pairs, threshold, shards):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'SPEC.json').write_text(json.dumps({'wm_ckpt': wm_ckpt, 'wm_config': wm_config, 'pairs': pairs,
                                               'threshold': threshold, 'shards': shards}, indent=1))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONPATH': f'{CODE}/src:{CODE}'}
    children, pids = [], []
    for shard in range(shards):
        log = (out / f'worker{shard:03d}.log').open('x')
        child = subprocess.Popen([sys.executable, '-B', __file__, 'worker', str(out), str(shard), str(shards)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        pids.append({'pid': child.pid, 'start_ticks': ticks, 'shard': shard})
        children.append(child)
    (out / 'WORKERS.json').write_text(json.dumps(pids, indent=1))
    codes = [child.wait() for child in children]
    done = {'exit_codes': codes, 'all_ok': all(c == 0 for c in codes), 't': time.time()}
    if done['all_ok']:
        with (out / 'combine.log').open('x') as log:
            done['combine_exit'] = subprocess.run([sys.executable, '-B', __file__, 'combine', str(out)],
                                                  env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (out / 'DONE.json').write_text(json.dumps(done))
    return 0 if done.get('combine_exit') == 0 else 1


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'worker':
        worker(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == 'combine':
        combine(Path(sys.argv[2]))
    else:
        sys.exit(launch(Path(sys.argv[2]), *sys.argv[3:7], int(sys.argv[7])))
