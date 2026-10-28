"""Prepare and explicitly launch two isolated candidates; never dispatch SUMO.

This task invokes inventory/prepare/dry-run only. `launch` is a later MAIN
command and refuses unresolved inputs or absent matching two-update receipts.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import (ACTORS, CODE, GATE, OUT, REL, WM, WM_CONFIG, WM_SHA,
                    now, qualified_gate, read, safe, sha, write_new)

sys.path[:0] = [str(CODE / 'src'), str(CODE)]


def descriptor(entrant):
    mode = 'on' if entrant else 'off'
    return {'schema': 'c4prep_launch/1', 'entrant_control': entrant,
            'source_commit': read(CODE / 'SOURCE.json')['commit'],
            'runtime': str(CODE), 'wm': str(WM), 'wm_sha256': WM_SHA,
            'wm_config': str(WM_CONFIG), 'gate': str(GATE),
            'candidates': [{'name': name, 'checkpoint': str(ACTORS[cycle][0]),
                            'sha256': ACTORS[cycle][1]} for name, cycle in (('A', 1), ('B', 3))],
            'base_toml': str(CODE / f'configs/c4_policy14_c4prep_{mode}.toml'),
            'updates': 150, 'gpu_memory_fraction': .45,
            'probes': False, 'critic_bootstrap': False,
            'ready': False, 'blockers': ['final entrant merge/verdict, registered WM3 gate, validated anchor and dry-run receipts required']}


def templates():
    for entrant in (False, True):
        mode = 'on' if entrant else 'off'
        path = OUT / f'CONFIG_{mode.upper()}.json'
        write_new(path, descriptor(entrant))
        print(json.dumps({'config': str(path), 'ready': False}))


def prepare(config, anchor, entrant_receipt):
    """Resolve immutable artifacts only after final integration; exclusive-create."""
    from wms.control.antiexploit import inference_signature
    from wms.control.dreamer import DreamerConfig
    from wms.control.train_cli import _read_config
    template = read(config)
    entrant = template['entrant_control']
    horizon, gate = qualified_gate(entrant)
    final = read(entrant_receipt)
    assert final['task'] == 'S-ENTRANTFIX' and final['final'] is True
    assert final['merged_source_commit'] == read(CODE / 'SOURCE.json')['commit'], 're-merge final entrant tip and restage'
    raw = _read_config(template['base_toml'])
    cfg = DreamerConfig(**raw['trainer'])
    artifact = read(anchor)
    assert artifact['schema'] == 'antiexploit_anchor/1'
    assert artifact['inference_signature'] == inference_signature(cfg)
    assert artifact['credit_baseline'] == 'quantile' and artifact['k'] == 8 and artifact['quantile'] == .1
    assert artifact['validation'] == 'world_disjoint'
    for h in {10, min(60, horizon), horizon}:
        model = artifact['horizons'].get(str(h))
        assert model and model['validated'], f'no validated native anchor at active horizon {h}'
        assert model['training_worlds'] and model['validation_worlds']
        assert not set(model['training_worlds']) & set(model['validation_worlds'])
    from wms.control.policy14_cli import load_members
    from wms.control.dreamer import _model_identity
    members, _ = load_members([str(WM)], [str(WM_CONFIG)], 'cpu')
    assert artifact['world_model_ids'] == [_model_identity(m, cfg.imagination_grad_decay) for m in members]
    for item in template['candidates']:
        assert sha(item['checkpoint']) == item['sha256']
    boundary = read(REL / 'arm/loop/LOOP14_STATE.json')['looplong_boundary']
    # No change to chain state, registration binding, STOP_DRIVER or budget.
    boundary.update(wm={'checkpoint': str(WM), 'sha256': WM_SHA, 'config': str(WM_CONFIG), 'source': 'cycle 3'},
        gate_file=str(GATE), gate_sha256=sha(GATE), horizon_s=horizon,
        previous_horizon_s=horizon, imagination_horizon_s=horizon, accepted_after_cycle=3,
        pairs=str(REL / 'arm/loop/data/pairs_c03.a1'),
        pairing_receipt=str(REL / 'arm/loop/returns/pairing_c03.a1.json'),
        policy_returns=str(REL / 'arm/loop/returns/returns_c03.a1.npz'),
        native_returns=str(REL / 'arm/loop/returns/native_returns_c03.a1.npz'),
        source_policy_checkpoint=str(ACTORS[3][0]), source_policy_sha256=ACTORS[3][1])
    boundary['flags']['hot_swap'] = False
    mode = 'ON' if entrant else 'OFF'
    dest = OUT / f'resolved_{mode}'
    assert not dest.exists(), 'use a new source snapshot/output revision after any failed preparation'
    dest.mkdir()
    write_new(dest / 'BOUNDARY.json', boundary)
    # Both arms use a single-stage, trusted horizon from their first update.
    # Per-penetration safety capping remains authoritative in the trainer.
    text = (f'extends = {json.dumps(template["base_toml"])}\n\n[trainer]\n'
            f'native_anchor_calibration = {json.dumps(str(safe(anchor).resolve()))}\n'
            '\n[horizon]\nenabled = true\n'
            f'horizons_s = [10, {horizon}]\ncurriculum_updates = [0, 1]\n'
            f'entry_pad_max = [32, {max(32, (32*horizon+9)//10)}]\n')
    with (dest / 'POLICY.toml').open('x') as stream:
        stream.write(text)
    result = {**template, 'ready': True, 'blockers': [], 'trusted_horizon_s': horizon,
              'anchor': str(anchor), 'anchor_sha256': sha(anchor),
              'entrant_receipt': str(entrant_receipt), 'entrant_receipt_sha256': sha(entrant_receipt),
              'gate_sha256': sha(GATE), 'policy_config': str(dest / 'POLICY.toml'),
              'boundary': str(dest / 'BOUNDARY.json')}
    result['policy_config_sha256'] = sha(result['policy_config'])
    result['boundary_sha256'] = sha(result['boundary'])
    write_new(dest / 'CONFIG.json', result)
    print(json.dumps({'config': str(dest / 'CONFIG.json'), 'ready_for_dry_run': True}))


def validate(config, *, launching=False):
    doc = read(config)
    if not doc.get('ready'):
        raise ValueError('; '.join(doc['blockers']))
    assert doc['source_commit'] == read(CODE / 'SOURCE.json')['commit']
    horizon, _ = qualified_gate(doc['entrant_control'])
    assert horizon == doc['trusted_horizon_s']
    for key in ('anchor', 'entrant_receipt', 'gate', 'policy_config', 'boundary'):
        assert sha(doc[key]) == doc[key + '_sha256']
    for item in doc['candidates']:
        assert sha(item['checkpoint']) == item['sha256']
    assert doc['updates'] == 150 and doc['gpu_memory_fraction'] == .45
    if launching:
        # MAIN chooses between ON/OFF after the final scientific verdict.
        assert read(doc['entrant_receipt'])['selected_entrant_control'] == doc['entrant_control']
        receipt = read(Path(config).parent / 'DRY_RUN_VERIFIED.json')
        assert receipt['config_sha256'] == sha(config) and receipt['passed']
    return doc


def arguments(doc, item, output, dry):
    argv = [sys.executable, '-B', '-m', 'wms.control.policy14_cli',
        '--config', doc['policy_config'], '--wm-ckpt', str(WM), '--wm-config', str(WM_CONFIG),
        '--real-returns', str(REL / 'arm/loop/returns/returns_c03.a1.npz'),
        '--longcritic-native-returns', str(REL / 'arm/loop/returns/native_returns_c03.a1.npz'),
        '--paircrit-receipt', str(REL / 'arm/loop/returns/pairing_c03.a1.json'),
        '--paircrit-policy-sha256', ACTORS[3][1],
        '--gate-panel', str(REL / 'arm/gate_panel.json'), '--horizon-gate', doc['gate'],
        '--looplong-boundary', doc['boundary'], '--native-init', '--phase-relative-signal-gate',
        '--real-credit', str(REL / 'arm/loop/sumocredit/REAL_CREDIT_c03.a1.json'),
        '--critic-warmup-updates', '0', '--resume', item['checkpoint'],
        '--device', 'cuda', '--seed', '965140', '--max-updates', '150',
        '--output-dir', str(output), '--compact']
    if dry:
        argv += ['--prep-dry-run-updates', '2']
    return argv


def run(config, dry):
    doc = validate(config, launching=not dry)
    import fcntl
    parent = Path(config).parent
    mode = 'DRY_RUN' if dry else 'POLICY'
    root = parent / mode
    root.mkdir(exist_ok=False)
    env = {**os.environ, **read(REL / 'arm/CONFIG_C3_LONGINT.json')['policy']['env'],
           'C4_GPU_MEM_FRACTION': '0.45', 'PYTHONPATH': f'{CODE}/src:{CODE}',
           'C4_ALLOWED_ROOTS': '/root/autodl-tmp/c4',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    children, samples = [], []
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context((OUT / 'candidates.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not dry:
            for name in ('autonomy.lock', 'driver.lock', 'cycle1.lock'):
                handle = stack.enter_context((REL / 'arm' / name).open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (REL / 'arm/loop/STOP_DRIVER').is_file()
        # Reject simultaneous config invocations and record every owned PID.
        for item in doc['candidates']:
            output = root / item['name']
            argv = arguments(doc, item, output, dry)
            log = stack.enter_context((root / (item['name'] + '.log')).open('x'))
            child = subprocess.Popen(argv, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((item, child, output))
            ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            write_new(root / (item['name'] + '_PID.json'),
                      {'pid': child.pid, 'start_ticks': ticks, 'argv': argv, 't': now()})
        while any(child.poll() is None for _, child, _ in children):
            result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                               '--format=csv,noheader,nounits'], text=True)
            owned = {child.pid for _, child, _ in children}
            for line in result.splitlines():
                pid, mib = line.split(',')
                if int(pid) in owned:
                    sample = {'t': now(), 'pid': int(pid), 'bytes': int(mib) * 1024**2}
                    samples.append(sample)
                    with (root / 'gpu_memory.jsonl').open('a') as stream:
                        stream.write(json.dumps(sample) + '\n')
            time.sleep(5)
    results = []
    for item, child, output in children:
        summary = read(output / 'RUN_SUMMARY.json')
        updates = summary.get('updates', [])
        peaks = [r.get('cuda_peak_reserved_bytes', 0) for r in updates]
        passed = (child.returncode == 0 and (not dry or
                  summary['status'] == 'completed_dry_run' and len(updates) == 2))
        results.append({'candidate': item['name'], 'exit': child.returncode, 'passed': passed,
                        'updates': len(updates), 'peak_reserved_bytes': max(peaks, default=0),
                        'summary_sha256': sha(output / 'RUN_SUMMARY.json')})
    receipt = {'config_sha256': sha(config), 'passed': all(r['passed'] for r in results),
               'dry_run': dry, 'results': results, 'process_memory_samples': len(samples),
               'new_sim_s': 0, 'automatic_collection': False}
    write_new(parent / ('DRY_RUN_VERIFIED.json' if dry else 'POLICY_COMPLETE.json'), receipt)
    print(json.dumps(receipt))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('templates', 'prepare', 'dry-run', 'launch', 'validate'))
    p.add_argument('--config')
    p.add_argument('--anchor')
    p.add_argument('--entrant-final-receipt')
    a = p.parse_args()
    if a.action == 'templates':
        templates()
    elif a.action == 'prepare':
        prepare(a.config, a.anchor, a.entrant_final_receipt)
    elif a.action in ('dry-run', 'launch'):
        raise SystemExit(run(a.config, a.action == 'dry-run'))
    else:
        validate(a.config)
