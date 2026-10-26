"""Scoped box-side S-RESTART setup/verification; never dispatch SUMO here."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
from tools.resumeprep import control  # noqa: E402

RELEASE = ROOT.parent
INPUTS = RELEASE.with_name(RELEASE.name+'_inputs')
ARM = RELEASE/'arm'
CPU_ROOT = '/home/ubuntu/c4/main14/restart20261026'
REJECTED = 'fa2427772a49e9462c0be3172e402c515616a15e2930a263e1eafeacb23a0d5d'


def prepare():
    snapshot = INPUTS/'core'
    old = snapshot/'autodl-tmp/c4/v5/mainstart/mainr16/restart'
    control.prepare(old, ARM, snapshot, shared_gpu=True)
    cfg = loop.read_json(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    p = cfg['policy']
    p.update(config='configs/c4_policy14_restart.toml', longcritic=True,
             initial_real_returns=str(INPUTS/'longcritic/current_real_returns.npz'),
             initial_native_returns=str(INPUTS/'longcritic/paired_native_returns.npz'))
    p['allowed_roots'] += [str(ROOT), str(RELEASE), str(INPUTS), '/root/autodl-tmp/c4/v5/data']
    p['threads'] = 2
    p['env']['OPENBLAS_NUM_THREADS'] = '2'
    p['extra_args'] += ['--longcritic-reliability', str(ROOT/'tools/longcritic/reliability_r16.json')]
    for name in ('cpu', 'cpuP', 'cpuQ'):
        cfg['hosts'][name]['code_root'] = CPU_ROOT+'/runtime'
    for host in cfg['mainstart']['collect_hosts'].values():
        host['code_root'] = CPU_ROOT+'/runtime'
    cfg['collect'].update(work_root=CPU_ROOT+'/work', long_group={'enabled': True},
                          long_single_hold={'enabled': True})
    cfg['wm']['env']['OPENBLAS_NUM_THREADS'] = '4'
    cfg['resumeprep']['operational_retries_per_error'] = 3
    reg = loop.read_json(cfg['registration'])
    reg['budget'].update(policy_updates_per_cycle=30, sim_s_per_cycle=36933,
                         sim_s_total=210000, gpu_h_wm_per_cycle=4.5, wm_steps_per_cycle=40)
    reg['collection'].update(primary_sim_s_per_cycle=36933, charged_sim_s_per_cycle=36933,
                             retry_reserve_sim_s_per_cycle=0, long_attempt_cap_sim_s=8000)
    reg['resources'].update(gpu='gpuA RTX4090D 48GB; main priority/shared round17',
                            policy_cuda_memory_fraction=.20, wm_cuda_memory_fraction=.20,
                            policy_threads=2, wm_threads=4, loop_output=cfg['loop_root'])
    reg['sources'].update(initial_wm_sha256=control.BASE_SHA,
                          source_overlay_sha256=loop.sha256(RELEASE/'SOURCE_OVERLAY.json'))
    reg['restart_amendment'] = {
        'id': 'S-RESTART', 'authority': 'owner renewed continuation after ENOSPC',
        'base_round': 16, 'new_cycle': 0, 'rejected_policy_never_adopted': REJECTED,
        'historical_charged_sim_s': 87399, 'native_residual': True,
        'native_relative_veto_before_adoption_and_collection': True,
        'early_guard_worlds': 4, 'horizon_ceiling_s': 60, 'initial_horizon_s': 10,
        'longwm_loss': False, 'longcritic': True, 'real_share_bootstrap': 1.0,
        'real_share_evidence_sha256': loop.sha256(ROOT/'tools/longcritic/REAL_SHARE_MEASURED.json'),
        'real_share_note': 'owner-approved measured mix; off-policy real critic bootstrap only; rejected actor/critic never loaded; subsequent matched-policy phases remeasure fixed rule',
        'persistent_exploration_s': [30, 60], 'long_primary_sim_s': 7800,
        'long_combined_attempt_cap_sim_s': 8000, 'overlap_policy': True,
        'wm_timeout_h': 4.5, 'wm_max_hours': 4.5,
        'smoke': 'S-C0DIAG-SMOKE-1', 'smoke_seeds': [968142, 968143], 'smoke_sim_s': 1922,
        'operational_retry': 'owner continuation3: three fixes/retries per NON-scientific error; scientific failure or E below300MB stops',
    }
    loop.write_atomic(Path(cfg['registration']), reg)
    state.update(real_returns=p['initial_real_returns'], native_real_returns=p['initial_native_returns'],
                 registration_sha256=loop.sha256(cfg['registration']))
    loop.write_atomic(ARM/'CONFIG.json', cfg)
    loop.save_state(cfg, state)
    # Bind the accepted-WM horizon receipt before MINI_PASS; the driver then
    # republishes the exact same parsed bytes, not a new measurement/config.
    horizon = loop.read_json(ARM/'horizon_gate_current.json')
    if horizon.get('wm_sha256') != control.BASE_SHA:
        raise loop.Refused('restart horizon receipt WM mismatch')
    loop.write_atomic(ARM/'horizon_gate_current.json', horizon)
    loop.write_atomic(ARM/f'horizon_{control.BASE_SHA}.json', horizon)
    # Prior continuation receipts remain history, not an active retry veto after
    # the owner's explicit expanded continuation3 authority.
    shutil.copy2(RELEASE/'OPERATIONAL_RETRY.json', ARM/'LEGACY_OPERATIONAL_RETRY.json')
    if loop.collect_sim_s(cfg) != 36933:
        raise loop.Refused('restart primary sim-s mismatch')
    loop.load_config(ARM/'CONFIG.json')
    loop.load_state(cfg)
    print(json.dumps({'prepared': str(ARM), 'config_sha256': loop.sha256(ARM/'CONFIG.json'),
                      'registration_sha256': loop.sha256(cfg['registration']),
                      'charged': state['sim_s_charged'], 'policy_adopted': state['policy']}))


def inspect():
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    rows = [json.loads(x) for x in (Path(cfg['loop_root'])/'seed_ledger.jsonl').read_text().splitlines()]
    used = set(state['seeds_used'])
    for row in rows:
        used.update(row.get('seeds', []))
        if 'seed' in row:
            used.add(row['seed'])
    rejected = INPUTS/'core/main14_loop/policy/c00.a2/policy_latest.pt'
    if loop.sha256(rejected) != REJECTED:
        raise loop.Refused('rejected predecessor SHA mismatch')
    from wms.control.nativeinit import verify_collection_gate
    try:
        verify_collection_gate({'checkpoint': str(rejected)})
    except (OSError, ValueError, KeyError) as error:
        blocked = str(error)
    else:
        raise loop.Refused('rejected checkpoint unexpectedly passed native-relative veto')
    paths = {'accepted_wm': state['accepted_wm']['checkpoint'],
             'accepted_wm_config': state['accepted_wm']['config'],
             'real_returns': state['real_returns'], 'native_real_returns': state['native_real_returns'],
             **{k: v for k, v in cfg['wm']['env'].items() if k.startswith('WMS_DATA_')},
             **{k: v for k, v in cfg['policy']['env'].items() if k.startswith('WMS_DATA_')}}
    missing = {k: v for k, v in paths.items() if not Path(v).exists()}
    if missing or used & {968142, 968143}:
        raise loop.Refused(f'missing inputs or smoke seeds unavailable: {missing}')
    receipt = {'status': 'PASS', 'rejected_checkpoint_sha256': REJECTED,
               'rejected_checkpoint_blocked': True, 'reason': blocked,
               'smoke_seeds_unused': True, 'historical_charged_sim_s': state['sim_s_charged'],
               'inputs': paths, 'native_dispatch': 'production dreamer/reference use explicit mode strings; no config-object caller',
               'native_residual_executor': 'zero residual abstains; nonzero SUMO car-following vs WM twin finalization remains unverified'}
    loop.write_atomic(ARM/'READINESS.json', receipt)
    print(json.dumps(receipt))


def tests():
    names = ['test_resumeprep.py', 'test_nativeinit.py', 'test_longcritic.py',
             'test_longcritic_followup.py', 'test_loop14.py', 'test_fastloop.py',
             'test_main14.py', 'test_coopw.py', 'test_policy14_critic.py',
             'test_sumocredit.py', 'test_sumocredit_kcheck.py', 'test_c4_dreamer.py',
             'test_loop14c.py', 'test_policy14_integration.py', 'test_policy14e.py',
             'test_policy14_lanes.py', 'test_restart.py']
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT),
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    argv = [sys.executable, '-B', '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider',
            '--basetemp', str(RELEASE/'pytest_integrated'),
            *['tests/unit/'+name for name in names]]
    with (RELEASE/'integrated_tests.log').open('w') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print((RELEASE/'integrated_tests.log').read_text())
    raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'inspect', 'tests'])
    args = p.parse_args()
    {'prepare': prepare, 'inspect': inspect, 'tests': tests}[args.action]()


if __name__ == '__main__':
    main()
