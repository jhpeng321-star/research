"""Independent authorized cycle-0 WM work after paired-critic fitting-data refusal.

No exporter retry, actor training, world allocation, collection, or loop advancement.
The original failed conversion and exhausted retry receipts remain authoritative.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RELEASE = Path('/root/autodl-tmp/c4/restart20261026')
ROOT, ARM = RELEASE/'runtime', RELEASE/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402


def inputs():
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    assert state['sim_s_charged'] == 102607 and state['cycles'][-1]['cycle'] == 0
    assert state['cycles'][0]['phases']['convert']['status'] == 'failed'
    assert state['policy']['sha256'] == loop.read_json(RELEASE/'AMENDMENT_C1.json')['cycle0']['policy_sha256']
    assert loop.read_json(ARM/'OPERATIONAL_PAUSED.json')['error'].startswith('OperationalPaused(')
    root = Path(cfg['loop_root'])
    os.environ['C4_ALLOWED_ROOTS'] = str(root)
    pairs = root/'data/pairs_c00.a1'
    manifest = loop.read_json(pairs/'C4_EPISODES_MANIFEST.json')
    assert len(manifest['episodes']) == 16
    assert sorted({e['seed'] for e in manifest['episodes']}) == list(range(968144, 968148))
    from wms.control.real_returns14 import NPZRealReturnProvider
    import torch
    paths = [root/'returns'/name for name in ('returns_c00.a1.npz', 'native_returns_c00.a1.npz')]
    policy, native = [NPZRealReturnProvider(p) for p in paths]
    assert policy._returns.numel() == native._returns.numel() == 8
    assert all(torch.equal(v, native._states[k]) for k, v in policy._states.items())
    assert (policy.provenance['horizon_s'] == 300).all() and (native.provenance['horizon_s'] == 300).all()
    proof = loop.read_json(root/'returns/pairing_c00.a1.json')
    assert proof['policy_checkpoint_sha256'] == state['policy']['sha256']
    assert proof['policy_sha256'] == loop.sha256(paths[0]) and proof['native_sha256'] == loop.sha256(paths[1])
    assert len(proof['pairs']) == 8 and all(p['same_saved_start'] for p in proof['pairs'])
    partition = {s: int(hashlib.sha256(f'965140:{s}'.encode()).hexdigest(), 16) % 5 for s in range(968144, 968148)}
    assert set(partition.values()) == {0}
    converted = {'pairs': str(pairs), 'returns': str(paths[0]), 'native_returns': str(paths[1]),
                 'targets': 8, 'episodes': 16, 'pair_groups': 8,
                 'pairing_not_fit_eligible': True, 'pairing_proof': str(root/'returns/pairing_c00.a1.json')}
    return cfg, state, converted, partition


def diagnose():
    cfg, state, converted, partition = inputs()
    doc = {'t': loop.now(), 'status': 'OPERATIONAL_PAUSED_NO_PAIRED_FITTING_WORLDS',
           'partition_rule': 'sha256(965140:world_seed) % 5 == 0 -> development',
           'partition_remainders': partition, 'fit_worlds': 0, 'development_worlds': 4,
           'real_pairs': 8, 'charged': state['sim_s_charged'], 'converted_outputs': converted,
           'scientific_threshold_or_partition_changed': False, 'export_retries_exhausted': 3,
           'cycle1_launched': False, 'next_action_requires_owner':
           'authorize a prospective fitting-data/partition amendment; current development worlds remain excluded',
           'independent_wm_authority': 'TASK S-LOOPRESUME: fine-tune on four completed worlds, core frozen, same acceptance'}
    loop.write_atomic(RELEASE/'LOOPRESUME_PAIRING_BLOCKED.json', doc)
    print(json.dumps(doc))


def launch():
    inputs()
    assert not (RELEASE/'LOOPRESUME_WM_PID.json').exists()
    with (ARM/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (RELEASE/'loopresume_wm_only.log').open('xb') as log:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__)), 'run'],
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        receipt = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid),
                   't': loop.now(), 'task': 'S-LOOPRESUME independent cycle0 WM/check only',
                   'script_sha256': loop.sha256(__file__)}
        loop.write_atomic(RELEASE/'LOOPRESUME_WM_PID.json', receipt)
    print(json.dumps(receipt))


def run():
    with (ARM/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg, state, converted, _ = inputs()
        snapshot = copy.deepcopy(state)
        root = Path(cfg['loop_root'])
        snapshot['cycles'][0]['phases']['convert']['receipt'] = converted
        doc = {'t': loop.now(), 'status': 'calibration', 'charged': state['sim_s_charged'],
               'converted_outputs': converted, 'actor_launch_blocked': True,
               'state_sha256_before': loop.sha256(loop.state_path(cfg))}
        try:
            local = copy.deepcopy(cfg)
            local['hosts']['cpu'] = local['hosts']['gpu']
            env = {'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT), 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'C4_ALLOWED_ROOTS': str(root)}
            doc['confidence_recalibration'] = loop.convert_sumocredit(local, snapshot,
                str(root/'merged/c00.a1'), 'c00.wm_only', env, root/'logs/calibration_c00.wm_only.log', root)
            doc['status'] = 'wm_ft_running'
            loop.write_atomic(RELEASE/'LOOPRESUME_WM_ONLY.json', doc)
            ft = loop.run_wm_ft(cfg, snapshot, {'attempt': 1}, root)
            snapshot['cycles'][0]['phases']['wm_ft']['receipt'] = ft
            doc.update(status='original_acceptance_running', wm_ft=ft)
            loop.write_atomic(RELEASE/'LOOPRESUME_WM_ONLY.json', doc)
            check = loop.run_check(cfg, snapshot, {'attempt': 1}, root)
            doc.update(status='completed', check=check, accepted_wm=snapshot['accepted_wm'],
                       finished=loop.now(), new_sim_s=0)
            # Persist separate evidence only: failed conversion/retry/stop state is untouched.
            loop.write_atomic(RELEASE/'LOOPRESUME_WM_ONLY.json', doc)
            loop.append(root/'cycle_ledger.jsonl', {'event': 'independent_cycle0_wm_completed', **doc})
        except Exception as error:
            doc.update(status='OPERATIONAL_PAUSED', error=repr(error), finished=loop.now())
            loop.write_atomic(RELEASE/'LOOPRESUME_WM_ONLY.json', doc)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['diagnose', 'launch', 'run'])
    globals()[parser.parse_args().action]()
