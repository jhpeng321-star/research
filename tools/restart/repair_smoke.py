"""Prepare explicitly registered executor-contract retry; never launches SUMO."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

RELEASE = Path('/root/autodl-tmp/c4/restart20261026')
ROOT = RELEASE/'runtime'
ARM = RELEASE/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
from wms.control.nativeinit import verify_collection_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    cfg = loop.load_config(ARM/'CONFIG.json')
    state = loop.load_state(cfg)
    old = loop.read_json(ARM/'mini/loop/LOOP14_STATE.json')
    diagnostic = loop.read_json(ARM/'SMOKE_MASK_DIAGNOSIS.json')
    assert old['cycles'][0]['phases']['policy']['status'] == 'done'
    assert old['cycles'][0]['phases']['collect']['status'] == 'failed'
    assert state['sim_s_charged'] == 89321 and old['sim_s_charged'] == 1922
    assert state['policy'] is None and not (ARM/'MINI_PASS.json').exists()
    assert all(w['nonfinite_on_acting_cavs'] == 0 and w['all_policy_parameters_finite']
               for w in diagnostic['worlds'])
    assert all(w['nonfinite_on_unobserved_cavs'] > 0 and all(x == -1 for x in w['unobserved_cav_lanes'])
               for w in diagnostic['worlds'])
    gate = verify_collection_gate(old['policy'])
    assert gate['wm_sha256'] == [state['accepted_wm']['sha256']]
    repair = {'classification': 'junction_absence_sentinel', 'retry_number': 1,
              'checkpoint_sha256': old['policy']['sha256'], 'failed_attempt_charged_sim_s': 1922,
              'new_attempt_reservation_sim_s': 1922, 'total_after_new_charge': 91243,
              'native_base_actual_sim_s_lower_bound': 722,
              'failed_partial_policy_sim_s': 'not measured; retained full charge covers failed attempt',
              'scientific_results_retried': False, 'completed_pairs_in_failed_attempt': 0,
              'authority': 'owner continuation3: up to3 fixes/retries per non-scientific error',
              'diagnosis_sha256': loop.sha256(ARM/'SMOKE_MASK_DIAGNOSIS.json'),
              'repair_source_commit': args.commit}
    for name in ('CONFIG.json', 'REGISTRATION.json', 'PREARM.json', 'prearm', 'prearm.log',
                 'SMOKE_BINDINGS.json', 'SHARED_SMOKE_HASHES_APPENDED.json', 'MINI_SMOKE_PID.json'):
        p = ARM/name
        if p.exists():
            dst = ARM/(name+'.before_junction_fix')
            assert not dst.exists()
            if name in ('CONFIG.json', 'REGISTRATION.json'):
                shutil.copy2(p, dst)
            else:
                p.rename(dst)
    root = ARM/'mini_retry1'
    assert not root.exists()
    cfg['resumeprep'].update(mini_root=str(root), smoke_setup_retry=repair)
    loop.write_atomic(ARM/'CONFIG.json', cfg)
    reg = loop.read_json(cfg['registration'])
    reg['smoke_setup_retry'] = repair
    reg['sources'].update(code_sha=args.commit, config_sha256=loop.sha256(ARM/'CONFIG.json'),
                          source_overlay_sha256=loop.sha256(RELEASE/'SOURCE_OVERLAY.json'))
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = loop.sha256(cfg['registration'])
    loop.save_state(cfg, state)
    one = copy.deepcopy(cfg)
    one.update(loop_root=str(root/'loop'), registration=str(root/'REGISTRATION.json'))
    one['fastloop']['overlap_policy'] = False
    one['policy']['updates_per_cycle'] = 10
    one['collect'].update(worlds_per_cycle=2, demands='1900', penetrations='0.05,0.4', pen_weights='1,1',
                          branch_points=1, branch_first=360, branch_horizon=300, warmup=300,
                          work_root=cfg['collect']['work_root']+'/smoke_retry1')
    for flag in ('sumocredit', 'long_group', 'long_single_hold'):
        one['collect'].pop(flag, None)
    one['wm']['extra_set'] += ['curriculum.stage_steps=[2]']
    mini_reg = copy.deepcopy(reg)
    mini_reg.update(seed_block=[968142, 968143], exclude=[])
    mini_reg['budget'].update(max_cycles=1, sim_s_per_cycle=1922, sim_s_total=1922,
                              wm_steps_per_cycle=2, policy_updates_per_cycle=10)
    root.mkdir()
    loop.write_atomic(root/'REGISTRATION.json', mini_reg)
    loop.write_atomic(root/'CONFIG.json', one)
    loop.init(one, argparse.Namespace(adopt_policy=None))
    retry_state = loop.load_state(one)
    retry_state['policy'] = copy.deepcopy(old['policy'])
    retry_state['cycles'][0]['phases']['policy'] = copy.deepcopy(old['cycles'][0]['phases']['policy'])
    retry_state['cycles'][0]['phases']['policy']['receipt']['executor_only_setup_retry'] = repair
    loop.save_state(one, retry_state)
    loop.write_atomic(ARM/'SMOKE_RETRY_PREPARED.json', repair)
    print(json.dumps({'prepared': repair, 'config_sha256': loop.sha256(ARM/'CONFIG.json'),
                      'registration_sha256': state['registration_sha256'],
                      'new_sumo': 0, 'new_actor_updates': 0, 'historical_charge_retained': True}))


if __name__ == '__main__':
    main()
