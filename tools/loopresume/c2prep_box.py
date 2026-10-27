"""Detached ACCEPT-bound C2 resource smoke -> policy -> WAIT_OWNER_BUDGET."""
import copy
import fcntl
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]


def drive():
    import loop14 as loop
    from tools.loopresume.continue_c1 import live
    from tools.loopresume.c2prep_common import POLICY_SHA, budget_need, policy_boundary, resource_fits
    import main14_autolaunch as waiter

    def receipt(status, **extra):
        loop.write_atomic(RELEASE/'C2PREP_PROGRESS.json', {'t': loop.now(), 'status': status, **extra})

    binding = loop.read_json(RELEASE/'C2PREP_BINDINGS.json')
    for name, sha in binding['source'].items():
        assert loop.sha256(ROOT/name) == sha, name
    receipt('WAIT_WM1_ACCEPT', wm_driver=binding['wm_driver'])
    # No state/config/register writes while the existing WM/check writer owns its lock.
    while live(binding['wm_driver']):
        time.sleep(30)
    with (ARM/'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = loop.load_config(ARM/'CONFIG.json')
        old = loop.load_state(cfg)
        try:
            assert (RELEASE/'C1COLLECT4_COMPLETE.json').is_file()
            state = policy_boundary(old)
            assert loop.sha256(state['policy']['checkpoint']) == POLICY_SHA
            ft = old['cycles'][1]['phases']['wm_ft']['receipt']
            assert loop.sha256(ft['candidate']) == ft['candidate_sha256']
            converted = old['cycles'][1]['phases']['convert']['receipt']
            assert converted['paircrit_eligibility']['eligible']
            assert loop.sha256(state['pairing_receipt']) == converted['pairing_receipt_sha256']
            for key in ('returns', 'native_returns'):
                assert loop.sha256(converted[key]) == converted[key+'_sha256']
            backup = RELEASE/'c2prep_before'
            backup.mkdir()
            for name in ('CONFIG.json', 'REGISTRATION.json', 'loop/LOOP14_STATE.json', 'PID.json'):
                p = ARM/name
                target = backup/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(p.read_bytes())
            cfg['policy'].update(config='configs/c4_policy14_c2prep.toml', paircrit=True,
                                 longcritic=False, breakdown=False)
            cfg['policy']['env']['C4_GPU_MEM_FRACTION'] = '0.65'
            cfg.setdefault('c2prep', {}).update(policy_only=True, collection_authorized=False,
                                               after_gate='WAIT_OWNER_BUDGET')
            reg = loop.check_registration(cfg)
            assert reg['budget']['sim_s_total'] == 210000
            need = budget_need(state['sim_s_charged'], loop.collect_sim_s(cfg), reg['budget']['sim_s_total'])
            assert need == budget_need()
            amendment = {**binding['amendment'], 'registered_at': loop.now(),
                         'accepted_wm': state['accepted_wm'], 'budget': need,
                         'previous_registration_sha256': loop.sha256(cfg['registration']),
                         'previous_budget_stop': old['stopped']}
            reg.setdefault('amendments', []).append(amendment)
            loop.write_atomic(ARM/'CONFIG.json', cfg)
            reg['sources']['config_sha256'] = loop.sha256(ARM/'CONFIG.json')
            loop.write_atomic(Path(cfg['registration']), reg)
            state['registration_sha256'] = loop.sha256(cfg['registration'])
            loop.save_state(cfg, state)
            loop.append(ARM/'loop/cycle_ledger.jsonl', {'event': 'S-C2PREP_registered', **amendment})
            loop.write_atomic(ARM/'PID_C2PREP.json', {
                'pid': os.getpid(), 'start_ticks': loop.fast_module().process_token(os.getpid()), 't': loop.now()})
            receipt('HORIZON_CHECK', wm1_check=old['cycles'][1]['phases']['check']['receipt'])
            waiter.prepare_horizon(cfg, state, ARM)
            gate = loop.read_json(ARM/'horizon_gate_current.json')
            assert gate['wm_sha256'] == state['accepted_wm']['sha256']
            available = [10]
            for horizon in (30, 60):
                if gate.get('horizons', {}).get(str(horizon), {}).get('verdict') != 'PASS':
                    break
                available.append(horizon)
            import torch
            checkpoint = torch.load(state['policy']['checkpoint'], map_location='cpu', weights_only=True)
            assert checkpoint['update_count'] == 180
            assert checkpoint['coopw']['anchor'] == 0
            del checkpoint
            updates = loop.measured_policy_updates(cfg, state)
            phase_started = time.monotonic()
            smokes, selected = [], None
            # Gate failures and actual cap/time failures are recorded, never overridden.
            for horizon in reversed(available):
                smoke_cfg = copy.deepcopy(cfg)
                smoke_out = ARM/f'loop/policy/c02_resource_h{horizon}'
                args, env = loop.policy_cmd(smoke_cfg, state, smoke_out)
                args[1:3] = ['-B', str(ROOT/'tools/loopresume/c2prep_smoke.py'),
                             '--horizon', str(horizon), '--receipt', str(RELEASE/f'C2PREP_SMOKE_{horizon}.json')]
                receipt('RESOURCE_SMOKE', horizon_s=horizon, accepted_wm=state['accepted_wm'])
                code = loop.run_cmd(loop.host(cfg, 'gpu'), args, env=env,
                    log=smoke_out.with_suffix('.log'), timeout_s=10800-(time.monotonic()-phase_started))
                smoke_path = RELEASE/f'C2PREP_SMOKE_{horizon}.json'
                smoke = loop.read_json(smoke_path) if smoke_path.exists() else {
                    'status': 'FAIL', 'error': f'smoke exit{code} produced no receipt', 'horizon_s': horizon}
                smokes.append(smoke)
                if resource_fits(smoke, updates, 10800-(time.monotonic()-phase_started)+smoke.get('wall_seconds', 0)):
                    selected = horizon
                    break
            assert selected is not None, 'no registered horizon fits measured resources/time'
            selection = {'t': loop.now(), 'horizon_s': selected, 'available': available,
                         'gate_sha256': loop.sha256(ARM/'horizon_gate_current.json'), 'smokes': smokes,
                         'updates': updates, 'wm1_check': old['cycles'][1]['phases']['check']['receipt']}
            if selected != 60:
                selection['reason'] = 'current accepted-WM gate or measured allocator/time budget; see gate and smoke receipts'
                # Preserve the original stages/pads but truncate unavailable stages.
                hs = [h for h in (10, 30, 60) if h <= selected]
                override = ROOT/'configs/c4_policy14_c2prep_resource.toml'
                override.write_text('extends = "c4_policy14_c2prep.toml"\n[horizon]\n'
                    +f'horizons_s = {hs}\ncurriculum_updates = {[0,150,190][:len(hs)]}\n'
                    +f'entry_pad_max = {[32,83,149][:len(hs)]}\n')
                cfg['policy']['config'] = str(override)
                selection['resource_config_sha256'] = loop.sha256(override)
            # Bind measured selection prospectively before the full phase.
            cfg['policy']['timeout_h'] = (10800-(time.monotonic()-phase_started))/3600
            assert cfg['policy']['timeout_h'] > 0
            reg = loop.check_registration(cfg)
            reg.setdefault('amendments', []).append({'id': 'S-C2PREP_RESOURCE', **selection})
            loop.write_atomic(ARM/'CONFIG.json', cfg)
            reg['sources']['config_sha256'] = loop.sha256(ARM/'CONFIG.json')
            loop.write_atomic(Path(cfg['registration']), reg)
            state['registration_sha256'] = loop.sha256(cfg['registration'])
            loop.save_state(cfg, state)
            loop.write_atomic(RELEASE/'C2PREP_SELECTION.json', selection)
            receipt('POLICY_RUNNING', horizon_s=selected, updates=updates)
            code = loop.run_phase(cfg, 'policy')
            assert code == 0, f'policy exit {code}; no scientific retry'
            state = loop.load_state(cfg)
            from wms.control.nativeinit import verify_collection_gate
            native = verify_collection_gate(state['policy'])
            assert native['wm_sha256'] == [state['accepted_wm']['sha256']]
            assert native['wm_config_sha256'] == [loop.sha256(state['accepted_wm']['config'])]
            summary = loop.read_json(Path(state['policy']['checkpoint']).with_name('RUN_SUMMARY.json'))
            assert summary['paircrit_monitor']['source_policy_sha256'] == POLICY_SHA
            horizons = [row['imagination_horizon_s'] for row in summary['updates']]
            assert max(horizons) <= selected <= 60
            assert selected in horizons, 'selected training horizon was never exercised'
            assert state['sim_s_charged'] == old['sim_s_charged']
            assert all(state['cycles'][2]['phases'][n]['status'] == 'pending'
                       for n in ('collect', 'convert', 'wm_ft', 'check'))
            state['stopped'] = 'WAIT_OWNER_BUDGET'
            loop.save_state(cfg, state)
            result = {'t': loop.now(), 'status': 'WAIT_OWNER_BUDGET', 'wm1_check': selection['wm1_check'],
                      'selection': selection, 'native_gate': native, 'policy': state['policy'],
                      'paircrit_monitor': summary['paircrit_monitor'], 'critic_warmup': summary['critic_warmup'],
                      'actual_training_horizons': {str(h): horizons.count(h) for h in sorted(set(horizons))},
                      'budget': need, 'collection_dispatched': False,
                      'registration_sha256': state['registration_sha256']}
            loop.write_atomic(RELEASE/'C2PREP_RESULT.json', result)
            loop.write_atomic(ARM/'WAIT_OWNER_BUDGET.json', result)
            loop.append(ARM/'loop/cycle_ledger.jsonl', {'event': 'S-C2PREP_WAIT_OWNER_BUDGET', **result})
            receipt('WAIT_OWNER_BUDGET', budget=need, horizon_s=selected, native_gate=native)
        except Exception as error:
            receipt('OPERATIONAL_PAUSED', error=repr(error), charged=loop.load_state(cfg)['sim_s_charged'])
            raise


if __name__ == '__main__':
    drive()
