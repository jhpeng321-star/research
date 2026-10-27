"""Owned, detached S-C2H60B operational recovery; original phase ends at383."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent / 'arm'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/c4')]

def drive():
    import loop14 as loop
    from tools.loopresume.c2h60_box import budget_authorization, verify_updates
    from tools.loopresume.continue_c1 import live
    from wms.control.nativeinit import verify_collection_gate

    binding = loop.read_json(RELEASE / 'C2H60B_BINDINGS.json')
    for name, sha in binding['source'].items():
        assert loop.sha256(ROOT / name) == sha, name
    recovery = loop.read_json(RELEASE / 'C2H60B_RECOVERY.json')
    assert loop.sha256(recovery['checkpoint']) == recovery['checkpoint_sha256']
    assert all(not live(p) for p in binding['failed_processes'])
    out = Path(recovery['retry_outputs'][0])
    saved = recovery['saved_update']

    def record(status, **extra):
        doc = {'t': loop.now(), 'status': status, **extra}
        loop.write_atomic(RELEASE / 'C2H60B_PROGRESS.json', doc)
        return doc

    with (ARM / 'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = loop.load_config(ARM / 'CONFIG.json')
        state = loop.load_state(cfg)
        reg = loop.check_registration(cfg)
        assert state['sim_s_charged'] == 176473
        assert state['cycles'][-1]['cycle'] == 2
        assert state['cycles'][2]['phases']['policy']['status'] == 'failed'
        assert state['accepted_wm']['sha256'] == binding['wm_sha256']
        assert loop.sha256(state['accepted_wm']['checkpoint']) == binding['wm_sha256']
        assert loop.sha256(ARM / 'horizon_gate_current.json') == binding['gate_sha256']
        assert not out.exists()
        cfg['policy']['config'] = str(ROOT / 'configs/c4_policy14_c2h60b.toml')
        cfg['policy']['env'].update(C4_GPU_MEM_FRACTION='0.92',
                                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        cfg['c2h60']['operational_retry'] = str(RELEASE / 'C2H60B_BINDINGS.json')
        reg.setdefault('amendments', []).append(binding['amendment'])
        loop.write_atomic(ARM / 'CONFIG.json', cfg)
        reg['sources']['config_sha256'] = loop.sha256(ARM / 'CONFIG.json')
        loop.write_atomic(Path(cfg['registration']), reg)
        state['registration_sha256'] = loop.sha256(cfg['registration'])
        # Build pairing arguments from the original C1source; recovery loads240.
        argv, env = loop.policy_cmd(cfg, state, out)
        argv[argv.index('--resume') + 1] = binding['previous_resume']
        argv[argv.index('--max-updates') + 1] = '150'
        ph = state['cycles'][2]['phases']['policy']
        state['cycles'][2].setdefault('failed_policy_attempts', []).append(copy.deepcopy(ph))
        ph.clear()
        ph.update(status='running', attempt=state['cycles'][2]['failed_policy_attempts'][-1]['attempt'] + 1,
                  started=loop.now(), pid=os.getpid(), task='S-C2H60B', output=str(out),
                  coordinator_start=loop.fast_module().process_token(os.getpid()),
                  updates=150, retained_updates=saved - 233, remaining_updates=383 - saved, horizon_s=60)
        state['stopped'] = None
        loop.save_state(cfg, state)
        event = {'event': 'S-C2H60B_registered', **binding['amendment']}
        for path in (ARM / 'loop/cycle_ledger.jsonl', RELEASE / 'C1_EVENTS.jsonl'):
            loop.append(path, event)
        try:
            with out.with_name(out.name + '.log').open('xb') as log:
                proc = subprocess.Popen(argv, cwd=ROOT,
                    env={**os.environ, **{k: str(v) for k, v in env.items()}},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                child = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid)}
                record('POLICY_RUNNING', child=child, output=str(out), argv=argv,
                       started_unix=time.time(), saved_update=saved, phase_start=233, phase_end=383)
                while proc.poll() is None:
                    sample = {'t': loop.now(), 'unix': time.time(), 'child': child}
                    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                    for line in query.stdout.splitlines():
                        pid, memory = line.split(',')
                        if int(pid.strip()) == proc.pid:
                            sample['used_memory_mib'] = int(memory.strip())
                    metrics = out / 'updates.jsonl'
                    if metrics.exists():
                        lines = metrics.read_text().splitlines()
                        if lines:
                            try:
                                sample['update'] = json.loads(lines[-1])['update']
                            except json.JSONDecodeError:
                                pass
                    loop.append(RELEASE / 'C2H60B_MEMORY.jsonl', sample)
                    time.sleep(5)
            assert proc.returncode == 0, f'policy exit{proc.returncode}; operational pause, inspect summary'
            summary = loop.read_json(out / 'RUN_SUMMARY.json')
            assert summary['status'] == 'completed'
            verify_updates(summary['updates'], 233, 150)
            policy = {'checkpoint': str(out / 'policy_latest.pt')}
            native = verify_collection_gate(policy)
            assert native['wm_sha256'] == [state['accepted_wm']['sha256']]
            assert native['wm_config_sha256'] == [loop.sha256(state['accepted_wm']['config'])]
            assert summary['paircrit_monitor']['source_policy_sha256'] == binding['pairing_source_policy_sha256']
            policy.update(sha256=loop.sha256(policy['checkpoint']), source='cycle2 S-C2H60B same phase',
                          trained_on_wm_sha256=state['accepted_wm']['sha256'])
            cfg = loop.load_config(ARM / 'CONFIG.json')
            state = loop.load_state(cfg)
            assert state['sim_s_charged'] == 176473
            state['policy'] = policy
            state['policy_wm'] = dict(state['accepted_wm'])
            state['cycles'][2]['phases']['policy'].update(status='done', finished=loop.now(),
                receipt={'out': str(out), 'updates': 150, 'native_gate': native,
                         'policy_sha256': policy['sha256'], 'resumed_from': summary['resumed_from']})
            reg = loop.check_registration(cfg)
            authorization = budget_authorization(reg, state['sim_s_charged'], loop.collect_sim_s(cfg))
            state['registration_sha256'] = loop.sha256(cfg['registration'])
            state['stopped'] = None if authorization else 'WAIT_OWNER_BUDGET'
            loop.save_state(cfg, state)
            result = record('BUDGET_AUTHORIZED' if authorization else 'WAIT_OWNER_BUDGET',
                policy=policy, native_gate=native, charged=state['sim_s_charged'],
                cap=reg['budget']['sim_s_total'], collection_need=loop.collect_sim_s(cfg),
                owner_budget_authorization=authorization, collection_dispatched=False)
            loop.write_atomic(RELEASE / 'C2H60B_RESULT.json', result)
            loop.append(ARM / 'loop/cycle_ledger.jsonl', {'event': 'S-C2H60B_phase_complete', **result})
            if not authorization:
                loop.write_atomic(ARM / 'WAIT_OWNER_BUDGET.json', result)
                return
            seeds = loop.allocate_seeds(cfg, state, reg, ARM / 'loop')
            cfg['collect']['startup_canary'] = {
                'cycle': 2, 'next_attempts': {str(seed): 1 for seed in seeds}, 'boxes': ['cpuP', 'cpuQ']}
            loop.write_atomic(ARM / 'CONFIG.json', cfg)
            reg['sources']['config_sha256'] = loop.sha256(ARM / 'CONFIG.json')
            loop.write_atomic(Path(cfg['registration']), reg)
            state['registration_sha256'] = loop.sha256(cfg['registration'])
            loop.save_state(cfg, state)
            for name in ('collect', 'convert', 'wm_ft', 'check'):
                code = loop.run_phase(cfg, name)
                if code:
                    record('OPERATIONAL_PAUSED', phase=name, exit=code)
                    return
            record('CYCLE2_COMPLETE', charged=loop.load_state(cfg)['sim_s_charged'])
        except Exception as error:
            state = loop.load_state(cfg)
            ph = state['cycles'][2]['phases']['policy']
            if ph['status'] == 'running':
                ph.update(status='failed', error=repr(error), finished=loop.now())
                loop.save_state(cfg, state)
            record('OPERATIONAL_PAUSED', error=repr(error), charged=state['sim_s_charged'])
            raise


if __name__ == '__main__':
    drive()
