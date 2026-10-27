"""S-C2H60: detached, SHA-bound 150-update60s phase and native/budget gates."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE, ARM = ROOT.parent, ROOT.parent / 'arm'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/c4')]


def verify_updates(rows, start, expected=150):
    if len(rows) != expected or [r['update'] for r in rows] != list(range(start + 1, start + expected + 1)):
        raise ValueError('wrong logical update count/sequence')
    for row in rows:
        if not all(math.isfinite(row[k]) for k in ('actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm')):
            raise ValueError('nonfinite training loss/gradient')
        if row['imagination_horizon_s'] not in (10, 60) or row['imagination_horizon_stage'] != 2:
            raise ValueError('60s stage not active (10s allowed only by inherited low-share mask)')
    if not any(row['imagination_horizon_s'] == 60 for row in rows):
        raise ValueError('60s training never exercised')


def budget_authorization(reg, charged, need):
    """A cap alone cannot override the policy-only authority: require owner/Markdown binding."""
    cap = reg['budget']['sim_s_total']
    for item in reversed(reg.get('amendments', [])):
        if (item.get('id') == 'S-C2H60_OWNER_BUDGET_RAISE'
                and item.get('owner_approved') is True
                and item.get('sim_s_total') == cap and cap > 210000
                and item.get('markdown_registration_sha256')
                and item.get('collection_authorized') is True
                and charged + need <= cap):
            return item
    return None


def drive():
    import loop14 as loop
    from tools.loopresume.continue_c1 import live
    from wms.control.nativeinit import verify_collection_gate

    def record(status, **extra):
        doc = {'t': loop.now(), 'status': status, **extra}
        loop.write_atomic(RELEASE / 'C2H60_PROGRESS.json', doc)
        return doc

    binding = loop.read_json(RELEASE / 'C2H60_BINDINGS.json')
    for name, sha in binding['source'].items():
        assert loop.sha256(ROOT / name) == sha, name
    assert not live(binding['superseded']['coordinator'])
    assert not live(binding['superseded']['child'])
    assert loop.sha256(binding['warm_start']['checkpoint']) == binding['warm_start']['sha256']
    with (ARM / 'cycle1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = loop.load_config(ARM / 'CONFIG.json')
        state = loop.load_state(cfg)
        assert state['sim_s_charged'] == 176473
        assert state['cycles'][-1]['cycle'] == 2
        assert state['cycles'][2]['phases']['policy']['status'] == 'failed'
        assert state['accepted_wm']['sha256'] == binding['wm_sha256']
        assert loop.sha256(state['accepted_wm']['checkpoint']) == binding['wm_sha256']
        assert loop.sha256(ARM / 'horizon_gate_current.json') == binding['gate_sha256']
        backup = RELEASE / f'c2h60_before_{os.getpid()}'
        backup.mkdir(exist_ok=False)
        for name in ('CONFIG.json', 'REGISTRATION.json', 'loop/LOOP14_STATE.json'):
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ARM / name).read_bytes())
        reg = loop.check_registration(cfg)
        cfg['policy']['config'] = str(ROOT / 'configs/c4_policy14_c2h60.toml')
        # The inherited numerical timeout remains historical; this coordinator never applies it.
        cfg['c2h60'] = {'horizon_s': 60, 'updates': 150, 'time_limit_waived': True,
                        'after_gate': 'WAIT_OWNER_BUDGET', 'binding': str(RELEASE / 'C2H60_BINDINGS.json')}
        reg.setdefault('amendments', []).append(binding['amendment'])
        loop.write_atomic(ARM / 'CONFIG.json', cfg)
        reg['sources']['config_sha256'] = loop.sha256(ARM / 'CONFIG.json')
        loop.write_atomic(Path(cfg['registration']), reg)
        state['registration_sha256'] = loop.sha256(cfg['registration'])
        # Build the paired-return arguments using their real C1source policy, NOT the warm actor.
        argv, env = loop.policy_cmd(cfg, state, ARM / 'loop/policy/c02.h60')
        argv[argv.index('--resume') + 1] = binding['warm_start']['checkpoint']
        argv[argv.index('--max-updates') + 1] = '150'
        out = ARM / 'loop/policy/c02.h60'
        assert not out.exists()
        ph = state['cycles'][2]['phases']['policy']
        historical = copy.deepcopy(ph)
        state['cycles'][2].setdefault('superseded_policy_phases', []).append({
            **historical, **binding['superseded'], 'superseded_by': 'S-C2H60'})
        ph.clear()
        ph.update(status='running', attempt=historical['attempt'] + 1, started=loop.now(), pid=os.getpid(),
                  coordinator_start=loop.fast_module().process_token(os.getpid()),
                  output=str(out), task='S-C2H60', updates=150, horizon_s=60)
        state['stopped'] = None
        loop.save_state(cfg, state)
        loop.append(ARM / 'loop/cycle_ledger.jsonl', {'event': 'S-C2H60_registered', **binding['amendment']})
        loop.append(RELEASE / 'C1_EVENTS.jsonl', {'event': 'S-C2H60_registered', **binding['amendment']})
        try:
            with out.with_name(out.name + '.log').open('xb') as log:
                proc = subprocess.Popen(argv, cwd=ROOT,
                    env={**os.environ, **{k: str(v) for k, v in env.items()}},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                child = {'pid': proc.pid, 'start_ticks': loop.fast_module().process_token(proc.pid)}
                record('POLICY_RUNNING', child=child, output=str(out), warm_start=binding['warm_start'],
                       started_unix=time.time(), argv=argv, time_limit_waived=True)
                samples = []
                while proc.poll() is None:
                    row = {'t': loop.now(), 'unix': time.time(), 'child': child}
                    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                    for line in query.stdout.splitlines():
                        pid, memory = line.split(',')
                        if int(pid.strip()) == proc.pid:
                            row['used_memory_mib'] = int(memory.strip())
                    metrics = out / 'updates.jsonl'
                    if metrics.exists():
                        lines = metrics.read_text().splitlines()
                        if lines:
                            try:
                                last = json.loads(lines[-1])
                                row.update(update=last['update'], horizon_s=last['imagination_horizon_s'])
                            except json.JSONDecodeError:
                                pass  # writer may be appending; next sample is authoritative
                    loop.append(RELEASE / 'C2H60_MEMORY.jsonl', row)
                    samples.append(row)
                    time.sleep(10)
                assert proc.returncode == 0, f'policy exit{proc.returncode}; no automatic scientific retry'
            summary = loop.read_json(out / 'RUN_SUMMARY.json')
            assert summary['status'] == 'completed'
            verify_updates(summary['updates'], binding['warm_start']['update'])
            policy = {'checkpoint': str(out / 'policy_latest.pt')}
            native = verify_collection_gate(policy)
            assert native['wm_sha256'] == [state['accepted_wm']['sha256']]
            assert native['wm_config_sha256'] == [loop.sha256(state['accepted_wm']['config'])]
            assert summary['paircrit_monitor']['source_policy_sha256'] == binding['pairing_source_policy_sha256']
            policy.update(sha256=loop.sha256(policy['checkpoint']), source='cycle2 S-C2H60',
                          trained_on_wm_sha256=state['accepted_wm']['sha256'])
            # Re-read registration after training so a prospective owner budget amendment is visible.
            cfg = loop.load_config(ARM / 'CONFIG.json')
            state = loop.load_state(cfg)
            assert state['sim_s_charged'] == 176473
            state['policy'] = policy
            ph = state['cycles'][2]['phases']['policy']
            ph.update(status='done', finished=loop.now(), receipt={
                'out': str(out), 'exit': 0, 'updates': 150, 'resumed_from': summary['resumed_from'],
                'native_gate': native, 'policy_sha256': policy['sha256']})
            reg = loop.check_registration(cfg)
            authorization = budget_authorization(reg, state['sim_s_charged'], loop.collect_sim_s(cfg))
            state['registration_sha256'] = loop.sha256(cfg['registration'])
            state['stopped'] = None if authorization else 'WAIT_OWNER_BUDGET'
            loop.save_state(cfg, state)
            result = record('BUDGET_AUTHORIZED' if authorization else 'WAIT_OWNER_BUDGET',
                policy=policy, native_gate=native, charged=state['sim_s_charged'],
                cap=reg['budget']['sim_s_total'], collection_need=loop.collect_sim_s(cfg),
                owner_budget_authorization=authorization, collection_dispatched=False)
            loop.write_atomic(RELEASE / 'C2H60_RESULT.json', result)
            loop.append(ARM / 'loop/cycle_ledger.jsonl', {'event': 'S-C2H60_phase_complete', **result})
            if not authorization:
                loop.write_atomic(ARM / 'WAIT_OWNER_BUDGET.json', result)
                return
            # Existing production collection path supplies serial canary barriers and charge-before-dispatch.
            seeds = loop.allocate_seeds(cfg, state, reg, ARM / 'loop')
            cfg['collect']['startup_canary'] = {
                'cycle': 2, 'next_attempts': {str(seed): 1 for seed in seeds},
                'boxes': ['cpuP', 'cpuQ']}
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
