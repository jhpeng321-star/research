#!/usr/bin/env python3
"""Owner-amended MAIN14 launcher; imports the immutable native-lane runtime."""
from __future__ import annotations

import argparse
import copy
try:
    import fcntl
except ImportError:
    fcntl = None
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

V5 = Path('/root/autodl-tmp/c4/v5')
CODE = V5 / 'repo_main14'
BASE_SHA = '952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8'
ARM = V5 / 'mainstart'


def identity(pid):
    try:
        f = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if f[0] == 'Z' else {'ppid': int(f[1]), 'start_ticks': f[19]}
    except FileNotFoundError:
        return None


def allowed_pairft(pid, registered, lookup=identity):
    """Only a live exact registered incarnation, or its current descendants."""
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        actual = lookup(pid)
        if actual is None:
            return False
        if str(pid) in registered:
            return actual['start_ticks'] == registered[str(pid)]
        pid = actual['ppid']
    return False


def gpu_rows_allowed(text, registered, lookup=identity):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pid = int(line.split(',', 1)[0])
        if not allowed_pairft(pid, registered, lookup):
            raise RuntimeError(f'unregistered CUDA process: {line}')
        rows.append(line.strip())
    return rows


def choose_swap(reference, candidates, choose_base, strict_ratio):
    """Reuse MAIN14's frozen scientific vetoes; lower eligible TTC wins."""
    eligible, decisions = [], []
    for item in candidates:
        try:
            decision = choose_base(reference, item['qualification'])
            ok = decision['base'] == '14b' and item['actual_sha256'] == item['scored_sha256']
            decisions.append({'label': item['label'], 'eligible': ok, **decision})
            if ok:
                eligible.append(item)
        except (KeyError, TypeError, ValueError) as error:
            decisions.append({'label': item.get('label'), 'eligible': False, 'error': str(error)})
    selected = min(eligible, key=lambda x: (strict_ratio(x['qualification']), x['label'] != '14b')) if eligible else None
    return selected, decisions


def improves_selected_base(chosen, previous, strict_ratio):
    """Do not reset an accepted fine-tune back to an already adopted external FINAL."""
    if chosen is None:
        return False
    if not previous:
        return True
    return (chosen['actual_sha256'] != previous['sha256']
            and strict_ratio(chosen['qualification']) < previous['strict_ttc_ratio'])


def remote_json(host, python, script, args=()):
    command = ' '.join(map(shlex.quote, [python, '-c', script, *args]))
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, command],
                       capture_output=True, text=True, timeout=90, check=False)
    if p.returncode:
        raise RuntimeError(f'{host}: {p.stderr[-500:]}')
    return json.loads(p.stdout)


READINESS = """
import hashlib,json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
receipt=root.parent/'COLLECTION_READY.json'
assert receipt.is_file(), 'collector readiness receipt absent'
r=json.loads(receipt.read_text()); assert r['source_manifest_sha256']==expected
for rel,sha in r['source_files'].items():
 p=root/rel; assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, rel
assert r['fixture_verified_without_new_sumo'] is True
import torch,numpy,libsumo,traci,sumolib
from traffic_shepherd.config import load_config
c=load_config(); assert c.network_path.is_file() and c.scenario_template_path.is_file()
mem=dict(x.split(':',1) for x in pathlib.Path('/proc/meminfo').read_text().splitlines())
available=int(mem['MemAvailable'].split()[0])*1024
workers=min(12,max(0,int((available-12*1024**3)/(1.8*1024**3))))
assert workers>=1, 'insufficient RAM headroom alongside scoring'
print(json.dumps({'ready':True,'workers':workers,'available_bytes':available,'receipt':r}))
"""


def collection_ready(cfg, arm, loop):
    errors = {}
    for alias in ('cpuP', 'cpuQ'):
        h = dict(cfg['mainstart']['collect_hosts'][alias])
        # Import paths are explicit; no startup shell files or assignment files are opened here.
        script = "import sys; sys.path[:0]=[sys.argv[1]+'/src',sys.argv[1]];\n" + READINESS
        try:
            ready = remote_json(alias, h['python'], script,
                                (h['code_root'], arm['native_source_manifest_sha256']))
            loop.disk_guard(cfg, h, cfg['collect']['work_root'], 13 * 60 / 1024)
            cfg['hosts']['cpu'] = h
            cfg['collect']['workers'] = min(12, int(ready['workers']))
            return {'host': alias, **ready}
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired, loop.Refused) as error:
            errors[alias] = str(error)
    return {'ready': False, 'errors': errors}


def boundary_swap(cfg, state, arm_dir, loop, verdict):
    """Packages are staged by the scoring handoff; absent evidence never selects a base."""
    cycle = state['cycles'][-1]['cycle']
    receipt = arm_dir / f'BASE_BOUNDARY_c{cycle:02d}.json'
    if receipt.exists():
        return
    package = arm_dir / 'BASE_CANDIDATES.json'
    record = {'cycle': cycle, 't': loop.now(), 'decision': 'retain', 'previous_wm': state['accepted_wm'],
              'reason': 'complete scored FINAL packages not yet staged', 'decisions': []}
    if package.is_file():
        doc = loop.read_json(package)
        ref = verdict.select_label(loop.read_json(doc['reference_qualification']), 'r13e')
        if doc['reference_checkpoint_sha256'] != BASE_SHA:
            raise RuntimeError('base-swap reference checkpoint mismatch')
        candidates = []
        for row in doc['candidates']:
            if row['label'] not in ('14b', 'round16'):
                raise RuntimeError('unregistered base candidate')
            actual = loop.sha256(row['checkpoint'])
            qualification = verdict.select_label(loop.read_json(row['qualification_path']), row['score_label'])
            candidates.append({**row, 'actual_sha256': actual, 'scored_sha256': row['scored_checkpoint_sha256'],
                               'qualification': qualification})
        chosen, record['decisions'] = choose_swap(ref, candidates, verdict.choose_base, verdict.strict_ratio)
        record.update(package_sha256=loop.sha256(package), reason='registered no-new-FAIL / strict-TTC comparison')
        if improves_selected_base(chosen, state.get('mainstart_selected_base'), verdict.strict_ratio):
            if not Path(loop.guard(chosen['config'])).is_file():
                raise RuntimeError('selected FINAL config missing')
            state['accepted_wm'] = {'checkpoint': chosen['checkpoint'], 'config': chosen['config'],
                                    'sha256': chosen['actual_sha256'], 'source': f"boundary swap {chosen['label']}"}
            state['mainstart_selected_base'] = {'sha256': chosen['actual_sha256'], 'label': chosen['label'],
                                                'strict_ttc_ratio': verdict.strict_ratio(chosen['qualification'])}
            loop.save_state(cfg, state)
            record.update(decision='swap', selected=chosen['label'], accepted_wm=state['accepted_wm'])
    loop.write_atomic(receipt, record)
    loop.append(Path(cfg['loop_root']) / 'cycle_ledger.jsonl', {'event': 'base_boundary', **record})


def main():
    sys.path[:0] = [str(CODE / 'scripts/c4'), str(CODE), str(CODE / 'src')]
    import main14_autolaunch as auto
    import loop14 as loop
    from tools.main14 import verdict_snapshot as verdict

    p = argparse.ArgumentParser()
    p.add_argument('--arm', default=str(ARM / 'ARMED.json'))
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    arm = loop.read_json(args.arm)
    arm_dir = Path(args.arm).parent
    cfg = loop.load_config(arm['config'])
    auto.verify_source(arm)
    loop.check_registration(cfg)
    if loop.sha256(__file__) != arm['wrapper_sha256']:
        raise RuntimeError('MAINSTART wrapper source mismatch')
    if identity(657216) and identity(657216)['start_ticks'] == '4159403954':
        raise RuntimeError('original waiter has not exited by itself')
    if not (V5 / 'main14_arm/STOP_ARM').is_file():
        raise RuntimeError('documented original disarm marker missing')
    original_guard = loop.gpu_guard
    context = {'phase': 'prelaunch'}

    def concurrent_guard(config, host):
        if context['phase'] not in ('policy', 'prelaunch'):
            return original_guard(config, host)
        command = ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader']
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
        try:
            rows = gpu_rows_allowed(result.stdout, arm['pairft_incarnations'])
        except (RuntimeError, ValueError) as error:
            raise loop.GpuBusy(str(error)) from error
        return {'compute_apps': rows, 'policy_memory_fraction': 0.45, 'registered_concurrency': 'S-PAIRFT only'}

    loop.gpu_guard = concurrent_guard

    def drive(config, directory):
        while True:
            if (directory / 'STOP_ARM').exists() or (Path(config['loop_root']) / 'STOP_DRIVER').exists():
                auto.say('graceful_stop')
                return
            state = loop.load_state(config)
            if state.get('stopped'):
                auto.say('loop_stopped', reason=state['stopped'], sim_s=state['sim_s_charged'])
                return
            phase = loop._phase(state)['name']
            cycle = state['cycles'][-1]['cycle']
            context['phase'] = phase
            if phase == 'policy':
                if cycle >= 1:
                    boundary_swap(config, state, directory, loop, verdict)
                    state = loop.load_state(config)
                auto.prepare_horizon(config, state, directory)
            if phase == 'collect':
                ready = collection_ready(config, arm, loop)
                loop.write_atomic(directory / 'COLLECT_READINESS.json', {'t': loop.now(), **ready})
                if not ready['ready']:
                    auto.say('collect_wait_no_seeds_charged', cycle=cycle, sim_s=state['sim_s_charged'], **ready)
                    time.sleep(120)
                    continue
                loop.write_atomic(directory / f'COLLECT_HOST_c{cycle:02d}.json', ready)
                loop.write_atomic(V5 / 'main14_launch_config.json', config)
            auto.say('phase_start', cycle=cycle, phase=phase)
            code = loop.run_phase(config, phase)
            if code == 5:
                auto.say('gpu_busy_wait', seconds=120)
                time.sleep(120)
                continue
            if code not in (0, 4):
                raise RuntimeError(f'phase {phase} exit {code}; no automatic retry')
            # Keep converted caches: this task never recursively removes data directories.

    auto.drive = drive
    if args.check_only:
        concurrent_guard(cfg, loop.host(cfg, 'gpu'))
        print(json.dumps({'verified': True, 'base_sha256': loop.sha256(cfg['wm']['initial_wm']),
                          'collector': collection_ready(copy.deepcopy(cfg), arm, loop)}))
        return
    if fcntl is None:
        raise RuntimeError('detached driver requires Linux flock')
    with (arm_dir / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        loop.write_atomic(arm_dir / 'PID.json', {'pid': os.getpid(), 'sid': os.getsid(0), **identity(os.getpid())})
        try:
            if args.resume:
                state = loop.load_state(cfg)
                if (arm_dir / 'WAIT_OWNER.json').exists() or state.get('stopped') or loop._phase(state)['status'] != 'pending':
                    raise RuntimeError('continuation requires a successful phase boundary, no stop/failure')
                loop.write_atomic(arm_dir / 'CONTINUED.json', {'t': loop.now(), 'pid': os.getpid(),
                                  'next': loop._phase(state)['name'], 'registration_sha256': state['registration_sha256'],
                                  'wrapper_sha256': arm['wrapper_sha256']})
                drive(cfg, arm_dir)
                return
            if loop.disk_guard(cfg, loop.host(cfg, 'gpu'), cfg['loop_root'], 0)['free_gb'] < 6:
                raise RuntimeError('launch output filesystem has less than 6 GiB free')
            if loop.sha256(cfg['wm']['initial_wm']) != BASE_SHA:
                raise RuntimeError('13E FINAL hash mismatch')
            selected = {'ready': True, 'base': '13e_b2', 'amendment': 'S-MAINSTART',
                        'reason': 'cpuX spot reclaimed; owner requested immediate start',
                        'reference_checkpoint_sha256': BASE_SHA,
                        'scored_checkpoint_sha256': loop.sha256(V5 / 'runs/c4_wm_v10_round14_b/wm_s.pt')}
            auto.launch(cfg, arm, selected, arm_dir)
        except Exception as error:
            loop.write_atomic(arm_dir / 'WAIT_OWNER.json', {'t': loop.now(), 'error': f'{type(error).__name__}: {error}'})
            auto.say('wait_owner', error=str(error))
            raise


if __name__ == '__main__':
    main()
