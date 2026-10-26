"""Box-local MINIPASS -> launch -> first gates -> portable mirror, own PIDs only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

RELEASE = Path('/root/autodl-tmp/c4/restart20261026')
ROOT = RELEASE/'runtime'
ARM = RELEASE/'arm'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/c4')]
import loop14 as loop  # noqa: E402
from wms.control.nativeinit import verify_collection_gate  # noqa: E402


def live(receipt):
    p = Path('/proc')/str(receipt['pid'])/'stat'
    if not p.exists():
        return False
    fields = p.read_text().rsplit(')', 1)[1].split()
    return fields[0] != 'Z' and fields[19] == receipt['start_ticks']


def event(name, **row):
    loop.append(ARM/'AUTO_EVENTS.jsonl', {'event': name, **row})


def mirror(cfg, state):
    """Copy only this arm/accepted base; original datasets remain on gpuR."""
    snapshot = ARM/'PORTABLE_STATE.json'
    loop.write_atomic(snapshot, state)
    target = '/home/ubuntu/c4/main14/restart20261026/mirror'
    remote = '/home/ubuntu/miniconda3/bin/python -c '+__import__('shlex').quote(
        "import pathlib; p=pathlib.Path('/home/ubuntu/c4/main14/restart20261026/mirror'); p.mkdir(exist_ok=True); (p/'arm').mkdir(exist_ok=True); (p/'base').mkdir(exist_ok=True)")
    subprocess.run(['ssh', 'cpuP', remote], check=True, timeout=60)
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                    str(ARM)+'/', 'cpuP:'+target+'/arm/'], check=True, timeout=600)
    base = Path(state['accepted_wm']['checkpoint']).parent
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
                    str(base)+'/', 'cpuP:'+target+'/base/'], check=True, timeout=300)
    checkpoint = Path(state['policy']['checkpoint']) if state.get('policy') else None
    expected = {'state': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                'base': state['accepted_wm']['sha256']}
    if checkpoint:
        expected['policy'] = state['policy']['sha256']
    rel = str(checkpoint.relative_to(ARM)) if checkpoint else None
    code = f'''import pathlib,json,hashlib
root=pathlib.Path({target!r}); expected={expected!r}
actual={{'state':hashlib.sha256((root/'arm/PORTABLE_STATE.json').read_bytes()).hexdigest(),
 'base':hashlib.sha256((root/'base'/{Path(state['accepted_wm']['checkpoint']).name!r}).read_bytes()).hexdigest()}}
if {rel!r} is not None: actual['policy']=hashlib.sha256((root/'arm'/{rel!r}).read_bytes()).hexdigest()
assert actual==expected, 'portable checkpoint/state SHA mismatch'
print(json.dumps(actual))
'''
    result = subprocess.run(['ssh', 'cpuP', '/home/ubuntu/miniconda3/bin/python -B -'],
                            input=code, text=True, capture_output=True, check=True, timeout=120)
    receipt = {'t': loop.now(), 'host': 'cpuP', 'root': target, 'sha256': json.loads(result.stdout),
               'restore_state': target+'/arm/PORTABLE_STATE.json',
               'path_remap': {str(ARM): target+'/arm', str(base): target+'/base',
                              str(ROOT): '/home/ubuntu/c4/main14/restart20261026/runtime'},
               'datasets': 'original full TRAIN corpora retained on gpuR; mirror is checkpoint+loop state/artifacts'}
    loop.write_atomic(ARM/'MIRROR_RECEIPT.json', receipt)
    event('mirror_verified', **receipt)


def main():
    mini_pid = loop.read_json(ARM/'MINI_SMOKE_PID.json')
    event('waiting_mini', pid=mini_pid)
    while not (ARM/'MINI_PASS.json').exists():
        if (ARM/'STOP_ARM').exists():
            event('owned_stop_before_launch')
            return 0
        if not live(mini_pid):
            loop.write_atomic(ARM/'AUTO_BLOCKED.json', {'reason': 'mini exited without MINI_PASS', 't': loop.now()})
            return 2
        time.sleep(15)
    subprocess.run([sys.executable, '-B', str(RELEASE/'validate_smoke.py')], check=True, timeout=180)
    if loop.read_json(ARM/'SMOKE_RESULT.json')['status'] != 'PASS':
        raise RuntimeError('scientific mechanics smoke validation failed')
    event('mini_and_mechanics_pass')
    if (ARM/'STOP_ARM').exists():
        return 0
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/resumeprep/control.py'), 'launch', '--arm', str(ARM)],
                   check=True, timeout=120)
    pid = loop.read_json(ARM/'PID.json')
    event('main_launched', pid=pid)
    cfg = loop.load_config(ARM/'CONFIG.json')
    first = False
    last_backup = 0.
    while True:
        state = loop.load_state(cfg)
        cycle = state['cycles'][0]
        ph = cycle['phases']['policy']
        if ph['status'] == 'done' and not first:
            policy = state['policy']
            gate = verify_collection_gate(policy)
            panel = loop.read_json(Path(ph['receipt']['out'])/'gate_panel_result.json')
            if not panel['passed'] or not gate['passed']:
                raise RuntimeError('scientific first policy/native gate failed')
            journal = Path(cfg['loop_root'])/'worlds_c00.json'
            guard = loop.read_json(journal).get('native_smoke') if journal.exists() else None
            receipt = {'t': loop.now(), 'status': 'POLICY_PASS_WAIT_REAL_GUARD' if guard is None else
                       'PASS' if guard['passed'] else 'SCIENTIFIC_WAIT_OWNER',
                       'driver': pid, 'driver_same_incarnation': live(pid),
                       'policy': policy, 'wm': state['accepted_wm'], 'updates': ph['receipt']['updates'],
                       'update10': {'passed': panel['passed'], 'mean_delta': panel['mean_delta'], 'n_improved': panel['n_improved']},
                       'native_gate': {'passed': gate['passed'], 'mean_delta': gate['mean_delta'], 'selected_update': gate['update']},
                       'real_guard': guard, 'sim_s_charged': state['sim_s_charged'], 'stopped': state.get('stopped')}
            loop.write_atomic(ARM/'FIRST_GATES.json', receipt)
            if guard is not None:
                first = True
                event('first_real_guard', **receipt)
        if (first or not live(pid)) and time.monotonic()-last_backup > 600:
            for attempt in range(1, 5):
                try:
                    mirror(cfg, state)
                    last_backup = time.monotonic()
                    break
                except (OSError, subprocess.SubprocessError) as error:
                    event('mirror_retry', attempt=attempt, error=str(error))
                    if attempt == 4:
                        loop.write_atomic(ARM/'MIRROR_PAUSED.json', {'t': loop.now(), 'error': str(error)})
                        return 3
                    time.sleep(10)
        if not live(pid) or state.get('stopped'):
            event('main_stopped', stopped=state.get('stopped'), wait_owner=(ARM/'WAIT_OWNER.json').exists())
            return 0
        time.sleep(10)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        event('auto_error', error=repr(error))
        raise
