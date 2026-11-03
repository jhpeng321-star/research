"""Launch3 start gate for the merged V2V + planned-command intent build.

CPU-only metadata module: it refuses to start unless every registered gate
holds, and otherwise only writes receipts. No SUMO, no GPU, no remote calls.
Gates, in order: registered merged commit, COOPFIX_GATE verdict PASS, no HOLD
file, resource gate (disk >= 6 GiB, live mover, mover not stopped), per-arm
config binding/hashing, and the budget re-quote including the intent head
cost within the global ceiling 17471840 sim-s.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

MERGED_COMMIT = '6c68cb7cafecffba8d4d6f8ca683792672769c51'
INTENT_INT_COMMIT = 'b37d895bc1c1114c232c55f3c69e1f60baf47660'
INTENT_HEAD_COMMIT = 'ad7fbc7b92c2a49435e709a282a4182315b99b7c'
V2V_REGISTRATION_COMMIT = '38366196'
CEILING_SIM_S = 17471840
REQUIRED_FREE_BYTES = 6 * 1024 ** 3
GATE_VERDICT = 'PASS'
GATE_NAME = 'COOPFIX_GATE.json'
HOLD_NAME = 'V2V_LAUNCH3_HOLD.json'
STARTED_NAME = 'V2V_STARTED.json'
SUPERSEDED_CHARGE = 811640
DEFAULT_ROOT = '/home/ubuntu/coopmwm20261103_v2v3'
DEFAULT_MOVER = '/home/ubuntu/coopmwm20261103_rolling1647_v3'
ARMS = ('A-v2v', 'B-v2v', 'R-v2v')
INTENT_OPTIONS = dict(v2v=True, v2v_intent=True, v2v_intent_horizon=5,
    v2v_intent_weight=.1)
BANK_OPTIONS = dict(bank_imitation='separate_full_pass', imitation_kl_cap=.045,
    bank_statistics='rule_v2')
SCHEDULE = dict(lessons=dict(A=[], B=[6, 12, 18, 24], R=[]),
    isolated=dict(A=[], B=[6, 12, 18, 24], R=[12, 24]),
    episode_tests=dict(A=[8, 24], B=[8, 24], R=[8, 24]),
    first_update=1, final_update=24,
    pacing_source='registered V2V runtime config (unchanged)')
V2V_PHASES = dict(v2v_train=2073600, v2v_interim=345600, v2v_final=345600,
    v2v_lessons=21840, v2v_isolated=185200)
INTENT_COMPUTE = dict(added_params=9541, base_params=20711, extra_sim_s=0,
    rule=('extra forward/backward only inside already-charged episodes '
          '(actor/critic first layers +33/+48 input columns, zero-init 64x5 '
          'head); no extra SUMO episode, so zero additional sim-s'))


class Refused(Exception):
    """Any refusal to start; carries a machine-readable reason."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def arm_config(arm):
    shared = dict(arms=['A-v2v', 'B-v2v', 'R-v2v'], common_start='original MAPPO u0, zero-init V2V',
        **INTENT_OPTIONS, schedule=json.loads(json.dumps(SCHEDULE)))
    if arm != 'A-v2v':
        shared.update(BANK_OPTIONS)
    return dict(arm=arm, **shared)


def build_launch3_configs():
    return {arm: arm_config(arm) for arm in ARMS}


def identity(cfg):
    return {k: v for k, v in cfg.items() if k != 'arm'}


def config_hashes(configs):
    return {arm: hashlib.sha256(canonical(identity(cfg)).encode()).hexdigest()
        for arm, cfg in configs.items()}


def check_config_binding(configs):
    if sorted(configs) != sorted(ARMS):
        raise Refused('launch3 requires exactly A-v2v, B-v2v, R-v2v configs')
    for arm, cfg in configs.items():
        if not cfg.get('v2v_intent') or int(cfg.get('v2v_intent_horizon', 0)) != 5:
            raise Refused('intent sharing must be on with horizon 5: ' + arm)
    hashes = config_hashes(configs)
    if hashes['B-v2v'] != hashes['R-v2v']:
        raise Refused('B-v2v and R-v2v configs must be identical')
    if identity(configs['A-v2v']) == identity(configs['B-v2v']):
        raise Refused('A-v2v must not carry the B/R bank options')
    for key in BANK_OPTIONS:
        if key in configs['A-v2v']:
            raise Refused('A-v2v must not carry ' + key)
    return dict(hashes=hashes, intent_on=sorted(ARMS),
        bank_arms=['B-v2v', 'R-v2v'], rule='intent ON all arms; imitation/bank B,R only, identical')


def read_gate(path):
    p = Path(path)
    if not p.exists():
        raise Refused('gate file missing: ' + str(p))
    try:
        gate = json.loads(p.read_text(encoding='utf-8'))
    except ValueError as error:
        raise Refused('gate file unreadable: ' + repr(error))
    if gate.get('verdict') != GATE_VERDICT:
        raise Refused('gate verdict is not ' + GATE_VERDICT + ': '
            + repr(gate.get('verdict')))
    return dict(verdict=gate['verdict'], sha256=digest(p), path=str(p),
        unix=gate.get('unix'), rule='refuse unless COOPFIX_GATE verdict == PASS')


def check_hold(control, root):
    for base in (Path(control), Path(root)):
        if (base / HOLD_NAME).exists():
            raise Refused('HOLD file present: ' + str(base / HOLD_NAME))
    return dict(hold_file=HOLD_NAME, held=False,
        rule='launch3 honours an operator HOLD and refuses while it exists')


def mover_alive(mover):
    d = Path(mover) / 'OWNER.json'
    if not d.exists():
        return False
    try:
        record = json.loads(d.read_text())
    except ValueError:
        return False
    try:
        stat = Path('/proc', str(record['pid']), 'stat').read_text().split()
        return (stat[21] == record['start_ticks'] and stat[2] != 'Z'
            and record['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError, KeyError, IndexError):
        return False


def decision(free, alive, stopped):
    return dict(ready=free >= REQUIRED_FREE_BYTES and alive and not stopped,
        disk_free_bytes=free, required_free_bytes=REQUIRED_FREE_BYTES,
        mover_alive=alive, mover_stopped=stopped,
        rule='MAIN2026-11-03 16:27/16:47 additive resource gate (disk>=6GiB, live mover)')


def resource_gate(root, mover):
    import shutil
    free = shutil.disk_usage(root).free
    stopped = (Path(mover) / 'STOP_GPU_SPACE.json').exists()
    return decision(free, mover_alive(mover), stopped)


def ledger_charged(path):
    p = Path(path)
    total = 0
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                total += int(json.loads(line)['sim_s'])
    return SUPERSEDED_CHARGE + total


def requote(charged):
    phases = dict(V2V_PHASES, intent_head=INTENT_COMPUTE['extra_sim_s'])
    additional = sum(V2V_PHASES.values()) + INTENT_COMPUTE['extra_sim_s']
    worst = additional + V2V_PHASES['v2v_train']
    remaining = CEILING_SIM_S - charged
    return dict(authority='MAIN2026-11-03 14:06 cap + launch3 intent re-quote',
        ceiling_sim_s=CEILING_SIM_S, charged_to_date_sim_s=charged,
        remaining_headroom_sim_s=remaining,
        registered_v2v_phases_sim_s=dict(V2V_PHASES),
        intent_head=INTENT_COMPUTE,
        quote_total_sim_s=additional, fits=charged + additional <= CEILING_SIM_S,
        shortfall_sim_s=max(0, charged + additional - CEILING_SIM_S),
        worst_case_compute_bound_sim_s=worst,
        worst_case_shortfall_sim_s=max(0, charged + worst - CEILING_SIM_S),
        worst_case_rule=('disclosure only: even if the whole v2v_train compute '
            'doubled, no additional SUMO episode is charged'),
        ledger_rule='charged = SUPERSEDED_CHARGE 811640 + sum(CHARGES.jsonl sim_s)')


def verify_commit(commit):
    if commit != MERGED_COMMIT:
        raise Refused('commit is not the registered merged commit ' + MERGED_COMMIT
            + ': ' + str(commit))
    git = None
    try:
        import subprocess
        def run(*args):
            return subprocess.run(['git', *args], capture_output=True, text=True,
                timeout=20, cwd=Path(__file__).resolve().parent).returncode == 0
        if run('rev-parse', '--is-inside-work-tree'):
            checks = dict(merged_is_ancestor=run('merge-base', '--is-ancestor', commit, 'HEAD'),
                intent_int_in=run('merge-base', '--is-ancestor', INTENT_INT_COMMIT, commit),
                intent_head_in=run('merge-base', '--is-ancestor', INTENT_HEAD_COMMIT, commit))
            if not all(checks.values()):
                raise Refused('commit ancestry check failed: ' + canonical(checks))
            git = dict(checks, verified=True)
    except (OSError, subprocess.SubprocessError):
        git = None
    return dict(commit=commit, intent_int=INTENT_INT_COMMIT,
        intent_head=INTENT_HEAD_COMMIT, git=git,
        rule='refuse unless the registered merged commit is supplied')


def bind_runtime(value):
    cfg = value.get('config', value)
    if int(cfg.get('budget', {}).get('total', -1)) != CEILING_SIM_S:
        raise Refused('runtime budget total differs from the registered ceiling')
    for key, want in INTENT_OPTIONS.items():
        if cfg.get(key) != want:
            raise Refused(f'runtime option {key} is {cfg.get(key)!r}, want {want!r}')
    for key, want in BANK_OPTIONS.items():
        have = cfg.get(key)
        if have is not None and have != want:
            raise Refused(f'runtime option {key} is {have!r}, want {want!r}')
    schedule = cfg.get('diagnostic_schedule', {})
    for arm, isolated in (('A', []), ('B', [6, 12, 18, 24]), ('R', [12, 24])):
        if list(schedule.get(arm, {}).get('isolated', [])) != isolated:
            raise Refused('runtime diagnostic schedule differs from the registered plan: '
                + arm)
    return dict(bound=True, cap_sim_s=CEILING_SIM_S,
        rule='deployed V2V_RUNTIME config must carry the registered options verbatim')


def preflight(commit, gate, control, root, mover, ledger=None):
    receipt = dict(unix=time.time(), arms=list(ARMS),
        commit=verify_commit(commit), gate=read_gate(gate),
        hold=check_hold(control, root))
    configs = build_launch3_configs()
    receipt['config_binding'] = check_config_binding(configs)
    runtime = Path(root) / 'V2V_RUNTIME.json'
    if runtime.exists():
        receipt['runtime_binding'] = bind_runtime(json.loads(runtime.read_text()))
    receipt['resource'] = resource_gate(root, mover)
    if not receipt['resource']['ready']:
        raise Refused('resource gate not ready: ' + canonical(receipt['resource']))
    charged = ledger_charged(ledger if ledger is not None
        else Path(root) / 'CHARGES.jsonl')
    receipt['budget'] = requote(charged)
    if not receipt['budget']['fits']:
        raise Refused('budget shortfall_sim_s='
            + str(receipt['budget']['shortfall_sim_s']) + ' sim-s'
            + ' (charged ' + str(charged) + ' vs ceiling '
            + str(CEILING_SIM_S) + '); quote_total '
            + str(receipt['budget']['quote_total_sim_s']))
    receipt['ready'] = True
    receipt['new_sim_s'] = 0
    return receipt


def start(commit, gate, control, root, mover, ledger=None):
    root = Path(root)
    started = root / STARTED_NAME
    if started.exists():
        raise Refused(STARTED_NAME + ' already exists; adopt instead of relaunching')
    receipt = preflight(commit, gate, control, root, mover, ledger)
    payload = dict(receipt, stage='launch3_start', first_update=1, final_update=24,
        resumed_after_smoke=None, launcher='tools/coopmwm/v2v_launch3.py',
        launcher_sha256=digest(__file__))
    with started.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    return dict(payload, wrote=str(started))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--commit', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--control', default='/home/ubuntu/coopmwm20261103_v2v_resource1727')
    p.add_argument('--root', default=DEFAULT_ROOT)
    p.add_argument('--mover', default=DEFAULT_MOVER)
    p.add_argument('--ledger', default=None)
    p.add_argument('--mode', choices=('check', 'start'), default='check')
    a = p.parse_args(argv)
    action = start if a.mode == 'start' else preflight
    try:
        value = action(a.commit, a.gate, a.control, a.root, a.mover, a.ledger)
    except Refused as error:
        print(json.dumps(dict(refused=True, reason=str(error), new_sim_s=0)))
        return 2
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
