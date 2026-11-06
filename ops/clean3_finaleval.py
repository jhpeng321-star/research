"""clean3 final paired evaluation: one gated launcher, zero-physics dry modes, paper tables.

Registered contract (clean3 CONFIG ``final_evaluation`` == ``rerun_cpus.FINAL_EVALUATION``):
A/B/R u24 on the 96 explicit TRAIN worlds of the root's ISOLATED_REGISTRATION.json (file SHA
bound by CONFIG ``isolated_registration_sha256``), full 1200 s windows, J_veh_s (lower is better),
whole-world paired percentile bootstrap 10000 reps / seed 20261104 / 95 %, safety counters
collisions/teleports/hard_brakes/ttc_events, phase ``v2v_final``; ``final_paired_branches`` is
false (episode-only). The A/B/R wave is dispatched by the EXISTING clean3 monitor
(launch_training.run_monitor -> launch_evaluation.scheduled(cfg, 24) ->
evaluation/PAIRED_u024.json). This launcher never runs, replaces or duplicates it.

It adds frozen comparators on the SAME worlds, windows, seeds and estimator (descriptive; the
registered primary criterion stays the u24 isolated synergy test):
  MAPPO_U0  speed-only MAPPO u0 = A_checkpoint_000.pt (== common_checkpoint_sha256) through the
            identical launch_evaluation.episode harness (evaluation_update=0);
  NATIVE    the qualified true-native plant (tools.coopnative.plant) on the same frozen runtime
            (manifest b3c031ad) that the clean3 episodes import;
  LANEON    lane-on MAPPO aecd7d70 (owner 2026-11-06 01:16, descriptive). Adopted from the
            monitor receipt evaluation/LANEON_u024.json when the deployed source runs it, else run
            through a deployed tools.coopmwm.laneon_baseline (specs/episode), charged to the
            clean3 v2v_final allocation that the quote reserved for it.
MAPPO_U0/NATIVE are NOT in the clean3 budget: they charge a separate ledger in the final root
whose sealed FINAL_EVAL_BUDGET.json must be bound by an owner OWNER_APPROVAL.json first.

Gate: u24 sealed (A/B/R_UPDATE_024 sealed, checkpoints, *_COMPLETE, LEARNING_MONITOR_u024),
registered health (no FAULT/PAUSE_SAFETY/active STOP/unreleased PAUSE_*), and safety clean
(every LEARNING_MONITOR_u001..u024 bound to this CONFIG, no PAUSE_SAFETY action, zero
collisions and teleports in every arm).  Seeds come only from the registered panel.

Commands (cpuV, from the clean3 deployed source environment; see FINAL_EVAL_READY.md):
  check     registration + budget + gate status, read-only
  register  create the final root and its sealed extra-baseline budget (zero physics)
  run       wait for the gate, run the frozen comparators, wait for the monitor's
            PAIRED_u024/replicated_u024 receipts, write FINAL_TABLES.json/.md
  analyse   tables only, from sealed receipts (zero physics)
  dry-run   full launcher path on a fake root built from a real CONFIG/registration copy with
            fake physics; never imports a simulator, never reserves on a real ledger
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

SCHEMA = 'coopmwm.clean3.finaleval.v1'
ROOT = '/home/ubuntu/coopmwm20261106_clean3_r6'
FINAL_ROOT = '/home/ubuntu/coopmwm20261106_clean3_finaleval'
UPDATE = 24
ARMS = ('A', 'B', 'R')
EPISODES = 96
HORIZON_S = 1200
SAFETY = ('collisions', 'teleports', 'hard_brakes', 'ttc_events')
HARD = ('collisions', 'teleports')
MARGIN_MIN = 0.15
LANEON_PHASE_SPLIT = 'v2v_final_laneon_mappo_aecd7d70'
LANEON_SHA = 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'
LANEON_CHECKPOINT = '/home/ubuntu/cpuq_final/e3final/inputs/mappo_v5_final.pt'
NATIVE_RUNTIME_SHA = 'b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'
FINAL_PHASES = dict(final_mappo_u0=133_200, final_native=133_200)   # 96 x 1200 x 1.156
BASELINES = dict(
    MAPPO_U0=dict(label='speed-only MAPPO u0 (frozen common checkpoint, identical harness)',
                  ledger='final', phase='final_mappo_u0', prefix='finaleval_mappo_u000'),
    NATIVE=dict(label='native SUMO CAV driving (qualified true-native plant)',
                ledger='final', phase='final_native', prefix='finaleval_native'),
    LANEON=dict(label='lane-on MAPPO aecd7d70 (owner 01:16; descriptive)',
                ledger='clean3', phase=None, prefix='finaleval_laneon'))
LABELS = dict(A='A u24 (plain MAPPO PPO, speed-only)', B='B u24 (verified cooperative bank)',
              R='R u24 (random-bank control)', **{k: v['label'] for k, v in BASELINES.items()})
ORDER = ('B', 'A', 'R', 'MAPPO_U0', 'LANEON', 'NATIVE')
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
             'hetdata_truth', 'vtype', 'vehicle_params', 'driver')
SYNERGY, TOTAL_BENEFIT = 'policy_cooperation_gain_veh_s', 'delta_all_new_veh_s'
PAIRED_FILE = 'evaluation/PAIRED_u024.json'
REPLICATED_FILE = 'track2_diagnostics/replicated_u024_live.json'
LANEON_MONITOR_FILE = 'evaluation/LANEON_u024.json'


class NotReady(Exception):
    """Gate not satisfied yet; waiting is allowed."""


class Blocked(Exception):
    """Never proceed automatically (safety, tampering, budget, interrupted work)."""


# ---------------------------------------------------------------- small helpers

def guard(path):
    for text in (str(path), str(Path(path).resolve())):
        s = text.replace('\\', '/').lower()
        hit = [t for t in FORBIDDEN if t in s]
        if hit:
            raise Blocked(f'REFUSED: forbidden path token {hit}: {path}')
    return Path(path)


def read(path):
    return json.loads(guard(path).read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(guard(path).read_bytes()).hexdigest()


def digest(value):
    from tools.coopmwm.protocol import digest as d
    return d(value)


def seal(value, key):
    from tools.coopmwm.exact_grade import _seal
    return _seal(value, key)


def verify(value, key, what):
    from tools.coopmwm.exact_grade import verify_sealed
    try:
        verify_sealed(value, key)
    except ValueError as error:
        raise Blocked(f'{what}: {error}')
    return value


def publish(path, value, *, replace=False):
    path = guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, indent=1)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if replace:
        os.replace(temporary, path)
    else:
        os.link(temporary, path)
        temporary.unlink()
    return path


def ledger_rows(root):
    path = Path(root)/'CHARGES.jsonl'
    if not path.is_file():
        return []
    return [json.loads(line) for line in guard(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def _finite(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Blocked(f'finite {what} required')
    return float(value)


# ---------------------------------------------------------------- registration (zero physics)

def panel(root, cfg):
    """The registered final panel; the ONLY seed source. Refuses a changed file."""
    from tools.coopmwm import rerun_cpus as rc
    path = Path(root)/'ISOLATED_REGISTRATION.json'
    if file_sha(path) != cfg.get('isolated_registration_sha256'):
        raise Blocked('ISOLATED_REGISTRATION.json differs from the CONFIG SHA binding')
    try:
        return rc.final_world_registration(read(path))
    except ValueError as error:
        raise Blocked(str(error))


def budget_phases(cfg):
    value = verify(deepcopy(cfg['clean3_budget']), 'receipt_sha256', 'clean3_budget seal')
    return value['phases']


def laneon_phase(cfg):
    return LANEON_PHASE_SPLIT if LANEON_PHASE_SPLIT in budget_phases(cfg) else 'v2v_final'


def final_budget(final_root):
    path = Path(final_root)/'FINAL_EVAL_BUDGET.json'
    if not path.is_file():
        return None
    return verify(read(path), 'receipt_sha256', 'FINAL_EVAL_BUDGET seal')


def approval(final_root):
    """Owner approval for the extra MAPPO_U0/NATIVE budget, bound to the sealed budget."""
    budget = final_budget(final_root)
    path = Path(final_root)/'OWNER_APPROVAL.json'
    if budget is None:
        return dict(status='UNREGISTERED', reason='no FINAL_EVAL_BUDGET.json (run register)')
    if not path.is_file():
        return dict(status='REGISTERED_UNAPPROVED', budget_sha256=budget['receipt_sha256'],
                    reason='OWNER_APPROVAL.json absent')
    value = read(path)
    if (value.get('approved') is not True or value.get('budget_sha256') != budget['receipt_sha256']
            or not str(value.get('authority', '')).strip()):
        return dict(status='REGISTERED_UNAPPROVED', budget_sha256=budget['receipt_sha256'],
                    reason='OWNER_APPROVAL.json does not approve this exact budget')
    return dict(status='APPROVED', budget_sha256=budget['receipt_sha256'],
                authority=value['authority'], dry_run=bool(value.get('dry_run')))


def registration_check(root, cfg, *, final_root=None):
    """Zero-physics registration/budget audit. Returns a report; never raises on a FAIL."""
    from tools.coopmwm import rerun_cpus as rc, protocol
    checks = []

    def check(name, ok, **detail):
        checks.append(dict(name=name, ok=bool(ok), **detail))
    root = Path(root)
    contract = cfg.get('final_evaluation')
    check('contract_equals_registered', contract == rc.FINAL_EVALUATION,
          arms=(contract or {}).get('arms'), comparisons=(contract or {}).get('comparisons'),
          metric=(contract or {}).get('metric'), bootstrap=(contract or {}).get('bootstrap'),
          safety_counters=(contract or {}).get('safety_counters'), phase=(contract or {}).get('phase'),
          episodes_per_arm=(contract or {}).get('episodes_per_arm'), horizon_s=(contract or {}).get('horizon_s'))
    check('episode_only_no_final_paired_branches', cfg.get('final_paired_branches') is False,
          final_paired_branches=cfg.get('final_paired_branches'))
    registration = None
    try:
        registration = panel(root, cfg)
        check('registration_sha_binding', True, isolated_registration_sha256=cfg.get('isolated_registration_sha256'),
              final_worlds_sha256=registration['final_worlds_sha256'])
    except Blocked as error:
        check('registration_sha_binding', False, error=str(error))
    if registration is not None:
        universe = {}
        for index in range(len(protocol.TRAINING_SEEDS)):
            for position, world in enumerate(protocol.worlds('train', index)):
                universe[digest(world)] = (index, position)
        where = [universe.get(digest(row['world'])) for row in registration['worlds']]
        first = sorted(p for i, p in filter(None, where) if i == 0)
        second = sorted(p for i, p in filter(None, where) if i == 1)
        check('panel_is_registered_train_universe', None not in where and first == list(range(72))
              and second == list(range(24)), worlds=len(where), primary_986xxx=len(first),
              secondary_987xxx_first=len(second),
              seeds=[min(r['world']['seed'] for r in registration['worlds']),
                     max(r['world']['seed'] for r in registration['worlds'])])
        check('world_id_is_seed', all(r['world_id'] == str(r['world']['seed']) for r in registration['worlds']))
    entries = [b for b in cfg.get('extra_final_baselines') or []
               if isinstance(b, dict) and b.get('commit') == 'aecd7d70']
    check('laneon_extra_baseline_registered', len(entries) == 1 and entries[0].get('episodes') == EPISODES,
          entry=entries[0] if entries else None)
    try:
        phases = budget_phases(cfg)
        rows_ = ledger_rows(root)
        spent = {}
        for row in rows_:
            spent[row['phase']] = spent.get(row['phase'], 0)+row['sim_s']
        need_abr, need_laneon = len(ARMS)*EPISODES*HORIZON_S, EPISODES*HORIZON_S

        def charged(prefixes):
            return sum(r['sim_s'] for r in rows_ if r['id'].startswith(prefixes))
        done_abr, done_laneon = charged(('paired_u024_',)), charged(('finaleval_laneon_', 'laneon_u024_'))
        if LANEON_PHASE_SPLIT in phases:
            allocation = {'v2v_final': (phases['v2v_final'], need_abr, done_abr),
                          LANEON_PHASE_SPLIT: (phases[LANEON_PHASE_SPLIT], need_laneon, done_laneon)}
        else:
            allocation = {'v2v_final': (phases['v2v_final'], need_abr+need_laneon, done_abr+done_laneon)}
        rows = {k: dict(allocated=a, need=n, margin=round(a/n-1, 4), spent=spent.get(k, 0),
                        charged_by_these_waves=d, other_spend=spent.get(k, 0)-d,
                        remaining=a-spent.get(k, 0), outstanding_need=n-d)
                for k, (a, n, d) in allocation.items()}
        check('clean3_final_phase_covers_ABR_and_laneon_with_margin',
              all(r['margin'] >= MARGIN_MIN and r['remaining'] >= r['outstanding_need'] and r['other_spend'] == 0
                  for r in rows.values()), phases=rows,
              u24_isolated_replicated=dict(allocated=phases.get('isolated_replicated'),
                                           spent=spent.get('isolated_replicated', 0)))
    except (Blocked, KeyError) as error:
        check('clean3_final_phase_covers_ABR_and_laneon_with_margin', False, error=repr(error))
    extra = dict(status='NOT_CHECKED')
    if final_root is not None:
        extra = approval(final_root)
        budget = final_budget(final_root)
        if budget is not None:
            spent = {}
            for row in ledger_rows(final_root):
                spent[row['phase']] = spent.get(row['phase'], 0)+row['sim_s']
            extra['phases'] = {k: dict(allocated=v, need=EPISODES*HORIZON_S,
                                       margin=round(v/(EPISODES*HORIZON_S)-1, 4), spent=spent.get(k, 0))
                               for k, v in budget['phases'].items()}
            extra['margin_ok'] = all(p['margin'] >= MARGIN_MIN for p in extra['phases'].values())
    status = 'PASS' if all(c['ok'] for c in checks) else 'FAIL'
    return dict(schema=SCHEMA+'.registration', status=status, root=str(root), checks=checks,
                extra_baseline_budget=extra, new_sim_s=0,
                note=('MAPPO_U0 and NATIVE are not in the clean3 registration/budget; they run only '
                      'under an owner-approved FINAL_EVAL_BUDGET.json in the final root'))


# ---------------------------------------------------------------- gate (zero physics)

def gate(root, cfg, runtime):
    """u24 sealed AND safety clean. NotReady = wait; Blocked = never proceed automatically."""
    root = Path(root)
    missing = [name for arm in ARMS for name in (f'{arm}_UPDATE_{UPDATE:03}.json',
               f'{arm}_checkpoint_{UPDATE:03}.pt', f'{arm}_COMPLETE.json') if not (root/name).is_file()]
    if not (root/f'LEARNING_MONITOR_u{UPDATE:03}.json').is_file():
        missing.append(f'LEARNING_MONITOR_u{UPDATE:03}.json')
    try:
        runtime.health(root)
    except Exception as error:   # registered health: FAULT / PAUSE_SAFETY / STOP / unreleased PAUSE_*
        raise NotReady(f'clean3 health not clear: {error!r}')
    if missing:
        raise NotReady('u24 not sealed: missing '+', '.join(missing))
    updates = {}
    for arm in ARMS:
        receipt = verify(read(root/f'{arm}_UPDATE_{UPDATE:03}.json'), 'update_sha256', f'{arm} u24 update')
        updates[arm] = receipt['update_sha256']
    binding, events, pauses = digest(cfg), {}, []
    for u in range(1, UPDATE+1):
        path = root/f'LEARNING_MONITOR_u{u:03}.json'
        if not path.is_file():
            raise NotReady(f'monitor receipt u{u:03} missing')
        row = verify(read(path), 'receipt_sha256', f'monitor u{u:03}')
        if row.get('config_sha256') != binding or row.get('update') != u:
            raise Blocked(f'monitor u{u:03} is not bound to this CONFIG')
        if sorted(a['arm'] for a in row['arms']) != list(ARMS):
            raise Blocked(f'monitor u{u:03} lacks an arm')
        for a in row['arms']:
            if a.get('action') == 'PAUSE_SAFETY':
                pauses.append((u, a['arm']))
            for k in HARD:
                value = a['metrics']['safety'][k]
                if value:
                    events[f'u{u:03}_{a["arm"]}_{k}'] = value
    if pauses or events:
        raise Blocked(f'safety not clean: pauses={pauses} events={events}')
    common = root/'A_checkpoint_000.pt'
    if file_sha(common) != cfg['common_checkpoint_sha256']:
        raise Blocked('A_checkpoint_000.pt is not the registered common MAPPO u0')
    return dict(schema=SCHEMA+'.gate', status='PASS', root=str(root), config_sha256=binding,
                update_sha256=updates, monitors=UPDATE, collisions=0, teleports=0,
                common_checkpoint_sha256=cfg['common_checkpoint_sha256'], unix=time.time(), new_sim_s=0)


# ---------------------------------------------------------------- specs / records

def baseline_specs(name, registration, cfg, root, *, laneon_checkpoint=LANEON_CHECKPOINT, laneon_sha=LANEON_SHA):
    if name == 'MAPPO_U0':
        checkpoint = dict(path=str(Path(root)/'A_checkpoint_000.pt'), file_sha256=cfg['common_checkpoint_sha256'],
                          update=0, components=['base'])
    elif name == 'NATIVE':
        checkpoint = dict(kind='native_reference', runtime=cfg['runtime'],
                          runtime_manifest_sha256=NATIVE_RUNTIME_SHA)
    elif name == 'LANEON':
        checkpoint = dict(path=str(laneon_checkpoint), file_sha256=laneon_sha, commit='aecd7d70')
    else:
        raise Blocked('unknown baseline '+name)
    extra = dict(baseline='mappo_laneon_aecd7d70', lane_change_mode=513) if name == 'LANEON' else {}
    return [dict(branch='shared_final', arm=name, world_id=row['world_id'], world=deepcopy(row['world']),
                 checkpoint=deepcopy(checkpoint), sim_s=HORIZON_S, window=dict(start_s=0, end_s=HORIZON_S),
                 final_worlds_sha256=registration['final_worlds_sha256'], **extra)
            for row in registration['worlds']]


def check_record(record, registration, arm):
    """One graded full-window record on a registered world; returns the normalised record."""
    if record.get('status') != 'graded':
        raise Blocked(f'{arm}: ungraded record; never drop/impute/retry')
    rows = {r['world_id']: r['world'] for r in registration['worlds']}
    if record.get('world_id') not in rows or record.get('world') != rows[record['world_id']]:
        raise Blocked(f'{arm}: record outside the registered panel')
    if record.get('window') != dict(start_s=0, end_s=HORIZON_S):
        raise Blocked(f'{arm}: full 1200 s window required')
    if record.get('final_worlds_sha256') != registration['final_worlds_sha256']:
        raise Blocked(f'{arm}: panel binding differs')
    safety = record.get('safety')
    if not isinstance(safety, dict) or set(safety) != set(SAFETY) or any(
            type(v) is not int or v < 0 for v in safety.values()):
        raise Blocked(f'{arm}: all four non-negative integer safety counters required')
    out = dict(record, arm=arm)
    out['J_veh_s'] = _finite(record.get('J_veh_s'), f'{arm} J_veh_s')
    return out


def complete(records, registration, arm):
    by_world = {}
    for record in records:
        record = check_record(record, registration, arm)
        if record['world_id'] in by_world:
            raise Blocked(f'{arm}: duplicate world {record["world_id"]}')
        by_world[record['world_id']] = record
    if set(by_world) != {r['world_id'] for r in registration['worlds']}:
        raise Blocked(f'{arm}: every registered world exactly once required')
    return by_world


# ---------------------------------------------------------------- real runtime (cpuV)

def _mappo_u0(spec, cfg, work):
    from tools.coopmwm import launch_evaluation
    record = launch_evaluation.episode(dict(spec, arm='A'), dict(cfg, evaluation_update=0), work)
    return dict(record, arm='MAPPO_U0', source_arm='A', checkpoint_update=0)


def _native(spec, cfg, work, approval_line):
    import types
    from tools.coopnative import plant as native
    from tools.coopnative.policy import Policy as NativePolicy
    from tools.coopmwm.protocol import guard as world_guard, install_seed_scope
    from tools.coopmwm.replay_bank import ttc_events
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))

    def setup(runtime):
        native.setup(runtime)
        install_seed_scope()

    def approved(text):
        if text != approval_line:
            raise ValueError('native reference requires the owner-approved final-eval budget')
    run = types.FunctionType(native.episode.__code__, dict(native.episode.__globals__,
        approval_line=approved, setup=setup, guard=world_guard))
    job = dict(id=Path(work).name, path=str(work), world=deepcopy(spec['world']), role='train', arm='native',
               weights=NativePolicy().state_dict(), runtime=cfg['runtime'], noise_seed=spec['world']['seed'],
               sim_s=HORIZON_S, stop=HORIZON_S, approval_line=approval_line)
    result = run(job)
    totals = result['metrics'][str(HORIZON_S-300)]
    safety = dict(collisions=int(totals['collisions']), teleports=int(totals['teleports']),
                  hard_brakes=int(totals['hard_brake']),
                  ttc_events=int(ttc_events(Path(work)/'observable.jsonl.gz', 300, HORIZON_S)))
    return dict({k: deepcopy(spec[k]) for k in ('world_id', 'world', 'checkpoint', 'window', 'final_worlds_sha256')},
                arm='NATIVE', status='graded', J_veh_s=float(totals['team_delay_vs']), safety=safety)


def _laneon(spec, cfg, work):
    from tools.coopmwm import laneon_baseline as lb
    record = lb.episode(spec, cfg, work)
    keep = {k: deepcopy(spec[k]) for k in ('world_id', 'world', 'window', 'final_worlds_sha256')}
    return dict(keep, arm='LANEON', status=record.get('status', 'graded'), J_veh_s=record['J_veh_s'],
                safety={k: record['safety'][k] for k in SAFETY}, checkpoint=deepcopy(spec['checkpoint']),
                lane_commands=record.get('lane_commands'))


def _worker(name, spec, cfg, work, extra):
    if name == 'MAPPO_U0':
        return _mappo_u0(spec, cfg, work)
    if name == 'NATIVE':
        return _native(spec, cfg, work, extra['approval_line'])
    if name == 'LANEON':
        return _laneon(spec, cfg, work)
    raise ValueError('unknown baseline '+name)


class Runtime:
    """Real cpuV bindings (clean3 deployed source). Tests/dry-run substitute FakeRuntime."""
    physics = True

    def config(self, root):
        from tools.coopmwm import launch_training as t
        return t.checked_config(Path(root)/'CONFIG.json')

    def health(self, root):
        from tools.coopmwm import launch_training as t
        t.health(root)

    def backend(self, cfg):
        from tools.coopmwm import launch_production as production
        return production.backend_config(cfg)

    def reserve_clean3(self, cfg, jobs, phase):
        from tools.coopmwm import track2_backend as p1
        p1._reserve(self.backend(cfg), jobs, phase)

    def reserve_final(self, final_root, budget, jobs, phase):
        import fcntl
        import types
        from tools.coopnative import storage
        reserve = types.FunctionType(storage.reserve.__code__, dict(storage.reserve.__globals__,
            CAP=sum(budget['phases'].values()), CAPS=dict(budget['phases'])))
        with (Path(final_root)/'ACCOUNTING.claim').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reserve(Path(final_root), jobs, phase)

    def laneon_runnable(self):
        try:
            from tools.coopmwm import laneon_baseline as lb
        except ImportError:
            return False
        return callable(getattr(lb, 'episode', None))

    def run(self, items, cfg, workers, extra):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        os.environ['COOPMWM_G5_RUNTIME'] = '1'   # same bound runtime env as the clean3 owners
        native = self.backend(cfg)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_worker, name, spec, native, work, extra) for name, spec, work in items]
            return [f.result(timeout=3600) for f in futures]


# ---------------------------------------------------------------- baselines

def laneon_mode(root, cfg, runtime):
    if (Path(root)/LANEON_MONITOR_FILE).is_file():
        return 'adopt'
    source = Path(cfg.get('source_root', ''))/'pytools/tools/coopmwm/launch_evaluation.py'
    if source.is_file() and 'laneon' in guard(source).read_text(encoding='utf-8'):
        return 'adopt'          # the deployed monitor runs it after PAIRED_u024
    return 'run' if runtime.laneon_runnable() else 'unavailable'


def adopt_laneon(root, registration):
    value = verify(read(Path(root)/LANEON_MONITOR_FILE), 'receipt_sha256', 'LANEON_u024')
    if value.get('registration', {}).get('final_worlds_sha256') not in (None, registration['final_worlds_sha256']):
        raise Blocked('LANEON_u024 panel differs')
    worlds = {row['world_id']: row['world'] for row in registration['worlds']}
    records = [dict({k: r[k] for k in ('world_id', 'J_veh_s', 'safety') if k in r},
                    world=r.get('world', worlds.get(r.get('world_id'))),
                    window=r.get('window', dict(start_s=0, end_s=HORIZON_S)),
                    final_worlds_sha256=r.get('final_worlds_sha256', registration['final_worlds_sha256']),
                    status=r.get('status', 'graded')) for r in value['results']]
    return complete(records, registration, 'LANEON'), dict(source=str(Path(root)/LANEON_MONITOR_FILE),
                                                         receipt_sha256=value['receipt_sha256'])


def run_baselines(names, *, root, final_root, cfg, registration, runtime, workers,
                  laneon_checkpoint=LANEON_CHECKPOINT, laneon_sha=LANEON_SHA):
    """Reserve every requested wave (all-or-nothing per wave) BEFORE dispatch, then run them in
    one pool. Never replays an interrupted wave and never re-runs a sealed one."""
    root, final_root = Path(root), Path(final_root)
    out, todo, sources = {}, [], {}
    extra = {}
    if {'MAPPO_U0', 'NATIVE'} & set(names):
        state = approval(final_root)
        if state['status'] != 'APPROVED':
            raise Blocked(f'MAPPO_U0/NATIVE need an owner-approved final budget: {state}')
        extra['approval_line'] = state['budget_sha256']
    for name in names:
        target = final_root/f'BASELINE_{name}.json'
        if target.is_file():
            value = verify(read(target), 'receipt_sha256', target.name)
            out[name] = complete(value['records'], registration, name)
            sources[name] = dict(source=str(target), receipt_sha256=value['receipt_sha256'])
            continue
        meta = BASELINES[name]
        ledger_root = root if meta['ledger'] == 'clean3' else final_root
        if any(r['id'].startswith(meta['prefix']+'_') for r in ledger_rows(ledger_root)):
            raise Blocked(f'{name}: interrupted wave in the ledger; never replay')
        specs = baseline_specs(name, registration, cfg, root, laneon_checkpoint=laneon_checkpoint,
                               laneon_sha=laneon_sha)
        if name == 'LANEON' and file_sha(laneon_checkpoint) != laneon_sha:
            raise Blocked('lane-on checkpoint SHA differs from the pinned aecd7d70 export')
        if name == 'MAPPO_U0' and file_sha(specs[0]['checkpoint']['path']) != cfg['common_checkpoint_sha256']:
            raise Blocked('MAPPO u0 checkpoint differs from the registered common checkpoint')
        jobs = [dict(id=f'{meta["prefix"]}_{i:04}', sim_s=s['sim_s']) for i, s in enumerate(specs)]
        todo.append((name, meta, specs, jobs))
    for name, meta, specs, jobs in todo:
        if meta['ledger'] == 'clean3':
            runtime.reserve_clean3(cfg, jobs, laneon_phase(cfg))
        else:
            runtime.reserve_final(final_root, final_budget(final_root), jobs, meta['phase'])
        publish(final_root/f'RESERVED_{name}.json', dict(name=name, jobs=[j['id'] for j in jobs],
                sim_s=sum(j['sim_s'] for j in jobs), ledger=meta['ledger'],
                phase=meta['phase'] or laneon_phase(cfg), unix=time.time()))
    items = [(name, spec, str(final_root/'evaluation'/job['id']))
             for name, _, specs, jobs in todo for spec, job in zip(specs, jobs)]
    results = runtime.run(items, cfg, workers, extra) if items else []
    position = 0
    for name, meta, specs, jobs in todo:
        records = results[position:position+len(specs)]
        position += len(specs)
        for spec, record in zip(specs, records):
            if record.get('world_id') != spec['world_id']:
                raise Blocked(f'{name}: result order/world mismatch')
        by_world = complete(records, registration, name)
        value = seal(dict(schema=SCHEMA+'.baseline', name=name, label=meta['label'], update=UPDATE,
                          ledger=meta['ledger'], phase=meta['phase'] or laneon_phase(cfg),
                          jobs=[j['id'] for j in jobs], sim_s=sum(j['sim_s'] for j in jobs),
                          final_worlds_sha256=registration['final_worlds_sha256'], physics=runtime.physics,
                          records=[by_world[r['world_id']] for r in registration['worlds']]), 'receipt_sha256')
        publish(final_root/f'BASELINE_{name}.json', value)
        out[name] = by_world
        sources[name] = dict(source=str(final_root/f'BASELINE_{name}.json'), receipt_sha256=value['receipt_sha256'])
    return out, sources


# ---------------------------------------------------------------- analysis (zero physics)

def paired_stats(treat, comp, bootstrap, *, pct=True):
    """Registered paired whole-world percentile bootstrap (planner_eval._paired_ci, identical
    resamples for every contrast), plus a ratio-of-means % change on the SAME resamples."""
    from tools.coopmwm import planner_eval as pe
    t, c = np.asarray(treat, float), np.asarray(comp, float)
    alpha = 1-bootstrap['level']
    ci = pe._paired_ci(t-c, bootstrap['reps'], bootstrap['seed'], alpha)
    out = dict(n=ci['n_worlds'], mean_treatment=float(t.mean()), mean_comparator=float(c.mean()),
               diff=ci['mean_delta_vs'], ci95=ci['ci95'], p_two_sided=ci['p_two_sided'],
               worlds_lower=ci['worlds_improved'], worlds_higher=int(np.sum(t-c > 0)))
    if pct:
        idx = np.random.default_rng(bootstrap['seed']).integers(0, len(t), (bootstrap['reps'], len(t)))
        mt, mc = t[idx].mean(1), c[idx].mean(1)
        if c.mean() != 0 and np.all(mc != 0):
            boot = 100*(mt-mc)/mc
            lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
            out.update(pct=float(100*(t.mean()-c.mean())/c.mean()), pct_ci95=[float(lo), float(hi)])
        else:
            out.update(pct=None, pct_ci95=[None, None])
    return out


def check_paired(paired, registration, contract):
    from tools.coopmwm import planner_eval as pe
    verify(paired, 'receipt_sha256', 'PAIRED_u024')
    if paired.get('update') != UPDATE or paired.get('status') != 'graded':
        raise Blocked('PAIRED_u024 is not the graded u24 receipt')
    if paired['registration'].get('final_worlds_sha256') != registration['final_worlds_sha256']:
        raise Blocked('PAIRED_u024 panel differs from the registration')
    for arm in ARMS:
        if paired['checkpoints'][arm].get('update') != UPDATE:
            raise Blocked('PAIRED_u024 checkpoint is not u24')
    try:
        again = pe.analyse_track2(registration=paired['registration'], checkpoints=paired['checkpoints'],
                                  records=paired['results'], contract=contract)
    except ValueError as error:
        raise Blocked(f'PAIRED_u024 records fail the registered analysis: {error}')
    if json.loads(json.dumps(again)) != json.loads(json.dumps(paired['analysis'])):
        raise Blocked('PAIRED_u024 analysis does not reproduce from its records')
    return {arm: complete([r for r in paired['results'] if r['arm'] == arm], registration, arm) for arm in ARMS}


def check_replicated(value):
    verify(value, 'receipt_sha256', 'replicated_u024')
    if value.get('update') != UPDATE or value.get('kind') != 'replicated' or value.get('status') != 'complete':
        raise Blocked('replicated u24 receipt is not the complete live u24 test')
    reports = value['metric_reports']
    keep = ('mean', 'ci_lo', 'ci_hi', 'n', 'level', 'reps', 'seed')
    return dict(receipt_sha256=value['receipt_sha256'], verdict=value.get('verdict'),
                criterion=value.get('criterion'), safety=value.get('safety'), safety_ok=value.get('safety_ok'),
                synergy={k: {f: v.get(f) for f in keep} for k, v in reports[SYNERGY]['comparisons'].items()},
                total_benefit={k: {f: v.get(f) for f in keep}
                               for k, v in reports[TOTAL_BENEFIT].get('comparisons', {}).items()},
                arm_means=dict(synergy=reports[SYNERGY].get('arm_means'),
                               total_benefit=reports[TOTAL_BENEFIT].get('arm_means')))


def analyse(*, registration, contract, arms, baselines, replicated=None, sources=None, missing=None,
            paired_analysis=None):
    """Paper tables. ``arms``/``baselines``: {name: {world_id: record}} (complete panels)."""
    bootstrap = contract['bootstrap']
    worlds = [r['world_id'] for r in registration['worlds']]
    have = dict(arms, **baselines)
    J = {name: np.array([have[name][w]['J_veh_s'] for w in worlds]) for name in have}
    S = {name: {k: np.array([have[name][w]['safety'][k] for w in worlds], float) for k in SAFETY} for name in have}
    names = [n for n in ORDER if n in have]
    summary = {n: dict(label=LABELS[n], n=len(worlds), mean_J_veh_s=float(J[n].mean()),
                       safety_totals={k: int(S[n][k].sum()) for k in SAFETY},
                       any_collision_or_teleport=bool(S[n]['collisions'].sum()+S[n]['teleports'].sum() > 0))
               for n in names}
    tables = dict(arms=summary)
    if 'MAPPO_U0' in have:
        tables['total_benefit_vs_mappo'] = dict(comparator='MAPPO_U0', convention='diff = X - MAPPO_U0; negative = less delay',
            rows={n: paired_stats(J[n], J['MAPPO_U0'], bootstrap) for n in names if n != 'MAPPO_U0'})
    else:
        tables['total_benefit_vs_mappo'] = dict(status='unavailable', reason=(missing or {}).get('MAPPO_U0', 'not run'),
            fallback_comparator='A', rows={n: paired_stats(J[n], J['A'], bootstrap) for n in names if n != 'A'})
    registered = {}
    for comparison in contract['comparisons']:
        a, b = comparison.split('-')
        registered[comparison] = paired_stats(J[a], J[b], bootstrap)
    if paired_analysis is not None:
        for comparison, row in registered.items():
            want = paired_analysis['paired_ci'][comparison]
            if (row['diff'], row['ci95'][0], row['ci95'][1]) != (want['mean'], want['low'], want['high']):
                raise Blocked(f'{comparison} does not reproduce the sealed PAIRED_u024 analysis')
    tables['registered_B_A_R'] = dict(convention='diff = first - second; negative = first has less delay',
                                      rows=registered, matches_paired_u024_analysis=paired_analysis is not None)
    decomposition = None
    if 'MAPPO_U0' in have:
        steps = (('A-MAPPO_U0', 'A', 'MAPPO_U0', 'continued speed-only PPO training (24 updates)'),
                 ('R-A', 'R', 'A', 'cooperation layer + trigger with a random bank (same move distribution)'),
                 ('B-R', 'B', 'R', 'verified cooperative bank (the cooperation-specific effect)'))
        total = J['B']-J['MAPPO_U0']
        error = float(np.max(np.abs(sum(J[t]-J[c] for _, t, c, _ in steps)-total)))
        decomposition = dict(identity='B-MAPPO_U0 = (A-MAPPO_U0) + (R-A) + (B-R), exact per world and per '
                                      'bootstrap resample (shared resample indices)',
            total=paired_stats(J['B'], J['MAPPO_U0'], bootstrap),
            components={k: dict(paired_stats(J[t], J[c], bootstrap, pct=False), meaning=m) for k, t, c, m in steps},
            max_identity_error_veh_s=error)
    tables['synergy'] = dict(panel_decomposition=decomposition or dict(status='unavailable (needs MAPPO_U0)'),
        registered_isolated_u24=replicated or dict(status='unavailable'))
    safety = dict(totals={n: summary[n]['safety_totals'] for n in names},
                  any_collision_or_teleport={n: summary[n]['any_collision_or_teleport'] for n in names})
    for comparator in ('MAPPO_U0', 'A'):
        if comparator in have:
            safety[f'vs_{comparator}'] = {n: {k: paired_stats(S[n][k], S[comparator][k], bootstrap, pct=False)
                                              for k in SAFETY} for n in names if n != comparator}
            safety[f'worlds_more_hard_events_than_{comparator}'] = {
                n: [w for i, w in enumerate(worlds) if any(S[n][k][i] > S[comparator][k][i] for k in HARD)]
                for n in names if n != comparator}
    tables['safety'] = safety
    comparator = 'MAPPO_U0' if 'MAPPO_U0' in have else 'A'
    pens = sorted({r['world']['penetration'] for r in registration['worlds']})
    strata = {}
    for i, p in enumerate(pens):
        mask = np.array([r['world']['penetration'] == p for r in registration['worlds']])
        boot = dict(bootstrap, seed=bootstrap['seed']+1000+i)
        strata[str(p)] = {n: paired_stats(J[n][mask], J[comparator][mask], boot)
                          for n in ('B', 'A', 'R', 'LANEON', 'NATIVE') if n in have and n != comparator}
    tables['by_penetration'] = dict(comparator=comparator, descriptive=True,
                                    note='within-stratum paired bootstrap, seed = registered + 1000 + stratum index',
                                    rows=strata)
    value = dict(schema=SCHEMA+'.tables', update=UPDATE, final_worlds_sha256=registration['final_worlds_sha256'],
                 bootstrap=bootstrap, metric=contract['metric'], horizon_s=contract['horizon_s'],
                 worlds=len(worlds), present=names, missing=missing or {}, sources=sources or {},
                 primary_criterion='u24 isolated B synergy CI95 low > 0 with no safety regression (track2 monitor)',
                 tables=tables)
    return seal(json.loads(json.dumps(value, allow_nan=False)), 'analysis_sha256')


def _f(x, digits=1):
    return 'n/a' if x is None else f'{x:,.{digits}f}'


def _ci(pair, digits=1):
    return 'n/a' if pair is None or None in pair else f'[{_f(pair[0], digits)}, {_f(pair[1], digits)}]'


def render(value):
    t = value['tables']
    lines = ['# clean3 final paired evaluation (u24) - paper tables', '',
             f"Panel: {value['worlds']} registered TRAIN worlds, final_worlds_sha256 `{value['final_worlds_sha256']}`; "
             f"metric {value['metric']} (veh-s, lower is better), full {value['horizon_s']} s; paired whole-world "
             f"percentile bootstrap {value['bootstrap']['reps']} reps, seed {value['bootstrap']['seed']}, "
             f"{int(100*value['bootstrap']['level'])}% CI. analysis_sha256 `{value['analysis_sha256']}`.", '']
    if value.get('dry_run'):
        lines += ['**DRY RUN - FAKE PHYSICS - NUMBERS ARE SYNTHETIC AND MEANINGLESS**', '']
    if value['missing']:
        lines += ['Not run: '+'; '.join(f'{k}: {v}' for k, v in value['missing'].items()), '']
    lines += ['## T0 Arms', '', '| arm | label | mean J (veh-s) | collisions | teleports | hard brakes | TTC events |',
              '|---|---|---:|---:|---:|---:|---:|']
    for n, r in t['arms'].items():
        s = r['safety_totals']
        lines.append(f"| {n} | {r['label']} | {_f(r['mean_J_veh_s'])} | {s['collisions']} | {s['teleports']} | "
                     f"{s['hard_brakes']} | {s['ttc_events']} |")

    def block(title, rows, note):
        out = ['', title, '', note, '', '| contrast | diff (veh-s) | 95% CI | % | % 95% CI | p | worlds lower/higher |',
               '|---|---:|---|---:|---|---:|---|']
        for k, r in rows.items():
            out.append(f"| {k} | {_f(r['diff'])} | {_ci(r['ci95'])} | {_f(r.get('pct'), 2)} | {_ci(r.get('pct_ci95'), 2)} "
                       f"| {_f(r['p_two_sided'], 4)} | {r['worlds_lower']}/{r['worlds_higher']} |")
        return out
    tb = t['total_benefit_vs_mappo']
    if tb.get('status') == 'unavailable':
        lines += block('## T1 Total benefit vs MAPPO (MAPPO_U0 not run; fallback comparator A u24)',
                       {f'{k}-A': v for k, v in tb['rows'].items()}, f"Reason: {tb['reason']}.")
    else:
        lines += block('## T1 Total benefit vs speed-only MAPPO u0', {f'{k}-MAPPO_U0': v for k, v in tb['rows'].items()},
                       tb['convention']+'.')
    lines += block('## T2 B vs A vs R (registered comparisons)', t['registered_B_A_R']['rows'],
                   t['registered_B_A_R']['convention']+'.')
    syn = t['synergy']
    lines += ['', '## T3 Synergy decomposition', '']
    dec = syn['panel_decomposition']
    if 'components' in dec:
        lines += [f"Panel identity: {dec['identity']} (max per-world error {dec['max_identity_error_veh_s']:.2e} veh-s).", '',
                  '| component | diff (veh-s) | 95% CI | p |', '|---|---:|---|---:|',
                  f"| total B-MAPPO_U0 | {_f(dec['total']['diff'])} | {_ci(dec['total']['ci95'])} | {_f(dec['total']['p_two_sided'], 4)} |"]
        for k, r in dec['components'].items():
            lines.append(f"| {k} | {_f(r['diff'])} | {_ci(r['ci95'])} | {_f(r['p_two_sided'], 4)} |")
    else:
        lines.append(f"Panel decomposition: {dec['status']}.")
    iso = syn['registered_isolated_u24']
    if 'synergy' in iso:
        lines += ['', f"Registered u24 isolated test (N=5 paired replicates; primary criterion): verdict **{iso['verdict']}**, "
                  f"safety_ok={iso['safety_ok']}, receipt `{iso['receipt_sha256']}`.", '',
                  '| metric | contrast | mean (veh-s) | 95% CI |', '|---|---|---:|---|']
        for metric in ('synergy', 'total_benefit'):
            for k, r in iso[metric].items():
                lines.append(f"| {metric} | {k} | {_f(r['mean'], 3)} | {_ci([r['ci_lo'], r['ci_hi']], 3)} |")
    else:
        lines.append('Registered u24 isolated test: unavailable.')
    lines += ['', '## T4 Safety (96-world totals and paired differences)', '',
              '| arm | collisions | teleports | hard brakes | TTC events | any collision/teleport |', '|---|---:|---:|---:|---:|---|']
    for n, s in t['safety']['totals'].items():
        lines.append(f"| {n} | {s['collisions']} | {s['teleports']} | {s['hard_brakes']} | {s['ttc_events']} | "
                     f"{t['safety']['any_collision_or_teleport'][n]} |")
    for comparator in ('MAPPO_U0', 'A'):
        key = f'vs_{comparator}'
        if key in t['safety']:
            lines += ['', f'Paired safety differences vs {comparator} (X - {comparator}, per world):', '',
                      '| arm | counter | mean diff | 95% CI | worlds lower/higher |', '|---|---|---:|---|---|']
            for n, rows in t['safety'][key].items():
                for k, r in rows.items():
                    lines.append(f"| {n} | {k} | {_f(r['diff'], 3)} | {_ci(r['ci95'], 3)} | {r['worlds_lower']}/{r['worlds_higher']} |")
    bp = t['by_penetration']
    lines += ['', f"## T5 By CAV penetration (descriptive, vs {bp['comparator']})", '',
              '| penetration | arm | diff (veh-s) | 95% CI | % | n |', '|---|---|---:|---|---:|---:|']
    for p, rows in bp['rows'].items():
        for n, r in rows.items():
            lines.append(f"| {p} | {n} | {_f(r['diff'])} | {_ci(r['ci95'])} | {_f(r.get('pct'), 2)} | {r['n']} |")
    return '\n'.join(lines)+'\n'


def tables(*, root, final_root, cfg, registration, runtime, baselines, sources, missing, dry_run=False):
    paired = read(Path(root)/PAIRED_FILE)
    arms = check_paired(paired, registration, cfg['final_evaluation'])
    sources = dict(sources, PAIRED_u024=dict(source=str(Path(root)/PAIRED_FILE), receipt_sha256=paired['receipt_sha256']))
    replicated = None
    path = Path(root)/REPLICATED_FILE
    if path.is_file():
        replicated = check_replicated(read(path))
        sources['replicated_u024'] = dict(source=str(path), receipt_sha256=replicated['receipt_sha256'])
    else:
        missing = dict(missing, replicated_u024='receipt absent')
    value = analyse(registration=registration, contract=cfg['final_evaluation'], arms=arms, baselines=baselines,
                    replicated=replicated, sources=sources, missing=missing, paired_analysis=paired['analysis'])
    if dry_run:
        value = seal(dict(value, dry_run=True), 'analysis_sha256')
    stem = 'FINAL_TABLES'
    existing = Path(final_root)/'FINAL_TABLES.json'
    if existing.is_file() and read(existing).get('analysis_sha256') != value['analysis_sha256']:
        stem = 'FINAL_TABLES_'+value['analysis_sha256'][:12]   # never overwrite a published table
    if not (Path(final_root)/f'{stem}.json').is_file():
        publish(Path(final_root)/f'{stem}.json', value)
    guard(Path(final_root)/f'{stem}.md').write_text(render(value), encoding='utf-8')
    return value


# ---------------------------------------------------------------- orchestration

class Status:
    def __init__(self, final_root):
        # Continue numbering after earlier launcher attempts; never overwrite their receipts.
        self.final_root = Path(final_root)
        self.serial = max([int(p.name.split('_')[1]) for p in self.final_root.glob('STATUS_[0-9][0-9][0-9]_*.json')],
                          default=0)

    def __call__(self, step, state, detail=None):
        self.serial += 1
        value = dict(schema=SCHEMA+'.status', step=step, status=state, sequence=self.serial, detail=detail,
                     unix=time.time(), timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
        publish(self.final_root/f'STATUS_{self.serial:03}_{step}_{state}.json', value)
        publish(self.final_root/'STATUS.json', value, replace=True)


def wait(step, fn, status, *, poll, timeout, sleep=time.sleep):
    started, last = time.monotonic(), None
    while True:
        try:
            return fn()
        except NotReady as error:
            if str(error) != last:
                status(step, 'WAITING', dict(reason=str(error)))
                last = str(error)
            if poll <= 0 or time.monotonic()-started > timeout:
                raise
            sleep(poll)


def execute(*, root, final_root, runtime, baselines=('MAPPO_U0', 'NATIVE', 'LANEON'), allow_missing=(),
            workers=24, poll=60, gate_timeout=7*86400, monitor_timeout=6*3600, laneon_checkpoint=LANEON_CHECKPOINT,
            laneon_sha=LANEON_SHA, dry_run=False, on_gate=None, sleep=time.sleep):
    root, final_root = Path(root), Path(final_root)
    if not final_root.is_dir():
        raise Blocked('final root missing: run register first')
    status = Status(final_root)
    current = 'CONFIG'
    try:
        status(current, 'RUNNING')
        cfg = runtime.config(root)
        registration = panel(root, cfg)
        report = registration_check(root, cfg, final_root=final_root)
        if report['status'] != 'PASS':
            raise Blocked('registration check failed: '+json.dumps([c for c in report['checks'] if not c['ok']]))
        missing, wanted = {}, []
        lane = laneon_mode(root, cfg, runtime) if 'LANEON' in baselines else None
        for name in baselines:
            if name in ('MAPPO_U0', 'NATIVE') and approval(final_root)['status'] != 'APPROVED':
                reason = 'extra budget not owner-approved: '+approval(final_root)['status']
            elif name == 'LANEON' and lane == 'unavailable':
                reason = 'no lane-on runner deployed and no monitor LANEON_u024 receipt'
            else:
                if not (name == 'LANEON' and lane == 'adopt'):
                    wanted.append(name)
                continue
            if name not in allow_missing:
                raise Blocked(f'{name}: {reason} (pass --allow-missing {name} to proceed without it)')
            missing[name] = reason
        status(current, 'PASS', dict(registration=report['status'], baselines=wanted, laneon_mode=lane, missing=missing))
        current = 'GATE'
        gated = wait(current, lambda: gate(root, cfg, runtime), status, poll=poll, timeout=gate_timeout, sleep=sleep)
        if not (final_root/'GATE_PASS.json').is_file():
            publish(final_root/'GATE_PASS.json', seal(gated, 'receipt_sha256'))
        status(current, 'PASS', gated)
        if on_gate is not None:
            on_gate()
        current = 'BASELINES'
        status(current, 'RUNNING', dict(names=wanted, workers=workers))
        results, sources = run_baselines(wanted, root=root, final_root=final_root, cfg=cfg, registration=registration,
                                         runtime=runtime, workers=workers, laneon_checkpoint=laneon_checkpoint,
                                         laneon_sha=laneon_sha)
        status(current, 'PASS', dict(names=sorted(results)))
        current = 'MONITOR_RECEIPTS'

        def receipts():
            needed = [PAIRED_FILE, REPLICATED_FILE]+([LANEON_MONITOR_FILE] if lane == 'adopt' else [])
            absent = [p for p in needed if not (root/p).is_file()]
            if absent:
                raise NotReady('waiting for the clean3 monitor: '+', '.join(absent))
            return needed
        wait(current, receipts, status, poll=poll, timeout=monitor_timeout, sleep=sleep)
        if lane == 'adopt':
            results['LANEON'], sources['LANEON'] = adopt_laneon(root, registration)
        status(current, 'PASS')
        current = 'TABLES'
        value = tables(root=root, final_root=final_root, cfg=cfg, registration=registration, runtime=runtime,
                       baselines=results, sources=sources, missing=missing, dry_run=dry_run)
        status(current, 'PASS', dict(analysis_sha256=value['analysis_sha256'], present=value['present']))
        return value
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


def register(*, root, final_root, runtime):
    """Zero physics: create the final root and its sealed extra-baseline budget (no approval)."""
    root, final_root = Path(root), Path(final_root)
    guard(final_root)
    if final_root.exists():
        raise Blocked('final root exists; never overwrite')
    cfg = runtime.config(root)
    registration = panel(root, cfg)
    final_root.mkdir(parents=True)
    (final_root/'CHARGES.jsonl').touch()
    (final_root/'ACCOUNTING.claim').touch()
    budget = seal(dict(schema=SCHEMA+'.budget', created=datetime.now(timezone.utc).isoformat(), clean3_root=str(root),
        final_worlds_sha256=registration['final_worlds_sha256'],
        isolated_registration_sha256=cfg['isolated_registration_sha256'], contract_sha256=digest(cfg['final_evaluation']),
        phases=dict(FINAL_PHASES), total_sim_s=sum(FINAL_PHASES.values()),
        basis={k: '96 TRAIN worlds x 1200 s = 115,200 sim-s + 15.6% margin' for k in FINAL_PHASES},
        baselines=dict(MAPPO_U0=dict(checkpoint_sha256=cfg['common_checkpoint_sha256'], harness='launch_evaluation.episode, evaluation_update=0'),
                       NATIVE=dict(runtime_manifest_sha256=NATIVE_RUNTIME_SHA, harness='tools.coopnative.plant native arm')),
        rule=('descriptive frozen comparators on the registered clean3 final panel, same worlds/windows/seeds/'
              'estimator; never a gate; registered primary criterion unchanged'),
        approval_required='OWNER_APPROVAL.json {approved: true, budget_sha256: <receipt_sha256>, authority: <owner verbatim>}'),
        'receipt_sha256')
    publish(final_root/'FINAL_EVAL_BUDGET.json', budget)
    with (final_root/'AUTHORITY.md').open('x', encoding='utf-8') as f:
        f.write(f"# clean3 final-eval extra baselines\n\nBudget receipt {budget['receipt_sha256']}: "
                f"{json.dumps(FINAL_PHASES)} sim-s on panel {registration['final_worlds_sha256']}. "
                'Requires OWNER_APPROVAL.json binding this receipt before any MAPPO_U0/NATIVE physics.\n')
    return budget


def check(*, root, final_root, runtime):
    cfg = runtime.config(root)
    report = registration_check(root, cfg, final_root=final_root if Path(final_root).is_dir() else None)
    try:
        report['gate'] = gate(root, cfg, runtime)
    except (NotReady, Blocked) as error:
        report['gate'] = dict(status=type(error).__name__, reason=str(error))
    report['laneon_mode'] = laneon_mode(root, cfg, runtime)
    report['monitor_receipts'] = {p: (Path(root)/p).is_file() for p in (PAIRED_FILE, REPLICATED_FILE, LANEON_MONITOR_FILE)}
    return report


# ---------------------------------------------------------------- dry run (fake physics)

def fake_J(world, arm):
    h = int(hashlib.sha256(f'{world["seed"]}:{arm}'.encode()).hexdigest()[:8], 16)/0xffffffff
    base = 20000+15*world['demand']*(1+world['penetration'])
    factor = dict(NATIVE=1.08, MAPPO_U0=1.0, LANEON=0.99, A=0.985, R=0.982, B=0.975).get(arm, 1.0)
    return float(base*factor*(1+0.02*(h-.5)))


def fake_record(spec, arm):
    h = int(hashlib.sha256(f'{spec["world"]["seed"]}:{arm}:s'.encode()).hexdigest()[:8], 16)
    return dict({k: deepcopy(spec[k]) for k in ('world_id', 'world', 'checkpoint', 'window', 'final_worlds_sha256')},
                arm=arm, status='graded', J_veh_s=fake_J(spec['world'], arm),
                safety=dict(collisions=0, teleports=0, hard_brakes=h % 7, ttc_events=h % 11))


class FakeRuntime:
    """No simulator, no process pool, no real ledger: fake physics for the dry run/tests."""
    physics = False

    def __init__(self, laneon=True):
        self.calls, self.laneon = [], laneon

    def config(self, root):
        cfg = read(Path(root)/'CONFIG.json')
        if Path(cfg['run_root']) != Path(root):
            raise Blocked('dry CONFIG must point at the dry root')
        return cfg

    def health(self, root):
        for name in ('FAULT.json', 'PAUSE_SAFETY.json', 'STOP.json'):
            if (Path(root)/name).exists():
                raise ValueError('dry health: '+name)

    def _reserve(self, root, jobs, phase, caps):
        rows = ledger_rows(root)
        ids = [j['id'] for j in jobs]
        if len(set(ids)) != len(ids) or set(ids) & {r['id'] for r in rows}:
            raise Blocked('duplicate charge/attempt')
        amount = sum(j['sim_s'] for j in jobs)
        if phase not in caps or sum(r['sim_s'] for r in rows if r['phase'] == phase)+amount > caps[phase]:
            raise Blocked('whole-wave budget blocked')
        with (Path(root)/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            for j in jobs:
                f.write(json.dumps(dict(id=j['id'], phase=phase, sim_s=j['sim_s']), sort_keys=True)+'\n')
        self.calls.append(('reserve', str(root), phase, amount))

    def reserve_clean3(self, cfg, jobs, phase):
        self._reserve(cfg['run_root'], jobs, phase, budget_phases(cfg))

    def reserve_final(self, final_root, budget, jobs, phase):
        self._reserve(final_root, jobs, phase, budget['phases'])

    def laneon_runnable(self):
        return self.laneon

    def run(self, items, cfg, workers, extra):
        self.calls.append(('run', len(items), workers))
        return [fake_record(spec, name) for name, spec, _ in items]


def build_fake_root(out, cfg, registration_bytes):
    """A disposable clean3-shaped root at u24: real CONFIG contract/budget/registration, fake
    checkpoints, sealed fake update/monitor receipts; no outcome of any real run is used."""
    from tools.coopmwm import rerun_cpus as rc
    root = Path(out)/'clean3_root'
    root.mkdir(parents=True)
    (root/'ISOLATED_REGISTRATION.json').write_bytes(registration_bytes)
    (root/'A_checkpoint_000.pt').write_bytes(b'dry-run fake common u0')
    cfg = dict(deepcopy(cfg), run_root=str(root), source_root=str(Path(out)/'no_source'),
               common_checkpoint_sha256=file_sha(root/'A_checkpoint_000.pt'))
    publish(root/'CONFIG.json', cfg)
    (root/'CHARGES.jsonl').touch()
    for arm in ARMS:
        (root/f'{arm}_checkpoint_{UPDATE:03}.pt').write_bytes(f'dry-run fake {arm} u24'.encode())
        publish(root/f'{arm}_UPDATE_{UPDATE:03}.json', seal(dict(arm=arm, update=UPDATE, dry_run=True), 'update_sha256'))
        publish(root/f'{arm}_COMPLETE.json', dict(arm=arm, updates=UPDATE, dry_run=True))
    zero = {k: 0 for k in SAFETY}
    for u in range(1, UPDATE+1):
        publish(root/f'LEARNING_MONITOR_u{u:03}.json', seal(dict(update=u, config_sha256=digest(cfg), dry_run=True,
            arms=[dict(arm=a, action='CONTINUE', metrics=dict(safety=dict(zero))) for a in ARMS]), 'receipt_sha256'))
    return root, cfg, rc.final_world_registration(json.loads(registration_bytes))


def fake_monitor(root, cfg, registration):
    """What the clean3 monitor writes after u24 (fake physics, real registered estimator)."""
    from tools.coopmwm import planner_eval as pe, track2_monitor as p3
    contract = cfg['final_evaluation']
    checkpoints = {arm: dict(path=str(Path(root)/f'{arm}_checkpoint_{UPDATE:03}.pt'),
                             file_sha256=file_sha(Path(root)/f'{arm}_checkpoint_{UPDATE:03}.pt'), update=UPDATE,
                             components=contract['checkpoint_components'][arm]) for arm in ARMS}
    specs = pe.prepare_track2(registration=registration, checkpoints=checkpoints, contract=contract)
    results = [fake_record(spec, spec['arm']) for spec, _ in specs]
    analysis = pe.analyse_track2(registration=registration, checkpoints=checkpoints, records=results, contract=contract)
    publish(Path(root)/PAIRED_FILE, seal(dict(update=UPDATE, status='graded', registration=registration,
        checkpoints=checkpoints, results=results, analysis=analysis, dry_run=True), 'receipt_sha256'))
    records = []
    for arm in ('B', 'R'):
        for i in range(5):
            gain = (3.0 if arm == 'B' else 0.5)+i*0.4
            records.append(dict(arm=arm, replicate=i, world_seed=986000+i, state_set_sha256=f's{i}', window_set_sha256=f'w{i}',
                                metric=gain, **{SYNERGY: gain, TOTAL_BENEFIT: 10+gain}, safety={k: 0 for k in SAFETY},
                                reference_safety={k: 0 for k in SAFETY}))
    analysis = p3._analyse(records, UPDATE, dict(p3.BOOTSTRAP))
    publish(Path(root)/REPLICATED_FILE, seal(dict(kind='replicated', update=UPDATE, status='complete', dry=False,
        records=records, dry_run=True, **analysis), 'receipt_sha256'))


def dry_run(*, config, registration_file, out, laneon=True):
    """Full launcher path with fake physics. Writes only under ``out``."""
    out = guard(Path(out))
    if out.exists() and any(out.iterdir()):
        raise Blocked('dry-run output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    root, cfg, registration = build_fake_root(out, read(config), guard(Path(registration_file)).read_bytes())
    runtime = FakeRuntime(laneon=laneon)
    final_root = out/'final_root'
    budget = register(root=root, final_root=final_root, runtime=runtime)
    publish(final_root/'OWNER_APPROVAL.json', dict(approved=True, budget_sha256=budget['receipt_sha256'],
            authority='DRY RUN ONLY - not an owner approval', dry_run=True))
    laneon_ck = out/'fake_laneon.pt'
    laneon_ck.write_bytes(b'dry-run stand-in, not the real aecd7d70 export')
    value = execute(root=root, final_root=final_root, runtime=runtime, poll=0, laneon_checkpoint=laneon_ck,
                    laneon_sha=file_sha(laneon_ck), dry_run=True,
                    on_gate=lambda: fake_monitor(root, cfg, registration))
    return dict(status='PASS', new_sim_s=0, physics=False, out=str(out), final_root=str(final_root),
                analysis_sha256=value['analysis_sha256'], present=value['present'], calls=runtime.calls,
                tables=str(final_root/'FINAL_TABLES.md'))


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('check', 'register', 'run', 'analyse', 'dry-run'))
    p.add_argument('--root', default=ROOT)
    p.add_argument('--final-root', default=FINAL_ROOT)
    p.add_argument('--baselines', default='MAPPO_U0,NATIVE,LANEON')
    p.add_argument('--allow-missing', default='')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--laneon-checkpoint', default=LANEON_CHECKPOINT)
    p.add_argument('--config', help='dry-run: CONFIG.json copy')
    p.add_argument('--registration', help='dry-run: ISOLATED_REGISTRATION.json copy')
    p.add_argument('--out', help='dry-run: empty output directory')
    a = p.parse_args(argv)
    names = tuple(x for x in a.baselines.split(',') if x)
    if set(names)-set(BASELINES):
        p.error('unknown baseline')
    if a.workers < 1 or a.workers > 24:
        p.error('workers must be 1..24 (16 monitor + 24 <= 40 SUMO workers on cpuV)')
    if a.command == 'dry-run':
        result = dry_run(config=a.config, registration_file=a.registration, out=a.out)
    elif a.command == 'check':
        result = check(root=a.root, final_root=a.final_root, runtime=Runtime())
    elif a.command == 'register':
        result = register(root=a.root, final_root=a.final_root, runtime=Runtime())
    elif a.command == 'analyse':
        runtime = Runtime()
        cfg = runtime.config(a.root)
        registration = panel(a.root, cfg)
        baselines, sources = {}, {}
        for name in names:
            path = Path(a.final_root)/f'BASELINE_{name}.json'
            if path.is_file():
                value = verify(read(path), 'receipt_sha256', path.name)
                baselines[name] = complete(value['records'], registration, name)
                sources[name] = dict(source=str(path), receipt_sha256=value['receipt_sha256'])
        if 'LANEON' in names and 'LANEON' not in baselines and (Path(a.root)/LANEON_MONITOR_FILE).is_file():
            baselines['LANEON'], sources['LANEON'] = adopt_laneon(a.root, registration)
        result = tables(root=a.root, final_root=a.final_root, cfg=cfg, registration=registration, runtime=runtime,
                        baselines=baselines, sources=sources,
                        missing={n: 'no sealed baseline receipt' for n in names if n not in baselines})
    else:
        import fcntl
        if not Path(a.final_root).is_dir():
            p.error('final root missing: run register first (it also holds the outputs)')
        with (Path(a.final_root)/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            publish(Path(a.final_root)/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                    replace=True)
            result = execute(root=a.root, final_root=a.final_root, runtime=Runtime(), baselines=names,
                             allow_missing=tuple(x for x in a.allow_missing.split(',') if x), workers=a.workers,
                             poll=a.poll, laneon_checkpoint=a.laneon_checkpoint)
    print(json.dumps(result if a.command != 'analyse' and a.command != 'run' else
                     dict(analysis_sha256=result['analysis_sha256'], present=result['present'],
                          missing=result['missing']), indent=1, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
