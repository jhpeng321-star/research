"""AMENDMENT 77 (S-MAPPO20C) evaluation: MAPPO20C vs MAPPO_CT vs U0 at D2800_p20 on ONE fresh 96-world panel.

Same code path as A52 / A54 / A66 (the A42 driver code cloned with the A77 names; ``A52.with_executor`` replaces only
the six executor fields of the U0 confirmation spec): every arm is the runner arm 'U0' (zero-residual u0, plant, 1,200 s,
control from 300 s, no hold executor) whose executor file is a sealed binding record (path + sha256, re-verified by the
runner at load). The panel is the cell D2800_p20 with the p20c plant / bindings (the A73 pattern: a process-local
nm_finaleval setting; the runner's seed table gains one block, ops/a77/host/patch_tables.py).

Arms (all run here; no stored arm is reused):
  U0        frozen speed-only MAPPO u0 export d9042cc5 (the paper's 20 % "MAPPO"; trained on the merge map)
  MAPPO_CT  A30 S-MAPPOCT policy_00012 069a4f85 (u0 + 12 updates x 8 episodes at D2800_p20; the paper's
            "MAPPO continued")
  MAPPO20C  the A77 training run's iteration-84 export (sha given at bind time and checked against the training
            run's own BOUND_00084.json receipt of the registered PLAN; no selection)
Fallback registered in advance: if the MAPPO_CT file cannot be retrieved before launch, the run uses --arms
U0,MAPPO20C (the contrasts with MAPPO_CT are then absent); nothing else changes.

Identity (before any confirmation episode; new host / tree): the A52 identity - the u0 export bound through this path
(IDENT_U0) on the 4 A42/A50 identity worlds (p20c 993100/993101, a35-D3200_p20 998100/998101) must reproduce the stored
A50 WAVE_A50_IDENT_U0 records exactly (J_veh_s difference 0, identical safety counters, same world); else STOP.

Budget: ONE ledger phase ``mappo20c_a77`` cap 480,000 sim-s (identity 4 x 1,200 = 4,800; confirmation 3 x 96 x 1,200
= 345,600; planned 350,400). No automatic replay: a failed dispatch raises and is reported.

Analysis (zero physics): the A42 driver's ``_contrast`` (paired percentile bootstrap 10,000, seed 20261104; diff =
treatment - reference, negative = less delay; total, five groups, CVaR90, hard brakes, TTC, collisions, teleports):
MAPPO20C - U0 (primary), MAPPO20C - MAPPO_CT, MAPPO_CT - U0.

Commands (python ops/a77_mappo20c.py <cmd> --final-root <root>):
  panel                      zero SUMO: build the panel file once (write-once; laptop, before registration)
  bind --which {MAPPO20C,MAPPO_CT,IDENT_U0} --policy F [--sha S --train-receipt BOUND_00084.json --train-plan PLAN.json]
  dry / status / budget
  identity ; identity-compare --a50-root <copy>      4 episodes ; IDENTITY_A77.json (PASS / FAILED)
  confirm [--arms U0,MAPPO_CT,MAPPO20C]              refused unless identity PASS
  analyse [--arms ...] --out <dir>                   A77_RESULTS_D2800_p20.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402   (_clone)
import a52_mappo60 as A52  # noqa: E402   (with_executor, _a50_records; never runs here)

AMENDMENT = '77'
CELL = 'D2800_p20'
SOURCE_SETTING = 'p20c'                    # the registered D2800_p20 plant / bindings (as A73)
SETTING = 'a77c_D2800_p20'
SEEDS = (997600, 997695)                   # 96 fresh worlds; 997696-997699 reserved unused
RUNNER_ATTR = 'A77_RANGES'                 # ((997600, 997699),) after ops/a77/host/patch_tables.py
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (987200, 987599))
A77_DIR = HERE/'a77'
PANEL_FILE = A77_DIR/f'nm_eval_panel_wz3to2_a77c_{CELL}.json'
#: pinned after the zero-SUMO panel build (ops/a77/nm_eval_panel_wz3to2_a77c_D2800_p20.json), before registration
PANEL_SHA256 = '2907e0aa042cd01a16158848db75818d64a3c31aa6946222d48a872cf72208fc'
PHASE = 'mappo20c_a77'
CAP_SIM_S = 480_000
CONFIRM_N = 96
IDENT_N = 4
#: the A77 training PLAN (Box /home/ubuntu/a77/run84/PLAN.json; registered in AMENDMENT 77 before any SUMO)
TRAIN_PLAN_SHA256 = '57f51364e2d56f89539d483fb445704c2bbb6e7fce8a824626082e25eb3d9f5d'
TRAIN_ITERATION = 84
MAPPO_CT_SHA256 = '069a4f85ab4ac7bdc5e01404dbe903f9b21957d5e1478c34d0929e5dcb2020c3'
#: MAPPO20C is not known before training ends: bind takes --sha (recorded in REGISTRATION.md by MAIN after training)
#: and accepts it only if it equals the file AND the training run's BOUND_00084.json of the registered PLAN.
PINS = dict(MAPPO20C=None, MAPPO_CT=MAPPO_CT_SHA256, IDENT_U0=N.FROZEN_MAPPO_SHA)
LABELS = dict(
    MAPPO20C='Amendment 77 run84 production policy_00084 (speed-only MAPPO, 20 % CAV, D2800_p20, 84 updates x 46 '
             'episodes from u0 d9042cc5, TRAIN 987200-987599; iteration fixed in advance)',
    MAPPO_CT='A30 S-MAPPOCT policy_00012 (speed-only MAPPO, u0 + 12 updates x 8 episodes at D2800_p20, TRAIN '
             '989200-989587)',
    IDENT_U0='u0 export d9042cc5 bound through the A77 path (identity check only)')
ARMS = ('U0', 'MAPPO_CT', 'MAPPO20C')
CONTRASTS = (('MAPPO20C', 'U0', 'primary'), ('MAPPO20C', 'MAPPO_CT', 'secondary'), ('MAPPO_CT', 'U0', 'context'))
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
IDENTITY_FILE = 'IDENTITY_A77.json'
A50_IDENT_WAVE = 'A50_IDENT_U0'
AUTHORITY = ('MAIN AMENDMENT 77 (S-MAPPO20C): MAPPO20C (the A77 training run, iteration 84) vs MAPPO_CT (A30) vs U0 at '
             'D2800_p20 on one fresh 96-world panel 997600-997695; same code path as A52/A54/A66; identity on the 4 '
             'A50 identity worlds; phase mappo20c_a77 cap 480,000')
DISCLOSURE = ('registered after the 20 % outcomes of U0 and MAPPO_CT (A28-A31 panels) and the 60 % outcome of MAPPO60C '
              '(A66) were known; MAPPO20C = 84 updates on top of the 169-iteration u0, iteration fixed before training; '
              'one panel, never pooled; MAPPO_CT had 96 training episodes, MAPPO20C 3,864')


def core():
    """The A42 driver code with the A77 names (ledger phase / cap / root / job ids / waves / runner seed table)."""
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A77',
        HOST_BASE=os.environ.get('A77_HOST_BASE', '/root/a77e'), CELL=CELL,
        VERIFY_SETTING=SETTING, CONFIRM_SETTING=SETTING,
        BANK_DIR='a77_banks', IDENT_WAVE='A77_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A77', JOB_PREFIX='a77',
        VERIFY_TAG='a77v', CONFIRM_TAG='a77c', IDENT_TAG='a77i', VERIFY_WAVE_PREFIX='A77V_', CONFIRM_WAVE_PREFIX='A77C_',
        RESULTS_FILE=f'A77_RESULTS_{CELL}.json', LOG_FILE='A77_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a77',
        RUNNER_RANGES_ATTR=RUNNER_ATTR, RESULT_LABEL='amendment 77; MAPPO comparators at 20 %; never pooled; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone('a77_core', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTING,)
    m.A42Runtime.phase = PHASE
    return m


D = core()


# ------------------------------------------------------------------------------------------------ setting / panel

def runner_block():
    from tools.coopmwm import nm_eval_runner as R
    table = getattr(R, RUNNER_ATTR, None)
    if table is None:
        raise N.Blocked(f'the runner has no {RUNNER_ATTR} (apply ops/a77/host/patch_tables.py to a runner copy)')
    (r0, r1), = table
    lo, hi = SEEDS
    if not r0 <= lo <= hi <= r1 or (r0, r1) != (997600, 997699):
        raise N.Blocked(f'{lo}-{hi} is not inside the registered runner block 997600-997699 ({r0}-{r1})')
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise N.Blocked('forbidden seed range')
    return r0, r1


def register_setting(pinned=True):
    """Process-local nm_finaleval setting (nm_finaleval unchanged): the p20c cell / plant / bindings with the A77
    block (= ops/a73_lcside20.register_setting)."""
    r0, r1 = runner_block()
    lo, hi = SEEDS
    N.SETTINGS[SETTING] = dict(
        N.SETTINGS[SOURCE_SETTING],
        N_WORLDS=hi-lo+1, EVAL_RANGE=(lo, r1), EVAL_EXTRA_RANGE=(lo, r1), SEEDS_RESERVED=(hi+1, r1),
        PANEL_SIZE_REASON=f'MAIN amendment 77: MAPPO comparator panel = {hi-lo+1} fresh worlds {lo}-{hi} (D2800_p20)',
        CELL_AUTHORITY='MAIN amendment 77 (S-MAPPO20C): MAPPO20C vs MAPPO_CT vs U0 at D2800_p20',
        PANEL_AUTHORITY=(f'A77 amendment 77: audited integers {lo}-{hi} (runner block {r0}-{r1}) for the MAPPO '
                         'comparator panel ONLY; executed through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(checked=('amendment-77 audit: ops/a77/SEED_AUDIT.json',),
                            disclosed='; amendment 77: fresh integers, registered at runtime by the runner'),
        PANEL_FILE=PANEL_FILE, PANEL_3C_FILE=None)
    if SETTING not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(SETTING,)
    pin = PANEL_SHA256 if pinned and len(PANEL_SHA256) == 64 else None
    N.SETTINGS_PINS[SETTING] = dict(PANEL_SHA256=pin, PANEL_3C_SHA256=None)
    return SETTING


def setup(*, need_panel=True):
    register_setting(pinned=need_panel)
    N.apply_setting(SETTING)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    if not need_panel:
        return None
    if len(PANEL_SHA256) != 64:
        raise N.Blocked('PANEL_SHA256 is not pinned in ops/a77_mappo20c.py')
    panel = N.load_panel(PANEL_FILE)
    got = [w['world']['seed'] for w in panel['worlds']]
    if got != list(range(SEEDS[0], SEEDS[1]+1)) or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'panel {PANEL_FILE.name} is not {CELL} {SEEDS[0]}-{SEEDS[1]}')
    if dict(panel['contract']['bootstrap']) != BOOTSTRAP:
        raise N.Blocked('panel bootstrap is not the registered one')
    return panel


# ------------------------------------------------------------------------------------------------ binding

def binding_file(which):
    if which not in PINS:
        raise N.Blocked(f'which is one of {sorted(PINS)}')
    return f'BOUND_A77_{which}.json'


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def train_receipt_check(sha, receipt, plan):
    """MAPPO20C: the file must be the A77 run's iteration-84 export, bound by launch.py bind of the REGISTERED PLAN."""
    rec = json.loads(Path(receipt).read_text(encoding='utf-8'))
    plan_sha = sha_file(plan)
    if len(TRAIN_PLAN_SHA256) != 64 or plan_sha != TRAIN_PLAN_SHA256:
        raise N.Blocked(f'training PLAN {plan} hashes to {plan_sha[:12]}, not the registered A77 PLAN')
    if rec.get('plan_sha256') != TRAIN_PLAN_SHA256 or int(rec.get('iteration', -1)) != TRAIN_ITERATION:
        raise N.Blocked(f'{receipt}: not the iteration-{TRAIN_ITERATION} receipt of the registered A77 PLAN')
    if rec.get('sha256') != sha or rec.get('lane_control') != 'native':
        raise N.Blocked(f'{receipt}: sha / lane control differ from the file being bound')
    return dict(receipt=str(receipt), receipt_sha256=sha_file(receipt), plan_sha256=plan_sha,
                iteration=TRAIN_ITERATION)


def bind(final_root, which, policy, *, check=None, sha=None, receipt=None, plan=None):
    """Seal BOUND_A77_<which>.json (A52.bind / A66.bind; write once; zero physics)."""
    final_root, policy = Path(final_root), Path(policy)
    target = final_root/binding_file(which)
    if target.exists():
        raise N.Blocked(f'{target}: exists (bound once; never rebound)')
    got = N.file_sha(policy)
    want = PINS[which]
    train = None
    if which == 'MAPPO20C':
        if not sha or len(sha) != 64 or not receipt or not plan:
            raise N.Blocked('MAPPO20C: --sha (as recorded in REGISTRATION.md), --train-receipt and --train-plan required')
        want = sha
        train = train_receipt_check(sha, receipt, plan)
    if got != want:
        raise N.Blocked(f'{policy} hashes to {got[:12]}, not the registered {str(want)[:12]}')
    meta = N.export_meta(policy)
    if meta.get('trainer') != 'mappo' or meta.get('lane_head') or meta.get('lane_control') != 'native':
        raise N.Blocked(f'{which}: a speed-only mappo export (native lanes, no lane head) is required: {meta}')
    value = N.seal(dict(schema=N.SCHEMA+'.comparator', arm='U0' if which == 'IDENT_U0' else which, kind=f'a77_{which}',
                        label=LABELS[which], path=str(policy), sha256=got, export=meta, authority=AUTHORITY,
                        rule='A77 (= A52 path): U0 arm with only the executor file replaced (path + sha256 verified '
                             'at every load)', load_check=check, training=train, amendment=AMENDMENT), 'binding_sha256')
    N.publish(target, value)
    return dict(value, binding_path=str(target))


def read_bound(final_root, which):
    path = Path(final_root)/binding_file(which)
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run bind --which {which} first)')
    value = N.verify(N.read(path), 'binding_sha256', path.name)
    if which == 'MAPPO20C':
        if (value.get('training') or {}).get('plan_sha256') != TRAIN_PLAN_SHA256:
            raise N.Blocked(f'{path.name}: not bound to the registered A77 training PLAN')
        want = value['sha256']
    else:
        want = PINS[which]
    if not want or value['sha256'] != want or value['kind'] != f'a77_{which}':
        raise N.Blocked(f'{path.name}: not the registered {which} executor')
    return dict(value, binding_path=str(path))


# ------------------------------------------------------------------------------------------------ specs

def confirm_wave(arm):
    return f'A77C_{arm}'


def confirm_specs(arm, panel, final_root):
    """U0: the A42-core U0 confirmation specs as they are (ids a77c_u0_*); MAPPO_CT / MAPPO20C: the same specs through
    A52.with_executor (ids a77c_<arm>_*)."""
    if arm not in ARMS:
        raise N.Blocked(f'no A77 arm {arm!r}')
    base = D.confirm_specs('U0', panel, final_root)
    if arm == 'U0':
        if any(s['checkpoint'].get('executor_sha256') != N.FROZEN_MAPPO_SHA for s in base):
            raise N.Blocked('U0: executor is not the frozen MAPPO u0')
        return base
    return A52.with_executor(base, read_bound(final_root, arm), f'{D.CONFIRM_TAG}_{arm.lower()}')


def ident_specs(final_root):
    """D.ident_specs (the 4 A42/A50 identity worlds) with the IDENT_U0 bound executor (= A52.ident_specs)."""
    return A52.with_executor(D.ident_specs(final_root), read_bound(final_root, 'IDENT_U0'))


def parse_arms(text):
    arms = tuple(a for a in (text or ','.join(ARMS)).split(',') if a)
    if arms not in (ARMS, ('U0', 'MAPPO20C')):
        raise N.Blocked(f'--arms must be {",".join(ARMS)} or the registered fallback U0,MAPPO20C; got {arms}')
    return arms


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, a50_root):
    specs = ident_specs(final_root)
    got, _, cov = N.published(final_root, D.IDENT_WAVE, specs, D.C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{D.IDENT_WAVE}: {cov}/{len(specs)} published')
    ref = A52._a50_records(a50_root, A50_IDENT_WAVE)
    rows = []
    for r in got:
        a = ref.get(r['world_id'])
        if a is None:
            raise N.Blocked(f'{r["world_id"]}: not an A50 identity world')
        rows.append(dict(world_id=r['world_id'], j_a77=r['J_veh_s'], j_a50=a['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a77=r['safety'], safety_a50=a['safety'],
                         safety_equal=r['safety'] == a['safety'], executor_loaded=r.get('executor_sha256_loaded'),
                         same_world=r['world'] == a['world'],
                         vehicle_log_equal=(r.get('vehicle_log') or {}).get('sha256')
                         == (a.get('vehicle_log') or {}).get('sha256')))
    ok = (len(rows) == IDENT_N and len({x['world_id'] for x in rows}) == IDENT_N
          and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] for x in rows))
    return N.seal(dict(schema='coopmwm.a77.identity.v1', amendment=AMENDMENT,
                       rule='A77 path (bind + with_executor) given u0 d9042cc5 reproduces the A50 WAVE_A50_IDENT_U0 '
                            'records exactly: J_veh_s difference 0 and identical safety counters on all 4 worlds',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: the panel may run; FAILED: stop and report, nothing else is run'),
                  'receipt_sha256')


def identity_passed(final_root):
    p = Path(final_root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ analysis

def analyse(final_root, panel, arms=ARMS):
    import numpy as np
    from a32_ablation import report_of
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    have, sources = {}, {}
    for arm in arms:
        have[arm], sources[arm] = D.C37._load(final_root, confirm_wave(arm), confirm_specs(arm, panel, final_root))
    reports = {a: {w: report_of(final_root, have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = []
    for a, ref, kind in CONTRASTS:
        if a in have and ref in have:
            rows.append(D._contrast(have, reports, worlds, a, ref, bootstrap, kind))
    means = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                     safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
             for a in have}
    ci = {(c['treatment'], c['reference']): c['total_minus_ref']['ci95'] for c in rows}
    reading = {f'{a}-{r}': dict(ci95=list(ci[(a, r)]), saves=bool(ci[(a, r)][1] < 0), worse=bool(ci[(a, r)][0] > 0))
               for a, r in ci}
    return N.seal(dict(schema='coopmwm.a77.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, arms_run=list(arms),
                       executors={a: dict(sha256=read_bound(final_root, a)['sha256'] if a != 'U0'
                                          else N.FROZEN_MAPPO_SHA, binding=sources[a]) for a in arms},
                       contrasts=rows, arms=means,
                       reading=dict(rule='per contrast: "saves" iff total-delay CI95 upper < 0, "worse" iff CI95 lower '
                                         '> 0 (diff = treatment - reference, negative = less delay); MAPPO20C - U0 '
                                         'is the primary contrast; nothing is pooled', per_contrast=reading),
                       identity=N.read(final_root/IDENTITY_FILE).get('receipt_sha256'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan(arms=ARMS):
    conf = len(arms)*CONFIRM_N*N.HORIZON_S
    ident = IDENT_N*N.HORIZON_S
    return dict(phase=PHASE, cap_sim_s=CAP_SIM_S, identity=ident, confirmation=conf, planned=ident+conf,
                margin=CAP_SIM_S-ident-conf, arms=list(arms))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('panel', 'bind', 'dry', 'identity', 'identity-compare', 'confirm', 'analyse',
                                       'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--which', choices=sorted(PINS))
    p.add_argument('--policy')
    p.add_argument('--sha')
    p.add_argument('--train-receipt')
    p.add_argument('--train-plan')
    p.add_argument('--a50-root')
    p.add_argument('--map-spec')
    p.add_argument('--arms')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    arms = parse_arms(a.arms)
    if a.command == 'budget':
        print(json.dumps(budget_plan(arms), indent=1))
        return
    if a.command == 'panel':
        setup(need_panel=False)
        import a37_celltailor as C37
        from datetime import datetime, timezone
        value = N.build_panel(a.map_spec or N.default_spec_path(), datetime.now(timezone.utc).isoformat())
        C37.write_once(PANEL_FILE, json.dumps(value, indent=1, sort_keys=True)+'\n')
        print(json.dumps(dict(panel_sha256=value['panel_sha256'], file=str(PANEL_FILE), cell=value['cell']['cell_key'],
                              seeds=[value['worlds'][0]['world']['seed'], value['worlds'][-1]['world']['seed']],
                              n=len(value['worlds']))))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    if a.command == 'bind':
        from tools.coopmwm.nm_eval_runner import mappo_load_check
        sha = a.sha if a.which == 'MAPPO20C' else PINS[a.which]
        v = bind(root, a.which, a.policy, check=mappo_load_check(a.policy, sha), sha=a.sha,
                 receipt=a.train_receipt, plan=a.train_plan)
        print(json.dumps(dict(which=a.which, binding_sha256=v['binding_sha256'], sha256=v['sha256'],
                              export=v['export'])))
        return
    panel = setup()

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        ident = ident_specs(root)
        conf = {arm: confirm_specs(arm, panel, root) for arm in arms
                if arm == 'U0' or (root/binding_file(arm)).is_file()}
        print(json.dumps(dict(
            budget=budget_plan(arms), panel_sha256=panel['panel_sha256'],
            ident=[(s['job_id'], s['world_id'], s['checkpoint'].get('executor_sha256')) for s in ident],
            confirm={arm: dict(n=len(c), first=c[0]['job_id'], last=c[-1]['job_id'], runner_arm=c[0]['arm'],
                               seeds=[c[0]['world']['seed'], c[-1]['world']['seed']], cell=c[0]['world']['cell_key'],
                               executor=c[0]['checkpoint']['executor_sha256']) for arm, c in conf.items()},
            unbound=[arm for arm in arms if arm not in conf], charged=D.charged(root)), indent=1))
    elif a.command in ('identity', 'confirm'):
        import fcntl
        if a.command == 'confirm' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(D.IDENT_WAVE, 'episode', ident_specs(root))]
            else:
                waves = [(confirm_wave(arm), 'episode', confirm_specs(arm, panel, root)) for arm in arms]
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log,
                            panel=None if a.command == 'identity' else panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        if not a.a50_root:
            p.error('identity-compare needs --a50-root')
        v = identity_compare(root, a.a50_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['world_id'], x['abs_diff'], x['safety_equal'],
                                                            x['same_world']) for x in v['worlds']])))
    elif a.command == 'analyse':
        v = analyse(root, panel, arms)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], reading=v['reading'],
                              means={k: round(x['mean_J_veh_s'], 1) for k, x in v['arms'].items()})))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root))
        for arm in arms:
            if arm == 'U0' or (root/binding_file(arm)).is_file():
                specs = confirm_specs(arm, panel, root)
                out[confirm_wave(arm)] = f'{N.published(root, confirm_wave(arm), specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
