"""AMENDMENT 52 (registered; MAIN 2026-11-11): MAPPO60 = the A50 confirmation U0 arm with a different executor file.

MAPPO60 is the U0 arm of ops/a50_fullp60.py (same runner arm 'U0', same zero-residual u0, same plant, same panel
a50c_D2800_p60 = worlds 996800-996895, same 1,200 s episodes, control from 300 s, no hold executor on top) whose
executor is a parameter: a sealed binding record (path + sha256, re-verified by the runner at load; the runner's
comparator_setup accepts exactly that sha). Nothing else in the episode spec changes (``with_executor`` replaces only
the six executor fields of the U0 checkpoint descriptor). The policy is S-MAPPOWZ run2 policy_00048.pt
(sha256 7f0c8869...364443); no other checkpoint is evaluated.

Identity (before any MAPPO60 episode): the same code path (bind -> with_executor) given the u0 export d9042cc5 on the
4 worlds A50 used for WAVE_A50_IDENT_U0 must reproduce the A50 records exactly (J_veh_s and the four safety counters).

Budget: ledger phase ``mappo60_a52`` cap 130,000 sim-s (identity 4 x 1,200 + 96 x 1,200 = 120,000; no automatic
replay: a failed dispatch stops and is reported). Own final root /root/a52/root, job ids a52*, stop file STOP_A52.

Analysis (zero physics): the A50 driver's own ``_contrast`` (paired percentile bootstrap 10,000, seed 20261104;
diff = treatment - reference, negative = less delay; groups; CVaR90 tail; safety) for MAPPO60 - U0,
DUO_HOLD - MAPPO60, BIG_BANK - MAPPO60, LEARNED_TRIGGER - MAPPO60. The A50 arms are read from a read-only COPY of the
finished A50 root (--a50-root; its records still name the host root /root/a50v/root).

Commands (python ops/a52_mappo60.py <cmd> --final-root <root>):
  bind --which {MAPPO60,IDENT_U0} --policy <file>   seal BOUND_A52_<which>.json (zero physics)
  dry                                                spec summary + budget (zero physics)
  identity / identity-compare --a50-root <copy>      4 episodes; IDENTITY_A52.json (PASS / FAILED)
  confirm                                            MAPPO60 on the 96 worlds (refused unless identity PASS)
  analyse --a50-root <copy> --out <dir>              A52_RESULTS_D2800_p60.json
  status / budget
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402   (the A50 driver: its D reads the A50 arms; never runs here)

AMENDMENT = '52'
CELL = 'D2800_p60'
ARM = 'MAPPO60'
MAPPO60_SHA256 = '7f0c88695161bae6a243801df94e92fbd27c87551d655be61d338fd369364443'
U0_SHA256 = N.FROZEN_MAPPO_SHA                                           # d9042cc5 (identity check only)
PINS = dict(MAPPO60=MAPPO60_SHA256, IDENT_U0=U0_SHA256)
LABELS = dict(MAPPO60='S-MAPPOWZ run2 policy_00048 (speed-only MAPPO, 60 % CAV, trained D3200_p60, it48)',
              IDENT_U0='u0 export d9042cc5 bound through the A52 path (identity check only)')
CAP_SIM_S = 130_000
IDENT_N = 4
CONFIRM_N = 96
EXECUTOR_FIELDS = ('comparator_kind', 'label', 'executor_path', 'executor_sha256', 'binding_sha256', 'binding_path')
A50_ROOT_ON_HOST = '/root/a50v/root'
A50_IDENT_WAVE = 'A50_IDENT_U0'
CONTRASTS = ((ARM, 'U0'), ('DUO_HOLD', ARM), ('BIG_BANK', ARM), ('LEARNED_TRIGGER', ARM))
AUTHORITY = ('MAIN AMENDMENT 52 (owner 2026-11-11: "can we just use the previously trained one?"): the existing '
             'S-MAPPOWZ run2 policy_00048 (7f0c8869) as one extra arm MAPPO60 on the A50 confirmation worlds '
             '996800-996895 (D2800_p60), paired with U0, DUO_HOLD, LEARNED_TRIGGER, BIG_BANK; phase a52 cap 130,000')
DISCLOSURE = ('trained at 3,200 veh/h, evaluated at 2,800; training stopped early (48 of 168 planned iterations on top '
              'of the 169-iteration u0); the earlier record labelled this run "NOT a comparator" for the 20 % study; '
              'paired with A50 arms on the same worlds; not an independent panel; never pooled')
OVERRIDES = dict(
    A42_PHASE='mappo60_a52', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A52',
    HOST_BASE=os.environ.get('A52_HOST_BASE', '/root/a52'),
    VERIFY_SETTING='a50v_D2800_p60', CONFIRM_SETTING='a50c_D2800_p60',
    BANK_DIR='a52_banks', IDENT_WAVE='A52_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A52', JOB_PREFIX='a52',
    VERIFY_TAG='a52v', CONFIRM_TAG='a52c', IDENT_TAG='a52i', VERIFY_WAVE_PREFIX='A52V_', CONFIRM_WAVE_PREFIX='A52C_',
    RESULTS_FILE=f'A52_RESULTS_{CELL}.json', LOG_FILE='A52_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a52',
    RUNNER_RANGES_ATTR='A50_RANGES', RESULT_LABEL='amendment 52; extra arm on the A50 panel; never pooled; one shot',
    AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a52_mappo60_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()            # the A42 driver code with the A52 names (ledger phase / cap / job ids / waves / stop file)
D50 = A50.D             # the A50 driver (read-only use: its confirmation specs and wave names)
CONFIRM_WAVE = f'A52C_{ARM}'
IDENTITY_FILE = 'IDENTITY_A52.json'


# ------------------------------------------------------------------------------------------------ binding

def binding_file(which):
    if which not in PINS:
        raise N.Blocked(f'which is one of {sorted(PINS)}')
    return f'BOUND_A52_{which}.json'


def bind(final_root, which, policy, *, check=None):
    """Seal BOUND_A52_<which>.json: the executor file (path + registered sha256, mappo trainer, speed-only export).
    Zero physics. Never rebinds (write once)."""
    final_root, policy = Path(final_root), Path(policy)
    target = final_root/binding_file(which)
    if target.exists():
        raise N.Blocked(f'{target}: exists (bound once; never rebound)')
    sha = N.file_sha(policy)
    if sha != PINS[which]:
        raise N.Blocked(f'{policy} hashes to {sha[:12]}, not the registered {PINS[which][:12]}')
    meta = N.export_meta(policy)
    if meta.get('trainer') != 'mappo' or meta.get('lane_head') or meta.get('lane_control') != 'native':
        raise N.Blocked(f'{which}: a speed-only mappo export (native lanes, no lane head) is required: {meta}')
    value = N.seal(dict(schema=N.SCHEMA+'.comparator', arm=ARM if which == 'MAPPO60' else 'U0', kind=f'a52_{which}',
                        label=LABELS[which], path=str(policy), sha256=sha, export=meta, authority=AUTHORITY,
                        rule='A52: U0 arm with only the executor file replaced (path + sha256 verified at every load)',
                        load_check=check, amendment=AMENDMENT), 'binding_sha256')
    N.publish(target, value)
    return dict(value, binding_path=str(target))


def read_bound(final_root, which):
    path = Path(final_root)/binding_file(which)
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run bind --which {which} first)')
    value = N.verify(N.read(path), 'binding_sha256', path.name)
    if value['sha256'] != PINS[which] or value['kind'] != f'a52_{which}':
        raise N.Blocked(f'{path.name}: not the registered {which} executor')
    return dict(value, binding_path=str(path))


def with_executor(specs, exe, tag=None):
    """The registered U0 specs with ONLY the executor fields of the checkpoint descriptor replaced (the runner arm,
    zero-residual u0, world, window, capture and sim_s unchanged); job ids <tag>_<i> (tag None: ids kept)."""
    out = []
    for i, s in enumerate(specs):
        if s['arm'] != 'U0' or s['checkpoint'].get('kind') != 'u0':
            raise N.Blocked(f'{s["job_id"]}: only U0 specs take a replaced executor')
        ck = deepcopy(s['checkpoint'])
        ck.update(comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                  executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'],
                  binding_path=exe['binding_path'])
        out.append(dict(s, checkpoint=ck, job_id=s['job_id'] if tag is None else f'{tag}_{i:04}'))
    return out


def ident_specs(final_root):
    """D.ident_specs (the 4 A50 identity worlds: p20c 993100/993101, a35-D3200 998100/998101) with the IDENT_U0 bound
    executor through with_executor."""
    return with_executor(D.ident_specs(final_root), read_bound(final_root, 'IDENT_U0'))


def confirm_specs(panel, final_root):
    """U0 confirmation specs of the A50 panel (T._arm_specs 'U0', same as A50) -> MAPPO60 executor, ids a52c_mappo60_*."""
    base = D.confirm_specs('U0', panel, final_root)
    return with_executor(base, read_bound(final_root, 'MAPPO60'), f'{D.CONFIRM_TAG}_{ARM.lower()}')


# ------------------------------------------------------------------------------------------------ identity

def _a50_records(a50_root, wave):
    """{world_id: record} from the sealed chunk / whole-wave receipts of a COPY of the A50 root (receipt seals
    verified; record checkpoints are the host's)."""
    a50_root = Path(a50_root)
    files = sorted(a50_root.glob(f'WAVE_{wave}__*.json')) or sorted(a50_root.glob(f'WAVE_{wave}.json'))
    if not files:
        raise N.Blocked(f'{a50_root}: no WAVE_{wave} receipts')
    out = {}
    for p in files:
        v = N.verify(N.read(p), 'receipt_sha256', p.name)
        for r in v['records']:
            out[r['world_id']] = r
    return out


def identity_compare(final_root, a50_root):
    specs = ident_specs(final_root)
    got, _, cov = N.published(final_root, D.IDENT_WAVE, specs, D.C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{D.IDENT_WAVE}: {cov}/{len(specs)} published')
    ref = _a50_records(a50_root, A50_IDENT_WAVE)
    rows = []
    for r in got:
        a = ref.get(r['world_id'])
        if a is None:
            raise N.Blocked(f'{r["world_id"]}: not an A50 identity world')
        rows.append(dict(world_id=r['world_id'], j_a52=r['J_veh_s'], j_a50=a['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a52=r['safety'], safety_a50=a['safety'],
                         safety_equal=r['safety'] == a['safety'],
                         executor_loaded=r.get('executor_sha256_loaded'),
                         same_world=r['world'] == a['world'],
                         vehicle_log_equal=r['vehicle_log'].get('sha256') == a.get('vehicle_log', {}).get('sha256')))
    ok = len(rows) == IDENT_N and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] for x in rows)
    return N.seal(dict(schema='coopmwm.a52.identity.v1', amendment=AMENDMENT,
                       rule='A52 path (bind + with_executor) given u0 d9042cc5 reproduces the A50 WAVE_A50_IDENT_U0 '
                            'records exactly: J_veh_s difference 0 and identical safety counters on all 4 worlds',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: MAPPO60 may run; FAILED: stop and report, MAPPO60 is not run'),
                  'receipt_sha256')


def identity_passed(final_root):
    p = Path(final_root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ analysis

@contextmanager
def host_root_view(host_root):
    """Read an offloaded copy whose records name ``host_root`` (nm_finaleval.binding_location remap), scoped."""
    old = os.environ.get(N.HOST_FINAL_ROOT_ENV)
    os.environ[N.HOST_FINAL_ROOT_ENV] = host_root
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(N.HOST_FINAL_ROOT_ENV, None)
        else:
            os.environ[N.HOST_FINAL_ROOT_ENV] = old


def load_a50_arms(panel, a50_root, host_root=A50_ROOT_ON_HOST):
    have, roots, sources = {}, {}, {}
    with host_root_view(host_root):
        for arm in D50.CONFIRM_ARMS:
            specs = D50.confirm_specs(arm, panel, a50_root, check_files=False)
            have[arm], sources[arm] = D50.C37._load(a50_root, D50.confirm_wave(arm), specs)
            roots[arm] = Path(a50_root)
    return have, roots, sources


def analyse(final_root, a50_root, panel, *, host_root=A50_ROOT_ON_HOST):
    import numpy as np
    from a32_ablation import report_of
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    have, roots, sources = {}, {}, {}
    have[ARM], sources[ARM] = D.C37._load(final_root, CONFIRM_WAVE, confirm_specs(panel, final_root))
    roots[ARM] = final_root
    h, r, s = load_a50_arms(panel, a50_root, host_root)
    have.update(h)
    roots.update(r)
    sources.update(s)
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = [D._contrast(have, reports, worlds, a, ref, bootstrap, 'a52') for a, ref in CONTRASTS]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in (ARM,)+tuple(D50.CONFIRM_ARMS)}
    hold = {f'{a}-{ARM}': bool(c['total_minus_ref']['ci95'][1] < 0)
            for c in contrasts for a in ('DUO_HOLD', 'BIG_BANK') if (c['treatment'], c['reference']) == (a, ARM)}
    return N.seal(dict(schema='coopmwm.a52.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, executor=dict(sha256=MAPPO60_SHA256, binding=sources[ARM]),
                       contrasts=contrasts, arms=arms,
                       reading=dict(rule='"MAPPO alone does not find the cooperation" is supported only if the hold '
                                         'arms remain below MAPPO60 with CI95 upper < 0',
                                    ci_upper_below_zero=hold, supported=all(hold.values()) and len(hold) == 2),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       a50_root_copy=str(a50_root), label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    planned = (IDENT_N+CONFIRM_N)*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, identity=IDENT_N*N.HORIZON_S,
                confirmation=CONFIRM_N*N.HORIZON_S, planned=planned, margin=CAP_SIM_S-planned)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('bind', 'dry', 'identity', 'identity-compare', 'confirm', 'analyse', 'status',
                                       'budget'))
    p.add_argument('--final-root')
    p.add_argument('--which', choices=sorted(PINS))
    p.add_argument('--policy')
    p.add_argument('--a50-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=97)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    if a.command == 'bind':
        from tools.coopmwm.nm_eval_runner import mappo_load_check
        check = mappo_load_check(a.policy, PINS[a.which])
        v = bind(root, a.which, a.policy, check=check)
        print(json.dumps(dict(which=a.which, binding_sha256=v['binding_sha256'], sha256=v['sha256'],
                              export=v['export'])))
        return
    setting = D.VERIFY_SETTING if a.command in ('identity', 'identity-compare') else D.CONFIRM_SETTING
    panel = D.setup(setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        ident = ident_specs(root)
        D.setup(D.CONFIRM_SETTING)
        conf = confirm_specs(N.load_panel(N.PANEL_FILE), root)
        print(json.dumps(dict(budget=budget_plan(), ident=[(s['job_id'], s['world_id']) for s in ident],
                              confirm=dict(n=len(conf), first=conf[0]['job_id'], last=conf[-1]['job_id'],
                                           seeds=[conf[0]['world']['seed'], conf[-1]['world']['seed']],
                                           cell=conf[0]['world']['cell_key'],
                                           executor=conf[0]['checkpoint']['executor_sha256']),
                              charged=D.charged(root)), indent=1))
    elif a.command in ('identity', 'confirm'):
        import fcntl
        if a.command == 'confirm' and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: MAPPO60 is not run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(D.IDENT_WAVE, 'episode', ident_specs(root))]
            else:
                waves = [(CONFIRM_WAVE, 'episode', confirm_specs(panel, root))]
            need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay (A52 brief): a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log,
                            panel=None if a.command == 'identity' else panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, a.a50_root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['world_id'], x['abs_diff'], x['safety_equal'])
                                                            for x in v['worlds']])))
    elif a.command == 'analyse':
        v = analyse(root, a.a50_root, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], reading=v['reading'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root))
        for name, specs in ((CONFIRM_WAVE, confirm_specs(panel, root)),):
            out[name] = f'{N.published(root, name, specs, D.C37.check_a37)[2]}/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
