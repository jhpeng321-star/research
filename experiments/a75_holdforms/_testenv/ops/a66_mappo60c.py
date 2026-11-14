"""AMENDMENT 66 (registered; MAIN 2026-11-12): MAPPO60C on both panels, MAPPO60 / MAPPO60B on panel 1.

Same code path as A52 / A54 (c4/a54-mappo60b 3c3bae51): the confirmation U0 arm of the panel (runner arm 'U0',
zero-residual u0, plant, 1,200 s, control from 300 s, no hold executor) whose executor is a sealed binding record
(A52.with_executor replaces only the six executor fields). The only parameters are the policy file and the panel.

Panels (never pooled):
  p1 = the A42 confirmation panel a42c_D2800_p60 (worlds 998600-998695): MAPPO60C, MAPPO60, MAPPO60B are run;
       stored arms U0, DUO_HOLD, BIG_BANK come from a read-only copy of the A42 root (records name /root/a42v/root).
  p2 = the A50 confirmation panel a50c_D2800_p60 (worlds 996800-996895): MAPPO60C is run; stored arms U0, DUO_HOLD,
       LEARNED_TRIGGER, BIG_BANK (A50 root), MAPPO60 (A52 root), MAPPO60B (A54 root) from read-only copies.

Policies: MAPPO60C = the Amendment 65 final policy (iteration 84; sha pinned below after training ends and before
any evaluation second), MAPPO60 = 7f0c8869 (A52), MAPPO60B = 30971d7e (A54), IDENT_U0 = u0 export d9042cc5.

Identity (before any confirmation episode of a panel; new host): 4 worlds per panel = the first four panel worlds;
U0 (u0 export bound through this path) on worlds 0-1 and one stored arm on worlds 2-3 (p1: DUO_HOLD through the
driver's own DUO_HOLD specs; p2: MAPPO60B through this path). The new records must equal the stored records exactly
(J_veh_s difference 0, identical safety counters, same world); otherwise FAILED and nothing else runs on that panel.
8 episodes in total = 9,600 sim-s (the registered identity budget).

Budget: ONE ledger phase ``mappo60c_a66`` cap 480,000 sim-s on ONE final root (p1 3 x 96 x 1,200 = 345,600;
p2 115,200; identity 9,600; margin 9,600). No automatic replay: a failed dispatch raises and is reported.

Analysis (zero physics; per panel): the A42 driver's own ``_contrast`` (paired percentile bootstrap 10,000, seed
20261104; diff = treatment - reference, negative = less delay; total, five groups, CVaR90, safety counters) for every
X run on the panel: X - U0, DUO_HOLD - X, BIG_BANK - X (p2 also LEARNED_TRIGGER - X); descriptive
MAPPO60C - MAPPO60B and MAPPO60C - MAPPO60. Registered readings per X.

Commands (python ops/a66_mappo60c.py <cmd> --panel {p1,p2} --final-root <root>):
  bind --which {MAPPO60C,MAPPO60,MAPPO60B,IDENT_U0} --policy <file>   seal BOUND_A66_<which>.json (no --panel)
  dry / status / budget
  load-refs <refs>                  zero physics: every stored arm of the panel loads 96/96
  selfcheck <refs> --a54-results F  p2 only, zero physics: this analysis path reproduces the A54 MAPPO60B contrasts
  identity ; identity-compare <refs>   4 episodes ; IDENTITY_A66_<panel>.json (PASS / FAILED)
  confirm                           the panel's MAPPO arms on its 96 worlds (refused unless identity PASS)
  analyse <refs> --out <dir>        A66_RESULTS_<panel>_D2800_p60.json
  <refs> = --a42-root (p1)  |  --a50-root --a52-root --a54-root (p2)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a50_fullp60 as A50  # noqa: E402
import a52_mappo60 as A52  # noqa: E402   (with_executor, host_root_view, stored A50 arms; never runs here)
import a54_mappo60b as A54  # noqa: E402  (stored A52 / A54 arms; never runs here)

AMENDMENT = '66'
CELL = 'D2800_p60'
#: the Amendment 65 final policy (run84 production policy_00084.pt); pinned after training ends, before bind
MAPPO60C_SHA256 = "2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8"  # A65 policy_00084.pt (fixed final iteration)
PINS = dict(MAPPO60C=MAPPO60C_SHA256, MAPPO60=A52.MAPPO60_SHA256, MAPPO60B=A54.MAPPO60B_SHA256,
            IDENT_U0=N.FROZEN_MAPPO_SHA)
LABELS = dict(
    MAPPO60C='Amendment 65 run84 production policy_00084 (speed-only MAPPO, 60 % CAV, D2800_p60, 84 updates from u0 '
             'd9042cc5; iteration fixed in advance)',
    MAPPO60=A52.LABELS['MAPPO60'], MAPPO60B=A54.LABEL_TEXT,
    IDENT_U0='u0 export d9042cc5 bound through the A66 path (identity check only)')
CAP_SIM_S = 480_000
PHASE = 'mappo60c_a66'
CONFIRM_N = 96
IDENT_U0_SLICE, IDENT_ARM_SLICE = slice(0, 2), slice(2, 4)
A42_ROOT_ON_HOST = '/root/a42v/root'
A50_ROOT_ON_HOST = A52.A50_ROOT_ON_HOST
A52_ROOT_ON_HOST = A54.A52_ROOT_ON_HOST
A54_ROOT_ON_HOST = '/root/a54/root'
PANELS = dict(
    p1=dict(verify='a42v_D2800_p60', confirm='a42c_D2800_p60', ranges='A42_RANGES',
            run=('MAPPO60C', 'MAPPO60', 'MAPPO60B'), stored=('U0', 'DUO_HOLD', 'BIG_BANK'),
            holds=('DUO_HOLD', 'BIG_BANK'), ident_arm='DUO_HOLD', ident_ref=dict(U0='A42C_U0', DUO_HOLD='A42C_DUO_HOLD'),
            worlds='998600-998695 (A42 confirmation panel)'),
    p2=dict(verify='a50v_D2800_p60', confirm='a50c_D2800_p60', ranges='A50_RANGES',
            run=('MAPPO60C',), stored=('U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'BIG_BANK', 'MAPPO60', 'MAPPO60B'),
            holds=('DUO_HOLD', 'BIG_BANK', 'LEARNED_TRIGGER'), ident_arm='MAPPO60B',
            ident_ref=dict(U0='A50C_U0', MAPPO60B='A54C_MAPPO60B'),
            worlds='996800-996895 (A50 confirmation panel)'))
AUTHORITY = ('MAIN AMENDMENT 66: MAPPO60C (the Amendment 65 final policy) on panel 2 (996800-996895) and MAPPO60C, '
             'MAPPO60, MAPPO60B on panel 1 (998600-998695), D2800_p60, paired with the stored arms of each panel; '
             'same code path as A52/A54; identity on 4 worlds per panel; phase a66 cap 480,000')
DISCLOSURE = ('MAPPO60C: 84 updates on top of the 169-iteration u0, iteration fixed before training; the panel-1 '
              'MAPPO60 / MAPPO60B evaluations are added after their panel-2 results were known; same worlds as the '
              'earlier arms of each panel, not independent panels; panels never pooled; evaluated on a new host '
              '(identity check per panel)')


def contrasts_of(pk):
    """Registered contrasts of the panel: per X run there, X - U0 and every hold arm - X; then the descriptive ones."""
    P = PANELS[pk]
    out = []
    for x in P['run']:
        out.append((x, 'U0', False))
        out += [(h, x, False) for h in P['holds']]
    out += [('MAPPO60C', 'MAPPO60B', True), ('MAPPO60C', 'MAPPO60', True)]
    return tuple(out)


def core(pk):
    """The A42 driver code with the A66 names of the panel (one ledger phase / cap / root for both panels)."""
    P, T = PANELS[pk], pk.upper()
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A66',
        HOST_BASE=os.environ.get('A66_HOST_BASE', '/root/a66'),
        VERIFY_SETTING=P['verify'], CONFIRM_SETTING=P['confirm'],
        BANK_DIR='a66_banks', IDENT_WAVE=f'A66{T}_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A66', JOB_PREFIX='a66',
        VERIFY_TAG=f'a66{pk}v', CONFIRM_TAG=f'a66{pk}c', IDENT_TAG=f'a66{pk}i',
        VERIFY_WAVE_PREFIX=f'A66{T}V_', CONFIRM_WAVE_PREFIX=f'A66{T}C_',
        RESULTS_FILE=f'A66_RESULTS_{pk}_{CELL}.json', LOG_FILE='A66_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a66',
        RUNNER_RANGES_ATTR=P['ranges'],
        RESULT_LABEL=f'amendment 66; extra MAPPO arms on panel {pk[1]}; never pooled; one shot', AUTHORITY=AUTHORITY)
    m = A50._clone(f'a66_{pk}_core', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


def identity_file(pk):
    return f'IDENTITY_A66_{pk}.json'


def confirm_wave(pk, arm):
    return f'A66{pk.upper()}C_{arm}'


def ident_wave(pk, arm):
    return f'A66{pk.upper()}_IDENT_{arm}'


# ------------------------------------------------------------------------------------------------ binding

def binding_file(which):
    if which not in PINS:
        raise N.Blocked(f'which is one of {sorted(PINS)}')
    return f'BOUND_A66_{which}.json'


def bind(final_root, which, policy, *, check=None):
    """Seal BOUND_A66_<which>.json (as A52.bind / A54.bind; write once; zero physics)."""
    final_root, policy = Path(final_root), Path(policy)
    target = final_root/binding_file(which)
    if PINS[which] is None:
        raise N.Blocked(f'{which}: sha not pinned yet (pin MAPPO60C_SHA256 after the A65 training ends)')
    if target.exists():
        raise N.Blocked(f'{target}: exists (bound once; never rebound)')
    sha = N.file_sha(policy)
    if sha != PINS[which]:
        raise N.Blocked(f'{policy} hashes to {sha[:12]}, not the registered {PINS[which][:12]}')
    meta = N.export_meta(policy)
    if meta.get('trainer') != 'mappo' or meta.get('lane_head') or meta.get('lane_control') != 'native':
        raise N.Blocked(f'{which}: a speed-only mappo export (native lanes, no lane head) is required: {meta}')
    value = N.seal(dict(schema=N.SCHEMA+'.comparator', arm='U0' if which == 'IDENT_U0' else which, kind=f'a66_{which}',
                        label=LABELS[which], path=str(policy), sha256=sha, export=meta, authority=AUTHORITY,
                        rule='A66 (= A52 path): U0 arm with only the executor file replaced (path + sha256 verified '
                             'at every load)', load_check=check, amendment=AMENDMENT), 'binding_sha256')
    N.publish(target, value)
    return dict(value, binding_path=str(target))


def read_bound(final_root, which):
    path = Path(final_root)/binding_file(which)
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run bind --which {which} first)')
    value = N.verify(N.read(path), 'binding_sha256', path.name)
    if PINS[which] is None or value['sha256'] != PINS[which] or value['kind'] != f'a66_{which}':
        raise N.Blocked(f'{path.name}: not the registered {which} executor')
    return dict(value, binding_path=str(path))


def confirm_specs(pk, arm, panel, final_root):
    """U0 confirmation specs of the panel -> the arm's executor (A52.with_executor), ids a66<pk>c_<arm>_*."""
    D = core(pk)
    if arm not in PANELS[pk]['run']:
        raise N.Blocked(f'{arm} is not run on {pk}')
    return A52.with_executor(D.confirm_specs('U0', panel, final_root), read_bound(final_root, arm),
                             f'{D.CONFIRM_TAG}_{arm.lower()}')


def ident_waves(pk, panel, final_root):
    """[(wave, 'episode', specs)]: U0 through this path on panel worlds 0-1; the stored arm on panel worlds 2-3."""
    D, P = core(pk), PANELS[pk]
    base = D.confirm_specs('U0', panel, final_root)
    u0 = A52.with_executor(base[IDENT_U0_SLICE], read_bound(final_root, 'IDENT_U0'), f'{D.IDENT_TAG}_u0')
    arm = P['ident_arm']
    if arm == 'DUO_HOLD':
        specs = D._retag(D.confirm_specs('DUO_HOLD', panel, final_root)[IDENT_ARM_SLICE], f'{D.IDENT_TAG}_duo_hold')
    else:
        specs = A52.with_executor(base[IDENT_ARM_SLICE], read_bound(final_root, arm), f'{D.IDENT_TAG}_{arm.lower()}')
    return [(ident_wave(pk, 'U0'), 'episode', u0), (ident_wave(pk, arm), 'episode', specs)]


# ------------------------------------------------------------------------------------------------ stored arms

def load_a42_arms(panel, a42_root, arms, host_root=A42_ROOT_ON_HOST):
    """Stored panel-1 arms from a read-only COPY of the finished A42 root (its records name /root/a42v/root)."""
    import a42_bigp60 as A42
    have, roots, sources = {}, {}, {}
    with A52.host_root_view(host_root):
        for arm in arms:
            specs = A42.confirm_specs(arm, panel, a42_root, check_files=False)
            have[arm], sources[arm] = A42.C37._load(a42_root, A42.confirm_wave(arm), specs)
            roots[arm] = Path(a42_root)
    return have, roots, sources


def load_a54_arm(panel, a54_root, host_root=A54_ROOT_ON_HOST):
    """MAPPO60B records from a read-only COPY of the finished A54 root (its records name /root/a54/root)."""
    with A52.host_root_view(host_root):
        exe = A54.read_bound(a54_root)
        exe['binding_path'] = N.binding_location(a54_root, exe['binding_path'])
        specs = A52.with_executor(A54.D.confirm_specs('U0', panel, a54_root), exe,
                                  f'{A54.D.CONFIRM_TAG}_{A54.ARM.lower()}')
        return A54.D.C37._load(a54_root, A54.CONFIRM_WAVE, specs)


def load_refs(pk, panel, refs):
    """(have, roots, sources) of every stored arm of the panel."""
    if pk == 'p1':
        return load_a42_arms(panel, refs['a42'], PANELS[pk]['stored'])
    have, roots, sources = A54.load_refs(panel, refs['a50'], refs['a52'])        # A50 arms + MAPPO60
    have['MAPPO60B'], sources['MAPPO60B'] = load_a54_arm(panel, refs['a54'])
    roots['MAPPO60B'] = Path(refs['a54'])
    if set(have) != set(PANELS[pk]['stored']):
        raise N.Blocked(f'stored arms {sorted(have)} != {sorted(PANELS[pk]["stored"])}')
    return have, roots, sources


def _ref_root(pk, arm, refs):
    if pk == 'p1':
        return refs['a42']
    return refs['a54'] if arm == 'MAPPO60B' else refs['a50']


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(pk, panel, final_root, refs):
    D, P = core(pk), PANELS[pk]
    rows = []
    for (wave, _, specs), arm in zip(ident_waves(pk, panel, final_root), ('U0', P['ident_arm'])):
        got, _, cov = N.published(final_root, wave, specs, D.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        ref = A52._a50_records(_ref_root(pk, arm, refs), P['ident_ref'][arm])
        for r in got:
            a = ref.get(r['world_id'])
            if a is None:
                raise N.Blocked(f'{r["world_id"]}: no stored {P["ident_ref"][arm]} record')
            rows.append(dict(arm=arm, stored_wave=P['ident_ref'][arm], world_id=r['world_id'], j_a66=r['J_veh_s'],
                             j_stored=a['J_veh_s'], abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a66=r['safety'],
                             safety_stored=a['safety'], safety_equal=r['safety'] == a['safety'],
                             executor_loaded=r.get('executor_sha256_loaded'), same_world=r['world'] == a['world'],
                             vehicle_log_equal=(r.get('vehicle_log') or {}).get('sha256')
                             == (a.get('vehicle_log') or {}).get('sha256')))
    ok = (len(rows) == 4 and len({x['world_id'] for x in rows}) == 4
          and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world'] for x in rows))
    return N.seal(dict(schema='coopmwm.a66.identity.v1', amendment=AMENDMENT, panel=pk,
                       rule='U0 (u0 d9042cc5 bound through the A66 path) on panel worlds 0-1 and the stored arm '
                            f'{P["ident_arm"]} on panel worlds 2-3 reproduce the stored records exactly on this host: '
                            'J_veh_s difference 0 and identical safety counters on all 4 worlds',
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: the panel may run; FAILED: stop and report, the panel is not run'),
                  'receipt_sha256')


def identity_passed(pk, final_root):
    p = Path(final_root)/identity_file(pk)
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ analysis

def _tables(pk, have, roots, sources, panel, contrasts, kind):
    import numpy as np
    from a32_ablation import report_of
    D = core(pk)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    if len(worlds) != CONFIRM_N:
        raise N.Blocked(f'{len(worlds)} worlds, not {CONFIRM_N}')
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = []
    for a, ref, descriptive in contrasts:
        c = D._contrast(have, reports, worlds, a, ref, bootstrap, kind)
        if descriptive:
            c['descriptive'] = True
        rows.append(c)
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    return worlds, bootstrap, rows, arms


def reading_of(x, rows):
    """The registered reading for X from its contrasts (total delay CI95)."""
    ci = {(c['treatment'], c['reference']): c['total_minus_ref']['ci95'] for c in rows}
    duo, big, u0 = ci[('DUO_HOLD', x)], ci[('BIG_BANK', x)], ci[(x, 'U0')]
    supported = bool(duo[1] < 0 and big[1] < 0)
    return dict(ci95_upper_below_zero={f'DUO_HOLD-{x}': bool(duo[1] < 0), f'BIG_BANK-{x}': bool(big[1] < 0)},
                does_not_find_the_hold='SUPPORTED' if supported else 'NOT SUPPORTED',
                closes_the_gap=not supported, x_minus_u0_saves=bool(u0[1] < 0),
                x_minus_u0_ci95=list(u0))


def analyse(pk, final_root, panel, refs):
    final_root, P = Path(final_root), PANELS[pk]
    D = core(pk)
    have, roots, sources = load_refs(pk, panel, refs)
    for arm in P['run']:
        have[arm], sources[arm] = D.C37._load(final_root, confirm_wave(pk, arm), confirm_specs(pk, arm, panel,
                                                                                               final_root))
        roots[arm] = final_root
    worlds, bootstrap, rows, arms = _tables(pk, have, roots, sources, panel, contrasts_of(pk), 'a66')
    return N.seal(dict(
        schema='coopmwm.a66.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL, panel=pk,
        panel_worlds=P['worlds'], disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
        bootstrap=bootstrap, executors={a: dict(sha256=PINS[a], binding=sources[a]) for a in P['run']},
        contrasts=rows, arms=arms,
        reading=dict(rule='per X: "MAPPO trained at 60 % (X) does not find the hold" SUPPORTED iff DUO_HOLD - X and '
                          'BIG_BANK - X both have CI95 upper < 0; "X closes the gap" if either CI includes 0 or is '
                          'positive; "X - U0 SAVES" iff CI95 upper < 0 (total delay; diff = treatment - reference)',
                     per_arm={x: reading_of(x, rows) for x in P['run']}),
        identity=N.read(final_root/identity_file(pk)).get('receipt_sha256'),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
        ref_root_copies={k: str(v) for k, v in refs.items() if v}, label=D.RESULT_LABEL), 'analysis_sha256')


def _max_rel(a, b):
    """Largest relative numeric difference between two JSON values; inf on any structural / non-numeric difference."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else float('inf')
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 0.0 if a == b else abs(a-b)/max(abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return float('inf') if set(a) != set(b) else max([_max_rel(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return float('inf') if len(a) != len(b) else max([_max_rel(x, y) for x, y in zip(a, b)] or [0.0])
    return float('inf')


def selfcheck(panel, refs, a54_results, tol=1e-9):
    """p2, zero physics: this analysis path on the STORED MAPPO60B arm reproduces the published A54 contrasts
    (structure and non-numeric fields identical; numbers within ``tol`` relative - a different CPU changes the
    last bits of float sums; the largest difference is reported)."""
    have, roots, sources = load_refs('p2', panel, refs)
    want = N.read(a54_results)
    cons = tuple((c['treatment'], c['reference'], False) for c in want['contrasts'])
    _, _, rows, _ = _tables('p2', have, roots, sources, panel, cons, 'a54')
    out = []
    for got, ref in zip(rows, want['contrasts']):
        g = {k: v for k, v in got.items() if k != 'descriptive'}
        r = {k: v for k, v in ref.items() if k != 'descriptive'}
        rel = _max_rel(g, r)
        out.append(dict(contrast=f'{got["treatment"]}-{got["reference"]}', exact=g == r, max_rel_diff=rel,
                        within_tol=rel <= tol, total_exact=got['total_minus_ref'] == ref['total_minus_ref'],
                        total_max_rel_diff=_max_rel(got['total_minus_ref'], ref['total_minus_ref']),
                        ci95=got['total_minus_ref']['ci95'], ci95_a54=ref['total_minus_ref']['ci95']))
    return dict(all_within_tol=all(x['within_tol'] for x in out) and len(out) == len(want['contrasts']), tol=tol,
                contrasts=out, a54_analysis_sha256=want.get('analysis_sha256'))


# ------------------------------------------------------------------------------------------------ CLI

def budget_plan():
    conf = {pk: len(P['run'])*CONFIRM_N*N.HORIZON_S for pk, P in PANELS.items()}
    ident = 2*4*N.HORIZON_S
    planned = sum(conf.values())+ident
    return dict(phase=PHASE, cap_sim_s=CAP_SIM_S, confirmation=conf, identity=ident, planned=planned,
                margin=CAP_SIM_S-planned)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('bind', 'dry', 'identity', 'identity-compare', 'confirm', 'load-refs',
                                       'selfcheck', 'analyse', 'status', 'budget'))
    p.add_argument('--panel', choices=sorted(PANELS))
    p.add_argument('--final-root')
    p.add_argument('--which', choices=sorted(PINS))
    p.add_argument('--policy')
    p.add_argument('--a42-root')
    p.add_argument('--a50-root')
    p.add_argument('--a52-root')
    p.add_argument('--a54-root')
    p.add_argument('--a54-results')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    if a.command == 'budget':
        print(json.dumps(budget_plan(), indent=1))
        return
    if not a.final_root and a.command != 'selfcheck':
        p.error('--final-root required')
    root = Path(a.final_root) if a.final_root else None
    if a.command == 'bind':
        from tools.coopmwm.nm_eval_runner import mappo_load_check
        if PINS[a.which] is None:
            raise N.Blocked(f'{a.which}: sha not pinned yet')
        v = bind(root, a.which, a.policy, check=mappo_load_check(a.policy, PINS[a.which]))
        print(json.dumps(dict(which=a.which, binding_sha256=v['binding_sha256'], sha256=v['sha256'],
                              export=v['export'])))
        return
    if not a.panel:
        p.error('--panel required')
    pk, P = a.panel, PANELS[a.panel]
    D = core(pk)
    setting = D.CONFIRM_SETTING
    panel = D.setup(setting)
    refs = dict(a42=a.a42_root, a50=a.a50_root, a52=a.a52_root, a54=a.a54_root)
    need_refs = ('a42',) if pk == 'p1' else ('a50', 'a52', 'a54')
    if a.command in ('identity-compare', 'load-refs', 'selfcheck', 'analyse') and any(not refs[k] for k in need_refs):
        p.error(f'{a.command} --panel {pk} needs ' + ' '.join(f'--{k}-root' for k in need_refs))

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting, panel=pk),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        ident = ident_waves(pk, panel, root)
        conf = {arm: confirm_specs(pk, arm, panel, root) for arm in P['run']}
        print(json.dumps(dict(
            budget=budget_plan(),
            ident=[(w, [(s['job_id'], s['world_id'], s['arm'], s['checkpoint'].get('executor_sha256')) for s in specs])
                   for w, _, specs in ident],
            confirm={arm: dict(n=len(c), first=c[0]['job_id'], last=c[-1]['job_id'],
                               seeds=[c[0]['world']['seed'], c[-1]['world']['seed']], cell=c[0]['world']['cell_key'],
                               executor=c[0]['checkpoint']['executor_sha256']) for arm, c in conf.items()},
            charged=D.charged(root)), indent=1))
    elif a.command in ('identity', 'confirm'):
        import fcntl
        if a.command == 'confirm' and not identity_passed(pk, root):
            raise N.Blocked(f'{identity_file(pk)} is not PASS: panel {pk} is not run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = ident_waves(pk, panel, root)
            else:
                waves = [(confirm_wave(pk, arm), 'episode', confirm_specs(pk, arm, panel, root)) for arm in P['run']]
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(pk, panel, root, refs)
        N.publish(root/identity_file(pk), v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], worlds=[(x['arm'], x['world_id'], x['abs_diff'], x['safety_equal'],
                                                            x['same_world']) for x in v['worlds']])))
    elif a.command == 'load-refs':
        have, _, sources = load_refs(pk, panel, refs)
        print(json.dumps({k: dict(n=len(v), receipt=sources[k].get('receipt_sha256')) for k, v in have.items()}))
    elif a.command == 'selfcheck':
        if pk != 'p2':
            p.error('selfcheck is p2 only')
        print(json.dumps(selfcheck(panel, refs, a.a54_results), indent=1))
    elif a.command == 'analyse':
        v = analyse(pk, root, panel, refs)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], reading=v['reading'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(pk, root))
        for arm in P['run']:
            if (root/binding_file(arm)).is_file():
                specs = confirm_specs(pk, arm, panel, root)
                out[confirm_wave(pk, arm)] = f'{N.published(root, confirm_wave(pk, arm), specs, D.C37.check_a37)[2]}' \
                                             f'/{len(specs)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
