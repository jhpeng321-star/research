"""AMENDMENT 54 (registered; MAIN 2026-11-11): MAPPO60B = the A52 MAPPO60 code path with a different policy file.

Same code path as A52 (c4/a52-mappo60 2bc62a97: ops/a52_mappo60.py): the A50 confirmation U0 arm (runner arm 'U0',
zero-residual u0, plant, panel a50c_D2800_p60 = worlds 996800-996895, 1,200 s, control from 300 s, no hold executor)
whose executor is a sealed binding record (A52.with_executor replaces only the six executor fields). The ONLY changed
parameter is the policy file: the Amendment 51 production policy_00024.pt (sha256 30971d7e...5ff9; speed-only MAPPO,
24 updates from u0 d9042cc5 at D2800_p60). Arm name MAPPO60B; own ledger phase / cap / job ids / stop file.

Registered deviation from A52: no identity re-run (code and host unchanged); ``confirm`` does not require it.

Budget: ledger phase ``mappo60b_a54`` cap 120,000 sim-s (96 x 1,200 = 115,200; no automatic replay).

Analysis (zero physics): the A50 driver's own ``_contrast`` (paired percentile bootstrap 10,000, seed 20261104;
diff = treatment - reference, negative = less delay) for MAPPO60B - U0, DUO_HOLD - MAPPO60B, BIG_BANK - MAPPO60B,
LEARNED_TRIGGER - MAPPO60B, and MAPPO60B - MAPPO60 (descriptive). A50 arms from a read-only COPY of the A50 root
(records name /root/a50v/root); MAPPO60 from a read-only COPY of the A52 root (records name /root/a52/root).

Commands (python ops/a54_mappo60b.py <cmd> --final-root <root>):
  bind --policy <file>                    seal BOUND_A54_MAPPO60B.json (zero physics)
  dry / status / budget
  confirm                                 MAPPO60B on the 96 worlds
  load-refs --a50-root <c> --a52-root <c> zero physics: the reference arms load (96/96 each)
  analyse --a50-root <c> --a52-root <c> --out <dir>   A54_RESULTS_D2800_p60.json
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
import a52_mappo60 as A52  # noqa: E402   (the A52 path: with_executor, A50 / A52 arm loading; never runs here)

AMENDMENT = '54'
CELL = 'D2800_p60'
ARM = 'MAPPO60B'
REF_ARM = 'MAPPO60'
MAPPO60B_SHA256 = '30971d7effb46539a8cf3a3d34da10eb0ac841d8c600be368676669b801b5ff9'
PINS = dict(MAPPO60B=MAPPO60B_SHA256)
LABEL_TEXT = ('Amendment 51 run24 production policy_00024 (speed-only MAPPO, 60 % CAV, D2800_p60, 24 updates from '
              'u0 d9042cc5)')
CAP_SIM_S = 120_000
CONFIRM_N = 96
A50_ROOT_ON_HOST = A52.A50_ROOT_ON_HOST
A52_ROOT_ON_HOST = '/root/a52/root'
CONTRASTS = ((ARM, 'U0'), ('DUO_HOLD', ARM), ('BIG_BANK', ARM), ('LEARNED_TRIGGER', ARM), (ARM, REF_ARM))
AUTHORITY = ('MAIN AMENDMENT 54: the Amendment 51 policy (run24 production policy_00024, 30971d7e) as one extra arm '
             'MAPPO60B on the A50 confirmation worlds 996800-996895 (D2800_p60), paired with U0, DUO_HOLD, '
             'LEARNED_TRIGGER, BIG_BANK and MAPPO60 (A52); same code path as A52; phase a54 cap 120,000')
DISCLOSURE = ('24 updates on top of the 169-iteration u0 (short fine-tune); same worlds as A50/A52, not an independent '
              'panel, never pooled; no identity re-run (code and host unchanged; registered deviation)')
OVERRIDES = dict(
    A42_PHASE='mappo60b_a54', A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A54',
    HOST_BASE=os.environ.get('A54_HOST_BASE', '/root/a54'),
    VERIFY_SETTING='a50v_D2800_p60', CONFIRM_SETTING='a50c_D2800_p60',
    BANK_DIR='a54_banks', IDENT_WAVE='A54_IDENT_U0', AMENDMENT=AMENDMENT, LABEL='A54', JOB_PREFIX='a54',
    VERIFY_TAG='a54v', CONFIRM_TAG='a54c', IDENT_TAG='a54i', VERIFY_WAVE_PREFIX='A54V_', CONFIRM_WAVE_PREFIX='A54C_',
    RESULTS_FILE=f'A54_RESULTS_{CELL}.json', LOG_FILE='A54_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a54',
    RUNNER_RANGES_ATTR='A50_RANGES', RESULT_LABEL='amendment 54; extra arm on the A50 panel; never pooled; one shot',
    AUTHORITY=AUTHORITY)


def _build():
    m = A50._clone('a54_mappo60b_core', HERE/'a42_bigp60.py', OVERRIDES)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (m.VERIFY_SETTING, m.CONFIRM_SETTING)
    m.A42Runtime.phase = m.A42_PHASE
    return m


D = _build()
D50 = A50.D
CONFIRM_WAVE = f'A54C_{ARM}'
BINDING_FILE = f'BOUND_A54_{ARM}.json'


def bind(final_root, policy, *, check=None):
    """Seal BOUND_A54_MAPPO60B.json (as A52.bind; write once; zero physics)."""
    final_root, policy = Path(final_root), Path(policy)
    target = final_root/BINDING_FILE
    if target.exists():
        raise N.Blocked(f'{target}: exists (bound once; never rebound)')
    sha = N.file_sha(policy)
    if sha != PINS[ARM]:
        raise N.Blocked(f'{policy} hashes to {sha[:12]}, not the registered {PINS[ARM][:12]}')
    meta = N.export_meta(policy)
    if meta.get('trainer') != 'mappo' or meta.get('lane_head') or meta.get('lane_control') != 'native':
        raise N.Blocked(f'{ARM}: a speed-only mappo export (native lanes, no lane head) is required: {meta}')
    value = N.seal(dict(schema=N.SCHEMA+'.comparator', arm=ARM, kind=f'a54_{ARM}', label=LABEL_TEXT, path=str(policy),
                        sha256=sha, export=meta, authority=AUTHORITY,
                        rule='A54 (= A52 path): U0 arm with only the executor file replaced (path + sha256 verified '
                             'at every load)', load_check=check, amendment=AMENDMENT), 'binding_sha256')
    N.publish(target, value)
    return dict(value, binding_path=str(target))


def read_bound(final_root):
    path = Path(final_root)/BINDING_FILE
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run bind first)')
    value = N.verify(N.read(path), 'binding_sha256', path.name)
    if value['sha256'] != PINS[ARM] or value['kind'] != f'a54_{ARM}':
        raise N.Blocked(f'{path.name}: not the registered {ARM} executor')
    return dict(value, binding_path=str(path))


def confirm_specs(panel, final_root):
    """U0 confirmation specs of the A50 panel -> MAPPO60B executor (A52.with_executor), ids a54c_mappo60b_*."""
    base = D.confirm_specs('U0', panel, final_root)
    return A52.with_executor(base, read_bound(final_root), f'{D.CONFIRM_TAG}_{ARM.lower()}')


def load_a52_arm(panel, a52_root, host_root=A52_ROOT_ON_HOST):
    """MAPPO60 records from a read-only COPY of the finished A52 root (its records name /root/a52/root)."""
    with A52.host_root_view(host_root):
        exe = A52.read_bound(a52_root, 'MAPPO60')
        # location-only remap of the binding path (as N.binding_location does for BOUND_U0); content still verified
        exe['binding_path'] = N.binding_location(a52_root, exe['binding_path'])
        specs = A52.with_executor(A52.D.confirm_specs('U0', panel, a52_root), exe,
                                  f'{A52.D.CONFIRM_TAG}_{A52.ARM.lower()}')
        return A52.D.C37._load(a52_root, A52.CONFIRM_WAVE, specs)


def load_refs(panel, a50_root, a52_root, *, a50_host=A50_ROOT_ON_HOST, a52_host=A52_ROOT_ON_HOST):
    have, roots, sources = A52.load_a50_arms(panel, a50_root, a50_host)
    have[REF_ARM], sources[REF_ARM] = load_a52_arm(panel, a52_root, a52_host)
    roots[REF_ARM] = Path(a52_root)
    return have, roots, sources


def analyse(final_root, a50_root, a52_root, panel, *, a50_host=A50_ROOT_ON_HOST, a52_host=A52_ROOT_ON_HOST):
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
    h, r, s = load_refs(panel, a50_root, a52_root, a50_host=a50_host, a52_host=a52_host)
    have.update(h)
    roots.update(r)
    sources.update(s)
    reports = {a: {w: report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    contrasts = [D._contrast(have, reports, worlds, a, ref, bootstrap, 'a54') for a, ref in CONTRASTS]
    for c in contrasts:
        if (c['treatment'], c['reference']) == (ARM, REF_ARM):
            c['descriptive'] = True
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in (ARM, REF_ARM)+tuple(D50.CONFIRM_ARMS)}
    hold = {f'{a}-{ARM}': bool(c['total_minus_ref']['ci95'][1] < 0)
            for c in contrasts for a in ('DUO_HOLD', 'BIG_BANK') if (c['treatment'], c['reference']) == (a, ARM)}
    return N.seal(dict(schema='coopmwm.a54.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, executor=dict(sha256=MAPPO60B_SHA256, binding=sources[ARM]),
                       contrasts=contrasts, arms=arms,
                       reading=dict(rule='as A52: "MAPPO alone does not find the cooperation" is supported only if '
                                         'the hold arms remain below MAPPO60B with CI95 upper < 0',
                                    ci_upper_below_zero=hold, supported=all(hold.values()) and len(hold) == 2),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       a50_root_copy=str(a50_root), a52_root_copy=str(a52_root), label=D.RESULT_LABEL),
                  'analysis_sha256')


def budget_plan():
    planned = CONFIRM_N*N.HORIZON_S
    return dict(phase=D.A42_PHASE, cap_sim_s=CAP_SIM_S, confirmation=planned, planned=planned,
                margin=CAP_SIM_S-planned)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('bind', 'dry', 'confirm', 'load-refs', 'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--policy')
    p.add_argument('--a50-root')
    p.add_argument('--a52-root')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=24)
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
        v = bind(root, a.policy, check=mappo_load_check(a.policy, PINS[ARM]))
        print(json.dumps(dict(binding_sha256=v['binding_sha256'], sha256=v['sha256'], export=v['export'])))
        return
    setting = D.CONFIRM_SETTING
    panel = D.setup(setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'dry':
        conf = confirm_specs(panel, root)
        print(json.dumps(dict(budget=budget_plan(),
                              confirm=dict(n=len(conf), first=conf[0]['job_id'], last=conf[-1]['job_id'],
                                           seeds=[conf[0]['world']['seed'], conf[-1]['world']['seed']],
                                           cell=conf[0]['world']['cell_key'],
                                           executor=conf[0]['checkpoint']['executor_sha256']),
                              charged=D.charged(root)), indent=1))
    elif a.command == 'confirm':
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = [(CONFIRM_WAVE, 'episode', confirm_specs(panel, root))]
            need = sum(len(s) for _, _, s in waves)*N.HORIZON_S
            if D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{D.A42_PHASE}: {D.charged(root)} + {need} > cap {CAP_SIM_S}')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root)))
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'load-refs':
        have, _, sources = load_refs(panel, a.a50_root, a.a52_root)
        print(json.dumps({k: dict(n=len(v), receipt=sources[k].get('receipt_sha256')) for k, v in have.items()}))
    elif a.command == 'analyse':
        v = analyse(root, a.a50_root, a.a52_root, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/D.RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], reading=v['reading'])))
    else:   # status
        specs = confirm_specs(panel, root)
        print(json.dumps(dict(charged=D.charged(root), cap=CAP_SIM_S,
                              **{CONFIRM_WAVE: f'{N.published(root, CONFIRM_WAVE, specs, D.C37.check_a37)[2]}'
                                               f'/{len(specs)}'}), indent=1))


if __name__ == '__main__':
    main()
