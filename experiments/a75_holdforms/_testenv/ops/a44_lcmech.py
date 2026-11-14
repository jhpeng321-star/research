"""AMENDMENT 44 mechanics (MAIN 2026-11-10 22:50; pre-outcome; NOT efficacy evidence, never reported as a result):
the lane-change executor variant (tools/coopmwm/a44_lcvariant.py, runner arms A44_PLAN / A44_BANK) at D2800_p60.

PREPARED, NOT LAUNCHED by the build agent. MAIN records the audited seed block and the host in REGISTRATION.md, pins
the panel (PANEL_SHA256 below) and launches.

(a) hold identity: 2 worlds x {A37 executor (runner arm A37_PLAN, digest 0a902437), A44 variant (A44_PLAN)} with the
    SAME hold plan (FIXTURE_PLANS.json 'a44_fix_hold', one bank file read by both): records must match exactly
    (J, safety, vehicle-log sha256, activations / members / releases / issued).
(b) lane-change fixture: one hand-made plan with one lc_target role ('a44_fix_lc'; NOT an A43 candidate) on up to 8
    worlds; reports commanded vs completed vs given-up lane changes, collisions, teleports.
Outcomes (delay) are never read for any choice (lc-report reads only the executor receipts and safety counters).

Budget: ONE ledger phase ``lcexec_a44_mech`` capped at 24,000 sim-s = identity 4 x 1,200 + fixture 8 x 1,200 +
reserve 9,600 (one charged retry per failed episode on the same world; never outcome-based).

Commands (python ops/a44_lcmech.py <cmd> --final-root <root> --seeds LO-HI):
  panel            zero physics: the TRAIN-type block panel (worlds LO..HI, <= 8, inside the runner A44_MECH_RANGE)
                   -> ops/a44/nm_eval_trainblock_wz3to2_a44m_D2800_p60.json (MAIN pins its panel_sha256 below)
  prepare          zero physics: <root>/a44_banks/A44_HOLD.json (A37 bank form) + A44_LCFIX.json (A44 bank form)
  dry              zero physics: build every spec, check bindings / digests, print counts and sim-s
  identity         (a) 4 episodes
  identity-compare zero physics: (a) record equality -> <root>/A44_IDENTITY.json
  lcfix            (b) up to --n-lc (default 8) episodes
  lc-report        zero physics: (b) receipts + safety -> <root>/A44_LC_MECH.json
  status           coverage per wave + ledger totals
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
import a32_ablation as A  # noqa: E402
import a37_celltailor as C37  # noqa: E402  (registers check_a37 for the A37 arms; A37 driver untouched)

A44_PHASE = 'lcexec_a44_mech'
A44_CAP_SIM_S = 24_000
N_IDENT_WORLDS, MAX_LC_WORLDS = 2, 8
IDENTITY_SIM_S = 2*N_IDENT_WORLDS*N.HORIZON_S
LC_SIM_S = MAX_LC_WORLDS*N.HORIZON_S
RESERVE_SIM_S = A44_CAP_SIM_S-IDENTITY_SIM_S-LC_SIM_S
STOP_A44 = 'STOP_A44'
HOST_BASE = os.environ.get('A44_HOST_BASE', '/root/a44m')
HOST_ROOT = f'{HOST_BASE}/root'
CELL = 'D2800_p60'
SETTING, SOURCE_SETTING = 'a44m_D2800_p60', 'a35_D2800_p60'
A44_DIR = HERE/'a44'
FIXTURE_FILE = A44_DIR/'FIXTURE_PLANS.json'
PANEL_FILE = A44_DIR/f'nm_eval_trainblock_wz3to2_{SETTING}.json'
#: pinned by MAIN after it records the audited block (before any A44 SUMO); None -> physics refused
PANEL_SHA256 = '49a271551277ab2d2378a733ce245a47768bcb02c47eedf5dfd6dc700a4fbb87'   # pinned 2026-11-12 (AMENDMENT 69 records the block 992400-992407)
BANK_DIR = 'a44_banks'
HOLD_BANK, LC_BANK = 'A44_HOLD.json', 'A44_LCFIX.json'
WAVE_IDENT = dict(A37='A44M_IDENT_A37', A44='A44M_IDENT_A44')
WAVE_LC = 'A44M_LCFIX'
AUTHORITY = ('MAIN 2026-11-10 22:50 AMENDMENT 44 (owner: "start route 2 now in parallel too"): lane-change-capable '
             'executor variant; mechanics phase lcexec_a44_mech cap 24,000 sim-s (hold identity 2 worlds x 2 executors '
             '+ lane-change fixture <= 8 episodes + reserve); not efficacy evidence')
REPLAY_AUTHORITY = ('MAIN AMENDMENT 44 registered rule: one charged retry per failed episode on the same world; '
                    'no outcome-based retries')


# ------------------------------------------------------------------------------------------------ fixture plans

def fixture_plans():
    """Hand-made (NOT an A43 candidate). Values from the registered strategy_search grids only: the incumbent
    wall_g_solo2 role (SUMO lane 2, WALL_G = v_w 22, dmax 1.0, hold 120 s) plus one SUMO lane-1 member with the same
    WALL_G program at the registered offset 20 s (lcm_plan.OFFSETS) and an adjacent-lane target SUMO 2 (WM 0; the G5
    path executes it); cooldown = max(onset + hold) (strategy_search._spec rule). The hold twin is the same plan
    without the lane change (the hold-identity plan)."""
    wall_g = dict(v_w=22.0, dmax=1.0, hold_s=120)

    def role(lane, onset, lc):
        return dict(lane=lane, rank=0, onset_s=onset, lc_target=lc, **wall_g)

    def spec(name, lc):
        return dict(name=name, trigger='wall', zone='full', spread_m=200.0, cooldown_s=140, slots=1, gate=None,
                    start_gate=None, phases=None, roles=[role(2, 0, None), role(1, 20, lc)])
    return dict(hold=spec('a44_fix_hold', None), lc=spec('a44_fix_lc', 2))


def fixture_doc():
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    from tools.coopmwm import oracle_plan as O
    f = fixture_plans()
    if C.executor_refusal(f['hold']) is not None or V.executor_refusal(f['lc']) is not None:
        raise N.Blocked('fixture plans not executable')
    if C.executor_refusal(f['lc']) != 'non_hold_plan' or not V.is_lc_plan(f['lc']):
        raise N.Blocked('the lc fixture must be an lc plan')
    body = dict(schema='coopmwm.a44.fixture.v1', amendment='44', plans=f,
                keys={k: O.strategy_key(v) for k, v in f.items()},
                note='hand-made mechanics fixture; not a search candidate; values = registered grids')
    return dict(body, doc_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())


def fixtures():
    if not FIXTURE_FILE.is_file():
        raise N.Blocked(f'{FIXTURE_FILE}: missing')
    doc = N.read(FIXTURE_FILE)
    if doc != fixture_doc():
        raise N.Blocked('FIXTURE_PLANS.json differs from the registered fixture (never edit)')
    return doc['plans']


# ------------------------------------------------------------------------------------------------ setting / seeds

def parse_seeds(text):
    lo, _, hi = str(text).partition('-')
    lo, hi = int(lo), int(hi or lo)
    from tools.coopmwm import nm_eval_runner as R
    r0, r1 = R.A44_MECH_RANGE
    if not r0 <= lo <= hi <= r1 or hi-lo+1 > MAX_LC_WORLDS or hi-lo+1 < N_IDENT_WORLDS:
        raise N.Blocked(f'--seeds {text}: must be {N_IDENT_WORLDS}..{MAX_LC_WORLDS} contiguous seeds inside the runner '
                        f'A44_MECH_RANGE {r0}-{r1}')
    for bad in ((964000, 964199), (962000, 962999), (977000, 977999)):
        if lo <= bad[1] and hi >= bad[0]:
            raise N.Blocked('forbidden seed range')
    return lo, hi


def register_setting(lo, hi):
    """Process-local nm_finaleval setting (nm_finaleval itself unchanged): the a35_D2800_p60 cell / plant with the
    A44 TRAIN-type block."""
    from tools.coopmwm import nm_eval_runner as R
    r0, r1 = R.A44_MECH_RANGE
    N.SETTINGS[SETTING] = dict(
        N.SETTINGS[SOURCE_SETTING],
        N_WORLDS=hi-lo+1, EVAL_RANGE=(lo, r1), EVAL_EXTRA_RANGE=(lo, r1), SEEDS_RESERVED=(hi+1, r1),
        PANEL_SIZE_REASON=f'MAIN amendment 44 mechanics: TRAIN-type block {lo}-{hi} (<= 8 worlds); never evaluation',
        CELL_AUTHORITY='MAIN amendment 44 (REGISTRATION 2026-11-10): lane-change executor mechanics at D2800_p60',
        PANEL_AUTHORITY=(f'A44 amendment 44: audited TRAIN-type integers {lo}-{hi} (runner A44_MECH_RANGE {r0}-{r1}) for '
                         'the executor-variant mechanics ONLY (never evaluation, outcomes never read)'),
        EXPOSURE_EXTRA=dict(checked=('amendment-44 audit: ops/a44/SEED_AUDIT.json',),
                            disclosed='; amendment 44: TRAIN-type mechanics block, registered at runtime by the runner'),
        PANEL_FILE=PANEL_FILE, PANEL_3C_FILE=None)
    if SETTING not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(SETTING,)
    N.SETTINGS_PINS[SETTING] = dict(PANEL_SHA256=PANEL_SHA256, PANEL_3C_SHA256=None)


def setup(seeds, *, need_panel=True):
    lo, hi = parse_seeds(seeds)
    register_setting(lo, hi)
    N.apply_setting(SETTING)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    if not need_panel:
        return None
    panel = N.load_panel(PANEL_FILE, pinned=None)
    got = [w['world']['seed'] for w in panel['worlds']]
    if got != list(range(lo, hi+1)):
        raise N.Blocked(f'panel seeds {got[0]}-{got[-1]} are not --seeds {lo}-{hi}')
    if panel['cell']['cell_key'] != CELL:
        raise N.Blocked('panel is not D2800_p60')
    return panel


def require_pin(panel):
    if PANEL_SHA256 is None or panel['panel_sha256'] != PANEL_SHA256:
        raise N.Blocked('A44: panel not pinned by MAIN (PANEL_SHA256) - physics refused')


# ------------------------------------------------------------------------------------------------ banks / specs

def prepare(final_root):
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    f = fixtures()
    out = {}
    for name, doc in ((HOLD_BANK, C.bank_doc([dict(family=f['hold']['name'], strategy=f['hold'], lb95_veh_s=0.0)], CELL)),
                      (LC_BANK, V.bank_doc([dict(family=f['lc']['name'], strategy=f['lc'], lb95_veh_s=0.0)], CELL))):
        p = C37.write_once(Path(final_root)/BANK_DIR/name, C37.bank_text(doc))
        out[name] = dict(path=str(p), file_sha256=N.file_sha(p))
    return out


def bank_entry(final_root, name, *, check_files=True):
    p = Path(final_root)/BANK_DIR/name
    if not p.is_file():
        if check_files:
            raise N.Blocked(f'{p}: missing (run prepare first)')
        return dict(path=f'{HOST_ROOT}/{BANK_DIR}/{name}', file_sha256='')
    return dict(path=f'{HOST_ROOT}/{BANK_DIR}/{name}', file_sha256=N.file_sha(p))


def a44_specs(arm, panel, final_root, entry, *, family, tag):
    """The registered LEARNED_TRIGGER plant specs (same plant / U0 binding) with arm -> A44 arm, checkpoint kind 'a44'
    (a37_celltailor._a37_specs pattern)."""
    from tools.coopmwm import a44_lcvariant as V
    u0, oracle = A.bindings(final_root)
    base = N.episode_specs('LEARNED_TRIGGER', panel, None, Path(final_root)/N.PRERUN_PLANT, 'mappo_frozen', oracle, u0)
    out = []
    for i, s in enumerate(base):
        ck = deepcopy(s['checkpoint'])
        ck.pop('learned_trigger_sha256')
        ck.pop('learned_trigger_model')
        ck.update(kind='a44', a44_arm=arm, a44_sha256=V.a44_digest(), a44_bank=dict(entry), a44_family=family)
        out.append(dict(s, arm=arm, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def waves(command, panel, final_root, *, n_lc=MAX_LC_WORLDS, check_files=True):
    f = fixtures()
    if command == 'identity':
        hold = bank_entry(final_root, HOLD_BANK, check_files=check_files)
        a37 = C37._a37_specs('A37_PLAN', panel, final_root, hold, family=f['hold']['name'], tag='a44m_ident_a37')
        a44 = a44_specs('A44_PLAN', panel, final_root, hold, family=f['hold']['name'], tag='a44m_ident_a44')
        return [(WAVE_IDENT['A37'], 'episode', a37[:N_IDENT_WORLDS]), (WAVE_IDENT['A44'], 'episode', a44[:N_IDENT_WORLDS])]
    if command == 'lcfix':
        if not 1 <= int(n_lc) <= MAX_LC_WORLDS:
            raise N.Blocked(f'--n-lc must be 1..{MAX_LC_WORLDS}')
        lc = bank_entry(final_root, LC_BANK, check_files=check_files)
        return [(WAVE_LC, 'episode', a44_specs('A44_PLAN', panel, final_root, lc, family=f['lc']['name'],
                                               tag='a44m_lcfix')[:int(n_lc)])]
    raise N.Blocked(command)


# ------------------------------------------------------------------------------------------------ checks / runtime

def check_a44(record, spec):
    from tools.coopmwm.nm_eval_runner import A44_ARMS
    if spec['arm'] not in A44_ARMS:
        return C37.check_a37(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = A._ORIG_CHECK(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a44_sha256_loaded') != spec['checkpoint']['a44_sha256']:
        raise N.Blocked(f'{what}: a44_lcvariant actually loaded differs from its registration')
    if not isinstance(record.get('a44'), dict) or record['a44'].get('a44_arm') != spec['arm']:
        raise N.Blocked(f'{what}: a44 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a44_sha256_loaded=record['a44_sha256_loaded'], a44=deepcopy(record['a44']))


N.CHECKS['episode'] = check_a44


def charged(final_root, prefix=None):
    rows = N.fe.ledger_rows(final_root)
    return sum(r['sim_s'] for r in rows if prefix is None or str(r['id']).startswith(prefix))


class A44Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase lcexec_a44_mech (cap 24,000)."""
    phase = A44_PHASE

    def _caps(self, amount):
        if charged(self.final_root)+amount > A44_CAP_SIM_S:
            raise N.Blocked(f'A44: {A44_PHASE} cap {A44_CAP_SIM_S} would be exceeded')

    def check_capacity(self, amount):
        self._caps(amount)

    def reserve(self, cfg, jobs):
        if not all(str(j['id']).startswith('a44m_') for j in jobs):
            raise N.Blocked('A44: only a44m_* jobs in this root')
        self._caps(sum(j['sim_s'] for j in jobs))
        N.own_reserve(self.final_root, jobs, A44_PHASE, cap=A44_CAP_SIM_S, calls=self.calls, allocation=A44_CAP_SIM_S)


def execute(final_root, wave_list, *, workers, chunk, log, panel=None, runtime=None):
    """a37_celltailor.execute with the A44 runtime and check."""
    final_root = Path(final_root)
    runtime = runtime or A44Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the A44 arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A44_ARMS', None) != ('A44_PLAN', 'A44_BANK') or getattr(R, 'A37_ARMS', None) != C37.A37_ARMS:
            raise N.Blocked('runner lacks the amendment-44 / amendment-37 arms')
        if panel is not None:
            N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a44)[2], len(specs)) for name, _, specs in wave_list}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in wave_list if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A44).exists():
            log(dict(stopped=True, coverage=cov))
            return dict(status='STOPPED', coverage=cov)
        budget, limits, selection = chunk, {}, []
        for name, kind, specs in pending:
            take = min(cov[name][1]-cov[name][0], budget)
            selection.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        N.run_waves(selection, root=plant, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published={k: v[0] for k, v in coverage().items()}, charged_sim_s=charged(final_root)))
    return dict(status='PASS', coverage=coverage(), charged_sim_s=charged(final_root))


def execute_with_retry(final_root, wave_list, *, log, **kw):
    """One retry per failed episode (registered; a37_celltailor.execute_with_retry rule)."""
    for attempt in range(3):
        try:
            return execute(final_root, wave_list, log=log, **kw)
        except Exception as e:  # noqa: BLE001
            jobs = N.interrupted_jobs(final_root)
            used = N._rounds_used(final_root)
            log(dict(failure=f'{type(e).__name__}: {str(e)[:500]}', interrupted=len(jobs), attempt=attempt))
            if not jobs or any(used.get(j, 0) >= 1 for j in jobs) or attempt == 2:
                raise
            value = N.authorize_replay(final_root=final_root, authority=REPLAY_AUTHORITY,
                                       reason=f'{type(e).__name__}: {str(e)[:300]}')
            log(dict(replay_authorized=len(value['jobs']), receipt=value['receipt_sha256']))
    raise N.Blocked('unreachable')


# ------------------------------------------------------------------------------------------------ zero-physics reads

EXECUTOR_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts', 'offers', 'plans')


def _published(final_root, name, specs):
    recs, _, cov = N.published(final_root, name, specs, check_a44)
    if cov != len(specs):
        raise N.Blocked(f'{name}: {cov}/{len(specs)} published')
    return {r['world_id']: r for r in recs}


def identity_compare(final_root, panel):
    """(a): per world, the A37-executor record and the A44-variant record of the same hold plan must match exactly:
    J_veh_s, safety, vehicle-log sha256, and the executor receipts (activations, members, releases, issued count,
    offers, admitted plan)."""
    (n37, _, s37), (n44, _, s44) = waves('identity', panel, final_root)
    r37, r44 = _published(final_root, n37, s37), _published(final_root, n44, s44)
    rows = []
    for wid in [s['world_id'] for s in s37]:
        a, b = r37[wid], r44[wid]
        ea, eb = a['a37'], b['a44']
        diff = [k for k in ('J_veh_s', 'safety') if a.get(k) != b.get(k)]
        if (a.get('vehicle_log') or {}).get('sha256') != (b.get('vehicle_log') or {}).get('sha256'):
            diff.append('vehicle_log.sha256')
        diff += [f'executor.{k}' for k in EXECUTOR_FIELDS if ea.get(k) != eb.get(k)]
        if eb.get('lc_members'):
            diff.append('executor.lc_members (hold plan produced lane-change members)')
        rows.append(dict(world_id=wid, equal=not diff, differences=diff, activations=ea.get('activations'),
                         issued=ea.get('issued')))
    return N.seal(dict(schema='coopmwm.a44.identity.v1', rule=('A37 executor (A37_PLAN, 0a902437) vs A44 variant '
                                                               '(A44_PLAN) on the same hold plan: J, safety, vehicle-log '
                                                               'sha256 and executor receipts identical'),
                       verdict='PASS' if rows and all(r['equal'] for r in rows) else 'FAILED',
                       non_vacuous=any((r['activations'] or 0) > 0 for r in rows), worlds=rows), 'receipt_sha256')


def lc_report(final_root, panel, n_lc=MAX_LC_WORLDS):
    """(b): commanded vs completed vs given-up lane changes, collisions, teleports. Reads the executor receipts and the
    safety counters only (no delay outcome). pass = >= 1 commanded change, every commanded change completed or
    cleanly ended (give-up point or the program's own release), zero collisions and teleports; any collision /
    teleport -> REVIEW (attribution to the command needs the vehicle log; MAIN decides)."""
    ((name, _, specs),) = waves('lcfix', panel, final_root, n_lc=n_lc)
    recs = _published(final_root, name, specs)
    eps, tot = [], dict(members=0, commanded=0, completed=0, gave_up=0, ended_uncompleted=0, never_requested=0,
                        collisions=0, teleports=0)
    for s in specs:
        r = recs[s['world_id']]
        lc = r['a44'].get('lc_members') or []
        ep = dict(world_id=s['world_id'], activations=r['a44'].get('activations'),
                  members=len(lc), commanded=sum(x['first_request'] is not None for x in lc),
                  completed=sum(x['completed'] is not None for x in lc),
                  gave_up=sum(x['first_request'] is not None and x['completed'] is None and x['gave_up'] is not None
                              for x in lc),
                  ended_uncompleted=sum(x['first_request'] is not None and x['completed'] is None and x['gave_up'] is None
                                        for x in lc),
                  never_requested=sum(x['first_request'] is None for x in lc),
                  collisions=int(r['safety'].get('collisions', 0)), teleports=int(r['safety'].get('teleports', 0)),
                  lc_member_ids=sorted({x['vid'] for x in lc}),
                  lc_members=[{k: x[k] for k in ('vid', 'start', 'sumo_from', 'sumo_target', 'onset_s', 'first_request',
                                                 'completed', 'gave_up', 'released')} for x in lc])
        eps.append(ep)
        for k in tot:
            tot[k] += ep[k]
    unsafe = tot['collisions']+tot['teleports'] > 0
    verdict = ('VACUOUS' if tot['commanded'] == 0 else 'REVIEW' if unsafe else
               'PASS' if tot['commanded'] == tot['completed']+tot['gave_up']+tot['ended_uncompleted'] else 'FAILED')
    return N.seal(dict(schema='coopmwm.a44.lc_mech.v1', authority=AUTHORITY, verdict=verdict, totals=tot, episodes=eps,
                       definitions=dict(
                           commanded='the executor put the lc target into the G5 lane request (role onset reached '
                                     'before the give-up point x = X_B - 300)',
                           completed='the member observed in the target WM lane while in its session',
                           gave_up='member reached x >= X_B - 300 before completing (request reverts to own lane)',
                           ended_uncompleted='session released (hold end / drop / left) before completion or give-up',
                           note=('G5 env per-tick outcomes (issued / deferred_edge_end / refused_*) are not in the '
                                 'episode record; collisions / teleports are episode totals (attribution via the '
                                 'vehicle log of lc_member_ids)')),
                       not_a_result='mechanics only; delay outcomes not read'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('panel', 'prepare', 'dry', 'identity', 'identity-compare', 'lcfix', 'lc-report',
                                       'status'))
    p.add_argument('--final-root')
    p.add_argument('--seeds', required=True, help='LO-HI contiguous TRAIN-type block inside the runner A44_MECH_RANGE')
    p.add_argument('--map-spec', help='panel: wz3to2 MAP_SPEC.json')
    p.add_argument('--n-lc', type=int, default=MAX_LC_WORLDS)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk', type=int, default=8)
    a = p.parse_args(argv)
    if a.command == 'panel':
        setup(a.seeds, need_panel=False)
        from datetime import datetime, timezone
        value = N.build_panel(a.map_spec or N.default_spec_path(), datetime.now(timezone.utc).isoformat())
        C37.write_once(PANEL_FILE, json.dumps(value, indent=1, sort_keys=True)+'\n')
        print(json.dumps(dict(panel_sha256=value['panel_sha256'], file=str(PANEL_FILE),
                              seeds=[value['worlds'][0]['world']['seed'], value['worlds'][-1]['world']['seed']])))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    panel = setup(a.seeds)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T')), sort_keys=True, default=str)
        print(line, flush=True)
        with (root/'A44_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root), indent=1))
    elif a.command == 'dry':
        from tools.coopmwm import a37_cellbank as C
        from tools.coopmwm import a44_lcvariant as V
        out = dict(setting=SETTING, panel_sha256=panel['panel_sha256'], pinned=PANEL_SHA256, cell=panel['cell']['cell_key'],
                   a37_sha256=C.a37_digest(), a44_sha256=V.a44_digest(), phase=A44_PHASE, cap_sim_s=A44_CAP_SIM_S,
                   charged=charged(root))
        for c in ('identity', 'lcfix'):
            ws = waves(c, panel, root, n_lc=a.n_lc)
            out[c] = {n: dict(n=len(s), first=s[0]['job_id'], arm=s[0]['arm'], family=s[0]['checkpoint'].get(
                'a44_family', s[0]['checkpoint'].get('a37_family')), seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                cell=s[0]['world']['cell_key']) for n, _, s in ws}
            out[c+'_sim_s'] = sum(len(s)*N.HORIZON_S for _, _, s in ws)
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'lcfix'):
        require_pin(panel)
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            ws = waves(a.command, panel, root, n_lc=a.n_lc)
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in ws], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=charged(root)))
            res = execute_with_retry(root, ws, workers=a.workers, chunk=a.chunk, log=log, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root, panel)
        N.publish(root/'A44_IDENTITY.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], non_vacuous=v['non_vacuous'],
                              worlds=[(r['world_id'], r['equal'], r['differences']) for r in v['worlds']])))
    elif a.command == 'lc-report':
        v = lc_report(root, panel, a.n_lc)
        N.publish(root/'A44_LC_MECH.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], totals=v['totals'])))
    else:   # status
        out = dict(charged=charged(root), cap=A44_CAP_SIM_S)
        for c in ('identity', 'lcfix'):
            for n, _, s in waves(c, panel, root, n_lc=a.n_lc, check_files=False):
                out[n] = f'{N.published(root, n, s, check_a44)[2]}/{len(s)}'
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
