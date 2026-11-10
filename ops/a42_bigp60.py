"""AMENDMENT 42 (MAIN 2026-11-10 ~22:20; registered before any A42 search, SUMO or outcome): WM search for 3-6 CAV group
plans at D2800_p60, verified in SUMO against the hand-written hold. Exploratory, non-design cell, never pooled.

Thin driver modelled on ops/a37_celltailor.py (same mirrored bank-root bindings / plant / U0 / DUO_HOLD /
LEARNED_TRIGGER as the amendment-35 D2800_p60 cell; waves through the unchanged nm_finaleval.run_waves: canary first,
per-episode ledger rows precharged in <final_root>/CHARGES.jsonl). Executor = the amendment-37 executor UNCHANGED
(runner arms A37_PLAN = one candidate, CELL_BANK = BIG_BANK; a37 digest re-verified per episode).

Budget: ONE ledger phase ``bigp60_a42`` capped at 720,000 sim-s = identity 4,800 + verification 7 x 24 x 1,200 =
201,600 + confirmation 4 x 96 x 1,200 = 460,800 + retry reserve 52,800. One retry per failed episode on the same world
(charged; receipts REPLAY_AUTHORIZED_*); never an outcome-based retry.

Stages / commands (python ops/a42_bigp60.py <cmd> --final-root <root> --setting <setting>):
  select           zero physics, laptop: --search STRATEGY_SEARCH_D2800_p60.json --out ops/a42/CANDIDATES_D2800_p60.json
                   (search `final` by WM score desc, skip comparator / banned / executor-refused, top K = 6; STOP rule)
  prepare          zero physics: a42_banks/A42_CANDS_D2800_p60.json (unconditional rows of the pinned CANDIDATES)
  dry              zero physics: build every spec of the setting, check bindings / digests, print counts
  identity         U0 on 2 worlds of each of the p20c and a35-D3200 panels (4 episodes, refs IDENTITY_REFS f049f546)
  identity-compare J (|dJ| <= 1e-6) and safety vs ops/a37/IDENTITY_REFS.json -> <root>/IDENTITY_GATE.json
  verify           (setting a42v_D2800_p60) U0 + the 6 candidates (A37_PLAN, start whenever feasible) x 24 worlds
  admit            (setting a42v_D2800_p60) zero physics: the A37 UNC rule per candidate -> ADMISSION_D2800_p60.json and,
                   if any plan passes, a42_banks/A42_BIG_BANK_D2800_p60.json (else BIG_BANK := U0, no confirmation)
  confirm          (setting a42c_D2800_p60) U0, DUO_HOLD, LEARNED_TRIGGER, BIG_BANK on the fresh 96-world panel
  reports          zero physics: vehicles_report.json next to every published episode log of the setting
  analyse          (setting a42c_D2800_p60) registered conjunction + secondaries -> A42_RESULTS_D2800_p60.json (--out)
  status           coverage per wave + ledger totals
"""
from __future__ import annotations

import argparse
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
import a32_ablation as A  # noqa: E402
import a35_transfer as T  # noqa: E402
import a37_celltailor as C37  # noqa: E402  (registers check_a37 for the A37 executor arms)

A42_PHASE = 'bigp60_a42'
A42_CAP_SIM_S = 720_000
IDENTITY_SIM_S, VERIFY_SIM_S, CONFIRM_SIM_S = 4_800, 7*24*1_200, 4*96*1_200
RETRY_RESERVE_SIM_S = A42_CAP_SIM_S-IDENTITY_SIM_S-VERIFY_SIM_S-CONFIRM_SIM_S
STOP_A42 = 'STOP_A42'
#: the final root on the run host: bank paths in the specs are host paths (identical specs on the laptop)
HOST_BASE = os.environ.get('A42_HOST_BASE', '/root/a42v')
HOST_ROOT = f'{HOST_BASE}/root'
CELL = 'D2800_p60'
VERIFY_SETTING, CONFIRM_SETTING = 'a42v_D2800_p60', 'a42c_D2800_p60'
SETTINGS = (VERIFY_SETTING, CONFIRM_SETTING)
CONFIRM_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'BIG_BANK')
RUNNER_ARM = dict(BIG_BANK='CELL_BANK', PLAN='A37_PLAN')     # the A37 executor arms (unchanged)
BANK_DIR = 'a42_banks'
A42_DIR = HERE/'a42'
CANDS_FILE = A42_DIR/f'CANDIDATES_{CELL}.json'
#: pinned when MAIN registers the candidates (before any A42 SUMO); None = not registered -> verify refuses
CANDIDATES_DOC_SHA256 = None
IDENT_WAVE = 'A42_IDENT_U0'
#: identity panels (A37/A41 pattern; no reference exists for an a35 D2800_p60 world, see PLAN note): tag, file, pin
IDENT_PANELS = (('p20c', 'nm_eval_panel_wz3to2_p20c_D2800.json',
                 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2'),
                ('a35d3200', 'nm_eval_panel_wz3to2_a35_D3200_p20.json',
                 '152dbfe93017f4b5c97f8925c832b332578fd66c53037208f4ad157733a6b280'))
IDENTITY_REFS_SHA256 = 'f049f546'          # receipt prefix of ops/a37/IDENTITY_REFS.json (file sha prefix checked)
AUTHORITY = ('MAIN 2026-11-10 ~22:20 AMENDMENT 42 (owner: "can we use the world model to search for more vehicle plans '
             'on the 60% cav coverage ... and possibly beat the hand written"): WM search forced to 3-6 CAV groups at '
             'D2800_p60, UNC verification, BIG_BANK vs DUO_HOLD on a fresh 96-world panel; phase bigp60_a42 cap 720,000')
PRIMARY_RULE = ('larger-group plans beat the hand-written hold at D2800_p60 iff BIG_BANK - DUO_HOLD total delay CI95 '
                'upper < 0 (paired percentile bootstrap 10,000, seed 20261104) AND BIG_BANK passes the group rule vs U0 '
                '(five registered groups present, each mean per-vehicle increase <= 10 s, U0 labels)')
SECONDARY = (('BIG_BANK', 'U0'), ('BIG_BANK', 'LEARNED_TRIGGER'))
CONTEXT = (('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'U0'), ('LEARNED_TRIGGER', 'DUO_HOLD'))
REPLAY_AUTHORITY = ('MAIN AMENDMENT 42 registered rule: one charged retry per failed episode on the same world; '
                    'no outcome-based retries')


def setup(setting):
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


def budget_plan():
    total = IDENTITY_SIM_S+VERIFY_SIM_S+CONFIRM_SIM_S
    if total+RETRY_RESERVE_SIM_S != A42_CAP_SIM_S or RETRY_RESERVE_SIM_S < 0:
        raise N.Blocked('A42 budget arithmetic')
    return dict(phase=A42_PHASE, cap_sim_s=A42_CAP_SIM_S, identity=IDENTITY_SIM_S, verification=VERIFY_SIM_S,
                confirmation=CONFIRM_SIM_S, planned=total, retry_reserve=RETRY_RESERVE_SIM_S)


def check_total_cap(total_cap, committed):
    """Registration: 'total_cap checked before launch'. committed = sum of every other registered phase cap / charge
    (MAIN's ledger). Blocks when this phase does not fit."""
    if int(committed)+A42_CAP_SIM_S > int(total_cap):
        raise N.Blocked(f'A42: {committed} + {A42_CAP_SIM_S} > total_cap {total_cap}')
    return dict(total_cap=int(total_cap), committed=int(committed), a42_cap=A42_CAP_SIM_S,
                headroom=int(total_cap)-int(committed)-A42_CAP_SIM_S)


# ------------------------------------------------------------------------------------------------ candidates / banks

def select(search_path, out):
    from tools.coopmwm import a42_bigbank as B
    search_path = Path(search_path)
    doc = B.seal(B.select(json.loads(search_path.read_text(encoding='utf-8')), N.file_sha(search_path)))
    out = Path(out)
    if out.exists():
        raise N.Blocked(f'{out}: exists (pre-outcome file; never overwrite)')
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return doc


def candidates(path=None, pinned=None):
    """The pinned pre-outcome candidates (K = 6); refuses an unpinned / edited file and a STOP verdict."""
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    path = Path(path or CANDS_FILE)
    pinned = CANDIDATES_DOC_SHA256 if pinned is None else pinned
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run select after the search, then MAIN pins CANDIDATES_DOC_SHA256)')
    doc = N.read(path)
    if B.doc_sha(doc) != doc.get('doc_sha256'):
        raise N.Blocked(f'{path.name}: doc_sha256 differs (never edit a pre-outcome file)')
    if pinned is None or doc['doc_sha256'] != pinned:
        raise N.Blocked(f'{path.name}: not the registered selection (pin {pinned})')
    if doc['cell'] != CELL or doc['k'] != B.K:
        raise N.Blocked(f'{path.name}: not the registered cell / K')
    if doc['STOP']:
        raise N.Blocked(f'A42 STOP: {doc["verdict"]} (no verification)')
    for c in doc['candidates']:
        if C.spec_sha(c['strategy']) != c['strategy_sha256']:
            raise N.Blocked(f'{c["family"]}: strategy sha differs')
    return doc['candidates']


def bank_path(final_root, name):
    return Path(final_root)/BANK_DIR/name


def prepare(final_root, cands=None):
    from tools.coopmwm import a37_cellbank as C
    cands = candidates() if cands is None else cands
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = C37.write_once(bank_path(final_root, f'A42_CANDS_{CELL}.json'), C37.bank_text(doc))
    return dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in cands])


def _bank_entry(final_root, name, *, check_files=True):
    p = bank_path(final_root, name)
    if not p.is_file():
        if check_files:
            raise N.Blocked(f'{p}: missing (run prepare / admit first)')
        return None
    return dict(path=f'{HOST_ROOT}/{BANK_DIR}/{name}', file_sha256=N.file_sha(p))


def _a37_specs(arm, panel, final_root, bank_entry, *, family=None, tag):
    """The registered LEARNED_TRIGGER specs (same plant / U0 binding) with arm -> A37 executor arm, kind 'a37', the
    world's own cell (D2800_p60, no bank_cell_key: the A42 plans were searched for this cell)."""
    from tools.coopmwm import a37_cellbank as C
    u0, oracle = A.bindings(final_root)
    base = N.episode_specs('LEARNED_TRIGGER', panel, None, Path(final_root)/N.PRERUN_PLANT, 'mappo_frozen', oracle, u0)
    out = []
    for i, s in enumerate(base):
        ck = deepcopy(s['checkpoint'])
        ck.pop('learned_trigger_sha256')
        ck.pop('learned_trigger_model')
        ck.update(kind='a37', a37_arm=arm, a37_sha256=C.a37_digest(), a37_bank=dict(bank_entry))
        if family is not None:
            ck['a37_family'] = family
        out.append(dict(s, arm=arm, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def _retag(specs, tag):
    return [dict(s, job_id=f'{tag}_{i:04}') for i, s in enumerate(specs)]


def verify_waves(panel, final_root, *, check_files=True, cands=None):
    """[(wave, 'episode', specs)]: U0 then every candidate (CANDIDATES order) on the 24-world TRAIN-type block."""
    u0, _ = A.bindings(final_root)
    cands = candidates() if cands is None else cands
    waves = [('A42V_U0', 'episode', _retag(N.episode_specs('U0', panel, None, Path(final_root)/N.PRERUN_PLANT,
                                                         'mappo_frozen', None, u0), 'a42v_u0'))]
    entry = _bank_entry(final_root, f'A42_CANDS_{CELL}.json', check_files=check_files) or dict(path='', file_sha256='')
    for c in cands:
        waves.append((f'A42V_{c["family"]}', 'episode',
                      _a37_specs(RUNNER_ARM['PLAN'], panel, final_root, entry, family=c['family'],
                                 tag=f'a42v_{c["key"][:8]}')))
    return waves


def admission(final_root):
    p = Path(final_root)/f'ADMISSION_{CELL}.json'
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def confirm_specs(arm, panel, final_root, *, check_files=True):
    if arm in ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD'):
        return _retag(T._arm_specs(arm, panel, final_root, check_files=check_files), f'a42c_{arm.lower()}')
    if arm == 'BIG_BANK':
        adm = admission(final_root)
        if adm is None:
            raise N.Blocked('no ADMISSION file (run verify + admit first)')
        if not adm['admitted']:
            return []                      # BIG_BANK := U0 by definition; no confirmation run
        entry = _bank_entry(final_root, f'A42_BIG_BANK_{CELL}.json')
        if entry['file_sha256'] != adm['big_bank']['file_sha256']:
            raise N.Blocked('BIG_BANK bank differs from the admission receipt')
        return _a37_specs(RUNNER_ARM['BIG_BANK'], panel, final_root, entry, tag='a42c_big_bank')
    raise N.Blocked(f'no A42 confirmation arm {arm!r}')


def confirm_wave(arm):
    return f'A42C_{arm}'


def ident_specs(final_root):
    u0, _ = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    out = []
    for tag, name, pin in IDENT_PANELS:
        panel = N.verify(N.read(HERE/name), 'panel_sha256', name)
        if panel['panel_sha256'] != pin:
            raise N.Blocked(f'identity panel {name}: sha differs from the registered panel')
        specs = N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)[:N.IDENTITY_WORLDS]
        out += [dict(s, job_id=f'a42i_{tag}_u0_{i:04}') for i, s in enumerate(specs)]
    return out


# ------------------------------------------------------------------------------------------------ runtime

def charged(final_root, prefix=None):
    rows = N.fe.ledger_rows(final_root)
    return sum(r['sim_s'] for r in rows if prefix is None or str(r['id']).startswith(prefix))


class A42Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase bigp60_a42 (cap 720,000)."""
    phase = A42_PHASE

    def check_capacity(self, amount):
        if charged(self.final_root)+amount > A42_CAP_SIM_S:
            raise N.Blocked(f'A42: {A42_PHASE} cap {A42_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        if any(not str(j['id']).startswith('a42') for j in jobs):
            raise N.Blocked('A42: job ids are a42*')
        self.check_capacity(sum(j['sim_s'] for j in jobs))
        N.own_reserve(self.final_root, jobs, A42_PHASE, cap=A42_CAP_SIM_S, calls=self.calls, allocation=A42_CAP_SIM_S)


def execute(final_root, waves, *, workers, chunk, log, panel=None, runtime=None):
    """a37_celltailor.execute with the A42 runtime: pending waves in chunks of ``chunk`` episodes."""
    final_root = Path(final_root)
    runtime = runtime or A42Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the A42 arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A37_ARMS', None) != C37.A37_ARMS or getattr(R, 'A42_RANGES', None) is None:
            raise N.Blocked('runner lacks the amendment-37 executor arms / amendment-42 ranges')
        if panel is not None:
            N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, C37.check_a37)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A42).exists():
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


def execute_with_retry(final_root, waves, *, log, **kw):
    """One retry per failed episode (registered): a failed dispatch leaves charged-unpublished jobs; if none of them
    was retried before, authorise ONE replay round (fresh <job>__r1 dirs, charged <job>#r1) and run again."""
    for attempt in range(3):
        try:
            return execute(final_root, waves, log=log, **kw)
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


def confirm_fits(final_root, n_arms):
    """Before confirmation: the remaining phase cap must hold every confirmation episode (no partial panel)."""
    need = n_arms*96*N.HORIZON_S
    left = A42_CAP_SIM_S-charged(final_root)
    if need > left:
        raise N.Blocked(f'A42: confirmation needs {need} sim-s, {left} left in {A42_PHASE} (MAIN decides)')
    return dict(need=need, left=left)


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, refs_path=None):
    refs_path = Path(refs_path or C37.CANDS_DIR/'IDENTITY_REFS.json')
    refs = N.read(refs_path)
    if not str(refs.get('receipt_sha256', '')).startswith(IDENTITY_REFS_SHA256):
        raise N.Blocked('IDENTITY_REFS is not the registered f049f546 receipt')
    specs = ident_specs(final_root)
    got, _, cov = N.published(final_root, IDENT_WAVE, specs, C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{IDENT_WAVE}: {cov}/{len(specs)} published')
    rows = []
    for s, r in zip(specs, got):
        tag = s['job_id'].split('_')[1]
        ref = refs['worlds'][f'{tag}:{r["world_id"]}']
        rows.append(dict(panel=tag, world_id=r['world_id'], j_host=r['J_veh_s'], j_ref=ref['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-ref['J_veh_s']), safety_equal=r['safety'] == ref['safety'],
                         ref_source=ref['source']))
    per = {}
    for x in rows:
        per[x['panel']] = per.get(x['panel'], True) and x['abs_diff'] <= 1e-6 and x['safety_equal']
    return N.seal(dict(schema='coopmwm.a42.identity.v1', rule='|J diff| <= 1e-6 veh-s and identical safety, U0',
                       verdict='PASS' if all(per.values()) else 'FAILED',
                       per_panel={k: 'PASS' if v else 'FAILED' for k, v in per.items()}, worlds=rows,
                       consequence=('as A37: the chain continues either way; every A42 arm (verification and '
                                    'confirmation) runs on this host and is analysed only among themselves')),
                  'receipt_sha256')


# ------------------------------------------------------------------------------------------------ admission / analysis

def _spec_map(cands):
    return {c['family']: c['strategy'] for c in cands}


def _group_sizes(recs, worlds, specs_by_family):
    from tools.coopmwm import a42_bigbank as B
    rows = []
    for w in worlds:
        rows += B.realised_sizes(recs[w].get('a37') or {}, specs_by_family)
    return B.group_size_summary(rows)


def admit(final_root, panel, cands=None):
    """A37 UNC rule per candidate on the verification block -> ADMISSION_D2800_p60.json (sealed) + the BIG_BANK bank."""
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    cands = candidates() if cands is None else cands
    if admission(final_root) is not None:
        raise N.Blocked(f'ADMISSION_{CELL}.json exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    waves = verify_waves(panel, final_root, cands=cands)
    have, sources = {}, {}
    for (name, _, specs), arm in zip(waves, ['U0']+[c['family'] for c in cands]):
        have[arm], sources[arm] = C37._load(final_root, name, specs)
    reports = C37._reports(final_root, have, worlds)
    verdicts, rows = {}, []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, 'U0', bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have['U0'][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        v.update(family=fam, wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'], key=cand['key'],
                 n_roles=len(cand['strategy']['roles']), cvar90_vs_u0=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_u0_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=_group_sizes(have[fam], worlds, _spec_map(cands)),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        verdicts[fam] = v
        rows.append(v)
    plans = C.admitted_plans(cands, verdicts)
    big = None
    if plans:
        p = C37.write_once(bank_path(final_root, f'A42_BIG_BANK_{CELL}.json'), C37.bank_text(C.bank_doc(plans, CELL)))
        big = dict(path=f'{HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p))
    value = N.seal(dict(schema='coopmwm.a42.admission.v1', amendment='42', cell=CELL, rule=C.UNC_RULE,
                        multiplicity=f'{len(rows)} picks each tested at 95% (disclosed, as A37)',
                        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
                        admitted=[p['family'] for p in plans], priority=[(p['family'], p['lb95_veh_s']) for p in plans],
                        big_bank=big, big_bank_is_u0=not plans,
                        consequence=('none admitted: BIG_BANK := U0, reported as the outcome, no confirmation run'
                                     if not plans else 'confirmation runs U0, DUO_HOLD, LEARNED_TRIGGER, BIG_BANK'),
                        u0_safety_totals={k: int(sum(have['U0'][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()}),
                   'receipt_sha256')
    N.publish(final_root/f'ADMISSION_{CELL}.json', value)
    return value


def _contrast(have, reports, worlds, arm, ref, bootstrap, kind):
    import numpy as np
    row = dict(treatment=arm, reference=ref, kind=kind,
               convention='diff = treatment - reference; negative = treatment has less delay')
    row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
    for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
        row[f'{k}_minus_ref'] = N.fe.paired_stats(np.array([have[arm][w]['safety'][k] for w in worlds], float),
                                                  np.array([have[ref][w]['safety'][k] for w in worlds], float),
                                                  bootstrap, pct=False)
    return row


def primary_verdict(big_minus_duo, big_minus_u0):
    """The registered conjunction from the two contrast rows (pure; unit-tested)."""
    ci = big_minus_duo['total_minus_ref']['ci95']
    groups = big_minus_u0['groups'] or {}
    part_a = bool(ci[1] < 0)
    part_b = groups.get('all_pass') is True
    return dict(rule=PRIMARY_RULE, a_total_ci95=ci, a_pass=part_a, b_group_rule_vs_u0=part_b,
                b_worst=max(((g, v.get('increase_mean_s')) for g, v in (groups.get('groups') or {}).items()
                             if v.get('registered') and 'increase_mean_s' in v), key=lambda x: x[1], default=None),
                supported=bool(part_a and part_b),
                verdict='SUPPORTED' if part_a and part_b else 'NOT SUPPORTED')


def analyse(final_root, panel, cands=None):
    """Registered amendment-42 conjunction + secondaries (paired percentile bootstrap 10,000, seed 20261104;
    diff = treatment - reference, negative = less delay). One panel, one shot, never pooled."""
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    adm = admission(final_root)
    if adm is None:
        raise N.Blocked('ADMISSION missing')
    if adm['big_bank_is_u0']:
        return N.seal(dict(schema='coopmwm.a42.analysis.v1', amendment='42', authority=AUTHORITY, cell=CELL,
                           primary=dict(rule=PRIMARY_RULE, supported=False, verdict='NOT SUPPORTED',
                                        note='no plan admitted: BIG_BANK := U0, reported as the outcome; no confirmation'),
                           admission=dict(admitted=[], receipt_sha256=adm['receipt_sha256'])), 'analysis_sha256')
    cands = candidates() if cands is None else cands
    have, sources = {}, {}
    for arm in CONFIRM_ARMS:
        have[arm], sources[arm] = C37._load(final_root, confirm_wave(arm),
                                            confirm_specs(arm, panel, final_root, check_files=False))
    reports = C37._reports(final_root, have, worlds)
    contrasts = [_contrast(have, reports, worlds, 'BIG_BANK', 'DUO_HOLD', bootstrap, 'primary_part_a')]
    contrasts += [_contrast(have, reports, worlds, a, r, bootstrap, 'secondary') for a, r in SECONDARY]
    contrasts += [_contrast(have, reports, worlds, a, r, bootstrap, 'context') for a, r in CONTEXT]
    prim = primary_verdict(contrasts[0], contrasts[1])
    arms = {}
    specs = _spec_map(cands)
    for a in CONFIRM_ARMS:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rec = [have[a][w].get('a37') or have[a][w].get('a32') or have[a][w].get('learned_trigger') or {}
               for w in worlds]
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam)
    arms['BIG_BANK']['realised_group_size'] = _group_sizes(have['BIG_BANK'], worlds, specs)
    return N.seal(dict(schema='coopmwm.a42.analysis.v1', amendment='42', authority=AUTHORITY, cell=CELL,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       primary=prim, contrasts=contrasts, arms=arms,
                       groups_vs_u0={r['treatment']: r['groups'] for r in contrasts if r['reference'] == 'U0'},
                       admission=dict(admitted=adm['admitted'], receipt_sha256=adm['receipt_sha256'],
                                      big_bank=adm['big_bank']),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       label='amendment 42; exploratory row at 60% CAVs; never pooled; one shot'), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def stage_waves(command, setting, panel, root, *, check_files=True):
    if command == 'verify':
        if setting != VERIFY_SETTING:
            raise N.Blocked(f'verify runs on {VERIFY_SETTING}')
        return verify_waves(panel, root, check_files=check_files)
    if command == 'confirm':
        if setting != CONFIRM_SETTING:
            raise N.Blocked(f'confirm runs on {CONFIRM_SETTING}')
        return [(confirm_wave(a), 'episode', s) for a in CONFIRM_ARMS
                if (s := confirm_specs(a, panel, root, check_files=check_files))]
    raise N.Blocked(command)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('select', 'prepare', 'dry', 'identity', 'identity-compare', 'verify', 'admit',
                                       'confirm', 'reports', 'analyse', 'status', 'budget'))
    p.add_argument('--final-root')
    p.add_argument('--setting', default=VERIFY_SETTING, choices=SETTINGS)
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=46)
    p.add_argument('--search')
    p.add_argument('--out')
    p.add_argument('--total-cap', type=int)
    p.add_argument('--committed', type=int)
    a = p.parse_args(argv)
    if a.command == 'select':
        doc = select(a.search, a.out or CANDS_FILE)
        print(json.dumps(dict(doc_sha256=doc['doc_sha256'], verdict=doc['verdict'],
                              skip_counts=doc['skip_counts'], n_eligible=doc['n_eligible'],
                              candidates=[(c['family'], c['score'], c['n_roles'], c['wm']['feasible'])
                                          for c in doc['candidates']])))
        return
    if a.command == 'budget':
        out = budget_plan()
        if a.total_cap is not None:
            out['total_cap_check'] = check_total_cap(a.total_cap, a.committed or 0)
        print(json.dumps(out, indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    panel = setup(a.setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=a.setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/'A42_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root), indent=1))
    elif a.command == 'dry':
        from tools.coopmwm import a37_cellbank as C
        out = dict(setting=a.setting, panel_sha256=panel['panel_sha256'], cell=panel['cell']['cell_key'],
                   a37_sha256=C.a37_digest(), ident=[s['job_id'] for s in ident_specs(root)], budget=budget_plan())
        c = 'verify' if a.setting == VERIFY_SETTING else 'confirm'
        try:
            ws = stage_waves(c, a.setting, panel, root)
        except N.Blocked as e:            # confirmation before admission: the three frozen arms only
            ws = [(confirm_wave(x), 'episode', confirm_specs(x, panel, root)) for x in CONFIRM_ARMS[:3]]
            out['note'] = str(e)
        out[c] = {n: dict(n=len(s), first=s[0]['job_id'], seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                          cell=s[0]['world']['cell_key'], arm=s[0]['arm']) for n, _, s in ws}
        out[c+'_sim_s'] = sum(len(s)*N.HORIZON_S for _, _, s in ws)
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'verify', 'confirm'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(IDENT_WAVE, 'episode', ident_specs(root))]
            else:
                waves = stage_waves(a.command, a.setting, panel, root)
                if a.command == 'confirm':
                    if not waves:
                        log(dict(done='confirm', status='SKIPPED_BIG_BANK_IS_U0'))
                        return
                    log(dict(confirm_fits=confirm_fits(root, len(waves))))
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=charged(root)))
            res = execute_with_retry(root, waves, workers=a.workers, chunk=a.chunk, log=log,
                                     panel=None if a.command == 'identity' else panel)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root)
        N.publish(root/'IDENTITY_GATE.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], per_panel=v['per_panel'])))
    elif a.command == 'admit':
        if a.setting != VERIFY_SETTING:
            p.error(f'admit runs on {VERIFY_SETTING}')
        v = admit(root, panel)
        print(json.dumps(dict(admitted=v['admitted'], big_bank_is_u0=v['big_bank_is_u0'],
                              receipt_sha256=v['receipt_sha256'])))
    elif a.command == 'analyse':
        if a.setting != CONFIRM_SETTING:
            p.error(f'analyse runs on {CONFIRM_SETTING}')
        v = analyse(root, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A42_RESULTS_{CELL}.json', v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], verdict=v['primary']['verdict'])))
    elif a.command == 'status':
        out = dict(charged=charged(root), cap=A42_CAP_SIM_S)
        for c in ('verify', 'confirm'):
            try:
                for n, _, s in stage_waves(c, a.setting, panel, root, check_files=False):
                    out[n] = f'{N.published(root, n, s, C37.check_a37)[2]}/{len(s)}'
            except N.Blocked:
                pass
        print(json.dumps(out, indent=1))
    else:   # reports
        n = 0
        waves = []
        for c in ('verify', 'confirm'):
            try:
                waves += stage_waves(c, a.setting, panel, root, check_files=False)
            except N.Blocked:
                pass
        if a.setting == VERIFY_SETTING:
            waves.append((IDENT_WAVE, 'episode', ident_specs(root)))
        for name, _, specs in waves:
            recs, _, _ = N.published(root, name, specs, C37.check_a37)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
