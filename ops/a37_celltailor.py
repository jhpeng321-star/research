"""AMENDMENT 37 (MAIN 2026-11-10; registered before any outcome of these arms): condition tailoring on wz3to2.

Do plans the WM searched FOR a demand cell do better AT that cell than the frozen D2800 bank (LEARNED_TRIGGER) and the
hand-written hold (DUO_HOLD)? Cells D3200_p20 and D2400_p20, never pooled. Thin driver over ops/nm_finaleval.py,
ops/a32_ablation.py and ops/a35_transfer.py (same mirrored bank-root bindings and plant; waves through the unchanged
nm_finaleval.run_waves: canary first, per-episode ledger rows precharged in <final_root>/CHARGES.jsonl).

Budget: ONE ledger phase ``celltailor_a37`` capped at 1,900,000 sim-s (identity + verification + evaluation + 37b +
retries); verification (job ids a37v_*) additionally capped at 600,000. One retry per failed episode (same world,
charged; MAIN-registered rule, receipts REPLAY_AUTHORIZED_*); never an outcome-based retry.

Stages / commands (python ops/a37_celltailor.py <cmd> --final-root <root> --setting <setting>):
  prepare            zero physics: write a37_banks/A37_CANDS_<cell>.json (unconditional rows of the CANDIDATES files)
  dry                zero physics: build every spec of the setting, check bindings/digests, print counts
  identity           U0 on 2 worlds of each of the p20c / a33 / a35-D3200 / a35-D2400 panels (8 episodes)
  identity-compare   J (|dJ| <= 1e-6) and safety vs ops/a37/IDENTITY_REFS.json -> <root>/IDENTITY_GATE.json
  verify             (setting a37v_<cell>) U0 + every candidate (A37_PLAN, start whenever feasible) x 24 TRAIN worlds
  admit              (setting a37v_<cell>) zero physics: the UNC rule per candidate -> ADMISSION_<cell>.json and,
                     if any plan passes, a37_banks/A37_CELL_BANK_<cell>.json (else CELL_BANK := U0)
  eval               (setting a35_<cell>) U0, LEARNED_TRIGGER, DUO_HOLD, CELL_BANK on the amendment-35 panel
  native             (setting a33) 37b: NATIVE on the a33 fresh panel (+ U0 when the a33 identity worlds failed)
  reports            zero physics: vehicles_report.json next to every published episode log of the setting
  analyse            (setting a35_<cell>) the registered contrasts -> A37_RESULTS_<cell>.json (--out)
  analyse-native     (setting a33) NATIVE - U0 (context, no pass/fail) -> A37B_NATIVE.json (--out, --a33-root)
  status             coverage per wave + ledger totals
  make-identity-refs (laptop) the reference J/safety of the 8 identity worlds from local sealed copies
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

A37_PHASE = 'celltailor_a37'
A37_CAP_SIM_S = 1_900_000
VERIFY_CAP_SIM_S = 600_000
STOP_A37 = 'STOP_A37'
#: the final root on the run host: bank paths in the specs are host paths (identical specs on the laptop)
HOST_ROOT = '/root/a37ct/root'
CELLS = ('D3200_p20', 'D2400_p20')
VERIFY_SETTING = {c: f'a37v_{c}' for c in CELLS}
EVAL_SETTING = {c: f'a35_{c}' for c in CELLS}
NATIVE_SETTING = 'a33'
SETTINGS = tuple(VERIFY_SETTING.values()) + tuple(EVAL_SETTING.values()) + (NATIVE_SETTING,)
EVAL_ARMS = ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD', 'CELL_BANK')
A37_ARMS = ('A37_PLAN', 'CELL_BANK')
BANK_DIR = 'a37_banks'
CANDS_DIR = HERE/'a37'
IDENT_WAVE = 'A37_IDENT_U0'
#: identity panels: (tag, panel file, pinned panel sha)
IDENT_PANELS = (('p20c', 'nm_eval_panel_wz3to2_p20c_D2800.json',
                 'd2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2'),
                ('a33', 'nm_eval_panel_wz3to2_a33_D2800.json',
                 'ab2933df19e84c2bb4c1f8ef2f97f3fdb2807763739b64a6943520941cfa47b5'),
                ('a35d3200', 'nm_eval_panel_wz3to2_a35_D3200_p20.json',
                 '152dbfe93017f4b5c97f8925c832b332578fd66c53037208f4ad157733a6b280'),
                ('a35d2400', 'nm_eval_panel_wz3to2_a35_D2400_p20.json',
                 '456c5f3f9721b679b16132ffeea24b51b553530453c4f00e5bd670f5f72eb5ed'))
AUTHORITY = ('MAIN 2026-11-10 AMENDMENT 37 (owner "build the two with opus and I will turn the cpu on tonight"): '
             'condition tailoring D3200_p20 / D2400_p20; CELL_BANK vs LEARNED_TRIGGER / DUO_HOLD / U0 on the a35 panels; '
             'phase celltailor_a37 cap 1,900,000 sim-s (verification <= 600,000)')
PRIMARY = ('CELL_BANK', 'LEARNED_TRIGGER')
CONTRASTS = (('CELL_BANK', 'LEARNED_TRIGGER'), ('CELL_BANK', 'DUO_HOLD'), ('CELL_BANK', 'U0'),
             ('LEARNED_TRIGGER', 'DUO_HOLD'))
CONTEXT = (('LEARNED_TRIGGER', 'U0'), ('DUO_HOLD', 'U0'))
REPLAY_AUTHORITY = ('MAIN AMENDMENT 37 registered rule: one retry per failed episode on the same world (charged); '
                    'no outcome-based retries')


def setup(setting):
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


def cell_of(setting):
    return next((c for c in CELLS if setting in (VERIFY_SETTING[c], EVAL_SETTING[c])), None)


# ------------------------------------------------------------------------------------------------ candidates / banks

def candidates(cell):
    from tools.coopmwm import a37_cellbank as C
    doc = N.read(CANDS_DIR/f'CANDIDATES_{cell}.json')
    body = {k: v for k, v in doc.items() if k != 'doc_sha256'}
    import hashlib
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != doc['doc_sha256']:
        raise N.Blocked(f'CANDIDATES_{cell}: doc_sha256 differs (never edit a pre-outcome file)')
    if doc['cell'] != cell or doc['search']['file_sha256'] != C.SEARCH_SHA256[cell] or doc['k'] != C.k_per_cell():
        raise N.Blocked(f'CANDIDATES_{cell}: not the registered selection')
    for c in doc['candidates']:
        if C.spec_sha(c['strategy']) != c['strategy_sha256']:
            raise N.Blocked(f'{c["family"]}: strategy sha differs')
    return doc['candidates']


def bank_path(final_root, name):
    return Path(final_root)/BANK_DIR/name


def write_once(path, value_text):
    """Publish a zero-outcome file once; an existing file must be byte-identical (never overwrite)."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != value_text.encode('utf-8'):
            raise N.Blocked(f'{path}: exists with different content (never overwrite)')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(value_text.encode('utf-8'))
    os.replace(tmp, path)
    return path


def bank_text(doc):
    return json.dumps(doc, indent=1, sort_keys=True)+'\n'


def prepare(final_root):
    from tools.coopmwm import a37_cellbank as C
    out = {}
    for cell in CELLS:
        cands = candidates(cell)
        doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], cell)
        p = write_once(bank_path(final_root, f'A37_CANDS_{cell}.json'), bank_text(doc))
        out[cell] = dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in cands])
    return out


def _bank_entry(final_root, name, *, check_files=True):
    p = bank_path(final_root, name)
    if not p.is_file():
        if check_files:
            raise N.Blocked(f'{p}: missing (run prepare / admit first)')
        return None
    return dict(path=f'{HOST_ROOT}/{BANK_DIR}/{name}', file_sha256=N.file_sha(p))


def _a37_specs(arm, panel, final_root, bank_entry, *, family=None, tag):
    """The registered LEARNED_TRIGGER specs (same plant / U0 binding) with arm -> A37 arm, checkpoint kind 'a37'."""
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


def verify_waves(cell, panel, final_root, *, check_files=True):
    """[(wave, 'episode', specs)]: U0 then every candidate (CANDIDATES order) on the 24-world TRAIN block."""
    u0, _ = A.bindings(final_root)
    c = cell.lower()
    waves = [(f'A37V_{cell}_U0', 'episode', _retag(N.episode_specs('U0', panel, None, Path(final_root)/N.PRERUN_PLANT,
                                                                    'mappo_frozen', None, u0), f'a37v_{c}_u0'))]
    entry = _bank_entry(final_root, f'A37_CANDS_{cell}.json', check_files=check_files) or dict(path='', file_sha256='')
    for cand in candidates(cell):
        waves.append((f'A37V_{cell}_{cand["family"]}', 'episode',
                      _a37_specs('A37_PLAN', panel, final_root, entry, family=cand['family'],
                                 tag=f'a37v_{c}_{cand["key"][:8]}')))
    return waves


def admission(final_root, cell):
    p = Path(final_root)/f'ADMISSION_{cell}.json'
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def eval_specs(arm, cell, panel, final_root, *, check_files=True):
    c = cell.lower()
    if arm in ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD'):
        return _retag(T._arm_specs(arm, panel, final_root, check_files=check_files), f'a37e_{c}_{arm.lower()}')
    if arm == 'CELL_BANK':
        adm = admission(final_root, cell)
        if adm is None:
            raise N.Blocked(f'{cell}: no ADMISSION file (run verify + admit first)')
        if not adm['admitted']:
            return []                      # CELL_BANK := U0 by definition; nothing runs
        entry = _bank_entry(final_root, f'A37_CELL_BANK_{cell}.json')
        if entry['file_sha256'] != adm['cell_bank']['file_sha256']:
            raise N.Blocked(f'{cell}: CELL_BANK bank differs from the admission receipt')
        return _a37_specs('CELL_BANK', panel, final_root, entry, tag=f'a37e_{c}_cell_bank')
    raise N.Blocked(f'no A37 eval arm {arm!r}')


def eval_wave(arm, cell):
    return f'A37E_{cell}_{arm}'


def ident_specs(final_root):
    u0, _ = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    out = []
    for tag, name, pin in IDENT_PANELS:
        panel = N.verify(N.read(HERE/name), 'panel_sha256', name)
        if panel['panel_sha256'] != pin:
            raise N.Blocked(f'identity panel {name}: sha differs from the registered panel')
        specs = N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)[:N.IDENTITY_WORLDS]
        out += [dict(s, job_id=f'a37i_{tag}_u0_{i:04}') for i, s in enumerate(specs)]
    return out


def native_specs(panel, final_root, arm):
    u0, _ = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    specs = N.episode_specs(arm, panel, None, plant, 'mappo_frozen', None, u0 if arm == 'U0' else None)
    return _retag(specs, f'a37n_{arm.lower()}')


def native_wave(arm):
    return f'A37N_D2800_p20_{arm}'


# ------------------------------------------------------------------------------------------------ checks / runtime

def check_a37(record, spec):
    if spec['arm'] not in A37_ARMS:
        return A.check_a32(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = A._ORIG_CHECK(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a37_sha256_loaded') != spec['checkpoint']['a37_sha256']:
        raise N.Blocked(f'{what}: a37_cellbank actually loaded differs from its registration')
    if not isinstance(record.get('a37'), dict) or record['a37'].get('a37_arm') != spec['arm']:
        raise N.Blocked(f'{what}: a37 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a37_sha256_loaded=record['a37_sha256_loaded'], a37=deepcopy(record['a37']))


N.CHECKS['episode'] = check_a37


def charged(final_root, prefix=None):
    rows = N.fe.ledger_rows(final_root)
    return sum(r['sim_s'] for r in rows if prefix is None or str(r['id']).startswith(prefix))


class A37Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase celltailor_a37 (cap 1,900,000), verification
    (a37v_*) <= 600,000."""
    phase = A37_PHASE

    def _caps(self, amount, verify):
        if charged(self.final_root)+amount > A37_CAP_SIM_S:
            raise N.Blocked(f'A37: {A37_PHASE} cap {A37_CAP_SIM_S} would be exceeded')
        if verify and charged(self.final_root, 'a37v_')+amount > VERIFY_CAP_SIM_S:
            raise N.Blocked(f'A37: verification cap {VERIFY_CAP_SIM_S} would be exceeded')

    def check_capacity(self, amount):
        self._caps(amount, getattr(self, 'verify', False))

    def reserve(self, cfg, jobs):
        verify = any(str(j['id']).startswith('a37v_') for j in jobs)
        if verify and not all(str(j['id']).startswith('a37v_') for j in jobs):
            raise N.Blocked('A37: a dispatch never mixes verification and other jobs')
        self._caps(sum(j['sim_s'] for j in jobs), verify)
        N.own_reserve(self.final_root, jobs, A37_PHASE, cap=A37_CAP_SIM_S, calls=self.calls, allocation=A37_CAP_SIM_S)


def execute(final_root, waves, *, workers, chunk, log, panel=None, verify=False, runtime=None):
    """a35_transfer.execute with the A37 runtime: pending waves in chunks of ``chunk`` episodes."""
    final_root = Path(final_root)
    runtime = runtime or A37Runtime(N.Runtime(N.RUNNER), final_root)
    runtime.verify = verify
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the A37 arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A37_ARMS', None) != A37_ARMS:
            raise N.Blocked('runner lacks the amendment-37 arms')
        if panel is not None:
            N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, check_a37)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A37).exists():
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


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, refs_path=None):
    refs = N.read(refs_path or CANDS_DIR/'IDENTITY_REFS.json')
    specs = ident_specs(final_root)
    got, _, cov = N.published(final_root, IDENT_WAVE, specs, check_a37)
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
        ok = x['abs_diff'] <= 1e-6 and x['safety_equal']
        per[x['panel']] = per.get(x['panel'], True) and ok
    return N.seal(dict(schema='coopmwm.a37.identity.v1', rule='|J diff| <= 1e-6 veh-s and identical safety, U0',
                       verdict='PASS' if all(per.values()) else 'FAILED', per_panel={k: 'PASS' if v else 'FAILED'
                                                                                 for k, v in per.items()},
                       worlds=rows, consequence=('PASS: the plant is identical; the a33 U0 records may pair with 37b '
                                                 'NATIVE. FAILED: every arm is (already) rerun on this host and '
                                                 'analysed only among themselves; 37b reruns U0 (budget permitting)')),
                  'receipt_sha256')


def make_identity_refs(out, *, a32_root, a33_root, a35_offload):
    """Laptop, zero physics: reference J / safety of the 8 identity worlds from sealed local copies."""
    import gzip
    import hashlib
    from tools.coopmwm.replay_bank import ttc_events
    worlds = {}

    def wave_records(root, prefix):
        recs = []
        for p in sorted(Path(root).glob(f'WAVE_{prefix}__*.json')):
            recs += N.verify(N.read(p), 'receipt_sha256', p.name)['records']
        return {r['world_id']: r for r in recs}
    p20c = wave_records(a32_root, 'A32_IDENT_U0')
    a33 = wave_records(a33_root, 'A33_U0')
    shas = {}
    for line in (Path(a35_offload)/'OFFLOAD.sha256').read_text().splitlines():
        h, f = line.split(None, 1)
        shas[f.strip()] = h
    for tag, name, _ in IDENT_PANELS:
        panel = N.verify(N.read(HERE/name), 'panel_sha256', name)
        for i, w in enumerate(panel['worlds'][:N.IDENTITY_WORLDS]):
            wid = w['world_id']
            if tag == 'p20c':
                r = p20c[wid]
                worlds[f'{tag}:{wid}'] = dict(J_veh_s=r['J_veh_s'], safety=r['safety'],
                                             source='a32_ablation/root WAVE_A32_IDENT_U0 (== sealed a28 U0, a32 gate)')
            elif tag == 'a33':
                r = a33[wid]
                worlds[f'{tag}:{wid}'] = dict(J_veh_s=r['J_veh_s'], safety=r['safety'],
                                             source='a33_combined/root WAVE_A33_U0 (gpuB)')
            else:
                cell = panel['cell']['cell_key'].lower()
                d = Path(a35_offload)/'evaluation'/f'a35_{cell}_nmfinal_u0_{i:04}'
                for x in ('RESULT.json.gz', 'observable.jsonl.gz'):
                    if hashlib.sha256((d/x).read_bytes()).hexdigest() != shas.get(f'{d.name}/{x}'):
                        raise N.Blocked(f'{d.name}/{x}: sha differs from the a35 offload receipt')
                res = json.load(gzip.open(d/'RESULT.json.gz'))
                if res['world'] != w['world']:
                    raise N.Blocked(f'{d.name}: not world {wid}')
                t = res['totals']
                worlds[f'{tag}:{wid}'] = dict(
                    J_veh_s=float(t['team_delay_vs']),
                    safety=dict(collisions=int(t['collisions']), hard_brakes=int(t['hard_brake']),
                                teleports=int(t['teleports']), ttc_events=int(ttc_events(d/'observable.jsonl.gz', 300, 1200))),
                    source=f'a35_transfer offload {d.name} (gpuB; sha-verified)')
    value = N.seal(dict(schema='coopmwm.a37.identity_refs.v1', worlds=worlds,
                        note='reference U0 per-world J and safety only; no other outcome read'), 'receipt_sha256')
    Path(out).write_text(json.dumps(value, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return value


# ------------------------------------------------------------------------------------------------ admission / analysis

def _load(final_root, wave, specs):
    recs, src, cov = N.published(final_root, wave, specs, check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {final_root}')
    return {r['world_id']: r for r in recs}, src


def _reports(root, have, worlds):
    reports = {a: {w: A.report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not r['ok'] for a in reports for r in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    return reports


def admit(final_root, cell, panel):
    """UNC rule per candidate on the verification block -> ADMISSION_<cell>.json (sealed) + the CELL_BANK bank."""
    from tools.coopmwm import a37_cellbank as C
    final_root = Path(final_root)
    if admission(final_root, cell) is not None:
        raise N.Blocked(f'ADMISSION_{cell}.json exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    waves = verify_waves(cell, panel, final_root)
    have, sources = {}, {}
    for (name, _, specs), arm in zip(waves, ['U0']+[c['family'] for c in candidates(cell)]):
        have[arm], sources[arm] = _load(final_root, name, specs)
    reports = _reports(final_root, have, worlds)
    verdicts, rows = {}, []
    for cand in candidates(cell):
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, 'U0', bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have['U0'][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        v.update(family=fam, wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'], key=cand['key'],
                 cvar90_vs_u0=cmp_['tail_cvar90'], groups=cmp_['groups'], total_minus_u0_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        verdicts[fam] = v
        rows.append(v)
    plans = C.admitted_plans(candidates(cell), verdicts)
    cell_bank = None
    if plans:
        p = write_once(bank_path(final_root, f'A37_CELL_BANK_{cell}.json'), bank_text(C.bank_doc(plans, cell)))
        cell_bank = dict(path=f'{HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p))
    value = N.seal(dict(schema='coopmwm.a37.admission.v1', amendment='37', cell=cell, rule=C.UNC_RULE,
                        multiplicity=f'{len(rows)} picks each tested at 95% (disclosed, as the r6 UNC multiple-pick rule)',
                        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
                        admitted=[p['family'] for p in plans], priority=[(p['family'], p['lb95_veh_s']) for p in plans],
                        cell_bank=cell_bank, cell_bank_is_u0=not plans,
                        u0_safety_totals={k: int(sum(have['U0'][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()}),
                   'receipt_sha256')
    N.publish(final_root/f'ADMISSION_{cell}.json', value)
    return value


def analyse(final_root, cell, panel):
    """Registered amendment-37 contrasts for one cell (never pooled): paired percentile bootstrap (10,000, seed
    20261104), diff = treatment - reference (negative = less delay)."""
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    adm = admission(final_root, cell)
    if adm is None:
        raise N.Blocked(f'{cell}: ADMISSION missing')
    have, sources = {}, {}
    for arm in EVAL_ARMS:
        specs = eval_specs(arm, cell, panel, final_root, check_files=False)
        if arm == 'CELL_BANK' and not specs:
            continue
        have[arm], sources[arm] = _load(final_root, eval_wave(arm, cell), specs)
    reports = _reports(final_root, have, worlds)
    if 'CELL_BANK' not in have:            # registered: no plan admitted -> CELL_BANK(cell) := U0
        have['CELL_BANK'], reports['CELL_BANK'] = have['U0'], reports['U0']
    contrasts = []
    for kind, pairs in (('registered', CONTRASTS), ('context', CONTEXT)):
        for arm, ref in pairs:
            row = dict(treatment=arm, reference=ref, kind=kind,
                       convention='diff = treatment - reference; negative = treatment has less delay')
            row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
            for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
                row[f'{k}_minus_ref'] = N.fe.paired_stats(
                    np.array([have[arm][w]['safety'][k] for w in worlds], float),
                    np.array([have[ref][w]['safety'][k] for w in worlds], float), bootstrap, pct=False)
            contrasts.append(row)
    prim = contrasts[0]
    arms = {}
    for a in EVAL_ARMS:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY},
                       is_u0=(a == 'CELL_BANK' and adm['cell_bank_is_u0']))
        rec = [have[a][w].get('a37') or have[a][w].get('a32') or have[a][w].get('learned_trigger') or {}
               for w in worlds] if not arms[a]['is_u0'] else []
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam)
    verdict = dict(rule='condition tailoring supported at the cell iff CELL_BANK - LEARNED_TRIGGER total delay CI95 upper < 0',
                   ci95=prim['total_minus_ref']['ci95'], supported=bool(prim['total_minus_ref']['ci95'][1] < 0),
                   cell_bank_is_u0=adm['cell_bank_is_u0'],
                   note=('no plan admitted: CELL_BANK := U0 ("search recommends no hold"), reported as the outcome'
                         if adm['cell_bank_is_u0'] else f'CELL_BANK = {adm["admitted"]}'))
    return N.seal(dict(schema='coopmwm.a37.analysis.v1', amendment='37', authority=AUTHORITY, cell=cell,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       primary=verdict, contrasts=contrasts, arms=arms, admission=dict(
                           admitted=adm['admitted'], receipt_sha256=adm['receipt_sha256'],
                           cell_bank=adm['cell_bank']),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       label='amendment 37; cells never pooled; one shot'), 'analysis_sha256')


def analyse_native(final_root, panel, a33_root=None):
    """37b context (no pass/fail): NATIVE - U0 on the a33 fresh panel. U0 = this host's U0 rerun if it exists, else
    the a33 (gpuB) U0 records, only when the identity gate passed on the a33 worlds."""
    import numpy as np
    final_root = Path(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    have, roots = {}, {}
    nat, _ = _load(final_root, native_wave('NATIVE'), native_specs(panel, final_root, 'NATIVE'))
    worlds = [w for w in worlds if w in nat]
    have['NATIVE'], roots['NATIVE'] = nat, final_root
    u0_specs = native_specs(panel, final_root, 'U0')
    got, _, cov = N.published(final_root, native_wave('U0'), u0_specs, check_a37)
    if cov:
        have['U0'], roots['U0'], u0_source = {r['world_id']: r for r in got}, final_root, 'this host (identity failed)'
        worlds = [w for w in worlds if w in have['U0']]
    else:
        gate = N.read(final_root/'IDENTITY_GATE.json')
        if gate['per_panel'].get('a33') != 'PASS':
            raise N.Blocked('37b: a33 identity worlds failed and no host U0 rerun exists')
        recs = []
        for p in sorted(Path(a33_root).glob('WAVE_A33_U0__*.json')):
            recs += N.verify(N.read(p), 'receipt_sha256', p.name)['records']
        have['U0'], roots['U0'], u0_source = {r['world_id']: r for r in recs}, Path(a33_root), 'a33 gpuB records'
    reports = {a: {w: A.report_of(roots[a], have[a][w]) for w in worlds} for a in have}
    row = N._bank_compare(have, reports, worlds, 'NATIVE', 'U0', bootstrap)
    for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
        row[f'{k}_minus_ref'] = N.fe.paired_stats(np.array([have['NATIVE'][w]['safety'][k] for w in worlds], float),
                                                  np.array([have['U0'][w]['safety'][k] for w in worlds], float),
                                                  bootstrap, pct=False)
    return N.seal(dict(schema='coopmwm.a37b.native.v1', amendment='37b', label='context for MAPPO strength; no pass/fail',
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), u0_source=u0_source,
                       contrast=dict(treatment='NATIVE', reference='U0', convention='diff = NATIVE - U0', **row),
                       mean_J={a: float(np.mean([have[a][w]['J_veh_s'] for w in worlds])) for a in have}),
                  'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def stage_waves(command, setting, panel, root, *, check_files=True):
    cell = cell_of(setting)
    if command == 'verify':
        if setting != VERIFY_SETTING.get(cell):
            raise N.Blocked('verify runs on an a37v_<cell> setting')
        return verify_waves(cell, panel, root, check_files=check_files)
    if command == 'eval':
        if setting != EVAL_SETTING.get(cell):
            raise N.Blocked('eval runs on an a35_<cell> setting')
        return [(eval_wave(a, cell), 'episode', s) for a in EVAL_ARMS
                if (s := eval_specs(a, cell, panel, root, check_files=check_files))]
    if command == 'native':
        if setting != NATIVE_SETTING:
            raise N.Blocked('native runs on the a33 setting')
        return [(native_wave('NATIVE'), 'episode', native_specs(panel, root, 'NATIVE'))]
    raise N.Blocked(command)


def native_plan(root, panel):
    """37b budget rule (registered fallback): NATIVE always (192 worlds); U0 too only if the a33 identity worlds failed;
    when both full arms do not fit in the remaining cap, both run on the first n worlds that fit (n >= 24, else
    37b is skipped and recorded)."""
    gate = N.read(Path(root)/'IDENTITY_GATE.json')
    need_u0 = gate['per_panel'].get('a33') != 'PASS'
    left = A37_CAP_SIM_S - charged(root)
    n_arms = 2 if need_u0 else 1
    n = min(len(panel['worlds']), left//(n_arms*N.HORIZON_S))
    return dict(need_u0=need_u0, n_worlds=n if n >= 24 else 0, left_sim_s=left)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('prepare', 'dry', 'identity', 'identity-compare', 'verify', 'admit', 'eval',
                                       'native', 'reports', 'analyse', 'analyse-native', 'status',
                                       'make-identity-refs'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--setting', default=VERIFY_SETTING['D3200_p20'], choices=SETTINGS)
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=46)
    p.add_argument('--out')
    p.add_argument('--a33-root')
    p.add_argument('--a32-root')
    p.add_argument('--a35-offload')
    a = p.parse_args(argv)
    root = Path(a.final_root)
    if a.command == 'make-identity-refs':
        v = make_identity_refs(a.out, a32_root=a.a32_root, a33_root=a.a33_root, a35_offload=a.a35_offload)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], worlds=len(v['worlds']))))
        return
    panel = setup(a.setting)
    cell = cell_of(a.setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=a.setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/'A37_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root), indent=1))
    elif a.command == 'dry':
        from tools.coopmwm import a37_cellbank as C
        out = dict(setting=a.setting, panel_sha256=panel['panel_sha256'], cell=panel['cell']['cell_key'],
                   a37_sha256=C.a37_digest(), ident=[s['job_id'] for s in ident_specs(root)])
        cmds = ('verify',) if a.setting.startswith('a37v_') else ('native',) if a.setting == NATIVE_SETTING else ('eval',)
        for c in cmds:
            try:
                ws = stage_waves(c, a.setting, panel, root)
            except N.Blocked as e:            # eval before admission: the three frozen arms only
                ws = [(eval_wave(x, cell), 'episode', eval_specs(x, cell, panel, root)) for x in EVAL_ARMS[:3]]
                out['note'] = str(e)
            out[c] = {n: dict(n=len(s), first=s[0]['job_id'], seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                              cell=s[0]['world']['cell_key'], arm=s[0]['arm']) for n, _, s in ws}
            out[c+'_sim_s'] = sum(len(s)*N.HORIZON_S for _, _, s in ws)
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'verify', 'eval', 'native'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves = [(IDENT_WAVE, 'episode', ident_specs(root))]
            elif a.command == 'native':
                plan = native_plan(root, panel)
                log(dict(native_plan=plan))
                if not plan['n_worlds']:
                    log(dict(done='native', status='SKIPPED_BUDGET', **plan))
                    return
                waves = [(native_wave('NATIVE'), 'episode', native_specs(panel, root, 'NATIVE')[:plan['n_worlds']])]
                if plan['need_u0']:
                    waves.append((native_wave('U0'), 'episode', native_specs(panel, root, 'U0')[:plan['n_worlds']]))
            else:
                waves = stage_waves(a.command, a.setting, panel, root)
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=charged(root)))
            res = execute_with_retry(root, waves, workers=a.workers, chunk=a.chunk, log=log,
                                     panel=None if a.command == 'identity' else panel, verify=a.command == 'verify')
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root)
        N.publish(root/'IDENTITY_GATE.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], per_panel=v['per_panel'])))
    elif a.command == 'admit':
        if not a.setting.startswith('a37v_'):
            p.error('admit runs on an a37v_<cell> setting')
        v = admit(root, cell, panel)
        print(json.dumps(dict(cell=cell, admitted=v['admitted'], receipt_sha256=v['receipt_sha256'])))
    elif a.command == 'analyse':
        v = analyse(root, cell, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A37_RESULTS_{cell}.json', v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], cell=cell, supported=v['primary']['supported'])))
    elif a.command == 'analyse-native':
        v = analyse_native(root, panel, a.a33_root)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/'A37B_NATIVE.json', v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], n_worlds=v['n_worlds'])))
    elif a.command == 'status':
        out = dict(charged=charged(root), verify_charged=charged(root, 'a37v_'), cap=A37_CAP_SIM_S)
        for c in ('verify', 'eval', 'native'):
            try:
                for n, _, s in stage_waves(c, a.setting, panel, root, check_files=False):
                    out[n] = f'{N.published(root, n, s, check_a37)[2]}/{len(s)}'
            except N.Blocked:
                pass
        print(json.dumps(out, indent=1))
    else:   # reports
        n = 0
        names = [IDENT_WAVE] if a.setting == VERIFY_SETTING['D3200_p20'] else []
        waves = []
        for c in ('verify', 'eval', 'native'):
            try:
                waves += stage_waves(c, a.setting, panel, root, check_files=False)
            except N.Blocked:
                pass
        if names:
            waves.append((IDENT_WAVE, 'episode', ident_specs(root)))
        for name, _, specs in waves:
            recs, _, _ = N.published(root, name, specs, check_a37)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
