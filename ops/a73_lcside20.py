"""AMENDMENT 73 (S-LCSIDE20; MAIN 2026-11-13): the A69 lane-change downstream at 20 % CAVs (cell D2800_p20).

Same design as ops/a69_lcside.py; only the cell changes. At 20 % the base controller is the frozen speed-only MAPPO
U0 (d9042cc5) itself, so no executor field is rebound: the A44 mechanics and the A69 identity collapse into ONE
mechanics stage, and every reference arm is run here on the new panels (no earlier root holds them).

Uses the A44 lane-change executor variant (tools/coopmwm/a44_lcvariant.py, runner arms A44_PLAN / A44_BANK).

Stages (own root, one ledger phase ``lcside20_a73``, cap 686,400 sim-s = 24,000 + 201,600 + 460,800):
  select    zero SUMO: top K = 6 plans of the search `final` list that contain >= 1 lane-change role (WM score desc,
            ties by key; skip comparator / banned / variant-executor refusal / feasibility 0)      [= A69 rule]
  panel     zero SUMO: build one panel (m: 992408-992415, v: 992524-992547, c: 996400-996495), write-once
  pin       zero SUMO: write ops/a73/PINS_A73.json from the built panels + the selection (MAIN, after recording them)
  mech      worlds 992408 / 992409: the hold plan a44_fix_hold by the A37 executor path (A37_PLAN) and by the variant
            (A44_PLAN) must give exactly equal records (decompressed vehicle logs); then the lane-change fixture
            a44_fix_lc on 992408-992415 -> lc-report (receipts + collisions / teleports only)
  verify    24 worlds 992524-992547: U0 and each candidate as one unconditional plan (A44_PLAN)
  admit     (plan - U0): delay CI95 upper < 0 AND five-group 10 s rule; LC_BANK = plans passing both, priority
            verified lb95 desc                                                                     [= A69 rule]
  confirm   96 worlds 996400-996495: U0, DUO_HOLD (hand-written two-CAV hold duo12_v20), LEARNED_TRIGGER (the
            existing 20 % searched hold bank fd385a2b + classifier 8e0ddd3e), LC_BANK_20 (A44_BANK)
  analyse   LC_BANK_20 - {U0, DUO_HOLD, LEARNED_TRIGGER}; paired percentile bootstrap 10,000, seed 20261104;
            diff = treatment - reference; never pooled
No automatic replay. Nothing is written outside the A73 root (and ops/a73 for the zero-SUMO files).

Commands (python ops/a73_lcside20.py <cmd> --final-root R):
  select --search F ; panel --stage m|v|c [--map-spec F] ; pin ; prepare-mech ; prepare ; dry --stage m|v|c ;
  identity ; identity-compare ; lcfix ; lc-report ; verify ; admit ; confirm ; analyse ; status --stage m|v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '73'
CELL = 'D2800_p20'
SOURCE_SETTING = 'p20c'                    # the registered D2800_p20 plant / bindings (the bank-root mirror)
PHASE = 'lcside20_a73'
IDENT_N, LC_N, VERIFY_N, CONFIRM_N, K = 2, 8, 24, 96, 6
HORIZON = 1_200
MECH_CAP = 24_000                          # = the A44 mechanics cap (planned 4 + 8 episodes = 14,400)
VERIFY_CAP = (K+1)*VERIFY_N*HORIZON        # U0 + <= 6 candidates
CONFIRM_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'LC_BANK_20')
CONFIRM_CAP = len(CONFIRM_ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+VERIFY_CAP+CONFIRM_CAP
#: stage -> (first seed, last seed, runner table attribute, index in that table or None for a single range)
SEEDS = dict(m=(992408, 992415, 'A44_MECH_RANGE', None), v=(992524, 992547, 'A68_RANGES', 0),
             c=(996400, 996495, 'A73_RANGES', 0))
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
SETTING = dict(m='a73m_D2800_p20', v='a73v_D2800_p20', c='a73c_D2800_p20')
A44_SHA8, A37_SHA8 = '53b64cbe', '0a902437'
HOST_BASE = os.environ.get('A73_HOST_BASE', '/root/a73')
HOST_ROOT = f'{HOST_BASE}/root'
BANK_DIR = 'a73_banks'
CANDS_BANK = f'A73_CANDS_{CELL}.json'
LC_BANK = f'A73_LC_BANK_{CELL}.json'
A73_DIR = HERE/'a73'
PANEL_FILE = {k: A73_DIR/f'nm_eval_panel_wz3to2_{v}.json' for k, v in SETTING.items()}
CANDS_FILE = A73_DIR/f'CANDIDATES_LC_{CELL}.json'
PINS_FILE = A73_DIR/'PINS_A73.json'
ADMISSION_FILE = f'A73_ADMISSION_{CELL}.json'
IDENTITY_FILE = 'IDENTITY_A73.json'
LC_MECH_FILE = 'A73_LC_MECH.json'
RESULTS_FILE = f'A73_RESULTS_{CELL}.json'
LOG_FILE = 'A73_LOG.jsonl'
REF = 'U0'
ARM = 'LC_BANK_20'
REF_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER')
CONTEXT = (('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'U0'), ('LEARNED_TRIGGER', 'DUO_HOLD'))
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
AUTHORITY = ('MAIN AMENDMENT 73 (S-LCSIDE20, 2026-11-13): the A69 lane-change downstream at D2800_p20; variant executor '
             '53b64cbe on the frozen MAPPO U0 (d9042cc5); phase lcside20_a73 cap 686,400')
DISCLOSURE = ('same rules as A69, cell D2800_p20; 6 picks each tested at 95 %; the empty-plan half of the identity '
              'cannot be built on the variant (it refuses a bank without a plan) and is not run; the frozen reference '
              'controllers (DUO_HOLD duo12_v20, LEARNED_TRIGGER bank fd385a2b) run as registered, no retuning')
ADMIT_RULE = ('(plan - U0) total delay CI95 upper < 0 on the 24 verification worlds (a37_cellbank.unc_verdict) AND all '
              'five groups present with mean increase <= 10 s relative to U0; reference = the U0 records of the same '
              'worlds (this root); LC_BANK priority lb95(gain) desc then strategy key')


def _mods():
    import nm_finaleval as N
    import a32_ablation as A
    return N, A


def _clone(name, path, overrides):
    """A private module object executed from ``path`` (same code), then the amendment names replaced
    (a50_fullp60._clone)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    for k, v in overrides.items():
        if not hasattr(m, k):
            raise AttributeError(f'{path.name} has no {k} (the A73 override list is out of date)')
        setattr(m, k, v)
    return m


def mech():
    """The A44 mechanics driver code (same file) with the A73 names: own phase / cap / root / bank dir / cell."""
    N = _mods()[0]
    m = _clone('a73_mech_core', HERE/'a44_lcmech.py', dict(
        A44_PHASE=PHASE, A44_CAP_SIM_S=CAP_SIM_S, STOP_A44='STOP_A73', HOST_BASE=HOST_BASE, HOST_ROOT=HOST_ROOT,
        BANK_DIR=BANK_DIR, CELL=CELL, HOLD_BANK='A73_HOLD.json', LC_BANK='A73_LCFIX.json', AUTHORITY=AUTHORITY,
        waves=_mech_waves))
    m.A44Runtime.phase = PHASE
    N.CHECKS['episode'] = m.check_a44
    return m


def runtime(root):
    N = _mods()[0]
    M = mech()

    class A73Runtime(M.A44Runtime):
        phase = PHASE

        def reserve(self, cfg, jobs):
            if not all(str(j['id']).startswith('a73') for j in jobs):
                raise N.Blocked('A73: only a73* jobs in this root')
            self._caps(sum(j['sim_s'] for j in jobs))
            N.own_reserve(self.final_root, jobs, PHASE, cap=CAP_SIM_S, calls=self.calls, allocation=CAP_SIM_S)
    return A73Runtime(N.Runtime(N.RUNNER), Path(root))


def digests():
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a44_lcvariant as V
    a37, a44 = C.a37_digest(), V.a44_digest()
    if not a37.startswith(A37_SHA8) or not a44.startswith(A44_SHA8):
        raise N.Blocked(f'executor digests differ from the registered ones: a37 {a37[:8]} a44 {a44[:8]}')
    return dict(a37_sha256=a37, a44_sha256=a44)


# ------------------------------------------------------------------------------------------------ settings / panels

def block(which):
    """(lo, hi, r0, r1): the registered seeds of the stage and the runner block that carries them."""
    N = _mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    lo, hi, attr, idx = SEEDS[which]
    table = getattr(R, attr, None)
    if table is None:
        raise N.Blocked(f'the runner has no {attr} (ops/a73/host/patch_tables.py)')
    r0, r1 = table if idx is None else table[idx]
    if not r0 <= lo <= hi <= r1:
        raise N.Blocked(f'{lo}-{hi} outside the runner block {r0}-{r1}')
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise N.Blocked('forbidden seed range')
    return lo, hi, r0, r1


def pins():
    return json.loads(PINS_FILE.read_text(encoding='utf-8')) if PINS_FILE.is_file() else {}


def register_setting(which):
    """Process-local nm_finaleval setting (nm_finaleval itself unchanged): the p20c cell / plant / bindings with
    the A73 block of the stage (a44_lcmech.register_setting pattern)."""
    N = _mods()[0]
    lo, hi, r0, r1 = block(which)
    what = dict(m='executor-variant mechanics (never evaluation, outcomes never read)',
                v='lane-change plan verification / admission', c='confirmation')[which]
    name = SETTING[which]
    N.SETTINGS[name] = dict(
        N.SETTINGS[SOURCE_SETTING],
        N_WORLDS=hi-lo+1, EVAL_RANGE=(lo, r1), EVAL_EXTRA_RANGE=(lo, r1), SEEDS_RESERVED=(hi+1, r1),
        PANEL_SIZE_REASON=f'MAIN amendment 73: {what} = {hi-lo+1} fresh worlds {lo}-{hi} (D2800_p20)',
        CELL_AUTHORITY='MAIN amendment 73 (S-LCSIDE20, 2026-11-13): lane-change plans at D2800_p20',
        PANEL_AUTHORITY=(f'A73 amendment 73: audited integers {lo}-{hi} (runner block {r0}-{r1}) for the {what} ONLY; '
                         'executed through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(checked=('amendment-73 audit: ops/a73/SEED_AUDIT.json',),
                            disclosed=f'; amendment 73: fresh integers ({what}), registered at runtime by the runner'),
        PANEL_FILE=PANEL_FILE[which], PANEL_3C_FILE=None)
    if name not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
    N.SETTINGS_PINS[name] = dict(PANEL_SHA256=pins().get(f'panel_{which}'), PANEL_3C_SHA256=None)
    return name


def setup(which, *, need_panel=True):
    N = _mods()[0]
    mech()
    digests()
    N.apply_setting(register_setting(which))
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    if not need_panel:
        return None
    lo, hi, _, _ = block(which)
    panel = N.load_panel(PANEL_FILE[which], pinned=None)
    got = [w['world']['seed'] for w in panel['worlds']]
    if got != list(range(lo, hi+1)) or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'panel {PANEL_FILE[which].name} is not {CELL} {lo}-{hi}')
    if dict(panel['contract']['bootstrap']) != BOOTSTRAP:
        raise N.Blocked('panel bootstrap is not the registered one')
    return panel


def require_pins(panel, which):
    """Physics only after MAIN recorded the panels and the selection (PINS_A73.json)."""
    N = _mods()[0]
    p = pins()
    if p.get(f'panel_{which}') != panel['panel_sha256']:
        raise N.Blocked(f'A73: panel {which} not pinned (ops/a73/{PINS_FILE.name}) - physics refused')


def make_pins():
    """Write-once pin file from the built panels and the sealed selection (zero SUMO; MAIN runs it after recording)."""
    N = _mods()[0]
    import a37_celltailor as C37
    doc = dict(amendment=AMENDMENT, candidates_doc_sha256=N.read(CANDS_FILE)['doc_sha256'])
    for w in 'mvc':
        doc[f'panel_{w}'] = N.read(PANEL_FILE[w])['panel_sha256']
    C37.write_once(PINS_FILE, json.dumps(doc, indent=1, sort_keys=True)+'\n')
    return doc


def published(root, wave, specs):
    N = _mods()[0]
    recs, src, cov = N.published(root, wave, specs, mech().check_a44)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {root}')
    return {r['world_id']: r for r in recs}, src


def log_content_sha(root, record):
    """sha256 of the DECOMPRESSED vehicle log of one record (the record's own sha covers the gzip container)."""
    N = _mods()[0]
    p = Path(root)/'evaluation'/record.get('work_dir', record['job_id'])/N.VEHICLE_LOG
    with gzip.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _retag(specs, tag):
    return [dict(s, job_id=f'{tag}_{i:04}') for i, s in enumerate(specs)]


def ref_specs(arm, panel, root, tag):
    """The frozen 20 % arms exactly as registered at their home cell (no bank_cell_key: world cell = bank cell)."""
    N, A = _mods()
    root = Path(root)
    u0, oracle = A.bindings(root)
    plant = root/N.PRERUN_PLANT
    if arm == 'U0':
        specs = N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)
    elif arm == 'LEARNED_TRIGGER':
        specs = N.episode_specs('LEARNED_TRIGGER', panel, None, plant, 'mappo_frozen', oracle, u0)
    elif arm == 'DUO_HOLD':
        specs = A.a32_specs('DUO_HOLD', panel, root)
    else:
        raise N.Blocked(f'no A73 reference arm {arm!r}')
    if any(s['checkpoint'].get('executor_sha256') != N.FROZEN_MAPPO_SHA for s in specs):
        raise N.Blocked(f'{arm}: executor is not the frozen MAPPO U0')
    return _retag(specs, tag)


# ------------------------------------------------------------------------------------------------ candidates

def select(search_path, out):
    """Zero SUMO, WM ranking only: the registered top-6 lane-change selection, sealed; never overwrites."""
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
    search_path, out = Path(search_path), Path(out)
    raw = search_path.read_bytes()
    search = json.loads(raw.decode('utf-8'))
    if search.get('cell') != CELL:
        raise SystemExit(f'search cell {search.get("cell")!r} is not {CELL}')
    rows = sorted(search['final'], key=lambda x: (-float(x['score']), x['key']))
    ranking, skipped, cands = [], [], []
    for i, x in enumerate(rows):
        s = x['strategy']
        if B.O.strategy_key(s) != x['key']:
            raise SystemExit(f'{s["name"]}: strategy_key differs from the search key')
        why = ('comparator' if x['comparator'] else 'banned' if x['banned'] else
               'no_lc_role' if not V.is_lc_plan(s) else V.executor_refusal(s))
        if why is None and not float(x['feasible']) > 0:
            why = B.NEVER_ACTIVATED
        entry = dict(wm_rank=i+1, name=s['name'], key=x['key'], score=x['score'], skip=why)
        ranking.append(entry)
        if why is not None:
            skipped.append(entry)
            continue
        if len(cands) < K:
            cands.append(dict(entry, family=s['name'], strategy=s, strategy_sha256=C.spec_sha(s),
                              n_roles=len(s['roles']),
                              lc_roles=[dict(index=j, lane=r['lane'], lc_target=r['lc_target'])
                                        for j, r in enumerate(s['roles']) if r.get('lc_target') is not None],
                              wm=dict({f: x.get(f) for f in ('feasible', 'gain_vs_base', 'gain_vs_base_lcb',
                                                               'gain_vs_inc', 'worst_group_inc', 'cvar90_inc',
                                                               'bounded', 'fair_cal')}, n_states=x.get('n_states')),
                              selected_rank=len(cands)+1))
    doc = B.seal(dict(
        schema='coopmwm.a73.candidates.v1', amendment=AMENDMENT, cell=CELL, k=K,
        search=dict(file_sha256=hashlib.sha256(raw).hexdigest(), schema=search.get('schema'), cell=search.get('cell'),
                    config=search.get('config'), evaluated=search.get('evaluated'), n_states=search.get('n_states'),
                    n_final=len(search['final'])),
        rule='search `final` list by WM score desc (ties: strategy key); skip comparator / banned rows, plans '
             'without a lane-change role, plans the variant executor refuses and rows with feasibility 0; top K = 6',
        STOP=not cands, skip_counts=dict(sorted(Counter(e['skip'] for e in skipped).items())),
        n_skipped=len(skipped), n_eligible=len(ranking)-len(skipped), n_selected=len(cands),
        role_counts=dict(sorted(Counter(str(c['n_roles']) for c in cands).items())), candidates=cands,
        skipped=skipped, ranking=ranking))
    if out.exists():
        raise SystemExit(f'{out}: exists (pre-outcome file; never overwrite)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return doc


def candidates(path=None, *, need_pin=True):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
    path = Path(path or CANDS_FILE)
    if not path.is_file():
        raise N.Blocked(f'{path}: missing (run select)')
    doc = N.read(path)
    pin = pins().get('candidates_doc_sha256')
    if B.doc_sha(doc) != doc.get('doc_sha256') or (need_pin and doc['doc_sha256'] != pin):
        raise N.Blocked(f'{path.name}: not the registered selection (pin {pin})')
    if doc['cell'] != CELL or doc['k'] != K or doc.get('amendment') != AMENDMENT or doc['STOP']:
        raise N.Blocked(f'{path.name}: not the registered cell / K, or no candidate')
    fams = [c['family'] for c in doc['candidates']]
    if len(set(fams)) != len(fams) or not 1 <= len(fams) <= K:
        raise N.Blocked(f'{path.name}: {len(fams)} candidates / duplicate family names')
    for c in doc['candidates']:
        s = c['strategy']
        if C.spec_sha(s) != c['strategy_sha256'] or V.executor_refusal(s) is not None or not V.is_lc_plan(s):
            raise N.Blocked(f'{c["family"]}: strategy sha differs / not an executable lane-change plan')
    return doc['candidates']


# ------------------------------------------------------------------------------------------------ mechanics

def ident_waves(panel, root):
    M = mech()
    fam = M.fixtures()['hold']['name']
    hold = M.bank_entry(root, M.HOLD_BANK)
    a37 = M.C37._a37_specs('A37_PLAN', panel, root, hold, family=fam, tag='a73i_a37')
    a44 = M.a44_specs('A44_PLAN', panel, root, hold, family=fam, tag='a73i_a44')
    return [('A73I_A37_HOLD', 'episode', a37[:IDENT_N]), ('A73I_A44_HOLD', 'episode', a44[:IDENT_N])]


def lcfix_waves(panel, root):
    M = mech()
    fam = M.fixtures()['lc']['name']
    specs = M.a44_specs('A44_PLAN', panel, root, M.bank_entry(root, M.LC_BANK), family=fam, tag='a73m_lcfix')
    return [('A73M_LCFIX', 'episode', specs[:LC_N])]


def _mech_waves(command, panel, final_root, *, n_lc=LC_N, check_files=True):
    """a44_lcmech.waves replacement inside the clone (its lc_report reads the A73 wave)."""
    N = _mods()[0]
    if int(n_lc) != LC_N:
        raise N.Blocked(f'A73: the lane-change fixture runs on {LC_N} worlds')
    return lcfix_waves(panel, final_root) if command == 'lcfix' else ident_waves(panel, final_root)


def identity_compare(panel, root):
    N = _mods()[0]
    M = mech()
    (n37, _, s37), (n44, _, s44) = ident_waves(panel, root)
    r37, r44 = published(root, n37, s37)[0], published(root, n44, s44)[0]
    rows = []
    for wid in [s['world_id'] for s in s37]:
        a, b = r37[wid], r44[wid]
        ea, eb = a['a37'], b['a44']
        diff = [k for k in ('J_veh_s', 'safety') if a.get(k) != b.get(k)]
        ca, cb = log_content_sha(root, a), log_content_sha(root, b)
        if ca != cb:
            diff.append('vehicle_log content')
        diff += [f'executor.{k}' for k in M.EXECUTOR_FIELDS if ea.get(k) != eb.get(k)]
        if eb.get('lc_members'):
            diff.append('executor.lc_members (hold plan produced lane-change members)')
        if not (a.get('executor_sha256_loaded') == b.get('executor_sha256_loaded') == N.FROZEN_MAPPO_SHA):
            diff.append('executor is not the frozen MAPPO U0 in both records')
        rows.append(dict(world_id=wid, equal=not diff, differences=diff, activations=ea.get('activations'),
                         issued=ea.get('issued'), vehicle_log_content_sha256=[ca, cb],
                         container_sha_equal=(a.get('vehicle_log') or {}).get('sha256')
                         == (b.get('vehicle_log') or {}).get('sha256')))
    ok = len(rows) == IDENT_N and all(r['equal'] for r in rows)
    return N.seal(dict(schema='coopmwm.a73.identity.v1', amendment=AMENDMENT,
                       rule='hold plan a44_fix_hold on U0 at D2800_p20: A37 executor path (A37_PLAN) vs the variant '
                            '(A44_PLAN) on worlds 992408 / 992409: J, safety, executor receipts identical, no '
                            'lane-change member, decompressed vehicle logs byte-identical',
                       not_run='the empty-plan half (the variant refuses a bank without a plan); disclosed',
                       verdict='PASS' if ok else 'FAILED',
                       non_vacuous=any((r['activations'] or 0) > 0 for r in rows), worlds=rows,
                       consequence='PASS (and the lane-change fixture report accepted by MAIN): A73 verification may '
                                   'run; FAILED: nothing further is simulated'),
                  'receipt_sha256')


def identity_passed(root):
    N = _mods()[0]
    p = Path(root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ verification

def prepare(root, cands):
    N = _mods()[0]
    M = mech()
    from tools.coopmwm import a44_lcvariant as V
    doc = V.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = M.C37.write_once(Path(root)/BANK_DIR/CANDS_BANK, M.C37.bank_text(doc))
    return dict(path=str(p), file_sha256=N.file_sha(p), families=[c['family'] for c in cands])


def verify_waves(panel, root, cands):
    N = _mods()[0]
    M = mech()
    entry = M.bank_entry(root, CANDS_BANK)
    waves = [('A73V_U0', 'episode', ref_specs('U0', panel, root, 'a73v_u0'))]
    for c in cands:
        waves.append((f'A73V_{c["family"]}', 'episode',
                      M.a44_specs('A44_PLAN', panel, root, entry, family=c['family'], tag=f'a73v_{c["key"][:8]}')))
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def _reports(root, have, worlds):
    N, A = _mods()
    reports = {a: {w: A.report_of(root, have[a][w]) for w in worlds} for a in have}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    return reports


def lc_counts(recs, worlds):
    tot = dict(members=0, requested=0, completed=0, gave_up=0, never_requested=0)
    for w in worlds:
        s = (recs[w].get('a44') or {}).get('lc_summary') or {}
        for k in tot:
            tot[k] += int(s.get(k) or 0)
    return tot


def admit(panel, root, cands):
    N = _mods()[0]
    M = mech()
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    from tools.coopmwm import a44_lcvariant as V
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources = {}, {}
    (uname, _, uspecs), *cwaves = verify_waves(panel, root, cands)
    have[REF], sources[REF] = published(root, uname, uspecs)
    for (name, _, specs), c in zip(cwaves, cands):
        have[c['family']], sources[c['family']] = published(root, name, specs)
    reports = _reports(root, have, worlds)
    smap = {c['family']: c['strategy'] for c in cands}
    verdicts, rows = {}, []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a44'] for w in worlds]
        sizes = []
        for w in worlds:
            sizes += B.realised_sizes(have[fam][w]['a44'], smap)
        v.update(family=fam, wm_rank=cand['wm_rank'], selected_rank=cand['selected_rank'], key=cand['key'],
                 wm_score=cand['score'], n_roles=len(cand['strategy']['roles']), lc_roles=cand['lc_roles'],
                 reference=REF, convention='diff = plan - U0 (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']), ADMITTED=bool(v['PASS']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=B.group_size_summary(sizes), lane_changes=lc_counts(have[fam], worlds),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        verdicts[fam] = v
        rows.append(v)
    plans = C.admitted_plans(cands, verdicts)
    bank = None
    if plans:
        p = M.C37.write_once(root/BANK_DIR/LC_BANK, M.C37.bank_text(V.bank_doc(plans, CELL)))
        bank = dict(path=f'{HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans])
    value = N.seal(dict(
        schema='coopmwm.a73.admission.v1', amendment=AMENDMENT, cell=CELL, rule=ADMIT_RULE, reference=REF,
        disclosure=DISCLOSURE, multiplicity=f'{len(rows)} picks each tested at 95% (disclosed)',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), verdicts=rows,
        admitted=[p['family'] for p in plans], bank=bank,
        consequence=('confirmation runs U0, DUO_HOLD, LEARNED_TRIGGER, LC_BANK_20' if plans
                     else 'no lane-change plan admitted: no confirmation run'),
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def confirm_waves(panel, root):
    N = _mods()[0]
    M = mech()
    adm = admission(root)
    if adm is None:
        raise N.Blocked('no A73 admission receipt')
    if not adm['bank']:
        return []
    entry = M.bank_entry(root, LC_BANK)
    if entry['file_sha256'] != adm['bank']['file_sha256']:
        raise N.Blocked('LC_BANK file differs from the admission receipt')
    waves = [(f'A73C_{a}', 'episode', ref_specs(a, panel, root, f'a73c_{a.lower()}')) for a in REF_ARMS]
    waves.append((f'A73C_{ARM}', 'episode', M.a44_specs('A44_BANK', panel, root, entry, family=None,
                                                        tag='a73c_lc_bank')))
    if any(len(s) != CONFIRM_N for _, _, s in waves) or tuple(w[5:] for w, _, _ in waves) != CONFIRM_ARMS:
        raise N.Blocked('confirmation waves: wrong size / arms')
    return waves


def _contrast(have, reports, worlds, arm, ref, bootstrap, kind):
    """a42_bigp60._contrast (same estimator: paired percentile bootstrap, whole-world resampling)."""
    import numpy as np
    N = _mods()[0]
    row = dict(treatment=arm, reference=ref, kind=kind,
               convention='diff = treatment - reference; negative = treatment has less delay')
    row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
    for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
        row[f'{k}_minus_ref'] = N.fe.paired_stats(np.array([have[arm][w]['safety'][k] for w in worlds], float),
                                                  np.array([have[ref][w]['safety'][k] for w in worlds], float),
                                                  bootstrap, pct=False)
    return row


def analyse(panel, root):
    import numpy as np
    N = _mods()[0]
    from tools.coopmwm import a42_bigbank as B
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    have, sources = {}, {}
    for wave, _, specs in confirm_waves(panel, root):
        have[wave[5:]], sources[wave[5:]] = published(root, wave, specs)
    if tuple(have) != CONFIRM_ARMS:
        raise N.Blocked('confirmation records incomplete (or no plan admitted)')
    reports = _reports(root, have, worlds)
    rows = [_contrast(have, reports, worlds, ARM, r, bootstrap, 'registered') for r in REF_ARMS]
    rows += [_contrast(have, reports, worlds, a, r, bootstrap, 'context') for a, r in CONTEXT]
    cands = {c['family']: c['strategy'] for c in candidates()}
    sizes, fam, any_ = [], {}, 0
    for w in worlds:
        rec = have[ARM][w]['a44']
        any_ += 1 if rec.get('activations') else 0
        for f, n in (rec.get('activations_by_family') or {}).items():
            fam[f] = fam.get(f, 0)+int(n)
        sizes += B.realised_sizes(rec, cands)
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    arms[ARM].update(activations=dict(worlds_with_any=any_, total=sum(fam.values()), by_family=fam,
                                      realised_group_size=B.group_size_summary(sizes),
                                      full_group=sum(1 for r in sizes if r['held'] == r['planned'])),
                     lane_changes=lc_counts(have[ARM], worlds))
    return N.seal(dict(schema='coopmwm.a73.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, kind='single cell D2800_p20; never pooled with any other panel',
                       admission=dict(receipt_sha256=adm['receipt_sha256'], admitted=adm['admitted'], bank=adm['bank']),
                       contrasts=rows, arms=arms,
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE = dict(identity='m', lcfix='m', verify='v', confirm='c')
STAGE_CAP = dict(m=MECH_CAP, v=VERIFY_CAP, c=CONFIRM_CAP)
STAGE_PREFIX = dict(m=('a73i_', 'a73m_'), v=('a73v_',), c=('a73c_',))
WHICH = dict(STAGE, **{'prepare-mech': 'm', 'identity-compare': 'm', 'lc-report': 'm', 'prepare': 'v', 'admit': 'v',
                       'analyse': 'c'})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('select', 'panel', 'pin', 'prepare-mech', 'prepare', 'dry', 'identity',
                                       'identity-compare', 'lcfix', 'lc-report', 'verify', 'admit', 'confirm',
                                       'analyse', 'status'))
    p.add_argument('--final-root')
    p.add_argument('--stage', choices=('m', 'v', 'c'), default='m')
    p.add_argument('--search')
    p.add_argument('--map-spec')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    a = p.parse_args(argv)
    if a.command == 'select':
        doc = select(a.search, a.out or CANDS_FILE)
        print(json.dumps(dict(doc_sha256=doc['doc_sha256'], STOP=doc['STOP'], skip_counts=doc['skip_counts'],
                              n_eligible=doc['n_eligible'], n_final=doc['search']['n_final'],
                              role_counts=doc['role_counts'],
                              candidates=[(c['family'], c['score'], c['n_roles'], c['lc_roles'], c['wm']['feasible'])
                                          for c in doc['candidates']])))
        return
    N = _mods()[0]
    if a.command == 'pin':
        print(json.dumps(make_pins()))
        return
    which = WHICH.get(a.command, a.stage)
    if a.command == 'panel':
        setup(which, need_panel=False)
        import a37_celltailor as C37
        from datetime import datetime, timezone
        value = N.build_panel(a.map_spec or N.default_spec_path(), datetime.now(timezone.utc).isoformat())
        C37.write_once(PANEL_FILE[which], json.dumps(value, indent=1, sort_keys=True)+'\n')
        print(json.dumps(dict(stage=which, panel_sha256=value['panel_sha256'], file=str(PANEL_FILE[which]),
                              cell=value['cell']['cell_key'],
                              seeds=[value['worlds'][0]['world']['seed'], value['worlds'][-1]['world']['seed']])))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    panel = setup(which)
    M = mech()

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), stage=which), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(command):
        if command == 'identity':
            return ident_waves(panel, root)
        if command == 'lcfix':
            return lcfix_waves(panel, root)
        if command == 'verify':
            return verify_waves(panel, root, candidates())
        return confirm_waves(panel, root)

    def stage_waves(w, need_pin=True):
        if w == 'm':
            return ident_waves(panel, root)+lcfix_waves(panel, root)
        if w == 'v':
            return verify_waves(panel, root, candidates(need_pin=need_pin))
        return confirm_waves(panel, root)

    def brief(ws):
        return {n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'], runner_arm=s[0]['arm'],
                        seeds=[s[0]['world']['seed'], s[-1]['world']['seed']], cell=s[0]['world']['cell_key'],
                        checkpoint={k: v for k, v in s[0]['checkpoint'].items() if k != 'label'})
                for n, _, s in ws}
    if a.command == 'prepare-mech':
        print(json.dumps(M.prepare(root), indent=1))
    elif a.command == 'prepare':
        print(json.dumps(prepare(root, candidates()), indent=1))
    elif a.command == 'dry':
        ws = stage_waves(which, need_pin=False)
        print(json.dumps(dict(stage=which, setting=SETTING[which], panel_sha256=panel['panel_sha256'],
                              pinned=pins().get(f'panel_{which}') == panel['panel_sha256'], digests=digests(),
                              waves=brief(ws), sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws),
                              stage_cap=STAGE_CAP[which], charged=M.charged(root), cap=CAP_SIM_S), indent=1))
    elif a.command in STAGE:
        import fcntl
        require_pins(panel, which)
        if a.command in ('verify', 'confirm') and not identity_passed(root):
            raise N.Blocked(f'{IDENTITY_FILE} is not PASS: nothing is run')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            if not waves:
                log(dict(done=a.command, status='SKIPPED_NO_PLAN_ADMITTED'))
                return
            pending = sum(len(s)-N.published(root, w, s, M.check_a44)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            used = M.charged(root, STAGE_PREFIX[which])
            if used+need > STAGE_CAP[which] or M.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[which]} (or phase cap)')
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=M.charged(root)))
            # no automatic replay: a failed dispatch raises and is reported
            res = M.execute(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=panel,
                            runtime=runtime(root))
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(panel, root)
        N.publish(root/IDENTITY_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], non_vacuous=v['non_vacuous'],
                              worlds=[(r['world_id'], r['equal'], r['differences'], r['activations'],
                                       r['container_sha_equal']) for r in v['worlds']])))
    elif a.command == 'lc-report':
        v = M.lc_report(root, panel, LC_N)
        N.publish(root/LC_MECH_FILE, v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'], totals=v['totals'])))
    elif a.command == 'admit':
        v = admit(panel, root, candidates())
        print(json.dumps(dict(admitted=v['admitted'], receipt_sha256=v['receipt_sha256'],
                              rows=[(x['family'], round(x['mean'], 1), [round(c, 1) for c in x['ci95']],
                                     x['group_rule_pass'], x['ADMITTED'], x['lane_changes'])
                                    for x in v['verdicts']])))
    elif a.command == 'analyse':
        v = analyse(panel, root)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'],
                              rows=[(r['treatment'], r['reference'], r['total_minus_ref']['diff'],
                                     r['total_minus_ref']['ci95']) for r in v['contrasts']],
                              lane_changes=v['arms'][ARM]['lane_changes'])))
    else:   # status
        out = dict(charged=M.charged(root), cap=CAP_SIM_S, identity_pass=identity_passed(root), stage=which,
                   stage_charged=M.charged(root, STAGE_PREFIX[which]), stage_cap=STAGE_CAP[which])
        try:
            for n, _, s in stage_waves(which, need_pin=False):
                out[n] = f'{N.published(root, n, s, M.check_a44)[2]}/{len(s)}'
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
