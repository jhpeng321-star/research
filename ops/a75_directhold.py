"""AMENDMENT 75, stage D1 (S-DIRECTHOLD60C; MAIN 2026-11-13; POST-HOC, exploratory): wider hold forms tested DIRECTLY
in SUMO at D2800_p60, without the world model.

A fixed seeded random draw (seed 20261113, N = 36) from a widened grid that uses only fields the current registered
executor already honours (oracle_plan.StrategyCore / select_members through the A37_PLAN arm; nothing in the executor
or runner is changed). Plans are frozen once drawn (sealed + pinned file); no hand-picking, no replacement.

Stages (own root; ledger phase ``directhold_a75``; stage caps: identity 4,800, admission 1,036,800, confirmation 115,200):
  draw      zero SUMO: the 36 plans (DRAW_RULE), sealed; never overwrites
  ident     MAPPO60C alone and the first A68 candidate plan on the first 2 admission worlds, run from THIS tree, must
            equal the stored A68 records exactly (J, safety, world, activations) -> the stored reference binds
  verify    each plan alone (A37_PLAN arm on MAPPO60C) on the 24 A68/A71 admission worlds 992500-992523; the MAPPO60C
            reference is the stored A68 wave A68V_MAPPO60C (read-only, not re-run; receipt + report shas pinned)
  admit     the A71 rule unchanged: (plan - MAPPO60C) CI95 upper < 0 AND the five-group 10 s rule; duplicates as A71
            (identical 24 per-world results as a plan earlier in draw order); bank = BANK_RULE (fixed before outcomes)
  confirm   one arm DIRECT_BANK_60C on the A70 96-world panel 996300-996395, paired with the stored A70 MAPPO60C and
            DUO_HOLD_60C runs (read-only). Not run if no plan is admitted.
  analyse   primary DIRECT_BANK_60C - MAPPO60C (CI95 upper < 0 AND five-group rule); vs DUO_HOLD_60C descriptive;
            worst-10% delay (CVaR90) as in A70 / A71.
--drop-raw (verify / confirm; THIS run's new output only): once an episode's sealed wave receipt is published and its
per-vehicle report is written and re-verified against the log, the raw archive files of that episode
(RESULT.json.gz, vehicles.jsonl.gz, observable.jsonl.gz) are removed and a sealed DROPPED_RAW.json (sizes + sha256 of
what was removed) is left in the episode directory. Nothing outside <root>/evaluation/a75[vc]_* is ever removed.
No automatic replay: a failed dispatch raises and is reported.

Commands (python ops/a75_directhold.py <cmd> --final-root R --a68-root R68 --a66-root R66 --ref68-reports RR68
          [--a70-root R70 --ref70-reports RR70]):
  draw [--out F] ; prepare ; dry --stage i|v|c ; ident ; identity-compare ; verify [--drop-raw] ; admit ;
  confirm [--drop-raw] ; analyse ; status --stage v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '75'
CELL = 'D2800_p60'
PHASE = 'directhold_a75'
IDENT_N, VERIFY_N, CONFIRM_N = 2, 24, 96
IDENT_CAP, VERIFY_CAP, CONFIRM_CAP = 4_800, 1_036_800, 115_200
CAP_SIM_S = IDENT_CAP+VERIFY_CAP+CONFIRM_CAP
# ---- the draw (frozen)
DRAW_SEED, DRAW_N = 20261113, 36
HOLDS = (10, 20, 30, 45, 60)
SIZES = (2, 3, 4)
V_W = (12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0)
DMAX = (1.0, 1.5)
ONSETS = (0, 20, 40, 60)
COOLDOWNS = (30, 60, 90, 140)
ZONES = ('far', 'full', 'near')
SPREADS = (200, 300)
P_WALL = 0.4
MAX_BANK = 4
SETTINGS = dict(v='a68v_D2800_p60', c='a70c_D2800_p60')
RANGES = dict(v='A68_RANGES', c='A70_RANGES')
SEEDS = dict(v=tuple(range(992500, 992524)), c=tuple(range(996300, 996396)))
PANEL_SHA256 = dict(v='989248f9d449b234724d74cd569e324e1be7bc1038529947429f336652412ea4',
                    c='1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1')
A75_DIR = HERE/'a75d1'
CANDS_FILE = A75_DIR/f'CANDIDATES_D1_{CELL}.json'
CANDS_PIN = A75_DIR/f'CANDIDATES_D1_{CELL}.sha256'
REF_PINS = A75_DIR/'REF_PINS.json'
BANK_DIR = 'a75_banks'
CANDS_BANK = f'A75_CANDS_{CELL}.json'
DIRECT_BANK = f'A75_DIRECT_BANK_{CELL}.json'
A68_BANK = f'A68_CANDS_{CELL}.json'
IDENTITY_FILE = 'IDENTITY_A75.json'
ADMISSION_FILE = f'A75_ADMISSION_{CELL}.json'
RESULTS_FILE = f'A75_RESULTS_{CELL}.json'
DROP_MANIFEST = 'A75_DROPPED_RAW.jsonl'
DROP_FILE = 'DROPPED_RAW.json'
RAW_FILES = ('RESULT.json.gz', 'vehicles.jsonl.gz', 'observable.jsonl.gz')
REF = 'MAPPO60C'
ARM = 'DIRECT_BANK_60C'
CONFIRM_WAVE = f'A75C_{ARM}'
REF_ARMS = ('MAPPO60C', 'DUO_HOLD_60C')
AUTHORITY = ('MAIN AMENDMENT 75 stage D1 (S-DIRECTHOLD60C, 2026-11-13; POST-HOC, exploratory): 36 hold plans drawn at '
             'random (seed 20261113) from a widened grid of fields the current executor honours, each verified in '
             'SUMO on the 24 A68 admission worlds vs the stored MAPPO60C; one bank; one arm on the A70 panel; phase '
             'directhold_a75, caps 4,800 + 1,036,800 + 115,200')
DISCLOSURE = ('POST-HOC and exploratory: run after A68 / A70 / A71 / A73 were seen; no world model is used anywhere. '
              'Multiplicity: 36 plans are each tested at 95% on the same 24 worlds, so some passes are expected by '
              'chance; admission is a screen, not a result, and the admission bounds that rank plans are selection-'
              'biased. The admission worlds were already used by A68 and A71, and the A70 panel was already viewed for '
              'SMALL1 / SMALL2 / WIDE_BANK; neither played a part in drawing the plans (seeded draw) or in the bank '
              'rule (fixed before any A75 episode). Never pooled with other panels')
DRAW_RULE = ('random.Random(20261113); 36 (hold_s, size) cells = every hold_s in {10,20,30,45,60} x size in {2,3,4} '
             'twice (30) + for each size 2 more distinct hold_s drawn at random (6); order shuffled = draw order. Per '
             'plan: one hold_s for all roles; trigger = wall with probability 0.4 (sizes <= 3 only; rank 0, distinct '
             'lanes, spread 200), else lanes with distinct (lane in 0..2, rank in 0..1) and spread_m in {200,300}; '
             'roles sorted by (lane, rank); per role independently v_w in {12,14,16,18,20,22,24} m/s, dmax in '
             '{1.0,1.5} m/s2, onset_s in {0,20,40,60} (if no role has onset 0 the first role is set to 0); zone in '
             '{far,full,near}; cooldown_s in {30,60,90,140}; slots 1; gate / start_gate / phases / lc_target none. A '
             'plan whose strategy key was already drawn is redrawn in place. No selection, no replacement afterwards')
ADMIT_RULE = ('(plan - MAPPO60C) total delay CI95 upper < 0 on the 24 A68 admission worlds (a37_cellbank.unc_verdict) '
              'AND the five-group 10 s rule relative to MAPPO60C; reference = the stored A68 wave A68V_MAPPO60C')
BANK_RULE = ('a plan whose 24 per-world results (per_seed, plan - MAPPO60C, 0.1 veh-s) equal those of a plan earlier '
             'in draw order is marked duplicate and cannot take a bank slot; for each hold_s class 10 / 20 / 30 / 45 / '
             '60 the single admitted non-duplicate plan with the most negative admission CI95 upper bound (ties: '
             'strategy key); if all five classes have one, the four with the most negative CI95 upper bound are kept '
             '(ties: strategy key) -> at most 4 plans; each plan keeps its admission lower bound as bank priority; '
             'plans and executor fields unchanged')
PRIMARY_RULE = ('SUPPORTED iff DIRECT_BANK_60C - MAPPO60C total delay CI95 upper < 0 (paired percentile bootstrap '
                '10,000, seed 20261104, n = 96) AND the five-group 10 s rule relative to MAPPO60C; diff = treatment - '
                'reference, negative = less delay')
DROP_RULE = ('after the sealed wave receipt of the episode is published: the per-vehicle report is computed from the '
             'log whose sha256 equals the sealed record, written as vehicles_report.json, re-read and compared; the '
             'RESULT.json pointer sha is checked against RESULT.json.gz; sizes and sha256 of the raw files are sealed '
             'in DROPPED_RAW.json; only then RESULT.json.gz, vehicles.jsonl.gz and observable.jsonl.gz of that episode '
             'are removed. Only episodes a75v_* / a75c_* of this root; identity episodes are never dropped')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    return N, A50, A67, A68


def _a70():
    import a70_smallbank as A70
    return A70


def core(which):
    N, A50 = _mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A75',
        HOST_BASE=os.environ.get('A75_HOST_BASE', '/root/a75'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A75_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A75', JOB_PREFIX='a75',
        VERIFY_TAG='a75v', CONFIRM_TAG='a75c', IDENT_TAG='a75i',
        VERIFY_WAVE_PREFIX='A75V_', CONFIRM_WAVE_PREFIX='A75C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE='A75_LOG.jsonl', SCHEMA_PREFIX='coopmwm.a75',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 75 stage D1; POST-HOC exploratory direct SUMO test of randomly drawn wider hold forms '
                     'on MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a75_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    return m


def setup(which):
    """-> (D, panel, Dref): Dref = the A68 (v) / A70 (c) driver core whose stored records are the references."""
    N, _, _, A68 = _mods()
    Dref, pref = (A68.setup('v') if which == 'v' else _a70().setup('c'))
    D = core(which)
    panel = D.setup(SETTINGS[which])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if (seeds != SEEDS[which] or panel['cell']['cell_key'] != CELL or panel['panel_sha256'] != pref['panel_sha256']
            or panel['panel_sha256'] != PANEL_SHA256[which]):
        raise N.Blocked(f'{SETTINGS[which]}: not the registered panel')
    return D, panel, Dref


# ------------------------------------------------------------------------------------------------ the draw

def _draw_plan(rng, hold, n):
    wall = n <= 3 and rng.random() < P_WALL
    if wall:
        slots = [(ln, 0) for ln in rng.sample((0, 1, 2), n)]
    else:
        slots = rng.sample([(ln, rk) for ln in (0, 1, 2) for rk in (0, 1)], n)
    roles = [dict(lane=ln, rank=rk, v_w=float(rng.choice(V_W)), dmax=float(rng.choice(DMAX)), hold_s=int(hold),
                  onset_s=int(rng.choice(ONSETS)), lc_target=None) for ln, rk in sorted(slots)]
    if all(r['onset_s'] for r in roles):
        roles[0]['onset_s'] = 0
    zone = rng.choice(ZONES)
    spread = 200 if wall else rng.choice(SPREADS)
    cooldown = rng.choice(COOLDOWNS)
    return dict(name='', trigger='wall' if wall else 'lanes', zone=zone, spread_m=float(spread), cooldown_s=int(cooldown),
                slots=1, gate=None, start_gate=None, phases=None, roles=roles)


def draw_strategies(seed=DRAW_SEED):
    """Pure (DRAW_RULE): the 36 strategies in draw order."""
    from tools.coopmwm import oracle_plan as O
    rng = random.Random(seed)
    cells = [(h, n) for h in HOLDS for n in SIZES]*2
    for n in SIZES:
        cells += [(h, n) for h in rng.sample(HOLDS, 2)]
    rng.shuffle(cells)
    out, seen = [], set()
    for hold, n in cells:
        for _ in range(1000):
            s = _draw_plan(rng, hold, n)
            key = O.strategy_key(s)
            if key not in seen:
                break
        else:
            raise RuntimeError('draw: no new plan')
        seen.add(key)
        s['name'] = f'd1_{key}'
        out.append(s)
    if len(out) != DRAW_N:
        raise RuntimeError('draw size')
    return out


def draw(out):
    """Zero SUMO: the registered candidate list, sealed; never overwrites."""
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    out = Path(out)
    cands = []
    for i, s in enumerate(draw_strategies()):
        why = C.executor_refusal(s)
        if why is not None:
            raise SystemExit(f'{s["name"]}: the registered executor refuses it ({why})')
        cands.append(dict(family=s['name'], name=s['name'], key=B.O.strategy_key(s), draw_rank=i+1, selected_rank=i+1,
                          n_roles=len(s['roles']), hold_s=s['roles'][0]['hold_s'], trigger=s['trigger'],
                          zone=s['zone'], strategy=s, strategy_sha256=C.spec_sha(s)))
    got, refused = B.O.bank_plans(C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0)
                                                 for c in cands], CELL), CELL)
    if len(got) != len(cands) or refused:
        raise SystemExit(f'bank_plans accepts {len(got)} of {len(cands)} (refused {refused})')

    def count(f):
        return dict(sorted(Counter(str(f(c)) for c in cands).items()))
    doc = B.seal(dict(
        schema='coopmwm.a75.candidates.v1', amendment=AMENDMENT, stage='D1', cell=CELL, rule=DRAW_RULE,
        draw=dict(seed=DRAW_SEED, n=DRAW_N, generator='python random.Random', hold_s=list(HOLDS), sizes=list(SIZES),
                  v_w=list(V_W), dmax=list(DMAX), onset_s=list(ONSETS), cooldown_s=list(COOLDOWNS), zone=list(ZONES),
                  spread_m=list(SPREADS), p_wall=P_WALL),
        world_model='none (direct SUMO test; no WM score, no WM rank)',
        n_candidates=len(cands), candidates_by_hold_s=count(lambda c: c['hold_s']),
        candidates_by_size=count(lambda c: c['n_roles']), candidates_by_zone=count(lambda c: c['zone']),
        candidates_by_trigger=count(lambda c: c['trigger']),
        candidates_by_cooldown_s=count(lambda c: c['strategy']['cooldown_s']),
        candidates_by_hold_and_size=count(lambda c: f'{c["hold_s"]}x{c["n_roles"]}'),
        STOP=not cands, candidates=cands))
    if out.exists():
        raise SystemExit(f'{out}: exists (pre-outcome file; never overwrite)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
    return doc


def candidates(path=None):
    """The pinned pre-outcome candidates; refuses an unpinned / edited file or one that is not the seeded draw."""
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    from tools.coopmwm import a42_bigbank as B
    path = Path(path or CANDS_FILE)
    if not path.is_file() or not CANDS_PIN.is_file():
        raise N.Blocked(f'{path} / {CANDS_PIN.name}: missing (run draw, record the doc sha in REGISTRATION.md)')
    doc = N.read(path)
    pin = CANDS_PIN.read_text(encoding='utf-8').split()[0]
    if B.doc_sha(doc) != doc.get('doc_sha256') or doc['doc_sha256'] != pin:
        raise N.Blocked(f'{path.name}: not the registered draw (pin {pin})')
    if doc['cell'] != CELL or doc.get('amendment') != AMENDMENT or doc['STOP'] or doc['n_candidates'] != DRAW_N:
        raise N.Blocked(f'{path.name}: not the registered cell / size')
    cands = doc['candidates']
    if [c['strategy'] for c in cands] != draw_strategies():
        raise N.Blocked(f'{path.name}: the candidates are not the seeded draw (hand-edited?)')
    for i, c in enumerate(cands):
        s = c['strategy']
        if (C.spec_sha(s) != c['strategy_sha256'] or C.executor_refusal(s) is not None or c['family'] != s['name']
                or c['key'] != B.O.strategy_key(s) or c['draw_rank'] != i+1 or len(s['roles']) != c['n_roles']
                or {r['hold_s'] for r in s['roles']} != {c['hold_s']} or c['hold_s'] not in HOLDS):
            raise N.Blocked(f'{c["family"]}: strategy sha differs / not an executable hold plan')
    return cands


def ref_pins():
    N = _mods()[0]
    if not REF_PINS.is_file():
        raise N.Blocked(f'{REF_PINS}: missing')
    return N.verify(N.read(REF_PINS), 'receipt_sha256', REF_PINS.name)


# ------------------------------------------------------------------------------------------------ verification

def prepare(D, root, cands, a68_root=None):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    doc = C.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    p = D.C37.write_once(D.bank_path(root, CANDS_BANK), D.C37.bank_text(doc))
    out = dict(path=str(p), file_sha256=N.file_sha(p), n=len(cands))
    if a68_root is not None:       # the A68 candidate bank bytes (identity stage only), copied, never edited
        src = Path(a68_root)/'a68_banks'/A68_BANK
        q = D.C37.write_once(D.bank_path(root, A68_BANK), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != N.file_sha(src):
            raise N.Blocked(f'{q}: not the A68 candidate bank bytes')
        out['a68_bank_sha256'] = N.file_sha(q)
    return out


def verify_waves(D, panel, root, a66_root, cands):
    """[(wave, 'episode', specs)]: every drawn plan (draw order) alone on the 24 worlds; the reference is not re-run."""
    N, _, A67, A68 = _mods()
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, CANDS_BANK)
    waves = []
    for c in cands:
        tag = f'a75v_{c["key"]}'
        base = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=c['family'], tag=tag)
        waves.append((f'A75V_{c["family"]}', 'episode', A67.with_exec(base, exe, tag)))
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def a68_ref(D68, panel, a68_root, a66_root):
    """The stored A68 MAPPO60C records of the 24 admission worlds (read-only; checked against their A68 specs and the
    pinned wave receipt)."""
    N, _, _, A68 = _mods()
    have, src = D68.C37._load(a68_root, 'A68V_MAPPO60C',
                              A68.mappo_specs(D68, panel, a68_root, A68.executor(a66_root), 'a68v_mappo60c'))
    got = src['receipt_sha256']
    if (got if isinstance(got, list) else [got]) != ref_pins()['a68v_mappo60c']['wave_receipt_sha256']:
        raise N.Blocked('A68V_MAPPO60C: not the pinned wave receipt')
    return have, src


def pinned_reports(reports_root, have, worlds, pin):
    """Reference reports from a reports-only root (evaluation/<job>/vehicles_report.json copied from the archived
    A68 / A70 episode records): file sha256 = pin, then a32_ablation.report_of (log sha of the sealed record + report
    version)."""
    N = _mods()[0]
    from a32_ablation import report_of
    out = {}
    for w in worlds:
        rec = have[w]
        job = rec.get('work_dir', rec['job_id'])
        p = Path(reports_root)/'evaluation'/job/'vehicles_report.json'
        if not p.is_file() or N.file_sha(p) != pin['reports'].get(job):
            raise N.Blocked(f'{job}: reference report missing / not the pinned file')
        if pin['vehicle_logs'].get(job) != rec['vehicle_log']['sha256']:
            raise N.Blocked(f'{job}: sealed vehicle-log sha differs from the pin')
        out[w] = report_of(reports_root, rec)
    return out


def _own_reports(root, have, worlds):
    from a32_ablation import report_of
    return {w: report_of(root, have[w]) for w in worlds}


def _check_reports(reports):
    N = _mods()[0]
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    return reports


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


# ------------------------------------------------------------------------------------------------ identity

def ident_waves(D, D68, panel, root, a66_root):
    """MAPPO60C alone and the first A68 candidate plan (A37_PLAN arm) on the first 2 admission worlds."""
    N, _, A67, A68 = _mods()
    exe = A68.executor(a66_root)
    c = A68.candidates()[0]
    entry = A68.bank_entry(D, root, A68_BANK)
    base = D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=c['family'], tag='a75i_plan')
    return [('A75_IDENT_MAPPO60C', 'episode', A68.mappo_specs(D, panel, root, exe, 'a75i_mappo60c')[:IDENT_N]),
            ('A75_IDENT_PLAN', 'episode', A67.with_exec(base, exe, 'a75i_plan')[:IDENT_N])]


def identity_compare(D, D68, panel, root, a68_root, a66_root):
    N, _, A67, A68 = _mods()
    waves = ident_waves(D, D68, panel, root, a66_root)
    ref60, _ = a68_ref(D68, panel, a68_root, a66_root)
    c = A68.candidates()[0]
    w68 = {n: s for n, _, s in A68.verify_waves(D68, panel, a68_root, a66_root, A68.candidates())}
    refplan, _ = D68.C37._load(a68_root, f'A68V_{c["family"]}', w68[f'A68V_{c["family"]}'])
    rows = []
    for (wave, _, specs), ref in zip(waves, (ref60, refplan)):
        got, _, cov = N.published(root, wave, specs, D.C37.check_a37)
        if cov != len(specs):
            raise N.Blocked(f'{wave}: {cov}/{len(specs)} published')
        for r in got:
            a = ref[r['world_id']]
            ra, aa = r.get('a37') or {}, a.get('a37') or {}
            rows.append(dict(wave=wave, world_id=r['world_id'], j_a75=r['J_veh_s'], j_stored=a['J_veh_s'],
                             abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_equal=r['safety'] == a['safety'],
                             same_world=r['world'] == a['world'],
                             executor_is_mappo60c=r.get('executor_sha256_loaded') == A67.MAPPO60C_SHA256
                             and a.get('executor_sha256_loaded') == A67.MAPPO60C_SHA256,
                             activations=[int(ra.get('activations') or 0), int(aa.get('activations') or 0)],
                             vehicle_log_equal=(r.get('vehicle_log') or {}).get('sha256')
                             == (a.get('vehicle_log') or {}).get('sha256')))
    ok = (len(rows) == 2*IDENT_N and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world']
                                         and x['executor_is_mappo60c']
                                         and x['activations'][0] == x['activations'][1] for x in rows))
    return N.seal(dict(schema='coopmwm.a75.identity.v1', amendment=AMENDMENT,
                       rule='MAPPO60C alone and the first A68 candidate plan on the first 2 admission worlds, run from '
                            'the A75 tree, equal the stored A68 records exactly: J_veh_s difference 0, identical '
                            'safety counters, same activations (vehicle-log sha equality is reported, not required)',
                       plan=c['family'], verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: the stored A68V_MAPPO60C reference binds and the plans may run; FAILED: '
                                   'stop and report'), 'receipt_sha256')


def identity_passed(root):
    N = _mods()[0]
    p = Path(root)/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ drop raw

def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _report_job(args):
    """Worker: compute + cache the per-vehicle report of one published episode (log sha checked inside)."""
    N = _mods()[0]
    root, rec = args
    rep = N.episode_log_report(root, rec)
    return rec['job_id'], bool(rep and rep.get('ok'))


def drop_raw(root, waves, check, *, procs=1, log=None):
    """DROP_RULE for every PUBLISHED episode of ``waves`` that still has raw files. -> counts. Idempotent."""
    N = _mods()[0]
    root = Path(root)
    base = (root/'evaluation').resolve()
    todo = []
    for name, _, specs in waves:
        recs, _, cov = N.published(root, name, specs, check)
        for r in recs[:cov]:
            job = r.get('work_dir', r['job_id'])
            d = (root/'evaluation'/job)
            if not r['job_id'].startswith(('a75v_', 'a75c_')) or d.resolve().parent != base or d.is_symlink():
                raise N.Blocked(f'{job}: not an A75 verification / confirmation episode directory')
            if (d/DROP_FILE).is_file():
                if any((d/f).exists() for f in RAW_FILES):
                    raise N.Blocked(f'{job}: {DROP_FILE} and raw files both present')
                continue
            if not (d/N.VEHICLE_LOG).is_file():
                raise N.Blocked(f'{job}: no vehicle log and no {DROP_FILE}')
            todo.append((name, r, d))
    if not todo:
        return dict(dropped=0, bytes=0)
    jobs = [(str(root), r) for _, r, _ in todo]
    if procs > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(procs, len(jobs))) as pool:
            done = dict(pool.imap_unordered(_report_job, jobs, chunksize=1))
    else:
        done = dict(map(_report_job, jobs))
    n = freed = 0
    for name, r, d in todo:
        job = d.name
        if done.get(r['job_id']) is not True:
            raise N.Blocked(f'{job}: no valid vehicle-log report; raw files kept')
        cache = d/'vehicles_report.json'
        c = N.read(cache)
        log_sha = _sha(d/N.VEHICLE_LOG)
        if (c.get('log_sha256') != r['vehicle_log']['sha256'] or log_sha != r['vehicle_log']['sha256']
                or c.get('version') != N.REPORT_VERSION or not c['report'].get('ok')):
            raise N.Blocked(f'{job}: cached report does not match the sealed record; raw files kept')
        ptr = N.read(d/'RESULT.json')
        files = []
        for f in RAW_FILES:
            if (d/f).is_file() and not (d/f).is_symlink():
                files.append(dict(file=f, bytes=(d/f).stat().st_size, sha256=log_sha if f == N.VEHICLE_LOG else _sha(d/f)))
        got = {x['file']: x['sha256'] for x in files}
        if ptr.get('file') != 'RESULT.json.gz' or got.get('RESULT.json.gz') != ptr.get('sha256'):
            raise N.Blocked(f'{job}: RESULT.json does not point at the RESULT.json.gz on disk; raw files kept')
        value = N.seal(dict(schema='coopmwm.a75.dropped_raw.v1', amendment=AMENDMENT, rule=DROP_RULE, job_id=r['job_id'],
                            work_dir=job, wave=name, world_id=r['world_id'], seed=r['world']['seed'],
                            vehicle_log_sha256_sealed=r['vehicle_log']['sha256'],
                            report=dict(file='vehicles_report.json', sha256=_sha(cache), version=c['version']),
                            result_pointer=dict(ptr), removed=files, unix=time.time()), 'receipt_sha256')
        N.publish(d/DROP_FILE, value)
        N.verify(N.read(d/DROP_FILE), 'receipt_sha256', DROP_FILE)
        if N.read(cache).get('log_sha256') != r['vehicle_log']['sha256']:
            raise N.Blocked(f'{job}: report unreadable after the drop receipt; raw files kept')
        for x in files:
            os.unlink(d/x['file'])
            freed += x['bytes']
        n += 1
        with (root/DROP_MANIFEST).open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(job_id=r['job_id'], wave=name, receipt_sha256=value['receipt_sha256'],
                                    bytes=sum(x['bytes'] for x in files)), sort_keys=True)+'\n')
        getattr(N, '_REPORTS', {}).pop(r['vehicle_log']['sha256'], None)
    out = dict(dropped=n, bytes=freed)
    if log:
        log(dict(drop_raw=out))
    return out


# ------------------------------------------------------------------------------------------------ admission

def bank_choice(rows):
    """Pure (BANK_RULE). rows: verdict rows with family, key, draw_rank, hold_s, PASS, ci95, per_seed.
    -> (chosen families, best first; {family: duplicate_of}; per-hold_s table)."""
    order = sorted(rows, key=lambda r: (r['draw_rank'], r['key']))
    first, dup = {}, {}
    for r in order:
        sig = json.dumps(sorted((str(k), float(v)) for k, v in r['per_seed'].items()))
        if sig in first:
            dup[r['family']] = first[sig]
        else:
            first[sig] = r['family']
    if any(r['hold_s'] not in HOLDS for r in rows):
        raise ValueError('a plan outside the registered hold_s classes')
    best, table = [], {}
    for h in HOLDS:
        of = [r for r in rows if r['hold_s'] == h]
        ok = sorted((r for r in of if r['PASS'] and r['family'] not in dup), key=lambda r: (r['ci95'][1], r['key']))
        table[str(h)] = dict(tested=len(of), delay_test_pass=sum(bool(r['ci95'][1] < 0) for r in of),
                             admitted=sum(bool(r['PASS']) for r in of),
                             admitted_non_duplicate=len(ok), duplicates=sum(r['family'] in dup for r in of),
                             best=ok[0]['family'] if ok else None, best_ci95_upper=ok[0]['ci95'][1] if ok else None)
        if ok:
            best.append(ok[0])
    best.sort(key=lambda r: (r['ci95'][1], r['key']))
    chosen = [r['family'] for r in best[:MAX_BANK]]
    for h in HOLDS:
        table[str(h)]['in_bank'] = table[str(h)]['best'] in chosen
    return chosen, dup, table


def admit(D, D68, panel, root, a68_root, a66_root, ref68_reports, cands):
    N = _mods()[0]
    from tools.coopmwm import a37_cellbank as C
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    if not identity_passed(root):
        raise N.Blocked('identity is not PASS')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, reports = {}, {}, {}
    have[REF], sources[REF] = a68_ref(D68, panel, a68_root, a66_root)
    reports[REF] = pinned_reports(ref68_reports, have[REF], worlds, ref_pins()['a68v_mappo60c'])
    for (name, _, specs), c in zip(verify_waves(D, panel, root, a66_root, cands), cands):
        have[c['family']], sources[c['family']] = D.C37._load(root, name, specs)
        reports[c['family']] = _own_reports(root, have[c['family']], worlds)
    _check_reports(reports)
    smap = D._spec_map(cands)
    rows = []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        acts = [have[fam][w]['a37'] for w in worlds]
        s = cand['strategy']
        v.update(family=fam, source='A75 (new run)', draw_rank=cand['draw_rank'], key=cand['key'],
                 n_roles=len(s['roles']), hold_s=cand['hold_s'], trigger=s['trigger'], zone=s['zone'],
                 cooldown_s=s['cooldown_s'], reference=REF,
                 convention='diff = plan - MAPPO60C (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activations_per_ep=sum(int(a.get('activations') or 0) for a in acts)/len(acts),
                 realised_group_size=D._group_sizes(have[fam], worlds, smap),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rows.append(v)
    chosen, dup, table = bank_choice(rows)
    for r in rows:
        r['duplicate_of'] = dup.get(r['family'])
        r['DIRECT_BANK'] = r['family'] in chosen
    plans = C.admitted_plans(cands, {r['family']: dict(r, PASS=r['family'] in chosen) for r in rows})
    bank = None
    if plans:
        p = D.C37.write_once(D.bank_path(root, DIRECT_BANK), D.C37.bank_text(C.bank_doc(plans, CELL)))
        bank = dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans],
                    sizes={x['family']: len(x['strategy']['roles']) for x in plans},
                    hold_s={x['family']: x['strategy']['roles'][0]['hold_s'] for x in plans})

    def by(f):
        t = {}
        for r in rows:
            e = t.setdefault(str(f(r)), dict(tested=0, delay_test_pass=0, admitted=0))
            e['tested'] += 1
            e['delay_test_pass'] += bool(r['ci95'][1] < 0)
            e['admitted'] += bool(r['PASS'])
        return dict(sorted(t.items()))
    value = N.seal(dict(
        schema='coopmwm.a75.admission.v1', amendment=AMENDMENT, stage='D1', cell=CELL, rule=ADMIT_RULE,
        bank_rule=BANK_RULE, reference=REF, disclosure=DISCLOSURE,
        multiplicity=f'{len(rows)} plans each tested at 95% on the same 24 worlds; some passes are expected by '
                     'chance; admission is a screen, not a result',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), n_tested=len(rows),
        per_hold_s=table, per_size=by(lambda r: r['n_roles']), per_zone=by(lambda r: r['zone']),
        per_trigger=by(lambda r: r['trigger']), admitted=[r['family'] for r in rows if r['PASS']],
        n_admitted=sum(bool(r['PASS']) for r in rows), n_delay_pass=sum(bool(r['ci95'][1] < 0) for r in rows),
        duplicates=dup, direct_bank=chosen, bank=bank, verdicts=rows,
        consequence='DIRECT_BANK_60C arm runs on the A70 panel' if plans else 'no plan admitted: no confirmation',
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        ref_pins_receipt_sha256=ref_pins()['receipt_sha256'],
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def confirm_waves(D, panel, root, a66_root):
    N, _, _, A68 = _mods()
    adm = admission(root)
    if adm is None or not adm['bank']:
        raise N.Blocked('no admission receipt / no plan admitted: no confirmation')
    specs = A68.bank_specs(D, panel, root, DIRECT_BANK, A68.executor(a66_root), 'a75c_direct_bank_60c',
                           adm['bank']['file_sha256'])
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{ARM}: {len(specs)} specs')
    return [(CONFIRM_WAVE, 'episode', specs)]


def analyse(D, D70, panel, root, a70_root, a66_root, ref70_reports):
    import numpy as np
    N, _, A67, _ = _mods()
    A70 = _a70()
    from tools.coopmwm import oracle_plan as O
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    sel70 = A70.selection(a70_root)
    pins = ref_pins()
    have, sources, reports = {}, {}, {}
    for arm in REF_ARMS:
        have[arm], sources[arm] = D70.C37._load(a70_root, f'A70C_{arm}',
                                                A70.confirm_specs(D70, arm, panel, a70_root, a66_root, sel70))
        pin = pins[f'a70c_{arm.lower()}']
        got = sources[arm]['receipt_sha256']
        if (got if isinstance(got, list) else [got]) != pin['wave_receipt_sha256']:
            raise N.Blocked(f'A70C_{arm}: not the pinned wave receipt')
        reports[arm] = pinned_reports(ref70_reports, have[arm], worlds, pin)
    (wave, _, specs), = confirm_waves(D, panel, root, a66_root)
    have[ARM], sources[ARM] = D.C37._load(root, wave, specs)
    reports[ARM] = _own_reports(root, have[ARM], worlds)
    _check_reports(reports)
    row = D._contrast(have, reports, worlds, ARM, REF, bootstrap, 'primary')
    ci = row['total_minus_ref']['ci95']
    groups = row['groups'] or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    primary = dict(rule=PRIMARY_RULE, contrast=f'{ARM} - MAPPO60C', diff=row['total_minus_ref']['diff'],
                   ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                   group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                   verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED',
                   worst_10pct_cvar90=row.get('tail_cvar90'))
    rows = [row, D._contrast(have, reports, worlds, ARM, 'DUO_HOLD_60C', bootstrap, 'secondary_descriptive'),
            D._contrast(have, reports, worlds, 'DUO_HOLD_60C', REF, bootstrap, 'secondary_descriptive')]
    pspecs = {p['family']: p['spec'] for p in O.bank_plans(N.read(D.bank_path(root, DIRECT_BANK)), CELL)[0]}
    arms = {}
    for a in have:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        if a != REF:
            arms[a]['activations_per_world'] = sum(int((have[a][w].get('a37') or have[a][w].get('a32') or {})
                                                       .get('activations') or 0) for w in worlds)/len(worlds)
    arms[ARM]['activations'] = A67.activation_table(have[ARM], worlds, pspecs)
    return N.seal(dict(schema='coopmwm.a75.analysis.v1', amendment=AMENDMENT, stage='D1', authority=AUTHORITY,
                       cell=CELL, disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, admission=dict(receipt_sha256=adm['receipt_sha256'], bank=adm['bank'],
                                                           per_hold_s=adm['per_hold_s'],
                                                           direct_bank=adm['direct_bank']),
                       arms_run=[ARM], arms_reused=dict(root=str(a70_root), arms=list(REF_ARMS)), root=str(root),
                       primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       ref_pins_receipt_sha256=pins['receipt_sha256'],
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

STAGE_OF = dict(i='ident', v='verify', c='confirm')
STAGE_CAP = dict(ident=IDENT_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIX = dict(ident='a75i_', verify='a75v_', confirm='a75c_')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('draw', 'prepare', 'dry', 'ident', 'identity-compare', 'verify', 'admit',
                                       'confirm', 'analyse', 'status', 'drop-raw'))
    p.add_argument('--final-root')
    p.add_argument('--a68-root')
    p.add_argument('--a70-root')
    p.add_argument('--a66-root')
    p.add_argument('--ref68-reports')
    p.add_argument('--ref70-reports')
    p.add_argument('--stage', choices=('i', 'v', 'c'), default='v')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    N = _mods()[0]
    if a.command == 'draw':
        doc = draw(a.out or CANDS_FILE)
        print(json.dumps({k: doc[k] for k in doc if k.startswith(('doc_sha', 'n_cand', 'candidates_by'))}))
        return
    root = Path(a.final_root)
    for other in (a.a68_root, a.a70_root, a.a66_root, a.ref68_reports, a.ref70_reports):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A75 final root is a new root')
    which = dict(confirm='c', analyse='c').get(a.command, 'c' if a.stage == 'c' else 'v')
    stage = dict(ident='ident', verify='verify', confirm='confirm').get(a.command, STAGE_OF[a.stage])
    D, panel, Dref = setup(which)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/D.LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(st):
        if st == 'ident':
            return ident_waves(D, Dref, panel, root, a.a66_root)
        if st == 'verify':
            return verify_waves(D, panel, root, a.a66_root, candidates())
        return confirm_waves(D, panel, root, a.a66_root)

    def brief(ws):
        return dict(n_waves=len(ws), first={n: dict(n=len(s), first=s[0]['job_id'], last=s[-1]['job_id'],
                                                   runner_arm=s[0]['arm'],
                                                   seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                                                   cell=s[0]['world']['cell_key'],
                                                   checkpoint={k: v for k, v in s[0]['checkpoint'].items()
                                                               if k != 'label'})
                                               for n, _, s in ws[:1]},
                    waves=[n for n, _, _ in ws])
    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, candidates(), a.a68_root)))
    elif a.command == 'dry':
        ws = waves_of(stage)
        out = dict(stage=stage, panel_sha256=panel['panel_sha256'], **brief(ws),
                   sim_s=sum(len(s)*N.HORIZON_S for _, _, s in ws), stage_cap=STAGE_CAP[stage],
                   charged=D.charged(root), cap=CAP_SIM_S)
        if stage == 'verify':      # the reference binding, checked now (zero SUMO)
            have, src = a68_ref(Dref, panel, a.a68_root, a.a66_root)
            worlds = [r['world_id'] for r in panel['worlds']]
            reps = pinned_reports(a.ref68_reports, have, worlds, ref_pins()['a68v_mappo60c'])
            out['reference'] = dict(wave='A68V_MAPPO60C', receipt_sha256=src['receipt_sha256'], n=len(have),
                                    reports_ok=all(x['ok'] for x in reps.values()),
                                    mean_J_veh_s=sum(have[w]['J_veh_s'] for w in worlds)/len(worlds))
        print(json.dumps(out, indent=1))
    elif a.command in STAGE_CAP:
        import fcntl
        if a.command != 'ident' and not identity_passed(root):
            raise N.Blocked('identity is not PASS: nothing runs')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(a.command)
            pending = sum(len(s)-N.published(root, w, s, D.C37.check_a37)[2] for w, _, s in waves)
            need = pending*N.HORIZON_S
            used = D.charged(root, STAGE_PREFIX[a.command])
            if used+need > STAGE_CAP[a.command] or D.charged(root)+need > CAP_SIM_S:
                raise N.Blocked(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[a.command]} (or phase cap)')
            drop = a.drop_raw and a.command != 'ident'
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), workers=a.workers,
                     chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    drop_raw(root, waves, D.C37.check_a37, procs=a.drop_procs, log=log)
            if drop:               # a previous round of an interrupted launch
                drop_raw(root, waves, D.C37.check_a37, procs=a.drop_procs, log=log)
            # no automatic replay: a failed dispatch raises and is reported
            res = D.execute(root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'drop-raw':
        print(json.dumps(drop_raw(root, waves_of(stage), D.C37.check_a37, procs=a.drop_procs, log=log)))
    elif a.command == 'identity-compare':
        v = identity_compare(D, Dref, panel, root, a.a68_root, a.a66_root)
        N.publish(root/IDENTITY_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], worlds=v['worlds'])))
    elif a.command == 'admit':
        v = admit(D, Dref, panel, root, a.a68_root, a.a66_root, a.ref68_reports, candidates())
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_hold_s=v['per_hold_s'], per_size=v['per_size'],
                              direct_bank=v['direct_bank'], bank=v['bank'], duplicates=v['duplicates'],
                              consequence=v['consequence'])))
    elif a.command == 'analyse':
        v = analyse(D, Dref, panel, root, a.a70_root, a.a66_root, a.ref70_reports)
        out = Path(a.out or root)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/RESULTS_FILE, v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=stage, admission=admission(root) is not None,
                   identity=identity_passed(root))
        try:
            ws = waves_of(stage)
            done = [N.published(root, n, s, D.C37.check_a37)[2] for n, _, s in ws]
            out.update(waves=len(ws), waves_complete=sum(d == len(s) for d, (_, _, s) in zip(done, ws)),
                       episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
