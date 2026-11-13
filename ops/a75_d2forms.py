"""AMENDMENT 75, stage D2 (S-HOLDFORMS-D2; MAIN 2026-11-13; POST-HOC, exploratory): the NEW hold forms of the A75
executor (tools/coopmwm/a75_holdforms.py, digest d1af7e0f; runner arms A75_PLAN / A75_BANK) tested DIRECTLY in SUMO
at D2800_p60 on top of MAPPO60C, without the world model.

A fixed seeded random draw (seed 20261113, stream 'a75d2-holdforms', N = 36) from the SPEC_HOLDFORMS per-field grids,
stratified by form (DRAW_RULE). Plans are frozen once drawn (pinned file); no hand-picking, no replacement.

Stages (own root; ledger phase ``holdforms_a75d2``; caps enforced in code: mech 24,000, admission 1,036,800,
confirmation 115,200 sim-s):
  draw        zero SUMO: the 36 plans (DRAW_RULE) -> ops/a75d2/CANDIDATES_D2_D2800_p60.json (+ .csv, .sha256)
  panel-m     zero SUMO: the 4-world mechanics panel on the coordinator-assigned block A75D2_MECH_SEED0 (write-once,
              pinned in the root before any mechanics episode)
  mech-ident  (setting a68v, worlds 992500/992501 = the first 2 admission worlds) the first A68 candidate plan by the
              A37 path (A37_PLAN) and by the A75 path (A75_PLAN) on MAPPO60C: 4 episodes
  mech-forms  (setting a75d2m, fresh worlds S0..S0+3) the six EXAMPLE_PLANS forms (A75_PLAN) + the six-plan bank
              (A75_BANK): 16 episodes
  mech-compare  zero SUMO -> A75D2_MECH.json PASS/FAIL (MECH_RULE); nothing else runs unless PASS
  verify      each drawn plan alone (A75_PLAN on MAPPO60C) on the 24 A68/A71 admission worlds 992500-992523; the
              MAPPO60C reference is the stored A68 wave A68V_MAPPO60C (read-only, receipt + report shas pinned)
  admit       ADMIT_RULE (A71 / D1 rule unchanged) + BANK_RULE (fixed before outcomes) -> A75D2_ADMISSION + bank
  confirm     one arm A75D2_BANK_60C (A75_BANK) on the A70 96-world panel 996300-996395, paired with the stored A70
              MAPPO60C and DUO_HOLD_60C runs (read-only). Not run if no plan is admitted.
  analyse     primary A75D2_BANK_60C - MAPPO60C; vs DUO_HOLD_60C descriptive; worst-10% delay (CVaR90), groups,
              hard braking, per-form activation counts.
--drop-raw (verify / confirm; THIS run's own episodes only): DROP_RULE (as D1). Mechanics episodes are never dropped.
No automatic replay: a failed dispatch raises and is reported.

Commands (python ops/a75_d2forms.py <cmd> --final-root R --a68-root R68 --a66-root R66 --ref68-reports RR68
          [--a70-root R70 --ref70-reports RR70]):
  draw [--out F] ; panel-m ; prepare ; dry --stage i|m|v|c ; mech-ident ; mech-forms ; mech-compare ; verify
  [--drop-raw] ; admit ; confirm [--drop-raw] ; analyse ; status --stage i|m|v|c ; drop-raw --stage v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
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
STAGE = 'D2'
CELL = 'D2800_p60'
PHASE = 'holdforms_a75d2'
A75_SHA256 = 'd1af7e0ff795f8cd1eb8345c097cc6c113791acd85754c3f664b4ad7dde7dfae'   # a75_holdforms.a75_digest()
HORIZON = 1_200
IDENT_N, VERIFY_N, CONFIRM_N = 2, 24, 96
MECH_EPISODES = 20
MECH_CAP, VERIFY_CAP, CONFIRM_CAP = MECH_EPISODES*HORIZON, 36*VERIFY_N*HORIZON, CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+VERIFY_CAP+CONFIRM_CAP
assert (MECH_CAP, VERIFY_CAP, CONFIRM_CAP) == (24_000, 1_036_800, 115_200)
# ---- the draw (frozen)
DRAW_SEED, DRAW_STREAM, DRAW_N = 20261113, 'a75d2-holdforms', 36
#: SPEC_HOLDFORMS.md v1 grids (no value outside them is ever drawn)
HOLD_GRID = (5, 10, 20, 30, 45, 60, 90, 120, 180)
SHORT_HOLDS = (10, 20, 30, 45)                       # the grid values inside the task's short-hold range 10-45
V_W = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0)
DMAX = (1.0, 1.5)                                    # the dmax values of every registered plan / D1 grid
ONSETS = (0, 10, 20, 40, 60, 90)
OLD_ZONES = ('far', 'full', 'near')
GAP_ZONES = ('near', 'mid', 'merge')
ZONE_D_UP = (1800.0, 1400.0, 1000.0, 700.0, 400.0)
ZONE_D_DOWN = (1000.0, 600.0, 400.0, 200.0, 100.0)
ZONE_D_MIN_SPAN = 200.0
SPREADS = (200.0, 300.0)
GAP_DX_LO = (-20.0, -10.0, 0.0)
GAP_DX_HI = (20.0, 40.0, 60.0)
GAP_DROP_M = (None, 150.0, 300.0, 600.0)
GAP_ARM_S = (30, 60, 120)
GAP_RELEASE = ('timer', 'clear')
GAP_EVENTS = (1, 3)
GAP_HOLDS = (5, 10, 15, 20)
DV_REL = (None, 2.0, 4.0, 6.0, 8.0)
TAPER = (0, 5, 10, 20)
PULSE = (5, 10, 20)
P_EXTRA_FORM_ROLE = 0.5
#: strata (label, count, form family, role-count choices) - draw order is the shuffled list of labels
STRATA = (('gap_alone', 4, 'gap', (1,)), ('gap_plus1', 4, 'gap', (2,)), ('gap_plus2', 4, 'gap', (3,)),
          ('short', 6, 'short', (1, 2, 3)), ('zone_mid', 2, 'zone', (1, 2, 3)), ('zone_merge', 2, 'zone', (1, 2, 3)),
          ('zone_custom', 2, 'zone', (1, 2, 3)), ('pulse', 4, 'pulse', (1, 2, 3)), ('taper', 4, 'taper', (1, 2, 3)),
          ('rolling', 4, 'rolling', (2, 3, 4)))
FORMS = ('gap', 'short', 'zone', 'pulse', 'taper', 'rolling')
MAX_BANK = 4
SETTINGS = dict(i='a68v_D2800_p60', v='a68v_D2800_p60', c='a70c_D2800_p60', m='a75d2m_D2800_p60')
RANGES = dict(i='A68_RANGES', v='A68_RANGES', c='A70_RANGES', m='A75D2_MECH_RANGE')
SEEDS = dict(i=tuple(range(992500, 992524)), v=tuple(range(992500, 992524)), c=tuple(range(996300, 996396)))
PANEL_SHA256 = dict(i='989248f9d449b234724d74cd569e324e1be7bc1038529947429f336652412ea4',
                    v='989248f9d449b234724d74cd569e324e1be7bc1038529947429f336652412ea4',
                    c='1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1')
MECH_N_WORLDS = 4
SEED_ENV = 'A75D2_MECH_SEED0'
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
D2_DIR = HERE/'a75d2'
CANDS_FILE = D2_DIR/f'CANDIDATES_D2_{CELL}.json'
CANDS_CSV = D2_DIR/f'CANDIDATES_D2_{CELL}.csv'
CANDS_PIN = D2_DIR/f'CANDIDATES_D2_{CELL}.sha256'
REF_PINS = D2_DIR/'REF_PINS.json'                      # byte copy of ops/a75d1/REF_PINS.json (receipt 79815fe9)
REF_PINS_RECEIPT = '79815fe94009a8519b5bab1d2a5d2ee6da35503fb15b6cdd75ec96d771ef1ef7'
EXAMPLES = D2_DIR/'EXAMPLE_PLANS.json'                 # byte copy of experiments/a75_holdforms/EXAMPLE_PLANS.json
EXAMPLE_FAMILIES = ('ex_short', 'ex_gap', 'ex_zone', 'ex_pulse', 'ex_taper', 'ex_rolling')
BANK_DIR = 'a75d2_banks'
CANDS_BANK = f'A75D2_CANDS_{CELL}.json'
EX_BANK = f'A75D2_EXAMPLES_{CELL}.json'
D2_BANK = f'A75D2_BANK_{CELL}.json'
A68_BANK = f'A68_CANDS_{CELL}.json'
PANEL_M = 'A75D2_PANEL_M.json'
PANEL_M_PIN = 'A75D2_PANEL_M.sha256'
MECH_FILE = 'A75D2_MECH.json'
ADMISSION_FILE = f'A75D2_ADMISSION_{CELL}.json'
RESULTS_FILE = f'A75D2_RESULTS_{CELL}.json'
LOG_FILE = 'A75D2_LOG.jsonl'
DROP_MANIFEST = 'A75D2_DROPPED_RAW.jsonl'
DROP_FILE = 'DROPPED_RAW.json'
RAW_FILES = ('RESULT.json.gz', 'vehicles.jsonl.gz', 'observable.jsonl.gz')
REF = 'MAPPO60C'
ARM = 'A75D2_BANK_60C'
CONFIRM_WAVE = f'A75D2C_{ARM}'
REF_ARMS = ('MAPPO60C', 'DUO_HOLD_60C')
PREFIX = dict(i='a75d2i_', m='a75d2m_', v='a75d2v_', c='a75d2c_')
STAGE_CAP = dict(mech=MECH_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIXES = dict(mech=(PREFIX['i'], PREFIX['m']), verify=(PREFIX['v'],), confirm=(PREFIX['c'],))
SAFETY_ZERO = ('collisions', 'teleports')
AUTHORITY = ('MAIN AMENDMENT 75 stage D2 (S-HOLDFORMS-D2, 2026-11-13; POST-HOC, exploratory): 36 plans of the NEW A75 '
             'hold forms drawn at random (seed 20261113, stream a75d2-holdforms) from the SPEC_HOLDFORMS grids, each '
             'verified in SUMO on the 24 A68 admission worlds vs the stored MAPPO60C; one bank; one arm on the A70 '
             'panel; phase holdforms_a75d2, caps 24,000 + 1,036,800 + 115,200')
DISCLOSURE = ('POST-HOC and exploratory: run after A68 / A70 / A71 / A73 / A75-D1 were seen; no world model is used. '
              'Multiplicity: 36 plans each tested at 95% on the same 24 worlds; admission is a screen, not a result, '
              'and the admission bounds that rank plans are selection-biased. The admission worlds were already used '
              'by A68 / A71 / D1, the A70 panel was already viewed; neither played a part in drawing the plans '
              '(seeded draw) or in the bank rule (fixed before any D2 episode). Never pooled with other panels')
DRAW_RULE = (
    'random.Random(int(sha256("20261113:a75d2-holdforms")[:16 hex], 16)); the 36 stratum labels (gap_alone 4, '
    'gap_plus1 4, gap_plus2 4, short 6, zone_mid 2, zone_merge 2, zone_custom 2, pulse 4, taper 4, rolling 4) are '
    'shuffled = draw order. Every plan: trigger lanes, slots 1, gate / start_gate / phases / lc_target none, spread_m '
    'in {200,300}; (lane, rank) slots distinct, roles sorted by (lane, rank); per role v_w in {8..24 step 2}, dmax in '
    '{1.0,1.5}, onset_s in {0,10,20,40,60,90} (if no role has onset 0 the first role gets 0; rolling excepted), '
    'hold_s in {5,10,20,30,45,60,90,120,180}; cooldown_s in {H, H+30, 60, 120} with H = max role hold_s. '
    'gap: one gap role on SUMO lane 1 (rank 0/1): start gap, gap_dx_lo {-20,-10,0}, gap_dx_hi {20,40,60}, gap_drop_m '
    '{none,150,300,600}, gap_arm_s {30,60,120}, gap_release {timer,clear}, gap_events {1,3}, hold_s {5,10,15,20}, '
    'dv_rel {none,2,4,6,8}, v_w in {0} + V_W grid when dv_rel is set (else V_W grid); + 0/1/2 ordinary roles in SUMO '
    'lanes 1/2; zone in {near,mid,merge}. short: hold_s in {10,20,30,45} per role, zone in {far,full,near}. zone_*: '
    'zone mid / merge / custom with zone_d = (d_up in {1800,1400,1000,700,400}, d_down in {1000,600,400,200,100}, '
    'd_up - d_down >= 200). pulse: role 1 pulsed, every other role pulsed with p 0.5: pulse_on_s, pulse_off_s in '
    '{5,10,20}, hold_s in the grid with hold_s >= on + off. taper: role 1 tapered, others with p 0.5: taper_in_s, '
    'taper_out_s in {0,5,10,20} not both 0, hold_s in the grid with hold_s >= in + out. rolling: 2-4 roles, onsets = '
    '0 then sorted distinct draws from {10,20,40,60,90}, assigned in role order. short / pulse / taper / rolling use '
    'zone in {far,full,near}. Every plan in a75_holdforms.canonical form; a plan the executor refuses '
    '(executor_refusal) or whose a75 strategy_key was already drawn is redrawn in place (same stratum). No '
    'selection, no replacement afterwards')
ADMIT_RULE = ('(plan - MAPPO60C) total delay CI95 upper < 0 on the 24 A68 admission worlds (a37_cellbank.unc_verdict: '
              'paired percentile bootstrap 10,000, seed 20261104) AND the five-group 10 s rule relative to MAPPO60C; '
              'reference = the stored A68 wave A68V_MAPPO60C (not re-run)')
BANK_RULE = ('the D1 BANK_RULE with the hold_s class replaced by the form family: a plan whose 24 per-world results '
             '(per_seed, plan - MAPPO60C, 0.1 veh-s) equal those of a plan earlier in draw order is marked duplicate '
             'and cannot take a bank slot; for each form family gap / short / zone / pulse / taper / rolling the single '
             'admitted non-duplicate plan with the most negative admission CI95 upper bound (ties: strategy key); of '
             'these family bests the (at most) 4 with the most negative CI95 upper bound are kept (ties: strategy key) '
             '-> at most 4 plans; bank priority = admission lb95(gain) (= -CI95 upper), order lb95 desc then key; plans '
             'and executor fields unchanged; written with a75_holdforms.bank_doc')
PRIMARY_RULE = ('SUPPORTED iff A75D2_BANK_60C - MAPPO60C total delay CI95 upper < 0 (paired percentile bootstrap '
                '10,000, seed 20261104, n = 96) AND the five-group 10 s rule relative to MAPPO60C; diff = treatment - '
                'reference, negative = less delay')
MECH_RULE = (
    'PASS iff all of: every mechanics record loaded a75_sha256 == registered (A75 records) and the MAPPO60C executor; '
    'collisions = 0 and teleports = 0 in every episode; I: per world (992500, 992501) A37_PLAN and A75_PLAN of the '
    'first A68 candidate give equal J_veh_s, safety counters, DECOMPRESSED vehicle-log sha256 and executor receipts '
    '(activations, activations_by_family, issued count, acts), A75 extras empty (extended = [], holdforms = []), and '
    'the A37_PLAN record equals the stored A68 record (J, safety, activations); F1 ex_short: >= 1 activation per '
    'world, every release in {hold_end, at_drop, left}, a member released hold_end was held exactly onset_s + hold_s '
    's; F2 ex_gap (4 worlds): gap_fired >= 1 per world, every gap event ended with end_why in {clear, timer, at_drop, '
    'left, hold_end, gap_arm_end}, trigger present with dx inside [gap_dx_lo, gap_dx_hi], issued_s <= hold_s, <= '
    'gap_events events per session, every session released; F3 ex_zone: >= 1 activation per world and every member '
    'SUMO x (vehicle-log s + S_OFF) at its activation second inside zone_d (+-1 m); F4 ex_pulse: >= 1 completed event '
    'per world and every event issued_s == the pulse count of its duration; F5 ex_taper: >= 1 completed event per '
    'world, issued_s == duration, a hold_end event lasted hold_s; F6 ex_rolling: >= 1 activation per world and every '
    'member i released hold_end held exactly role-i onset_s + hold_s; F7 bank: >= 1 activation per world and no '
    'member in two overlapping sessions. Descriptive (reported, not gating): vehicle-log re-derivation of gap events '
    '(trigger lane at fire, trigger lane change within 20 s, member speed 5 s after fire vs v0), families activated '
    'by the bank, hard brakes')
DROP_RULE = ('after the sealed wave receipt of the episode is published: the per-vehicle report is computed from the '
             'log whose sha256 equals the sealed record, written as vehicles_report.json, re-read and compared; the '
             'RESULT.json pointer sha is checked against RESULT.json.gz; sizes and sha256 of the raw files are sealed '
             'in DROPPED_RAW.json; only then RESULT.json.gz, vehicles.jsonl.gz and observable.jsonl.gz of that episode '
             'are removed. Only episodes a75d2v_* / a75d2c_* directly under <root>/evaluation; mechanics episodes are '
             'never dropped')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    return N, A50, A67, A68


def _H():
    from tools.coopmwm import a75_holdforms as H
    return H


# ================================================================================================ the draw (pure)

def draw_rng(seed=DRAW_SEED, stream=DRAW_STREAM):
    return random.Random(int(hashlib.sha256(f'{seed}:{stream}'.encode()).hexdigest()[:16], 16))


def _slots(rng, n, lanes=(0, 1, 2), exclude=()):
    pool = [(ln, rk) for ln in lanes for rk in (0, 1) if (ln, rk) not in exclude]
    return rng.sample(pool, n)


def _plain_role(rng, lane, rank, holds=HOLD_GRID):
    return dict(lane=int(lane), rank=int(rank), v_w=float(rng.choice(V_W)), dmax=float(rng.choice(DMAX)),
                hold_s=int(rng.choice(holds)), onset_s=int(rng.choice(ONSETS)), lc_target=None)


def _gap_role(rng, rank):
    dv = rng.choice(DV_REL)
    r = dict(lane=1, rank=int(rank), v_w=float(rng.choice(((0.0,) + V_W) if dv is not None else V_W)),
             dmax=float(rng.choice(DMAX)), hold_s=int(rng.choice(GAP_HOLDS)), onset_s=int(rng.choice(ONSETS)),
             lc_target=None, start='gap', gap_dx_lo=float(rng.choice(GAP_DX_LO)), gap_dx_hi=float(rng.choice(GAP_DX_HI)),
             gap_drop_m=rng.choice(GAP_DROP_M), gap_arm_s=int(rng.choice(GAP_ARM_S)),
             gap_release=rng.choice(GAP_RELEASE), gap_events=int(rng.choice(GAP_EVENTS)))
    if dv is not None:
        r['dv_rel'] = float(dv)
    return r


def _pulse(rng, r):
    on, off = int(rng.choice(PULSE)), int(rng.choice(PULSE))
    r.update(pulse_on_s=on, pulse_off_s=off, hold_s=int(rng.choice([h for h in HOLD_GRID if h >= on+off])))


def _taper(rng, r):
    while True:
        tin, tout = int(rng.choice(TAPER)), int(rng.choice(TAPER))
        if tin or tout:
            break
    r.update(taper_in_s=tin, taper_out_s=tout, hold_s=int(rng.choice([h for h in HOLD_GRID if h >= tin+tout])))


def _onset_zero(roles):
    if all(r['onset_s'] for r in roles):
        roles[0]['onset_s'] = 0


def draw_plan(rng, label):
    """One plan of stratum ``label`` (DRAW_RULE). Pure given the rng state."""
    _, _, form, sizes = next(s for s in STRATA if s[0] == label)
    n = int(rng.choice(sizes))
    zone, zone_d = rng.choice(OLD_ZONES), None
    if form == 'gap':
        g = _gap_role(rng, rng.choice((0, 1)))
        roles = [g] + [_plain_role(rng, ln, rk) for ln, rk in _slots(rng, n-1, lanes=(1, 2),
                                                                      exclude=((1, g['rank']),))]
        zone = rng.choice(GAP_ZONES)
    else:
        roles = [_plain_role(rng, ln, rk, SHORT_HOLDS if form == 'short' else HOLD_GRID) for ln, rk in _slots(rng, n)]
    roles.sort(key=lambda r: (r['lane'], r['rank']))
    if form == 'zone':
        zone = label.split('_', 1)[1]
        if zone == 'custom':
            while True:
                d_up, d_down = rng.choice(ZONE_D_UP), rng.choice(ZONE_D_DOWN)
                if d_up - d_down >= ZONE_D_MIN_SPAN:
                    break
            zone_d = [float(d_up), float(d_down)]
    if form in ('pulse', 'taper'):
        for i, r in enumerate(roles):
            if i == 0 or rng.random() < P_EXTRA_FORM_ROLE:
                (_pulse if form == 'pulse' else _taper)(rng, r)
    if form == 'rolling':
        on = [0] + sorted(rng.sample(ONSETS[1:], n-1))
        for r, o in zip(roles, on):
            r['onset_s'] = int(o)
    else:
        _onset_zero(roles)
    hmax = max(r['hold_s'] for r in roles)
    s = dict(name='', trigger='lanes', zone=zone, spread_m=float(rng.choice(SPREADS)),
             cooldown_s=int(rng.choice((hmax, hmax+30, 60, 120))), slots=1, gate=None, start_gate=None, phases=None,
             roles=roles)
    if zone_d is not None:
        s['zone_d'] = zone_d
    return s


def draw_strategies(seed=DRAW_SEED):
    """Pure (DRAW_RULE): [(label, strategy)] in draw order."""
    H = _H()
    rng = draw_rng(seed)
    labels = [s[0] for s in STRATA for _ in range(s[1])]
    rng.shuffle(labels)
    out, seen = [], set()
    for label in labels:
        for _ in range(1000):
            s = H.canonical(draw_plan(rng, label))
            if H.executor_refusal(s) is not None:
                continue
            key = H.strategy_key(s)
            if key not in seen:
                break
        else:
            raise RuntimeError(f'draw: no new executable plan for {label}')
        seen.add(key)
        s['name'] = f'd2_{key}'
        out.append((label, s))
    if len(out) != DRAW_N:
        raise RuntimeError('draw size')
    return out


def _sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def seal_doc(doc, field='doc_sha256'):
    body = {k: v for k, v in doc.items() if k != field}
    return dict(body, **{field: _sha_text(json.dumps(body, sort_keys=True))})


def candidate_rows(drawn):
    H = _H()
    out = []
    for i, (label, s) in enumerate(drawn):
        form = next(x[2] for x in STRATA if x[0] == label)
        out.append(dict(family=s['name'], name=s['name'], key=H.strategy_key(s), draw_rank=i+1, stratum=label,
                        form=form, n_roles=len(s['roles']), extended=bool(H.is_extended(s)),
                        hold_s=sorted({r['hold_s'] for r in s['roles']}), trigger=s['trigger'], zone=s['zone'],
                        zone_d=s.get('zone_d'), strategy=s,
                        strategy_sha256=hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()))
    return out


def candidates_doc():
    """Pure: the sealed candidate document (no timestamps)."""
    H = _H()
    cands = candidate_rows(draw_strategies())
    bank = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    got, refused = H.a75_bank_plans(bank, CELL)
    if len(got) != len(cands) or refused:
        raise RuntimeError(f'a75_bank_plans accepts {len(got)} of {len(cands)} (refused {refused})')

    def count(f):
        return dict(sorted(Counter(str(f(c)) for c in cands).items()))
    return seal_doc(dict(
        schema='coopmwm.a75d2.candidates.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, rule=DRAW_RULE,
        executor=dict(module='tools/coopmwm/a75_holdforms.py', a75_sha256=A75_SHA256),
        draw=dict(seed=DRAW_SEED, stream=DRAW_STREAM, n=DRAW_N, generator='python random.Random',
                  strata={s[0]: dict(n=s[1], form=s[2], sizes=list(s[3])) for s in STRATA},
                  hold_grid=list(HOLD_GRID), short_holds=list(SHORT_HOLDS), v_w=list(V_W), dmax=list(DMAX),
                  onset_s=list(ONSETS), old_zones=list(OLD_ZONES), gap_zones=list(GAP_ZONES),
                  zone_d_up=list(ZONE_D_UP), zone_d_down=list(ZONE_D_DOWN), spread_m=list(SPREADS),
                  gap_dx_lo=list(GAP_DX_LO), gap_dx_hi=list(GAP_DX_HI), gap_drop_m=list(GAP_DROP_M),
                  gap_arm_s=list(GAP_ARM_S), gap_release=list(GAP_RELEASE), gap_events=list(GAP_EVENTS),
                  gap_hold_s=list(GAP_HOLDS), dv_rel=list(DV_REL), taper=list(TAPER), pulse=list(PULSE),
                  p_extra_form_role=P_EXTRA_FORM_ROLE),
        world_model='none (direct SUMO test; no WM score, no WM rank)',
        n_candidates=len(cands), candidates_by_form=count(lambda c: c['form']),
        candidates_by_stratum=count(lambda c: c['stratum']), candidates_by_size=count(lambda c: c['n_roles']),
        candidates_by_zone=count(lambda c: c['zone']), STOP=not cands, candidates=cands))


CSV_FIELDS = ('draw_rank', 'family', 'form', 'stratum', 'n_roles', 'zone', 'zone_d', 'spread_m', 'cooldown_s', 'roles')


def _role_text(r):
    base = f'L{r["lane"]}r{r["rank"]} v_w={r["v_w"]:g} dmax={r["dmax"]:g} hold={r["hold_s"]} onset={r["onset_s"]}'
    extra = ' '.join(f'{k}={r[k]}' for k in sorted(r) if k not in ('lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s',
                                                                      'lc_target'))
    return base + (' ' + extra if extra else '')


def csv_text(doc):
    buf = io.StringIO(newline='')
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_FIELDS)
    for c in doc['candidates']:
        s = c['strategy']
        w.writerow([c['draw_rank'], c['family'], c['form'], c['stratum'], c['n_roles'], s['zone'],
                    '' if s.get('zone_d') is None else '/'.join(f'{x:g}' for x in s['zone_d']), f'{s["spread_m"]:g}',
                    s['cooldown_s'], ' | '.join(_role_text(r) for r in s['roles'])])
    return buf.getvalue()


def draw(out=None):
    """Zero SUMO: write the candidate JSON + CSV + pin (never overwrites)."""
    out = Path(out or CANDS_FILE)
    doc = candidates_doc()
    text = json.dumps(doc, indent=1, sort_keys=True)+'\n'
    csvp, pin = out.with_suffix('.csv'), out.with_suffix('.sha256')
    for p in (out, csvp, pin):
        if p.exists():
            raise SystemExit(f'{p}: exists (pre-outcome file; never overwrite)')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode('utf-8'))
    csvp.write_bytes(csv_text(doc).encode('utf-8'))
    pin.write_bytes(f'{doc["doc_sha256"]}  doc_sha256 {out.name}\n{_sha_text(text)}  file_sha256 {out.name}\n'
                    f'{_sha_text(csv_text(doc))}  file_sha256 {csvp.name}\n'.encode('utf-8'))
    return doc


def candidates(path=None):
    """The pinned pre-outcome candidates; refuses an unpinned / edited file or one that is not the seeded draw."""
    N = _mods()[0]
    H = _H()
    path = Path(path or CANDS_FILE)
    pin_file = path.with_suffix('.sha256')
    if not path.is_file() or not pin_file.is_file():
        raise N.Blocked(f'{path} / {pin_file.name}: missing (run draw, record the doc sha)')
    doc = json.loads(path.read_text(encoding='utf-8'))
    pin = pin_file.read_text(encoding='utf-8').split()[0]
    if seal_doc(doc)['doc_sha256'] != doc.get('doc_sha256') or doc['doc_sha256'] != pin:
        raise N.Blocked(f'{path.name}: not the registered draw (pin {pin})')
    if doc != candidates_doc():
        raise N.Blocked(f'{path.name}: the candidates are not the seeded draw (hand-edited / other python?)')
    if doc['cell'] != CELL or doc['STOP'] or doc['n_candidates'] != DRAW_N:
        raise N.Blocked(f'{path.name}: not the registered cell / size')
    for c in doc['candidates']:
        if H.executor_refusal(c['strategy']) is not None or H.strategy_key(c['strategy']) != c['key']:
            raise N.Blocked(f'{c["family"]}: not an executable A75 plan')
    return doc['candidates']


def examples():
    """The six EXAMPLE_PLANS strategies + their bank doc (re-built with a75_holdforms.bank_doc, must equal the file)."""
    N = _mods()[0]
    H = _H()
    d = json.loads(EXAMPLES.read_text(encoding='utf-8'))
    strat = {s['name']: s for s in d['strategies']}
    if tuple(strat) != EXAMPLE_FAMILIES or d.get('a75_digest') != A75_SHA256:
        raise N.Blocked('EXAMPLE_PLANS.json: not the registered six examples / digest')
    bank = d['bank']
    got, refused = H.a75_bank_plans(bank, CELL)
    keys = {p['family']: H.strategy_key(p['spec']) for p in got}
    if refused or keys != {f: H.strategy_key(H.canonical(strat[f])) for f in EXAMPLE_FAMILIES}:
        raise N.Blocked(f'EXAMPLE_PLANS.json: the shipped bank does not execute the six examples ({refused})')
    return strat, bank


# ================================================================================================ the bank rule (pure)

def bank_choice(rows):
    """Pure (BANK_RULE). rows: verdict rows with family, key, draw_rank, form, PASS, ci95, per_seed.
    -> (chosen families best first, {family: duplicate_of}, per-form table)."""
    order = sorted(rows, key=lambda r: (r['draw_rank'], r['key']))
    first, dup = {}, {}
    for r in order:
        sig = json.dumps(sorted((str(k), float(v)) for k, v in r['per_seed'].items()))
        if sig in first:
            dup[r['family']] = first[sig]
        else:
            first[sig] = r['family']
    if any(r['form'] not in FORMS for r in rows):
        raise ValueError('a plan outside the registered form families')
    best, table = [], {}
    for f in FORMS:
        of = [r for r in rows if r['form'] == f]
        ok = sorted((r for r in of if r['PASS'] and r['family'] not in dup), key=lambda r: (r['ci95'][1], r['key']))
        table[f] = dict(tested=len(of), delay_test_pass=sum(bool(r['ci95'][1] < 0) for r in of),
                        admitted=sum(bool(r['PASS']) for r in of), admitted_non_duplicate=len(ok),
                        duplicates=sum(r['family'] in dup for r in of), best=ok[0]['family'] if ok else None,
                        best_ci95_upper=ok[0]['ci95'][1] if ok else None)
        if ok:
            best.append(ok[0])
    best.sort(key=lambda r: (r['ci95'][1], r['key']))
    chosen = [r['family'] for r in best[:MAX_BANK]]
    for f in FORMS:
        table[f]['in_bank'] = table[f]['best'] in chosen
    return chosen, dup, table


def bank_plans(cands, rows, chosen):
    """The bank rows (BANK_RULE order: lb95(gain) = -CI95 upper desc, then a75 strategy key)."""
    H = _H()
    v = {r['family']: r for r in rows}
    out = [dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=float(-v[c['family']]['ci95'][1]))
           for c in cands if c['family'] in chosen]
    out.sort(key=lambda p: (-p['lb95_veh_s'], H.strategy_key(p['strategy'])))
    return out


# ================================================================================================ seeds (pure)

def mech_seed0(env=None):
    """The coordinator-assigned mechanics block start (A75D2_MECH_SEED0); refuses when unset / malformed."""
    v = (os.environ if env is None else env).get(SEED_ENV)
    if v is None or not str(v).strip():
        raise SystemExit(f'{SEED_ENV} is unset: the mechanics seed block is assigned by the coordinator at '
                         'registration; nothing runs')
    s0 = int(str(v).strip())
    if not 900000 <= s0 <= 999900 or s0 % 100:
        raise SystemExit(f'{SEED_ENV}={s0}: must be a 100-seed block start (multiple of 100, 900000-999900)')
    lo, hi = s0, s0+99
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise SystemExit(f'{SEED_ENV}={s0}: forbidden seed range')
    return s0


def check_block_free(s0, table):
    """table: runner EVAL_HELDOUT_RANGES rows (lo, hi, base). The block must be its own row and overlap no other."""
    lo, hi = s0, s0+99
    own = [r for r in table if (r[0], r[1]) == (lo, hi)]
    other = [r for r in table if (r[0], r[1]) != (lo, hi) and lo <= r[1] and hi >= r[0]]
    if len(own) != 1 or other:
        raise ValueError(f'mechanics block {lo}-{hi}: own rows {own}, overlapping {other}')
    return True


# ================================================================================================ mechanics checks (pure)

def _acts(rec):
    return list((rec or {}).get('acts') or [])


def _sessions_of(rec):
    """[(family, vid, role index, t0, t1, why)] from the executor acts (t1 = act t + release elapsed; open: None)."""
    out = []
    for act in _acts(rec):
        for i, vid in enumerate(act['members']):
            rel = (act.get('release') or {}).get(vid)
            out.append((act.get('family', act.get('strategy')), vid, i, int(act['t']),
                        None if rel is None else int(act['t'])+int(rel[0]), None if rel is None else rel[1]))
    return out


def check_short(rec, role):
    """Plain short hold: a member released hold_end was held exactly onset_s + hold_s seconds (the executor issues
    one command per second from onset to the end; the per-second list is not in the receipts)."""
    errs = [] if int(rec.get('activations') or 0) else ['no activation']
    for fam, vid, i, t0, t1, why in _sessions_of(rec):
        if why not in ('hold_end', 'at_drop', 'left'):
            errs.append(f'{vid}: release {why}')
        if why == 'hold_end' and t1 - t0 != int(role['onset_s']) + int(role['hold_s']):
            errs.append(f'{vid}: held {t1-t0} s != onset + hold_s {int(role["onset_s"]) + int(role["hold_s"])}')
    return errs


def check_gap(rec, role, horizon=HORIZON):
    """An event / session still running at the episode end is allowed only if it started inside the last
    onset_s + gap_arm_s + hold_s + 1 seconds (it could not have ended)."""
    late = horizon - (int(role['onset_s']) + int(role.get('gap_arm_s') or 0) + int(role['hold_s']) + 1)
    errs = []
    summ = rec.get('holdform_summary') or {}
    if not int(summ.get('gap_fired') or 0):
        errs.append('gap_fired = 0')
    per = Counter()
    for e in rec.get('holdforms') or []:
        if e['start_kind'] != 'gap':
            continue
        per[(e['vid'], e['armed_t'])] += 1
        if e.get('end_t') is None and int(e['fire_t']) >= horizon - int(role['hold_s']) - 1:
            pass
        elif e.get('end_t') is None or e.get('end_why') not in ('clear', 'timer', 'at_drop', 'left', 'hold_end',
                                                               'gap_arm_end'):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: end {e.get("end_t")} {e.get("end_why")}')
        if int(e['issued_s']) > int(role['hold_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: issued {e["issued_s"]} > hold_s')
        if e.get('trigger_vid') is None or not float(role['gap_dx_lo']) <= float(e['dx_m']) <= float(role['gap_dx_hi']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: trigger / dx {e.get("trigger_vid")} {e.get("dx_m")} outside window')
    if any(n > int(role.get('gap_events', 1)) for n in per.values()):
        errs.append('more events than gap_events in a session')
    if any(x.get('released') is None and int(x['start']) < late for x in rec.get('holdform_sessions') or []):
        errs.append('a session never released')
    return errs


def check_zone(rec, bounds, pos, tol=1.0):
    """pos(vid, t) -> SUMO x of the member at the activation second (from the vehicle log) or None."""
    errs = [] if int(rec.get('activations') or 0) else ['no activation']
    for fam, vid, i, t0, t1, why in _sessions_of(rec):
        x = pos(vid, t0)
        if x is None:
            errs.append(f'{vid}@{t0}: no position in the vehicle log')
        elif not bounds[0] - tol <= x <= bounds[1] + tol:
            errs.append(f'{vid}@{t0}: activation x {x:.1f} outside {bounds}')
    return errs


def _events(rec):
    return [e for e in rec.get('holdforms') or [] if e.get('end_t') is not None]


def check_pulse(rec, role):
    H = _H()
    ev = _events(rec)
    errs = [] if ev else ['no completed event']
    for e in ev:
        want = sum(bool(H.pulse_on(role, h)) for h in range(int(e['end_t'])-int(e['fire_t'])))
        if int(e['issued_s']) != want:
            errs.append(f'{e["vid"]}@{e["fire_t"]}: issued {e["issued_s"]} != pulse count {want}')
    return errs


def check_taper(rec, role):
    ev = _events(rec)
    errs = [] if ev else ['no completed event']
    for e in ev:
        if int(e['issued_s']) != int(e['end_t'])-int(e['fire_t']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: issued {e["issued_s"]} != duration')
        if e['end_why'] == 'hold_end' and int(e['end_t'])-int(e['fire_t']) != int(role['hold_s']):
            errs.append(f'{e["vid"]}@{e["fire_t"]}: timed hold lasted {int(e["end_t"])-int(e["fire_t"])} s')
    return errs


def check_rolling(rec, roles):
    """Plain rolling plan: member i (role i) released hold_end exactly at its own onset_s + hold_s."""
    errs = [] if int(rec.get('activations') or 0) else ['no activation']
    for fam, vid, i, t0, t1, why in _sessions_of(rec):
        r = roles[i]
        if why == 'hold_end' and t1 - t0 != int(r['onset_s']) + int(r['hold_s']):
            errs.append(f'{vid}: role {i} held {t1-t0} s != onset {r["onset_s"]} + hold {r["hold_s"]}')
    return errs


def check_bank(rec):
    errs = [] if int(rec.get('activations') or 0) else ['no activation']
    by = {}
    for fam, vid, i, t0, t1, _ in _sessions_of(rec):
        by.setdefault(vid, []).append((t0, 10**9 if t1 is None else t1, fam))
    for vid, xs in by.items():
        xs.sort()
        for a, b in zip(xs, xs[1:]):
            if b[0] < a[1]:
                errs.append(f'{vid}: in {a[2]} and {b[2]} at once')
    return errs


IDENT_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts')


def ident_row(r37, r75, log37, log75, stored):
    """One identity world: A37_PLAN record, A75_PLAN record, decompressed log shas, stored A68 record."""
    a, b = r37.get('a37') or {}, r75.get('a75') or {}
    row = dict(world_id=r37['world_id'], j_a37=r37['J_veh_s'], j_a75=r75['J_veh_s'],
               j_equal=r37['J_veh_s'] == r75['J_veh_s'], safety_equal=r37['safety'] == r75['safety'],
               log_equal=log37 == log75 and log37 is not None,
               receipts_equal={k: json.dumps(a.get(k), sort_keys=True) == json.dumps(b.get(k), sort_keys=True)
                               for k in IDENT_FIELDS},
               a75_extras_empty=(b.get('extended') == [] and b.get('holdforms') == []),
               stored_j_equal=stored['J_veh_s'] == r37['J_veh_s'], stored_safety_equal=stored['safety'] == r37['safety'],
               stored_activations_equal=int((stored.get('a37') or {}).get('activations') or 0)
               == int(a.get('activations') or 0),
               stored_vehicle_log_equal=(stored.get('vehicle_log') or {}).get('sha256')
               == (r37.get('vehicle_log') or {}).get('sha256'))
    row['PASS'] = bool(row['j_equal'] and row['safety_equal'] and row['log_equal'] and all(row['receipts_equal'].values())
                       and row['a75_extras_empty'] and row['stored_j_equal'] and row['stored_safety_equal']
                       and row['stored_activations_equal'])
    return row


def read_vlog(path):
    """(header {vid: veh}, ticks {t: {veh: (s, v, a, lane)}}) of one vehicles.jsonl.gz."""
    ids, ticks = {}, {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if 'veh' in d:
                ids[d['id']] = d['veh']
            else:
                ticks[int(d['t'])] = {k: (s, v, a, ln) for k, s, v, a, ln in zip(d['k'], d['s'], d['v'], d['a'],
                                                                                 d['lane'])}
    return ids, ticks


def vlog_pos(vlog):
    """pos(vid, t) -> SUMO x (= logged WM s + oracle_plan.S_OFF) at second t, or None."""
    from tools.coopmwm import oracle_plan as O
    ids, ticks = vlog

    def pos(vid, t):
        k = ids.get(str(vid))
        x = (ticks.get(int(t)) or {}).get(k)
        return None if x is None else float(x[0]) + float(O.S_OFF)
    return pos


def gap_rederive(rec, vlog):
    """Descriptive: per fired gap event, the trigger vehicle lane at fire_t, its lane changes within 20 s, the
    member speed 5 s after fire vs v0 (vehicle-log lanes are logged as the plant frame gives them)."""
    ids, ticks = vlog
    out = []
    for e in rec.get('holdforms') or []:
        if e['start_kind'] != 'gap':
            continue
        t, tv, mv = int(e['fire_t']), ids.get(str(e.get('trigger_vid'))), ids.get(str(e['vid']))
        lane0 = (ticks.get(t) or {}).get(tv, (None,)*4)[3]
        lanes = [(ticks.get(t+k) or {}).get(tv, (None,)*4)[3] for k in range(1, 21)]
        v5 = (ticks.get(t+5) or {}).get(mv, (None,)*4)[1]
        out.append(dict(vid=e['vid'], fire_t=t, trigger_vid=e.get('trigger_vid'), trigger_lane_at_fire=lane0,
                        trigger_lane_changed_20s=any(x is not None and lane0 is not None and x != lane0 for x in lanes),
                        member_v_fire_plus5=v5, v0=e['v0'],
                        slowed=None if v5 is None else bool(v5 < float(e['v0']))))
    return out


# ================================================================================================ host-side driver

def core(which):
    """The a42 driver code (as D1 uses it) with the D2 names: own phase / caps / prefixes / bank dir."""
    N, A50 = _mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A75D2',
        HOST_BASE=os.environ.get('A75D2_HOST_BASE', '/root/a75d2c'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A75D2_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A75D2', JOB_PREFIX='a75d2',
        VERIFY_TAG='a75d2v', CONFIRM_TAG='a75d2c', IDENT_TAG='a75d2i',
        VERIFY_WAVE_PREFIX='A75D2V_', CONFIRM_WAVE_PREFIX='A75D2C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE=LOG_FILE, SCHEMA_PREFIX='coopmwm.a75d2',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 75 stage D2; POST-HOC exploratory direct SUMO test of randomly drawn NEW hold forms '
                     'on MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a75d2_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = check_d2
    return m


def check_d2(record, spec):
    """A75 arms: the A37 record check + a75 sha / receipts; every other arm: a37_celltailor.check_a37."""
    from copy import deepcopy
    import a37_celltailor as C37
    import a32_ablation as A
    N = _mods()[0]
    if spec['arm'] not in ('A75_PLAN', 'A75_BANK'):
        return C37.check_a37(record, spec)
    what = f'{spec["arm"]} {spec["world_id"]}'
    out = A._ORIG_CHECK(record, spec)
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise N.Blocked(f'{what}: executor actually loaded differs from its binding (never silent)')
    if record.get('a75_sha256_loaded') != spec['checkpoint']['a75_sha256'] or spec['checkpoint']['a75_sha256'] != A75_SHA256:
        raise N.Blocked(f'{what}: a75_holdforms actually loaded differs from its registration')
    r = record.get('a75')
    if not isinstance(r, dict) or r.get('a75_arm') != spec['arm'] or 'holdform_summary' not in r:
        raise N.Blocked(f'{what}: a75 receipts missing')
    return dict(out, executor_sha256_loaded=record['executor_sha256_loaded'],
                a75_sha256_loaded=record['a75_sha256_loaded'], a75=deepcopy(r))


def register_mech_setting(root, s0):
    """Process-local nm_finaleval setting for the mechanics worlds S0..S0+3 (nm_finaleval itself unchanged)."""
    N = _mods()[0]
    name = SETTINGS['m']
    if name not in N.SETTINGS:
        N.SETTINGS[name] = dict(
            N.SETTINGS[SETTINGS['v']], N_WORLDS=MECH_N_WORLDS, EVAL_RANGE=(s0, s0+99), EVAL_EXTRA_RANGE=(s0, s0+99),
            SEEDS_RESERVED=(s0+MECH_N_WORLDS, s0+99),
            PANEL_SIZE_REASON=f'MAIN amendment 75 D2: executor mechanics = {MECH_N_WORLDS} fresh worlds {s0}-'
                              f'{s0+MECH_N_WORLDS-1} (D2800_p60); never evaluation, outcomes not used',
            CELL_AUTHORITY='MAIN amendment 75 stage D2 (S-HOLDFORMS-D2, 2026-11-13): hold-form executor mechanics',
            PANEL_AUTHORITY=(f'A75 D2: coordinator-assigned audited integers {s0}-{s0+99} (runner A75D2_MECH_RANGE) '
                             'for executor mechanics ONLY; executed through the eval runner (scope label only)'),
            EXPOSURE_EXTRA=dict(checked=('amendment-75 D2 audit: ops/a75d2/SEED_AUDIT.json + coordinator registration',),
                                disclosed='; amendment 75 D2: fresh mechanics integers, registered at runtime'),
            PANEL_FILE=Path(root)/PANEL_M, PANEL_3C_FILE=None)
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
        N.SETTINGS_PINS[name] = dict(PANEL_SHA256=None, PANEL_3C_SHA256=None)
    return name


def setup(which, root):
    """-> (D, panel, Dref)."""
    N, _, _, A68 = _mods()
    if which == 'm':
        s0 = mech_seed0()
        register_mech_setting(root, s0)
        from tools.coopmwm import nm_eval_runner as R
        if tuple(getattr(R, 'A75D2_MECH_RANGE', ())) != (s0, s0+99):
            raise N.Blocked(f'runner A75D2_MECH_RANGE is not {s0}-{s0+99} (ops/a75d2/host/patch_tables_d2.py)')
        check_block_free(s0, R.EVAL_HELDOUT_RANGES)
        D = core('m')
        p = Path(root)/PANEL_M
        if not p.is_file():
            return D, None, None
        pin = (Path(root)/PANEL_M_PIN).read_text(encoding='utf-8').split()[0]
        if N.file_sha(p) != pin:
            raise N.Blocked(f'{PANEL_M}: not the pinned file')
        panel = D.setup(SETTINGS['m'])
        seeds = [r['world']['seed'] for r in panel['worlds']]
        if seeds != list(range(s0, s0+MECH_N_WORLDS)) or panel['cell']['cell_key'] != CELL:
            raise N.Blocked(f'{PANEL_M}: not {CELL} {s0}-{s0+MECH_N_WORLDS-1}')
        return D, panel, None
    Dref, pref = (A68.setup('v') if which in 'iv' else _a70().setup('c'))
    D = core(which)
    panel = D.setup(SETTINGS[which])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if (seeds != SEEDS[which] or panel['cell']['cell_key'] != CELL or panel['panel_sha256'] != pref['panel_sha256']
            or panel['panel_sha256'] != PANEL_SHA256[which]):
        raise N.Blocked(f'{SETTINGS[which]}: not the registered panel')
    return D, panel, Dref


def _a70():
    import a70_smallbank as A70
    return A70


def build_panel_m(root):
    N = _mods()[0]
    from datetime import datetime, timezone
    import a37_celltailor as C37
    s0 = mech_seed0()
    D, panel, _ = setup('m', root)
    if panel is not None:
        raise N.Blocked(f'{PANEL_M} exists')
    N.apply_setting(SETTINGS['m'])
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    value = N.build_panel(N.default_spec_path(), datetime.now(timezone.utc).isoformat())
    if [w['world']['seed'] for w in value['worlds']] != list(range(s0, s0+MECH_N_WORLDS)):
        raise N.Blocked('mechanics panel: wrong seeds')
    p = C37.write_once(Path(root)/PANEL_M, json.dumps(value, indent=1, sort_keys=True)+'\n')
    C37.write_once(Path(root)/PANEL_M_PIN, f'{N.file_sha(p)}  {PANEL_M} panel_sha256 {value["panel_sha256"]}\n')
    return dict(file=str(p), file_sha256=N.file_sha(p), panel_sha256=value['panel_sha256'], seeds=[s0, s0+3])


def a75_from_a37(specs, arm, entry, *, family, tag):
    """A37 executor specs already bound to MAPPO60C (A67.with_exec) -> A75 specs (ops/a75/a75_specs.a75_specs after
    dropping the a37_* keys); only the executor arm / digest / bank change."""
    from copy import deepcopy
    sys.path.insert(0, str(HERE/'a75'))
    import a75_specs as S
    base = []
    for s in specs:
        ck = deepcopy(s['checkpoint'])
        for k in ('a37_arm', 'a37_sha256', 'a37_bank', 'a37_family'):
            ck.pop(k, None)
        base.append(dict(s, checkpoint=ck))
    return S.a75_specs(base, arm, entry, family=family, tag=tag, digest=A75_SHA256)


def prepare(D, root, which, a68_root=None):
    """Write-once bank files of the stage (candidate bank / example bank / A68 candidate bank copy)."""
    N = _mods()[0]
    H = _H()
    out = {}
    if which == 'v':
        cands = candidates()
        doc = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
        p = D.C37.write_once(D.bank_path(root, CANDS_BANK), D.C37.bank_text(doc))
        out[CANDS_BANK] = N.file_sha(p)
    if which == 'm':
        _, bank = examples()
        p = D.C37.write_once(D.bank_path(root, EX_BANK), D.C37.bank_text(bank))
        out[EX_BANK] = N.file_sha(p)
    if which == 'i':
        src = Path(a68_root)/'a68_banks'/A68_BANK
        q = D.C37.write_once(D.bank_path(root, A68_BANK), src.read_bytes().decode('utf-8'))
        if N.file_sha(q) != N.file_sha(src):
            raise N.Blocked(f'{q}: not the A68 candidate bank bytes')
        out[A68_BANK] = N.file_sha(q)
    return out


def _plan_specs(D, panel, root, a66_root, bank, arm, family, tag):
    N, _, A67, A68 = _mods()
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, bank)
    base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family=family or '_', tag=tag), exe,
                         tag)
    if arm == 'A37_PLAN':
        return base
    return a75_from_a37(base, arm, entry, family=family, tag=tag)


def ident_waves(D, panel, root, a66_root):
    A68 = _mods()[3]
    fam = A68.candidates()[0]['family']
    return [('A75D2_I_A37', 'episode', _plan_specs(D, panel, root, a66_root, A68_BANK, 'A37_PLAN', fam,
                                                    PREFIX['i']+'a37')[:IDENT_N]),
            ('A75D2_I_A75', 'episode', _plan_specs(D, panel, root, a66_root, A68_BANK, 'A75_PLAN', fam,
                                                    PREFIX['i']+'a75')[:IDENT_N])]


#: forms stage: (wave, family or None for the bank, number of worlds)
FORM_WAVES = (('A75D2_F1_SHORT', 'ex_short', 2), ('A75D2_F2_GAP', 'ex_gap', 4), ('A75D2_F3_ZONE', 'ex_zone', 2),
              ('A75D2_F4_PULSE', 'ex_pulse', 2), ('A75D2_F5_TAPER', 'ex_taper', 2), ('A75D2_F6_ROLLING', 'ex_rolling', 2),
              ('A75D2_F7_BANK', None, 2))
assert 2*IDENT_N + sum(n for _, _, n in FORM_WAVES) == MECH_EPISODES


def form_waves(D, panel, root, a66_root):
    out = []
    for wave, fam, n in FORM_WAVES:
        tag = PREFIX['m'] + wave.split('_', 2)[2].lower()
        arm = 'A75_BANK' if fam is None else 'A75_PLAN'
        out.append((wave, 'episode', _plan_specs(D, panel, root, a66_root, EX_BANK, arm, fam, tag)[:n]))
    return out


def verify_waves(D, panel, root, a66_root, cands):
    N = _mods()[0]
    waves = [(f'A75D2V_{c["family"]}', 'episode',
              _plan_specs(D, panel, root, a66_root, CANDS_BANK, 'A75_PLAN', c['family'], f'{PREFIX["v"]}{c["key"]}'))
             for c in cands]
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def confirm_waves(D, panel, root, a66_root):
    N, _, A67, A68 = _mods()
    adm = admission(root)
    if adm is None or not adm['bank']:
        raise N.Blocked('no admission receipt / no plan admitted: no confirmation')
    exe = A68.executor(a66_root)
    entry = A68.bank_entry(D, root, D2_BANK, adm['bank']['file_sha256'])
    base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family='_', tag=PREFIX['c']+'bank'),
                         exe, PREFIX['c']+'bank')
    specs = a75_from_a37(base, 'A75_BANK', entry, family=None, tag=PREFIX['c']+'bank')
    if len(specs) != CONFIRM_N:
        raise N.Blocked(f'{ARM}: {len(specs)} specs')
    return [(CONFIRM_WAVE, 'episode', specs)]


def _load(root, wave, specs):
    N = _mods()[0]
    recs, src, cov = N.published(root, wave, specs, check_d2)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {root}')
    return {r['world_id']: r for r in recs}, src


def ref_pins():
    N = _mods()[0]
    v = N.verify(N.read(REF_PINS), 'receipt_sha256', REF_PINS.name)
    if v['receipt_sha256'] != REF_PINS_RECEIPT:
        raise N.Blocked('REF_PINS.json: not the D1 reference pins 79815fe9')
    return v


def a68_ref(D68, panel, a68_root, a66_root):
    N, _, _, A68 = _mods()
    have, src = D68.C37._load(a68_root, 'A68V_MAPPO60C',
                              A68.mappo_specs(D68, panel, a68_root, A68.executor(a66_root), 'a68v_mappo60c'))
    got = src['receipt_sha256']
    if (got if isinstance(got, list) else [got]) != ref_pins()['a68v_mappo60c']['wave_receipt_sha256']:
        raise N.Blocked('A68V_MAPPO60C: not the pinned wave receipt')
    return have, src


def pinned_reports(reports_root, have, worlds, pin):
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


def _log_sha(root, rec):
    N = _mods()[0]
    p = Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
    if not p.is_file():
        return None
    with gzip.open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def mech_compare(root, a68_root, a66_root):
    """Zero SUMO: MECH_RULE over the published mechanics waves -> sealed verdict (PASS / FAIL)."""
    N, _, A67, A68 = _mods()
    root = Path(root)
    strat, _ = examples()
    rows, fails = dict(identity=[], forms={}), []
    # ---- identity (setting v; the A68 reference driver binds the stored records)
    Di, panel_i, D68 = setup('i', root)
    (w37, _, s37), (w75, _, s75) = ident_waves(Di, panel_i, root, a66_root)
    r37, _ = _load(root, w37, s37)
    r75, _ = _load(root, w75, s75)
    c = A68.candidates()[0]
    w68 = {n: s for n, _, s in A68.verify_waves(D68, panel_i, a68_root, a66_root, A68.candidates())}
    stored, _ = D68.C37._load(a68_root, f'A68V_{c["family"]}', w68[f'A68V_{c["family"]}'])
    safety = []
    for w in r37:
        row = ident_row(r37[w], r75[w], _log_sha(root, r37[w]), _log_sha(root, r75[w]), stored[w])
        row['executor_is_mappo60c'] = (r37[w].get('executor_sha256_loaded') == r75[w].get('executor_sha256_loaded')
                                       == A67.MAPPO60C_SHA256)
        row['PASS'] = row['PASS'] and row['executor_is_mappo60c']
        rows['identity'].append(row)
        safety += [(w37, w, r37[w]['safety']), (w75, w, r75[w]['safety'])]
        if not row['PASS']:
            fails.append(f'identity {w}')
    if len(rows['identity']) != IDENT_N:
        fails.append('identity: not 2 worlds')
    return rows, fails, safety, strat


def mech_compare_forms(root, a66_root, rows, fails, safety, strat):
    """Second process (setting a75d2m): the form checks; seals A75D2_MECH.json."""
    N = _mods()[0]
    H = _H()
    root = Path(root)
    D, panel, _ = setup('m', root)
    if panel is None:
        raise N.Blocked(f'{PANEL_M} missing')
    for wave, fam, specs in [(w, f, s) for (w, _, s), (_, f, _) in zip(form_waves(D, panel, root, a66_root),
                                                                       FORM_WAVES)]:
        have, _ = _load(root, wave, specs)
        out = []
        for w, rec in have.items():
            r = rec['a75']
            safety.append((wave, w, rec['safety']))
            errs = []
            if r.get('refused') not in (None, {}, []):
                errs.append(f'refused {r.get("refused")}')
            desc = {}
            if fam is None:
                errs += check_bank(r)
                desc['families_activated'] = sorted((r.get('activations_by_family') or {}).keys())
            else:
                s = strat[fam]
                role = s['roles'][0]
                p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
                vl = read_vlog(p) if fam in ('ex_gap', 'ex_zone') and p.is_file() else ({}, {})
                errs += dict(ex_short=lambda: check_short(r, role), ex_gap=lambda: check_gap(r, role),
                             ex_zone=lambda: check_zone(r, H.zone_bounds(s), vlog_pos(vl)),
                             ex_pulse=lambda: check_pulse(r, role), ex_taper=lambda: check_taper(r, role),
                             ex_rolling=lambda: check_rolling(r, s['roles']))[fam]()
                if fam == 'ex_gap':
                    desc = dict(gap_events=gap_rederive(r, vl))
            out.append(dict(world_id=w, activations=int(r.get('activations') or 0),
                            holdform_summary=r.get('holdform_summary'), errors=errs, PASS=not errs, descriptive=desc,
                            hard_brakes=rec['safety'].get('hard_brakes')))
            if errs:
                fails.append(f'{wave} {w}: {errs[:3]}')
        rows['forms'][wave] = out
    for wave, w, sf in safety:
        if any(int(sf.get(k) or 0) for k in SAFETY_ZERO):
            fails.append(f'{wave} {w}: collisions/teleports {[sf.get(k) for k in SAFETY_ZERO]}')
    return N.seal(dict(schema='coopmwm.a75d2.mech.v1', amendment=AMENDMENT, stage=STAGE, rule=MECH_RULE,
                       a75_sha256=A75_SHA256, seed_block=[mech_seed0(), mech_seed0()+99],
                       panel_m_sha256=N.file_sha(root/PANEL_M), verdict='PASS' if not fails else 'FAIL',
                       failures=fails, identity=rows['identity'], forms=rows['forms'],
                       deviation=('F2 runs on 4 D2800_p60 worlds instead of 2 x D2800_p60 + 2 x D2800_p20: the D2 root '
                                  'carries only the p60 plant / MAPPO60C binding; identity runs on the first 2 A68 '
                                  'admission worlds (stored A68 record also bound) instead of fresh worlds'),
                       consequence='PASS: verify / admit / confirm may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _mods()[0]
    p = Path(root)/MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ drop raw (D1 rule)

def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _report_job(args):
    N = _mods()[0]
    root, rec = args
    rep = N.episode_log_report(root, rec)
    return rec['job_id'], (bool(rep and rep.get('ok')),
                           _gz_content(Path(root)/'evaluation'/rec.get('work_dir', rec['job_id'])/'RESULT.json.gz'))


def _gz_content(path):
    """-> [sha256, bytes] of the DECOMPRESSED content (what the runner's RESULT.json pointer records), or None
    (D1 fix 498f273c)."""
    import zlib
    if not Path(path).is_file():
        return None
    h, n = hashlib.sha256(), 0
    try:
        with gzip.open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
                n += len(block)
    except (OSError, EOFError, zlib.error):
        return None
    return [h.hexdigest(), n]


def pointer_ok(ptr, content, files):
    """Pure: the RESULT.json pointer (coopmwm-lossless-gzip-v1) records sha256 + size of the DECOMPRESSED result."""
    return (ptr.get('file') == 'RESULT.json.gz' and ptr.get('encoding') == 'coopmwm-lossless-gzip-v1'
            and content is not None and 'RESULT.json.gz' in files
            and content == [ptr.get('sha256'), ptr.get('uncompressed_bytes')])


def drop_raw(root, waves, *, procs=1, log=None):
    """DROP_RULE for every PUBLISHED a75d2v_/a75d2c_ episode of ``waves`` that still has raw files. Idempotent."""
    N = _mods()[0]
    root = Path(root)
    base = (root/'evaluation').resolve()
    todo = []
    for name, _, specs in waves:
        recs, _, cov = N.published(root, name, specs, check_d2)
        for r in recs[:cov]:
            job = r.get('work_dir', r['job_id'])
            d = root/'evaluation'/job
            if not r['job_id'].startswith((PREFIX['v'], PREFIX['c'])) or d.resolve().parent != base or d.is_symlink():
                raise N.Blocked(f'{job}: not an A75 D2 verification / confirmation episode directory')
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
        ok, content = done.get(r['job_id']) or (False, None)
        if ok is not True:
            raise N.Blocked(f'{job}: no valid vehicle-log report; raw files kept')
        cache = d/'vehicles_report.json'
        c = N.read(cache)
        log_sha = _sha(d/N.VEHICLE_LOG)
        if (c.get('log_sha256') != r['vehicle_log']['sha256'] or log_sha != r['vehicle_log']['sha256']
                or c.get('version') != N.REPORT_VERSION or not c['report'].get('ok')):
            raise N.Blocked(f'{job}: cached report does not match the sealed record; raw files kept')
        ptr = N.read(d/'RESULT.json')
        files = [dict(file=f, bytes=(d/f).stat().st_size, sha256=log_sha if f == N.VEHICLE_LOG else _sha(d/f))
                 for f in RAW_FILES if (d/f).is_file() and not (d/f).is_symlink()]
        got = {x['file']: x['sha256'] for x in files}
        if not pointer_ok(ptr, content, got):
            raise N.Blocked(f'{job}: RESULT.json does not point at the RESULT.json.gz on disk; raw files kept')
        value = N.seal(dict(schema='coopmwm.a75d2.dropped_raw.v1', amendment=AMENDMENT, stage=STAGE, rule=DROP_RULE,
                            job_id=r['job_id'], work_dir=job, wave=name, world_id=r['world_id'],
                            seed=r['world']['seed'], vehicle_log_sha256_sealed=r['vehicle_log']['sha256'],
                            report=dict(file='vehicles_report.json', sha256=_sha(cache), version=c['version']),
                            result_pointer=dict(ptr), result_content=dict(sha256=content[0], bytes=content[1]),
                            removed=files, unix=time.time()), 'receipt_sha256')
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

def form_activations(recs, worlds):
    tot = Counter()
    for w in worlds:
        r = recs[w].get('a75') or {}
        tot['activations'] += int(r.get('activations') or 0)
        for k, v in (r.get('holdform_summary') or {}).items():
            tot[k] += int(v or 0)
    return dict(tot)


def admit(D, D68, panel, root, a68_root, a66_root, ref68_reports, cands):
    N = _mods()[0]
    H = _H()
    from tools.coopmwm import a37_cellbank as C
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    if not mech_passed(root):
        raise N.Blocked('mechanics is not PASS')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, reports = {}, {}, {}
    have[REF], sources[REF] = a68_ref(D68, panel, a68_root, a66_root)
    reports[REF] = pinned_reports(ref68_reports, have[REF], worlds, ref_pins()['a68v_mappo60c'])
    for (name, _, specs), c in zip(verify_waves(D, panel, root, a66_root, cands), cands):
        have[c['family']], sources[c['family']] = _load(root, name, specs)
        reports[c['family']] = _own_reports(root, have[c['family']], worlds)
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        s = cand['strategy']
        v.update(family=fam, source='A75 D2 (new run)', draw_rank=cand['draw_rank'], key=cand['key'],
                 form=cand['form'], stratum=cand['stratum'], n_roles=len(s['roles']), zone=s['zone'],
                 reference=REF, convention='diff = plan - MAPPO60C (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activation_counts=form_activations(have[fam], worlds),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rows.append(v)
    chosen, dup, table = bank_choice(rows)
    for r in rows:
        r['duplicate_of'] = dup.get(r['family'])
        r['IN_BANK'] = r['family'] in chosen
    plans = bank_plans(cands, rows, chosen)
    bank = None
    if plans:
        p = D.C37.write_once(D.bank_path(root, D2_BANK), D.C37.bank_text(H.bank_doc(plans, CELL)))
        bank = dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans],
                    forms={c['family']: c['form'] for c in cands if c['family'] in chosen})
    value = N.seal(dict(
        schema='coopmwm.a75d2.admission.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, rule=ADMIT_RULE,
        bank_rule=BANK_RULE, reference=REF, disclosure=DISCLOSURE,
        multiplicity=f'{len(rows)} plans each tested at 95% on the same 24 worlds; admission is a screen, not a result',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), n_tested=len(rows), per_form=table,
        admitted=[r['family'] for r in rows if r['PASS']], n_admitted=sum(bool(r['PASS']) for r in rows),
        n_delay_pass=sum(bool(r['ci95'][1] < 0) for r in rows), duplicates=dup, d2_bank=chosen, bank=bank,
        verdicts=rows, consequence=f'{ARM} runs on the A70 panel' if plans else 'no plan admitted: no confirmation',
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        ref_pins_receipt_sha256=ref_pins()['receipt_sha256'],
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


# ------------------------------------------------------------------------------------------------ confirmation

def analyse(D, D70, panel, root, a70_root, a66_root, ref70_reports):
    import numpy as np
    N = _mods()[0]
    A70 = _a70()
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
    have[ARM], sources[ARM] = _load(root, wave, specs)
    reports[ARM] = _own_reports(root, have[ARM], worlds)
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    row = D._contrast(have, reports, worlds, ARM, REF, bootstrap, 'primary')
    ci = row['total_minus_ref']['ci95']
    groups = row['groups'] or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    primary = dict(rule=PRIMARY_RULE, contrast=f'{ARM} - MAPPO60C', diff=row['total_minus_ref']['diff'],
                   ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                   group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                   verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED',
                   worst_10pct_cvar90=row.get('tail_cvar90'), groups=groups,
                   hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'))
    rows = [row, D._contrast(have, reports, worlds, ARM, 'DUO_HOLD_60C', bootstrap, 'secondary_descriptive'),
            D._contrast(have, reports, worlds, 'DUO_HOLD_60C', REF, bootstrap, 'secondary_descriptive')]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    by_fam = Counter()
    for w in worlds:
        for f, n in ((have[ARM][w]['a75'].get('activations_by_family')) or {}).items():
            by_fam[f] += int(n)
    forms = (adm['bank'] or {}).get('forms') or {}
    by_form = Counter()
    for f, n in by_fam.items():
        by_form[forms.get(f, '?')] += n
    arms[ARM].update(activation_counts=form_activations(have[ARM], worlds), activations_by_family=dict(by_fam),
                     activations_by_form=dict(by_form))
    return N.seal(dict(schema='coopmwm.a75d2.analysis.v1', amendment=AMENDMENT, stage=STAGE, authority=AUTHORITY,
                       cell=CELL, disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, admission=dict(receipt_sha256=adm['receipt_sha256'], bank=adm['bank'],
                                                           per_form=adm['per_form'], d2_bank=adm['d2_bank']),
                       arms_run=[ARM], arms_reused=dict(root=str(a70_root), arms=list(REF_ARMS)), root=str(root),
                       primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       ref_pins_receipt_sha256=pins['receipt_sha256'],
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ execution

def execute(D, root, waves, *, workers, chunk, log, panel=None):
    """a42_bigp60.execute with the D2 check (A75 + A37 arms) and the D2 runtime / phase / cap."""
    N = _mods()[0]
    import a32_ablation as A
    root = Path(root)
    runtime = D.A42Runtime(N.Runtime(N.RUNNER), root)
    plant = root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the A75 D2 arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A75_ARMS', None) != ('A75_PLAN', 'A75_BANK') or getattr(R, 'A37_ARMS', None) != D.C37.A37_ARMS:
            raise N.Blocked('runner lacks the amendment-75 / amendment-37 arms (ops/a75/host/patch_runner_a75.py)')
        if panel is not None:
            N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(root)
    u0, _ = A.bindings(root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(root, name, specs, check_d2)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (root/D.STOP_A42).exists():
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
        N.run_waves(selection, root=plant, final_root=root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published={k: v[0] for k, v in coverage().items()}, charged_sim_s=D.charged(root)))
    return dict(status='PASS', coverage=coverage(), charged_sim_s=D.charged(root))


def stage_charged(D, root, stage):
    return sum(D.charged(root, p) for p in STAGE_PREFIXES[stage])


def cap_check(used_stage, used_total, need, stage):
    """Pure: refuse when the stage or phase cap would be exceeded."""
    if used_stage+need > STAGE_CAP[stage] or used_total+need > CAP_SIM_S:
        raise SystemExit(f'{stage}: {used_stage} + {need} > stage cap {STAGE_CAP[stage]} (or phase cap {CAP_SIM_S})')
    return True


# ------------------------------------------------------------------------------------------------ CLI

COMMAND_STAGE = {'mech-ident': 'i', 'mech-forms': 'm', 'verify': 'v', 'admit': 'v', 'confirm': 'c', 'analyse': 'c',
                 'mech-compare': 'i', 'panel-m': 'm'}
CAP_STAGE = {'mech-ident': 'mech', 'mech-forms': 'mech', 'verify': 'verify', 'confirm': 'confirm'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('draw', 'panel-m', 'prepare', 'dry', 'mech-ident', 'mech-forms', 'mech-compare',
                                       'verify', 'admit', 'confirm', 'analyse', 'status', 'drop-raw'))
    p.add_argument('--final-root')
    p.add_argument('--a68-root')
    p.add_argument('--a70-root')
    p.add_argument('--a66-root')
    p.add_argument('--ref68-reports')
    p.add_argument('--ref70-reports')
    p.add_argument('--stage', choices=('i', 'm', 'v', 'c'), default='v')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    if a.command == 'draw':
        doc = draw(a.out)
        print(json.dumps({k: doc[k] for k in doc if k.startswith(('doc_sha', 'n_cand', 'candidates_by'))}))
        return
    N = _mods()[0]
    root = Path(a.final_root)
    for other in (a.a68_root, a.a70_root, a.a66_root, a.ref68_reports, a.ref70_reports):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A75 D2 final root is a new root')
    which = COMMAND_STAGE.get(a.command, a.stage)
    if a.command == 'panel-m':
        print(json.dumps(build_panel_m(root)))
        return
    if a.command == 'mech-compare':
        rows, fails, safety, strat = mech_compare(root, a.a68_root, a.a66_root)
        # the forms half needs the a75d2m setting: a second process (one setting per process)
        tmp = root/'A75D2_MECH_IDENT_PART.json'
        tmp.write_text(json.dumps(dict(rows=rows, fails=fails, safety=safety), sort_keys=True), encoding='utf-8')
        import subprocess
        rc = subprocess.call([sys.executable, __file__, 'mech-compare-forms', '--final-root', str(root),
                              '--a66-root', a.a66_root])
        sys.exit(rc)
    D, panel, Dref = setup(which, root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(st):
        if st == 'i':
            return ident_waves(D, panel, root, a.a66_root)
        if st == 'm':
            return form_waves(D, panel, root, a.a66_root)
        if st == 'v':
            return verify_waves(D, panel, root, a.a66_root, candidates())
        return confirm_waves(D, panel, root, a.a66_root)

    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, which, a.a68_root)))
    elif a.command == 'dry':
        ws = waves_of(which)
        out = dict(stage=which, panel_sha256=(panel or {}).get('panel_sha256'), n_waves=len(ws),
                   waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'])
                          for n, _, s in ws],
                   sim_s=sum(len(s)*HORIZON for _, _, s in ws), charged=D.charged(root), cap=CAP_SIM_S,
                   a75_sha256=_H().a75_digest())
        if which == 'v':
            have, src = a68_ref(Dref, panel, a.a68_root, a.a66_root)
            worlds = [r['world_id'] for r in panel['worlds']]
            reps = pinned_reports(a.ref68_reports, have, worlds, ref_pins()['a68v_mappo60c'])
            out['reference'] = dict(wave='A68V_MAPPO60C', receipt_sha256=src['receipt_sha256'], n=len(have),
                                    reports_ok=all(x['ok'] for x in reps.values()))
        print(json.dumps(out, indent=1, default=str))
    elif a.command in CAP_STAGE:
        import fcntl
        if _H().a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command in ('verify', 'confirm') and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        if a.command == 'mech-forms':
            mech_seed0()
        st = CAP_STAGE[a.command]
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(which)
            pending = sum(len(s)-N.published(root, w, s, check_d2)[2] for w, _, s in waves)
            cap_check(stage_charged(D, root, st), D.charged(root), pending*HORIZON, st)
            drop = a.drop_raw and st != 'mech'
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), workers=a.workers,
                     chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    drop_raw(root, waves, procs=a.drop_procs, log=log)
            if drop:
                drop_raw(root, waves, procs=a.drop_procs, log=log)
            res = execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'drop-raw':
        if which not in 'vc':
            raise N.Blocked('drop-raw: verification / confirmation only')
        print(json.dumps(drop_raw(root, waves_of(which), procs=a.drop_procs, log=log)))
    elif a.command == 'admit':
        v = admit(D, Dref, panel, root, a.a68_root, a.a66_root, a.ref68_reports, candidates())
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_form=v['per_form'], d2_bank=v['d2_bank'],
                              bank=v['bank'], duplicates=v['duplicates'], consequence=v['consequence'])))
    elif a.command == 'analyse':
        v = analyse(D, Dref, panel, root, a.a70_root, a.a66_root, a.ref70_reports)
        N.publish(root/RESULTS_FILE, v)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=which, mech=mech_passed(root),
                   admission=admission(root) is not None)
        try:
            ws = waves_of(which)
            done = [N.published(root, n, s, check_d2)[2] for n, _, s in ws]
            out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


def main_forms(argv):
    """Second process of mech-compare (setting a75d2m)."""
    p = argparse.ArgumentParser()
    p.add_argument('command')
    p.add_argument('--final-root', required=True)
    p.add_argument('--a66-root', required=True)
    a = p.parse_args(argv)
    N = _mods()[0]
    root = Path(a.final_root)
    part = json.loads((root/'A75D2_MECH_IDENT_PART.json').read_text(encoding='utf-8'))
    strat, _ = examples()
    v = mech_compare_forms(root, a.a66_root, part['rows'], part['fails'], [tuple(x) for x in part['safety']], strat)
    N.publish(root/MECH_FILE, v)
    print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'][:20])))
    if v['verdict'] != 'PASS':
        sys.exit(5)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'mech-compare-forms':
        main_forms(sys.argv[1:])
    else:
        main()
