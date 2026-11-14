"""AMENDMENT 76 (A76-D2P20; POST-HOC, exploratory): the 36 A75-D2 NEW hold-form plans (doc 9025e3b1, drawn once by
seed 20261113, stream a75d2-holdforms; NOT redrawn, NOT edited) tested directly in SUMO at 20 % CAVs (D2800_p20, map
wz3to2) on top of a COMPARATOR POLICY that is a parameter (path + sha256 pin), without the world model.

Changed variable vs A75-D2: CAV share 60 % -> 20 % (cell D2800_p20, the registered p20c plant / bindings). At 20 % no
stored reference exists on fresh worlds, so the comparator arm is RUN here on the same worlds (paired).

Comparator (parameter; env A76_COMP_POLICY + A76_COMP_SHA256, both required):
  * sha == d9042cc5 (default registration): the frozen speed-only MAPPO U0 of the p20c bank root (BOUND_U0.json; path
    must equal the bound path). Specs are the registered U0 / A37 specs unchanged (no executor field is rebound).
  * any other sha: a sealed, write-once BOUND_A76_COMP.json in this root (command bind-comp: file sha, mappo export,
    speed-only unless A76_COMP_ALLOW_LANE=1), and ONLY the six executor fields of every spec (comparator arm AND plan
    arms) are replaced (A52.with_executor / A67.with_exec pattern). One root per comparator (A76_COMPARATOR.json).

Stages (own root; ledger phase ``holdforms_a76p20``; caps enforced in code):
  panel --stage m|v|c / pin   zero SUMO: write-once panels (m 997100-997103, v 997200-997223, c 997300-997395) and
                              A76_PINS.json (panels, candidate doc, comparator, executor digest, fixture / example shas)
  mech      (setting m) 4 identity episodes: the plain hold fixture a44_fix_hold by the A37 path (A37_PLAN) and the A75
            path (A75_PLAN) on worlds 997100/997101 + 16 form episodes (the six EXAMPLE_PLANS forms + the six-plan
            bank, as A75-D2) on 997100-997103 -> mech-compare (A76_MECH.json PASS/FAIL; nothing later runs unless PASS)
            cap 24,000 sim-s
  verify    (setting v) the comparator + each of the 36 plans alone (A75_PLAN) on the 24 admission worlds;
            cap 1,065,600 sim-s (36 x 28,800 plans + 28,800 comparator)
  admit     zero physics: ADMIT_RULE + BANK_RULE (= A75-D2 rules, reference = the comparator) -> A76_ADMISSION + bank
  confirm   (setting c) the comparator + A76_BANK (A75_BANK) on the 96 confirmation worlds; cap 230,400 sim-s
            (not run if nothing is admitted)
  analyse   primary A76_BANK - comparator; worst-10 % delay (CVaR90), groups, hard braking, per-form activations.
--drop-raw (verify / confirm; own a76v_ / a76c_ episodes only): the A75-D2 DROP_RULE. No automatic replay.

Commands (python ops/a76_d2p20.py <cmd> --final-root R): panel --stage m|v|c ; pin ; bind-comp --policy P --sha S
  --authority TEXT ; prepare --stage m|v ; dry --stage m|v|c ; mech ; mech-compare ; verify [--drop-raw] ; admit ;
  confirm [--drop-raw] ; analyse ; status --stage m|v|c ; drop-raw --stage v|c
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '76'
STAGE = 'D2P20'
CELL = 'D2800_p20'
SOURCE_SETTING = 'p20c'                     # the registered D2800_p20 plant / bindings (bank-root mirror, as A73)
PHASE = 'holdforms_a76p20'
HORIZON = 1_200
IDENT_N, MECH_WORLDS, VERIFY_N, CONFIRM_N, DRAW_N = 2, 4, 24, 96, 36
MECH_EPISODES = 20
MECH_CAP = MECH_EPISODES*HORIZON
VERIFY_CAP = (DRAW_N+1)*VERIFY_N*HORIZON              # 36 plans + the comparator
CONFIRM_CAP = 2*CONFIRM_N*HORIZON                     # A76_BANK + the comparator
CAP_SIM_S = MECH_CAP+VERIFY_CAP+CONFIRM_CAP
assert (MECH_CAP, VERIFY_CAP, CONFIRM_CAP, CAP_SIM_S) == (24_000, 1_065_600, 230_400, 1_320_000)
#: stage -> (first seed, last seed used, runner block end); runner table A76_RANGES (ops/a76/host/patch_tables_a76.py)
BLOCKS = dict(m=(997100, 997103, 997199), v=(997200, 997223, 997299), c=(997300, 997395, 997399))
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
SETTING = dict(m='a76m_D2800_p20', v='a76v_D2800_p20', c='a76c_D2800_p20')
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
A75_SHA256 = 'd1af7e0ff795f8cd1eb8345c097cc6c113791acd85754c3f664b4ad7dde7dfae'   # a75_holdforms.a75_digest()
CANDS_DOC_SHA256 = '9025e3b1b09c7f05c99621f16e99a905098080b529bfa5e9e4bb8ec5364dae8d'  # A75-D2 candidates (p60 file)
FIXTURE_SHA256 = 'c9999d06668b1beb6ba5dad325322d6104a2e997d5ed9b6e1d1eb78dd6e991a4'   # ops/a44/FIXTURE_PLANS.json
EXAMPLES_SHA256 = 'c7d6b8ef3a7860441fa5e54ad57acf62c86b4bc1352e6b1a8ba82b4e5a3f4db6'  # ops/a75d2/EXAMPLE_PLANS.json
FROZEN_U0_SHA256 = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
COMP_ENV, COMP_SHA_ENV, COMP_LANE_ENV = 'A76_COMP_POLICY', 'A76_COMP_SHA256', 'A76_COMP_ALLOW_LANE'
HOST_BASE = os.environ.get('A76_HOST_BASE', '/root/a76p20')
BANK_DIR = 'a76_banks'
IDENT_BANK = f'A76_IDENT_HOLD_{CELL}.json'
EX_BANK = f'A76_EXAMPLES_{CELL}.json'
CANDS_BANK = f'A76_CANDS_{CELL}.json'
BANK = f'A76_BANK_{CELL}.json'
PANEL = {k: f'A76_PANEL_{k.upper()}.json' for k in 'mvc'}
PINS_FILE = 'A76_PINS.json'
COMP_FILE = 'A76_COMPARATOR.json'
BOUND_COMP = 'BOUND_A76_COMP.json'
MECH_FILE = 'A76_MECH.json'
ADMISSION_FILE = f'A76_ADMISSION_{CELL}.json'
RESULTS_FILE = f'A76_RESULTS_{CELL}.json'
LOG_FILE = 'A76_LOG.jsonl'
STOP_FILE = 'STOP_A76'
REF = 'COMP'
ARM = 'A76_BANK'
PREFIX = dict(i='a76i_', m='a76m_', v='a76v_', c='a76c_')
WAVE = dict(i37='A76I_A37', i75='A76I_A75', vref='A76V_COMP', cref='A76C_COMP', cbank='A76C_BANK')
STAGE_CAP = dict(mech=MECH_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP)
STAGE_PREFIXES = dict(mech=(PREFIX['i'], PREFIX['m']), verify=(PREFIX['v'],), confirm=(PREFIX['c'],))
SAFETY_ZERO = ('collisions', 'teleports')
#: mechanics: receipts errors that mean "the form never ran on that world" (at 20 % a group may never form)
NOT_EXERCISED = ('no activation', 'no completed event', 'gap_fired = 0')
AUTHORITY = ('MAIN AMENDMENT 76 (A76-D2P20, 2026-11-13; POST-HOC, exploratory): the 36 A75-D2 new hold-form plans '
             '(doc 9025e3b1, unchanged) at D2800_p20 vs a parameterised comparator policy (path + sha pin); fresh '
             'worlds 997100-997103 / 997200-997223 / 997300-997395; phase holdforms_a76p20, caps 24,000 + 1,065,600 + '
             '230,400')
DISCLOSURE = ('POST-HOC and exploratory: run after A75-D2 (60 %) was registered; the plans were drawn for 60 % and are '
              'run unchanged at 20 % (a plan whose CAV roles cannot be filled simply does not activate; no plan, '
              'threshold or executor field is edited). Multiplicity: 36 plans each tested at 95 % on the same 24 worlds; '
              'admission is a screen, the confirmation on the separate 96 worlds is the claim-bearing test. Fresh '
              'worlds, never pooled with other panels; no world model is used')
ADMIT_RULE = ('(plan - comparator) total delay CI95 upper < 0 on the 24 admission worlds 997200-997223 '
              '(a37_cellbank.unc_verdict: paired percentile bootstrap 10,000, seed 20261104) AND the five-group 10 s '
              'rule relative to the comparator; the comparator runs on the same worlds in this root (A76V_COMP)')
BANK_RULE = ('the A75-D2 BANK_RULE unchanged: a plan whose 24 per-world results (plan - comparator) equal those of a '
             'plan earlier in draw order is a duplicate and takes no slot; per form family gap / short / zone / pulse / '
             'taper / rolling the admitted non-duplicate plan with the most negative admission CI95 upper bound (ties: '
             'strategy key); of these the (at most) 4 most negative are kept; bank priority = -CI95 upper; plans and '
             'executor fields unchanged; written with a75_holdforms.bank_doc for cell D2800_p20')
PRIMARY_RULE = ('SUPPORTED iff A76_BANK - comparator total delay CI95 upper < 0 (paired percentile bootstrap 10,000, '
                'seed 20261104, n = 96 worlds 997300-997395) AND the five-group 10 s rule relative to the comparator; '
                'diff = treatment - reference, negative = less delay')
MECH_RULE = (
    'PASS iff all of: every record ran the bound comparator executor (and a75 sha == registered for A75 arms); '
    'collisions = 0 and teleports = 0 in all 20 episodes; IDENTITY: on 997100 / 997101 the plain hold fixture '
    'a44_fix_hold by A37_PLAN and by A75_PLAN gives equal J_veh_s, safety counters, DECOMPRESSED vehicle-log sha256 and '
    'executor receipts (activations, activations_by_family, issued, acts), A75 extras empty, and >= 1 activation over '
    'the two worlds (non-vacuous); FORMS: the A75-D2 checks F1-F7 (short / gap / zone / pulse / taper / rolling / bank) '
    'on every world where the form ran; a world where it never ran ("no activation", "no completed event", '
    '"gap_fired = 0") is reported NOT EXERCISED, not failed (20 %: groups may not form); >= 1 form wave exercised on '
    '>= 1 world. Descriptive: per-wave exercised counts, gap re-derivation, hard brakes')
DROP_RULE = ('the A75-D2 DROP_RULE for this root: after the sealed wave receipt of the episode is published, the '
             'per-vehicle report is written and re-verified, the RESULT.json pointer is checked against the '
             'decompressed RESULT.json.gz, sizes and sha256 of the raw files are sealed in DROPPED_RAW.json, then '
             'RESULT.json.gz, vehicles.jsonl.gz, observable.jsonl.gz of that episode are removed. Only a76v_* / a76c_* '
             'episodes directly under <root>/evaluation; mechanics episodes are never dropped')
#: form waves of the mechanics stage (A75-D2 FORM_WAVES, own names): (wave, example family or None = bank, worlds)
FORM_WAVES = (('A76M_F1_SHORT', 'ex_short', 2), ('A76M_F2_GAP', 'ex_gap', 4), ('A76M_F3_ZONE', 'ex_zone', 2),
              ('A76M_F4_PULSE', 'ex_pulse', 2), ('A76M_F5_TAPER', 'ex_taper', 2), ('A76M_F6_ROLLING', 'ex_rolling', 2),
              ('A76M_F7_BANK', None, 2))
assert 2*IDENT_N + sum(n for _, _, n in FORM_WAVES) == MECH_EPISODES


def _mods():
    import nm_finaleval as N
    import a32_ablation as A
    import a75_d2forms as D2
    return N, A, D2


def _H():
    from tools.coopmwm import a75_holdforms as H
    return H


# ================================================================================================ pure helpers

def check_sha(text, what='sha256'):
    if not re.fullmatch('[0-9a-f]{64}', str(text or '')):
        raise SystemExit(f'{what}: a full lowercase sha256 is required (got {text!r})')
    return str(text)


def comp_env(env=None):
    """(policy path or None, sha256) of the coordinator-chosen comparator; refuses when the sha is unset."""
    env = os.environ if env is None else env
    sha = env.get(COMP_SHA_ENV)
    if not sha:
        raise SystemExit(f'{COMP_SHA_ENV} is unset: the comparator policy is pinned at registration; nothing runs')
    return (env.get(COMP_ENV) or None), check_sha(sha, COMP_SHA_ENV)


EXEC_FIELDS = ('comparator_kind', 'label', 'executor_path', 'executor_sha256', 'binding_sha256', 'binding_path')


def with_exe(specs, comp, tag):
    """Registered U0 / A37 specs (frozen d9042cc5 executor) -> ONLY the six executor fields replaced by the comparator
    (no change when the comparator is the frozen U0 itself); job ids <tag>_<i>."""
    out = []
    for i, s in enumerate(specs):
        ck = deepcopy(s['checkpoint'])
        if ck.get('executor_sha256') != FROZEN_U0_SHA256 or ck.get('kind') not in ('u0', 'a37'):
            raise ValueError(f'{s["job_id"]}: not a registered U0 / A37 spec on the frozen U0 executor')
        if comp['rebind']:
            ck.update(comparator_kind=comp['kind'], label=comp['label'], executor_path=comp['path'],
                      executor_sha256=comp['sha256'], binding_sha256=comp['binding_sha256'],
                      binding_path=comp['binding_path'])
        out.append(dict(s, checkpoint=ck, job_id=f'{tag}_{i:04}'))
    return out


def comp_record(comp):
    """The identity of a comparator that is sealed into the root (A76_COMPARATOR.json) and every receipt."""
    return {k: comp[k] for k in ('rebind', 'name', 'kind', 'label', 'path', 'sha256', 'binding_sha256', 'export')}


def runner_block(which, table):
    """(lo, hi, r0, r1) of the stage; table = runner A76_RANGES ((lo, hi) per stage m, v, c)."""
    lo, hi, end = BLOCKS[which]
    r0, r1 = tuple(table)['mvc'.index(which)]
    if (r0, r1) != (lo, end) or not r0 <= lo <= hi <= r1:
        raise ValueError(f'runner block {r0}-{r1} is not the registered {lo}-{end}')
    if any(lo <= b1 and end >= b0 for b0, b1 in FORBIDDEN):
        raise ValueError('forbidden seed range')
    return lo, hi, r0, r1


def check_blocks_free(table, heldout):
    """heldout: runner EVAL_HELDOUT_RANGES rows (lo, hi, base): each A76 block is its own row, overlapping no other."""
    for lo, hi in table:
        own = [r for r in heldout if (r[0], r[1]) == (lo, hi)]
        other = [r for r in heldout if (r[0], r[1]) != (lo, hi) and lo <= r[1] and hi >= r[0]]
        if len(own) != 1 or other:
            raise ValueError(f'block {lo}-{hi}: own rows {own}, overlapping {other}')
    return True


def split_errors(errs):
    """-> (exercised, real errors): a 'never ran' message is not a correctness failure at 20 %."""
    never = [e for e in errs if e in NOT_EXERCISED]
    return not never, [e for e in errs if e not in NOT_EXERCISED]


IDENT_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts')


def ident_row(r37, r75, log37, log75, comp_sha):
    """One identity world: A37_PLAN vs A75_PLAN of the plain hold fixture on the comparator executor."""
    a, b = r37.get('a37') or {}, r75.get('a75') or {}
    row = dict(world_id=r37['world_id'], j_a37=r37['J_veh_s'], j_a75=r75['J_veh_s'],
               j_equal=r37['J_veh_s'] == r75['J_veh_s'], safety_equal=r37['safety'] == r75['safety'],
               log_equal=log37 == log75 and log37 is not None, vehicle_log_content_sha256=[log37, log75],
               receipts_equal={k: json.dumps(a.get(k), sort_keys=True) == json.dumps(b.get(k), sort_keys=True)
                               for k in IDENT_FIELDS},
               a75_extras_empty=(b.get('extended') == [] and b.get('holdforms') == []),
               executor_is_comparator=(r37.get('executor_sha256_loaded') == r75.get('executor_sha256_loaded')
                                       == comp_sha),
               activations=int(a.get('activations') or 0))
    row['PASS'] = bool(row['j_equal'] and row['safety_equal'] and row['log_equal'] and all(row['receipts_equal'].values())
                       and row['a75_extras_empty'] and row['executor_is_comparator'])
    return row


def mech_verdict(ident_rows, form_rows, safety):
    """Pure MECH_RULE. form_rows {wave: [row(world_id, exercised, errors)]}; safety [(wave, world, counters)]."""
    fails = []
    if len(ident_rows) != IDENT_N or not all(r['PASS'] for r in ident_rows):
        fails.append('identity: ' + ', '.join(f'{r["world_id"]} PASS={r["PASS"]}' for r in ident_rows))
    if not sum(r['activations'] for r in ident_rows):
        fails.append('identity vacuous: the hold fixture never activated on the identity worlds')
    for wave, rows in form_rows.items():
        for r in rows:
            if r['errors']:
                fails.append(f'{wave} {r["world_id"]}: {r["errors"][:3]}')
    if not any(r['exercised'] for rows in form_rows.values() for r in rows):
        fails.append('no form wave exercised on any world')
    for wave, w, sf in safety:
        if any(int(sf.get(k) or 0) for k in SAFETY_ZERO):
            fails.append(f'{wave} {w}: collisions/teleports {[sf.get(k) for k in SAFETY_ZERO]}')
    exercised = {wave: sum(bool(r['exercised']) for r in rows) for wave, rows in form_rows.items()}
    return ('PASS' if not fails else 'FAIL'), fails, exercised


def cap_check(used_stage, used_total, need, stage):
    if used_stage+need > STAGE_CAP[stage] or used_total+need > CAP_SIM_S:
        raise SystemExit(f'{stage}: {used_stage} + {need} > stage cap {STAGE_CAP[stage]} (or phase cap {CAP_SIM_S})')
    return True


# ================================================================================================ plans / banks (zero SUMO)

def candidates():
    """The 36 pinned A75-D2 plans (seeded draw verified by a75_d2forms.candidates; doc 9025e3b1), unchanged."""
    N, _, D2 = _mods()
    cands = D2.candidates()
    doc = N.read(D2.CANDS_FILE)
    if doc['doc_sha256'] != CANDS_DOC_SHA256 or len(cands) != DRAW_N:
        raise N.Blocked('the A75-D2 candidates are not the registered doc 9025e3b1')
    return cands


def examples_bank20():
    """(strategies, bank doc for D2800_p20): the shipped six-example bank rebuilt by a75_holdforms.bank_doc, which must
    reproduce the shipped D2800_p60 bank byte-for-byte as JSON before the cell is changed."""
    N, _, D2 = _mods()
    H = _H()
    if N.file_sha(D2.EXAMPLES) != EXAMPLES_SHA256:
        raise N.Blocked('EXAMPLE_PLANS.json is not the registered file')
    strat, bank60 = D2.examples()
    rows = [dict(family=r['family'], strategy=H.canonical(strat[r['family']]), lb95_veh_s=r['lb95_veh_s'])
            for r in sorted(bank60['rows'], key=lambda r: -float(r['lb95_veh_s']))]
    if json.dumps(H.bank_doc(rows, D2.CELL), sort_keys=True) != json.dumps(bank60, sort_keys=True):
        raise N.Blocked('the example bank cannot be rebuilt from its strategies (cell change refused)')
    bank = H.bank_doc(rows, CELL)
    got, refused = H.a75_bank_plans(bank, CELL)
    if refused and any(refused.values()) or len(got) != len(D2.EXAMPLE_FAMILIES):
        raise N.Blocked(f'the D2800_p20 example bank does not execute the six examples ({refused})')
    return strat, bank


def hold_fixture():
    """The plain hold fixture a44_fix_hold (ops/a44/FIXTURE_PLANS.json, registered with A44; used by A73 at 20 %)."""
    N = _mods()[0]
    p = HERE/'a44'/'FIXTURE_PLANS.json'
    if N.file_sha(p) != FIXTURE_SHA256:
        raise N.Blocked('ops/a44/FIXTURE_PLANS.json is not the registered fixture file')
    plans = N.read(p)['plans']
    return plans['hold']


def ident_bank_doc():
    from tools.coopmwm import a37_cellbank as C
    h = hold_fixture()
    return C.bank_doc([dict(family=h['name'], strategy=h, lb95_veh_s=0.0)], CELL)


def cands_bank_doc(cands):
    H = _H()
    doc = H.bank_doc([dict(family=c['family'], strategy=c['strategy'], lb95_veh_s=0.0) for c in cands], CELL)
    got, refused = H.a75_bank_plans(doc, CELL)
    if len(got) != len(cands) or any(refused.values()):
        raise ValueError(f'the D2800_p20 candidate bank executes {len(got)} of {len(cands)} ({refused})')
    return doc


# ================================================================================================ setting / panels

def core():
    """The a42 driver code (as A75-D2 uses it) with the A76 names: own phase / cap / prefixes / bank dir / root."""
    N, A, D2 = _mods()
    import a50_fullp60 as A50
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42=STOP_FILE, HOST_BASE=HOST_BASE, CELL=CELL,
        VERIFY_SETTING=SETTING['v'], CONFIRM_SETTING=SETTING['c'], BANK_DIR=BANK_DIR, IDENT_WAVE='A76_IDENT_UNUSED',
        AMENDMENT=AMENDMENT, LABEL='A76', JOB_PREFIX='a76', VERIFY_TAG='a76v', CONFIRM_TAG='a76c', IDENT_TAG='a76i',
        VERIFY_WAVE_PREFIX='A76V_', CONFIRM_WAVE_PREFIX='A76C_', RESULTS_FILE=RESULTS_FILE, LOG_FILE=LOG_FILE,
        SCHEMA_PREFIX='coopmwm.a76', RUNNER_RANGES_ATTR='A76_RANGES',
        RESULT_LABEL='amendment 76; POST-HOC exploratory direct SUMO test of the A75-D2 new hold forms at 20 % CAVs; '
                     'one shot', AUTHORITY=AUTHORITY)
    m = A50._clone('a76_core', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = tuple(SETTING.values())
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = D2.check_d2
    return m


def block(which):
    N = _mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    table = getattr(R, 'A76_RANGES', None)
    if table is None:
        raise N.Blocked('the runner has no A76_RANGES (ops/a76/host/patch_tables_a76.py)')
    check_blocks_free(table, R.EVAL_HELDOUT_RANGES)
    return runner_block(which, table)


def pins(root):
    N = _mods()[0]
    p = Path(root)/PINS_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def register_setting(which, root):
    """Process-local nm_finaleval setting (nm_finaleval itself unchanged): the p20c cell / plant / bindings with the
    A76 block of the stage (a73_lcside20.register_setting pattern)."""
    N = _mods()[0]
    lo, hi, r0, r1 = block(which)
    what = dict(m='hold-form executor mechanics (never evaluation, outcomes not used)',
                v='new hold-form admission', c='new hold-form confirmation')[which]
    name = SETTING[which]
    N.SETTINGS[name] = dict(
        N.SETTINGS[SOURCE_SETTING],
        N_WORLDS=hi-lo+1, EVAL_RANGE=(lo, r1), EVAL_EXTRA_RANGE=(lo, r1), SEEDS_RESERVED=(hi+1, r1),
        PANEL_SIZE_REASON=f'MAIN amendment 76: {what} = {hi-lo+1} fresh worlds {lo}-{hi} (D2800_p20)',
        CELL_AUTHORITY='MAIN amendment 76 (A76-D2P20, 2026-11-13): new hold forms at D2800_p20',
        PANEL_AUTHORITY=(f'A76: audited integers {lo}-{hi} (runner block {r0}-{r1}) for the {what} ONLY; executed '
                         'through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(checked=('amendment-76 audit: ops/a76/SEED_AUDIT.json',),
                            disclosed=f'; amendment 76: fresh integers ({what}), registered at runtime by the runner'),
        PANEL_FILE=Path(root)/PANEL[which], PANEL_3C_FILE=None)
    if name not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
    pin = (pins(root) or {}).get('panels', {}).get(which)
    N.SETTINGS_PINS[name] = dict(PANEL_SHA256=pin, PANEL_3C_SHA256=None)
    return name


def setup(which, root, *, need_panel=True):
    """-> (D, panel or None)."""
    N = _mods()[0]
    D = core()
    N.apply_setting(register_setting(which, root))
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    p = Path(root)/PANEL[which]
    if not need_panel or not p.is_file():
        if need_panel:
            raise N.Blocked(f'{p.name}: missing (run panel --stage {which})')
        return D, None
    pin = (pins(root) or {}).get('panels', {}).get(which)
    panel = N.load_panel(p, pinned=pin)
    lo, hi, _, _ = block(which)
    if [w['world']['seed'] for w in panel['worlds']] != list(range(lo, hi+1)) or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'{p.name} is not {CELL} {lo}-{hi}')
    if dict(panel['contract']['bootstrap']) != BOOTSTRAP:
        raise N.Blocked('panel bootstrap is not the registered one')
    return D, panel


def build_panel(which, root):
    N = _mods()[0]
    import a37_celltailor as C37
    from datetime import datetime, timezone
    setup(which, root, need_panel=False)
    p = Path(root)/PANEL[which]
    if p.exists():
        raise N.Blocked(f'{p.name} exists (write-once)')
    value = N.build_panel(N.default_spec_path(), datetime.now(timezone.utc).isoformat())
    lo, hi, _, _ = block(which)
    if [w['world']['seed'] for w in value['worlds']] != list(range(lo, hi+1)) or value['cell']['cell_key'] != CELL:
        raise N.Blocked(f'panel {which}: wrong seeds / cell')
    C37.write_once(p, json.dumps(value, indent=1, sort_keys=True)+'\n')
    return dict(stage=which, file=str(p), file_sha256=N.file_sha(p), panel_sha256=value['panel_sha256'],
                seeds=[lo, hi], cell=value['cell']['cell_key'])


# ================================================================================================ comparator

def comparator(root, env=None):
    """The comparator of this root (COMP_ENV / COMP_SHA_ENV); the frozen U0 or the sealed BOUND_A76_COMP.json."""
    N, A, _ = _mods()
    root = Path(root)
    path, sha = comp_env(env)
    u0, _ = A.bindings(root)
    if sha == FROZEN_U0_SHA256:
        if u0['sha256'] != sha or (path is not None and path != u0['path']):
            raise N.Blocked(f'{COMP_ENV}={path}: the frozen U0 is bound at {u0["path"]} (BOUND_U0.json)')
        if N.file_sha(u0['path']) != sha:
            raise N.Blocked(f'{u0["path"]} does not hash to the frozen U0 {sha[:8]}')
        comp = dict(rebind=False, name='U0', kind=u0['kind'], label=u0['label'], path=u0['path'], sha256=sha,
                    binding_sha256=u0['binding_sha256'], binding_path=u0['binding_path'], export=u0.get('export'))
    else:
        b = root/BOUND_COMP
        if not b.is_file():
            raise N.Blocked(f'{b}: missing (run bind-comp with the registered policy first)')
        v = N.verify(N.read(b), 'binding_sha256', b.name)
        if v['sha256'] != sha or (path is not None and path != v['path']) or v.get('amendment') != AMENDMENT:
            raise N.Blocked(f'{b.name}: binds {v["sha256"][:8]} {v["path"]}, not the requested comparator')
        if N.file_sha(v['path']) != sha:
            raise N.Blocked(f'{v["path"]} does not hash to {sha[:8]}')
        comp = dict(rebind=True, name='COMP', kind=v['kind'], label=v['label'], path=v['path'], sha256=sha,
                    binding_sha256=v['binding_sha256'], binding_path=str(b), export=v.get('export'))
    rec = root/COMP_FILE
    if rec.is_file():
        got = N.verify(N.read(rec), 'receipt_sha256', rec.name)
        if got['comparator'] != comp_record(comp):
            raise N.Blocked(f'{COMP_FILE}: this root is bound to comparator {got["comparator"]["sha256"][:8]} '
                            f'({got["comparator"]["path"]}); one root per comparator')
    return comp


def seal_comparator(root, comp):
    N = _mods()[0]
    rec = Path(root)/COMP_FILE
    if rec.is_file():
        return N.verify(N.read(rec), 'receipt_sha256', rec.name)
    value = N.seal(dict(schema='coopmwm.a76.comparator.v1', amendment=AMENDMENT, comparator=comp_record(comp),
                        rule='one root per comparator; every A76 spec of this root runs this executor'),
                   'receipt_sha256')
    N.publish(rec, value)
    return value


def bind_comp(root, policy, sha, authority, allow_lane=False):
    """Seal BOUND_A76_COMP.json (zero physics, write-once): a mappo export, speed-only unless allow_lane."""
    N = _mods()[0]
    root, policy = Path(root), Path(policy)
    sha = check_sha(sha)
    target = root/BOUND_COMP
    if target.exists():
        raise N.Blocked(f'{target}: exists (bound once; a different comparator needs a new root)')
    if sha == FROZEN_U0_SHA256:
        raise N.Blocked('the frozen U0 needs no binding (it is BOUND_U0.json)')
    if any(root.glob('RESERVED_*.json')) or any(root.glob('WAVE_*.json')):
        raise N.Blocked('this root already reserved waves; a comparator can no longer be bound')
    if not str(authority).strip():
        raise N.Blocked('authority (verbatim registration line) required')
    if N.file_sha(policy) != sha:
        raise N.Blocked(f'{policy} does not hash to {sha}')
    meta = N.export_meta(policy)
    if meta.get('trainer') != 'mappo':
        raise N.Blocked(f'{policy}: not a mappo export ({meta})')
    if (meta.get('lane_head') or meta.get('lane_control') != 'native') and not allow_lane:
        raise N.Blocked(f'{policy}: lane head / policy lanes ({meta}); set {COMP_LANE_ENV}=1 only if registered')
    value = N.seal(dict(schema=N.SCHEMA+'.comparator', arm='U0', kind='a76_comparator',
                        label=f'A76 comparator {sha[:12]} ({"lane-capable" if meta.get("lane_head") else "speed-only"})',
                        path=str(policy), sha256=sha, export=meta, authority=str(authority), amendment=AMENDMENT,
                        rule='A76: comparator / plan executor with only the six executor fields replaced (path + '
                             'sha256 verified at every load)', speed_only=not meta.get('lane_head')),
                   'binding_sha256')
    N.publish(target, value)
    return value


# ================================================================================================ specs / waves

def u0_specs(panel, root, comp, tag):
    N, A, _ = _mods()
    u0, _ = A.bindings(root)
    base = N.episode_specs('U0', panel, None, Path(root)/N.PRERUN_PLANT, 'mappo_frozen', None, u0)
    return with_exe(base, comp, tag)


def plan_specs(D, panel, root, bank, arm, family, tag, comp):
    """A37 specs of the p20c plant (D._a37_specs) -> comparator executor (with_exe) -> A75 arm if asked."""
    _, _, D2 = _mods()
    entry = D._bank_entry(root, bank)
    base = with_exe(D._a37_specs('A37_PLAN', panel, root, entry, family=family or '_', tag=tag), comp, tag)
    if arm == 'A37_PLAN':
        return base
    return D2.a75_from_a37(base, arm, entry, family=family, tag=tag)


def mech_waves(D, panel, root, comp):
    fam = hold_fixture()['name']
    out = [(WAVE['i37'], 'episode', plan_specs(D, panel, root, IDENT_BANK, 'A37_PLAN', fam, PREFIX['i']+'a37',
                                               comp)[:IDENT_N]),
           (WAVE['i75'], 'episode', plan_specs(D, panel, root, IDENT_BANK, 'A75_PLAN', fam, PREFIX['i']+'a75',
                                               comp)[:IDENT_N])]
    for wave, ex, n in FORM_WAVES:
        tag = PREFIX['m'] + wave.split('_', 2)[2].lower()
        out.append((wave, 'episode', plan_specs(D, panel, root, EX_BANK, 'A75_BANK' if ex is None else 'A75_PLAN', ex,
                                                tag, comp)[:n]))
    return out


def verify_waves(D, panel, root, comp, cands):
    N = _mods()[0]
    waves = [(WAVE['vref'], 'episode', u0_specs(panel, root, comp, PREFIX['v']+'comp'))]
    waves += [(f'A76V_{c["family"]}', 'episode',
               plan_specs(D, panel, root, CANDS_BANK, 'A75_PLAN', c['family'], f'{PREFIX["v"]}{c["key"]}', comp))
              for c in cands]
    if any(len(s) != VERIFY_N for _, _, s in waves) or len({w for w, _, _ in waves}) != len(waves):
        raise N.Blocked('verification waves: wrong size / duplicate wave')
    return waves


def admission(root):
    N = _mods()[0]
    p = Path(root)/ADMISSION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def confirm_waves(D, panel, root, comp):
    N, _, D2 = _mods()
    adm = admission(root)
    if adm is None or not adm['bank']:
        raise N.Blocked('no admission receipt / no plan admitted: no confirmation')
    entry = D._bank_entry(root, BANK)
    if entry['file_sha256'] != adm['bank']['file_sha256']:
        raise N.Blocked(f'{BANK}: differs from the admission receipt')
    base = with_exe(D._a37_specs('A37_PLAN', panel, root, entry, family='_', tag=PREFIX['c']+'bank'), comp,
                    PREFIX['c']+'bank')
    bank = D2.a75_from_a37(base, 'A75_BANK', entry, family=None, tag=PREFIX['c']+'bank')
    waves = [(WAVE['cref'], 'episode', u0_specs(panel, root, comp, PREFIX['c']+'comp')),
             (WAVE['cbank'], 'episode', bank)]
    if any(len(s) != CONFIRM_N for _, _, s in waves):
        raise N.Blocked('confirmation waves: wrong size')
    return waves


def prepare(root, which):
    """Write-once bank files of the stage + the comparator record of the root."""
    N = _mods()[0]
    import a37_celltailor as C37
    root = Path(root)
    comp = comparator(root)
    out = dict(comparator=seal_comparator(root, comp)['receipt_sha256'])
    docs = dict(m=((IDENT_BANK, ident_bank_doc()), (EX_BANK, examples_bank20()[1])),
                v=((CANDS_BANK, cands_bank_doc(candidates())),))[which]
    for name, doc in docs:
        p = root/BANK_DIR/name
        text = C37.bank_text(doc)
        if p.is_file():
            if p.read_bytes().decode('utf-8') != text:
                raise N.Blocked(f'{p}: exists with different content')
        else:
            C37.write_once(p, text)
        out[name] = N.file_sha(p)
    return out


def make_pins(root):
    """Write-once A76_PINS.json from the three built panels + the registered inputs (zero SUMO)."""
    N, _, D2 = _mods()
    root = Path(root)
    if pins(root) is not None:
        raise N.Blocked(f'{PINS_FILE} exists (write-once)')
    comp = comparator(root)
    panels, files = {}, {}
    for w in 'mvc':
        p = root/PANEL[w]
        if not p.is_file():
            raise N.Blocked(f'{p.name}: missing (run panel --stage {w})')
        panels[w] = N.read(p)['panel_sha256']
        files[w] = N.file_sha(p)
    value = N.seal(dict(schema='coopmwm.a76.pins.v1', amendment=AMENDMENT, cell=CELL, panels=panels, panel_files=files,
                        seeds={w: list(BLOCKS[w][:2]) for w in 'mvc'}, candidates_doc_sha256=CANDS_DOC_SHA256,
                        candidates_file_sha256=N.file_sha(D2.CANDS_FILE), a75_sha256=_H().a75_digest(),
                        fixture_sha256=FIXTURE_SHA256, examples_sha256=EXAMPLES_SHA256,
                        comparator=comp_record(comp), caps=dict(mech=MECH_CAP, verify=VERIFY_CAP, confirm=CONFIRM_CAP,
                                                                total=CAP_SIM_S)), 'receipt_sha256')
    if value['a75_sha256'] != A75_SHA256:
        raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
    N.publish(root/PINS_FILE, value)
    return value


def require_pins(root, panel, which):
    N = _mods()[0]
    p = pins(root)
    if p is None or p['panels'].get(which) != panel['panel_sha256']:
        raise N.Blocked(f'{PINS_FILE} missing / panel {which} not pinned: physics refused')
    if p['comparator'] != comp_record(comparator(root)):
        raise N.Blocked(f'{PINS_FILE}: pinned comparator differs from the requested one')
    return p


# ================================================================================================ loading / mechanics

def _load(root, wave, specs):
    N, _, D2 = _mods()
    recs, src, cov = N.published(root, wave, specs, D2.check_d2)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {root}')
    return {r['world_id']: r for r in recs}, src


def _executor_check(have, comp, what):
    N = _mods()[0]
    for w, r in have.items():
        got = r.get('executor_sha256_loaded')
        if got is not None and got != comp['sha256']:
            raise N.Blocked(f'{what} {w}: executor {got[:8]} is not the comparator {comp["sha256"][:8]}')


def mech_compare(D, panel, root, comp):
    """Zero SUMO: MECH_RULE over the published mechanics waves -> sealed verdict (PASS / FAIL)."""
    N, _, D2 = _mods()
    H = _H()
    root = Path(root)
    strat, _ = examples_bank20()
    waves = mech_waves(D, panel, root, comp)
    (w37, _, s37), (w75, _, s75) = waves[:2]
    r37, r75 = _load(root, w37, s37)[0], _load(root, w75, s75)[0]
    safety, ident = [], []
    for w in r37:
        ident.append(ident_row(r37[w], r75[w], D2._log_sha(root, r37[w]), D2._log_sha(root, r75[w]), comp['sha256']))
        safety += [(w37, w, r37[w]['safety']), (w75, w, r75[w]['safety'])]
    forms = {}
    for (wave, _, specs), (_, fam, _) in zip(waves[2:], FORM_WAVES):
        have, _ = _load(root, wave, specs)
        _executor_check(have, comp, wave)
        rows = []
        for w, rec in have.items():
            r = rec['a75']
            safety.append((wave, w, rec['safety']))
            errs, desc = [], {}
            if r.get('refused') not in (None, {}, []):
                errs.append(f'refused {r.get("refused")}')
            if fam is None:
                errs += D2.check_bank(r)
                desc['families_activated'] = sorted((r.get('activations_by_family') or {}).keys())
            else:
                s = strat[fam]
                role = s['roles'][0]
                p = root/'evaluation'/rec.get('work_dir', rec['job_id'])/N.VEHICLE_LOG
                vl = D2.read_vlog(p) if fam in ('ex_gap', 'ex_zone') and p.is_file() else ({}, {})
                errs += dict(ex_short=lambda: D2.check_short(r, role), ex_gap=lambda: D2.check_gap(r, role),
                             ex_zone=lambda: D2.check_zone(r, H.zone_bounds(s), D2.vlog_pos(vl)),
                             ex_pulse=lambda: D2.check_pulse(r, role), ex_taper=lambda: D2.check_taper(r, role),
                             ex_rolling=lambda: D2.check_rolling(r, s['roles']))[fam]()
                if fam == 'ex_gap':
                    desc = dict(gap_events=D2.gap_rederive(r, vl))
            exercised, real = split_errors(errs)
            rows.append(dict(world_id=w, exercised=exercised, errors=real, raw_messages=errs,
                             activations=int(r.get('activations') or 0), holdform_summary=r.get('holdform_summary'),
                             descriptive=desc, hard_brakes=rec['safety'].get('hard_brakes')))
        forms[wave] = rows
    verdict, fails, exercised = mech_verdict(ident, forms, safety)
    return N.seal(dict(schema='coopmwm.a76.mech.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, rule=MECH_RULE,
                       a75_sha256=A75_SHA256, comparator=comp_record(comp), panel_sha256=panel['panel_sha256'],
                       seeds=list(BLOCKS['m'][:2]), verdict=verdict, failures=fails, exercised_worlds=exercised,
                       identity=ident, forms=forms,
                       deviation=('A75-D2 identity compared A37_PLAN / A75_PLAN with a stored A68 record at 60 %; at '
                                  '20 % no stored record exists on fresh worlds: identity = A37_PLAN vs A75_PLAN of the '
                                  'A44 hold fixture (the plan A73 used at 20 %); a form that never ran on a world is '
                                  'NOT EXERCISED, not failed'),
                       consequence='PASS: verify / admit / confirm may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _mods()[0]
    p = Path(root)/MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ================================================================================================ admission / analysis

def admit(D, panel, root, comp, cands):
    N, A, D2 = _mods()
    H = _H()
    from tools.coopmwm import a37_cellbank as C
    import a37_celltailor as C37
    root = Path(root)
    if admission(root) is not None:
        raise N.Blocked(f'{ADMISSION_FILE} exists (one shot)')
    if not mech_passed(root):
        raise N.Blocked('mechanics is not PASS')
    worlds = [r['world_id'] for r in panel['worlds']]
    seed = {r['world_id']: r['world']['seed'] for r in panel['worlds']}
    bootstrap = dict(panel['contract']['bootstrap'])
    waves = verify_waves(D, panel, root, comp, cands)
    have, sources, reports = {}, {}, {}
    for (name, _, specs), fam in zip(waves, [REF]+[c['family'] for c in cands]):
        have[fam], sources[fam] = _load(root, name, specs)
        _executor_check(have[fam], comp, name)
        reports[fam] = {w: A.report_of(root, have[fam][w]) for w in worlds}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    rows = []
    for cand in cands:
        fam = cand['family']
        cmp_ = N._bank_compare(have, reports, worlds, fam, REF, bootstrap)
        v = C.unc_verdict({seed[w]: have[fam][w]['J_veh_s'] for w in worlds},
                          {seed[w]: have[REF][w]['J_veh_s'] for w in worlds}, cmp_['groups'])
        s = cand['strategy']
        v.update(family=fam, source='A76 (new run, D2800_p20)', draw_rank=cand['draw_rank'], key=cand['key'],
                 form=cand['form'], stratum=cand['stratum'], n_roles=len(s['roles']), zone=s['zone'],
                 reference=comp['name'], convention='diff = plan - comparator (negative = less delay); gain = -diff',
                 delay_test_pass=bool(v['ci95'][1] < 0), group_rule_pass=bool(v['allowed']),
                 cvar90_vs_ref=cmp_['tail_cvar90'], groups=cmp_['groups'],
                 total_minus_ref_secondary=cmp_['total_minus_ref'],
                 activation_counts=D2.form_activations(have[fam], worlds),
                 safety_totals={k: int(sum(have[fam][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rows.append(v)
    chosen, dup, table = D2.bank_choice(rows)
    for r in rows:
        r['duplicate_of'] = dup.get(r['family'])
        r['IN_BANK'] = r['family'] in chosen
    plans = D2.bank_plans(cands, rows, chosen)
    bank = None
    if plans:
        p = C37.write_once(root/BANK_DIR/BANK, C37.bank_text(H.bank_doc(plans, CELL)))
        bank = dict(path=f'{D.HOST_ROOT}/{BANK_DIR}/{p.name}', file_sha256=N.file_sha(p),
                    families=[x['family'] for x in plans], priority=[(x['family'], x['lb95_veh_s']) for x in plans],
                    forms={c['family']: c['form'] for c in cands if c['family'] in chosen})
    value = N.seal(dict(
        schema='coopmwm.a76.admission.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL, rule=ADMIT_RULE,
        bank_rule=BANK_RULE, reference=comp_record(comp), disclosure=DISCLOSURE,
        multiplicity=f'{len(rows)} plans each tested at 95% on the same 24 worlds; admission is a screen, not a result',
        panel_sha256=panel['panel_sha256'], seeds=sorted(seed.values()), n_tested=len(rows), per_form=table,
        admitted=[r['family'] for r in rows if r['PASS']], n_admitted=sum(bool(r['PASS']) for r in rows),
        n_delay_pass=sum(bool(r['ci95'][1] < 0) for r in rows), duplicates=dup, a76_bank=chosen, bank=bank,
        verdicts=rows, consequence=f'{ARM} runs on the confirmation panel' if plans else 'no plan admitted: no '
                                                                                         'confirmation',
        ref_mean_J_veh_s=sum(have[REF][w]['J_veh_s'] for w in worlds)/len(worlds),
        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    N.publish(root/ADMISSION_FILE, value)
    return value


def analyse(D, panel, root, comp):
    import numpy as np
    N, A, D2 = _mods()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != BOOTSTRAP or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    adm = admission(root)
    have, sources, reports = {}, {}, {}
    for (name, _, specs), arm in zip(confirm_waves(D, panel, root, comp), (REF, ARM)):
        have[arm], sources[arm] = _load(root, name, specs)
        _executor_check(have[arm], comp, name)
        reports[arm] = {w: A.report_of(root, have[arm][w]) for w in worlds}
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    row = D._contrast(have, reports, worlds, ARM, REF, bootstrap, 'primary')
    ci = row['total_minus_ref']['ci95']
    groups = row['groups'] or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    primary = dict(rule=PRIMARY_RULE, contrast=f'{ARM} - comparator ({comp["name"]} {comp["sha256"][:8]})',
                   diff=row['total_minus_ref']['diff'], ci95=list(ci), p_two_sided=row['total_minus_ref'].get(
                       'p_two_sided'), delay_pass=a_pass, group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                   verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED',
                   worst_10pct_cvar90=row.get('tail_cvar90'), groups=groups,
                   hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'))
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
    arms[ARM].update(activation_counts=D2.form_activations(have[ARM], worlds), activations_by_family=dict(by_fam),
                     activations_by_form=dict(by_form))
    return N.seal(dict(schema='coopmwm.a76.analysis.v1', amendment=AMENDMENT, stage=STAGE, authority=AUTHORITY,
                       cell=CELL, disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       bootstrap=bootstrap, comparator=comp_record(comp),
                       admission=dict(receipt_sha256=adm['receipt_sha256'], bank=adm['bank'], per_form=adm['per_form'],
                                      a76_bank=adm['a76_bank']),
                       arms_run=[REF, ARM], root=str(root), primary=primary, contrasts=[row], arms=arms,
                       multiplicity='one registered primary; every other number is descriptive',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, label=D.RESULT_LABEL),
                  'analysis_sha256')


# ================================================================================================ raw drop / execution

def dropper():
    """a75_d2forms.drop_raw (D1-fixed DROP_RULE code) with the A76 prefixes / names (private module copy)."""
    import a50_fullp60 as A50
    D2 = _mods()[2]
    return A50._clone('a76_drop', Path(D2.__file__), dict(
        PREFIX=PREFIX, AMENDMENT=AMENDMENT, STAGE=STAGE, DROP_RULE=DROP_RULE, DROP_MANIFEST='A76_DROPPED_RAW.jsonl'))


def stage_charged(D, root, stage):
    return sum(D.charged(root, p) for p in STAGE_PREFIXES[stage])


COMMAND_STAGE = {'mech': 'm', 'mech-compare': 'm', 'verify': 'v', 'admit': 'v', 'confirm': 'c', 'analyse': 'c'}
CAP_STAGE = {'mech': 'mech', 'verify': 'verify', 'confirm': 'confirm'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('panel', 'pin', 'bind-comp', 'prepare', 'dry', 'mech', 'mech-compare', 'verify',
                                       'admit', 'confirm', 'analyse', 'status', 'drop-raw'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--stage', choices=('m', 'v', 'c'), default='v')
    p.add_argument('--policy')
    p.add_argument('--sha')
    p.add_argument('--authority')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    N, A, D2 = _mods()
    root = Path(a.final_root)
    if a.command == 'bind-comp':
        core()
        v = bind_comp(root, a.policy, a.sha, a.authority, allow_lane=os.environ.get(COMP_LANE_ENV) == '1')
        print(json.dumps(dict(binding_sha256=v['binding_sha256'], path=v['path'], sha256=v['sha256'],
                              export=v['export'])))
        return
    if a.command == 'pin':
        setup('m', root, need_panel=False)
        v = make_pins(root)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], file_sha256=N.file_sha(root/PINS_FILE),
                              panels=v['panels'], comparator=v['comparator']['sha256'])))
        return
    which = COMMAND_STAGE.get(a.command, a.stage)
    if a.command == 'panel':
        print(json.dumps(build_panel(which, root)))
        return
    if a.command == 'prepare':
        setup(which, root, need_panel=False)
        print(json.dumps(prepare(root, which)))
        return
    D, panel = setup(which, root)
    comp = comparator(root)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    def waves_of(st):
        if st == 'm':
            return mech_waves(D, panel, root, comp)
        if st == 'v':
            return verify_waves(D, panel, root, comp, candidates())
        return confirm_waves(D, panel, root, comp)

    if a.command == 'dry':
        note = None
        try:
            ws = waves_of(which)
        except N.Blocked as e:            # confirmation before admission: show the comparator wave only
            if which != 'c':
                raise
            ws, note = [(WAVE['cref'], 'episode', u0_specs(panel, root, comp, PREFIX['c']+'comp'))], str(e)
        print(json.dumps(dict(
            note=note,
            stage=which, setting=SETTING[which], panel_sha256=panel['panel_sha256'],
            pinned=(pins(root) or {}).get('panels', {}).get(which) == panel['panel_sha256'],
            comparator=dict(name=comp['name'], sha256=comp['sha256'], path=comp['path'], rebind=comp['rebind'],
                            export=comp['export']),
            n_waves=len(ws), waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'],
                                     s[-1]['world']['seed'], s[0]['checkpoint']['executor_sha256'][:8]) for n, _, s in ws],
            sim_s=sum(len(s)*HORIZON for _, _, s in ws), charged=D.charged(root), cap=CAP_SIM_S,
            stage_cap=STAGE_CAP[dict(m='mech', v='verify', c='confirm')[which]], a75_sha256=_H().a75_digest()),
            indent=1, default=str))
    elif a.command in CAP_STAGE:
        import fcntl
        if _H().a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        require_pins(root, panel, which)
        if a.command in ('verify', 'confirm') and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        st = CAP_STAGE[a.command]
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = waves_of(which)
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in waves)
            cap_check(stage_charged(D, root, st), D.charged(root), pending*HORIZON, st)
            drop = a.drop_raw and st != 'mech'
            log(dict(start=a.command, n_waves=len(waves), episodes=sum(len(w[2]) for w in waves), workers=a.workers,
                     chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop,
                     comparator=comp['sha256']))
            X = dropper() if drop else None

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    X.drop_raw(root, waves, procs=a.drop_procs, log=log)
            if drop:
                X.drop_raw(root, waves, procs=a.drop_procs, log=log)
            res = D2.execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'mech-compare':
        v = mech_compare(D, panel, root, comp)
        N.publish(root/MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'][:20],
                              exercised_worlds=v['exercised_worlds'])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        if which not in 'vc':
            raise N.Blocked('drop-raw: verification / confirmation only')
        print(json.dumps(dropper().drop_raw(root, waves_of(which), procs=a.drop_procs, log=log)))
    elif a.command == 'admit':
        v = admit(D, panel, root, comp, candidates())
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_form=v['per_form'], a76_bank=v['a76_bank'],
                              bank=v['bank'], duplicates=v['duplicates'], consequence=v['consequence'])))
    elif a.command == 'analyse':
        v = analyse(D, panel, root, comp)
        N.publish(root/RESULTS_FILE, v)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary=v['primary'])))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=which, mech=mech_passed(root),
                   admission=admission(root) is not None, comparator=comp['sha256'][:12])
        try:
            ws = waves_of(which)
            done = [N.published(root, n, s, D2.check_d2)[2] for n, _, s in ws]
            out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
