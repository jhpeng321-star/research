"""AMENDMENT 81 (S-BIGCONFIRM-A81; POST-HOC, exploratory): a LARGER FRESH-panel confirmation of the frozen A75-D2 bank
against the hand-written hold at D2800_p60 (60% CAVs, map wz3to2, 2,800 veh/h), all three arms RUN on the new worlds.

Follows the A75-D2 confirmation (96 A70 worlds 996300-996395): bank - MAPPO60C -3,606 veh-s CI95 [-5,134, -2,131]
(SUPPORTED); bank - DUO_HOLD_60C -708 [-1,740, +293] (not separated). That result stays reported; this panel is never
pooled with it.

Arms (each exactly as configured in A70 / A75-D2; no retuning, no re-admission):
  A75D2_BANK_60C  A75_BANK runner arm on the frozen bank file A75D2_BANK_D2800_p60.json (sha256 c325ec1d, byte copy;
                  plans d2_d86b1d6ac653e8ad rolling, d2_b5fb46c0bc659861 pulse) on MAPPO60C (as a75_d2forms.confirm_waves)
  DUO_HOLD_60C    the A67 / A70 DUO_HOLD arm with the six executor fields bound to MAPPO60C (as a70_smallbank)
  MAPPO60C        the A66 MAPPO60C executor alone (A65 policy_00084, 2abed317; as a70_smallbank / a68 mappo_specs)

Stages (own root /root/a81big/root; ledger phase ``bigconfirm_a81``; caps enforced in code: mech 7,200, confirm
1,382,400 sim-s; no retry reserve, no automatic replay):
  panel-c      zero SUMO: the fresh 384-world panel 990100-990483 (block 990100-990499; 990484-990499 reserved unused)
               built by nm_finaleval.build_panel exactly as the A70 panel (setting = the a70c setting with only the
               seed block / size / labels changed); write-once + pin file in the root before any A81 episode
  prepare      zero SUMO: byte copy of the frozen bank into <root>/a75d2_banks (sha checked)
  mech         setting a70c, the FIRST 2 A70 worlds (996300, 996301): the three arms run from this tree; MECH_RULE
               compares them with the stored A70 (MAPPO60C, DUO_HOLD_60C) and A75-D2 (bank) records
  mech-compare zero SUMO -> A81_MECH.json PASS / FAIL; confirm refuses unless PASS
  confirm      the three arms on the 384 fresh worlds (3 x 384 episodes), raw archives dropped per round (DROP_RULE)
  analyse      PRIMARY_RULE (registered here before any A81 outcome) + descriptive secondaries -> A81_RESULTS_*.json

Commands (python ops/a81_bigconfirm.py <cmd> --final-root R --a66-root R66 --a70-root R70 --ref70-reports RR70
          --a75d2-root RD2):
  panel-c ; prepare ; dry --stage i|c ; mech ; mech-compare ; confirm [--drop-raw] ; analyse ; status --stage i|c ;
  drop-raw
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

AMENDMENT = '81'
CELL = 'D2800_p60'
PHASE = 'bigconfirm_a81'
HORIZON = 1_200
IDENT_N, CONFIRM_N = 2, 384
BANK_ARM = 'A75D2_BANK_60C'
ARMS = (BANK_ARM, 'DUO_HOLD_60C', 'MAPPO60C')
MECH_EPISODES = len(ARMS)*IDENT_N
MECH_CAP, CONFIRM_CAP = MECH_EPISODES*HORIZON, len(ARMS)*CONFIRM_N*HORIZON
CAP_SIM_S = MECH_CAP+CONFIRM_CAP
assert (MECH_CAP, CONFIRM_CAP, CAP_SIM_S) == (7_200, 1_382_400, 1_389_600)
BLOCK = (990100, 990499)
SEEDS_C = tuple(range(BLOCK[0], BLOCK[0]+CONFIRM_N))
RESERVED_C = (BLOCK[0]+CONFIRM_N, BLOCK[1])
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
SETTINGS = dict(i='a70c_D2800_p60', c='a81c_D2800_p60')
RANGES = dict(i='A70_RANGES', c='A81_RANGES')
A70_PANEL_SHA256 = '1b4b59ee4ab717b931647fbae7cfa489726cd2cbacad30e268c57e32cb2c78e1'
A70_SEEDS = tuple(range(996300, 996396))
PANEL_C = 'A81_PANEL_C.json'
PANEL_C_PIN = 'A81_PANEL_C.sha256'
BANK_DIR = 'a75d2_banks'
BANK_FILE = f'A75D2_BANK_{CELL}.json'
BANK_SHA256 = 'c325ec1da034d0ad22d242f97e926c30cfa998981f16cef935b1b6498565c30c'
BANK_FAMILIES = ('d2_d86b1d6ac653e8ad', 'd2_b5fb46c0bc659861')
BANK_FORMS = {'d2_d86b1d6ac653e8ad': 'rolling', 'd2_b5fb46c0bc659861': 'pulse'}
A75D2_ADMISSION_RECEIPT = '99f6c64d76b631a125c598ff59b1e9aaeee2b502b62bb9521d5582f7ce09419c'
#: the stored wave receipts the mechanics identity compares against (A70 pins = ops/a75d2/REF_PINS.json; bank = the
#: A75-D2 analysis 'sources' entry)
STORED_RECEIPTS = dict(MAPPO60C='f235a47dc32cbe4f279ca8a17f892ade3e3a0ee9b3f003bb93882d23698923d6',
                       DUO_HOLD_60C='0af22b0c9baed8971d4aed45859a9274146160887f348c12f3acf11ff87048e4',
                       A75D2_BANK_60C='b4a17a6b325455ead474dc4e993a6a799037b26450b44a556cf5e88883e5be09')
PREFIX = dict(i='a81i_', c='a81c_')
TAG = {BANK_ARM: 'bank', 'DUO_HOLD_60C': 'duo_hold_60c', 'MAPPO60C': 'mappo60c'}
STAGE_CAP = dict(mech=MECH_CAP, confirm=CONFIRM_CAP)
MECH_FILE = 'A81_MECH.json'
RESULTS_FILE = f'A81_RESULTS_{CELL}.json'
LOG_FILE = 'A81_LOG.jsonl'
DROP_MANIFEST = 'A81_DROPPED_RAW.jsonl'
DROP_FILE = 'DROPPED_RAW.json'
RAW_FILES = ('RESULT.json.gz', 'vehicles.jsonl.gz', 'observable.jsonl.gz')
IDENT_FIELDS = ('activations', 'activations_by_family', 'issued', 'acts')
RECEIPT_KEYS = ('a37', 'a75', 'oracle', 'a32', 'learned_trigger')
AUTHORITY = ('MAIN AMENDMENT 81 (S-BIGCONFIRM-A81, 2026-11-13; POST-HOC, exploratory): the frozen A75-D2 bank (c325ec1d) '
             'vs DUO_HOLD_60C and MAPPO60C, all three run on a fresh 384-world D2800_p60 panel 990100-990483; phase '
             'bigconfirm_a81, caps mech 7,200 + confirm 1,382,400')
DISCLOSURE = ('POST-HOC and exploratory: designed after the A75-D2 96-world confirmation, in which bank - DUO_HOLD_60C '
              'was NOT separated (-708 veh-s, CI95 [-1,740, +293]); the larger panel is motivated by that result. Bank, '
              'plans, executors and every arm definition are frozen (no retuning, no re-admission). Fresh worlds; '
              'never pooled with the A70 / A75-D2 96-world panel or any other panel')
PRIMARY_RULE = ('SUPPORTED iff A75D2_BANK_60C - DUO_HOLD_60C total delay CI95 upper < 0 (paired percentile bootstrap '
                '10,000, seed 20261104, n = 384) AND the five-group 10 s rule relative to DUO_HOLD_60C '
                '(nm_finaleval._bank_compare groups.all_pass); diff = treatment - reference, negative = less delay')
SECONDARY = ((BANK_ARM, 'MAPPO60C'), ('DUO_HOLD_60C', 'MAPPO60C'))
MECH_RULE = ('PASS iff, for each arm (A75D2_BANK_60C, DUO_HOLD_60C, MAPPO60C) on each of the first 2 A70 worlds '
             '(996300, 996301), the record run from the A81 tree equals the stored record (A70C_MAPPO60C / '
             'A70C_DUO_HOLD_60C wave receipts f235a47d / 0af22b0c in the A70 root; A75D2C_A75D2_BANK_60C receipt '
             'b4a17a6b in the A75-D2 root): J_veh_s exactly equal, safety counters equal, same world dict, executor '
             'loaded = MAPPO60C 2abed317, behavioural executor receipts (activations, activations_by_family, issued, '
             'acts of every a37 / a75 / oracle / a32 receipt present) equal, per-vehicle vehicle-log report '
             '(vehicles_report.json report payload, pinned for the stored side) equal; bank records loaded '
             'a75_sha256 d1af7e0f. The gzip-file sha of the vehicle log is reported, not gating (gzip header bytes '
             'differ between runs; A75-D2 identity showed the same)')
DROP_RULE = ('as the A75-D2 DROP_RULE: after the sealed wave receipt of the episode is published the per-vehicle '
             'report is computed from the log whose sha256 equals the sealed record, written as vehicles_report.json, '
             're-read and compared; the RESULT.json pointer sha is checked against RESULT.json.gz; sizes and sha256 '
             'of the raw files are sealed in DROPPED_RAW.json; only then RESULT.json.gz, vehicles.jsonl.gz and '
             'observable.jsonl.gz of that episode are removed. Only a81c_* episodes directly under <root>/evaluation; '
             'mechanics episodes are never dropped')


def _mods():
    import nm_finaleval as N
    import a50_fullp60 as A50
    import a67_plans_on_60c as A67
    import a68_research60c as A68
    return N, A50, A67, A68


def _D2():
    import a75_d2forms as D2
    return D2


def _A70():
    import a70_smallbank as A70
    return A70


# ================================================================================================ pure helpers

def block_free(block, table):
    """table: runner EVAL_HELDOUT_RANGES rows (lo, hi, base). The block must be exactly one row (base = lo) and
    overlap no other row nor a forbidden range."""
    lo, hi = block
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise ValueError(f'{lo}-{hi}: forbidden range')
    own = [r for r in table if (r[0], r[1]) == (lo, hi)]
    other = [r for r in table if (r[0], r[1]) != (lo, hi) and lo <= r[1] and hi >= r[0]]
    if len(own) != 1 or own[0][2] != lo or other:
        raise ValueError(f'A81 block {lo}-{hi}: own rows {own}, overlapping {other}')
    return True


def primary_verdict(row):
    """Pure (PRIMARY_RULE) on a _contrast row of A75D2_BANK_60C - DUO_HOLD_60C."""
    ci = row['total_minus_ref']['ci95']
    groups = row.get('groups') or {}
    a_pass, b_pass = bool(ci[1] < 0), groups.get('all_pass') is True
    return dict(rule=PRIMARY_RULE, contrast=f'{BANK_ARM} - DUO_HOLD_60C', diff=row['total_minus_ref']['diff'],
                ci95=list(ci), p_two_sided=row['total_minus_ref'].get('p_two_sided'), delay_pass=a_pass,
                group_rule_pass=b_pass, supported=bool(a_pass and b_pass),
                verdict='SUPPORTED' if a_pass and b_pass else 'NOT SUPPORTED',
                worst_10pct_cvar90=row.get('tail_cvar90'), groups=groups,
                hard_brakes_minus_ref=row.get('hard_brakes_minus_ref'))


def spec_diff(a, b, path=''):
    """Pure: the flattened keys whose values differ between two spec dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += spec_diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else str(k))
        return out
    return [] if a == b else [path]


def receipts_of(rec):
    return {k: {f: json.dumps((rec.get(k) or {}).get(f), sort_keys=True) for f in IDENT_FIELDS}
            for k in RECEIPT_KEYS if isinstance(rec.get(k), dict)}


def ident_row(arm, new, ref, rep_new, rep_ref, mappo_sha):
    rn, rr = receipts_of(new), receipts_of(ref)
    row = dict(arm=arm, world_id=new['world_id'], j_new=new['J_veh_s'], j_stored=ref['J_veh_s'],
               j_equal=new['J_veh_s'] == ref['J_veh_s'], safety_equal=new['safety'] == ref['safety'],
               same_world=new['world'] == ref['world'],
               executor_is_mappo60c=new.get('executor_sha256_loaded') == ref.get('executor_sha256_loaded') == mappo_sha,
               receipt_keys=[sorted(rn), sorted(rr)], receipts_equal=rn == rr and bool(rn) == bool(rr),
               report_equal=json.dumps(rep_new, sort_keys=True) == json.dumps(rep_ref, sort_keys=True),
               report_ok=bool(rep_new.get('ok')) and bool(rep_ref.get('ok')),
               vehicle_log_gz_sha_equal=(new.get('vehicle_log') or {}).get('sha256')
               == (ref.get('vehicle_log') or {}).get('sha256'),
               safety=new['safety'])
    if arm == BANK_ARM:
        row['a75_sha256_loaded'] = new.get('a75_sha256_loaded')
    row['PASS'] = bool(row['j_equal'] and row['safety_equal'] and row['same_world'] and row['executor_is_mappo60c']
                       and row['receipts_equal'] and row['report_equal'] and row['report_ok']
                       and (arm != BANK_ARM or row['a75_sha256_loaded'] == _D2().A75_SHA256))
    return row


# ================================================================================================ host-side driver

def core(which):
    """The a42 driver code (as A70 / A75-D2 use it) with the A81 names: own phase / caps / prefixes."""
    N, A50 = _mods()[:2]
    over = dict(
        A42_PHASE=PHASE, A42_CAP_SIM_S=CAP_SIM_S, STOP_A42='STOP_A81',
        HOST_BASE=os.environ.get('A81_HOST_BASE', '/root/a81big'),
        VERIFY_SETTING=SETTINGS[which], CONFIRM_SETTING=SETTINGS[which],
        BANK_DIR=BANK_DIR, IDENT_WAVE='A81_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A81', JOB_PREFIX='a81',
        VERIFY_TAG='a81v', CONFIRM_TAG=f'a81{which}', IDENT_TAG='a81i',
        VERIFY_WAVE_PREFIX='A81V_', CONFIRM_WAVE_PREFIX='A81C_',
        RESULTS_FILE=RESULTS_FILE, LOG_FILE=LOG_FILE, SCHEMA_PREFIX='coopmwm.a81',
        RUNNER_RANGES_ATTR=RANGES[which],
        RESULT_LABEL='amendment 81; POST-HOC exploratory larger fresh-panel confirmation of the frozen A75-D2 bank vs '
                     'DUO_HOLD_60C and MAPPO60C; one shot',
        AUTHORITY=AUTHORITY)
    m = A50._clone(f'a81_core_{which}', HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTINGS[which],)
    m.A42Runtime.phase = PHASE
    N.CHECKS['episode'] = _D2().check_d2
    return m


def register_c_setting(root):
    """Process-local nm_finaleval setting for the A81 panel (nm_finaleval itself unchanged): the registered A70 a70c
    setting with ONLY the seed block, size and labels changed (same plant, cell, contract, bindings)."""
    N = _mods()[0]
    name = SETTINGS['c']
    if name not in N.SETTINGS:
        lo, hi = BLOCK
        N.SETTINGS[name] = dict(
            N.SETTINGS[SETTINGS['i']], N_WORLDS=CONFIRM_N, EVAL_RANGE=BLOCK, EVAL_EXTRA_RANGE=BLOCK,
            SEEDS_RESERVED=RESERVED_C,
            PANEL_SIZE_REASON=(f'MAIN amendment 81: larger confirmation = {CONFIRM_N} fresh worlds {lo}-{lo+CONFIRM_N-1} '
                               f'(D2800_p60); {RESERVED_C[0]}-{hi} reserved unused'),
            CELL_AUTHORITY=('MAIN amendment 81 (S-BIGCONFIRM-A81, post-hoc): D2800_p60 larger fresh-panel confirmation '
                            'of the frozen A75-D2 bank vs DUO_HOLD_60C and MAPPO60C'),
            PANEL_AUTHORITY=(f'A81 amendment 81: fresh audited integers {lo}-{lo+CONFIRM_N-1} (D2800_p60) for the A81 '
                             'confirmation ONLY; executed through the eval runner (scope label only); never pooled'),
            EXPOSURE_EXTRA=dict(
                checked=(f'amendment-81 audit: {lo}-{hi} as standalone integers / inside ranges absent from '
                         'REGISTRATION.md, protocols/experiment_history.json, the repo text files, the c4/* branch '
                         'tips, every panel file and runner range table under /root on the run host '
                         '(ops/a81/SEED_AUDIT.json)',),
                disclosed='; amendment 81: fresh integers, registered at runtime by the runner'),
            PANEL_FILE=Path(root)/PANEL_C, PANEL_3C_FILE=None)
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
        N.SETTINGS_PINS[name] = dict(PANEL_SHA256=None, PANEL_3C_SHA256=None)
    return name


def _runner_check():
    N = _mods()[0]
    from tools.coopmwm import nm_eval_runner as R
    if tuple(getattr(R, 'A81_RANGES', ())) != (BLOCK,):
        raise N.Blocked(f'runner A81_RANGES is not {BLOCK} (ops/a81/host/patch_tables_a81.py)')
    block_free(BLOCK, R.EVAL_HELDOUT_RANGES)
    return R


def setup(which, root):
    """-> (D, panel). which = i (A70 panel, mechanics) or c (the A81 panel; panel None before panel-c)."""
    N = _mods()[0]
    if which == 'i':
        D = core('i')
        panel = D.setup(SETTINGS['i'])
        seeds = tuple(r['world']['seed'] for r in panel['worlds'])
        if seeds != A70_SEEDS or panel['cell']['cell_key'] != CELL or panel['panel_sha256'] != A70_PANEL_SHA256:
            raise N.Blocked(f'{SETTINGS["i"]}: not the registered A70 panel')
        return D, panel
    register_c_setting(root)
    _runner_check()
    D = core('c')
    p = Path(root)/PANEL_C
    if not p.is_file():
        return D, None
    pin = (Path(root)/PANEL_C_PIN).read_text(encoding='utf-8').split()
    if N.file_sha(p) != pin[0]:
        raise N.Blocked(f'{PANEL_C}: not the pinned file')
    panel = D.setup(SETTINGS['c'])
    seeds = tuple(r['world']['seed'] for r in panel['worlds'])
    if seeds != SEEDS_C or panel['cell']['cell_key'] != CELL or panel['panel_sha256'] != pin[3]:
        raise N.Blocked(f'{PANEL_C}: not {CELL} {SEEDS_C[0]}-{SEEDS_C[-1]} / not the pinned panel sha')
    want = os.environ.get('A81_PANEL_SHA256')
    if want and want != panel['panel_sha256']:
        raise N.Blocked(f'{PANEL_C}: panel sha {panel["panel_sha256"]} != registered A81_PANEL_SHA256 {want}')
    return D, panel


def build_panel_c(root):
    """Zero SUMO, write-once: the A81 panel by nm_finaleval.build_panel (as the A70 panel)."""
    N = _mods()[0]
    from datetime import datetime, timezone
    import a37_celltailor as C37
    D, panel = setup('c', root)
    if panel is not None:
        raise N.Blocked(f'{PANEL_C} exists')
    N.apply_setting(SETTINGS['c'])
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    value = N.build_panel(N.default_spec_path(), datetime.now(timezone.utc).isoformat())
    if [w['world']['seed'] for w in value['worlds']] != list(SEEDS_C):
        raise N.Blocked('A81 panel: wrong seeds')
    if dict(value['contract']['bootstrap']) != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('A81 panel: bootstrap contract is not the registered one')
    N.validate_panel(value)
    p = C37.write_once(Path(root)/PANEL_C, json.dumps(value, indent=1, sort_keys=True)+'\n')
    C37.write_once(Path(root)/PANEL_C_PIN, f'{N.file_sha(p)}  {PANEL_C} panel_sha256 {value["panel_sha256"]}\n')
    return dict(file=str(p), file_sha256=N.file_sha(p), panel_sha256=value['panel_sha256'],
                seeds=[SEEDS_C[0], SEEDS_C[-1]], n=len(value['worlds']), reserved=list(RESERVED_C))


def prepare(D, root, a75d2_root):
    """Write-once byte copy of the frozen A75-D2 bank (sha c325ec1d) into this root."""
    N = _mods()[0]
    src = Path(a75d2_root)/BANK_DIR/BANK_FILE
    if N.file_sha(src) != BANK_SHA256:
        raise N.Blocked(f'{src}: not the frozen A75-D2 bank {BANK_SHA256[:8]}')
    q = D.C37.write_once(D.bank_path(root, BANK_FILE), src.read_bytes().decode('utf-8'))
    if N.file_sha(q) != BANK_SHA256 or q.read_bytes() != src.read_bytes():
        raise N.Blocked(f'{q}: not the frozen bank bytes')
    doc = json.loads(q.read_text(encoding='utf-8'))
    from tools.coopmwm import a75_holdforms as H
    got, refused = H.a75_bank_plans(doc, CELL)
    if refused or tuple(p['family'] for p in got) != BANK_FAMILIES:
        raise N.Blocked(f'{BANK_FILE}: executor plans {[p["family"] for p in got]} refused {refused}')
    return {BANK_FILE: N.file_sha(q), 'families': [p['family'] for p in got]}


def arm_specs(D, arm, panel, root, a66_root, tag):
    """The arm's episode specs exactly as A70 (MAPPO60C, DUO_HOLD_60C) / A75-D2 (bank) build them."""
    N, _, A67, A68 = _mods()
    D2 = _D2()
    exe = A68.executor(a66_root)
    if arm == 'MAPPO60C':
        return A68.mappo_specs(D, panel, root, exe, tag)
    if arm == 'DUO_HOLD_60C':
        return A67.with_exec(D.confirm_specs('DUO_HOLD', panel, root), exe, tag)
    if arm == BANK_ARM:
        entry = A68.bank_entry(D, root, BANK_FILE, BANK_SHA256)
        base = A67.with_exec(D._a37_specs(D.RUNNER_ARM['PLAN'], panel, root, entry, family='_', tag=tag), exe, tag)
        return D2.a75_from_a37(base, 'A75_BANK', entry, family=None, tag=tag)
    raise N.Blocked(f'no A81 arm {arm!r}')


def waves(D, panel, root, a66_root, which):
    N = _mods()[0]
    n = IDENT_N if which == 'i' else CONFIRM_N
    out = []
    for arm in ARMS:
        specs = arm_specs(D, arm, panel, root, a66_root, f'{PREFIX[which]}{TAG[arm]}')[:n]
        if len(specs) != n:
            raise N.Blocked(f'{arm}: {len(specs)} specs (want {n})')
        out.append((f'A81{which.upper()}_{arm}', 'episode', specs))
    if len({s['job_id'] for _, _, ss in out for s in ss}) != len(ARMS)*n:
        raise N.Blocked('duplicate job ids')
    return out


def _load(root, wave, specs):
    N = _mods()[0]
    recs, src, cov = N.published(root, wave, specs, _D2().check_d2)
    if cov != len(specs):
        raise N.Blocked(f'{wave}: {cov}/{len(specs)} published in {root}')
    return {r['world_id']: r for r in recs}, src


def _receipt(src):
    got = src['receipt_sha256']
    return got if isinstance(got, list) else [got]


def stored_refs(panel, a70_root, ref70_reports, a75d2_root, a66_root, worlds):
    """The stored A70 / A75-D2 records + pinned per-vehicle reports of ``worlds`` (read-only; receipts pinned)."""
    N = _mods()[0]
    D2, A70 = _D2(), _A70()
    from a32_ablation import report_of
    D70 = A70.core('c')
    sel70 = A70.selection(a70_root)
    pins = D2.ref_pins()
    have, reps, srcs = {}, {}, {}
    for arm in ('MAPPO60C', 'DUO_HOLD_60C'):
        h, src = D70.C37._load(a70_root, f'A70C_{arm}', A70.confirm_specs(D70, arm, panel, a70_root, a66_root, sel70))
        if _receipt(src) != [STORED_RECEIPTS[arm]] or _receipt(src) != pins[f'a70c_{arm.lower()}']['wave_receipt_sha256']:
            raise N.Blocked(f'A70C_{arm}: not the pinned wave receipt')
        have[arm] = {w: h[w] for w in worlds}
        reps[arm] = D2.pinned_reports(ref70_reports, have[arm], worlds, pins[f'a70c_{arm.lower()}'])
        srcs[arm] = _receipt(src)
    Dd2 = D2.core('c')
    adm = D2.admission(a75d2_root)
    if adm is None or adm['receipt_sha256'] != A75D2_ADMISSION_RECEIPT or adm['bank']['file_sha256'] != BANK_SHA256:
        raise N.Blocked('A75-D2 admission receipt / bank is not the registered one')
    (wave, _, specs), = D2.confirm_waves(Dd2, panel, a75d2_root, a66_root)
    h, src = D2._load(a75d2_root, wave, specs)
    if _receipt(src) != [STORED_RECEIPTS[BANK_ARM]]:
        raise N.Blocked(f'{wave}: not the pinned wave receipt')
    have[BANK_ARM] = {w: h[w] for w in worlds}
    reps[BANK_ARM] = {}
    for w in worlds:
        rec = have[BANK_ARM][w]
        d = Path(a75d2_root)/'evaluation'/rec.get('work_dir', rec['job_id'])
        drop = N.verify(N.read(d/DROP_FILE), 'receipt_sha256', DROP_FILE)
        if drop['report']['sha256'] != N.file_sha(d/'vehicles_report.json'):
            raise N.Blocked(f'{d.name}: stored report is not the one sealed in its drop receipt')
        reps[BANK_ARM][w] = report_of(a75d2_root, rec)
    srcs[BANK_ARM] = _receipt(src)
    return have, reps, srcs


def mech_compare(root, a70_root, ref70_reports, a75d2_root, a66_root):
    """Zero SUMO: MECH_RULE -> sealed verdict."""
    N, _, A67, _ = _mods()
    from a32_ablation import report_of
    root = Path(root)
    D, panel = setup('i', root)
    ws = waves(D, panel, root, a66_root, 'i')
    worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
    ref, ref_rep, srcs = stored_refs(panel, a70_root, ref70_reports, a75d2_root, a66_root, worlds)
    rows, fails = [], []
    for (wave, _, specs), arm in zip(ws, ARMS):
        have, _ = _load(root, wave, specs)
        if sorted(have) != sorted(worlds):
            fails.append(f'{wave}: worlds {sorted(have)}')
            continue
        for w in worlds:
            row = ident_row(arm, have[w], ref[arm][w], report_of(root, have[w]), ref_rep[arm][w], A67.MAPPO60C_SHA256)
            rows.append(row)
            if not row['PASS']:
                fails.append(f'{arm} {w}')
    if len(rows) != MECH_EPISODES:
        fails.append(f'{len(rows)} identity rows (want {MECH_EPISODES})')
    return N.seal(dict(schema='coopmwm.a81.mech.v1', amendment=AMENDMENT, rule=MECH_RULE, panel_sha256=panel['panel_sha256'],
                       worlds=worlds, stored_receipts=srcs, verdict='PASS' if not fails else 'FAIL', failures=fails,
                       identity=rows, consequence='PASS: confirm may run; FAIL: stop and report'), 'receipt_sha256')


def mech_passed(root):
    N = _mods()[0]
    p = Path(root)/MECH_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name)['verdict'] == 'PASS'


# ------------------------------------------------------------------------------------------------ drop raw (D2 rule)

def drop_raw(root, ws, *, procs=1, log=None):
    """DROP_RULE (= a75_d2forms.drop_raw with the A81 prefix / manifest / labels) for every PUBLISHED a81c_ episode."""
    N = _mods()[0]
    D2 = _D2()
    root = Path(root)
    base = (root/'evaluation').resolve()
    todo = []
    for name, _, specs in ws:
        recs, _, cov = N.published(root, name, specs, D2.check_d2)
        for r in recs[:cov]:
            job = r.get('work_dir', r['job_id'])
            d = root/'evaluation'/job
            if not r['job_id'].startswith(PREFIX['c']) or d.resolve().parent != base or d.is_symlink():
                raise N.Blocked(f'{job}: not an A81 confirmation episode directory')
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
            done = dict(pool.imap_unordered(D2._report_job, jobs, chunksize=1))
    else:
        done = dict(map(D2._report_job, jobs))
    n = freed = 0
    for name, r, d in todo:
        job = d.name
        ok, content = done.get(r['job_id']) or (False, None)
        if ok is not True:
            raise N.Blocked(f'{job}: no valid vehicle-log report; raw files kept')
        cache = d/'vehicles_report.json'
        c = N.read(cache)
        log_sha = D2._sha(d/N.VEHICLE_LOG)
        if (c.get('log_sha256') != r['vehicle_log']['sha256'] or log_sha != r['vehicle_log']['sha256']
                or c.get('version') != N.REPORT_VERSION or not c['report'].get('ok')):
            raise N.Blocked(f'{job}: cached report does not match the sealed record; raw files kept')
        ptr = N.read(d/'RESULT.json')
        files = [dict(file=f, bytes=(d/f).stat().st_size, sha256=log_sha if f == N.VEHICLE_LOG else D2._sha(d/f))
                 for f in RAW_FILES if (d/f).is_file() and not (d/f).is_symlink()]
        got = {x['file']: x['sha256'] for x in files}
        if not D2.pointer_ok(ptr, content, got):
            raise N.Blocked(f'{job}: RESULT.json does not point at the RESULT.json.gz on disk; raw files kept')
        value = N.seal(dict(schema='coopmwm.a81.dropped_raw.v1', amendment=AMENDMENT, rule=DROP_RULE,
                            job_id=r['job_id'], work_dir=job, wave=name, world_id=r['world_id'],
                            seed=r['world']['seed'], vehicle_log_sha256_sealed=r['vehicle_log']['sha256'],
                            report=dict(file='vehicles_report.json', sha256=D2._sha(cache), version=c['version']),
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


# ------------------------------------------------------------------------------------------------ analysis

def analyse(D, panel, root, a66_root):
    """PRIMARY_RULE + descriptive secondaries, all three arms from THIS root (never pooled)."""
    import numpy as np
    N, _, A67, _ = _mods()
    D2 = _D2()
    root = Path(root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile') or len(worlds) != CONFIRM_N:
        raise N.Blocked('panel bootstrap / size is not the registered one')
    if [r['world']['seed'] for r in panel['worlds']] != list(SEEDS_C):
        raise N.Blocked('not the A81 panel')
    have, sources, reports = {}, {}, {}
    for (wave, _, specs), arm in zip(waves(D, panel, root, a66_root, 'c'), ARMS):
        have[arm], sources[arm] = _load(root, wave, specs)
        reports[arm] = D2._own_reports(root, have[arm], worlds)
    if any(not x['ok'] for a in reports for x in reports[a].values()):
        raise N.Blocked('invalid vehicle-log report')
    row = D._contrast(have, reports, worlds, BANK_ARM, 'DUO_HOLD_60C', bootstrap, 'primary')
    primary = primary_verdict(row)
    rows = [row]+[D._contrast(have, reports, worlds, a, r, bootstrap, 'secondary_descriptive') for a, r in SECONDARY]
    arms = {a: dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
            for a in have}
    by_fam = Counter()
    for w in worlds:
        for f, n in ((have[BANK_ARM][w]['a75'].get('activations_by_family')) or {}).items():
            by_fam[f] += int(n)
    by_form = Counter()
    for f, n in by_fam.items():
        by_form[BANK_FORMS.get(f, '?')] += n
    arms[BANK_ARM].update(activation_counts=D2.form_activations(have[BANK_ARM], worlds),
                          activations_by_family=dict(by_fam), activations_by_form=dict(by_form))
    arms['DUO_HOLD_60C']['activations_per_world'] = sum(
        int((have['DUO_HOLD_60C'][w].get('a37') or have['DUO_HOLD_60C'][w].get('oracle') or {}).get('activations') or 0)
        for w in worlds)/len(worlds)
    return N.seal(dict(schema='coopmwm.a81.analysis.v1', amendment=AMENDMENT, authority=AUTHORITY, cell=CELL,
                       disclosure=DISCLOSURE, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                       seeds=[SEEDS_C[0], SEEDS_C[-1]], bootstrap=bootstrap,
                       bank=dict(file=BANK_FILE, file_sha256=BANK_SHA256, families=list(BANK_FAMILIES),
                                 forms=BANK_FORMS, a75d2_admission_receipt_sha256=A75D2_ADMISSION_RECEIPT),
                       arms_run=list(ARMS), arms_reused=None, pooled_with=None, root=str(root),
                       primary=primary, contrasts=rows, arms=arms,
                       multiplicity='one registered primary; every other contrast is descriptive',
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()},
                       label=D.RESULT_LABEL), 'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('panel-c', 'prepare', 'dry', 'mech', 'mech-compare', 'confirm', 'analyse',
                                       'status', 'drop-raw'))
    p.add_argument('--final-root', required=True)
    p.add_argument('--a66-root')
    p.add_argument('--a70-root')
    p.add_argument('--ref70-reports')
    p.add_argument('--a75d2-root')
    p.add_argument('--stage', choices=('i', 'c'), default='c')
    p.add_argument('--out')
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-raw', action='store_true')
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    N = _mods()[0]
    D2 = _D2()
    root = Path(a.final_root)
    for other in (a.a66_root, a.a70_root, a.ref70_reports, a.a75d2_root):
        if other and root.resolve() == Path(other).resolve():
            raise N.Blocked('the A81 final root is a new root')
    if a.command == 'panel-c':
        print(json.dumps(build_panel_c(root)))
        return
    which = dict(mech='i', **{'mech-compare': 'i'}, confirm='c', analyse='c', prepare='c').get(a.command, a.stage)
    if a.command == 'drop-raw':
        which = 'c'
    if a.command == 'mech-compare':
        v = mech_compare(root, a.a70_root, a.ref70_reports, a.a75d2_root, a.a66_root)
        N.publish(root/MECH_FILE, v, replace=True)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'],
                              rows=[(x['arm'], x['world_id'], x['j_equal'], x['report_equal'], x['receipts_equal'],
                                     x['PASS']) for x in v['identity']])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
        return
    D, panel = setup(which, root)
    if which == 'c' and panel is None and a.command != 'prepare':
        raise N.Blocked(f'{PANEL_C} missing (run panel-c)')

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTINGS[which]),
                          sort_keys=True, default=str)
        print(line, flush=True)
        with (root/LOG_FILE).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    if a.command == 'prepare':
        print(json.dumps(prepare(D, root, a.a75d2_root)))
    elif a.command == 'dry':
        ws = waves(D, panel, root, a.a66_root, which)
        out = dict(stage=which, setting=SETTINGS[which], panel_sha256=panel['panel_sha256'], n_waves=len(ws),
                   waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[-1]['job_id'], s[0]['world']['seed'],
                           s[-1]['world']['seed']) for n, _, s in ws],
                   episodes=sum(len(s) for _, _, s in ws), sim_s=sum(len(s)*HORIZON for _, _, s in ws),
                   stage_cap=STAGE_CAP['mech' if which == 'i' else 'confirm'], charged=D.charged(root), cap=CAP_SIM_S,
                   a75_sha256=__import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest(),
                   checkpoints={n: {k: v for k, v in s[0]['checkpoint'].items() if k != 'label'} for n, _, s in ws},
                   horizon=[s[0].get('sim_s') for _, _, s in ws], windows=[s[0].get('window') for _, _, s in ws])
        if which == 'i':
            # the A70 / A75-D2 drivers' own specs for the same worlds: every differing field is listed
            A70 = _A70()
            D70 = A70.core('c')
            sel70 = A70.selection(a.a70_root)
            Dd2 = D2.core('c')
            (_, _, sd2), = D2.confirm_waves(Dd2, panel, a.a75d2_root, a.a66_root)
            theirs = {'MAPPO60C': A70.confirm_specs(D70, 'MAPPO60C', panel, a.a70_root, a.a66_root, sel70),
                      'DUO_HOLD_60C': A70.confirm_specs(D70, 'DUO_HOLD_60C', panel, a.a70_root, a.a66_root, sel70),
                      BANK_ARM: sd2}
            out['spec_diff_vs_stored_driver'] = {arm: sorted({k for x, y in zip(s, theirs[arm]) for k in spec_diff(x, y)})
                                                 for (_, _, s), arm in zip(ws, ARMS)}
            worlds = [r['world_id'] for r in panel['worlds']][:IDENT_N]
            _, reps, srcs = stored_refs(panel, a.a70_root, a.ref70_reports, a.a75d2_root, a.a66_root, worlds)
            out['stored_refs'] = dict(receipts=srcs, reports_ok={k: all(x['ok'] for x in v.values()) for k, v in reps.items()})
        print(json.dumps(out, indent=1, default=str))
    elif a.command in ('mech', 'confirm'):
        import fcntl
        if __import__('tools.coopmwm.a75_holdforms', fromlist=['x']).a75_digest() != D2.A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command == 'confirm' and not mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            ws = waves(D, panel, root, a.a66_root, which)
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in ws)
            used = D.charged(root, PREFIX[which])
            need = pending*HORIZON
            if used+need > STAGE_CAP[a.command] or D.charged(root)+need > CAP_SIM_S:
                raise SystemExit(f'{a.command}: {used} + {need} > stage cap {STAGE_CAP[a.command]} (or phase cap '
                                 f'{CAP_SIM_S})')
            drop = a.drop_raw and a.command == 'confirm'
            log(dict(start=a.command, n_waves=len(ws), episodes=sum(len(w[2]) for w in ws), workers=a.workers,
                     chunk=a.chunk, pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop))

            def hook(x):
                log(x)
                if drop and 'round' in x:
                    drop_raw(root, ws, procs=a.drop_procs, log=log)
            if drop:
                drop_raw(root, ws, procs=a.drop_procs, log=log)
            res = D2.execute(D, root, ws, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res))
    elif a.command == 'drop-raw':
        print(json.dumps(drop_raw(root, waves(D, panel, root, a.a66_root, 'c'), procs=a.drop_procs, log=log)))
    elif a.command == 'analyse':
        v = analyse(D, panel, root, a.a66_root)
        N.publish(root/RESULTS_FILE, v)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], primary={k: v['primary'][k] for k in (
            'rule', 'contrast', 'diff', 'ci95', 'delay_pass', 'group_rule_pass', 'verdict')})))
    else:   # status
        out = dict(charged=D.charged(root), cap=CAP_SIM_S, stage=which, mech=mech_passed(root))
        try:
            ws = waves(D, panel, root, a.a66_root, which)
            done = [N.published(root, n, s, D2.check_d2)[2] for n, _, s in ws]
            out.update(waves=len(ws), episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}')
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
