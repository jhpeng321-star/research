"""AMENDMENT 96 part (b) (A96-P20ADM; POST-HOC, exploratory): SUMO admission at 20 % CAVs (D2800_p20) of <= 12 plans
per root vs MAPPO20C (A77 iteration-84 export, speed-only), run ONCE as a stored reference on 24 FRESH worlds
988900-988923 (runner block 988900-988999, rest unused). DRAFT prep 2026-11-14; nothing here has run.

This is the A82 part-A driver (ops/a82_s3b20.py 2b33146b = the 20 % version of A80: a80_s2b MAPPING_RULE / FORM_RULE /
plan_checks / mech_verdict, a76_d2p20.admit ADMIT_RULE + BANK_RULE, a75_d2forms execute / drop rule) imported
UNCHANGED, with three things re-pointed (process-local, same pattern A82 used for A76 and A80 for A75-D2):
  1. worlds: the A96 block (runner table A96_RANGES, patch_tables_a96.py) instead of the reused A76 worlds; the panel is
     built once in the REFERENCE root and byte-copied into every admission root (panel sha pinned);
  2. reference: the stored wave A96V_MAPPO20C of the reference root (/root/a96ref): the U0 specs (a76_d2p20.u0_specs)
     through A52.with_executor with the a77_mappo20c-bound executor (= a77_mappo20c.confirm_specs / a82_fourarm
     MAPPO20C arm), run once with the A75-D2 drop rule; admission roots read it read-only with pinned wave receipts and
     the 24 DROPPED_RAW-sealed reports (the A82 stored-reference path);
  3. the executor check of a76_d2p20.admit: plan records must carry the U0 executor d9042cc5 (plans run on U0, as every
     arm of this study), reference records the bound MAPPO20C sha. Nothing else in the admission computation changes.
Mechanics: the A82 20 % rule (first 2 verification episodes of every candidate, costless; A80 'where required' = every
plan at 20 %, because no 20 % fixture covers any token).

Reference root commands (--final-root /root/a96ref):
  ref-panel                zero SUMO, write-once A96_PANEL_V.json (988900-988923)
  ref-bind --which W       zero SUMO, write-once BOUND_A77_<W>.json (MAPPO20C: env A96_MAPPO20C_POLICY / _SHA256 /
                           A96_TRAIN_RECEIPT / A96_TRAIN_PLAN; IDENT_U0 only in a synth root)
  ref-dry                  zero SUMO
  ref-run                  24 episodes (28,800 sim-s, cap in code), raw dropped
  ref-seal                 zero SUMO: A96_REF.json (wave receipts + 24 sealed reports + executor) - read by admission
Admission root commands (--final-root /root/a96adm_* --ref-root /root/a96ref/root):
  pin-top / pin (panel copy + comparator) / prepare / dry / pin-ref (after ref-seal) / mech / mech-compare / verify /
  admit / status / drop-raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a82_s3b20 as MA  # noqa: E402  (A82 part A driver, unchanged; re-pointed below)
import a76_d2p20 as M76  # noqa: E402
import a80_s2b as M80  # noqa: E402
import a75_d2forms as D2  # noqa: E402

AMENDMENT = '96'
STAGE = 'P20ADM'
CELL = MA.CELL                                   # D2800_p20
assert CELL == 'D2800_p20'
HORIZON = MA.HORIZON
FROZEN_U0_SHA256 = MA.FROZEN_U0_SHA256
A75_SHA256 = MA.A75_SHA256
BLOCK = (988900, 988923, 988999)                 # (lo, hi, runner block end)
SEEDS = tuple(range(BLOCK[0], BLOCK[1]+1))
assert len(SEEDS) == MA.VERIFY_N == 24
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (988200, 988891),
             (992400, 992523))
SETTING_V = 'a96v_D2800_p20'
PANEL_V = 'A96_PANEL_V.json'
REF_WAVE = 'A96V_MAPPO20C'
REF_TAG = 'a96r_mappo20c'
REF_PREFIX = 'a96r_'
REF_CAP = MA.VERIFY_N*HORIZON                    # 28,800
REF_FILE = 'A96_REF.json'
REF_PINS_FILE = 'A96_REF_PINS.json'
REFPIN_FILE = 'A96_REFPIN.json'                  # admission root: the reference pinned after ref-seal
PHASE_REF = 'ref_a96p20'
PHASE_ADM = 'holdforms_a96p20'
RANGES_ATTR = 'A96_RANGES'
AUTHORITY = ('MAIN AMENDMENT 96 part (b) (A96-P20ADM, 2026-11-14; POST-HOC, exploratory): SUMO admission at D2800_p20 '
             'of <= 12 plans per root vs MAPPO20C run once on 24 fresh worlds 988900-988923; A80/A82 admission code and '
             'rule unchanged (CI95 upper < 0 AND five-group rule; bank = best admitted per form)')
DISCLOSURE = ('POST-HOC and exploratory (Amendment 96). Plans run on the frozen U0 d9042cc5; the reference is MAPPO20C, '
              'so plan - reference mixes coordination and base policy (registered). 24 fresh worlds; up to 12 plans '
              'each tested at 95%; admission is a screen; never pooled. Labels inherited from the reused A76 / A82 code '
              '(schema coopmwm.a76.admission.v1, source "A76 (new run, D2800_p20)", wave names A82V_*) are cosmetic')
_RT = dict(mode=None, base=None, which_ref='MAPPO20C')
M76_ADMIT_RULE = M76.ADMIT_RULE                 # the unchanged A76 text (before any re-pointing)


def _N():
    return D2._mods()[0]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _A77():
    import a77_mappo20c as A77
    return A77


def _A52():
    import a52_mappo60 as A52
    return A52


# ================================================================================================ re-pointing

def block(which='v'):
    """(lo, hi, r0, r1) of the A96 block; the runner must carry A96_RANGES = ((988900, 988999),) as its own held-out
    row overlapping no other (a76_d2p20.check_blocks_free)."""
    N = _N()
    from tools.coopmwm import nm_eval_runner as R
    table = getattr(R, RANGES_ATTR, None)
    if which != 'v' or table is None or tuple(table) != ((BLOCK[0], BLOCK[2]),):
        raise N.Blocked(f'the runner has no {RANGES_ATTR} = ((988900, 988999),) (patch_tables_a96.py)')
    M76.check_blocks_free(table, R.EVAL_HELDOUT_RANGES)
    lo, hi, end = BLOCK
    if any(lo <= b1 and end >= b0 for b0, b1 in FORBIDDEN):
        raise N.Blocked('forbidden seed range')
    return lo, hi, lo, end


def register_setting(which, root):
    """a76_d2p20.register_setting with the A96 block / labels (process-local; nm_finaleval unchanged)."""
    N = _N()
    lo, hi, r0, r1 = block(which)
    name = SETTING_V
    N.SETTINGS[name] = dict(
        N.SETTINGS[M76.SOURCE_SETTING],
        N_WORLDS=hi-lo+1, EVAL_RANGE=(lo, r1), EVAL_EXTRA_RANGE=(lo, r1), SEEDS_RESERVED=(hi+1, r1),
        PANEL_SIZE_REASON=f'MAIN amendment 96: 20 % admission = {hi-lo+1} fresh worlds {lo}-{hi} (D2800_p20)',
        CELL_AUTHORITY='MAIN amendment 96 (A96-P20ADM, 2026-11-14): 20 % admission vs MAPPO20C at D2800_p20',
        PANEL_AUTHORITY=(f'A96: audited integers {lo}-{hi} (runner block {r0}-{r1}) for the 20 % admission ONLY '
                         '(reference MAPPO20C + plans); executed through the eval runner; never pooled'),
        EXPOSURE_EXTRA=dict(checked=('amendment-96 audit: Box 3 seed_audit_box.py 988900-988999 (0 overlaps)',),
                            disclosed='; amendment 96: fresh integers (20 % admission), registered at runtime'),
        PANEL_FILE=Path(root)/PANEL_V, PANEL_3C_FILE=None)
    if name not in N.BANK_SETTINGS:
        N.BANK_SETTINGS = N.BANK_SETTINGS+(name,)
    pin = (MA.read_pins(root) or {}).get('panels', {}).get('v') if _RT['mode'] == 'adm' else \
        (_read(Path(root)/REF_PINS_FILE) or {}).get('panel_sha256')
    N.SETTINGS_PINS[name] = dict(PANEL_SHA256=pin, PANEL_3C_SHA256=None)
    return name


def core():
    """The a42 driver code with the A96 names (own phase / cap / prefixes / runner table)."""
    N = _N()
    import a50_fullp60 as A50
    ref = _RT['mode'] == 'ref'
    over = dict(
        A42_PHASE=PHASE_REF if ref else PHASE_ADM, A42_CAP_SIM_S=REF_CAP if ref else MA.CAP_SIM_S, STOP_A42='STOP_A96',
        HOST_BASE=_RT['base'], CELL=CELL, VERIFY_SETTING=SETTING_V, CONFIRM_SETTING=SETTING_V,
        BANK_DIR=MA.BANK_DIR, IDENT_WAVE='A96_IDENT_UNUSED', AMENDMENT=AMENDMENT, LABEL='A96', JOB_PREFIX='a96',
        VERIFY_TAG='a96v', CONFIRM_TAG='a96c_unused', IDENT_TAG='a96i_unused', VERIFY_WAVE_PREFIX='A96V_',
        CONFIRM_WAVE_PREFIX='A96C_', RESULTS_FILE='A96_RESULTS_UNUSED.json', LOG_FILE=MA.LOG_FILE,
        SCHEMA_PREFIX='coopmwm.a96', RUNNER_RANGES_ATTR=RANGES_ATTR,
        RESULT_LABEL=('amendment 96 (b); POST-HOC exploratory 20 % ' + ('MAPPO20C reference run' if ref else
                      'admission vs the stored MAPPO20C reference') + '; one shot'), AUTHORITY=AUTHORITY)
    m = A50._clone('a96_core_' + ('ref' if ref else 'adm'), HERE/'a42_bigp60.py', over)
    m.HOST_ROOT = f'{m.HOST_BASE}/root'
    m.SETTINGS = (SETTING_V,)
    m.A42Runtime.phase = over['A42_PHASE']
    N.CHECKS['episode'] = D2.check_d2
    return m


def ref_specs(panel, refroot, comp, which=None):
    """= a82_fourarm.policy_specs / a77_mappo20c.confirm_specs: U0 specs (built on the reference root) through
    A52.with_executor with the a77-bound executor of the reference root."""
    base = M76.u0_specs(panel, refroot, comp, REF_TAG)
    return _A52().with_executor(base, _A77().read_bound(refroot, which or _RT['which_ref']), REF_TAG)


def _repoint_common(mode, base):
    _RT.update(mode=mode, base=str(base))
    M76.SETTING = dict(M76.SETTING, v=SETTING_V)
    M76.PANEL = dict(M76.PANEL, v=PANEL_V)
    M76.block = block
    M76.register_setting = register_setting


def repoint_adm(base, refroot, pins):
    """A82 part A names -> A96 (process-local), then a82_s3b20._point (which re-points a76_d2p20)."""
    _repoint_common('adm', base)
    MA.AMENDMENT, MA.STAGE, MA.PHASE = AMENDMENT, STAGE, PHASE_ADM
    MA.HOST_BASE_ENV = 'A96_HOST_BASE'
    MA.TOP_ENV, MA.TOP_SHA_ENV = 'A80_TOP_FILE', 'A80_TOP_SHA256'
    MA.TOP_COPY = f'A96_{MA.TOP_NAME}'
    MA.CANDS_DOC, MA.REFUSED_DOC = 'A96_CANDIDATES.json', 'A96_MAPPING_REFUSED.json'
    MA.PINS_FILE, MA.COMP_FILE, MA.BANK_DIR = 'A96_PINS.json', 'A96_COMPARATOR.json', 'a96_banks'
    MA.CANDS_BANK, MA.BANK = f'A96_CANDS_{CELL}.json', f'A96_BANK_{CELL}.json'
    MA.MECH_FILE, MA.ADMISSION_FILE = 'A96_MECH.json', f'A96_ADMISSION_{CELL}.json'
    MA.EXTRA_FILE, MA.LOG_FILE, MA.STOP_FILE = 'A96_ADMISSION_EXTRA.json', 'A96_LOG.jsonl', 'STOP_A96'
    MA.DROP_MANIFEST = 'A96_DROPPED_RAW.jsonl'
    MA.PREFIX = dict(i='a96i_unused_', m='a96m_unused_', v='a96v_', c='a96c_unused_')
    MA.REF_WAVE, MA.REF_TAG, MA.ARM = REF_WAVE, REF_TAG, 'A96_BANK'
    MA.WRITER = 'a75ss.main TOP_OFFER_CONDITIONS (D2800_p20 search) / A95(a) hand list (same writer)'
    MA.AUTHORITY, MA.DISCLOSURE = AUTHORITY, DISCLOSURE
    MA.ADMIT_RULE = M76_ADMIT_RULE.replace(
        'the comparator runs on the same worlds in this root (A76V_COMP)',
        'the reference = the STORED A96V_MAPPO20C wave of the A96 reference root (MAPPO20C on the same 24 worlds; '
        'run once; wave receipts and per-vehicle reports pinned); plans on U0 d9042cc5')
    assert MA.ADMIT_RULE != M76_ADMIT_RULE
    MA.host_base = lambda env=None: str(base)
    MA.core = core
    MA.a76_root = lambda env=None: Path(refroot)

    def admit_waves(D, panel, root, comp, cands):
        return [(REF_WAVE, 'episode', ref_specs(panel, refroot, comp))] + MA.plan_waves(D, panel, root, comp, cands)
    MA.admit_waves = admit_waves
    MA._point(refroot, pins)
    ref_sha = (pins or {}).get('ref', {}).get('executor_sha256')

    def _executor_check(have, comp, what):
        N = _N()
        want = ref_sha if what == REF_WAVE else comp['sha256']
        if what == REF_WAVE and not ref_sha:
            raise N.Blocked('reference executor not pinned (run pin-ref)')
        for w, r in have.items():
            got = r.get('executor_sha256_loaded')
            if got != want:
                raise N.Blocked(f'{what} {w}: executor {str(got)[:8]} is not {want[:8]}')
    M76._executor_check = _executor_check


def repoint_ref(base):
    _repoint_common('ref', base)
    M76.core = core
    M76.HOST_BASE = str(base)


# ================================================================================================ helpers

def _read(p):
    N = _N()
    return N.verify(N.read(p), 'receipt_sha256', p.name) if Path(p).is_file() else None


def comparator(root):
    N = _N()
    comp = M76.comparator(root, env={M76.COMP_SHA_ENV: FROZEN_U0_SHA256})
    if comp['rebind'] or comp['sha256'] != FROZEN_U0_SHA256:
        raise N.Blocked('plans run on the frozen U0 d9042cc5')
    return comp


def load_panel(root, want=None):
    N = _N()
    panel = N.load_panel(Path(root)/PANEL_V, pinned=want)
    if tuple(w['world']['seed'] for w in panel['worlds']) != SEEDS or panel['cell']['cell_key'] != CELL:
        raise N.Blocked(f'{PANEL_V}: not {CELL} 988900-988923')
    if dict(panel['contract']['bootstrap']) != M76.BOOTSTRAP:
        raise N.Blocked('panel bootstrap is not the registered one')
    return panel


# ================================================================================================ reference root

def ref_setup(root):
    N = _N()
    D = core()
    N.apply_setting(register_setting('v', root))
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return D


def ref_panel(root):
    N = _N()
    import a37_celltailor as C37
    from datetime import datetime, timezone
    root = Path(root)
    ref_setup(root)
    p = root/PANEL_V
    if p.exists():
        raise N.Blocked(f'{p.name} exists (write-once)')
    value = N.build_panel(N.default_spec_path(), datetime.now(timezone.utc).isoformat())
    if tuple(w['world']['seed'] for w in value['worlds']) != SEEDS or value['cell']['cell_key'] != CELL:
        raise N.Blocked('panel: wrong seeds / cell')
    C37.write_once(p, json.dumps(value, indent=1, sort_keys=True)+'\n')
    pins = N.seal(dict(schema='coopmwm.a96.ref_pins.v1', amendment=AMENDMENT, cell=CELL, seeds=[SEEDS[0], SEEDS[-1]],
                       panel_file=PANEL_V, panel_file_sha256=N.file_sha(p), panel_sha256=value['panel_sha256'],
                       authority=AUTHORITY), 'receipt_sha256')
    N.publish(root/REF_PINS_FILE, pins)
    return dict(file_sha256=pins['panel_file_sha256'], panel_sha256=pins['panel_sha256'], seeds=pins['seeds'])


def ref_bind(root, which, env=None):
    env = os.environ if env is None else env
    A77 = _A77()
    if which == 'IDENT_U0':
        if not env.get('A96_SYNTH'):
            raise _N().Blocked('IDENT_U0 is a synthetic-dry stand-in only (A96_SYNTH=1 in a synth root)')
        import a32_ablation as A
        u0, _ = A.bindings(root)
        return A77.bind(root, 'IDENT_U0', u0['path'])
    return A77.bind(root, 'MAPPO20C', env['A96_MAPPO20C_POLICY'], sha=env['A96_MAPPO20C_SHA256'],
                    receipt=env['A96_TRAIN_RECEIPT'], plan=env['A96_TRAIN_PLAN'])


def ref_context(root):
    N = _N()
    root = Path(root)
    pins = _read(root/REF_PINS_FILE)
    if pins is None:
        raise N.Blocked(f'{REF_PINS_FILE} missing (run ref-panel)')
    D = ref_setup(root)
    panel = load_panel(root, pins['panel_sha256'])
    comp = comparator(root)
    bound = _A77().read_bound(root, _RT['which_ref'])
    specs = ref_specs(panel, root, comp)
    if len(specs) != MA.VERIFY_N or any(s['checkpoint']['executor_sha256'] != bound['sha256'] for s in specs):
        raise N.Blocked('reference specs: size / executor')
    return D, panel, comp, bound, [(REF_WAVE, 'episode', specs)]


def ref_dropper():
    import a50_fullp60 as A50
    return A50._clone('a96_ref_drop', Path(D2.__file__), dict(
        PREFIX=dict(i='a96i_unused_', m='a96m_unused_', v=REF_PREFIX, c='a96c_unused_'), AMENDMENT=AMENDMENT,
        STAGE=STAGE, DROP_MANIFEST='A96_REF_DROPPED_RAW.jsonl',
        DROP_RULE=D2.DROP_RULE.replace('a75d2v_* / a75d2c_*', REF_PREFIX+'*')))


def ref_seal(root):
    """Zero SUMO, write-once: the published reference wave (receipts, executor, 24 DROPPED_RAW-sealed reports)."""
    N = _N()
    root = Path(root)
    if (root/REF_FILE).is_file():
        raise N.Blocked(f'{REF_FILE} exists')
    D, panel, comp, bound, waves = ref_context(root)
    (name, _, specs), = waves
    have, src = M76._load(root, name, specs)
    reports = {}
    for w in [r['world_id'] for r in panel['worlds']]:
        rec = have[w]
        if rec.get('executor_sha256_loaded') != bound['sha256']:
            raise N.Blocked(f'{w}: executor is not the bound reference')
        reports[rec.get('work_dir', rec['job_id'])] = MA.ref_episode_check(root, rec, None)
    rs = src['receipt_sha256'] if isinstance(src['receipt_sha256'], list) else [src['receipt_sha256']]
    v = N.seal(dict(schema='coopmwm.a96.ref.v1', amendment=AMENDMENT, cell=CELL, wave=REF_WAVE, wave_receipts=rs,
                    reports=reports, n=len(reports), executor_sha256=bound['sha256'], executor=_RT['which_ref'],
                    binding_sha256=bound['binding_sha256'], panel_sha256=panel['panel_sha256'],
                    charged=D.charged(root), note='hashes only; no outcome is read by this step'), 'receipt_sha256')
    if v['n'] != MA.VERIFY_N:
        raise N.Blocked('reference size')
    N.publish(root/REF_FILE, v)
    return v


# ================================================================================================ admission root

def adm_pin(root, refroot):
    """Zero SUMO, write-once: panel byte copy from the reference root + comparator (U0); no reference yet."""
    N = _N()
    import a37_celltailor as C37
    root, refroot = Path(root), Path(refroot)
    if MA.read_pins(root) is not None:
        raise N.Blocked(f'{MA.PINS_FILE} exists (write-once)')
    rp = _read(refroot/REF_PINS_FILE)
    if rp is None:
        raise N.Blocked('reference root has no panel pins')
    src = refroot/PANEL_V
    if N.file_sha(src) != rp['panel_file_sha256']:
        raise N.Blocked(f'{src}: not the pinned reference panel')
    q = C37.write_once(root/PANEL_V, src.read_bytes().decode('utf-8'))
    if N.file_sha(q) != rp['panel_file_sha256']:
        raise N.Blocked('panel copy is not byte-identical')
    D, _ = M76.setup('v', root, need_panel=False)
    panel = load_panel(root, rp['panel_sha256'])
    comp = comparator(root)
    v = N.seal(dict(schema='coopmwm.a96.pins.v1', amendment=AMENDMENT, stage=STAGE, cell=CELL,
                    panels=dict(v=panel['panel_sha256']), panel_files=dict(v=N.file_sha(q)),
                    seeds=dict(v=[SEEDS[0], SEEDS[-1]]), ref_root=str(refroot),
                    ref_pins_receipt_sha256=rp['receipt_sha256'], comparator=M76.comp_record(comp),
                    a75_sha256=MA._H().a75_digest(), caps=dict(verify=MA.VERIFY_CAP, total=MA.CAP_SIM_S),
                    note='reference pinned later by pin-ref (A96_REFPIN.json), before any plan episode'),
               'receipt_sha256')
    if v['a75_sha256'] != A75_SHA256:
        raise N.Blocked('a75 digest')
    N.publish(root/MA.PINS_FILE, v)
    return v


def adm_pin_ref(root, refroot):
    N = _N()
    root, refroot = Path(root), Path(refroot)
    if (root/REFPIN_FILE).is_file():
        raise N.Blocked(f'{REFPIN_FILE} exists')
    ref = _read(refroot/REF_FILE)
    if ref is None:
        raise N.Blocked(f'{refroot/REF_FILE}: missing (reference stage not sealed)')
    pins = MA.read_pins(root)
    if ref['panel_sha256'] != pins['panels']['v']:
        raise N.Blocked('reference panel differs from this root')
    v = N.seal(dict(schema='coopmwm.a96.refpin.v1', amendment=AMENDMENT, ref_root=str(refroot),
                    ref_receipt_sha256=ref['receipt_sha256'], wave=ref['wave'], wave_receipts=ref['wave_receipts'],
                    reports=ref['reports'], n=ref['n'], executor_sha256=ref['executor_sha256'],
                    binding_sha256=ref['binding_sha256']), 'receipt_sha256')
    N.publish(root/REFPIN_FILE, v)
    return v


def merged_pins(root):
    pins = MA.read_pins(root)
    rp = _read(Path(root)/REFPIN_FILE)
    if pins is None:
        return None
    out = dict(pins)
    if rp is not None:
        out['ref'] = dict(wave=rp['wave'], wave_receipts=rp['wave_receipts'], reports=rp['reports'], n=rp['n'],
                          executor_sha256=rp['executor_sha256'])
        out['receipt_sha256'] = rp['receipt_sha256']
    return out


# ================================================================================================ CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command')
    p.add_argument('--final-root', required=True)
    p.add_argument('--ref-root')
    p.add_argument('--which', choices=('MAPPO20C', 'IDENT_U0'))
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=96)
    p.add_argument('--drop-procs', type=int, default=16)
    a = p.parse_args(argv)
    root = Path(a.final_root)
    base = root.parent
    if os.environ.get('A96_SYNTH'):
        _RT['which_ref'] = 'IDENT_U0'
    N = _N()

    def log(x, lf):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=SETTING_V), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/lf).open('a', encoding='utf-8') as f:
            f.write(line+'\n')

    if a.command.startswith('ref-'):
        repoint_ref(base)
        if a.command == 'ref-panel':
            print(json.dumps(ref_panel(root)))
        elif a.command == 'ref-bind':
            v = ref_bind(root, a.which)
            print(json.dumps(dict(which=a.which, sha256=v['sha256'], binding_sha256=v['binding_sha256'])))
        elif a.command == 'ref-dry':
            D, panel, comp, bound, waves = ref_context(root)
            (n, _, s), = waves
            print(json.dumps(dict(stage='ref', panel_sha256=panel['panel_sha256'], wave=n, n=len(s),
                                  seeds=[s[0]['world']['seed'], s[-1]['world']['seed']], job0=s[0]['job_id'],
                                  executor=s[0]['checkpoint']['executor_sha256'][:12], which=_RT['which_ref'],
                                  sim_s=len(s)*HORIZON, cap=REF_CAP, charged=D.charged(root)), indent=1))
        elif a.command == 'ref-run':
            import fcntl
            if _RT['which_ref'] != 'MAPPO20C':
                raise N.Blocked('physics only with the MAPPO20C binding')
            D, panel, comp, bound, waves = ref_context(root)
            with (root/'OWNER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in waves)
                if D.charged(root)+pending*HORIZON > REF_CAP:
                    raise N.Blocked(f'reference cap {REF_CAP}')
                X = ref_dropper()
                lf = 'A96_REF_LOG.jsonl'
                log(dict(start='ref-run', pending=pending, workers=a.workers, charged_before=D.charged(root)), lf)

                def hook(x):
                    log(x, lf)
                    if 'round' in x:
                        X.drop_raw(root, waves, procs=a.drop_procs, log=lambda y: log(y, lf))
                N.CHECKS['episode'] = D2.check_d2
                res = D2.execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
                X.drop_raw(root, waves, procs=a.drop_procs, log=lambda y: log(y, lf))
                log(dict(done='ref-run', **res), lf)
        elif a.command == 'ref-seal':
            v = ref_seal(root)
            print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n=v['n'], executor=v['executor_sha256'][:12],
                                  wave_receipts=v['wave_receipts'])))
        else:
            raise SystemExit(f'unknown command {a.command}')
        return

    refroot = Path(a.ref_root or '/root/a96ref/root')
    if root.resolve() == refroot.resolve():
        raise N.Blocked('admission root != reference root')
    if a.command == 'pin-top':
        repoint_adm(base, refroot, None)
        print(json.dumps(MA.pin_top(root)))
        return
    if a.command == 'pin':
        repoint_adm(base, refroot, None)
        v = adm_pin(root, refroot)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], panel=v['panels']['v'])))
        return
    if a.command == 'pin-ref':
        repoint_adm(base, refroot, None)
        v = adm_pin_ref(root, refroot)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n=v['n'], executor=v['executor_sha256'][:12])))
        return
    repoint_adm(base, refroot, None)                 # A96 file names first, then the pinned reference
    pins = merged_pins(root)
    if pins is None:
        raise N.Blocked('run pin first')
    repoint_adm(base, refroot, pins)
    if a.command == 'prepare':
        M76.setup('v', root, need_panel=False)
        print(json.dumps(MA.prepare(root)))
        return
    D, panel = M76.setup('v', root)
    if panel['panel_sha256'] != pins['panels']['v']:
        raise N.Blocked('panel is not the pinned one')
    comp = comparator(root)
    if M76.comp_record(comp) != pins['comparator']:
        raise N.Blocked('comparator differs from the pinned one')
    cands = MA.candidates(root)
    lf = MA.LOG_FILE
    if a.command in ('mech', 'verify', 'admit') and 'ref' not in pins:
        raise N.Blocked('reference not pinned (run pin-ref after the reference stage)')
    if a.command == 'dry':
        ws = MA.plan_waves(D, panel, root, comp, cands)
        mw = MA.mech_waves(D, panel, root, comp, cands)
        out = dict(stage='v', panel_sha256=panel['panel_sha256'], comparator=comp['sha256'][:12], n_waves=len(ws),
                   waves=[(n, len(s), s[0]['arm'], s[0]['job_id'], s[0]['world']['seed'], s[-1]['world']['seed'],
                           s[0]['checkpoint']['executor_sha256'][:8]) for n, _, s in ws],
                   sim_s=sum(len(s)*HORIZON for _, _, s in ws),
                   mech=dict(n_waves=len(mw), sim_s=sum(len(s)*HORIZON for _, _, s in mw)),
                   charged=D.charged(root), cap=MA.CAP_SIM_S, a75_sha256=MA._H().a75_digest())
        rb = refroot/'BOUND_A77_MAPPO20C.json'
        sb = refroot/'BOUND_A77_IDENT_U0.json'
        if rb.is_file() or sb.is_file():
            rs = MA.admit_waves(D, panel, root, comp, cands)[0][2]
            out['reference_specs'] = dict(n=len(rs), job0=rs[0]['job_id'], executor=rs[0]['checkpoint']['executor_sha256'][:12],
                                          which=_RT['which_ref'])
        if 'ref' in pins:
            (rn, _, rsp), = MA.admit_waves(D, panel, root, comp, cands)[:1]
            have, src = MA._load(root, rn, rsp)
            reps = {w: MA._mods_proxy()[1].report_of(root, have[w]) for w in have}
            out['reference'] = dict(wave=rn, n=len(have), receipts=src['receipt_sha256'],
                                    reports_ok=all(x.get('ok') for x in reps.values()))
        print(json.dumps(out, indent=1, default=str))
    elif a.command in ('mech', 'verify'):
        import fcntl
        if MA._H().a75_digest() != A75_SHA256:
            raise N.Blocked('a75_holdforms in this tree is not the registered executor d1af7e0f')
        if a.command == 'verify' and not MA.mech_passed(root):
            raise N.Blocked('mechanics is not PASS: nothing runs')
        if a.command == 'mech' and (root/MA.MECH_FILE).is_file():
            raise N.Blocked(f'{MA.MECH_FILE} exists: mechanics ran')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            waves = (MA.mech_waves if a.command == 'mech' else MA.plan_waves)(D, panel, root, comp, cands)
            pending = sum(len(s)-N.published(root, w, s, D2.check_d2)[2] for w, _, s in waves)
            MA.cap_check(D.charged(root, MA.PREFIX['v']), D.charged(root), pending*HORIZON)
            drop = a.command == 'verify'
            log(dict(start=a.command, n_waves=len(waves), pending=pending, workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=D.charged(root), drop_raw=drop), lf)
            X = MA.dropper() if drop else None

            def hook(x):
                log(x, lf)
                if drop and 'round' in x:
                    X.drop_raw(root, waves, procs=a.drop_procs, log=lambda y: log(y, lf))
            if drop:
                X.drop_raw(root, waves, procs=a.drop_procs, log=lambda y: log(y, lf))
            N.CHECKS['episode'] = D2.check_d2
            res = D2.execute(D, root, waves, workers=a.workers, chunk=a.chunk, log=hook, panel=panel)
            log(dict(done=a.command, **res), lf)
    elif a.command == 'mech-compare':
        if (root/MA.MECH_FILE).is_file():
            raise N.Blocked(f'{MA.MECH_FILE} exists (one shot)')
        v = MA.mech_compare(D, panel, root, comp, cands)
        N.publish(root/MA.MECH_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'], failures=v['failures'][:20])))
        if v['verdict'] != 'PASS':
            sys.exit(5)
    elif a.command == 'drop-raw':
        print(json.dumps(MA.dropper().drop_raw(root, MA.plan_waves(D, panel, root, comp, cands), procs=a.drop_procs,
                                               log=lambda y: log(y, lf))))
    elif a.command == 'admit':
        if not MA.mech_passed(root):
            raise N.Blocked('mechanics is not PASS')
        v = MA.admit(D, panel, root, comp, cands)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], n_tested=v['n_tested'], n_admitted=v['n_admitted'],
                              n_delay_pass=v['n_delay_pass'], per_form=v['per_form'], bank=v['bank'],
                              consequence=v['consequence'])))
    elif a.command == 'status':
        ws = MA.plan_waves(D, panel, root, comp, cands)
        done = [N.published(root, n, s, D2.check_d2)[2] for n, _, s in ws]
        print(json.dumps(dict(charged=D.charged(root), cap=MA.CAP_SIM_S, mech=MA.mech_passed(root),
                              ref_pinned='ref' in pins, admission=(root/MA.ADMISSION_FILE).is_file(),
                              episodes=f'{sum(done)}/{sum(len(s) for _, _, s in ws)}'), indent=1))
    else:
        raise SystemExit(f'unknown command {a.command}')


if __name__ == '__main__':
    main()
