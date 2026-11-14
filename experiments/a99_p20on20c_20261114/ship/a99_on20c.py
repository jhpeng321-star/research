"""AMENDMENT 99 (S-P20-ON-20C; POST-HOC, exploratory): the A96(b) 20 % admission driver ops/a96_p20adm.py (85cdef84,
imported UNCHANGED) with ONE change - the plan executor's base policy is MAPPO20C 606e59e0 instead of U0 d9042cc5.

How (process-local, no shipped file edited):
  * the a76 'comparator' of an admission root (= the executor the plan specs get through a76_d2p20.with_exe, and the
    sha every plan record must carry in the admit / mech executor checks) is the A77 MAPPO20C binding sealed in the A96
    reference root (/root/a96ref/root/BOUND_A77_MAPPO20C.json) - the very six executor fields the stored reference ran;
  * the stored reference specs are rebuilt exactly as in A96 (U0 comparator, no rebind -> A52.with_executor MAPPO20C),
    so the stored A96V_MAPPO20C wave is read unchanged (pinned receipts);
  * receipt text (authority / disclosure / admit-rule wording) says MAPPO20C; file names stay A96_* (cosmetic).
Identity gate (A67 rule; base /root/a99ident only): runner arm ORACLE (oracle_plan.OracleCore) on the bound p20c oracle
bank, which must admit NO plan for D2800_p20 (checked, zero physics), with the six executor fields = MAPPO20C, worlds
988900-988903; PASS iff J_veh_s difference 0 and identical safety counters vs the stored A96V_MAPPO20C records, zero
plans / activations / issued. Cap 4,800 sim-s. Admission commands refuse unless IDENTITY_A99.json is PASS.
Extra commands: ident-dry | ident-run | ident-compare. Every other command -> a96_p20adm.main.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import a96_p20adm as M96  # noqa: E402  (unchanged)

M76, MA, D2 = M96.M76, M96.MA, M96.D2
REFROOT = Path(os.environ.get('A99_REF_ROOT', '/root/a96ref/root'))
IDENT_BASE = Path('/root/a99ident')
IDENTITY_FILE = 'IDENTITY_A99.json'
IDENT_WAVE = 'A99I_EMPTY_ON20C_R2'           # R2: wave name of failed launch 1 holds a stale reservation receipt
IDENT_TAG = 'a96i_a99empty'                   # runner reserve() requires the a96 job prefix
IDENT_N = 4
IDENT_CAP = IDENT_N*M96.HORIZON                  # 4,800
MAPPO20C_SHA256 = '606e59e0f579a4c7c3be88b44da2237ce9ddbdc065780dbd4b385128775ad659'
EXEC_FIELDS = ('comparator_kind', 'label', 'executor_path', 'executor_sha256', 'binding_sha256', 'binding_path')

M96.AUTHORITY = ('MAIN AMENDMENT 99 (S-P20-ON-20C, 2026-11-14; POST-HOC, exploratory): the A96(b) S4-p20 / S5-p20 '
                 'candidates admitted with the plan executor on MAPPO20C 606e59e0 vs the stored A96 MAPPO20C reference '
                 '(988900-988923); A96(b) code and rule unchanged (CI95 upper < 0 AND five-group rule; bank = best '
                 'admitted per form)')
M96.DISCLOSURE = ('POST-HOC and exploratory (Amendment 99). Plans run on MAPPO20C 606e59e0 (plan executor base policy); '
                  'the reference is the stored A96V_MAPPO20C wave, so plan - reference isolates the plans on the same base. '
                  'Worlds 988900-988923 reused from A96(b) (already viewed); 12 plans per search each tested at 95%; '
                  'a screen; never pooled. File names A96_* / waves A82V_* / schema coopmwm.a76.admission.v1 / source '
                  '"A76 (new run, D2800_p20)" are cosmetic labels of the reused code')
_ORIG = dict(comparator=M76.comparator, ref_specs=M96.ref_specs, repoint_adm=M96.repoint_adm)


def _N():
    return M96._N()


def mappo_comp():
    """The A77 MAPPO20C binding of the reference root as an a76 comparator dict (rebind: six fields replaced)."""
    N = _N()
    b = M96._A77().read_bound(REFROOT, 'MAPPO20C')
    if b['sha256'] != MAPPO20C_SHA256 or N.file_sha(b['path']) != MAPPO20C_SHA256:
        raise N.Blocked('reference-root MAPPO20C binding is not 606e59e0')
    return dict(rebind=True, name='MAPPO20C', kind=b['kind'], label=b['label'], path=b['path'], sha256=b['sha256'],
                binding_sha256=b['binding_sha256'], binding_path=b['binding_path'], export=b.get('export'))


def comparator(root, env=None):
    N = _N()
    comp = mappo_comp()
    rec = Path(root)/M76.COMP_FILE
    if rec.is_file():
        got = N.verify(N.read(rec), 'receipt_sha256', rec.name)
        if got['comparator'] != M76.comp_record(comp):
            raise N.Blocked(f'{rec.name}: this root is bound to another comparator')
    return comp


def u0_comp():
    c = _ORIG['comparator'](REFROOT, env={M76.COMP_SHA_ENV: M96.FROZEN_U0_SHA256})
    if c['rebind'] or c['sha256'] != M96.FROZEN_U0_SHA256:
        raise _N().Blocked('reference specs need the frozen U0 comparator')
    return c


def ref_specs(panel, refroot, comp, which=None):
    """The stored reference specs exactly as A96 built them (U0 comparator -> A52.with_executor MAPPO20C)."""
    return _ORIG['ref_specs'](panel, refroot, u0_comp(), which)


def repoint_adm(base, refroot, pins):
    _ORIG['repoint_adm'](base, refroot, pins)
    rule = MA.ADMIT_RULE.replace('plans on U0 d9042cc5', 'plans on MAPPO20C 606e59e0 (A99: plan executor base policy)')
    assert rule != MA.ADMIT_RULE
    MA.ADMIT_RULE = M76.ADMIT_RULE = rule
    MA.AUTHORITY = M76.AUTHORITY = M96.AUTHORITY
    MA.DISCLOSURE = M76.DISCLOSURE = M96.DISCLOSURE


M76.comparator = comparator
M96.comparator = comparator
M96.ref_specs = ref_specs
M96.repoint_adm = repoint_adm


# ================================================================================================ identity

def identity_passed():
    N = _N()
    p = IDENT_BASE/'root'/IDENTITY_FILE
    return p.is_file() and N.verify(N.read(p), 'receipt_sha256', p.name).get('verdict') == 'PASS'


EMPTY_BANK = 'oracle_banks/A99_EMPTY_ORACLE_BANK.json'
FROZEN_ORACLE_BANK_SHA256 = 'fd385a2b8b1cbaf33897f8617b6d4c0abeb9f2dea40c9b72dca91d718bfc3c82'


def empty_bank(root, write=False):
    """The frozen p20c oracle bank fd385a2b admits 5 plans at D2800_p20 (it is NOT empty at 20 %, unlike A67 at 60 %),
    so the empty plan set is that bank document with rows = [] (every other key byte-for-byte the same JSON value),
    written once into the identity root. Zero physics."""
    N = _N()
    from tools.coopmwm import oracle_plan as O
    _, A, _ = M76._mods()
    _, oracle = A.bindings(root)
    frozen = Path(root)/'oracle_banks'/Path(N.arm_bank(oracle, 'ORACLE')['path']).name
    if N.arm_bank(oracle, 'ORACLE')['file_sha256'] != FROZEN_ORACLE_BANK_SHA256 or N.file_sha(frozen) != FROZEN_ORACLE_BANK_SHA256:
        raise N.Blocked('bound oracle bank is not fd385a2b')
    p = Path(root)/EMPTY_BANK
    if write and not p.exists():
        doc = json.loads(frozen.read_text(encoding='utf-8'))
        doc['rows'] = []
        p.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding='utf-8')
    if not p.is_file():
        raise N.Blocked(f'{p}: missing (ident-bank)')
    plans, refused = O.bank_plans(N.read(p), M96.CELL, None)
    if plans or refused:
        raise N.Blocked(f'empty bank not empty for {M96.CELL}')
    return dict(path=str(p), file_sha256=N.file_sha(p), source=str(frozen), source_sha256=FROZEN_ORACLE_BANK_SHA256,
                cell=M96.CELL, plans=0, refused=refused)


def ident_specs(panel, root):
    N = _N()
    _, A, _ = M76._mods()
    eb = empty_bank(root)
    u0, oracle = A.bindings(root)
    base = N.episode_specs('ORACLE', panel, None, Path(root)/N.PRERUN_PLANT, 'mappo_frozen', oracle, u0)[:IDENT_N]
    exe = mappo_comp()
    out = []
    for i, s in enumerate(base):
        ck = deepcopy(s['checkpoint'])
        if ck.get('executor_sha256') != M96.FROZEN_U0_SHA256 or ck.get('kind') != 'oracle':
            raise N.Blocked(f'{s["job_id"]}: not a u0 ORACLE spec')
        ck.update(comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                  executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'], binding_path=exe['binding_path'])
        ck['oracle_bank'] = dict(path=eb['path'], file_sha256=eb['file_sha256'])
        out.append(dict(s, checkpoint=ck, job_id=f'{IDENT_TAG}_{i:04}'))
    if [s['world']['seed'] for s in out] != list(M96.SEEDS[:IDENT_N]):
        raise N.Blocked('identity worlds are not 988900-988903')
    return out


def stored_ref(panel):
    """Stored A96V_MAPPO20C records (reference root, read-only; wave receipts == A96_REF.json)."""
    N = _N()
    ref = N.verify(N.read(REFROOT/M96.REF_FILE), 'receipt_sha256', M96.REF_FILE)
    specs = ref_specs(panel, REFROOT, None)
    have, src = MA._ORIG['load'](REFROOT, M96.REF_WAVE, specs)
    got = src['receipt_sha256'] if isinstance(src['receipt_sha256'], list) else [src['receipt_sha256']]
    if got != ref['wave_receipts']:
        raise N.Blocked('stored reference wave receipts differ from A96_REF.json')
    return have, specs


def ident_compare(panel, root):
    N = _N()
    specs = ident_specs(panel, root)
    got, _, cov = N.published(root, IDENT_WAVE, specs, D2.check_d2)
    if cov != len(specs):
        raise N.Blocked(f'{IDENT_WAVE}: {cov}/{len(specs)} published')
    ref, rspecs = stored_ref(panel)
    rows = []
    for r in got:
        a = ref[r['world_id']]
        o = r.get('oracle') or {}
        rows.append(dict(world_id=r['world_id'], seed=r['world'].get('seed'), j_a99=r['J_veh_s'], j_stored=a['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-a['J_veh_s']), safety_a99=r['safety'], safety_stored=a['safety'],
                         safety_equal=r['safety'] == a['safety'], same_world=r['world'] == a['world'],
                         executor_a99=r.get('executor_sha256_loaded'), executor_stored=a.get('executor_sha256_loaded'),
                         plans=len(o.get('plans') or []), activations=int(o.get('activations') or 0),
                         issued=int(o.get('issued') or 0)))
    ok = (len(rows) == IDENT_N and len({x['world_id'] for x in rows}) == IDENT_N
          and all(x['abs_diff'] == 0.0 and x['safety_equal'] and x['same_world']
                  and x['executor_a99'] == MAPPO20C_SHA256 == x['executor_stored']
                  and x['plans'] == 0 and x['activations'] == 0 and x['issued'] == 0 for x in rows))
    return N.seal(dict(schema='coopmwm.a99.identity.v1', amendment='99',
                       rule=('the plan-executor path with an empty plan set (runner arm ORACLE, oracle_plan.OracleCore, '
                             'bound p20c oracle bank: no plan for D2800_p20) with the six executor fields = MAPPO20C '
                             '606e59e0 (reference-root A77 binding) reproduces the stored A96V_MAPPO20C records on '
                             '988900-988903: J_veh_s difference 0, identical safety counters, zero plans / activations'),
                       verdict='PASS' if ok else 'FAILED', worlds=rows,
                       ident_specs_executor={k: specs[0]['checkpoint'][k] for k in EXEC_FIELDS},
                       ref_specs_executor={k: rspecs[0]['checkpoint'][k] for k in EXEC_FIELDS},
                       consequence='PASS: A99 admission may run; FAILED: stop and report, nothing else runs'),
                  'receipt_sha256')


def ident_main(a):
    N = _N()
    root = Path(a.final_root)
    base = root.parent
    if base != IDENT_BASE:
        raise N.Blocked('identity commands only in /root/a99ident')
    M96.repoint_adm(base, REFROOT, None)
    pins = M96.merged_pins(root)
    if pins is None or 'ref' not in pins:
        raise N.Blocked('run pin and pin-ref in the identity root first')
    M96.repoint_adm(base, REFROOT, pins)
    D, panel = M76.setup('v', root)
    if panel['panel_sha256'] != pins['panels']['v']:
        raise N.Blocked('panel is not the pinned one')
    if a.command == 'ident-bank':
        print(json.dumps(empty_bank(root, write=True)))
        return
    specs = ident_specs(panel, root)
    waves = [(IDENT_WAVE, 'episode', specs)]
    if a.command == 'ident-dry':
        _, rs = stored_ref(panel)
        print(json.dumps(dict(empty=empty_bank(root), n=len(specs), oracle_bank=specs[0]['checkpoint']['oracle_bank'], arm=specs[0]['arm'], kind=specs[0]['checkpoint']['kind'],
                              jobs=[s['job_id'] for s in specs], seeds=[s['world']['seed'] for s in specs],
                              ident_exec={k: specs[0]['checkpoint'][k] for k in EXEC_FIELDS},
                              ref_exec={k: rs[0]['checkpoint'][k] for k in EXEC_FIELDS},
                              exec_equal=all(specs[0]['checkpoint'][k] == rs[0]['checkpoint'][k] for k in EXEC_FIELDS),
                              ref_n=len(rs), charged=D.charged(root), cap=IDENT_CAP), indent=1, default=str))
    elif a.command == 'ident-run':
        import fcntl
        if (root/IDENTITY_FILE).is_file():
            raise N.Blocked('identity already compared')
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pending = len(specs)-N.published(root, IDENT_WAVE, specs, D2.check_d2)[2]
            if D.charged(root)+pending*M96.HORIZON > IDENT_CAP:
                raise N.Blocked(f'identity cap {IDENT_CAP}')
            N.CHECKS['episode'] = D2.check_d2

            def log(x):
                print(json.dumps(dict(x, t=time.strftime('%F %T')), default=str), flush=True)
            log(dict(start='ident-run', pending=pending, charged_before=D.charged(root)))
            res = D2.execute(D, root, waves, workers=min(a.workers, IDENT_N), chunk=IDENT_N, log=log, panel=panel)
            log(dict(done='ident-run', charged=D.charged(root), **res))
    elif a.command == 'ident-compare':
        if (root/IDENTITY_FILE).is_file():
            raise N.Blocked('identity already compared (one shot)')
        v = ident_compare(panel, root)
        N.publish(root/IDENTITY_FILE, v)
        print(json.dumps(dict(receipt_sha256=v['receipt_sha256'], verdict=v['verdict'],
                              worlds=[(x['world_id'], x['abs_diff'], x['safety_equal'], x['activations'])
                                      for x in v['worlds']]), default=str))
        if v['verdict'] != 'PASS':
            sys.exit(5)


def main(argv=None):
    import argparse
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else ''
    if cmd.startswith('ident-'):
        p = argparse.ArgumentParser()
        p.add_argument('command', choices=('ident-bank', 'ident-dry', 'ident-run', 'ident-compare'))
        p.add_argument('--final-root', required=True)
        p.add_argument('--ref-root')
        p.add_argument('--workers', type=int, default=4)
        ident_main(p.parse_args(argv))
        return
    if cmd in ('mech', 'verify', 'admit') and not identity_passed():
        raise _N().Blocked(f'{IDENT_BASE}/root/{IDENTITY_FILE} is not PASS: nothing runs')
    if cmd.startswith('ref-'):
        raise _N().Blocked('A99 never runs or seals a reference (stored A96 reference only)')
    M96.main(argv)


if __name__ == '__main__':
    main()
