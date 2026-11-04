"""CPU tests for the c4/rerun-cpus standalone prep (no SUMO, no GPU, no ssh/remote).

Covers the owner 2026-11-04 requirements:
  own ledger    - init_root creates the root's OWN single ledger (CHARGES.jsonl + lock +
                  AUTHORITY.md) and refuses an existing root, a symlinked/shared ledger or a
                  missing/wrong authority; every charge lands in THIS root's ledger only;
  foreign ledger- a ledger shared with another root (hard link or symlink) is refused;
  3-arm pins    - ALL THREE arms A/B/R from the common MAPPO u0 with identical code, B/R with
                  the full fix set (kl_targeted, per_seed_v1 cooperative_only, practice_starts
                  0.25/cap 2, imitation_target_space 'effective'), A the same runtime minus
                  bank/practice; the effective-imitation commit pinned to f35b193a;
  A results local - control arm A runs in the SAME new root (no remote A reads);
  quote         - 3 arms incl. the exact gate, with the u8/u24 test numbers written out;
  staging       - the read-only manifest resolves the bank through the REAL readers
                  (bank_decisions + per_seed_v1 confirmations on a real all8 bank built by the
                  test verifiers) relative to /home/ubuntu, and the dry preflight refuses
                  without a fully bound OWNER_APPROVAL.json and byte-exact staged files.
"""
import gzip
import hashlib
import json
import os
import shutil

import pytest
import torch

from tools.coopnative.storage import canonical
from tools.coopmwm import accel_storage
from tools.coopmwm import replay_bank as rb, rerun_cpus as rcpu, sumo_verify as sv
from tools.coopmwm import verify_perseed as vp
from tools.coopmwm.exact_grade import _seal, verify_sealed
from tools.coopmwm.test_replay_bank_confirm_cpu import fp, plan, snapshot
from tools.coopmwm.test_verify_perseed_cpu import SeedPlant

PHASE = 'bank_verification_ext'
LEGACY_REMOTE = '/home/ubuntu/coopmwm20261102_warm1'
COLLECTION_REMOTE = '/home/ubuntu/ls_collection_u3frozen'
SHA = 'a'*64


# ---------------------------------------------------------------- quote / budget / wall clock

def test_quote_three_arms_including_gate_with_u8_u24_numbers():
    q = rcpu.quote()
    assert q['a_train_sim_s'] == 24*24*1200 == 691200                     # A: plain PPO, no bank/practice
    assert q['b_train_sim_s'] == q['r_train_sim_s'] == 24*30*1200 == 864000   # 24 PPO + 6 practice refs
    assert q['v2v_train'] == 691200+2*864000 == 2419200
    assert q['v2v_lessons'] == 4*5280 == 21120
    assert q['v2v_isolated'] == (4+2)*32610 == 195660                    # B 6/12/18/24 + R 12/24
    assert q['gate_bank_verification_ext'] == 35820
    assert q['interim_u8'] == dict(per_arm=96*1200, arms=3, updates=8, episodes_per_arm=96)
    assert q['final_u24'] == dict(per_arm=96*1200, arms=3, updates=24, episodes_per_arm=96)
    assert q['interim_u8_sim_s'] == q['final_u24_sim_s'] == 3*96*1200 == 345600
    assert q['total_without_tests'] == 691200+1728000+21120+195660+35820 == 2671800
    assert q['total'] == 2671800+345600+345600 == 3363000


def test_budget_is_single_ledger_and_covers_the_quote():
    b = rcpu.budget()
    assert b['single_ledger'] is True and b['arms'] == ['A', 'B', 'R']
    assert sum(b['phases'].values()) == b['total'] == 3363720            # lessons floor 21840
    assert b['phases']['v2v_train'] == 2419200 and b['quote_total_sim_s'] == 3363000
    for phase, cost in (('v2v_train', 2419200), ('v2v_lessons', 21120), ('v2v_isolated', 195660),
                        ('gate_bank_verification_ext', 35820), ('v2v_interim', 345600),
                        ('v2v_final', 345600)):
        assert b['phases'][phase] >= cost


def test_wall_clock_paced():
    w = rcpu.wall_clock()
    assert w['per_update_s'] == 1800
    assert w['A_s'] == 24*1800 and w['B_s'] == 24*1800+4*900 and w['R_s'] == w['B_s']+1800
    assert w['tests_s'] == 2*3*96*300/2


# ---------------------------------------------------------------- 3-arm runtime config

def config(tmp_path, **over):
    base = over.pop('base', dict(replay_sources={'B': [COLLECTION_REMOTE], 'R': []}))
    return rcpu.runtime_config(base, root=str(tmp_path/'root_c4_cpus'),
                               reference_root=str(tmp_path/'refs'), source_root=str(tmp_path/'src'),
                               common_checkpoint_sha256=over.pop('sha', SHA), **over)


def test_three_arms_from_mappo_u0_with_identical_code(tmp_path):
    cfg = config(tmp_path)
    assert rcpu.check_pins(cfg)
    assert cfg['active_arms'] == ['A', 'B', 'R'] and cfg['rerun_arms'] == ['A', 'B', 'R']
    assert set(cfg['start_checkpoint_sha256'].values()) == {SHA}          # ALL arms from MAPPO u0
    assert cfg['final_updates'] == dict(A=24, B=24, R=24)
    assert cfg['stop_after_update'] == 24 and cfg['A_stop_after_update'] == 24
    for key, want in rcpu.PINS.items():                                   # identical B/R fix set
        assert cfg[key] == want, key
    assert cfg['diagnostic_schedule'] == rcpu.SCHEDULE
    assert cfg['bank_arms'] == dict(B='B', R='R')                         # A has no bank
    from tools.coopmwm.practice_starts import enabled
    assert not enabled(cfg, 'A') and enabled(cfg, 'B') and enabled(cfg, 'R')
    from tools.coopmwm.v2v_launch3 import (imitation_filter_for, imitation_step_mode_for,
                                           imitation_target_space_for, practice_options_for)
    for check in (imitation_filter_for, imitation_step_mode_for, imitation_target_space_for,
                  practice_options_for):
        assert check(cfg, 'B') == check(cfg, 'R')                         # B == R, identical options
    with pytest.raises(ValueError, match='only arms B and R'):
        imitation_filter_for(cfg, 'A')                                    # A has no bank/filter
    assert cfg['effective_imitation_commit'] == rcpu.EFFECTIVE_IMITATION_COMMIT
    assert cfg['budget'] == rcpu.budget()


def test_standalone_own_ledger_and_local_a_bindings(tmp_path):
    cfg = config(tmp_path)
    assert cfg['run_root'] == cfg['accounting_root'] == str(tmp_path/'root_c4_cpus').replace('\\', '/')
    assert cfg['single_ledger'] == dict(root=cfg['run_root'], charges='CHARGES.jsonl',
                                        lock='ACCOUNTING.claim', authority='AUTHORITY.md',
                                        foreign_refused=True)
    # A results are LOCAL: the control arm runs in the SAME new root; no remote A read anywhere.
    assert cfg['control_arm'] == dict(arm='A', root=cfg['run_root'], update=24)
    assert cfg['a_results']['root'] == cfg['run_root']
    assert 'A_UPDATE' in cfg['a_results']['rule'] and 'THIS root' in cfg['a_results']['rule']
    assert cfg['legacy_root'] != cfg['run_root']                    # only the legacy-u3 refs are staged


def test_runtime_config_refusals(tmp_path):
    with pytest.raises(ValueError, match='checkpoint hash required'):
        config(tmp_path, sha='a'*63)
    with pytest.raises(ValueError, match='must not be the staged legacy reference root'):
        rcpu.runtime_config({}, root=str(tmp_path/'x'), reference_root=str(tmp_path/'x'),
                            source_root=str(tmp_path/'src'), common_checkpoint_sha256=SHA)
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match='pin mismatch'):
        rcpu.check_pins(dict(cfg, imitation_step_mode='fixed'))
    with pytest.raises(ValueError, match='must account in its OWN root'):
        rcpu.check_pins(dict(cfg, accounting_root=str(tmp_path/'elsewhere')))
    with pytest.raises(ValueError, match='no remote A reads'):
        rcpu.check_pins(dict(cfg, control_arm=dict(arm='A', root='coop3', update=24)))
    with pytest.raises(ValueError, match='not pinned'):
        rcpu.check_pins(dict(cfg, effective_imitation_commit='b'*40))
    with pytest.raises(ValueError, match='all three arms'):
        rcpu.check_pins(dict(cfg, rerun_arms=['B', 'R']))                 # the old rerun_clean shape
    with pytest.raises(ValueError, match='registered 3-arm quote'):
        rcpu.check_pins(dict(cfg, budget=dict(cfg['budget'], total=1)))


# ---------------------------------------------------------------- own single ledger

def test_init_root_creates_the_own_single_ledger(tmp_path):
    root = tmp_path/'new_root'
    bound = rcpu.init_root(root, amendment_shas=('c'*64,))
    assert bound['foreign'] is False
    assert (root/'CHARGES.jsonl').read_bytes() == b'' and (root/'ACCOUNTING.claim').exists()
    text = (root/'AUTHORITY.md').read_text(encoding='utf-8')
    assert 'SINGLE LEDGER' in text and rcpu.EFFECTIVE_IMITATION_COMMIT in text and 'c'*64 in text
    assert rcpu.own_ledger(root)['root'] == str(root.resolve())
    with pytest.raises(ValueError, match='already exists'):               # refuse existing root
        rcpu.init_root(root)
    with pytest.raises(ValueError, match='already exists'):
        rcpu.init_root(root, amendment_shas=())


def test_charge_lands_only_in_this_roots_own_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    a, b = tmp_path/'a', tmp_path/'b'
    rcpu.init_root(a)
    rcpu.init_root(b)
    rcpu.charge(a, [dict(id='u001_a_1', sim_s=1200)], 'v2v_train', budget=rcpu.budget())
    from tools.coopmwm.storage import ledger
    assert [e['id'] for e in ledger(a)] == ['u001_a_1']
    assert ledger(b) == []                                                # own-ledger only: b untouched
    with pytest.raises(ValueError, match='duplicate charge'):
        rcpu.charge(a, [dict(id='u001_a_1', sim_s=1200)], 'v2v_train', budget=rcpu.budget())
    with pytest.raises(ValueError, match='whole-wave budget blocked'):
        rcpu.charge(a, [dict(id='x1', sim_s=30000)], 'v2v_lessons', budget=rcpu.budget())


def test_refuses_foreign_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    a, b = tmp_path/'a', tmp_path/'b'
    rcpu.init_root(a)
    rcpu.init_root(b)
    # (1) ledger shared with another root through a hard link (works unprivileged everywhere)
    (a/'CHARGES.jsonl').unlink()
    os.link(b/'CHARGES.jsonl', a/'CHARGES.jsonl')
    with pytest.raises(ValueError, match="another root's ledger|another directory entry"):
        rcpu.own_ledger(a)
    with pytest.raises(ValueError, match='single ledger|another'):
        rcpu.charge(a, [dict(id='j1', sim_s=1200)], 'v2v_train', budget=rcpu.budget())
    # (2) the classic shared-ledger pattern (symlink into another root) on symlink-capable hosts
    (a/'CHARGES.jsonl').unlink()
    try:
        os.symlink(b/'CHARGES.jsonl', a/'CHARGES.jsonl')
    except OSError:
        pytest.skip('Windows host without symlink privilege (hard-link case already covered)')
    with pytest.raises(ValueError, match='own single ledger'):
        rcpu.own_ledger(a)
    # (3) missing files / wrong authority
    (a/'CHARGES.jsonl').unlink()
    (a/'CHARGES.jsonl').write_text('', encoding='utf-8')
    (a/'AUTHORITY.md').unlink()
    with pytest.raises(ValueError, match='own single ledger'):
        rcpu.own_ledger(a)
    (a/'AUTHORITY.md').write_text('unrelated root authority\n', encoding='utf-8')
    with pytest.raises(ValueError, match='not the standalone single-ledger rule'):
        rcpu.own_ledger(a)
    (a/'AUTHORITY.md').write_text('SINGLE LEDGER without the commit pin\n', encoding='utf-8')
    with pytest.raises(ValueError, match='effective-imitation commit'):
        rcpu.own_ledger(a)


# ---------------------------------------------------------------- staging manifest (real readers)

def _budget_file(root):
    from tools.coopmwm.warm_protocol import budget
    b = budget()
    b['total'] += 2600000
    b['allocated'] += 2600000
    b['phases'] = dict(b['phases'], bank_verification_ext=2600000)
    (root/'B_REPLAY_PHASE_RECONCILIATION_0627.json').write_text(json.dumps(
        dict(budget=b, amendment_sha256='test-bank-budget')))
    (root/'AUTHORITY.md').write_text('test-bank-budget')


def real_all8_bank(root):
    """One REAL decision through the test verifiers (no SUMO): sumo_verification receipt +
    branches, snapshots, decisions link, all8 record, seed-2 and both-solo verifications."""
    _budget_file(root)
    bank = rb.bank_root(root, 'B')
    snap = snapshot(600)
    plans = [plan(f'p{i+1}', 0.1) for i in range(8)]
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6}
    plant = SeedPlant(bank/'sumo_verification', gains)
    r = sv.verify(snap, plans, plant, sv.local_charge(root/'first_charges'), 'wm',
                  root=bank/'sumo_verification')
    assert r['status'] == 'accepted'
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    torch.save(snap, bank/'snapshots'/'s1.pt')
    (bank/'snapshots'/'s1.json').write_text(json.dumps(dict(state_id='s1')), encoding='utf-8')
    (bank/'decisions').mkdir(parents=True, exist_ok=True)
    (bank/'decisions'/'s1.json').write_text(json.dumps(dict(state_id='s1', verification_id=r['verification_id'],
                                                            status=r['status'], policy_update=5)),
                                            encoding='utf-8')
    def runner_for(work):
        return SeedPlant(work, gains)
    out = rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8)
    assert out == [True]
    assert (bank/vp.RECORDS).is_dir() and (bank/vp.SEED2_ROOT/'verifications').is_dir()
    return r


def ls_label_source(mirror):
    """The frozen-u3 LS-COLLECT label root for arm B (one label + its branch tapes/traces)."""
    src = mirror/'ls_collection_u3frozen'
    vid = 'ls_u3_label_0001'
    receipt = _seal(dict(verification_id=vid, status='accepted',
                         episode=dict(arm='B', id='ls_e1', world=dict(seed=7), update=3, role='train'),
                         timing=dict(finished_unix=1.0), grade_veh_s=1.0,
                         branches=[dict(branch='policy'), dict(branch='plan')],
                         window=dict(start_s=600, end_s=720)), 'receipt_sha256')
    (src/'labels').mkdir(parents=True)
    (src/'labels'/(vid+'.json')).write_text(json.dumps(dict(search_id=vid, verification=receipt)),
                                            encoding='utf-8')
    from tools.coopmwm.accel_storage import write as awrite
    for branch in ('policy', 'plan'):
        d = src/'branches'/vid/branch
        d.mkdir(parents=True)
        awrite(d/'RESULT.json', dict(prefixes={}, final=fp(720, 5.), steps=[]))
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(dict(time_s=600))+'\n')
    return src


@pytest.fixture
def mirror(tmp_path):
    """A local mirror laid out like a filesystem root (home/ubuntu/...) holding the legacy run
    root + the LS label source; /home/ubuntu maps onto the mirror's home/ubuntu."""
    mirror = tmp_path/'mirror'
    root = mirror/'home'/'ubuntu'/'coopmwm20261102_warm1'
    root.mkdir(parents=True)
    real_all8_bank(root)
    ls_label_source(mirror/'home'/'ubuntu')
    (root/'B_checkpoint_000.pt').write_bytes(b'MAPPO-u0-common-checkpoint')
    (root/'B_checkpoint_003.pt').write_bytes(b'legacy B u3 reference')
    (root/'R_checkpoint_003.pt').write_bytes(b'legacy R u3 reference')
    (root/'SAFETY_RULE_0957.json').write_text(json.dumps(dict(authority='MAIN2026-11-03 09:57')),
                                              encoding='utf-8')
    rcpu.manifest_source(mirror)
    return mirror


def test_staging_manifest_resolves_the_bank_through_real_readers(mirror):
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    verify_sealed(m, 'manifest_sha256')
    assert m['home'] == '/home/ubuntu'
    keys = set(m['entries'])
    legacy = 'home/ubuntu/coopmwm20261102_warm1/'
    # resolved via the REAL readers: decisions link, verifications receipt, tapes + traces
    assert legacy+'replay_bank/B/decisions/s1.json' in keys
    receipts = list((mirror/'home'/'ubuntu'/'coopmwm20261102_warm1'/'replay_bank'/'B'
                     /'sumo_verification'/'verifications').glob('*.json'))
    assert len(receipts) == 1
    assert (legacy+'replay_bank/B/sumo_verification/verifications/'+receipts[0].name) in keys
    vid = json.loads(receipts[0].read_text(encoding='utf-8'))['verification_id']
    receipt = json.loads(receipts[0].read_text(encoding='utf-8'))
    assert receipt['branches'][0]['branch'] == 'policy'
    for branch in receipt['branches']:                     # every graded branch tape + trace staged
        base = legacy+'replay_bank/B/sumo_verification/branches/'+vid+'/'+branch['branch']+'/'
        assert base+'RESULT.json' in keys and base+'observable.jsonl.gz' in keys
    assert 'cand1' in {b['branch'] for b in receipt['branches']}
    traces = [k for k in keys if k.endswith('observable.jsonl.gz')]
    assert len(traces) >= 2                                        # seed-2 + solo traces too
    # all8 pipeline artifacts (real reader output dirs)
    for sub in (vp.RECORDS, vp.SEED2_ROOT, vp.SOLO_ROOTS[1], vp.SOLO_ROOTS[2]):
        assert any(k.startswith(legacy+'replay_bank/B/'+sub+'/') for k in keys), sub
    all8 = next(iter((mirror/'home'/'ubuntu'/'coopmwm20261102_warm1'/'replay_bank'/'B'/vp.RECORDS).glob('*.json')))
    assert legacy+'replay_bank/B/confirmations_all8/'+all8.name in keys
    # snapshots (practice starts may restart from any own-bank example)
    assert legacy+'replay_bank/B/snapshots/s1.pt' in keys and legacy+'replay_bank/B/snapshots/s1.json' in keys
    # LS source label + tapes staged from the source root, relative to /home/ubuntu
    assert any(k.startswith('home/ubuntu/ls_collection_u3frozen/labels/') for k in keys)
    assert any(k.startswith('home/ubuntu/ls_collection_u3frozen/branches/') and k.endswith('RESULT.json')
               for k in keys)
    # MAPPO checkpoints + composite rule, with the staging roles written out
    assert m['entries'][legacy+'B_checkpoint_000.pt']['kind'] == 'checkpoint_u0'
    assert m['entries'][legacy+'B_checkpoint_003.pt']['kind'] == 'checkpoint_reference_u3'
    assert m['entries'][legacy+'R_checkpoint_003.pt']['kind'] == 'checkpoint_reference_u3'
    assert m['entries'][legacy+'SAFETY_RULE_0957.json']['kind'] == 'safety_rule'
    # every sha matches the actual file, and every key is /home/ubuntu-relative posix
    import hashlib
    for key, entry in m['entries'].items():
        assert key.startswith('home/ubuntu/') and '\\' not in key
        path = mirror.joinpath(*key.split('/'))
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    assert m['counts']['checkpoint_u0'] == 1 and m['counts']['checkpoint_reference_u3'] == 2
    command = rcpu.staging_files_command(m)
    assert 'STAGING_FILES.txt' in command and '--files-from' in command and 'rsync' in command


def test_staging_manifest_refuses_forbidden_and_missing_inputs(mirror):
    with pytest.raises(ValueError, match='forbidden path token'):
        (mirror/'home'/'ubuntu'/'coopmwm20261102_warm1'/'replay_bank'/'B'/'snapshots'/'holdout_x.json'
         ).write_text('{}', encoding='utf-8')
        rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    (mirror/'home'/'ubuntu'/'coopmwm20261102_warm1'/'replay_bank'/'B'/'snapshots'/'holdout_x.json'
     ).unlink()
    mirror2 = mirror.parent/'mirror2'
    shutil.copytree(mirror, mirror2)
    (mirror2/'home'/'ubuntu'/'coopmwm20261102_warm1'/'R_checkpoint_003.pt').unlink()
    rcpu.manifest_source(mirror2)
    with pytest.raises(ValueError, match='required staging input missing'):
        rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    rcpu.manifest_source(mirror)
    with pytest.raises(ValueError, match='staged roots must live under'):
        rcpu.staging_manifest(remote_root='/data/elsewhere', sources={})
    with pytest.raises(ValueError, match='register the local mirror'):
        saved = rcpu._mirror[0]
        rcpu._mirror[0] = None
        try:
            rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={})
        finally:
            rcpu._mirror[0] = saved


def test_staging_manifest_is_read_only(mirror, tmp_path):
    """The manifest builder must never write into the mirror (the temperature rule stays
    unwritten, no replay_bank file changes) while still resolving everything."""
    before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
              for p in mirror.rglob('*') if p.is_file()}
    rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    after = {str(p): (p.stat().st_mtime_ns, p.stat().st_size)
             for p in mirror.rglob('*') if p.is_file()}
    assert before == after


def test_verify_staged_refuses_any_mismatch(mirror, tmp_path):
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    stage = tmp_path/'stage'
    shutil.copytree(mirror, stage)
    assert rcpu.verify_staged(m, stage)['files'] == len(m['entries'])
    victim = next(iter((stage/'home'/'ubuntu'/'coopmwm20261102_warm1'/'replay_bank'/'B'/'decisions').glob('*.json')))
    victim.write_text(victim.read_text(encoding='utf-8')+' ', encoding='utf-8')
    with pytest.raises(SystemExit, match='sha differs'):
        rcpu.verify_staged(m, stage)
    victim.unlink()
    with pytest.raises(SystemExit, match='missing'):
        rcpu.verify_staged(m, stage)
    with pytest.raises(SystemExit, match='seal broken'):
        rcpu.verify_staged(dict(m, manifest_sha256='0'*64), stage)


# ---------------------------------------------------------------- approval-gated dry preflight

def prep_dir(tmp_path, mirror, *, verdict='positive', approval_sha=True, commit=None,
             owner_approved=True):
    prep = tmp_path/'prep'
    prep.mkdir()
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(m), encoding='utf-8')
    request = dict(quote=rcpu.quote(), budget=rcpu.budget(), new_sim_s=0)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(request), encoding='utf-8')
    (prep/rcpu.OWNER_QUOTE).write_text('owner quote: 3,363,000 sim-s for the 3-arm standalone rerun',
                                       encoding='utf-8')
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict=verdict)), encoding='utf-8')
    sha_of = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(
        budget_request_sha256=sha_of(prep/rcpu.BUDGET_REQUEST) if approval_sha else '0'*64,
        owner_approved=owner_approved, owner_quote_sha256=sha_of(prep/rcpu.OWNER_QUOTE),
        verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE),
        effective_imitation_commit=commit or rcpu.EFFECTIVE_IMITATION_COMMIT)), encoding='utf-8')
    stage = tmp_path/'stage'
    shutil.copytree(mirror, stage)
    return prep, stage


def test_preflight_dry_passes_only_with_full_binding_and_matching_staged_shas(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    out = rcpu.preflight_dry(prep, stage_dir=stage)
    assert out['ready_for_registered_launch'] is True and out['new_sim_s'] == 0
    assert out['staged']['files'] == len(json.loads((prep/rcpu.MANIFEST_FILE).read_text())['entries'])
    assert out['manifest_sha256']


def test_preflight_dry_refusals(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    sha_of = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    good = dict(budget_request_sha256=sha_of(prep/rcpu.BUDGET_REQUEST), owner_approved=True,
                owner_quote_sha256=sha_of(prep/rcpu.OWNER_QUOTE),
                verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE),
                effective_imitation_commit=rcpu.EFFECTIVE_IMITATION_COMMIT)
    (prep/rcpu.BUDGET_REQUEST).unlink()
    with pytest.raises(SystemExit, match='budget request missing'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(dict(quote=dict(total=1))), encoding='utf-8')
    with pytest.raises(SystemExit, match='differs from the 3-arm quote'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(dict(quote=rcpu.quote(), budget=rcpu.budget(),
                                                          new_sim_s=0)), encoding='utf-8')
    (prep/rcpu.APPROVAL).unlink()
    with pytest.raises(SystemExit, match='OWNER_APPROVAL'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, budget_request_sha256='0'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='budget sha differs'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, owner_approved=False)), encoding='utf-8')
    with pytest.raises(SystemExit, match='owner approval not positive'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, owner_quote_sha256='1'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='owner quote'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, verdict_file_sha256='2'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='verdict file'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict='negative')), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE))),
                                    encoding='utf-8')
    with pytest.raises(SystemExit, match='verdict not positive'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict='positive')), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, effective_imitation_commit='b'*40)),
                                    encoding='utf-8')
    with pytest.raises(SystemExit, match='not pinned'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.APPROVAL).write_text(json.dumps(good), encoding='utf-8')
    (prep/rcpu.MANIFEST_FILE).unlink()
    with pytest.raises(SystemExit, match='staging manifest missing'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(dict(good_check='x')), encoding='utf-8')
    with pytest.raises(SystemExit, match='seal broken'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(m), encoding='utf-8')
    victim = next(iter((stage/'home'/'ubuntu'/'coopmwm20261102_warm1').glob('B_checkpoint_000.pt')))
    victim.write_bytes(b'tampered')
    with pytest.raises(SystemExit, match='sha differs'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    victim.write_bytes(b'MAPPO-u0-common-checkpoint')
    assert rcpu.preflight_dry(prep, stage_dir=stage)['ready_for_registered_launch'] is True
