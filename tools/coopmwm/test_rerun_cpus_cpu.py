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
  Track 2 T2C   - explicitly selected clean B/R profile: trigger + frozen layer/credit/features,
                  WM/random teachers, own-bank augmentation and equal practice weighting;
                  extra coop_reward OFF; cpuS shared
                  40-worker ceiling (24 episode + 16 diagnostic slots); isolated cooperation at
                  u6/12/18/24 (N=8 paired-world replicates, bootstrap CIs, sealed
                  pre-registered criterion), the full budget quote and the criterion-bound
                  approval-gated dry preflight;
  A results local - control arm A runs in the SAME new root (no remote A reads);
  quote         - 3 arms incl. the exact gate, group-credit verification allowance, the
                  replicated isolated test and the u8/u24 test numbers written out;
  staging       - the read-only manifest resolves the bank through the REAL readers
                  (bank_decisions + per_seed_v1 confirmations on a real all8 bank built by the
                  test verifiers) relative to /home/ubuntu, and the dry preflight refuses
                  without a fully bound OWNER_APPROVAL.json and byte-exact staged files.

The T1A/T2A/T2B modules (moves/trigger_policy/dense_teacher) are owned by sibling tracks and
are NOT on this branch: per the shared rules, rerun_cpus codes against their APIs and these
tests install small local stubs (autouse fixture below).
"""
import gzip
import hashlib
import json
import os
import shutil
import sys
import types

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


# ------------------------------------------------- T1A/T2A/T2B API stubs (tests only, per rules)

def _t2_stub_modules():
    moves = types.ModuleType('tools.coopmwm.moves')
    moves.FAMILIES = ('none', 'ease_off', 'smoothing', 'gap_open', 'wall', 'ramp_assist')
    moves.candidates = lambda obs_state, rng, n: (_ for _ in ()).throw(
        NotImplementedError('T1A owns the real move library'))
    moves.offsets = lambda move, t: {}
    moves.from_plan = lambda verified_plan: None
    trigger_policy = types.ModuleType('tools.coopmwm.trigger_policy')
    dense_teacher = types.ModuleType('tools.coopmwm.dense_teacher')
    return dict(moves=moves, trigger_policy=trigger_policy, dense_teacher=dense_teacher)


@pytest.fixture(autouse=True)
def t2_sibling_module_stubs(monkeypatch):
    """rerun_cpus codes against the sibling-track APIs (moves/trigger_policy/dense_teacher);
    those modules live on their own branches, so the tests install the documented-API stubs."""
    import tools.coopmwm as pkg
    for name, mod in _t2_stub_modules().items():
        monkeypatch.setitem(sys.modules, 'tools.coopmwm.'+name, mod)
        monkeypatch.setattr(pkg, name, mod, raising=False)
    yield


# ---------------------------------------------------------------- quote / budget / wall clock

def test_quote_three_arms_including_gate_teacher_verify_and_replicated_isolated():
    q = rcpu.quote()
    assert q['a_train_sim_s'] == 24*24*1200 == 691200                     # A: plain PPO, no bank/practice
    assert q['b_train_sim_s'] == q['r_train_sim_s'] == 24*36*1200
    assert q['v2v_train'] == 691200+2*24*36*1200
    assert q['v2v_lessons'] == 4*5280 == 21120
    assert q['teacher_verify_sim_s'] == q['group_credit_verify_sim_s'] == 2304*35820
    assert q['augment_verify_sim_s'] == 144*35820
    assert q['early_probes_sim_s'] == 2*3*3*32610
    assert q['smoke_sim_s'] == 6*1200+2*1200+18*35820
    cells = 3*4
    assert q['isolated_replicated'] == dict(arms=['A', 'B', 'R'], updates=[6, 12, 18, 24],
                                            replicates=8, cells=cells, replicates_total=cells*8,
                                            per_replicate_sim_s=32610)
    assert q['isolated_replicated_sim_s'] == cells*8*32610 == 3130560     # supersedes v2v_isolated
    assert 'v2v_isolated' not in q
    assert q['gate_bank_verification_ext'] == 35820
    assert q['interim_u8'] == dict(per_arm=96*1200, arms=3, updates=8, episodes_per_arm=96)
    assert q['final_u24'] == dict(per_arm=96*1200, arms=3, updates=24, episodes_per_arm=96,
                                  harness='tools.coopmwm.planner_eval', shared_worlds=True,
                                  bootstrap=rcpu.FINAL_EVALUATION['bootstrap'])
    assert q['interim_u8_sim_s'] == q['final_u24_sim_s'] == 3*96*1200 == 345600
    assert q['total_without_tests'] == sum(q[k] for k in (
        'v2v_train', 'v2v_lessons', 'teacher_verify_sim_s', 'group_credit_verify_sim_s',
        'augment_verify_sim_s', 'early_probes_sim_s', 'smoke_sim_s',
        'isolated_replicated_sim_s', 'gate_bank_verification_ext'))
    assert q['total'] == q['total_without_tests']+345600+345600


def test_budget_is_single_ledger_and_covers_the_quote():
    b = rcpu.budget()
    assert b['single_ledger'] is True and b['arms'] == ['A', 'B', 'R']
    assert sum(b['phases'].values()) == b['total'] == rcpu.quote()['total']+720
    assert b['phases']['v2v_train'] == rcpu.quote()['v2v_train']
    assert 'replicated' in b['isolated_test'] and 'pre-registered' in b['isolated_test']
    for phase, cost in (('v2v_train', 2419200), ('v2v_lessons', 21120), ('teacher_verify', 0),
                         ('group_credit_verify', 35820),
                        ('isolated_replicated', 3130560), ('gate_bank_verification_ext', 35820),
                        ('v2v_interim', 345600), ('v2v_final', 345600)):
        assert b['phases'][phase] >= cost


def test_wall_clock_paced():
    w = rcpu.wall_clock()
    assert w['per_update_s'] == 1800
    assert w['A_s'] == 24*1800 and w['B_s'] == 24*1800+4*900 and w['R_s'] == w['B_s']+1800
    assert w['tests_s'] == 2*3*96*300/24
    assert w['isolated_s'] == 3130560/20 == 156528.0                     # 36,000 sim-s per 1,800 s pace
    assert w['total_h'] == round((w['R_s']+w['isolated_s']+w['verification_s']+w['tests_s'])/3600, 1)
    assert w['estimate_only'] and w['target']['max_sumo_workers'] == 40


# ---------------------------------------------------------------- 3-arm runtime config

def config(tmp_path, **over):
    base = over.pop('base', dict(replay_sources={'B': [COLLECTION_REMOTE], 'R': []}))
    return rcpu.runtime_config(base, root=str(tmp_path/'root_c4_cpus'),
                               reference_root=str(tmp_path/'refs'), source_root=str(tmp_path/'src'),
                               common_checkpoint_sha256=over.pop('sha', SHA),
                               track2=over.pop('track2', True), **over)


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
    assert cfg['diagnostic_phases'] == dict(lessons='v2v_lessons', isolated='isolated_replicated', probes='early_probes')
    assert cfg['isolated_test'] == rcpu.ISOLATED_TEST                     # replicated test registration
    assert cfg['criterion_sha256'] == rcpu.criterion_document()['criterion_sha256']
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


def test_t2c_trigger_head_and_teachers_per_arm(tmp_path):
    """A is plain PPO; B/R heads and WM/random teachers are enabled in the opted-in profile."""
    import tools.coopmwm as pkg
    cfg = config(tmp_path)
    head, teachers = cfg['trigger_head'], cfg['teacher']
    assert head['module'] == 'tools.coopmwm.trigger_policy'
    assert tuple(head['arms']) == ('B', 'R') and head['v2v_dim'] == 48 == cfg['v2v_dim']
    assert head['decision_interval_s'] == 60                              # registered cadence
    assert head['families_source'] == 'moves.FAMILIES'
    assert head['candidates_api'] == 'moves.candidates(obs_state, rng, n)'
    assert head['bank_label_api'] == 'moves.from_plan(verified_plan)'
    assert 'moves.offsets' in head['low_level'] and 'HDVs never commanded' in head['low_level']
    assert rcpu.trigger_head_enabled(cfg, 'B') and rcpu.trigger_head_enabled(cfg, 'R')
    assert not rcpu.trigger_head_enabled(cfg, 'A')
    assert [rcpu.teacher_kind(cfg, a) for a in ('A', 'B', 'R')] == [None, 'wm_dense', 'random']
    b = teachers['B']
    assert b['module'] == 'tools.coopmwm.dense_teacher' and b['kind'] == 'wm_dense'
    assert 'DAgger' in b['mode'] and 'cross-entropy' in b['mode'] and 'T1B' in b['mode']
    assert b['verify_fraction'] == .1 and b['enabled'] is True
    assert b['verified_through'] == 'all8'
    r = teachers['R']
    assert r['kind'] == 'random' and 'same move distribution' in r['distribution']
    assert teachers['A'] is None


def test_t2c_coop_reward_slot_is_off_by_default(tmp_path):
    """The coop_reward shaper is a config SLOT only (built on c4/coop-reward): present, OFF,
    and refused when modified or enabled."""
    cfg = config(tmp_path)
    assert cfg['coop_reward'] == rcpu.COOP_REWARD
    assert cfg['coop_reward']['enabled'] is False and cfg['coop_reward']['weight'] == 0.0
    assert cfg['coop_reward']['built_on'] == 'c4/coop-reward'
    assert cfg['coop_reward']['off_is_byte_identical'] is True
    with pytest.raises(ValueError, match='pin mismatch.*coop_reward'):
        rcpu.check_pins(dict(cfg, coop_reward=dict(rcpu.COOP_REWARD, enabled=True, weight=0.3)))
    with pytest.raises(ValueError, match='pin mismatch.*coop_reward'):
        rcpu.check_pins(dict(cfg, coop_reward=dict(rcpu.COOP_REWARD, module='x')))


def test_t2c_isolated_test_registration_matches_schedule_and_quote():
    spec = rcpu.ISOLATED_TEST
    assert spec['updates'] == [6, 12, 18, 24] and spec['arms'] == ['A', 'B', 'R']
    assert spec['replicates'] == 8 and spec['sim_s_per_replicate'] == 32610
    assert spec['pairing'] == 'same world seed across arms (replicate index)'
    assert spec['bootstrap'] == dict(reps=10000, level=0.95, seed=20261104, method='percentile')
    assert spec['safety_counters'] == ['collisions', 'teleports', 'hard_brakes', 'ttc_events']
    for arm in ('A', 'B', 'R'):
        assert rcpu.SCHEDULE[arm]['isolated'] == spec['updates']          # every arm, every update
    assert rcpu.SCHEDULE == dict(A=dict(lessons=[], isolated=[6, 12, 18, 24]),
                                 B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 18, 24]),
                                 R=dict(lessons=[], isolated=[6, 12, 18, 24]))
    assert rcpu.quote()['isolated_replicated_sim_s'] == \
        len(spec['arms'])*len(spec['updates'])*spec['replicates']*spec['sim_s_per_replicate']


def test_t2c_stubs_expose_the_sibling_track_apis():
    """Only moves has a fixed symbol API in the spec; no guessed sibling head constants."""
    import tools.coopmwm as pkg
    assert 'none' in tuple(pkg.moves.FAMILIES)
    assert set(pkg.moves.FAMILIES) >= {'ease_off', 'gap_open', 'wall', 'ramp_assist'}
    assert all(callable(getattr(pkg.moves, n)) for n in ('candidates', 'offsets', 'from_plan'))


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
    with pytest.raises(ValueError, match='trigger head applies to B and R only'):
        rcpu.check_pins(dict(cfg, trigger_head=dict(cfg['trigger_head'], arms=('A', 'B', 'R'))))
    with pytest.raises(ValueError, match='trigger head applies to B and R only'):
        rcpu.check_pins(dict(cfg, trigger_head=None))                     # missing head slot
    with pytest.raises(ValueError, match='registered 60 s decision cadence'):
        rcpu.check_pins(dict(cfg, trigger_head=dict(cfg['trigger_head'], decision_interval_s=30)))
    with pytest.raises(ValueError, match='V2V obs width'):
        rcpu.check_pins(dict(cfg, trigger_head=dict(cfg['trigger_head'], v2v_dim=32)))
    with pytest.raises(ValueError, match='teachers must be A=none, B=wm_dense .*R=random'):
        rcpu.check_pins(dict(cfg, teacher=dict(cfg['teacher'], A=dict(kind='wm_dense'))))
    with pytest.raises(ValueError, match='teachers must be A=none, B=wm_dense .*R=random'):
        rcpu.check_pins(dict(cfg, teacher=dict(cfg['teacher'], B=dict(kind='random'))))
    with pytest.raises(ValueError, match='all8 machinery'):
        rcpu.check_pins(dict(cfg, teacher=dict(cfg['teacher'],
                                               B=dict(cfg['teacher']['B'], verified_through='manual'))))
    with pytest.raises(ValueError, match='SAME move distribution'):
        rcpu.check_pins(dict(cfg, teacher=dict(cfg['teacher'], R=dict(kind='random'))))
    with pytest.raises(ValueError, match='replicated isolated-test registration'):
        rcpu.check_pins(dict(cfg, isolated_test=dict(cfg['isolated_test'], replicates=4)))
    with pytest.raises(ValueError, match='pre-registered criterion'):
        rcpu.check_pins(dict(cfg, criterion_sha256='0'*64))
    sched = {a: dict(v) for a, v in cfg['diagnostic_schedule'].items()}
    sched['R'] = dict(sched['R'], isolated=[12, 24])                      # the old non-replicated plan
    with pytest.raises(ValueError, match='replicated isolated test at'):
        rcpu.check_pins(dict(cfg, diagnostic_schedule=sched))
    with pytest.raises(ValueError, match='isolated_replicated phase'):
        rcpu.check_pins(dict(cfg, diagnostic_phases=dict(lessons='v2v_lessons',
                                                         isolated='v2v_isolated')))


# ---------------------------------------------------------------- own single ledger

def test_init_root_creates_the_own_single_ledger(tmp_path):
    root = tmp_path/'new_root'
    bound = rcpu.init_root(root, amendment_shas=('c'*64,), track2=True)
    assert bound['foreign'] is False
    assert (root/'CHARGES.jsonl').read_bytes() == b'' and (root/'ACCOUNTING.claim').exists()
    crit_sha = rcpu.criterion_document()['criterion_sha256']
    text = (root/'AUTHORITY.md').read_text(encoding='utf-8')
    assert 'SINGLE LEDGER' in text and rcpu.EFFECTIVE_IMITATION_COMMIT in text and 'c'*64 in text
    assert 'isolated-test criterion '+crit_sha in text                    # pre-registration bound
    assert rcpu.own_ledger(root, track2=True)['root'] == str(root.resolve())
    with pytest.raises(ValueError, match='already exists'):               # refuse existing root
        rcpu.init_root(root)
    with pytest.raises(ValueError, match='already exists'):
        rcpu.init_root(root, amendment_shas=())
    (root/'AUTHORITY.md').write_text('SINGLE LEDGER '+rcpu.EFFECTIVE_IMITATION_COMMIT+'\n',
                                     encoding='utf-8')
    with pytest.raises(ValueError, match='pre-registered.*criterion'):
        rcpu.own_ledger(root, track2=True)                                # criterion line missing
    (root/'AUTHORITY.md').write_text('SINGLE LEDGER\n'+rcpu.EFFECTIVE_IMITATION_COMMIT
                                     +'\nisolated-test criterion '+'0'*64+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='pre-registered.*criterion'):
        rcpu.own_ledger(root, track2=True)                                # criterion sha differs


def test_charge_lands_only_in_this_roots_own_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    a, b = tmp_path/'a', tmp_path/'b'
    rcpu.init_root(a, track2=True)
    rcpu.init_root(b, track2=True)
    rcpu.charge(a, [dict(id='u001_a_1', sim_s=1200)], 'v2v_train', budget=rcpu.budget(), track2=True)
    from tools.coopmwm.storage import ledger
    assert [e['id'] for e in ledger(a)] == ['u001_a_1']
    assert ledger(b) == []                                                # own-ledger only: b untouched
    with pytest.raises(ValueError, match='duplicate charge'):
        rcpu.charge(a, [dict(id='u001_a_1', sim_s=1200)], 'v2v_train', budget=rcpu.budget(), track2=True)
    with pytest.raises(ValueError, match='whole-wave budget blocked'):
        rcpu.charge(a, [dict(id='x1', sim_s=30000)], 'v2v_lessons', budget=rcpu.budget(), track2=True)


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
    shutil.copytree(root/'replay_bank'/'B', root/'replay_bank'/'R')
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
        rcpu.guard_path('holdout_x.json')  # refuse before creating/opening a prohibited path
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


# ------------------------------------------- pre-registered criterion + replicated isolated test

def test_criterion_document_is_sealed_and_stable():
    d1, d2 = rcpu.criterion_document(), rcpu.criterion_document()
    assert d1 == d2                                                       # reproducible pre-registration
    verify_sealed(d1, 'criterion_sha256')
    crit = d1['criterion']
    assert crit['primary_update'] == 24 and crit['secondary_updates'] == [6, 12, 18]
    assert crit['metric'] == 'policy_cooperation_gain_veh_s' and crit['higher_is_better'] is True
    assert crit['updates'] == [6, 12, 18, 24] and crit['replicates'] == 8
    assert crit['bootstrap']['reps'] == 10000 and crit['bootstrap']['level'] == 0.95
    assert crit['safety_tolerance'] == 0
    assert crit['r_arm']['gated'] is False and 'same move distribution' in crit['r_arm']['rule']
    assert 'CI lying entirely above 0' in crit['pass_rule']
    assert d1['isolated_test'] == rcpu.ISOLATED_TEST
    assert d1['arms']['B'].startswith('T2A trigger head + frozen cooperation layer + WM bank')
    assert d1['arms']['R'].endswith('own random bank')
    assert d1['quote'] == rcpu.quote()
    d2['criterion']['updates'].append(99)
    assert rcpu.criterion_document() == d1                               # no global aliases
    tampered = dict(d1, primary_update=12)
    with pytest.raises(ValueError, match='mismatch'):
        verify_sealed(dict(tampered, **{'criterion_sha256': d1['criterion_sha256']}),
                      'criterion_sha256')


def test_bootstrap_ci_is_deterministic_percentile_with_registered_defaults():
    vals = [0.50, 0.42, 0.71, 0.33, 0.66, 0.58, 0.29, 0.80]
    d1 = rcpu.bootstrap_ci(vals)
    assert d1 == rcpu.bootstrap_ci(vals)                                  # same seed -> identical
    assert d1['seed'] == 20261104 and d1['reps'] == 10000 and d1['level'] == 0.95
    assert d1['n'] == 8 and d1['lo'] <= d1['mean'] <= d1['hi']
    assert rcpu.bootstrap_ci(vals, reps=500, seed=7) == rcpu.bootstrap_ci(vals, reps=500, seed=7)
    assert rcpu.bootstrap_ci(vals, reps=500, seed=7) != rcpu.bootstrap_ci(vals, reps=500, seed=8)
    assert rcpu.bootstrap_ci([5.0]*8) == dict(mean=5.0, lo=5.0, hi=5.0, n=8, reps=10000,
                                              level=0.95, seed=20261104)
    strictly = rcpu.bootstrap_ci([0.15]*4+[0.16]*4)
    assert strictly['lo'] > 0 and strictly['hi'] < 0.2
    with pytest.raises(ValueError, match='at least two'):
        rcpu.bootstrap_ci([1.0])
    with pytest.raises(ValueError, match='bad bootstrap'):
        rcpu.bootstrap_ci([1.0, 2.0], level=1.0)
    with pytest.raises(ValueError, match='bad bootstrap'):
        rcpu.bootstrap_ci([1.0, 2.0], reps=0)


def _registration():
    return dict(role='train', metric=rcpu.ISOLATED_TEST['metric'],
                final_evaluation=dict(role='train', worlds=[dict(world_id=f'w{i}', world=dict(seed=986100+i))
                                                           for i in range(rcpu.TEST_EPISODES)]),
                replicates=[dict(replicate=r, world_seed=986000+r,
                                 state_set_sha256='a'*64, window_set_sha256='b'*64)
                            for r in range(8)])


def _evaluate(rows):
    return rcpu.evaluate_isolated_test(rows, registration=_registration())


def _isolated_grid(b_delta=0.15, r_delta=0.0, collisions_b=0.0, at_update=None):
    """A full registered grid: 3 arms x u6/12/18/24 x 8 paired-world replicates."""
    return [dict(j, status='graded', metric=0.5+{'A': 0., 'B': b_delta, 'R': r_delta}[j['arm']],
                 collisions=collisions_b if j['arm'] == 'B' and j['update'] == at_update else 0,
                 teleports=0, hard_brakes=0, ttc_events=0) for j in rcpu.isolated_jobs(_registration())]


def test_evaluate_isolated_test_applies_the_registered_criterion():
    out = _evaluate(_isolated_grid())
    assert out['verdict'] == 'pass' and out['primary']['pass'] is True
    assert out['primary']['update'] == 24 and out['primary']['pair'] == 'B-A'
    assert out['primary']['mean'] == pytest.approx(0.15)
    assert out['primary']['ci_lo'] == pytest.approx(0.15)
    assert out['primary']['ci_excludes_zero_positive'] is True and out['primary']['safety_ok'] is True
    assert out['cells'] == 96 and out['replicates'] == 8
    assert all(s['ok'] for s in out['safety'].values())
    assert out['per_update']['6']['B-A']['ci_lo'] > 0                      # secondary, reported
    assert out['secondary_updates'] == [6, 12, 18]
    assert out['r_arm'].startswith('random-bank fairness control')
    assert out['per_update']['24']['B-R']['mean'] == pytest.approx(0.15)   # teacher ablation reported
    assert out['criterion_sha256'] == rcpu.criterion_document()['criterion_sha256']
    # R clearly worse than A everywhere: reported, NEVER gated (B still passes).
    out_r_bad = _evaluate(_isolated_grid(r_delta=-0.4))
    assert out_r_bad['verdict'] == 'pass' and out_r_bad['per_update']['24']['R-A']['ci_hi'] < 0
    # B loses at the primary update -> fail.
    out_b_loses = _evaluate(_isolated_grid(b_delta=-0.1))
    assert out_b_loses['verdict'] == 'fail' and out_b_loses['primary']['ci_lo'] < 0
    assert out_b_loses['primary']['pass'] is False
    # B wins coop_score but regresses safety (collisions 8 vs 0 across replicates) -> fail.
    out_unsafe = _evaluate(_isolated_grid(collisions_b=1.0, at_update=24))
    assert out_unsafe['verdict'] == 'fail'
    assert out_unsafe['primary']['ci_excludes_zero_positive'] is True
    assert out_unsafe['primary']['safety_ok'] is False
    assert out_unsafe['safety']['collisions'] == dict(A=0.0, B=8.0, ok=False)
    # safety regression at a SECONDARY update never gates the primary verdict
    out_secondary_unsafe = _evaluate(_isolated_grid(collisions_b=1.0, at_update=6))
    assert out_secondary_unsafe['verdict'] == 'pass'


def test_evaluate_isolated_test_refuses_bad_records():
    good = _isolated_grid()
    with pytest.raises(ValueError, match='non-registered arm'):
        _evaluate(good+[dict(arm='C', update=24, replicate=0, metric=0.5,
                                               collisions=0.0, teleports=0.0, hard_brakes=0.0)])
    with pytest.raises(ValueError, match='non-registered update'):
        _evaluate(good+[dict(arm='A', update=8, replicate=0, metric=0.5,
                                               collisions=0.0, teleports=0.0, hard_brakes=0.0)])
    with pytest.raises(ValueError, match='outside the registered range'):
        _evaluate(good+[dict(arm='A', update=24, replicate=8, metric=0.5,
                                               collisions=0.0, teleports=0.0, hard_brakes=0.0)])
    with pytest.raises(ValueError, match='duplicate'):
        _evaluate(good+[good[0]])
    dropped = good[:-1]
    with pytest.raises(ValueError, match='incomplete isolated-test grid'):
        _evaluate(dropped)
    with pytest.raises(ValueError, match='non-finite'):
        _evaluate([dict(r, metric=float('nan')) if r['arm'] == 'B' and
                                     r['update'] == 24 and r['replicate'] == 0 else r
                                     for r in good])
    with pytest.raises(ValueError, match='safety counter'):
        _evaluate([dict(r, collisions=-1.0) if r['arm'] == 'A' and
                                     r['update'] == 24 and r['replicate'] == 0 else r
                                     for r in good])


def test_component_adapters_refuse_wrong_moves_api(tmp_path, monkeypatch):
    import tools.coopmwm as pkg
    cfg = config(tmp_path)
    adapters = {n: lambda **kw: object() for n in
                ('merge_features', 'trigger_head', 'coop_layer', 'group_credit',
                 'dense_teacher', 'plan_augment', 'practice_weighting')}
    monkeypatch.setattr(pkg.moves, 'FAMILIES', ('ease_off', 'gap_open'))
    with pytest.raises(ValueError, match='none family'):
        rcpu.track2_components(cfg, 'B', adapters=adapters)
    monkeypatch.setattr(pkg.moves, 'FAMILIES', ('none', 'ease_off'))
    monkeypatch.setattr(pkg.moves, 'offsets', None)
    with pytest.raises(ValueError, match='moves API'):
        rcpu.track2_components(cfg, 'B', adapters=adapters)


# ---------------------------------------------------------------- approval-gated dry preflight

def prep_dir(tmp_path, mirror, *, verdict='positive', approval_sha=True, commit=None,
             owner_approved=True, criterion_sha=True):
    prep = tmp_path/'prep'
    prep.mkdir()
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(m), encoding='utf-8')
    cfg, registration = config(tmp_path, sha=hashlib.sha256(b'MAPPO-u0-common-checkpoint').hexdigest()), _registration()
    rcpu.init_root(cfg['run_root'], track2=True)
    (prep/rcpu.CONFIG_FILE).write_text(json.dumps(cfg), encoding='utf-8')
    (prep/rcpu.ISOLATED_REGISTRATION_FILE).write_text(json.dumps(registration), encoding='utf-8')
    request = rcpu.budget_request_document(cfg, registration, m)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(request), encoding='utf-8')
    (prep/rcpu.OWNER_QUOTE).write_text('owner quote: 6,333,720 sim-s for the 3-arm standalone '
                                       'rerun with the replicated isolated test',
                                       encoding='utf-8')
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict=verdict)), encoding='utf-8')
    (prep/rcpu.CRITERION_FILE).write_text(json.dumps(rcpu.criterion_document()), encoding='utf-8')
    sha_of = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(
        budget_request_sha256=sha_of(prep/rcpu.BUDGET_REQUEST) if approval_sha else '0'*64,
        owner_approved=owner_approved, owner_quote_sha256=sha_of(prep/rcpu.OWNER_QUOTE),
        verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE),
        criterion_sha256=sha_of(prep/rcpu.CRITERION_FILE) if criterion_sha else '0'*64,
        config_file_sha256=sha_of(prep/rcpu.CONFIG_FILE),
        registration_file_sha256=sha_of(prep/rcpu.ISOLATED_REGISTRATION_FILE),
        manifest_file_sha256=sha_of(prep/rcpu.MANIFEST_FILE),
        effective_imitation_commit=commit or rcpu.EFFECTIVE_IMITATION_COMMIT)), encoding='utf-8')
    stage = tmp_path/'stage'
    shutil.copytree(mirror, stage)
    return prep, stage


def test_preflight_dry_passes_only_with_full_binding_and_matching_staged_shas(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    out = rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    assert out['preflight_pass'] is True and out['new_sim_s'] == 0
    assert out['ready_for_registered_launch'] is False and out['integration_required'] is True
    assert out['staged']['files'] == len(json.loads((prep/rcpu.MANIFEST_FILE).read_text())['entries'])
    assert out['manifest_sha256']
    assert out['criterion_sha256'] == rcpu.sha(prep/rcpu.CRITERION_FILE)
    assert out['criterion_document_sha256'] == rcpu.criterion_document()['criterion_sha256']
    assert json.loads((prep/rcpu.CRITERION_FILE).read_text())['criterion'] == rcpu.CRITERION


def test_preflight_dry_refusals(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    sha_of = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    good = dict(budget_request_sha256=sha_of(prep/rcpu.BUDGET_REQUEST), owner_approved=True,
                owner_quote_sha256=sha_of(prep/rcpu.OWNER_QUOTE),
                verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE),
                criterion_sha256=sha_of(prep/rcpu.CRITERION_FILE),
                config_file_sha256=sha_of(prep/rcpu.CONFIG_FILE),
                registration_file_sha256=sha_of(prep/rcpu.ISOLATED_REGISTRATION_FILE),
                manifest_file_sha256=sha_of(prep/rcpu.MANIFEST_FILE),
                effective_imitation_commit=rcpu.EFFECTIVE_IMITATION_COMMIT)
    request = json.loads((prep/rcpu.BUDGET_REQUEST).read_text())
    (prep/rcpu.BUDGET_REQUEST).unlink()
    with pytest.raises(SystemExit, match='budget request missing'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(dict(quote=dict(total=1))), encoding='utf-8')
    with pytest.raises(SystemExit, match='differs from the 3-arm quote'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(request), encoding='utf-8')
    (prep/rcpu.APPROVAL).unlink()
    with pytest.raises(SystemExit, match='OWNER_APPROVAL'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, budget_request_sha256='0'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='budget sha differs'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, owner_approved=False)), encoding='utf-8')
    with pytest.raises(SystemExit, match='owner approval not positive'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, owner_quote_sha256='1'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='owner quote'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, verdict_file_sha256='2'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='verdict file'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict='negative')), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, verdict_file_sha256=sha_of(prep/rcpu.VERDICT_FILE))),
                                    encoding='utf-8')
    with pytest.raises(SystemExit, match='verdict not positive'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.VERDICT_FILE).write_text(json.dumps(dict(verdict='positive')), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, effective_imitation_commit='b'*40)),
                                    encoding='utf-8')
    with pytest.raises(SystemExit, match='not pinned'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(good), encoding='utf-8')
    # pre-registered criterion bindings: missing file, unbound sha, broken seal, foreign content
    (prep/rcpu.CRITERION_FILE).unlink()
    with pytest.raises(SystemExit, match='pre-registered criterion missing'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.CRITERION_FILE).write_text(json.dumps(rcpu.criterion_document()), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, criterion_sha256='3'*64)), encoding='utf-8')
    with pytest.raises(SystemExit, match='criterion sha differs'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_text(json.dumps(good), encoding='utf-8')
    (prep/rcpu.CRITERION_FILE).write_text(json.dumps(dict(criterion=rcpu.CRITERION,
                                                          criterion_sha256='0'*64)),
                                          encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, criterion_sha256=sha_of(
        prep/rcpu.CRITERION_FILE))), encoding='utf-8')
    with pytest.raises(SystemExit, match='criterion seal broken'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    foreign = _seal(dict(schema='x', criterion=dict(rcpu.CRITERION, primary_update=12),
                         isolated_test=rcpu.ISOLATED_TEST, quote=rcpu.quote()), 'criterion_sha256')
    (prep/rcpu.CRITERION_FILE).write_text(json.dumps(foreign), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(dict(good, criterion_sha256=sha_of(
        prep/rcpu.CRITERION_FILE))), encoding='utf-8')
    with pytest.raises(SystemExit, match='does not carry the registered criterion'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)              # re-sealed edit: still refused
    (prep/rcpu.CRITERION_FILE).write_text(json.dumps(rcpu.criterion_document()), encoding='utf-8')
    (prep/rcpu.APPROVAL).write_text(json.dumps(good), encoding='utf-8')
    (prep/rcpu.MANIFEST_FILE).unlink()
    with pytest.raises(SystemExit, match='staging manifest missing'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(dict(good_check='x')), encoding='utf-8')
    with pytest.raises(SystemExit, match='seal broken'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    m = rcpu.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    (prep/rcpu.MANIFEST_FILE).write_text(json.dumps(m), encoding='utf-8')
    victim = next(iter((stage/'home'/'ubuntu'/'coopmwm20261102_warm1').glob('B_checkpoint_000.pt')))
    victim.write_bytes(b'tampered')
    with pytest.raises(SystemExit, match='sha differs'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    victim.write_bytes(b'MAPPO-u0-common-checkpoint')
    assert rcpu.preflight_dry(prep, stage_dir=stage, track2=True)['preflight_pass'] is True


# ----------------------- essentials-first integration and pairing/approval regression checks

def test_essentials_profile_serializes_without_sibling_modules(tmp_path, monkeypatch):
    """The preparation path works before sibling merges, and survives JSON transport."""
    import tools.coopmwm as pkg
    for name in ('moves', 'trigger_policy', 'dense_teacher'):
        monkeypatch.delattr(pkg, name)
        monkeypatch.setitem(sys.modules, 'tools.coopmwm.'+name, None)
    cfg = json.loads(json.dumps(config(tmp_path)))
    assert rcpu.check_pins(cfg)
    assert cfg['cpu']['workers']+cfg['cpu']['diagnostic_workers'] == 40 < 48
    assert cfg['job_namespace'] == 't2c' and cfg['target_host'] == 'cpuS'
    for name in ('coop_layer', 'group_credit', 'merge_features', 'dense_teacher', 'plan_augment', 'practice_weighting'):
        assert cfg[name]['enabled'] is True
        assert cfg[name]['arms'] == ['B', 'R']
    for name in ('coop_reward',):
        assert cfg[name]['enabled'] is False
    assert cfg['coop_layer']['frozen_base'] and cfg['coop_layer']['separate_optimizer']
    assert cfg['group_credit']['layer_advantage_only']
    assert rcpu.episode_options(cfg, 'A') == {}
    assert rcpu.track2_components(cfg, 'A', adapters={}) == {}
    # Caller mutation cannot change registrations, future configs, or criterion hashes.
    expected = rcpu.criterion_document()
    cfg['isolated_test']['updates'].append(99)
    cfg['coop_layer']['arms'].append('A')
    assert rcpu.criterion_document() == expected
    assert rcpu.check_pins(config(tmp_path))


def test_injected_adapters_receive_dependencies_and_off_slots_never_run(tmp_path):
    cfg = config(tmp_path)
    calls = []
    tokens = {n: object() for n in ('merge_features', 'trigger_head', 'coop_layer', 'group_credit',
                                   'dense_teacher', 'plan_augment', 'practice_weighting')}
    def adapter(name):
        def build(*, cfg, arm, moves, components):
            assert arm == 'B'
            assert cfg['v2v_dim'] == 48 and callable(moves.offsets)
            assert set(components) == set(calls)
            assert all(components[k] is tokens[k] for k in components)
            calls.append(name)
            cfg['coop_layer']['enabled'] = False  # adapters cannot mutate the approved config
            return tokens[name]
        return build
    adapters = {n: adapter(n) for n in tokens}
    def forbidden(**kw):
        pytest.fail('deferred component imported/called')
    adapters.update(coop_reward=forbidden)
    assert rcpu.track2_components(cfg, 'B', adapters=adapters) == tokens
    assert calls == list(tokens)
    assert cfg['coop_layer']['enabled'] is True
    assert rcpu.episode_options(cfg, 'B') == rcpu.episode_options(cfg, 'R') | {
        'track2_teacher': cfg['teacher']['B']}
    with pytest.raises(ValueError, match='adapters missing'):
        rcpu.track2_components(cfg, 'R', adapters={})
    with pytest.raises(ValueError, match='unknown Track 2 arm'):
        rcpu.episode_options(cfg, 'X')


@pytest.mark.parametrize('slot', ['coop_layer', 'group_credit', 'merge_features',
                                  'dense_teacher', 'plan_augment', 'practice_weighting'])
def test_essentials_slots_cannot_be_silently_disabled_or_enabled(tmp_path, slot):
    cfg = config(tmp_path)
    cfg[slot]['enabled'] = not cfg[slot]['enabled']
    with pytest.raises(ValueError, match='clean-run slot differs'):
        rcpu.check_pins(cfg)


def test_worker_checkpoint_and_budget_pins_refuse_drift(tmp_path):
    cfg = config(tmp_path)
    for bad in (dict(cfg, cpu=dict(cfg['cpu'], max_sumo_workers=48)),
                dict(cfg, workers=40), dict(cfg, device='cuda'),
                dict(cfg, start_checkpoint_sha256=dict(cfg['start_checkpoint_sha256'], R='b'*64)),
                dict(cfg, final_updates=dict(A=24, B=24, R=12)),
                dict(cfg, budget=dict(cfg['budget'], phases={})),
                dict(cfg, teacher=dict(cfg['teacher'], B=dict(cfg['teacher']['B'], enabled=False)))):
        with pytest.raises(ValueError, match='REFUSED'):
            rcpu.check_pins(bad)


def test_concurrency_does_not_shorten_planner_or_episode_horizons(tmp_path):
    base = dict(search=dict(horizon_s=600, final_horizon_s=720),
                planner=dict(lookahead_s=600), workers=2)
    cfg = config(tmp_path, base=base)
    assert cfg['search'] == base['search'] and cfg['planner'] == base['planner']
    assert cfg['workers'] == 24 and base['workers'] == 2
    assert rcpu.EPISODE_SIM_S == 1200 and cfg['v2v_intent_horizon'] == 5


def test_isolated_jobs_are_dry_paired_and_match_full_budget():
    registration = _registration()
    jobs = rcpu.isolated_jobs(registration)
    assert jobs == rcpu.isolated_jobs(registration)
    assert len({j['id'] for j in jobs}) == 96
    assert sum(j['sim_s'] for j in jobs) == rcpu.quote()['isolated_replicated_sim_s']
    for u in (6, 12, 18, 24):
        for r in range(8):
            paired = [j for j in jobs if j['update'] == u and j['replicate'] == r]
            assert {j['arm'] for j in paired} == {'A', 'B', 'R'}
            assert len({(j['world_seed'], j['state_set_sha256'], j['window_set_sha256'])
                        for j in paired}) == 1
    for bad in (dict(registration, role='eval'),
                dict(registration, replicates=registration['replicates'][:-1]),
                dict(registration, replicates=[registration['replicates'][0]]*8)):
        with pytest.raises(ValueError, match='REFUSED'):
            rcpu.isolated_jobs(bad)


@pytest.mark.parametrize('field,value', [('world_seed', 99999), ('state_set_sha256', 'c'*64),
                                        ('window_set_sha256', 'd'*64),
                                        ('registration_sha256', 'e'*64),
                                        ('status', 'failed'), ('collisions', .5)])
def test_analysis_refuses_unpaired_or_ungraded_results(field, value):
    rows = _isolated_grid()
    rows[0][field] = value
    with pytest.raises(ValueError, match='REFUSED'):
        _evaluate(rows)


def test_existing_synergy_decomposition_adapter():
    job = rcpu.isolated_jobs(_registration())[0]
    row = rcpu.isolated_record(job, delta_all_new_veh_s=12.,
                               delta_only_member_new_veh_s={'cav1': 3., 'cav2': 8.},
                               safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0))
    assert row['metric'] == 4. and row['status'] == 'graded'
    assert row['world_seed'] == job['world_seed']
    with pytest.raises(ValueError, match='2..4 group members'):
        rcpu.isolated_record(job, delta_all_new_veh_s=12.,
                             delta_only_member_new_veh_s={'cav1': 3.}, safety={})


def test_preflight_binds_whole_request_config_registration_and_root(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    original = {n: (prep/n).read_bytes() for n in
                (rcpu.BUDGET_REQUEST, rcpu.APPROVAL, rcpu.CONFIG_FILE,
                 rcpu.ISOLATED_REGISTRATION_FILE, rcpu.CRITERION_FILE)}
    def approve_edit(field, filename):
        app = json.loads(original[rcpu.APPROVAL])
        app[field] = rcpu.sha(prep/filename)
        (prep/rcpu.APPROVAL).write_text(json.dumps(app), encoding='utf-8')
    request = json.loads(original[rcpu.BUDGET_REQUEST])
    request['budget']['phases']['isolated_replicated'] = 1  # total unchanged
    (prep/rcpu.BUDGET_REQUEST).write_text(json.dumps(request), encoding='utf-8')
    approve_edit('budget_request_sha256', rcpu.BUDGET_REQUEST)
    with pytest.raises(SystemExit, match='full budget request'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    for n, data in original.items():
        (prep/n).write_bytes(data)
    cfg = json.loads(original[rcpu.CONFIG_FILE])
    cfg['device'] = 'cuda'
    (prep/rcpu.CONFIG_FILE).write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(SystemExit, match='unbound config_file_sha256'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    approve_edit('config_file_sha256', rcpu.CONFIG_FILE)
    with pytest.raises(SystemExit, match='CPU-only'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    for n, data in original.items():
        (prep/n).write_bytes(data)
    (prep/rcpu.ISOLATED_REGISTRATION_FILE).unlink()
    with pytest.raises(SystemExit, match='registration_file_sha256'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    for n, data in original.items():
        (prep/n).write_bytes(data)
    root = config(tmp_path)['run_root']
    from pathlib import Path
    (Path(root)/'AUTHORITY.md').write_text('wrong authority', encoding='utf-8')
    with pytest.raises(SystemExit, match='single-ledger rule'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)


def test_on_preflight_still_requires_each_track2_binding(tmp_path, mirror):
    from pathlib import Path
    prep, stage = prep_dir(tmp_path, mirror)
    assert rcpu.preflight_dry(prep, stage_dir=stage, track2=True)['preflight_pass']
    # A Track 2 quote cannot be accepted by accidentally omitting the opt-in.
    with pytest.raises(SystemExit, match='differs from the 3-arm quote'):
        rcpu.preflight_dry(prep, stage_dir=stage)
    original_app = (prep/rcpu.APPROVAL).read_bytes()
    for field, error in (('criterion_sha256', 'criterion sha differs'),
                         ('config_file_sha256', 'unbound config_file_sha256'),
                         ('registration_file_sha256', 'unbound registration_file_sha256'),
                         ('manifest_file_sha256', 'unbound manifest_file_sha256')):
        app = json.loads(original_app)
        del app[field]
        (prep/rcpu.APPROVAL).write_text(json.dumps(app), encoding='utf-8')
        with pytest.raises(SystemExit, match=error):
            rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    (prep/rcpu.APPROVAL).write_bytes(original_app)
    for name, error in ((rcpu.CRITERION_FILE, 'criterion missing'),
                        (rcpu.CONFIG_FILE, 'config_file_sha256'),
                        (rcpu.ISOLATED_REGISTRATION_FILE, 'registration_file_sha256')):
        path = prep/name
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(SystemExit, match=error):
            rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
        path.write_bytes(original)
    cfg = json.loads((prep/rcpu.CONFIG_FILE).read_text())
    authority = Path(cfg['run_root'])/'AUTHORITY.md'
    original = authority.read_bytes()
    authority.write_text(rcpu.AUTHORITY_LINE+'effective-imitation commit '
                         +rcpu.EFFECTIVE_IMITATION_COMMIT+'\n', encoding='utf-8')
    with pytest.raises(SystemExit, match='pre-registered.*criterion'):
        rcpu.preflight_dry(prep, stage_dir=stage, track2=True)
    authority.write_bytes(original)
    assert rcpu.preflight_dry(prep, stage_dir=stage, track2=True)['preflight_pass']
