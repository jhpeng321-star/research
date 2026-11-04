"""T2H: larger practice share, own augmented states, fairness and default parity."""
import json
from collections import Counter

import pytest
import torch

from tools.coopmwm import exact_grade as xg, practice_starts as starts
from tools.coopmwm import practice_weighting as pw, replay_bank as rb
from tools.coopmwm import sumo_verify as sv, v2v_launch3 as launch, v2v_runner
from tools.coopmwm.test_bank_perseed_cpu import ps as bank_fixture  # noqa: F401
from tools.coopmwm.test_practice_starts_cpu import (
    _cfg, bank_with_snapshots, run_one_update, snapshot)
from tools.coopmwm.test_v2v_live_cpu import charges_ledger  # noqa: F401


@pytest.fixture(name='ps')
def per_seed(bank_fixture):
    return bank_fixture


def augment(root, arm='B', *, rows=None, verified=True):
    bank = rb.bank_root(root, arm)
    state = f'aug_state_{arm}'
    snap = snapshot(world_seed=986020, gen_seed=21)
    snap['episode']['arm'] = arm
    torch.save(snap, bank/'snapshots'/(state+'.pt'))
    (bank/'decisions'/(state+'.json')).write_text(json.dumps(
        dict(verification_id='aug1', state_id=state)))
    row = dict(example_id='aug1#s1', parent_example_id='v0#s1',
               verification_id='aug1', state_id=state, seed_index=1,
                wm_cooperative=True, snapshot_sha256=sv.snapshot_sha(snap))
    prelaunch = dict(phase='track2_augmentation', allocation_sim_s=200000,
                     provenance='registered prelaunch allocation; CPU fixture')
    if verified:
        from .test_bank_perseed_cpu import make, win, lose
        from . import bank_statistics as bs
        from .protocol import digest
        make(root, 'aug1', win(), lose(), arm=arm, syn1=win())
        path = bank/'sumo_verification'/'verifications'/'aug1.json'
        receipt = json.loads(path.read_text())
        receipt.update(snapshot_sha256=sv.snapshot_sha(snap), episode=snap['episode'],
                       branch_sim_s=1200, charged_sim_s=2400)
        receipt = xg._seal(receipt, 'receipt_sha256')
        path.write_text(json.dumps(receipt))
        ids = ['aug1_policy', 'aug1_cand1']
        row['sumo_evidence'] = dict(receipt_path='sumo_verification/verifications/aug1.json',
            receipt_sha256=receipt['receipt_sha256'], plan_sha256=digest(receipt['chosen_plan']),
            charge_ids=ids)
        (root/'CHARGES.jsonl').write_text(''.join(json.dumps(dict(id=i,
            phase=prelaunch['phase'], sim_s=1200))+'\n' for i in ids))
        bs._CACHE.clear()
    record = xg._seal(dict(schema=pw.SCHEMA, arm=arm, rows=rows if rows is not None else [row]),
                      'practice_augmented_sha256')
    record = xg._seal(dict(record, prelaunch=prelaunch), 'practice_augmented_sha256')
    (bank/'PRACTICE_AUGMENTED.json').write_text(json.dumps(record))
    return row, record


def test_optional_config_and_default_bytes(tmp_path):
    legacy, _ = _cfg(tmp_path, practice_starts=True)
    assert not set(pw.KEYS) & legacy.keys()
    explicit, _ = _cfg(tmp_path, practice_starts=True, practice_weighting=False)
    assert {k: v for k, v in explicit.items() if k not in pw.KEYS} == legacy
    assert pw.effective_fraction(legacy) == pw.effective_fraction(explicit) == .25
    weighted, _ = _cfg(tmp_path, practice_starts=True, practice_weighting=True)
    assert pw.effective_fraction(weighted) == .5
    assert weighted['budget']['phases']['v2v_train'] == (
        launch.V2V_PHASES['v2v_train'] + 2*24*12*1200)
    assert launch.practice_reference_quote(.5) == 691200
    assert sum(weighted['budget']['phases'].values()) <= launch.CEILING_SIM_S


@pytest.mark.parametrize('bad', [dict(practice_weighting='yes'),
    dict(practice_weighting=1), dict(practice_weighted_fraction=.75),
    dict(practice_weighted_fraction=float('nan')), dict(practice_weighted_fraction=True)])
def test_unregistered_weighting_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        _cfg(tmp_path, practice_starts=True, **bad)


def test_br_equality_and_practice_required(tmp_path):
    with pytest.raises(ValueError, match='requires practice_starts'):
        _cfg(tmp_path, practice_weighting=True)
    cfg, _ = _cfg(tmp_path, practice_starts=True, practice_weighting=True)
    opts = launch.practice_options_for(cfg, 'B')
    assert opts == launch.practice_options_for(cfg, 'R')
    equal = dict(cfg, practice_starts_by_arm=dict(B=opts, R=dict(opts)))
    launch.practice_options_for(equal, 'R')
    equal['practice_starts_by_arm']['R']['practice_weighting'] = False
    with pytest.raises(ValueError, match='identical'):
        launch.practice_options_for(equal, 'R')
    with pytest.raises(ValueError, match='only arms'):
        launch.practice_options_for(cfg, 'A')


def test_default_never_reads_augmentation_and_plan_bytes_equal(tmp_path, monkeypatch, ps):
    import time
    monkeypatch.setattr(time, 'time', lambda: 123.)
    bank_with_snapshots(tmp_path/'one', vids=('v0', 'v1'))
    bank_with_snapshots(tmp_path/'two', vids=('v0', 'v1'))
    monkeypatch.setattr(pw, 'augmented_examples', lambda *a: pytest.fail('default read augmentation'))
    args = dict(fraction=.25, repeat_cap=2, count=6, training_seed=4)
    first = starts.plan_for_update(tmp_path/'one', 'B', 1, **args)
    second = starts.plan_for_update(tmp_path/'two', 'B', 1, weighting=False, **args)
    assert first == second
    assert starts.plan_path(tmp_path/'one', 'B', 1).read_bytes() == (
        starts.plan_path(tmp_path/'two', 'B', 1).read_bytes())


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bank_plus_augmented_distinct_states_and_frozen_resume(tmp_path, monkeypatch, ps, arm):
    bank_with_snapshots(tmp_path, arm, vids=('v0',))
    row, manifest = augment(tmp_path, arm)
    plan = starts.plan_for_update(tmp_path, arm, 1, fraction=.5, repeat_cap=2,
        count=12, training_seed=4, weighting=True)
    assert len(plan['rows']) == 6  # two bank seed states + one new augmented state, cap 2
    counts = Counter(r['example_id'] for r in plan['rows'])
    assert counts == {'v0#s1': 2, 'v0#s2': 2, 'aug1#s1': 2}
    assert plan['weighting']['augmented_selected'] == 2
    assert plan['weighting']['augmentation_sha256'] == manifest['practice_augmented_sha256']
    loaded = starts.load_snapshot(tmp_path, arm, next(
        r for r in plan['rows'] if r['example_id'] == row['example_id']))
    assert loaded['episode']['world']['seed'] == 986020
    monkeypatch.setattr(pw, 'augmented_examples', lambda *a: pytest.fail('resume resampled'))
    assert starts.plan_for_update(tmp_path, arm, 1, fraction=.5, repeat_cap=2,
        count=12, training_seed=4, weighting=True) == plan


@pytest.mark.parametrize('change', [dict(parent_example_id='foreign#s1'),
    dict(verification_id='foreign'), dict(state_id='../escape'), dict(wm_cooperative=False),
    dict(seed_index=2), dict(snapshot_sha256='0'*64)])
def test_invalid_augmentation_rejected_before_plan(tmp_path, ps, change):
    bank_with_snapshots(tmp_path, vids=('v0',))
    row, _ = augment(tmp_path)
    augment(tmp_path, rows=[dict(row, **change)])
    with pytest.raises(ValueError):
        starts.plan_for_update(tmp_path, 'B', 1, fraction=.5, repeat_cap=2,
            count=12, training_seed=4, weighting=True)
    assert not starts.plan_path(tmp_path, 'B', 1).exists()


def test_manifest_integrity_arm_and_duplicate_state(tmp_path, ps):
    bank_with_snapshots(tmp_path, vids=('v0',))
    row, record = augment(tmp_path)
    path = rb.bank_root(tmp_path, 'B')/'PRACTICE_AUGMENTED.json'
    record['arm'] = 'R'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='mismatch'):
        pw.augmented_examples(tmp_path, 'B', starts.examples(tmp_path, 'B'))
    # An alias of a verified example is not independently admitted evidence.
    augment(tmp_path, rows=[row, dict(row, example_id='aug2#s1')])
    with pytest.raises(ValueError, match='own admitted'):
        pw.augmented_examples(tmp_path, 'B', starts.examples(tmp_path, 'B'))


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_wm_only_child_never_enters_plan(tmp_path, ps, arm):
    bank_with_snapshots(tmp_path, arm, vids=('v0',))
    augment(tmp_path, arm, verified=False)
    with pytest.raises(ValueError, match='receipt path'):
        starts.plan_for_update(tmp_path, arm, 1, fraction=.5, repeat_cap=2,
                              count=12, training_seed=4, weighting=True)
    assert not starts.plan_path(tmp_path, arm, 1).exists()


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('failure', ['missing', 'failed', 'world', 'snapshot', 'arm', 'plan',
    'seed', 'admission', 'budget', 'charges', 'receipt_hash', 'overdrawn', 'wrong_phase',
    'undercharged'])
def test_child_verification_and_prelaunch_binding(tmp_path, ps, arm, failure):
    from . import bank_statistics as bs
    bank_with_snapshots(tmp_path, arm, vids=('v0',))
    row, manifest = augment(tmp_path, arm)
    bank = rb.bank_root(tmp_path, arm)
    path = bank/'sumo_verification'/'verifications'/'aug1.json'
    receipt = json.loads(path.read_text())
    if failure == 'missing':
        path.unlink()
    elif failure == 'seed':
        manifest['rows'][0]['seed_index'] = 2
    elif failure == 'admission':
        cpath = bank/'confirmations'/'aug1.json'
        confirmation = json.loads(cpath.read_text())
        confirmation['checks']['safety'] = False
        confirmation['safety_deltas']['replication']['collisions'] = 1
        confirmation['safety_deltas']['extended']['collisions'] = 1
        cpath.write_text(json.dumps(xg._seal(confirmation, 'confirmation_sha256')))
    elif failure == 'budget':
        manifest['prelaunch']['allocation_sim_s'] = 200001
    elif failure == 'charges':
        (tmp_path/'CHARGES.jsonl').write_text('')
    elif failure in ('overdrawn', 'wrong_phase', 'undercharged'):
        path_ledger = tmp_path/'CHARGES.jsonl'
        charges = [json.loads(line) for line in path_ledger.read_text().splitlines()]
        if failure == 'overdrawn':
            charges.append(dict(id='other_prelaunch', phase=manifest['prelaunch']['phase'], sim_s=200000))
        elif failure == 'wrong_phase':
            charges[0]['phase'] = 'other_phase'
        else:
            charges[0]['sim_s'] = 1
        path_ledger.write_text(''.join(json.dumps(c)+'\n' for c in charges))
    elif failure == 'receipt_hash':
        manifest['rows'][0]['sumo_evidence']['receipt_sha256'] = '0'*64
    else:
        if failure == 'failed':
            receipt['status'] = 'failed'
        elif failure == 'world':
            receipt['episode']['world']['seed'] += 1
        elif failure == 'snapshot':
            receipt['snapshot_sha256'] = '0'*64
        elif failure == 'arm':
            receipt['episode']['arm'] = 'A'
        elif failure == 'plan':
            receipt['chosen_plan']['id'] = 'different'
        receipt = xg._seal(receipt, 'receipt_sha256')
        path.write_text(json.dumps(receipt))
        manifest['rows'][0]['sumo_evidence']['receipt_sha256'] = receipt['receipt_sha256']
    bs._CACHE.clear()
    (bank/'PRACTICE_AUGMENTED.json').write_text(json.dumps(
        xg._seal(manifest, 'practice_augmented_sha256')))
    with pytest.raises((ValueError, FileNotFoundError)):
        starts.plan_for_update(tmp_path, arm, 1, fraction=.5, repeat_cap=2,
                              count=12, training_seed=4, weighting=True)


def test_frozen_augmented_plan_rechecks_evidence_before_wave(tmp_path, ps):
    bank_with_snapshots(tmp_path, vids=('v0',))
    augment(tmp_path)
    plan = starts.plan_for_update(tmp_path, 'B', 1, fraction=.5, repeat_cap=2,
                                 count=12, training_seed=4, weighting=True)
    child = next(r for r in plan['rows'] if r['source'] == 'augmented')
    (rb.bank_root(tmp_path, 'B')/'sumo_verification'/'verifications'/'aug1.json').unlink()
    with pytest.raises(FileNotFoundError):
        starts.wave_call(tmp_path, 'B', 'training', 'train', child, 0, {})


def test_real_wave_receipt_twelve_of_twenty_four_and_r_matches_realised(tmp_path, monkeypatch, ps):
    calls, receipt = run_one_update(tmp_path/'b', monkeypatch, arm='B', practice=True,
        weighting=True, vids=tuple(f'v{i}' for i in range(4)))
    assert [len(c['worlds']) for c in calls] == [12, 12]
    assert receipt['practice']['practice'] == receipt['practice']['normal'] == 12
    assert receipt['practice']['weighting'] == dict(enabled=True, practice=12,
                                                   cooperative=12, augmented=0)
    for item in calls[1]['extras']['practice_batch']:
        assert item['practice']['charged_sim_s'] == 1200
        assert item['practice']['weighting']['enabled'] is True
    calls, receipt = run_one_update(tmp_path/'r', monkeypatch, arm='R', practice=True,
        weighting=True, vids=tuple(f'v{i}' for i in range(4)), b_practice=9)
    assert [len(c['worlds']) for c in calls] == [15, 9]
    assert receipt['practice']['practice'] == 9


def test_augmented_state_reaches_real_wave_and_update_receipt(tmp_path, monkeypatch, ps):
    root = tmp_path/'v2v'
    bank_with_snapshots(root, vids=('v0',))
    augment(root)
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, weighting=True)
    assert receipt['practice']['weighting']['augmented'] == 2
    assert len(calls[1]['worlds']) == 6
    assert sum(w['seed'] == 986020 for w in calls[1]['worlds']) == 2


def test_empty_weighted_bank_uses_normal_starts(tmp_path, monkeypatch, ps):
    calls, receipt = run_one_update(tmp_path, monkeypatch, arm='B', practice=True, weighting=True)
    assert len(calls) == 1 and len(calls[0]['worlds']) == 24
    assert receipt['practice']['practice'] == 0
    plan = json.loads(starts.plan_path(tmp_path/'v2v', 'B', 1).read_text())
    assert plan['weighting']['enabled'] is True and not plan['rows']


def test_weighted_r_exhausts_cooperative_capacity_first(tmp_path, ps):
    from tools.coopmwm.test_bank_perseed_cpu import make, win
    bank_with_snapshots(tmp_path, 'R', vids=('v0',))
    make(tmp_path, 'other', win(), win(), arm='R', cooperation=False)
    bank = rb.bank_root(tmp_path, 'R')
    (bank/'decisions'/'other.json').write_text(json.dumps(
        dict(verification_id='other', state_id='other')))
    plan = starts.plan_for_update(tmp_path, 'R', 1, fraction=.5, repeat_cap=2,
                                  count=6, training_seed=4, weighting=True)
    assert len(plan['rows']) == 6
    assert [r['fallback'] for r in plan['rows']] == [False]*4 + [True]*2
    assert plan['weighting']['cooperative_selected'] == 4


@pytest.mark.parametrize('stage,arm,expected', [
    ('smoke', 'B', 36*1200), ('train', 'R', 23*36*1200), ('smoke', 'A', 24*1200),
    (None, None, launch.V2V_PHASES['v2v_train'] + 691200)])
def test_weighted_dispatch_quotes(tmp_path, monkeypatch, stage, arm, expected):
    from tools.coopmwm import v2v_live as live, replay_start
    from tools.coopmwm.test_v2v_launch_wiring_cpu import authority
    root, cfg, _ = authority(tmp_path)
    cfg.update(practice_starts=True, practice_weighting=True, practice_fraction=.25)
    monkeypatch.setattr(replay_start, 'alive', lambda *a: False)
    quoted = []
    monkeypatch.setattr(launch, 'wait_for_preflight',
                        lambda *a, **kw: quoted.append(kw['phase_costs']))
    live.launch3_check(root, cfg, stage, arm)
    assert quoted[0]['v2v_train'] == expected


def test_weighted_resume_quote_and_smoke_credit(tmp_path, charges_ledger):
    from tools.coopmwm import v2v_live as live
    from tools.coopmwm.test_v2v_live_cpu import (
        NS, started_root, smoke_charges, write_charges)
    cfg = dict(job_namespace=NS, practice_starts=True, practice_weighting=True,
               practice_fraction=.25)
    root, _ = started_root(tmp_path/'started')
    remaining = live.arm_remaining(root, cfg, 'B', live=False)
    assert remaining['remaining_reference_sim_s'] == 22*12*1200
    fresh = tmp_path/'fresh'
    fresh.mkdir()
    rows = smoke_charges(fresh, 'B', 1)
    write_charges(fresh, rows)
    assert live.remaining_phase_costs(fresh, cfg)['v2v_train'] == (
        launch.V2V_PHASES['v2v_train'] + 691200 - 1200)


@pytest.mark.parametrize('practice', [False, True])
def test_default_bytes_against_shared_base(tmp_path, monkeypatch, ps, practice):
    """Differential regression oracle, independent of the new sampling implementation."""
    import subprocess
    import time
    import types
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    def original(name):
        source = subprocess.check_output(['git', 'show',
            f'13f184b5:tools/coopmwm/{name}.py'], cwd=root)
        module = types.ModuleType('tools.coopmwm._base_'+name)
        module.__package__ = 'tools.coopmwm'
        exec(compile(source, '<shared-base-'+name+'>', 'exec'), module.__dict__)
        return module
    old_starts, old_runner = original('practice_starts'), original('v2v_runner')
    monkeypatch.setattr(time, 'time', lambda: 123.)
    old_cfg = old_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v',
        legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
        common_checkpoint_sha256='a'*64, practice_starts=practice)
    current, _ = _cfg(tmp_path, practice_starts=practice)
    assert json.dumps(current, sort_keys=True) == json.dumps(old_cfg, sort_keys=True)
    for arm in ('B', 'R'):
        bank_with_snapshots(tmp_path/'old', arm, vids=('v0', 'v1'))
        bank_with_snapshots(tmp_path/'new', arm, vids=('v0', 'v1'))
        args = dict(fraction=.25, repeat_cap=2, count=6, training_seed=4)
        old = old_starts.plan_for_update(tmp_path/'old', arm, 1, **args)
        new = starts.plan_for_update(tmp_path/'new', arm, 1, **args)
        assert new == old
        assert starts.plan_path(tmp_path/'old', arm, 1).read_bytes() == (
            starts.plan_path(tmp_path/'new', arm, 1).read_bytes())
