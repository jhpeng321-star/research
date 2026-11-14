"""CPU tests: walls-only MINED_MOVES rows carry their own verified situation binding and
are ingested without a replay-bank decision link (c4/clean3-minedlink).

A mined row binds (seed, situation id, t0, TRAIN world, executable fields, measured synergy
with one-sided 95% lower bound, n_reps); its practice restart snapshot comes from the row's
OWN sealed SUMO verification branch snapshot. Rows missing binding fields are refused (fail
closed). Legacy bank rows keep the decision-link path byte-identically, link first."""
from copy import deepcopy
import hashlib
import json
import shutil

import pytest
import torch

from . import bank_breadth as breadth, group_lane_bank as gb, replay_bank as rb
from . import practice_starts as ps, sumo_verify as sv, track2_backend as p1, verify_perseed as vp
from . import exact_grade as xg
from .protocol import digest
from .test_bank_breadth_cpu import bind_fixture_snapshots, observable_bank  # noqa: F401
from .test_practice_starts_cpu import bank_with_snapshots

VID = 'v0'
EXTRA_SEED = 5


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def wall_bank(root, arm='B'):
    """A native per_seed_v1 cooperative bank whose chosen plan is an executable wall
    profile plan (profile/policy_profile/perturbation_profile), with snapshot-bound
    receipts and own decision links."""
    from .test_bank_perseed_cpu import D
    bank_with_snapshots(root, arm, vids=(VID,), snapshot_seed=1)
    bank = rb.bank_root(root, arm)
    for kind in ('sumo_verification', 'confirm_replication'):
        path = bank/kind/'verifications'/(VID+'.json')
        receipt = json.loads(path.read_text())
        receipt['chosen_plan'].update(profile=[[0.3]*D, [0.3]*D], policy_profile=[[0.6]*D]*2,
                                      perturbation_profile=[[-0.3]*D]*2)
        write(path, xg._seal(receipt, 'receipt_sha256'))
    bind_fixture_snapshots(root, arm)
    return bank


def probe_bank(root, arm='B', extra_seed=EXTRA_SEED):
    """A walls-probe-shaped bank: sealed first-seed/replication/solo receipts, branch tapes,
    per_seed_v1 confirmations and the decision snapshot under
    sumo_verification/branches/<vid>/SNAPSHOT.pt - but NO replay-bank decision link and NO
    snapshots/<state_id>.pt (the shape that raised 'no decision link' on cpuV)."""
    bank = wall_bank(root, arm)
    snap = torch.load(bank/'snapshots'/f'state_{arm}_{VID}.pt', weights_only=False)
    directory = bank/'sumo_verification'/'branches'/VID
    torch.save(snap, directory/'SNAPSHOT.pt')
    (directory/'SNAPSHOT.sha256').write_text(
        hashlib.sha256((directory/'SNAPSHOT.pt').read_bytes()).hexdigest()+'\n')
    (bank/'snapshots'/f'state_{arm}_{VID}.pt').unlink()
    (bank/'decisions'/f'state_{arm}_{VID}.json').unlink()
    first = json.loads((bank/'sumo_verification'/'verifications'/(VID+'.json')).read_text())
    second = json.loads((bank/'confirm_replication'/'verifications'/(VID+'.json')).read_text())
    extra_root = bank/f'confirm_seed{extra_seed}_all8'
    shutil.copytree(bank/'confirm_replication', extra_root)
    salt = vp.extra_seed_salt(VID, extra_seed)
    third = xg._seal(dict(second, snapshot_sha256=sv.snapshot_sha(rb.replicate(snap, salt))),
                     'receipt_sha256')
    write(extra_root/'verifications'/(VID+'.json'), third)
    with p1._bank_view(dict(run_root=str(root), training_seed=4)):
        record = deepcopy(rb.confirmations(root, arm)[VID])
    record.update(verify_mode=vp.ALL8, arm=arm,
        first_seed=dict(verification_id=VID, receipt_sha256=first['receipt_sha256']),
        seed2=dict(verification_id=VID, receipt_sha256=second['receipt_sha256'],
                   snapshot_sha256=second['snapshot_sha256'], salt=VID),
        extra={str(extra_seed): dict(verification_id=VID, receipt_sha256=third['receipt_sha256'],
                                     snapshot_sha256=third['snapshot_sha256'], salt=salt)})
    ex = deepcopy(record['per_seed_v1']['examples'][1])
    ex.update(seed_index=extra_seed, example_id=f'{VID}#s{extra_seed}')
    ex['tape']['branches_dir'] = str(extra_root/'branches')
    record['per_seed_v1']['examples'].append(ex)
    write(bank/vp.RECORDS/(VID+'.json'), xg._seal(record, 'confirmation_sha256'))
    return snap, record, first


def mined_row(root, seed, plan, snap, **overrides):
    duration = int(plan['duration_s'])
    row = dict(schema=gb.SCHEMA, arm='B', run_root=str(root), verification_id=VID,
               plan_sha256=digest(plan), replay_sources={}, seed=seed,
               situation_id=f"{snap['episode']['id']}_t{snap['t0']}", t0=int(snap['t0']),
               world=dict(snap['episode']['world']), family='wall', members=list(plan['members']),
               duration_s=duration, profile=[[0.3]*duration for _ in plan['members']],
               policy_profile=[[0.6]*duration for _ in plan['members']],
               perturbation_profile=[[-0.3]*duration for _ in plan['members']],
               synergy_veh_s={str(seed): 336.7}, synergy_lower95={str(seed): 73.5},
               advantage_veh_s={str(seed): 490.3}, adv_lower95={str(seed): 135.5},
               best_solo=plan['members'][0], n_reps=6,
               statistic='per-seed one-sided 95% circular-block (30 s) paired bootstrap',
               base='frozen MAPPO u0', source='cpu test probe')
    row.update(overrides)
    return row


def mined_config(tmp_path, rows):
    path = tmp_path/'MINED_MOVES.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    return dict(run_root=str(tmp_path), training_seed=4, group_lane_moves=True,
                mined_moves=str(path))


def mined_probe(tmp_path):
    """(config, run root, snapshot, plan, rows) of a probe bank plus rows binding its seed-1
    and seed-5 examples (the seed-2 example of the same plan has no row: not a mined example)."""
    root = tmp_path/'probe'
    snap, _, first = probe_bank(root)
    plan = first['chosen_plan']
    rows = [mined_row(root, seed, plan, snap) for seed in (1, EXTRA_SEED)]
    return mined_config(tmp_path, rows), root, snap, plan, rows


def test_mined_rows_with_full_binding_ingest_as_bank_examples_and_practice_starts(
        tmp_path, observable_bank):
    config, root, snap, plan, _rows = mined_probe(tmp_path)
    seen = []
    breadth.visit(config, 'B', lambda **row: seen.append(row))
    by_seed = {r['example']['seed_index']: r for r in seen}
    assert set(by_seed) == {1, EXTRA_SEED}
    assert all(r['pairs'] for r in seen)
    bindings = {seed: r['cfg']['_mined_binding'] for seed, r in by_seed.items()}
    assert set(bindings) == {1, EXTRA_SEED}
    assert bindings[1]['plan_sha256'] == digest(plan) and bindings[1]['world']['seed'] == 986000
    assert bindings[EXTRA_SEED]['synergy_lower95'] == {str(EXTRA_SEED): 73.5}
    pool = breadth.practice_pool(config, 'B')
    assert {r['seed_index'] for r in pool} == {1, EXTRA_SEED}   # seed 2 has no row: skipped
    assert all(r['mined'] and r['demonstrate'] and r['mined_plan_sha256'] == digest(plan)
               for r in pool)
    first_restart = ps.load_snapshot(str(tmp_path), 'B', next(r for r in pool if r['seed_index'] == 1))
    assert first_restart['t0'] == snap['t0'] == 400
    assert sv.snapshot_sha(first_restart) == sv.snapshot_sha(snap)
    extra_restart = ps.load_snapshot(str(tmp_path), 'B',
                                     next(r for r in pool if r['seed_index'] == EXTRA_SEED))
    expected = rb.replicate(snap, vp.extra_seed_salt(VID, EXTRA_SEED))
    assert sv.snapshot_sha(extra_restart) == sv.snapshot_sha(expected)
    assert torch.equal(extra_restart['generator_state'], expected['generator_state'])
    row = next(r for r in pool if r['seed_index'] == EXTRA_SEED)
    assert row['snapshot_lookup']['mined_binding']['seed'] == EXTRA_SEED
    call = ps.wave_call(str(tmp_path), 'B', 'training', 'train', row, 0, {})
    demo = call['extras']['practice']['demonstration']
    assert demo['mined'] is True and demo['plan_sha256'] == digest(plan)


@pytest.mark.parametrize('field', ['n_reps', 'synergy_lower95', 't0', 'profile'])
def test_mined_row_missing_binding_field_refuses_fail_closed(tmp_path, observable_bank, field):
    config, _root, _snap, plan, rows = mined_probe(tmp_path)
    for row in rows:
        row.pop(field)
    config = mined_config(tmp_path, rows)
    with pytest.raises(ValueError, match='verified situation binding'):
        breadth.practice_pool(config, 'B')


def test_mined_row_missing_branch_decision_snapshot_refuses(tmp_path, observable_bank):
    config, root, _snap, _plan, _rows = mined_probe(tmp_path)
    (rb.bank_root(root, 'B')/'sumo_verification'/'branches'/VID/'SNAPSHOT.pt').unlink()
    with pytest.raises(ValueError, match='no sealed decision snapshot'):
        breadth.practice_pool(config, 'B')


def test_mined_binding_tampering_against_the_sealed_verification_refuses(tmp_path, observable_bank):
    config, root, snap, plan, rows = mined_probe(tmp_path)
    rows[1] = mined_row(root, EXTRA_SEED, plan, dict(snap, t0=snap['t0']-1))
    with pytest.raises(ValueError, match='decision time'):
        breadth.practice_pool(mined_config(tmp_path, rows), 'B')
    config, root, snap, plan, rows = mined_probe(tmp_path/'again')
    other = mined_row(root, EXTRA_SEED, plan, snap, world=dict(snap['episode']['world'], seed=986071))
    with pytest.raises(ValueError, match='world differs'):
        breadth.practice_pool(mined_config(tmp_path/'again', [other]), 'B')


def test_decision_link_path_stays_byte_identical_and_wins_over_the_mined_binding(
        tmp_path, observable_bank):
    # A legacy bank keeps its decision link: the snapshot comes from snapshots/<state_id>.pt
    # even when the row also carries a full binding; the branch snapshot is never read.
    legacy = tmp_path/'legacy'
    bank = wall_bank(legacy, 'B')
    snap = torch.load(bank/'snapshots'/f'state_B_{VID}.pt', weights_only=False)
    first = json.loads((bank/'sumo_verification'/'verifications'/(VID+'.json')).read_text())
    assert not (bank/'sumo_verification'/'branches'/VID/'SNAPSHOT.pt').exists()
    rows = [mined_row(legacy, 1, first['chosen_plan'], snap)]
    pool = breadth.practice_pool(mined_config(tmp_path, rows), 'B')
    assert {r['seed_index'] for r in pool} == {1}
    restart = ps.load_snapshot(str(tmp_path), 'B', pool[0])
    assert torch.equal(restart['generator_state'], snap['generator_state'])
    assert pool[0]['state_id'] == f'state_B_{VID}'
    # Without a mined row and without a link, the legacy refusal is unchanged.
    (bank/'decisions'/f'state_B_{VID}.json').unlink()
    with pytest.raises(ValueError, match='no decision link for '+VID):
        breadth.practice_pool(dict(run_root=str(legacy), training_seed=4), 'B')


def test_decision_snapshot_link_wins_and_legacy_refusal_stays_registered(tmp_path):
    bank = rb.bank_root(tmp_path, 'B')
    link = dict(verification_id=VID, state_id='state1')
    write(bank/'decisions'/'state1.json', link)
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    torch.save(dict(observable='own'), bank/'snapshots'/'state1.pt')
    r = dict(verification_id=VID, t0=400, _policy_version='unknown')
    assert rb._decision_snapshot(tmp_path, 'B', r, {VID: link}) == dict(observable='own')
    with pytest.raises(ValueError, match='no decision link for '+VID):
        rb._decision_snapshot(tmp_path, 'B', r, {})
    full = gb.binding(mined_row(tmp_path, 1, dict(members=['c1', 'c2'], duration_s=20),
                                dict(episode=dict(id='e', world=dict(seed=986000)), t0=400)))
    with pytest.raises(ValueError, match='no sealed decision snapshot'):
        rb._decision_snapshot(tmp_path, 'B', r, {}, mined=full)


def test_binding_validator_full_partial_and_v1_pointer():
    full = dict(seed=2, situation_id='s', t0=455, world=dict(seed=986061), members=['a', 'b'],
                duration_s=3, profile=[[.1]*3]*2, policy_profile=[[0.]*3]*2,
                perturbation_profile=[[.1]*3]*2, synergy_veh_s={'2': 1.}, synergy_lower95={'2': .5},
                n_reps=6)
    value = gb.binding(dict(schema=gb.SCHEMA, arm='B', verification_id='v', plan_sha256='p', **full))
    assert value['seed'] == 2 and value['world'] == dict(seed=986061)
    assert value['synergy_veh_s'] == {'2': 1.0} and gb.binding(value) == value
    assert gb.binding(dict(verification_id='v', plan_sha256='p')) is None   # v1 pointer row
    for mutation in (dict(full, seed=0), dict(full, n_reps=True), dict(full, duration_s=0),
                     dict(full, t0='x'), dict(full, situation_id=''), dict(full, world=dict(seed=1)),
                     dict(full, members=['a', 'a']), dict(full, profile=[[.1]*3]),
                     dict(full, profile=[[.1]*2]*2), dict(full, profile=[[True]*3]*2),
                     dict(full, synergy_lower95={'1': .5}), dict(full, synergy_veh_s={'2': None})):
        with pytest.raises(ValueError, match='MINED_MOVES binding'):
            gb.binding(mutation)
    partial = {k: v for k, v in full.items() if k != 'n_reps'}
    with pytest.raises(ValueError, match='partial'):
        gb.binding(partial)
