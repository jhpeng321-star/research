"""W2 negative "do not wall" labels: schema, per-seed rule, ingestion and B = R (CPU fakes)."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from . import launch_production as launch, wall_negatives as wn, wall_release as wr
from .moves import Move
from .protocol import digest
from .test_track2_backend_cpu import cfg, cpu_threads, rows, sample  # noqa: F401


def frames(t0=300):
    return {str(t): wn.whitelist(rows()) for t in (t0-1, t0)}


def neg_row(arm='B', seed=2, synergy=-120., advantage=-90., t0=300, members=('c1', 'c2'), release=0):
    move = Move('wall', members, 0, 20, .6, release, 'linear' if release else 'step')
    plan = wr.plan(move, [[0.]*wr.duration(move)]*len(members))
    return wn.example(arm=arm, situation_id='ep_t0300', verification_id='sv_ep_t0300_x', plan=plan,
                      seed=seed, t0=t0, world=dict(seed=986061), frames=frames(t0), source='CPU fixture',
                      synergy=synergy, synergy_lower95=None if synergy is None else synergy-50,
                      advantage=advantage)


def test_per_seed_rule_and_basis():
    assert neg_row(synergy=-1.)['basis'] == 'synergy'
    assert neg_row(synergy=None, advantage=-5.)['basis'] == 'advantage'
    assert neg_row(synergy=3., advantage=-5.) is None          # measured synergy >= 0: not a negative
    assert neg_row(synergy=None, advantage=4.) is None
    assert neg_row(synergy=0.) is None
    row = neg_row(seed=5, synergy=-2.)
    assert set(row['synergy_veh_s']) == {'5'} and row['label'] == wn.LABEL
    # Non-wall plans never become wall negatives.
    gap = dict(id='g', members=['c1', 'c2'], duration_s=20, perturbation_profile=[[-.3]*20, [0.]*20])
    assert wn.example(arm='B', situation_id='s', verification_id='v', plan=gap, seed=1, t0=300,
                      world=dict(seed=986061), frames=frames(), source='x', synergy=-1.) is None


@pytest.mark.parametrize('mutate', [
    lambda r: r['synergy_veh_s'].update({'2': 4.}),                       # not a losing seed
    lambda r: r['synergy_veh_s'].update({'3': -4.}),                      # another seed's value
    lambda r: r.update(basis='advantage'),                               # synergy measured
    lambda r: r['frames'][str(r['t0'])][0].update(tau=1.2),               # unwhitelisted field
    lambda r: r['frames'][str(r['t0'])][0].update(s_m=-1.),               # frames hash mismatch
    lambda r: r.update(world=dict(seed=964001)),                         # evaluation world
    lambda r: r.update(arm='A'),
    lambda r: r['move'].update(family='gap_open'),
    lambda r: r.update(extra=1)])
def test_validation_fails_closed(mutate):
    row = deepcopy(neg_row())
    mutate(row)
    with pytest.raises(ValueError):
        wn.validate(row)


def write_rows(path, rows_):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows_), encoding='utf-8')


def test_trigger_records_target_none_against_the_losing_wall_b_and_r_identical(cfg, tmp_path):
    path = tmp_path/'WALL_NEGATIVES.jsonl'
    b, r = neg_row('B'), neg_row('R')
    write_rows(path, [b, r, neg_row('B', seed=3, release=10)])
    config = dict(cfg, wall_negative_labels=True, wall_negatives=str(path))
    policy_b, _, _ = sample(config)
    records, contexts, receipt = wn.trigger_records(config, 'B', policy_b)
    assert receipt['rows'] == 2 and receipt['contexts'] == len(records) == 4   # 2 rows x 2 observing members
    assert receipt['basis'] == {'synergy': 2, 'advantage': 0}
    for batch, weight in records:
        assert batch.actions.tolist() == [0] and bool(batch.mask[0, 1])
        assert batch.tables[0][1].move.family == 'wall' and weight == 1.
        assert batch.context['members'][0] in batch.tables[0][1].move.members
    assert {c['family'] for c in contexts} == {wn.FAMILY_KEY} and all(c['negative'] for c in contexts)
    # R reads only its own rows, with the identical rule/observer: same batches as B's same row.
    policy_r, _, _ = sample(config)
    policy_r.arm = 'R'
    r_records, _, r_receipt = wn.trigger_records(config, 'R', policy_r)
    assert r_receipt['rows'] == 1 and len(r_records) == 2
    for (rb_, _), (bb_, _) in zip(r_records, records[:2]):
        assert torch.equal(rb_.descriptors, bb_.descriptors) and rb_.context == bb_.context
    # The CE on these labels moves probability from the losing wall to 'none'.
    batch = records[0][0]
    before = policy_b.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0].softmax(-1)
    for _ in range(5):
        policy_b.trigger_optimizer.zero_grad()
        policy_b.trigger.imitation_loss(batch, batch.actions).backward()
        policy_b.trigger_optimizer.step()
    after = policy_b.trigger.distribution(batch.context, batch.descriptors, batch.mask)[0].softmax(-1)
    assert after[0, 0] > before[0, 0] and after[0, 1] < before[0, 1]


def test_unobservable_wall_is_counted_not_labelled(cfg, tmp_path):
    row = neg_row()
    far = deepcopy(row)
    for t in far['frames']:
        for v in far['frames'][t]:
            if v['vid'] == 'c2':
                v['s_m'] = 5000.
    far['frames_sha256'] = digest(far['frames'])
    write_rows(tmp_path/'WALL_NEGATIVES.jsonl', [wn.validate(far)])
    config = dict(cfg, wall_negative_labels=True)
    policy, _, _ = sample(config)
    records, _, receipt = wn.trigger_records(config, 'B', policy)
    assert records == [] and receipt['unobservable_rows'] == 1


def test_configured_missing_file_refuses_and_default_absent_is_empty(cfg, tmp_path):
    policy, _, _ = sample(cfg)
    with pytest.raises(ValueError, match='missing'):
        wn.trigger_records(dict(cfg, wall_negative_labels=True, wall_negatives=str(tmp_path/'nope.jsonl')),
                           'B', policy)
    assert wn.trigger_records(dict(cfg, wall_negative_labels=True), 'B', policy)[0] == []


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bank_supervision_adds_negatives_only_when_on(cfg, monkeypatch, tmp_path, arm):
    from . import bank_breadth as breadth, bank_v2v
    policy, recorded, _ = sample(cfg)
    plan = dict(id='wall', members=['c1', 'c2'], duration_s=30, perturbation_profile=[[-.3]*30]*2)

    def visit(config, requested_arm, consume):
        for i in range(3):
            consume(cfg=config, receipt=dict(verification_id=f'{requested_arm}_{i}', t0=300),
                    example=dict(example_id=f'{requested_arm}_{i}#s1'), plan=plan,
                    pairs=[(dict(recorded, intent_link=dict(t=300)), 20.)], trace='CPU')
    monkeypatch.setattr(breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path:
                        {299: dict(vehicles=rows()), 300: dict(vehicles=rows())})
    write_rows(tmp_path/'WALL_NEGATIVES.jsonl', [neg_row('B'), neg_row('R')])
    off = launch.bank_move_supervision(SimpleNamespace(cfg=cfg, policy=deepcopy(policy), arm=arm))
    assert 'wall_negatives' not in off and off['family_counts'] == {'wall': 3}
    on_cfg = dict(cfg, wall_negative_labels=True)
    on = launch.bank_move_supervision(SimpleNamespace(cfg=on_cfg, policy=deepcopy(policy), arm=arm))
    assert on['wall_negatives']['rows'] == 1 and on['wall_negatives']['arm'] == arm
    assert on['family_counts'] == {'wall': 3, wn.FAMILY_KEY: 2}
    assert on['family_mass'] == {'wall': .5, wn.FAMILY_KEY: .5}
    assert on['contexts'][:3] == off['contexts']
    assert on['examples'] == off['examples']+2


def probe_plan(members, onset=0, hold=30, depth=1., family='wall'):
    """c4/coop-probe ease_plan format (absolute profile + policy reference, no wall_macro)."""
    dur = onset+hold
    off = [[0.]*onset+[-depth]*hold for _ in members]
    base = [[.1]*dur for _ in members]
    return dict(id='g_wall_probe', family=family, members=list(members), duration_s=dur, policy_profile=base,
                profile=[[b+o for b, o in zip(br, orow)] for br, orow in zip(base, off)],
                perturbation_profile=off)


def test_only_exactly_representable_probe_walls_become_negatives():
    def label(plan):
        return wn.example(arm='B', situation_id='s', verification_id='v', plan=plan, seed=2, t0=300,
                          world=dict(seed=986061), frames=frames(), source='probe', advantage=-200.)
    four = label(probe_plan(['c1', 'c2', 'c3', 'c4']))
    assert four is not None and four['move']['members'] == ['c1', 'c2', 'c3', 'c4']
    assert four['move']['hold'] == 30 and four['move']['depth'] == pytest.approx(1.)
    assert label(probe_plan(['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])) is None   # projects to a sub-wall
    assert label(probe_plan(['c1', 'c2', 'c3'], family='easing')) is None     # not a wall plan
