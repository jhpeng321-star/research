"""G4LD practice OFFER mode and demonstration-onset imitation: CPU contracts (no SUMO/GPU).

Real EpisodePolicy / TriggerPolicy / lane-drop protocol code paths with the N2 CPU fixtures.
"""
from copy import deepcopy

import numpy as np
import pytest

from . import lane_drop as ld, practice_offer as po, practice_starts as ps, track2_backend as tb
from .test_lane_drop_cpu import (INITIATE_CP, Prefer, StubMap, advance, ld_cfg, make, pair_bank,
                                 pair_frames, row, tick)
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

PROV = 'G4LD CPU test (MAIN 2026-11-06 LEARNAUDIT finding 2)'


def offer_cfg(cfg, fraction=.5):
    return dict(ld_cfg(cfg, range_m=200.), practice_offer=po.config_keys(fraction=fraction, provenance=PROV))


def verified_plan():
    move = ld.LaneDropMove('closing_pair', ('m1', 'm2'), 0, 30, 1., 18., 2., False, (2, 1))
    positions = np.array([[-1300.+20*t for t in range(30)], [-1290.+20*t for t in range(30)]])
    return move, ld.plan(move, StubMap(), policy_profile=np.zeros((2, 30)).tolist(),
                         member_speeds=np.full((2, 30), 20.).tolist(), member_positions=positions.tolist())


def scene(t, t0=300):
    return advance([row('m1', -1300, 2), row('m2', -1290, 1), row('h1', -1250, 1, cav=False)], t-t0)


def test_settings_default_off_and_registration_shape(cfg):
    assert po.settings(cfg) is None and po.settings(dict(cfg, practice_offer=False)) is None
    assert po.offered(cfg, 1, 0) is False
    for bad in (dict(po.config_keys(fraction=.5, provenance=PROV), fraction=0.),
                dict(po.config_keys(fraction=.5, provenance=PROV), fraction=1.5),
                dict(po.config_keys(fraction=.5, provenance=PROV), rule='x'),
                dict(po.config_keys(fraction=.5, provenance=PROV), extra=1)):
        with pytest.raises(ValueError):
            po.settings(dict(cfg, practice_offer=bad))


def test_slot_selection_is_outcome_and_arm_blind_and_follows_the_fraction(cfg):
    c = offer_cfg(cfg, .5)
    picks = [po.offered(c, u, i) for u in range(1, 25) for i in range(8)]
    assert 0.35 < sum(picks)/len(picks) < 0.65
    assert all(po.offered(offer_cfg(cfg, 1.), u, i) for u in (1, 7) for i in range(8))
    _, plan = verified_plan()
    rows = [dict(demonstrate=True, mined=True, family='closing_pair', snapshot_lookup=dict(plan=plan),
                 context_id=f'{arm}{i}') for arm, i in (('B', 0), ('B', 1), ('B', 2), ('B', 3))]
    legacy = dict(rows[0], snapshot_lookup=dict(plan=dict(plan, lane_drop_macro=None)))
    marked = po.mark_rows(c, 3, rows+[legacy])
    expected = [po.offered(c, 3, i) for i in range(4)]
    assert [bool(r.get('offer')) for r in marked[:4]] == expected
    assert 'offer' not in marked[4]                     # non-N2 demonstrations stay forced
    assert all('offer' not in r for r in rows)          # frozen practice rows are not mutated


def test_offer_puts_the_verified_initiation_in_slot_one_and_leaves_the_rest_identical(cfg):
    c = offer_cfg(cfg)
    move, plan = verified_plan()
    plain = make(c)
    offered = make(c)
    offered.offer = po.offer_spec(plan, 300)
    for p in (plain, offered):
        p.observe(scene(299), 299, native_commands={'m1': .3, 'm2': .3})
    a = tick(plain, scene(301), 301)
    b = tick(offered, scene(301), 301)
    ta, tb_ = a['track2_trigger'], b['track2_trigger']
    i = tb_['context']['members'].index('m1')
    slot = tb_['tables'][i][1]
    assert slot['family'] == 'lane_drop_initiate' and slot['kind'] == 'closing_pair'
    assert slot['wall_id'] == 'm1@301' and slot['role_lane'] == 2 and slot['hold'] == 29
    assert (slot['target_mps'], slot['gap_dv_mps'], slot['depth']) == (18., 2., 1.)
    assert b['practice_offer']['offered'] == {'m1': slot}
    assert tb_['mask'][i][1] is True
    for j, (x, y) in enumerate(zip(ta['tables'], tb_['tables'])):
        assert (x[2:] if j == i else x) == (y[2:] if j == i else y)
    assert plain.candidate_rng.bit_generator.state == offered.candidate_rng.bit_generator.state
    assert 'practice_offer' not in a and offered.offer is None   # one-shot


def test_offered_initiation_chosen_starts_the_session_and_is_a_ppo_row(cfg):
    c = offer_cfg(cfg)
    _, plan = verified_plan()
    p = make(c)
    # Only the offered (verified, gap_dv 2) initiation carries a gap_dv column: it dominates.
    p.trigger.trigger_head = Prefer(p.trigger, dict(INITIATE_CP, ld_gap_dv_frac=200.))
    p.offer = po.offer_spec(plan, 300)
    p.observe(scene(300), 300, native_commands={'m1': .3, 'm2': .3})
    s = tick(p, scene(301), 301, deterministic=False, seed=3)
    trig = s['track2_trigger']
    i = trig['context']['members'].index('m1')
    assert trig['actions'][i] == 1 and trig['tables'][i][1]['family'] == 'lane_drop_initiate'
    assert trig['accepted']['m1'] is True and 'm1@301' in p.ld_protocol.sessions
    # The decision row is an ordinary stochastic, non-executed trigger sample: PPO-eligible,
    # and its recorded table replays to the recorded log-probabilities.
    assert not s.get('deterministic') and not s.get('behaviour', {}).get('executed')
    logp, _, _ = p.trigger.evaluate(tb._batch(deepcopy(trig)))
    assert logp.tolist() == pytest.approx(trig['logp'])


def test_offer_not_representable_when_a_member_is_absent_or_busy(cfg):
    c = offer_cfg(cfg)
    _, plan = verified_plan()
    p = make(c)
    p.offer = po.offer_spec(plan, 300)
    vehicles = [r for r in scene(301) if r['vid'] != 'm2']
    s = tick(p, vehicles, 301)
    assert s['practice_offer']['offered'] == {} and p.offer is None


def test_snapshot_call_offers_instead_of_forcing(cfg, monkeypatch):
    _, plan = verified_plan()
    snap = dict(t0=305, episode=dict(world=dict(seed=989201)), prefixes={str(t): 'x' for t in range(300, 306)})
    monkeypatch.setattr(ps, 'load_snapshot', lambda root, arm, row: snap)
    monkeypatch.setattr(ps.sv, 'snapshot_sha', lambda value: 'f'*64)
    base = dict(verification_id='v', example_id='v#s1', seed_index=1, mined=True, demonstrate=True,
                mined_plan_sha256=ps.digest(plan), snapshot_lookup=dict(plan=plan),
                context_id='ctx', family='closing_pair', source='verified_breadth')
    forced = ps.wave_call('root', 'B', 'id', 'train', base, 0, {})['extras']['practice']
    offered = ps.wave_call('root', 'B', 'id', 'train', dict(base, offer=True), 0, {})['extras']['practice']
    assert forced['plan'] == plan and 'offer' not in forced and forced['demonstration']
    assert 'plan' not in offered and 'demonstration' not in offered
    assert offered['offer'] == po.offer_spec(plan, 305)


def test_demo_onset_imitation_reaches_every_member_through_the_bank_ce_path(cfg):
    """The forced demo restores the example's own snapshot; its onset context is the one the
    bank CE path (bank_breadth.observable_contexts -> lane_drop.protocol_contexts) labels for
    EVERY member each update: initiator -> lane_drop_initiate at t0, others -> join."""
    from . import bank_breadth as breadth
    frames = pair_frames()
    move, plan, pairs = pair_bank(frames)
    p = make(ld_cfg(cfg, range_m=200.))
    got = [(o.family, o.members[0], t) for _, o, _, _, t in
           breadth.observable_contexts(p, plan, pairs, frames, t0=300)]
    assert got == [('lane_drop_initiate', 'm1', 300), ('lane_drop_join', 'm2', 301)]
    assert {m for _, m, _ in got} == set(move.members)
