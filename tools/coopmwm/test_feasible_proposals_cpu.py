"""F2 synthetic observable geometry and real CPU policy/controller contracts."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from . import feasible_proposals as fp, moves, track2_backend as tb
from .test_track2_backend_cpu import cfg, cpu_threads, policy, rows  # noqa: F401
from .acceleration import Controller
from .v2v_runtime import controller_class


def row(vid, position, lane, cav=True, speed=1.):
    return dict(vid=vid, s_m=float(position), v_mps=speed, lane=lane,
                is_cav=cav, length_m=5.)


@pytest.mark.parametrize('geometry,families', [
    ([row('ego', -300, 0), row('p', -305, 1)], {'ease_off', 'wall'}),
    ([row('ego', -30, 1), row('p', -35, 2)], {'ease_off', 'gap_open', 'ramp_assist'}),
    ([row('ego', -300, 0)], {'ease_off'}),
])
def test_only_feasible_families_balanced_even_with_remainder(geometry, families):
    for n in (0, 1, 2, 4, 8, 17):
        stats = {}
        slate = fp.candidates(dict(self='ego', rows=geometry, proposal_stats=stats),
                              np.random.default_rng(51), n)
        assert len(slate) == n
        counts = Counter(m.family for m in slate)
        assert set(counts) <= families
        per_family = [counts[f] for f in families]
        assert max(per_family)-min(per_family) <= 1
        assert stats['family_checks'] == (4 if n else 0)
        assert stats['infeasible_family_checks'] == (4-len(families) if n else 0)
    first = {fp.candidates(dict(self='ego', rows=geometry),
             np.random.default_rng(seed), 1)[0].family for seed in range(40)}
    assert first == families  # no fixed preference when action budget < family count


@pytest.mark.parametrize('owned', ['ego', 'p'])
def test_ownership_range_and_hdv_excluded_before_proposing(owned):
    geometry = [row('ego', -30, 1), row('p', -35, 2),
                row('hdv', -32, 2, cav=False), row('far', 10000, 2)]
    available = {r['vid'] for r in geometry}-{owned}
    slate = fp.candidates(dict(self='ego', rows=geometry, acquirable_cav_ids=available),
                          np.random.default_rng(4), 8)
    if owned == 'ego':
        assert slate == []
    else:
        assert {m.family for m in slate} == {'ease_off'}
        assert {v for m in slate for v in m.members} == {'ego'}


def test_wall_discovers_other_lane_beyond_nearest_same_lane_members():
    geometry = [row('ego', -300, 0), row('p1', -301, 0), row('p2', -302, 0),
                row('p3', -303, 0), row('p4', -310, 1)]
    slate = fp.candidates(dict(self='ego', rows=geometry), np.random.default_rng(7), 8)
    walls = [m for m in slate if m.family == 'wall']
    assert len(walls) == 4
    assert all('p4' in m.members and len(m.members) <= moves.MOVE_MAX_MEMBERS for m in walls)
    assert all(m.onset+m.hold <= 190 for m in walls)


def test_unavailable_or_remote_geometry_cannot_change_local_slate():
    geometry = [row('ego', -300, 0), row('p', -305, 1)]
    def propose(extra):
        return fp.candidates(dict(self='ego', rows=geometry+extra,
            acquirable_cav_ids=['ego', 'p']), np.random.default_rng(8), 8)
    assert propose([]) == propose([row('owned', -302, 2), row('outside', 10000, 2)])


def test_ramp_role_uses_full_acquirable_local_mainline_pool():
    geometry = [row('ego', -45, 1), row('ahead', -30, 1), row('ramp', -50, 2)]
    slate = fp.candidates(dict(self='ego', rows=geometry), np.random.default_rng(8), 8)
    assert Counter(m.family for m in slate) == {'ease_off': 4, 'gap_open': 4}


def force_family(p, family):
    # Scores use real candidate descriptors; real trigger masks and samples stay active.
    index = p.trigger.families.index(family)
    def distribution(context, descriptors, mask):
        logits = (descriptors[:, :, index]*100).masked_fill(~mask, -torch.inf)
        return logits, torch.zeros(len(context['members']))
    p.trigger.distribution = distribution


def run_sample(p, geometry, now=300, deterministic=True):
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), deterministic)
    p.observe(geometry, now)
    controller.apply(geometry, {r['vid']: 0. for r in geometry if r['is_cav']}, now)
    return controller.events[-1]['sample']


def test_same_tick_acquisition_masks_later_ego_and_preserves_ppo(cfg):
    p = policy(cfg)
    force_family(p, 'wall')
    s = run_sample(p, rows())
    record = s['track2_trigger']
    assert record['accepted'] == dict(c1=True, c2=False)
    assert all(m is None for m in record['tables'][1])
    assert record['mask'][1] == [True]+[False]*cfg['trigger_candidates']
    assert len(p.executor.active) == 1
    assert set(p.executor.active[0]['move'].members) == {'c1', 'c2'}
    replay = tb._batch(record)
    logp, values, _ = p.trigger.evaluate(replay)
    assert torch.equal(logp, replay.logp)
    assert torch.equal(values, replay.values)
    assert record['proposals']['selected_non_none'] == 1
    assert record['proposals']['rejected_non_none'] == 0
    json.dumps(s)


def test_real_trigger_conditional_logps_replay_in_original_context_order(cfg):
    from tools.coopnative.policy import encode
    p = policy(cfg)
    p.observe(rows(), 300)
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    context['trigger_v2v'] = {v: p.local_intents[v] for v in context['members']}
    results = []
    for reverse in (False, True):
        ordered = deepcopy(context)
        if reverse:
            ordered['members'].reverse()
            ordered['x'].reverse()
        batch, accepted, stats = fp.decide_and_start(p.trigger, ordered,
            obs_states=p.local_states, observable_cavs=p.visible,
            executor=tb.MoveExecutor(moves_api=tb.MoveAPI()), now=300,
            rng=np.random.default_rng(3), generator=torch.Generator().manual_seed(3))
        logp, values, _ = p.trigger.evaluate(batch)
        torch.testing.assert_close(logp, batch.logp)
        torch.testing.assert_close(values, batch.values)
        results.append(({v: None if m is None else m.to_dict()
                         for v, m in batch.selected().items()}, accepted, stats))
    assert results[0] == results[1]


def test_existing_commitment_blocks_every_member_until_exact_expiry(cfg):
    p = policy(cfg)
    force_family(p, 'wall')
    move = tb.TriggerMove(moves.Move('wall', ('c1', 'c2'), 0, 20, .3))
    p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=420)
    first = run_sample(p, rows(), 300)['track2_trigger']
    assert first['actions'] == [0, 0]
    assert first['proposals']['infeasible_family_checks'] == 8
    later = run_sample(p, rows(), 420)['track2_trigger']
    assert later['accepted'] == dict(c1=True, c2=False)
    assert later['actions'][0] > 0


def test_current_control_eligibility_excludes_visible_nonacting_partner(cfg):
    from tools.coopnative.policy import encode
    p = policy(cfg)
    force_family(p, 'wall')
    p.observe(rows(), 300)
    context = encode(rows()[:1])
    context['v2v'] = {'c1': [0.]*48}
    sample, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert all(m is None or m['members'] == ['c1']
               for m in sample['track2_trigger']['tables'][0])


@pytest.mark.parametrize('deterministic', [False, True])
def test_b_r_identical_slates_masks_acquisitions_and_random_streams(cfg, deterministic):
    b = policy(cfg)
    r = tb.EpisodePolicy(deepcopy(b.layer.base), cfg, 'R')
    r.restore_worker(b.worker_state())
    sb, sr = (run_sample(p, rows(), deterministic=deterministic) for p in (b, r))
    assert sb['track2_trigger'] == sr['track2_trigger']
    assert sb['action'] == sr['action']
    assert b.candidate_rng.bit_generator.state == r.candidate_rng.bit_generator.state
    assert torch.equal(b.layer_rng.get_state(), r.layer_rng.get_state())


def test_a_episode_hook_keeps_original_policy_controller_and_rng(cfg):
    base = policy(cfg).layer.base
    before = deepcopy(base.state_dict())
    rng = torch.random.get_rng_state().clone()
    out, controller = tb.episode_components(dict(track2_backend=True,
        track2_config=cfg, arm='A'), base, Controller)
    assert out is base and controller is Controller
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())


def test_update_metrics_log_weighted_rates_and_monitor_retains_them(cfg, tmp_path):
    from . import launch_production as launch, track2_monitor as monitor
    from .test_track2_monitor_cpu import metrics, wm_config
    p = policy(cfg)
    force_family(p, 'wall')
    s = run_sample(p, rows())
    stats = deepcopy(s['track2_trigger']['proposals'])
    second = deepcopy(stats)
    second.update(selected_non_none=3, rejected_non_none=1)
    other = deepcopy(s)
    other['track2_trigger']['proposals'] = second
    episodes = [dict(totals=dict(collisions=0, teleports=0, hard_brake=0), ttc_events=0,
                     steps=[dict(sample=sample, reward=0.)]) for sample in (s, other)]
    result = launch.update_metrics('B', episodes, dict(layer=[], bank_move_ce=0.),
                                  dict(receipt={}), augmented_examples=0)
    receipt = result['move_proposals']
    assert receipt['infeasible_rate'] == .25  # ratio of sums, not mean of rates
    assert receipt['family_checks'] == 16
    assert receipt['proposed_by_family'] == {f: 2*stats['proposed_by_family'][f]
                                             for f in moves.STRUCTURED}
    m = metrics(move_proposals=receipt)
    recorded = monitor.monitor_update(wm_config(tmp_path), 'B', 1, m)
    assert recorded['metrics']['move_proposals'] == receipt
    assert launch.update_metrics('A', episodes, {}, None,
                                augmented_examples=0).get('move_proposals') is None
    assert fp.aggregate([]) is None
    stats.update(selected_non_none=0, rejected_non_none=0)
    assert fp.aggregate([stats])['infeasible_rate'] is None
