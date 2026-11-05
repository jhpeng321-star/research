"""G2 synthetic observables + real CPU trigger/PPO/controller contracts."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from . import coordination_opportunities as co, feasible_proposals as fp, track2_backend as tb
from .test_track2_backend_cpu import cfg, cpu_threads, policy  # noqa: F401
from .test_feasible_proposals_cpu import row, force_family
from .trigger_policy import ppo_update


def enable(cfg):
    return dict(cfg, coordination_opportunities=dict(enabled=True, rule=co.RULE,
                provenance=co.PROVENANCE))


def views(geometry):
    cavs = [r['vid'] for r in geometry if r['is_cav']]
    return ({v: dict(self=v, rows=deepcopy(geometry)) for v in cavs},
            {v: cavs[:] for v in cavs})


def geometry():
    # The lexically earliest CAV would otherwise acquire a singleton/context
    # before the useful mainline role gets a joint opportunity.
    return [row('a_ramp', -35, 2, speed=10.), row('z_main', -30, 1, speed=10.)]


def run(p, rows, now, *, deterministic=True):
    from tools.coopnative.policy import encode
    p.observe(rows, now)
    context = encode(rows)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    return p.sample(context, torch.Generator().manual_seed(99), deterministic, {})[0]


def test_graph_is_local_role_aware_and_excludes_truth_and_unavailable():
    rows = geometry() + [row('hdv', -33, 2, cav=False), row('far', 1000, 2),
                         row('downstream', 90, 2), row('other_lane', -32, 0)]
    states, visible = views(rows)
    edges = co.graph(states, visible, visible, now=300)
    assert [e.members for e in edges] == [('z_main', 'a_ramp')]
    assert edges[0].remaining_s == pytest.approx((75.44+35)/10)
    assert edges[0].to_dict()['roles'] == dict(z_main='mainline_yielder', a_ramp='ramp_merger')
    # Other CAVs' observations cannot affect the initiating ego's edges.
    changed = deepcopy(states)
    changed['a_ramp']['rows'] = [row('a_ramp', 99999, 2)]
    changed['a_ramp']['rows'][0]['hidden_type'] = 'not an input'
    assert co.graph(changed, visible, visible, now=300) == edges
    local_visible = deepcopy(visible)
    local_visible['z_main'].remove('a_ramp')
    assert not co.graph(states, local_visible, visible, now=300)
    assert not co.graph(states, visible, visible, now=300, occupied=['a_ramp'])
    reservations = {'z_main': {'a_ramp': 360}}
    assert not co.graph(states, visible, visible, now=359, command_reservations=reservations)
    assert co.graph(states, visible, visible, now=360, command_reservations=reservations) == edges
    # Nonlocal ownership in a different ego's receipt is not a policy input.
    assert co.graph(states, visible, visible, now=300,
                    command_reservations={'far': {'a_ramp': 360}}) == edges


def test_endpoint_is_precision_consistent_without_zone_padding():
    rows = geometry()
    rows[0]['s_m'] = float(np.float32(tb.moves._ZONE_S1))
    rows[1]['s_m'] = 70.
    states, visible = views(rows)
    assert co.graph(states, visible, visible, now=0)
    assert states['z_main']['rows'][0]['s_m'] != tb.moves._ZONE_S1  # not mutated
    rows[0]['s_m'] = float(np.nextafter(np.float32(tb.moves._ZONE_S1), np.float32(np.inf)))
    states, visible = views(rows)
    assert not co.graph(states, visible, visible, now=0)


def test_stopped_endpoint_pair_uses_immediate_joint_onset():
    state = dict(self='main', rows=[row('main', 70., 1),
        row('ramp', float(np.float32(tb.moves._ZONE_S1)), 2, speed=0.)],
        coordination_partner='ramp', acquirable_cav_ids=['main', 'ramp'])
    states, visible = views(state['rows'])
    assert co.graph(states, visible, visible, now=0)[0].remaining_s == 0.
    slate = fp.candidates(state, np.random.default_rng(7), 8)
    assert all(m.onset == 0 for m in slate if m.family != 'ease_off')


def test_matching_is_disjoint_urgent_and_deterministic():
    rows = [row('r1', -30, 2, speed=20.), row('m1', -29, 1),
            row('r2', -80, 2, speed=1.), row('m2', -79, 1)]
    states, visible = views(rows)
    edges = co.graph(states, visible, visible, now=0)
    assert len(edges) == 4
    pairs = co.disjoint(edges)
    assert [p.members for p in pairs] == [('m1', 'r1'), ('m2', 'r2')]
    assert len({v for p in pairs for v in p.members}) == 2*len(pairs)
    reverse, rev_visible = views(rows[::-1])
    assert co.disjoint(co.graph(reverse, rev_visible, reverse, now=0)) == pairs


@pytest.mark.parametrize('budget', [2, 3, 4, 8, 17])
def test_pair_slate_keeps_single_none_and_library_contract(budget):
    state = dict(self='z_main', rows=geometry(), acquirable_cav_ids=['a_ramp', 'z_main'],
                 coordination_partner='a_ramp', proposal_stats={})
    original = deepcopy(state['rows'])
    slate = fp.candidates(state, np.random.default_rng(123), budget)
    counts = Counter(m.family for m in slate)
    assert len(slate) == budget and counts['ramp_assist'] and counts['ease_off']
    assert max(counts[f] for f in ('ramp_assist', 'ease_off', 'gap_open')) - min(
        counts[f] for f in ('ramp_assist', 'ease_off', 'gap_open')) <= 1
    for move in slate:
        assert set(move.members) <= {'a_ramp', 'z_main'}
        assert 20 <= move.hold <= 90 and .3 <= move.depth <= 1.
        if move.family != 'ease_off':
            assert move.onset < (75.44+35)/10
    assert state['rows'] == original
    broken = dict(state, acquirable_cav_ids=['z_main'])
    with pytest.raises(ValueError, match='free members'):
        fp.candidates(broken, np.random.default_rng(123), budget)


def test_pair_first_acquires_both_before_lexical_singleton_and_replays(cfg):
    p = policy(enable(cfg))
    force_family(p, 'ramp_assist')
    sample = run(p, geometry(), 300)
    record = sample['track2_trigger']
    members = record['context']['members']
    assert record['accepted'] == {'a_ramp': False, 'z_main': True}
    assert len(p.executor.active) == 1
    assert set(p.executor.active[0]['move'].actors) == {'a_ramp', 'z_main'}
    follower = members.index('a_ramp')
    assert record['mask'][follower] == [True]+[False]*cfg['trigger_candidates']
    assert record['proposals']['rejected_non_none'] == 0
    stats = record['proposals']['coordination_opportunities']
    assert stats['decision_order'] == ['z_main', 'a_ramp']
    assert stats['offered_joint'] == 1 and stats['disjoint_pairs'] == 1
    batch = tb._batch(record)
    logp, values, _ = p.trigger.evaluate(batch)
    torch.testing.assert_close(logp, batch.logp)
    torch.testing.assert_close(values, batch.values)
    json.dumps(sample, allow_nan=False)


def test_actual_ppo_uses_exact_conditional_slates_no_regeneration(cfg, monkeypatch):
    p = policy(enable(cfg))
    sample = run(p, geometry(), 300, deterministic=False)
    record = sample['track2_trigger']
    batch = tb._batch(record)
    old = deepcopy(tb._batch_dict(batch))
    logp, values, _ = p.trigger.evaluate(batch)
    torch.testing.assert_close(logp, batch.logp, rtol=0, atol=0)
    torch.testing.assert_close(values, batch.values)
    base_before = deepcopy(p.layer.base.state_dict())
    head_before = deepcopy(p.trigger.trigger_head.state_dict())
    monkeypatch.setattr(fp, 'candidates', lambda *a, **k: pytest.fail('PPO regenerated slate'))
    metrics = ppo_update(p.trigger, p.trigger_optimizer, batch,
        [1., -1.], [1., 1.], **cfg['trigger_ppo'])
    assert np.isfinite(metrics['loss'])
    assert any(not torch.equal(v, head_before[k]) for k, v in p.trigger.trigger_head.state_dict().items())
    assert all(torch.equal(v, base_before[k]) for k, v in p.layer.base.state_dict().items())
    assert tb._batch_dict(batch) == old


def test_none_or_single_remain_real_choices(cfg):
    for family in ('none', 'ease_off'):
        p = policy(enable(cfg))
        force_family(p, family)
        record = run(p, geometry(), 300)['track2_trigger']
        selected = tb._batch(record).selected()['z_main']
        if family == 'none':
            assert selected is None and not p.executor.active
        else:
            assert selected.family == 'ease_off' and selected.members == ('z_main',)
        main_table = record['tables'][record['context']['members'].index('z_main')]
        assert any(m and m['family'] == 'ramp_assist' for m in main_table)


def test_new_approach_event_does_not_shorten_periodic_cadence_or_horizon(cfg):
    p = policy(enable(cfg))
    force_family(p, 'none')
    upstream = geometry() + [row('unrelated', -500, 0)]
    for row_ in upstream[:2]:
        row_['s_m'] -= 200.
    assert 'track2_trigger' in run(p, upstream, 300)
    approaching = geometry() + [row('unrelated', -500, 0)]
    first = run(p, approaching, 310)['track2_trigger']
    stats = first['proposals']['coordination_opportunities']
    assert stats['event_decisions'] == 1
    unrelated = first['context']['members'].index('unrelated')
    assert first['mask'][unrelated] == [True]+[False]*cfg['trigger_candidates']
    assert p.last_decision == 300
    assert 'track2_trigger' not in run(p, approaching, 311)
    # Disappearance/reappearance cannot churn a previously offered role pair.
    assert 'track2_trigger' not in run(p, upstream, 312)
    assert 'track2_trigger' not in run(p, approaching, 313)
    periodic = run(p, approaching, 360)['track2_trigger']
    assert periodic['proposals']['coordination_opportunities']['event_decisions'] == 0
    assert p.last_decision == 360 and p.cfg['decision_interval_s'] == 60
    assert 'horizon' not in co.settings(p.cfg)


@pytest.mark.parametrize('ownership', ['executor', 'controller'])
def test_event_waits_for_authoritative_expiry(cfg, ownership):
    p = policy(enable(cfg))
    force_family(p, 'none')
    p.last_decision = 300
    move = tb.TriggerMove(tb.moves.Move('ease_off', ('a_ramp',), 0, 20, .3))
    if ownership == 'executor':
        p.executor.start(move, ego='a_ramp', observable_cav_ids=['a_ramp'], now=300, until=320)
    for now in (319, 320):
        p.observe(geometry(), now)
        if ownership == 'controller':
            p.command_reservations = {'z_main': {'a_ramp': 320}, 'a_ramp': {'a_ramp': 320}}
        from tools.coopnative.policy import encode
        context = encode(geometry())
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        sample = p.sample(context, torch.Generator().manual_seed(99), True, {})[0]
        assert ('track2_trigger' in sample) == (now == 320)
    assert len(p.approach_events.seen) == 1


def test_offered_joint_rate_aggregates_into_existing_monitor_receipt(cfg):
    p = policy(enable(cfg))
    force_family(p, 'none')
    first = run(p, geometry(), 300)['track2_trigger']['proposals']
    second = run(p, geometry(), 360)['track2_trigger']['proposals']
    out = fp.aggregate([first, second])['coordination_opportunities']
    assert out['decisions'] == 4 and out['offered_joint'] == 4
    # With none selected, ramp's later ordinary periodic slate also offers joint.
    assert out['offered_joint_rate'] == 1.
    assert out['disjoint_pairs'] == 2 and out['event_decisions'] == 0
    legacy = dict(first)
    legacy.pop('coordination_opportunities')
    assert 'coordination_opportunities' not in fp.aggregate([legacy])
    with pytest.raises(ValueError, match='mixed G2'):
        fp.aggregate([first, legacy])


def test_multiple_pair_acquisitions_are_disjoint_and_masks_follow_ownership(cfg):
    p = policy(enable(cfg))
    force_family(p, 'ramp_assist')
    rows = [row('r1', -30, 2, speed=20.), row('m1', -29, 1),
            row('r2', -80, 2, speed=1.), row('m2', -79, 1)]
    sample = run(p, rows, 300)
    active = p.executor.active
    assert len(active) == 2
    assert set(active[0]['move'].members).isdisjoint(active[1]['move'].members)
    record = sample['track2_trigger']
    assert record['proposals']['coordination_opportunities']['decision_order'][:2] == ['m1', 'm2']
    assert record['proposals']['rejected_non_none'] == 0
    for v in ('r1', 'r2'):
        assert record['mask'][record['context']['members'].index(v)] == [True]+[False]*cfg['trigger_candidates']


def test_conditional_slates_are_context_order_invariant(cfg):
    from tools.coopnative.policy import encode
    p = policy(enable(cfg))
    p.observe(geometry(), 300)
    context = encode(geometry())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    context['trigger_v2v'] = p.local_intents
    results = []
    for reverse in (False, True):
        ordered = deepcopy(context)
        if reverse:
            ordered['members'].reverse()
            ordered['x'].reverse()
        batch, accepted, stats = fp.decide_and_start(p.trigger, ordered,
            obs_states=p.local_states, observable_cavs=p.visible,
            executor=tb.MoveExecutor(moves_api=tb.MoveAPI()), now=300,
            rng=np.random.default_rng(4), generator=torch.Generator().manual_seed(4),
            coordination=p.coordination)
        logp, values, _ = p.trigger.evaluate(batch)
        torch.testing.assert_close(logp, batch.logp)
        torch.testing.assert_close(values, batch.values)
        results.append(({v: None if m is None else m.to_dict()
                         for v, m in batch.selected().items()}, accepted, stats))
    assert results[0] == results[1]


def test_monitor_retains_actual_g2_telemetry(cfg, tmp_path):
    from . import launch_production as launch, track2_monitor as monitor
    from .test_track2_monitor_cpu import metrics, wm_config
    p = policy(enable(cfg))
    force_family(p, 'ramp_assist')
    s = run(p, geometry(), 300)
    episodes = [dict(totals=dict(collisions=0, teleports=0, hard_brake=0), ttc_events=0,
                     steps=[dict(sample=s, reward=0.)])]
    result = launch.update_metrics('B', episodes, dict(layer=[], bank_move_ce=0.),
                                   dict(receipt={}), augmented_examples=0)
    proposals = result['move_proposals']
    assert proposals['coordination_opportunities']['offered_joint_rate'] == .5
    recorded = monitor.monitor_update(wm_config(tmp_path), 'B', 1, metrics(move_proposals=proposals))
    assert recorded['metrics']['move_proposals'] == proposals


@pytest.mark.parametrize('invalid', ['duplicate', 'stale'])
def test_supplied_pair_plan_is_revalidated_before_any_decision(cfg, invalid):
    from tools.coopnative.policy import encode
    p = policy(enable(cfg))
    p.observe(geometry(), 300)
    pairs = co.disjoint(co.graph(p.local_states, p.visible, p.visible, now=300))
    reservations = None
    if invalid == 'duplicate':
        pairs = pairs + pairs
    else:
        reservations = {'z_main': {'a_ramp': 360}}
    with pytest.raises(ValueError, match='disjoint currently'):
        fp.decide_and_start(p.trigger, encode(geometry()), obs_states=p.local_states,
            observable_cavs=p.visible, executor=p.executor, now=300,
            rng=p.candidate_rng, generator=p.layer_rng, coordination=p.coordination,
            opportunity_pairs=pairs, command_reservations=reservations)
    assert not p.executor.active


def test_production_controller_hook_obeys_ownership_then_executes_pair(cfg):
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    cfg = enable(cfg)
    initial = policy(cfg)
    for parameter in initial.layer.base.parameters():
        parameter.data.zero_()
    job = dict(track2_backend=True, track2_config=cfg, arm='B',
               track2_state=initial.worker_state(), noise_seed=123)
    p, factory = tb.episode_components(job, initial.layer.base, controller_class(Controller))
    force_family(p, 'ramp_assist')
    controller = factory(p, torch.Generator().manual_seed(9), True)
    p.last_decision = 300
    controller.active_plans = [dict(start_s=300, plan=dict(id='existing', family='wall', members=['a_ramp'],
        duration_s=20, knots=[[-.4]*3], knot_s=10))]
    commands = dict(a_ramp=0., z_main=0.)
    before = controller.apply(geometry(), commands, 319)
    assert 'track2_trigger' not in controller.events[-1]['sample']
    assert before['a_ramp']['requested_raw'] == -.4
    controller.apply(geometry(), commands, 320)
    selected = controller.events[-1]['sample']['track2_trigger']
    assert selected['accepted'] == dict(a_ramp=False, z_main=True)
    assert selected['proposals']['coordination_opportunities']['event_decisions'] == 1
    assert p.last_decision == 300 and controller.active_plans == []
    move = p.executor.active[0]['move']
    now = 320+move.onset
    issued = controller.apply(geometry(), commands, now)
    expected = tb.moves.offsets(move.move, move.onset)
    assert len(move.actors) == 2
    assert controller.events[-1]['sample']['context']['move_offsets'] == expected
    assert {v: issued[v]['requested_raw'] for v in move.members} == pytest.approx(expected)


def test_default_off_and_common_br_are_identical_and_tensor_compatible(cfg):
    implicit, explicit = policy(cfg), policy(dict(cfg, coordination_opportunities=False))
    assert run(implicit, geometry(), 300) == run(explicit, geometry(), 300)
    enabled = enable(cfg)
    b = policy(enabled)
    r = tb.EpisodePolicy(deepcopy(b.layer.base), enabled, 'R')
    r.restore_worker(b.worker_state())
    sb, sr = run(b, geometry(), 300), run(r, geometry(), 300)
    sr['track2']['arm'] = 'B'
    assert sb == sr
    assert implicit.worker_state()['trigger'].keys() == b.worker_state()['trigger'].keys()
    assert all(implicit.worker_state()['trigger'][k].shape == v.shape
               for k, v in b.worker_state()['trigger'].items())
    b.restore_worker(implicit.worker_state())
    assert tb.episode_components({}, implicit.layer.base, object) == (implicit.layer.base, object)
    # Even with G2 configured, A returns the original base/controller objects.
    base = implicit.layer.base
    assert tb.episode_components(dict(track2_backend=True, track2_config=enabled, arm='A'),
                                 base, object) == (base, object)


@pytest.mark.parametrize('change', [True, {}, {'enabled': True},
    {'enabled': True, 'rule': co.RULE, 'provenance': ''},
    {'enabled': True, 'rule': co.RULE, 'provenance': 'x', 'B': {}}])
def test_fail_closed_registration(cfg, change):
    with pytest.raises(ValueError, match='G2'):
        co.settings(dict(cfg, coordination_opportunities=change))


def test_small_slate_fails_closed(cfg):
    with pytest.raises(ValueError, match='two candidates'):
        co.settings(dict(enable(cfg), trigger_candidates=1))
