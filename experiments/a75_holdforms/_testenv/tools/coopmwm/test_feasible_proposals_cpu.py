"""F2 synthetic observable geometry and real CPU policy/controller contracts."""
from collections import Counter
from copy import deepcopy
import hashlib
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


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('source', ['snapshot', 'candidate', 'live_controller'])
def test_production_hook_honours_practice_and_controller_ownership(cfg, arm, source):
    from .launch_production import OPTIMIZATION
    from .sumo_verify import verify_controller_class
    from tools.coopnative.policy import encode

    cfg = dict(cfg, **deepcopy(OPTIMIZATION))
    initial = policy(cfg)
    geometry = [row('c0', -195, 0)] + rows() + [row('remote', 10000, 0)]
    commands = {r['vid']: 0. for r in geometry if r['is_cav']}
    plan = dict(id='practice-owned', family='wall', members=['c1', 'c2'],
                duration_s=60, knots=[[-.4]*7, [-.4]*7])
    # Include nonlocal and non-CAV reservations: neither may enter local F2
    # bookkeeping. The zero command still reserves its declared member.
    extra = dict(id='other-owned', family='wall', members=['remote', 'h1'],
                 duration_s=60, knots=[[0.]*7, [0.]*7])
    commitments = [dict(plan=p, start_s=300) for p in (plan, extra)]
    generator = torch.Generator().manual_seed(19)
    context = encode(geometry)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    cached, memory = initial.layer.base.sample(context, torch.Generator().manual_seed(19), True, {})
    snapshot = dict(t0=300, samples={'300': cached}, generator_state=generator.get_state(),
        rng=dict(generator_state_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                 memory_after_t0=memory),
        committed_plans=commitments if source == 'snapshot' else [])
    verify = controller_class(verify_controller_class())

    def practice_factory(p, g, *args, **kwargs):
        return verify(p, g, snapshot, plan if source == 'candidate' else None)

    job = dict(track2_backend=True, track2_config=cfg, arm=arm,
               track2_state=initial.worker_state(), noise_seed=81)
    out, factory = tb.episode_components(job, initial.layer.base, practice_factory)
    force_family(out, 'wall')
    controller = factory(out, generator)
    controller.apply(geometry, commands, 300)  # real snapshot/CRN restoration
    controller.deterministic = True  # keep free egos at none unless wall is feasible
    if source == 'live_controller':
        # Live plans can be added after construction, not just from snapshots.
        controller.active_plans = deepcopy(commitments)
    original = deepcopy(controller.active_plans)
    for now in (301, 359):
        out.last_decision = None  # probe eligibility immediately before expiry
        requested = controller.apply(geometry, commands, now)
        record = controller.events[-1]['sample']['track2_trigger']
        table = dict(zip(controller.events[-1]['sample']['context']['members'], record['tables']))
        for ego in ('c1', 'c2'):
            assert all(m is None for m in table[ego])
            assert record['accepted'][ego] is False
        assert {v for m in table['c0'] if m is not None for v in m['members']} == {'c0'}
        assert not out.executor.active
        assert {v: requested[v]['requested_raw'] for v in plan['members']} == {'c1': -.4, 'c2': -.4}
        assert controller.active_plans == original  # original expiry is untouched
        assert out.command_reservations['c0'] == {'c1': 360, 'c2': 360}
        assert out.command_reservations['remote'] == ({} if source == 'candidate' else {'remote': 360})
        assert record['proposals']['controller_owned_cav_checks'] == (7 if source != 'candidate' else 6)
        assert record['proposals']['owned_cav_checks'] == record['proposals']['controller_owned_cav_checks']
        assert record['proposals']['infeasible_family_checks'] == (15 if source != 'candidate' else 14)
        assert record['proposals']['proposed_by_family'] == dict(
            ease_off=8 if source != 'candidate' else 16, gap_open=0, wall=0, ramp_assist=0)
        assert record['proposals']['rejected_non_none'] == 0

    out.last_decision = None
    controller.apply(geometry, commands, 360)  # now == start + duration is free
    record = controller.events[-1]['sample']['track2_trigger']
    assert controller.active_plans == []
    assert record['proposals']['controller_owned_cav_checks'] == 0
    assert record['accepted']['c0'] is True
    move = out.executor.active[0]['move']
    assert move.family == 'wall' and 'c0' in move.members
    assert set(move.members) & {'c1', 'c2'}
    tables = dict(zip(controller.events[-1]['sample']['context']['members'], record['tables']))
    acquired = {v for entry in out.executor.active for v in entry['move'].members}
    assert all(v in acquired or any(m is not None for m in tables[v]) for v in ('c1', 'c2'))
    assert record['proposals']['rejected_non_none'] == 0


@pytest.mark.parametrize('arm', ['A', 'B', 'R'])
@pytest.mark.parametrize('path', ['lesson', 'isolated_all', 'paired'])
def test_diagnostic_hooks_refresh_live_controller_ownership(cfg, monkeypatch, tmp_path, arm, path):
    from . import accel_plant, launch_evaluation as evaluation, launch_production as launch
    from . import protocol, rerun_tracks_backend, sumo_verify as sv
    from .policy_synergy import MixedPolicy
    from tools.coopnative.policy import encode

    cfg = dict(cfg, **deepcopy(launch.OPTIMIZATION), evaluation_update=24, warm_history=1)
    initial = policy(cfg)
    weights = deepcopy(initial.layer.base.state_dict())
    if arm != 'A':
        weights['p1.arm'] = torch.tensor(1 if arm == 'B' else 2)
        for name, state in initial.worker_state().items():
            if name in ('layer', 'trigger'):
                weights.update({f'p1.{name}.{k}': v for k, v in state.items()})
    geometry = [row('c0', -195, 0, speed=2.)] + rows() + [row('c3', -190, 0, speed=2.)]
    commands = {r['vid']: 0. for r in geometry if r['is_cav']}
    plan = dict(id='diagnostic-owned', family='wall', members=['c1', 'c2'],
                duration_s=60, knots=[[-.4]*7, [-.4]*7])
    commitments = [dict(plan=plan, start_s=300)]
    generator = torch.Generator().manual_seed(19)
    context = encode(geometry)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    cached, memory = initial.layer.base.sample(context, torch.Generator().manual_seed(19), True, {})
    snapshot = dict(t0=300, samples={'300': cached}, generator_state=generator.get_state(),
        rng=dict(generator_state_sha256=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
                 memory_after_t0=memory), committed_plans=deepcopy(commitments),
        sumo=dict(noise_seed=81), weights=weights)
    if path == 'isolated_all':
        snapshot.update(u3_weights=deepcopy(initial.layer.base.state_dict()),
            policy_synergy=dict(group=['c0', 'c3'], new_members=['c0', 'c3'], background_old=True))
    window = dict(start_s=0 if path == 'paired' else 300, end_s=1200, truncated=False)
    spec = dict(arm=arm, world_id='train-world', world=protocol.worlds('train', 0)[0],
        checkpoint=dict(file_sha256='a'*64), window=window, final_worlds_sha256='b'*64)
    exercised = []

    def exercise(controller, new, *, replay):
        assert isinstance(new, tb.EpisodePolicy) == (arm != 'A')
        if arm != 'A':
            force_family(new, 'wall')  # real slates, masks, serial acquisition and executor
        if replay:
            requested = controller.apply(geometry, commands, 300)
            assert controller.audit['crn_verified_t0'] is True
            replayed = controller.events[-1]['sample']
            expected_actions = deepcopy(cached['action'])
            for v in plan['members']:
                expected_actions[context['members'].index(v)] = [-.4]
            assert replayed['action'] == expected_actions  # inherited authoritative overlay
            assert replayed['own_action'] == cached['own_action']
            assert snapshot['samples']['300'] == cached
            assert 'track2_trigger' not in replayed  # t0 stays the cached base sample
            expected_rng = torch.Generator().set_state(snapshot['generator_state'])
            torch.rand((), generator=expected_rng)
            assert torch.equal(controller.generator.get_state(), expected_rng.get_state())
        else:
            # Paired evaluations may acquire commitments after hook installation.
            controller.active_plans = deepcopy(commitments)
        controller.deterministic = True
        for now in (301, 359, 360):
            if arm != 'A':
                new.last_decision = None  # inspect eligibility on both sides of exact expiry
            if now == 359:
                # Refresh the live ledger, rather than caching the restored list.
                extra = dict(plan=dict(plan, id='late-owned', members=['c3'], duration_s=40,
                                      knots=[[-.2]*7]), start_s=320)
                remote = dict(plan=dict(plan, id='remote-owned', members=['remote'], knots=[[0.]*7]),
                              start_s=300)
                controller.active_plans = deepcopy(commitments) + [extra, remote]
            before = deepcopy(controller.active_plans)
            requested = controller.apply(geometry, commands, now)
            sample = controller.events[-1]['sample']
            if now < 360:
                assert controller.active_plans == before
                assert {v: requested[v]['requested_raw'] for v in plan['members']} == {'c1': -.4, 'c2': -.4}
                if now == 359:
                    assert requested['c3']['requested_raw'] == -.2
            else:
                assert controller.active_plans == []
            if arm == 'A':
                assert 'track2_trigger' not in sample and 'track2' not in sample
                continue
            record = sample['track2_trigger']
            tables = dict(zip(sample['context']['members'], record['tables']))
            stats = record['proposals']
            assert stats['rejected_non_none'] == 0
            if now < 360:
                owned = ['c1', 'c2'] + (['c3'] if now == 359 else [])
                assert new.command_reservations == {v: dict.fromkeys(owned, 360) for v in commands}
                for ego in owned:
                    assert all(m is None for m in tables[ego])
                    assert record['accepted'][ego] is False
                for ego in set(commands)-set(owned):
                    assert {v for m in tables[ego] if m for v in m['members']} == {ego}
                assert not new.executor.active
                assert stats['controller_owned_cav_checks'] == stats['owned_cav_checks'] == 4*len(owned)
                assert stats['infeasible_family_checks'] == 12+len(owned)
                assert stats['proposed_by_family'] == dict(
                    ease_off=8*(4-len(owned)), gap_open=0, wall=0, ramp_assist=0)
            else:
                assert new.command_reservations == {v: {} for v in commands}
                assert stats['controller_owned_cav_checks'] == 0
                assert record['accepted']['c0'] is True
                acquired = set()
                for entry in new.executor.active:
                    members = set(entry['move'].members)
                    assert not members & acquired
                    acquired.update(members)
                first = new.executor.active[0]['move']
                assert first.family == 'wall' and set(first.members) & {'c1', 'c2'}
                # First acquisition is visible to every later ego in the same tick.
                assert all(all(m is None for m in tables[v]) for v in first.members if v != 'c0')
                assert stats['owned_cav_checks'] > 0
        exercised.append(True)

    if path == 'paired':
        monkeypatch.setattr(evaluation.rc, 'sha', lambda path: 'a'*64)
        monkeypatch.setattr(launch, 'checkpoint_weights', lambda *args: weights)
        monkeypatch.setattr(rerun_tracks_backend, 'safety', lambda *args: dict(collisions=0))

        def fake_episode(job, *, controller_hook, policy_loader):
            assert job['sim_s'] == 1200 and job['guided'] is False
            new = policy_loader(job['weights'])  # actual production loader
            controller = controller_class(Controller)(new, generator, True)
            controller_hook(controller, job, {})
            exercise(controller, new, replay=False)
            return dict(totals=dict(team_delay_vs=0.))

        monkeypatch.setattr(accel_plant, 'episode', fake_episode)
        assert evaluation.episode(spec, cfg, tmp_path)['window'] == window
    else:
        def fake_branch(given_spec, given_snapshot, runner):
            assert given_spec['window'] == dict(start_s=300, end_s=1200, truncated=False)
            assert given_snapshot is snapshot
            new = runner.policy_loader(snapshot['weights'])  # actual production loader
            controller = controller_class(sv.verify_controller_class())(new, generator, snapshot, None)
            runner.controller_hook(controller, dict(accel_bounds=cfg['accel_bounds']), {})
            if path == 'isolated_all':
                assert isinstance(controller.policy, MixedPolicy)
                assert controller.policy.new is new
                assert controller.policy.new_members_only == {'c0', 'c3'}
            exercise(controller, new, replay=True)
            return dict(status='cpu-exercised', window=given_spec['window'])

        monkeypatch.setattr(sv, 'run_branch', fake_branch)  # replace physics/grading only
        assert launch.run_branch(spec, snapshot, cfg, tmp_path)['window'] == window
    assert exercised == [True]


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
    assert receipt['controller_owned_cav_checks'] == 0
    assert receipt['owned_cav_checks'] == 4  # same-tick joint acquisition, twice
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
    old = deepcopy(stats)
    for key in ('owned_cav_checks', 'controller_owned_cav_checks'):
        old.pop(key)
    assert fp.aggregate([old, stats])['controller_owned_cav_checks'] is None
    assert fp.aggregate([old, stats])['owned_cav_checks'] is None
