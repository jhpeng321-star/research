"""P1 production path contract tests; real models/controllers, no simulator."""
from copy import deepcopy
import json
import sys
import types

import numpy as np
import pytest
import torch

from . import track2_backend as tb
from .acceleration import Controller
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class


@pytest.fixture(autouse=True)
def cpu_threads(monkeypatch):
    from . import exploration
    # Deployment supplies leverprobe; no imported simulator in these CPU tests.
    monkeypatch.setattr(exploration, 'regime', lambda rows: 'ordinary')
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cfg(tmp_path):
    return dict(track2_backend=True, run_root=str(tmp_path), training_seed=123,
        v2v=True, v2v_dim=48, v2v_intent=True, merge_features=True, host='cpuV',
        layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045,
            clip_ratio=.2, max_grad_norm=.5), trigger_candidates=4, trigger_partner_slots=5,
        decision_interval_s=60, trigger_lr=.0003, layer_epochs=1,
        trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5),
        worlds_by_update={'1': [dict(seed=1), dict(seed=2)]}, practice_starts=False,
        runtime='unused', checkpoint='unused', accel_bounds=[-3, 2.6], gpu_deadline=9999999999,
        workers=2, load_max=40, budget=dict(total=5000000, phases=dict(track2_train=4000000)),
        safety_reference=dict(by_world={}, unmatched_zero_event_benchmark=True, source='CPU fixture'),
        imitation_epochs=5)


def rows(extra=False):
    out = [dict(vid='c1', s_m=-200., v_mps=2., lane=0, is_cav=True, length_m=5.),
           dict(vid='c2', s_m=-205., v_mps=2., lane=1, is_cav=True, length_m=5.),
           dict(vid='h1', s_m=-210., v_mps=0., lane=2, is_cav=False, length_m=5.)]
    if extra:
        out.append(dict(vid='far', s_m=500., v_mps=30., lane=2, is_cav=False, length_m=5.))
    return out


def policy(cfg):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, 'B')


def sample(cfg, now=300):
    p = policy(cfg)
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), now)
    controller.apply(rows(), {r['vid']: 0. for r in rows() if r['is_cav']}, now)
    return p, controller.events[-1]['sample'], controller


def fake_load(monkeypatch):
    def load(cfg, arm, update):
        torch.manual_seed(7)
        base = IntentPolicy()
        return base, torch.optim.Adam(base.parameters(), lr=.0003), torch.Generator().manual_seed(3), {}
    monkeypatch.setattr(tb, '_load_base', load)


def test_default_off_exact_objects():
    base = object()
    assert tb.episode_components({}, base, Controller) == (base, Controller)


def test_live_zero_init_identity_rng_and_local_receipts(cfg):
    p, s, controller = sample(cfg)
    context = s['context']
    expected, _ = p.layer.base.sample(context, torch.Generator().manual_seed(9), False, {})
    assert s['action'] == expected['action']
    assert torch.equal(controller.generator.get_state(), torch.Generator().manual_seed(9).get_state()) is False
    # Same generator advancement, including stochastic trigger draws on a separate stream.
    g = torch.Generator().manual_seed(9)
    p.layer.base.sample(context, g, False, {})
    assert torch.equal(g.get_state(), controller.generator.get_state())
    assert s['track2_trigger']['tables'][0][1] is not None  # real T1A -> T2A adapter
    assert set(context['merge_features']) == {'c1', 'c2'}
    assert all(len(x) == tb.MERGE_DIM for x in context['merge_features'].values())
    assert float(p.layer.score(s)[0].detach()) == pytest.approx(s['logp'])
    assert set(controller.events[-1]['requests']) <= {'c1', 'c2'}
    assert s['track2']['schema'] == tb.SCHEMA


def test_sensor_privacy_and_t_minus_one_reports(cfg):
    p = policy(cfg)
    p.observe(rows(), 300)
    first = deepcopy(p.blocks)
    p.observe(rows(extra=True), 301)
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_report_count')] == 1
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_report_age_s')] == 1
    assert 'far' not in {r['vid'] for r in p.local_states['c1']['rows']}
    assert first['c1'][tb.MERGE_FEATURES.index('partner_reports_missing')] == 1
    p.observe(rows(extra=True), 303)
    assert p.blocks['c1'][tb.MERGE_FEATURES.index('partner_reports_missing')] == 1
    from tools.coopnative.policy import encode
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    a, _ = p.sample(context, torch.Generator().manual_seed(1))
    context['rsu'] = [1e6]*12
    b, _ = p.sample(context, torch.Generator().manual_seed(1))
    assert tb._features(a['context']).tolist() == tb._features(b['context']).tolist()
    assert p.trigger._hidden(a['context']).tolist() == p.trigger._hidden(b['context']).tolist()


def test_real_move_commitment_hold_release_and_no_hdv(cfg):
    p = policy(cfg)
    move = tb.TriggerMove(tb.moves.Move('wall', ('c1', 'c2'), 0, 20, .3))
    assert p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=320)
    assert not p.executor.start(move, ego='c2', observable_cav_ids=['c1', 'c2'], now=300, until=320)
    from tools.coopnative.policy import encode
    context = encode(rows())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    p.last_decision = 300
    p.layer.trigger_gain.data.fill_(.5)
    p.observe(rows(), 301)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=-.3, c2=-.3)
    assert np.asarray(s['action']).ravel().tolist() == pytest.approx([-.3*np.tanh(.5)]*2)
    p.observe(rows(), 320)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=0., c2=0.)
    # Walls release at the observable merge boundary even before expiry.
    p.executor = tb.MoveExecutor(moves_api=tb.MoveAPI())
    p.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=320, until=340)
    near = rows()
    near[0]['s_m'] = 0.
    p.observe(near, 321)
    s, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    assert s['context']['move_offsets'] == dict(c1=0., c2=0.)


def test_spawn_state_and_practice_factory_wrapper(cfg):
    p = policy(cfg)
    job = dict(track2_backend=True, track2_config=cfg, arm='B', track2_state=p.worker_state(), noise_seed=81)
    # Practice supplies a callable controller factory, not a subclassable type.
    out, factory = tb.episode_components(job, p.layer.base,
        lambda *a, **k: controller_class(Controller)(*a, **k))
    controller = factory(out, torch.Generator().manual_seed(2), False)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    json.dumps(s)
    json.dumps(tb._batch_dict(tb._batch(s['track2_trigger'])))
    assert out.now == 300
    assert out.layer.base.state_dict().keys() == p.layer.base.state_dict().keys()
    for key, value in p.layer.base.state_dict().items():
        assert torch.equal(out.layer.base.state_dict()[key], value)
    deepcopy(out)  # bank imitation's transactional reference must be copyable


def test_group_credit_and_dense_updates_freeze_base(cfg, monkeypatch):
    fake_load(monkeypatch)
    hooks = tb.build_arm_runtime(cfg, 'B')
    runtime = hooks['runtime']
    runtime.safety_checked = True  # safety's own paired adapter is tested separately
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    episodes = [dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s, reward=-999999.)])]
    before = deepcopy(p.layer.base.state_dict())
    weights = deepcopy(p.layer.residual.state_dict())
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([], {'pairs': 0}))
    result = hooks['train_update'](episodes,
        credit=[dict(episode_id='ep1', time_s=300, group_advantage=1.)],
        labels=[dict(episode_id='ep1', time_s=300, targets=[1, 1], calibrated_probability=.8)])
    assert result['layer'][0]['accepted']
    assert result['trigger'][0]['cav_count'] == 2
    assert result['dense'][0]['ce'] > 0
    assert any(not torch.equal(weights[k], v) for k, v in p.layer.residual.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in p.layer.base.state_dict().items())
    assert all(v.grad is None and not v.requires_grad for v in p.layer.base.parameters())


@pytest.mark.parametrize('probability', [.79, float('nan'), float('inf'), 1.1])
def test_bad_dense_confidence_refuses(cfg, monkeypatch, probability):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'R')['runtime']
    with pytest.raises(ValueError, match='unfiltered'):
        runtime._imitate_labels([dict(calibrated_probability=probability)], {})


def test_group_credit_required_and_no_positional_join(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    with pytest.raises(ValueError, match='P2 group credit'):
        runtime.train([])
    with pytest.raises(ValueError, match='unmatched'):
        runtime.train([], credit=[dict(episode_id='unknown', time_s=300, group_advantage=1.)])


def test_collect_uses_real_wave_contract_and_refuses_without_approval(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import accel_protocol
    monkeypatch.setattr(accel_protocol, 'guard', lambda w, role: None)
    calls = []
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: calls.append((a, k)) or [])
    with pytest.raises(FileNotFoundError):
        tb.collect_update(cfg, 'B', 1)
    assert not calls
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    tb.collect_update(cfg, 'B', 1)
    args, kw = calls[0]
    assert args[2] == cfg['worlds_by_update']['1']
    assert args[4]['track2_backend'] is True and args[4]['b_replay'] is True
    assert set(args[1]['B'].state_dict()) == set(IntentPolicy().state_dict())
    calls.clear()
    tb.collect_update(cfg, 'A', 1)
    assert 'track2_backend' not in calls[0][0][4]
    assert calls[0][0][4]['b_replay'] is False


def test_bank_reader_is_real_per_seed_effective_and_restored(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import replay_bank as rb
    original = rb.confirmations
    calls = []
    def pairs(*a, **k):
        assert rb.confirmations.bank_rule == 'per_seed_v1'
        calls.append((a, k))
        return [], {'pairs': 0}
    monkeypatch.setattr(rb, 'bank_pairs', pairs)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.bank_pairs()
    assert calls[0][1]['imitation_target_space'] == 'effective'
    assert calls[0][1]['imitation_bank_filter'] == 'cooperative_only'
    assert rb.confirmations is original


def test_save_resume_includes_trainable_weights_and_optimizers(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.policy.layer.trigger_gain.data.fill_(.123)
    target = runtime.save()
    saved = torch.load(target, map_location='cpu', weights_only=False)
    assert saved['charged'] == 960
    with pytest.raises(ValueError, match='already exists'):
        runtime.save()
    restored = tb.build_arm_runtime(dict(cfg, update=2), 'B')['runtime']
    assert restored.policy.layer.trigger_gain.item() == pytest.approx(.123)


@pytest.mark.parametrize('change', [dict(track2_backend=False), dict(coop_reward=True),
    dict(host='cpuS'), dict(device='cuda'), dict(v2v=False), dict(merge_features=False)])
def test_config_refusals_before_loading(cfg, monkeypatch, change):
    monkeypatch.setattr(tb, '_load_base', lambda *a: pytest.fail('loaded before validation'))
    with pytest.raises(ValueError):
        tb.build_arm_runtime(dict(cfg, **change), 'B')


def test_nonempty_effective_bank_trains_layer_only(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    runtime.safety_checked = True
    p = runtime.policy
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(2), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    s = controller.events[-1]['sample']
    bank = deepcopy(s)
    bank['behaviour'] = dict(executed=True, proposal_members=['c1', 'c2'])
    bank['action'] = [[.5], [.5]]
    bank['binding_limits'] = dict(c1=.3, c2=.3)
    before = deepcopy(p.layer.base.state_dict())
    monkeypatch.setattr(runtime, 'bank_pairs', lambda: ([(bank, 1.)], {'pairs': 1}))
    report = runtime.train([dict(id='ep1', arm='B', steps=[dict(time_s=300, sample=s, reward=-100)])],
        credit=[dict(episode_id='ep1', time_s=300, group_advantage=1.)])
    assert report['bank']['imitation_target_space'] == 'effective'
    assert report['bank']['optimizer_steps'] >= 5
    assert report['bank']['effective']['binding_share'] == 1.
    assert report['bank']['kl'] <= .045
    assert all(torch.equal(before[k], v) for k, v in p.layer.base.state_dict().items())


def test_bank_cap_explicit_no_truth_lookup(cfg, monkeypatch):
    from . import effective_imitation as ei
    captured = []
    original = lambda *a, **k: captured.append(k['cap'])
    monkeypatch.setattr(ei, 'attach_limits', original)
    with tb._bank_view(cfg):
        ei.attach_limits()
    assert captured == [1.2]
    assert ei.attach_limits is original


def test_weighted_practice_uses_existing_frozen_plan_and_batch(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, accel_protocol, practice_weighting as pw
    cfg.update(practice_starts=True, practice_weighting=True, practice_fraction=.25,
               practice_weighted_fraction=.5, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .5)
    requests = []
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: requests.append((a, k)) or
                        dict(rows=[dict(example_id='x')]))
    monkeypatch.setattr(ps, 'wave_call', lambda *a: dict(worlds=[dict(seed=5)],
        extras=dict(practice=dict(slot='clean1_warm_u001_p01', t0=400), branch={'test': True})))
    calls = []
    def wave(*a, **k):
        calls.append((a, k))
        if a[3].endswith('_practice'):
            return [dict(id='practice', arm='B', world=dict(seed=5), practice=dict(t0=400),
                         steps=[dict(time_s=400), dict(time_s=401)])]
        return [dict(id='normal', arm='B', world=dict(seed=1), steps=[])]
    monkeypatch.setattr(tb, '_wave', wave)
    episodes = tb.collect_update(cfg, 'B', 1)
    assert requests[0][1]['weighting'] is True
    assert requests[0][1]['fraction'] == .5
    assert calls[0][0][2] == [cfg['worlds_by_update']['1'][0]]
    assert calls[1][1]['used_seeds'] == {1}
    assert calls[1][0][4]['practice_batch'][0]['branch'] == {'test': True}
    assert episodes[1]['steps'] == [dict(time_s=401)]


def test_r_practice_uses_b_realised_count(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, accel_protocol, practice_weighting as pw
    cfg.update(practice_starts=True, practice_fraction=.25, practice_repeat_cap=2)
    (tb._path(cfg['run_root'])/'OWNER_APPROVAL.json').write_text('{"approved":true}')
    monkeypatch.setattr(accel_protocol, 'guard', lambda *a: None)
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .25)
    monkeypatch.setattr(ps, 'realised_count', lambda r, arm, u: 0)
    captured = []
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: captured.append(k['count']) or dict(rows=[]))
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: [])
    tb.collect_update(cfg, 'R', 1)
    assert captured == [0]


def test_practice_safety_uses_pre_update_a_separate_pairs(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import practice_starts as ps, warm_health, accel_storage, practice_weighting as pw
    cfg.update(practice_starts=True, practice_fraction=.25, practice_repeat_cap=2)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    monkeypatch.setattr(pw, 'effective_fraction', lambda c: .25)
    monkeypatch.setattr(ps, 'plan_for_update', lambda *a, **k: dict(rows=[dict(example_id='x')]))
    monkeypatch.setattr(ps, 'a_reference_call', lambda *a: dict(ident='ref_B_pa01', worlds=[dict(seed=1)],
        extras=dict(practice={}, branch={})))
    reference = dict(id='normal-A', world=dict(seed=1))
    monkeypatch.setattr(accel_storage, 'read', lambda p: reference)
    matched = dict(id='ref_B_pa01_A_1', world=dict(seed=1))
    waves = []
    monkeypatch.setattr(tb, '_wave', lambda *a, **k: waves.append(a) or [matched])
    loads = []
    original_load = tb._load_base
    monkeypatch.setattr(tb, '_load_base', lambda *a: loads.append(a) or original_load(*a))
    checks = []
    monkeypatch.setattr(warm_health, 'braking', lambda *a: checks.append(a))
    report = runtime.safety([dict(id='normal-B', world=dict(seed=1)), dict(id='practice-B',
        world=dict(seed=1), practice=dict(slot='clean1_warm_u001_p01', example_id='x'))])
    assert loads[0][1:] == ('A', 0)
    assert checks[0][4] == [reference]
    assert checks[0][5] == {'practice-B': matched}
    assert 'track2_backend' not in waves[0][4]
    assert waves[0][3].endswith('_B_practiceref')
    assert report['new_reference_sim_s'] == 1200


def test_train_refuses_before_paired_safety(cfg, monkeypatch):
    fake_load(monkeypatch)
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    with pytest.raises(ValueError, match='paired safety'):
        runtime.train([])


def test_real_wave_job_serializes_base_and_track2_state(cfg, monkeypatch):
    fake_load(monkeypatch)
    from . import accel_runner
    runtime = tb.build_arm_runtime(cfg, 'B')['runtime']
    monkeypatch.setattr(accel_runner, 'ledger', lambda r: [])
    extra = dict(update=1, v2v=True, v2v_dim=48, b_replay=True, track2_backend=True,
                 track2_config=cfg, track2_state=runtime.policy.worker_state())
    jobs, receipts = accel_runner.wave_jobs(tb._path(cfg['run_root']), cfg, {'B': runtime.base},
        cfg['worlds_by_update']['1'], 'train', 'clean1_warm_u001', 'track2_train', False, extra)
    assert not receipts and len(jobs) == 2
    assert jobs[0]['sim_s'] == 1200
    assert jobs[0]['policy_state']  # real WM base-actor whitelist, not wrapper-prefixed weights
    assert jobs[0]['track2_state']['schema'] == tb.SCHEMA
    assert 'actor.0.weight' in jobs[0]['weights']


def test_reservation_own_cap_prior_charge_phase_and_no_double_charge(cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(flock=lambda *a: None, LOCK_EX=1))
    cfg['budget'] = dict(total=5000000, phases=dict(track2_train=2400))
    tb._reserve(cfg, [dict(id='ep1', sim_s=1200)], 'track2_train')
    with pytest.raises(ValueError, match='duplicate'):
        tb._reserve(cfg, [dict(id='ep1', sim_s=1200)], 'track2_train')
    with pytest.raises(ValueError, match='budget'):
        tb._reserve(cfg, [dict(id='ep2', sim_s=2400)], 'track2_train')
    cfg['prior_charged_sim_s'] = 4998000
    with pytest.raises(ValueError, match='budget'):
        tb._reserve(cfg, [dict(id='ep3', sim_s=1200)], 'track2_train')


def test_real_accel_plant_full_episode_with_fake_physics(cfg, monkeypatch):
    """Run the actual 900-tick worker, V2V, layer, receipts and compression path."""
    from . import accel_plant
    p = policy(cfg)
    calls = []
    leverprobe = types.ModuleType('tools.leverprobe')
    exact = types.ModuleType('tools.leverprobe.backend')
    free = types.ModuleType('tools.coopfree.backend')
    class Backend:
        def __init__(self, path, world, sim_s, checkpoint):
            self.actual = 300
            self.totals = dict(team_delay_vs=0., collisions=0, teleports=0, hard_brake=0, ttc_below_3_vs=0)
        def fingerprint(self):
            return str(self.actual)
        def close(self):
            calls.append('closed')
    exact.Backend = Backend
    exact.setup = lambda *a: calls.append('fake_setup')
    def tick(backend, controller, trace):
        t = backend.actual
        sensed = rows()
        commands = dict(c1=0., c2=0.)
        requests = controller.apply(sensed, commands, t)
        assert set(requests) <= {'c1', 'c2'}
        trace.write(json.dumps(dict(time_s=t, vehicles=sensed, actions=[]))+'\n')
        backend.actual += 1
        return [dict(clamped=False) for v in requests]
    free.tick = tick
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', leverprobe)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', exact)
    monkeypatch.setitem(sys.modules, 'tools.coopfree.backend', free)
    monkeypatch.setattr(accel_plant, 'install_seed_scope', lambda: None)
    monkeypatch.setattr(accel_plant, 'guard', lambda *a: None)
    job = dict(id='fake-full-episode', path=str(tb._path(cfg['run_root'])/'episode'),
        runtime='unused', checkpoint='unused', arm='B', world=dict(seed=1), role='train', sim_s=1200,
        weights=p.layer.base.state_dict(), noise_seed=81, deterministic=False, v2v=True,
        b_replay=True, track2_backend=True, track2_config=cfg, track2_state=p.worker_state())
    result = accel_plant.episode(job)
    assert len(result['steps']) == 900
    assert result['actual_sim_s'] == 1200 and result['charged_sim_s'] == 1200
    assert sum('track2_trigger' in s['sample'] for s in result['steps']) == 15
    assert all('track2' in s['sample'] for s in result['steps'])
    assert set(result['metrics']) == {'600', '900'}
    assert calls == ['fake_setup', 'closed']
    assert (tb._path(job['path'])/'RESULT.json.gz').is_file()
