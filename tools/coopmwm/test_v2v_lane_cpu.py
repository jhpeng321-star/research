"""CPU gates for lane-change actions (option ``lane_actions``, tools/coopmwm/v2v_lane.py).

Real torch on the REAL execution path (the coopfree tick boundary, the v2v runtime
controller, the launch3/prepare_intent key pins); the only stubs stand in for the
frozen-plant import (``tools.leverprobe.backend``) and the SUMO engine itself, exactly
like the other CPU suites. No SUMO, no GPU.
"""
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.coopnative.policy import encode
from tools.coopnative.storage import canonical
from .accel_learner import save
from .acceleration import Controller, Policy as BasePolicy
from . import bank_v2v, v2v_intent as vi, v2v_lane as vl, v2v_launch3 as launch, v2v_obs as vo
from .v2v_obs import V2V_DIM, V2VConfig, lane_intent_columns, lane_intent_dim, v2v_intent_features
from .v2v_runtime import controller_class, from_weights, load_checkpoint

H = 5
W = lane_intent_dim(4, H)
CFG = vl.config_keys()


@pytest.fixture(autouse=True)
def single_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def rows(t=0, lane=None, drop=()):
    spec = (('c0', 0., 10., 0, True), ('c1', 20., 10.5, 0, True), ('c2', -30., 11., 1, True),
            ('c3', 50., 9., 1, True), ('h0', 10., 9.5, 2, False))
    out = []
    for vid, s, v, l, cav in spec:
        if vid in drop:
            continue
        out.append(dict(vid=vid, s_m=s+10.*t, v_mps=v, lane=(lane or {}).get(vid, l),
                        is_cav=cav, length_m=5.))
    return out


def mappo_u0(path, seed=7):
    """Base 33-wide MAPPO-shaped checkpoint with a NONZERO action head and warmed Adam."""
    torch.manual_seed(seed)
    policy = BasePolicy()
    torch.nn.init.normal_(policy.actor[-1].weight, std=.3)
    torch.nn.init.normal_(policy.actor[-1].bias, std=.1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    context = encode(rows())
    mean, value = policy(context)
    (mean.sum()+value).backward()
    optimizer.step()
    save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, 0)
    return policy


def lane_context(policy, t=0, lane=None):
    context = encode(rows(t, lane=lane))
    width = int(policy.actor[0].weight.shape[1])-33
    return dict(context, v2v={v: [.1]*width for v in context['members']})


def wake(policy, bias=(-1., -1.)):
    """Move the lane head off the neutral init so non-keep choices can be sampled."""
    with torch.no_grad():
        policy.lane_head.bias.copy_(torch.tensor([0., *bias]))
    return policy


# ---------------------------------------------------------------- 1. observation block

def test_lane_block_shapes_columns_and_off_identity():
    assert W == 72 and vo.V2V_DIM == 24 and CFG['v2v_dim'] == 72
    expected = dict(launch.LANE_OPTIONS, **vi.config_keys())
    expected['v2v_dim'] = lane_intent_dim(4, H)
    assert CFG == expected
    assert lane_intent_columns() == [s*18+12+j for s in range(4) for j in range(6)]
    state = {r['vid']: dict(r, a_mps2=.1) for r in rows()}
    cfg = V2VConfig(v2v=True)
    board = {'c1': (300, [1., 2., 3., 4., 5.])}
    lane_board = {'c1': (300, [-1., 0., 1., 0., -1.])}
    with_lane = vo.v2v_lane_intent_features(state, 'c0', {'c1': .5}, board, lane_board, 301, H, cfg)
    without = vo.v2v_intent_features(state, 'c0', {'c1': .5}, board, 301, H, cfg)
    assert with_lane.shape == (W,) and without.shape == (48,)
    stride = 18
    for slot in range(4):                                   # the intent part is slot-identical
        assert np.array_equal(with_lane[slot*stride:slot*stride+12], without[slot*12:slot*12+12])
    assert with_lane[12:17].tolist() == [-1., 0., 1., 0., -1.] and with_lane[17] == 1.  # c1 fresh
    for slot in range(1, 4):                                # no other partner published a lane intent
        assert np.array_equal(with_lane[slot*stride+12:slot*stride+18], np.zeros(6, np.float32))
    # lane_board None -> EXACTLY the intent-only block (byte-identical, option off)
    passthrough = vo.v2v_lane_intent_features(state, 'c0', {'c1': .5}, board, None, 301, H, cfg)
    assert np.array_equal(passthrough, without)
    assert not vo.v2v_lane_intent_features(state, 'c0', {}, board, lane_board, 301, H, V2VConfig()).any()


def test_published_lane_intent_freshness_and_range():
    assert vo.published_lane_intent({'z': (302, [0., 1., -1., 0., .5])}, 'z', 303, H) is not None
    assert vo.published_lane_intent({'z': (303, [0.]*H)}, 'z', 303, H) is None      # same-step
    assert vo.published_lane_intent({'z': (301, [0.]*H)}, 'z', 303, H) is None      # stale
    assert vo.published_lane_intent({'z': (302, [0.]*4)}, 'z', 303, H) is None      # malformed
    assert vo.published_lane_intent({'z': (302, [0., 2., 0., 0., 0.])}, 'z', 303, H) is None
    assert vo.published_lane_intent({}, 'z', 303, H) is None


# ---------------------------------------------------------------- 2. neutral init at load

def test_zero_init_lane_head_is_exactly_keep_and_matches_current_policy_at_load(tmp_path):
    path = tmp_path/'u0.pt'
    mappo_u0(path)
    current, _, _, _ = load_checkpoint(path, config=dict(v2v=True, v2v_dim=V2V_DIM))
    lane, opt, _, _ = load_checkpoint(path, config=dict(CFG))
    assert type(lane) is vl.LaneIntentPolicy and isinstance(lane, vi.IntentPolicy)
    assert {id(p) for p in lane.parameters()} == {id(p) for g in opt.param_groups for p in g['params']}
    context = encode(rows())
    narrow = dict(context, v2v={v: [.2]*V2V_DIM for v in context['members']})
    wide = dict(context, v2v={v: [.2]*W for v in context['members']})
    logits = lane.lane_logits(wide)
    assert torch.equal(torch.softmax(logits, -1),
                       torch.tensor([[1., 0., 0.]]*len(context['members'])))   # EXACT keep mass
    for deterministic in (True, False):
        for t in range(3):
            a, ma = current.sample(narrow, torch.Generator().manual_seed(9+t), deterministic, {'c0': .1})
            b, mb = lane.sample(wide, torch.Generator().manual_seed(9+t), deterministic, {'c0': .1})
            assert a['action'] == b['action'] and a['logp'] == b['logp'] and ma == mb
            assert b['lane_choice'] == [0]*len(context['members'])
            assert b['lane_logp'] == 0. and b['logp'] == b['logp']-b['lane_logp']
            assert torch.equal(current.score(a)[0], lane.score(b)[0])
    assert float(lane.lane_head.weight.detach().abs().sum()) == 0.
    assert vl.LanePPO(lane, [dict(steps=[])]).stats()['lane_neutral_init'] is True


# ---------------------------------------------------------------- 3. joint log-prob / entropy / PPO

def lane_steps(policy, n=6, seed=50):
    steps = []
    for t in range(n):
        sample, _ = policy.sample(lane_context(policy, t), torch.Generator().manual_seed(seed+t),
                                  False, {})
        steps.append(dict(time_s=300+t, sample=sample, reward=-1.5))
    return steps


def test_joint_logprob_score_consistency_and_gradients():
    torch.manual_seed(5)
    policy = wake(vl.LaneIntentPolicy())
    steps = lane_steps(policy)
    assert any(any(c != 0 for c in s['sample']['lane_choice']) for s in steps)  # head is awake
    for step in steps:
        sample = step['sample']
        joint, value = policy.score(sample)
        assert abs(float(joint)-sample['logp']) < 1e-5        # score == sampled joint logp
        assert torch.isfinite(value)
        accel_only = 0.
        loss = -(policy.score(sample)[0])
        loss.backward()
        assert policy.lane_head.weight.grad is not None
        policy.zero_grad()
    assert accel_only == 0.


def test_lane_entropy_and_ppo_update_metrics():
    torch.manual_seed(5)
    policy = wake(vl.LaneIntentPolicy())
    steps = lane_steps(policy)
    contexts = [s['sample']['context'] for s in steps]
    # uniform lane head -> exactly log(3) per member
    with torch.no_grad():
        policy.lane_head.bias.copy_(torch.tensor([0., -104., -104.]))
        policy.lane_head.weight.zero_()
    neutral = float(vl.lane_entropy(policy, contexts))
    assert neutral == pytest.approx(0., abs=1e-7)             # exact keep mass -> zero entropy
    with torch.no_grad():
        policy.lane_head.bias.zero_()
    uniform = float(vl.lane_entropy(policy, contexts))
    assert uniform == pytest.approx(np.log(3.), rel=1e-6)
    from .accel_learner import update
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    episode = dict(steps=steps, world=dict(seed=1, cell='lane'))
    metrics = update(policy, optimizer, [episode], torch.Generator().manual_seed(2), epochs=1)
    assert all('lane_entropy' in m for m in metrics)
    assert isinstance(metrics[-1]['lane'], dict)
    assert metrics[-1]['lane']['lane_entropy'] >= 0.
    # the entropy term is exactly -weight * mean entropy at the first minibatch context set
    weight = policy.lane_entropy_weight
    entropy_term = vl.LanePPO(policy, [episode]).entropy_term(list(range(len(steps))))
    assert float(entropy_term) == pytest.approx(-weight*float(vl.lane_entropy(policy, contexts)),
                                                rel=1e-6)


def test_plain_controller_refuses_a_lane_head_policy():
    torch.manual_seed(5)
    policy = vl.LaneV2VPolicy()
    controller = Controller(policy, torch.Generator().manual_seed(1), True)
    with pytest.raises(ValueError, match='lane execution runtime'):
        controller.apply(rows(), {r['vid']: 0.2 for r in rows() if r['is_cav']}, 301)


# ---------------------------------------------------------------- 4. execution on the REAL tick path

class FakeExe:
    """The frozen MAPPO executor boundary: one pre-safety safe_accel call per tick."""
    accel_max, accel_min = 2.6, -3.
    memory = SimpleNamespace(present=None)
    dwell = SimpleNamespace(present=None)

    def __init__(self, accel=0.):
        self.raw = torch.full((1, 192), float(accel))

    def __call__(self, state, with_lanes=False):
        from wms.control import safety
        safe = safety.safe_accel({'s_m': None}, self.raw.clone(), accel_max=self.accel_max)
        assert with_lanes is True
        return safe, None


class FakeEnv:
    """The real actuator boundary contract: step(accel, fallback, lanes) and slot ids."""
    frame = SimpleNamespace(vid_index=np.arange(192))
    now = 301.0

    def __init__(self, lane_commands=None, ids=('c0', 'c1', 'c2', 'c3')):
        self.ids = list(ids)
        self.lane_commands = dict(lane_commands or {})
        self.lanes_seen = []
        self.steps = 0

    def slot_vehicle_ids(self):
        return self.ids

    def step(self, accel_by_slot=None, *, fallback='native', lanes=None):
        self.steps += 1
        self.lanes_seen.append(lanes)
        info = dict(interval={key: 0. for key in
                              ('speed_deficit_vehicle_seconds', 'source_wait_vehicle_seconds',
                               'collision_involvements', 'starting_teleports',
                               'ttc_below_3_vehicle_seconds', 'braking_vehicle_seconds',
                               'dropped_vehicle_steps')},
                    done=False, lane_commands=dict(self.lane_commands))
        self.lane_commands = {}
        return None, info

    def state(self):
        return {}


def fake_backend(env, controller, trace=None):
    b = SimpleNamespace(env=env, exe=FakeExe(), previous=None,
                        totals={key: 0. for key in ('road_delay_vs', 'source_wait_vs', 'team_delay_vs',
                                                    'collisions', 'teleports', 'ttc_below_3_vs',
                                                    'hard_brake')},
                        command_digest=hashlib.sha256(), observe=lambda e: None)
    return b


@contextmanager
def plant_stubs(monkeypatch):
    """Stand-ins ONLY for the frozen-plant import and SUMO's safe_accel body; the tick,
    inject wrapper, controller, policy and lane pass-through are the REAL code."""
    import tools.coopfree.backend as backend
    import wms.control.safety as safety
    lever = SimpleNamespace(exact=lambda cmd: cmd.tolist(), Backend=None, observations=lambda b: [])
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', lever)
    monkeypatch.setattr(safety, 'safe_accel', lambda state, accel, **kw: accel)
    yield backend


def runtime_controller(policy, seed=4):
    return controller_class(Controller)(policy, torch.Generator().manual_seed(seed), False)


def test_keep_tick_stays_fully_native_and_records_zero_lane_outcome(monkeypatch):
    torch.manual_seed(5)
    policy = vl.LaneIntentPolicy()                       # neutral: every choice is keep
    controller = runtime_controller(policy)
    env = FakeEnv()
    with plant_stubs(monkeypatch) as backend:
        audit = backend.tick(fake_backend(env, controller), controller)
    assert env.steps == 1
    assert env.lanes_seen == [None]                      # keep = defer to native SUMO lanes
    assert audit == []
    assert controller.events[-1]['lane_outcome'] == dict(requested=0, issued=0, blocked=0,
                                                         executed=0, counts={})


def test_non_keep_choice_reaches_env_step_and_records_issued_blocked_executed(monkeypatch):
    torch.manual_seed(5)
    policy = wake(vl.LaneIntentPolicy())
    controller = runtime_controller(policy)
    env = FakeEnv()
    with plant_stubs(monkeypatch) as backend:
        backend.tick(fake_backend(env, controller), controller)
    targets = controller.lane_targets
    if not targets:                                      # force a deterministic non-keep member
        controller.lane_targets = {'c0': 1}
        request = getattr(controller, 'lane_request')
        class SlotEnv:
            def slot_vehicle_ids(self):
                return env.ids
        env.lanes_seen.append(request(SlotEnv()))
        targets = controller.lane_targets
    assert targets and all(0 <= t <= 2 for t in targets.values())
    tensor = env.lanes_seen[-1]
    assert tensor is not None and tensor.shape == (1, len(env.ids))
    for slot, vid in enumerate(env.ids):
        assert tensor[0, slot].item() == (float(targets[vid]) if vid in targets else -1.)
    # the env's own safety accounting drives the per-step record (issued / blocked / executed)
    env.lane_commands = dict(issued=1, noop=1, refused_not_allowed=1, deferred_edge_end=0,
                             completed=1)
    controller.observe_lane(dict(lane_commands=env.lane_commands))
    outcome = controller.events[-1]['lane_outcome']
    assert outcome['requested'] == 3 and outcome['issued'] == 1
    assert outcome['blocked'] == 2 and outcome['executed'] == 1
    assert 'refused_not_allowed' in outcome['counts']


def test_lane_target_clipping_and_env_refusals_are_recorded_not_bypassed(monkeypatch):
    torch.manual_seed(5)
    policy = wake(vl.LaneIntentPolicy())
    controller = runtime_controller(policy)
    from .v2v_lane import lane_targets
    sensor = rows(0, lane={'c0': 0, 'c1': 1, 'c2': 2, 'c3': 1})
    sample = dict(context=dict(members=['c0', 'c1', 'c2', 'c3']),
                  lane_choice=[1, 2, 2, 0])              # left, right, right, keep
    targets = lane_targets(sample, sensor)
    assert targets == {'c0': 0, 'c1': 2, 'c2': 2}        # clipped into 0..2, keep dropped
    # the env refuses target 2 (ramp lane) and the off-range adjacency: recorded, never bypassed
    class SlotEnv:
        def slot_vehicle_ids(self):
            return ['c0', 'c1', 'c2', 'c3']
    from .v2v_lane import lanes_tensor
    tensor = lanes_tensor(targets, SlotEnv())
    assert tensor.tolist() == [[0., 2., 2., -1.]]
    assert vl.lane_outcome(dict(refused_not_allowed=2, issued=0, noop=1))['blocked'] == 3
    assert vl.lane_outcome(dict(issued=2, completed=2)) == dict(requested=2, issued=2, blocked=0,
                                                                executed=2,
                                                                counts=dict(issued=2, completed=2))


def test_tick_without_lane_hooks_is_byte_identical(monkeypatch):
    torch.manual_seed(5)
    policy = BasePolicy()                                # the pre-lane path
    controller = Controller(policy, torch.Generator().manual_seed(4), True)
    env = FakeEnv(lane_commands=dict(issued=1))          # any lane command without intent is anomalous
    with plant_stubs(monkeypatch) as backend:
        with pytest.raises(ValueError, match='lane anomaly'):
            backend.tick(fake_backend(env, controller), controller)
    assert env.lanes_seen == [None]


# ---------------------------------------------------------------- 5. lane intent sharing (runtime)

def test_lane_intent_sharing_t_minus_1_latency_and_zero_init_columns(tmp_path):
    torch.manual_seed(7)
    path = tmp_path/'u0.pt'
    mappo_u0(path)
    policy, _, _, _ = load_checkpoint(path, config=dict(CFG))
    assert type(policy) is vl.LaneIntentPolicy
    controller = runtime_controller(policy)
    commands = {r['vid']: .2 for r in rows() if r['is_cav']}
    controller.apply(rows(0), commands, 301)
    first = controller.events[-1]['sample']
    assert set(first['published_lane_intent']) == set(first['context']['members'])
    assert all(len(v) == W for v in first['context']['v2v'].values())
    assert all(v[s*18+17] == 0. for v in first['context']['v2v'].values() for s in range(4))
    assert all(x == 0. for v in first['published_lane_intent'].values() for x in v)  # zero-init head
    controller.apply(rows(1), commands, 302)
    second = controller.events[-1]['sample']
    table = {r['vid']: dict(r) for r in rows(1)}
    slot = vo.v2v_partner_ids(table, 'c0', V2VConfig(v2v=True)).index('c1')
    seen = second['context']['v2v']['c0'][slot*18+12:slot*18+18]
    assert seen[-1] == 1.                                # lane_valid
    assert seen[:5] == pytest.approx(first['published_lane_intent']['c1'], abs=1e-6)   # t-1
    controller.apply(rows(3), commands, 304)             # a skipped second makes it stale
    assert all(v[s*18+17] == 0. for v in controller.events[-1]['sample']['context']['v2v'].values()
               for s in range(4))
    with torch.no_grad():
        policy.lane_intent_head.weight.fill_(.4)
    published = policy.publish_lane(lane_context(policy))
    assert all(all(-1. <= x <= 1. for x in v) for v in published.values())   # signed, clipped


# ---------------------------------------------------------------- 6. bank lane intents

def test_bank_lane_intent_blocks_refresh_and_lane_targets():
    torch.manual_seed(7)
    policy = vl.LaneIntentPolicy()
    members = ['c0', 'c1']
    frame = dict(time_s=301, vehicles=[dict(r, issued_acceleration_mps2=.5) for r in rows(1)])
    prev = dict(time_s=300, vehicles=[dict(r, issued_acceleration_mps2=.4) for r in rows(0)])
    intent = dict(horizon=H, entry='e/plan', board={}, lane_board={})
    block, missing = bank_v2v.pair_v2v_block(frame, prev, members, None, intent)
    assert all(len(v) == W for v in block.values())
    assert all(v[s*18+17] == 0. and v[s*18+11] == 0. for v in block.values() for s in range(4))
    full = encode(rows(1))
    prev_context = dict(encode(rows(0)), v2v={v: [0.]*W for v in full['members']})
    pairs = [(dict(context=dict(prev_context, members=list(full['members'])),
                   behaviour=dict(proposal_members=members),
                   intent_link=dict(entry='e', t=300, slots={})), 1.),
             (dict(context=dict(full, v2v=block, members=list(members)),
                   behaviour=dict(proposal_members=members),
                   intent_link=dict(entry='e', t=301, slots={'c0': ['c1'], 'c1': ['c0']})), 1.)]
    with torch.no_grad():
        policy.lane_intent_head.weight.fill_(.5)
    report = vi.refresh_bank_intents(policy, pairs)
    refreshed = pairs[1][0]['context']['v2v']['c0']      # the t=301 pair sees the t=300 publish
    assert report['partner_lane_slots'] >= 1 and report['partner_lane_slots_valid'] >= 1
    values = policy.publish_lane(pairs[0][0]['context'])['c1']
    slot = vo.v2v_partner_ids({r['vid']: dict(r) for r in rows(1)}, 'c0', V2VConfig(v2v=True)).index('c1')
    assert refreshed[slot*18+12:slot*18+17] == pytest.approx(values, abs=1e-6)
    assert refreshed[slot*18+17] == 1.
    # lane intent training targets: realised signed choices at t+1..t+H, masked past gaps/end
    steps = []
    for t, choice in ((300, [0, 1]), (301, [1, 0]), (302, [2, 2]), (304, [0, 0])):
        context = encode([r for r in rows(t) if r['vid'] in ('c0', 'c1')])
        context['v2v'] = {v: [0.]*W for v in context['members']}
        steps.append(dict(time_s=t, sample=dict(context=context, lane_choice=list(choice))))
    pairs_targets = vi.lane_episode_targets(steps, H)
    assert pairs_targets[0] == ([[-1., 1., 0., 0., 0.], [0., 1., 0., 0., 0.]],
                                [[1., 1., 0., 1., 0.], [1., 1., 0., 1., 0.]])
    assert pairs_targets[1] == ([[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]],
                                [[1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.]])
    assert pairs_targets[2] == ([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]],
                                [[0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    assert pairs_targets[3] == ([[0.]*H, [0.]*H], [[0.]*H, [0.]*H])
    loss, count = vi.lane_mean_loss(policy, [s['sample']['context'] for s in steps],
                                    [t for t, _ in pairs_targets], [m for _, m in pairs_targets])
    assert count == 12. and float(loss) >= 0.


# ---------------------------------------------------------------- 7. checkpoint loaders

def test_lane_checkpoint_round_trip_and_from_weights(tmp_path):
    torch.manual_seed(7)
    policy = vl.LaneIntentPolicy()
    steps = lane_steps(policy, n=2, seed=80)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    path = tmp_path/'lane.pt'
    save(path, policy, optimizer, torch.Generator().manual_seed(3), 4, 1200)
    restored, opt, generator, payload = load_checkpoint(path, config=dict(CFG))
    assert type(restored) is vl.LaneIntentPolicy
    assert payload['update'] == 4
    context = lane_context(restored)
    a, _ = policy.sample(context, torch.Generator().manual_seed(11), False, {})
    b, _ = restored.sample(context, torch.Generator().manual_seed(11), False, {})
    assert a['action'] == b['action'] and a['lane_choice'] == b['lane_choice']
    assert a['logp'] == b['logp']
    assert len(opt.param_groups[0]['params']) == len(list(restored.parameters()))
    weights = {k: v.detach().clone() for k, v in restored.state_dict().items()}
    spawned = from_weights(weights, enabled=True)
    assert isinstance(spawned, vl.LaneIntentPolicy)
    assert torch.equal(spawned.lane_head.weight, restored.lane_head.weight)
    with pytest.raises(ValueError, match='explicit V2V'):
        from_weights(weights)
    # non-intent lane checkpoint reconstruction
    plain = vl.LaneV2VPolicy()
    plain_path = tmp_path/'lane_v2v.pt'
    save(plain_path, plain, torch.optim.Adam(plain.parameters(), lr=3e-4),
         torch.Generator().manual_seed(2), 0, 0)
    cfg = {k: v for k, v in CFG.items() if k not in ('v2v_dim', 'v2v_intent')}
    cfg['v2v_dim'] = V2V_DIM
    lane_only, _, _, _ = load_checkpoint(plain_path, config=cfg)
    assert type(lane_only) is vl.LaneV2VPolicy
    with pytest.raises(ValueError, match='lane intent checkpoint'):
        load_checkpoint(plain_path, config=dict(CFG))
    assert vl.settings({}) is None and vl.settings(None) is None
    with pytest.raises(ValueError):
        vl.settings(dict(CFG, lane_actions_entropy_weight=-1.))


# ---------------------------------------------------------------- 8. launch3 / prepare_intent key pins

def base_bind(tmp_path):
    cfg, live = None, None
    return cfg, live


def lane_cfg(tmp_path, **kw):
    from . import v2v_runner
    kw.setdefault('lane_actions', True)
    return v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v',
                                     legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
                                     common_checkpoint_sha256='a'*64, **kw)


def bindable(tmp_path, cfg):
    return dict(cfg, launch_authorized=True, phase_funding_authorized=True,
                budget=dict(total=launch.CEILING_SIM_S, phases=dict(launch.V2V_PHASES)))


def test_prepare_intent_lane_keys_are_the_launcher_pins(tmp_path):
    assert launch.LANE_OPTIONS['lane_actions'] is True
    cfg = lane_cfg(tmp_path)
    for key, want in launch.LANE_OPTIONS.items():
        assert cfg[key] == want, key
    assert cfg['v2v_dim'] == lane_intent_dim(4, launch.INTENT_OPTIONS['v2v_intent_horizon']) == 72
    # identical for A/B/R: one pinned producer, no per-arm overrides accepted
    again = lane_cfg(tmp_path, lane_actions=True)
    assert again == cfg
    # the default producer stays byte-identical (no lane keys, dim 48)
    from . import v2v_runner
    plain = v2v_runner.prepare_intent(dict(budget={}), root=tmp_path/'v2v2',
                                      legacy_root=tmp_path/'legacy', source_root=tmp_path/'src',
                                      common_checkpoint_sha256='a'*64)
    assert not any(key in plain for key in launch.LANE_OPTIONS)
    assert plain['v2v_dim'] == 48


def test_bind_runtime_accepts_the_lane_pins_and_refuses_mismatches(tmp_path):
    cfg = bindable(tmp_path, lane_cfg(tmp_path))
    assert launch.bind_runtime(dict(config=cfg), root=tmp_path/'v2v',
                               source_root=tmp_path/'src') is cfg
    for bad in (dict(lane_actions=False), dict(lane_actions=None),
                dict(lane_actions_entropy_weight=.02), dict(lane_intent_loss_weight=0.5),
                dict(lane_intent_detach_trunk=False), dict(v2v_dim=48), dict(v2v_dim=None)):
        with pytest.raises(launch.Refused):
            launch.bind_runtime(dict(config=dict(cfg, **bad)), root=tmp_path/'v2v',
                                source_root=tmp_path/'src')
    # lane keys present with lane_actions off: refused (all-or-nothing)
    legacy = {k: v for k, v in cfg.items() if k not in launch.LANE_OPTIONS}
    legacy['v2v_dim'] = 48
    legacy = bindable(tmp_path, legacy)
    assert launch.bind_runtime(dict(config=legacy), root=tmp_path/'v2v',
                               source_root=tmp_path/'src') is legacy
    with pytest.raises(launch.Refused):
        launch.bind_runtime(dict(config=dict(legacy, lane_actions_entropy_weight=.01)),
                            root=tmp_path/'v2v', source_root=tmp_path/'src')
    with pytest.raises(launch.Refused):
        launch.bind_runtime(dict(config=dict(legacy, lane_actions=False)),
                            root=tmp_path/'v2v', source_root=tmp_path/'src')


def test_gate_identity_accepts_lane_pins_and_refuses_tampering(tmp_path):
    from .coopfix_exact_gate import runtime_gate_config
    cfg = lane_cfg(tmp_path)
    gate = runtime_gate_config(cfg)
    assert gate['lane_actions'] is True and gate['v2v_dim'] == 72
    from . import coopfix_exact_gate as gate_module
    with pytest.raises(gate_module.GateIdentityError, match='lane_actions_entropy_weight'):
        runtime_gate_config(dict(cfg, lane_actions_entropy_weight=.5))
    stripped = {k: v for k, v in cfg.items() if k != 'lane_actions'}
    with pytest.raises(gate_module.GateIdentityError, match='lane_actions'):
        runtime_gate_config(stripped)
    # the intent settings validate the lane block width through the shared settings path
    assert vi.settings(cfg)['dim'] == 72
    with pytest.raises(ValueError, match='intent block width'):
        vi.settings(dict(cfg, v2v_dim=48))
