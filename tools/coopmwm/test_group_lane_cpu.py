"""G5 contracts using real policies/controllers/envelope; fake actuator, CPU only."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from . import bank_breadth, feasible_proposals, group_lane as gl, group_lane_bank as gb, moves
from . import track2_backend as tb
from .acceleration import Controller
from .protocol import digest
from .test_track2_backend_cpu import cfg, cpu_threads, policy  # noqa: F401
from .test_bank_perseed_cpu import ps as native_per_seed_reader  # noqa: F401
from .v2v_runtime import controller_class


def geometry():
    return [dict(vid='main', s_m=-30., v_mps=2., lane=1, is_cav=True, length_m=5.),
            dict(vid='ramp', s_m=-10., v_mps=2., lane=2, is_cav=True, length_m=5.),
            dict(vid='hdv', s_m=-70., v_mps=2., lane=0, is_cav=False, length_m=5.)]


def state(ego='main', rows=None, **kwargs):
    return dict(self=ego, rows=geometry() if rows is None else rows,
                group_lane_moves=True, observed_lanes=(0, 1, 2), **kwargs)


def macro(family='yield_lane', onset=2):
    return moves.Move(family, ('main', 'ramp'), onset, 20, .5)


def plan_for(move):
    duration = 60  # full verification window, not shortened to macro length
    offsets = [[moves.offsets(move, t)[v] for t in range(duration)] for v in move.members]
    return dict(id='G5_CPU', family='profile', members=list(move.members), duration_s=duration,
                perturbation_profile=offsets, profile=offsets, **gl.plan_fields(move, duration))


@pytest.mark.parametrize('family', gl.FAMILIES)
def test_exact_macro_roundtrip_and_independent_command_timeline(family):
    m = macro(family)
    assert moves.Move.from_dict(m.to_dict()) == m
    assert moves.from_plan(plan_for(m)) == m
    for t in range(60):
        expected = (-.5 if family == 'open_gap_merge' and 2 <= t < 22 else 0.)
        assert moves.offsets(m, t) == dict(main=expected, ramp=0.)
        lane = ({'main': 0} if family == 'yield_lane' and t == 2 else
                {'ramp': 1} if family == 'open_gap_merge' and t == 3 else {})
        assert gl.targets(m, t) == lane
    assert sum(sum(row) for row in plan_for(m)['lane_actions']) == 1


@pytest.mark.parametrize('change', ['lane', 'offset', 'role', 'duration', 'metadata'])
def test_bank_projection_never_discards_or_retimes_lane_effects(change):
    plan = plan_for(macro())
    if change == 'lane':
        plan['lane_actions'][0][2], plan['lane_actions'][0][3] = 0, 1
    elif change == 'offset':
        plan['perturbation_profile'][0][3] = -.1
    elif change == 'role':
        plan['members'].reverse()
    elif change == 'duration':
        plan['duration_s'] = 20
    else:
        plan.pop('g5_move')
    with pytest.raises(ValueError):
        moves.from_plan(plan)


def test_g5_default_off_preserves_vocab_slate_and_worker_shapes(cfg):
    original = policy(cfg)
    explicit_off = policy(dict(cfg, group_lane_moves=False))
    assert gl.families(cfg) == moves.FAMILIES
    assert gl.families(dict(cfg, group_lane_moves=True)) == moves.ALL_FAMILIES
    assert all(torch.equal(v, explicit_off.worker_state()['trigger'][k])
               for k, v in original.worker_state()['trigger'].items())
    base_state = state()
    base_state.pop('group_lane_moves')
    assert not set(m.family for m in moves.candidates(base_state, np.random.default_rng(3), 30)) & set(gl.FAMILIES)
    with pytest.raises(ValueError, match='bool'):
        gl.enabled(dict(group_lane_moves=1))
    # A returns the identical policy/controller even when the common flag is on.
    base = original.layer.base
    assert tb.episode_components(dict(track2_backend=True, track2_config=dict(cfg, group_lane_moves=True),
        arm='A'), base, Controller) == (base, Controller)


@pytest.mark.parametrize('ego', ['main', 'ramp'])
def test_observable_proposals_balanced_roles_and_both_arm_identity(cfg, ego):
    slate = moves.candidates(state(ego), np.random.default_rng(4), 60)
    counts = Counter(m.family for m in slate)
    assert set(gl.FAMILIES) <= set(counts)
    assert max(counts.values())-min(counts.values()) <= 1
    assert all(m.members == ('main', 'ramp') for m in slate if m.family in gl.FAMILIES)
    cfg['group_lane_moves'] = True
    a = policy(cfg)
    b = policy(cfg); b.arm = 'R'
    for p in (a, b):
        p.observe(geometry(), 300)
    from tools.coopnative.policy import encode
    context = encode(geometry())
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sa, _ = a.sample(context, torch.Generator().manual_seed(3), False)
    sb, _ = b.sample(context, torch.Generator().manual_seed(3), False)
    assert sa['track2_trigger'] == sb['track2_trigger']
    assert sa['action'] == sb['action']
    batch = tb._batch(sa['track2_trigger'])
    assert torch.allclose(a.trigger.evaluate(batch)[0], batch.logp, atol=1e-6)


@pytest.mark.parametrize('obstruction', ['coverage', 'hdv_overlap', 'rear_closing', 'not_cav', 'edge', 'nan'])
def test_yield_unavailable_when_target_gap_is_unobservable_or_unsafe(obstruction):
    s = state()
    if obstruction == 'coverage':
        s['observed_lanes'] = (1, 2)
    elif obstruction == 'hdv_overlap':
        s['rows'][2]['s_m'] = -30.
    elif obstruction == 'rear_closing':
        s['rows'][2]['s_m'], s['rows'][2]['v_mps'] = -45., 20.
    elif obstruction == 'not_cav':
        s['rows'][0]['is_cav'] = False
    elif obstruction == 'edge':
        s['rows'][0]['s_m'] = -1.
    else:
        s['rows'][0]['v_mps'] = float('nan')
    result = gl.candidate('yield_lane', s, s['rows'][0], [s['rows'][1]], np.random.default_rng(1))
    assert result is None
    safe, blocked = gl.safe_targets({'main': 0}, {'main': s}, {'main'})
    assert safe == {} and 'main' in blocked


def test_owned_cavs_are_gap_obstacles_and_never_group_members(cfg):
    cfg['group_lane_moves'] = True
    p = policy(cfg)
    p.observe(geometry(), 300)
    no_partner = dict(p.local_states['main'], acquirable_cav_ids=['main'])
    slate = feasible_proposals.candidates(no_partner, np.random.default_rng(5), 30)
    assert all(m.members == ('main',) for m in slate)
    blocked = geometry()
    # An unacquirable CAV remains a sensed obstacle in the target lane.
    blocked.append(dict(vid='owned', s_m=-30., v_mps=2., lane=0, is_cav=True, length_m=5.))
    s = state(rows=blocked, acquirable_cav_ids=['main', 'ramp'])
    slate = feasible_proposals.candidates(s, np.random.default_rng(5), 30)
    assert 'yield_lane' not in {m.family for m in slate}
    assert all('owned' not in m.members for m in slate)


@pytest.mark.parametrize('family', gl.FAMILIES)
@pytest.mark.parametrize('arm', ['B', 'R'])
def test_real_controller_macro_requests_and_current_safety_recheck(cfg, family, arm):
    cfg['group_lane_moves'] = True
    p = policy(cfg); p.arm = arm
    c = controller_class(Controller)(p, torch.Generator().manual_seed(9), True)
    move = macro(family)
    p.executor.start(tb.TriggerMove(move), ego='main', observable_cav_ids=['main', 'ramp'],
                     now=300, until=322)
    p.last_decision = 300
    attempt = 302 if family == 'yield_lane' else 303
    observed = geometry()
    if family == 'open_gap_merge':
        # Inside the real merge zone (not variety's approach-zone label).
        observed[0]['s_m'], observed[1]['s_m'] = -30., 10.
    for now in (300, attempt):
        p.observe(observed, now)
        c.apply(observed, dict(main=0., ramp=0.), now)
        requested = c.lane_request(SimpleNamespace(slot_vehicle_ids=lambda: ['main', 'ramp', 'hdv']))
        if now == attempt:
            expected = [0., -1., -1.] if family == 'yield_lane' else [-1., 1., -1.]
            assert requested.tolist() == [expected]
        else:
            assert requested is None
    sample = c.events[-1]['sample']
    assert sample['context']['move_offsets'] == dict(main=-.5 if family == 'open_gap_merge' else 0., ramp=0.)
    c.observe_lane(dict(lane_commands=dict(issued=1, completed=0)))
    assert c.events[-1]['g5_lane_outcome']['issued'] == 1
    assert c.events[-1]['g5_lane_outcome']['executed'] == 0
    # Safety changed after the macro was selected: replay is rechecked, not forced.
    observed.append(dict(vid='new_hdv', s_m=observed[0 if family == 'yield_lane' else 1]['s_m'],
                         v_mps=2., lane=0 if family == 'yield_lane' else 1, is_cav=False, length_m=5.))
    c.replay = {str(attempt): sample}
    p.observe(observed, attempt)
    c.apply(observed, dict(main=0., ramp=0.), attempt)
    assert c.lane_request(SimpleNamespace(slot_vehicle_ids=lambda: ['main', 'ramp'])) is None
    assert c.events[-1]['sample']['g5_lane_gate']['blocked']


def test_full_command_expiry_and_counterfactual_release(cfg):
    cfg['group_lane_moves'] = True
    p = policy(cfg)
    move = macro('open_gap_merge', onset=0)
    p.last_decision = 300
    p.executor.start(tb.TriggerMove(move), ego='main', observable_cav_ids=['main', 'ramp'], now=300, until=320)
    c = controller_class(Controller)(p, torch.Generator().manual_seed(9), True)
    for now in range(300, 321):
        obs = geometry(); obs[1]['s_m'] = 10.
        p.observe(obs, now)
        c.apply(obs, dict(main=0., ramp=0.), now)
        assert c.events[-1]['sample']['context']['move_offsets']['main'] == (-.5 if now < 320 else 0.)
    assert not p.executor.active
    sample = dict(g5_lane_requests={'main': 0}, context=dict(members=[]))
    c.events = [dict(sample=sample)]
    c.branch = dict(time_s=300, until=330, group=['main', 'ramp'], revert=['main'])
    assert gl.controller_targets(c, p, geometry(), 302) == {}
    c.branch['revert'] = []
    assert gl.controller_targets(c, p, geometry(), 330) == {}
    assert c.events[-1]['sample']['g5_lane_requests'] == {}


def test_practice_plan_lane_request_supersedes_sample_and_checks_live_safety(cfg):
    cfg['group_lane_moves'] = True
    p = policy(cfg); p.observe(geometry(), 302)
    c = controller_class(Controller)(p, torch.Generator(), True)
    c.events = [dict(sample=dict(g5_lane_requests={'ramp': 1}))]
    c.active_plans = [dict(plan=plan_for(macro()), start_s=300)]
    assert gl.controller_targets(c, p, geometry(), 302) == {'main': 0}
    assert gl.controller_targets(c, p, geometry(), 303) == {}


def test_native_verification_base_policy_executes_bound_lane_plan_locally(cfg):
    p = policy(cfg).layer.base
    c = controller_class(Controller)(p, torch.Generator(), True)
    c.active_plans = [dict(plan=plan_for(macro()), start_s=300)]
    c.apply(geometry(), dict(main=0., ramp=0.), 302)
    assert c.g5_lane_targets == {'main': 0}
    plain = Controller(p, torch.Generator(), True, replay={'302': c.events[-1]['sample']})
    plain.active_plans = deepcopy(c.active_plans)
    with pytest.raises(ValueError, match='single-writer'):
        plain.apply(geometry(), dict(main=0., ramp=0.), 302)
    # A native verifier must also execute recorded G5 prefix samples without a
    # live trigger/layer, otherwise its CRN prefix would silently diverge.
    replayed = controller_class(Controller)(p, torch.Generator(), True,
        replay={'302': c.events[-1]['sample']})
    replayed.apply(geometry(), dict(main=0., ramp=0.), 302)
    assert replayed.g5_lane_targets == {'main': 0}


@pytest.mark.parametrize('family', gl.FAMILIES)
@pytest.mark.parametrize('defect', ['none', 'blocked', 'wrong_actor', 'no_transition', 'wrong_request'])
def test_mined_lane_effect_requires_actual_actor_transition_and_actuator_receipt(family, defect):
    move = macro(family)
    elapsed = 2 if family == 'yield_lane' else 3
    expected = gl.targets(move, elapsed)
    actor, target = next(iter(expected.items()))
    pre = geometry()
    post = deepcopy(pre)
    next(r for r in post if r['vid'] == actor)['lane'] = target
    outcome = dict(requested=1, issued=1, blocked=0, executed=1)
    if defect == 'blocked':
        outcome.update(issued=0, blocked=1, executed=0)
    elif defect == 'wrong_actor':
        next(r for r in post if r['vid'] == actor)['is_cav'] = False
    elif defect == 'no_transition':
        post = deepcopy(pre)
    elif defect == 'wrong_request':
        expected = {'hdv': target}
    tape = dict(steps=[dict(time_s=300+elapsed, rows=pre, sample=dict(g5_lane_requests=expected),
                            g5_lane_outcome=outcome),
                       dict(time_s=301+elapsed, rows=post)])
    if defect == 'none':
        gl.verify_tape(plan_for(move), tape, 300)
    else:
        with pytest.raises(ValueError, match='actual commanded CAV lane effect'):
            gl.verify_tape(plan_for(move), tape, 300)


@pytest.mark.parametrize('flag', [False, True])
def test_real_tick_reuses_one_env_step_and_records_only_actual_outcome(monkeypatch, flag):
    from tools.coopfree import backend
    exact = ModuleType('tools.leverprobe.backend')
    exact.exact = lambda x: x.tolist()
    # Only the two tiny backend ABI dependencies are faked; tick itself is real.
    parent = ModuleType('tools.leverprobe'); parent.backend = exact
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', parent)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', exact)
    monkeypatch.setattr(backend, 'observations', lambda b: geometry())
    @contextmanager
    def inject(*args):
        yield
    monkeypatch.setattr(backend, 'inject', inject)
    class FakeEnv:
        now = 300
        frame = SimpleNamespace(vid_index=np.array([1, 2, 3]))
        def state(self):
            return {}
        def slot_vehicle_ids(self):
            return ['main', 'ramp', 'hdv']
        def step(self, cmd, lanes):
            calls.append((cmd, lanes))
            return None, dict(done=False, lane_commands=dict(issued=1, completed=0) if flag else {},
                interval=dict(speed_deficit_vehicle_seconds=0., source_wait_vehicle_seconds=0.))
    class Exe:
        memory = SimpleNamespace(present=None)
        dwell = SimpleNamespace(present=None)
        def __call__(self, *args, **kwargs):
            return torch.zeros((1, 3)), None
    outcomes, calls = [], []
    targets = {'main': 0} if flag else {}
    c = SimpleNamespace(lane_request=lambda env: gl.lanes_tensor(targets, env),
                        observe_lane=lambda info: outcomes.append(gl.lane_outcome(info['lane_commands'])))
    b = SimpleNamespace(env=FakeEnv(), exe=Exe(), previous=None, command_digest=hashlib.sha256(),
        totals=dict.fromkeys(('road_delay_vs', 'source_wait_vs', 'team_delay_vs', 'collisions',
                             'teleports', 'ttc_below_3_vs', 'hard_brake'), 0.), observe=lambda env: None)
    backend.tick(b, c)
    assert len(calls) == 1
    assert calls[0][1].tolist() == [[0., -1., -1.]] if flag else calls[0][1] is None
    assert outcomes[0]['issued'] == int(flag) and outcomes[0]['executed'] == 0


def mined_record(root, **kwargs):
    return dict(schema=gb.SCHEMA, arm='B', run_root=str(root), verification_id='verified',
                plan_sha256=digest(plan_for(macro())), **kwargs)


def write_mined(root, records):
    path = root/'MINED_MOVES.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records)+'\n', encoding='utf-8')
    return path


def test_mined_inputs_default_off_and_path_validation(tmp_path):
    write_mined(tmp_path, [dict(unverified_label=macro().to_dict())])
    assert gb.inputs(dict(run_root=str(tmp_path))) == []
    with pytest.raises(ValueError, match='native verification'):
        gb.inputs(dict(run_root=str(tmp_path), group_lane_moves=True))
    row = mined_record('library')
    row['replay_sources'] = dict(B=['guidance/B'], R=['guidance/R'])
    write_mined(tmp_path, [row, row])
    source, = gb.inputs(dict(run_root=str(tmp_path), group_lane_moves=True))
    assert source['run_root'] == str((tmp_path/'library').resolve())
    assert len(source['_g5_mined']) == 1
    assert source['replay_sources']['R'] == [str((tmp_path/'guidance/R').resolve())]
    row['run_root'] = 'sealed/no_read'
    write_mined(tmp_path, [row])
    with pytest.raises(ValueError, match='restricted'):
        gb.inputs(dict(run_root=str(tmp_path), group_lane_moves=True))


def test_mined_record_verified_marker_cannot_bypass_real_native_admission(tmp_path):
    write_mined(tmp_path, [dict(mined_record('unverified_library'), verified=True)])
    with pytest.raises(ValueError, match='no usable native verified example'):
        bank_breadth.visit(dict(run_root=str(tmp_path), group_lane_moves=True, training_seed=1),
                          'B', lambda **kwargs: pytest.fail('unverified callback'))


@pytest.mark.parametrize('wrong_hash', [False, True])
def test_f3_mined_source_uses_native_reader_and_exact_plan_binding(tmp_path, monkeypatch, wrong_hash):
    row = mined_record('library')
    if wrong_hash:
        row['plan_sha256'] = '0'*64
    write_mined(tmp_path, [row])
    calls, seen = [], []
    class NativeReader:
        def bank_pairs(self, root, arm, seed, **kwargs):
            calls.append((root, arm, kwargs))
            if root.name == 'library':
                kwargs['example_consumer'](receipt=dict(verification_id='verified'), example=None,
                    plan=plan_for(macro()), pairs=[({}, 1.)], trace=root/'trace.jsonl')
    @contextmanager
    def view(cfg):
        yield NativeReader()
    monkeypatch.setattr(tb, '_bank_view', view)
    from . import practice_weighting, practice_starts
    monkeypatch.setattr(practice_weighting, 'augmented_examples', lambda *a, **k: ([], ''))
    monkeypatch.setattr(practice_starts, 'examples', lambda *a: [])
    config = dict(run_root=str(tmp_path), group_lane_moves=True, training_seed=1)
    if wrong_hash:
        with pytest.raises(ValueError, match='no usable native verified example'):
            bank_breadth.visit(config, 'B', lambda **kw: seen.append(kw))
        assert not seen
    else:
        bank_breadth.visit(config, 'B', lambda **kw: seen.append(kw))
        assert len(seen) == 1 and moves.from_plan(seen[0]['plan']).family == 'yield_lane'
    assert calls[-1][2]['imitation_bank_filter'] == 'cooperative_only'
    assert calls[-1][2]['imitation_target_space'] == 'effective'


@pytest.mark.parametrize('defect', ['none', 'blocked', 'unbound_tape'])
def test_native_statistical_bank_admits_only_hash_bound_actual_lane_effects(
        tmp_path, native_per_seed_reader, defect):
    from . import replay_bank as rb, exact_grade as xg, bank_statistics
    from .test_bank_perseed_cpu import make, win
    from .test_bank_boot_cpu import T0
    make(tmp_path, 'lane', win(), win(), syn1=win(), syn2=win(), members=['main', 'ramp'])
    root = rb.bank_root(tmp_path, 'B')
    move = macro(onset=0)
    plan = plan_for(move)
    plan.update(id='p', duration_s=20,
        profile=[r[:20] for r in plan['profile']],
        perturbation_profile=[r[:20] for r in plan['perturbation_profile']],
        **gl.plan_fields(move, 20))
    own_receipts = {}
    for kind in ('sumo_verification', 'confirm_replication'):
        tape_path = root/kind/'branches'/'lane'/'cand1'/'RESULT.json'
        tape = json.loads(tape_path.read_text())
        for step in tape['steps']:
            observed = geometry()
            if step['time_s'] > T0:
                observed[0]['lane'] = 0
            step['rows'] = observed
            if step['time_s'] == T0:
                step['sample']['g5_lane_requests'] = {'main': 0}
                step['g5_lane_outcome'] = dict(issued=1, executed=1)
                if defect == 'blocked':
                    step['g5_lane_outcome'] = dict(issued=0, executed=0)
        tape_path.write_text(json.dumps(tape), encoding='utf-8')
        receipt_path = root/kind/'verifications'/'lane.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['chosen_plan'] = plan
        receipt['branches'][1]['result_sha256'] = digest(tape)
        if defect == 'unbound_tape':
            receipt['branches'][1].pop('result_sha256')
        receipt = xg._seal(receipt, 'receipt_sha256')
        receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
        own_receipts[kind] = receipt
    path = root/'confirmations'/'lane.json'
    confirmation = json.loads(path.read_text())
    confirmation['replication']['receipt_sha256'] = own_receipts['confirm_replication']['receipt_sha256']
    path.write_text(json.dumps(xg._seal(confirmation, 'confirmation_sha256')), encoding='utf-8')
    bank_statistics._CACHE.clear()
    if defect == 'none':
        pairs, report = rb.bank_pairs(tmp_path, 'B', 123, imitation_bank_filter='cooperative_only')
        assert pairs and len(pairs) == 20*len(report['entries'])
    else:
        with pytest.raises(ValueError, match='G5'):
            rb.bank_pairs(tmp_path, 'B', 123, imitation_bank_filter='cooperative_only')
