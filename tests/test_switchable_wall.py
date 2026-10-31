"""CPU-only real wrapper/owner paths; synthetic telemetry, never start SUMO."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import inspect
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tools.c1switch.core import Handover
from tools.c1switch.design import CONTROLLERS
from tools.c1switch.library import SWITCHABLE_CONTROLLERS, make_handover
from tools.c1switch.sumo import Controller
from traffic_shepherd.c4.baselines import WallLaw
from traffic_shepherd.c4.env import CavSumoEnv
from traffic_shepherd.c4.switchable_wall import SwitchableWallController
from traffic_shepherd.config import load_config
from traffic_shepherd.damping import CirclesController, direct_target
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from traffic_shepherd.wall_group import PHASE_HOLD
from traffic_shepherd.wall_group_v22 import MERGE_START_S, WallGroupSessionV22
from wms.control.wall import WallRolloutController
from wms.physics.wall_exec import NumpyWallExecutor, TorchWallExecutor, V21, V22


def telemetry(*, positions=(-1000., -1050., -500., -200.), speeds=(20., 20., 18., 22.)):
    return NS(s=np.array(positions), v=np.array(speeds), lane=np.array([0, 1, 2, 1]),
        lane_carry=np.array([0, 1, 2, 1]), present=np.ones(4, bool),
        is_cav=np.array([True, True, True, False]), junction=np.zeros(4, bool),
        vid_index=np.arange(4))


def reports(frame, now):
    return {str(i): SensorFrame(str(i), now, float(frame.v[i]), 0., 31.29,
        int(frame.lane[i]), distance_to_merge_m=-float(frame.s[i]),
        front=NeighborMeasurement(30., 0.), front_coverage_m=100., rear_coverage_m=100.)
        for i in np.flatnonzero(frame.present & frame.is_cav)}


def fake_env(now=300.):
    """Use CavSumoEnv's REAL control tick, PI, CIRCLES, direct_target and owner."""
    calls = []
    env = NS(now=now, frame=telemetry(), cav_ids={'0', '1', '2'}, k=4,
        config=load_config(), mode='direct', warmup_s=0., fleet=None,
        pi_session=None, _direct_pi=False, _previous_live={'0', '1', '2', '3'},
        _retired_cavs=set(), _lc_original={}, slow_sensor=None, _table={},
        direct_accel_max_mps2=2.6, recorder=None, command_counts=Counter(),
        fallback_counts=Counter(), lane_command_counts=Counter(),
        totals=Counter(), window_totals=Counter())
    def write(kind, *args):
        caller = inspect.currentframe().f_back.f_back
        assert caller.f_code.co_filename.endswith('ownership.py'), 'independent writer'
        calls.append((env.now, kind, *args))
    vehicle = NS(getIDList=lambda: ['0', '1', '2', '3'], getSpeedMode=lambda v: 31,
        slowDown=lambda *a: write('slowDown', *a), setSpeed=lambda *a: write('setSpeed', *a))
    def physics_tick():
        env.now += .5  # CPU fake clock only, no simulator process or plant.
    env.api = NS(vehicle=vehicle, simulation=NS(getArrivedIDList=lambda: []),
                 simulationStep=physics_tick)
    env.builder = NS(vehicle_ids=['0', '1', '2', '3', '4'],
                     pool=NS(slot_of=lambda vid: int(vid) if int(vid) < 4 else None))
    env.sensor = NS(bind=lambda table: None,
                    capture=lambda vid, now_s: reports(env.frame, now_s)[vid])
    env._circles = CirclesController(env.config)
    env.ownership = CommandOwnership(env.api, env.cav_ids, env._circles)
    env._accel_by_vehicle = lambda a: CavSumoEnv._accel_by_vehicle(env, a)
    env._lane_control = lambda p, live: CavSumoEnv._lane_control(env, p, live)
    env._after_physics = lambda *args: Counter()
    env.state = lambda: model_state(env.frame)
    env.tick = lambda a, kw: CavSumoEnv._control_tick(env, a, **kw)
    return env, calls


def model_state(frame):
    return {key: torch.as_tensor(value.copy())[None] for key, value in dict(
        s_m=frame.s, v_mps=frame.v, a_mps2=np.zeros(4), lane=frame.lane,
        present=frame.present, is_cav=frame.is_cav, junction=frame.junction,
        vid_index=frame.vid_index, length_m=np.full(4, 5.)).items()}


class Plant:
    physics = NS(dt_s=.5)
    def physics_accel(self, state):
        return torch.full_like(state['s_m'], 1.2)


def motion(state, acceleration):
    speed = (state['v_mps'] + .5 * acceleration).clamp_min(0)
    position = state['s_m'] + .25 * (state['v_mps'] + speed)
    return torch.stack((position, state['lane'], speed, acceleration), -1)


def fallback(state):
    return motion(state, torch.full_like(state['s_m'], -.25))


@pytest.mark.parametrize('now', [0., 17., 301., 9999.])
@pytest.mark.parametrize('phase', ['empty', 'assembly', 'hold', 'cooldown'])
def test_arbitrary_start_stop_restart(now, phase):
    env, calls = fake_env(now)
    controller = SwitchableWallController()
    controller.start(now_s=now)
    if phase == 'empty':
        env.frame.is_cav[:] = False
    controller.command(env)
    if phase == 'hold':
        env.now += 1
        env.frame.s[:2], env.frame.v[:2] = [-900, -903], [17, 17]
        controller.command(env)
        assert controller.law.executor.state.phase[0] == 1
    if phase == 'cooldown':
        env.now += 60
        controller.command(env)
        assert controller.law.executor is None
    controller.stop(now_s=env.now + 1)
    controller.stop(now_s=env.now + 1)  # idempotent
    assert controller.law is None
    assert controller.command(env) == (None, {'fallback': 'native'})
    assert calls == []  # Lifecycle and request production never write.
    env.now += 100
    env.frame = telemetry()
    controller.start(now_s=env.now)
    accel, kw = controller.command(env)
    assert np.isfinite(accel).sum() == 2 and kw == {'fallback': 'damping'}
    assert controller.law.executor.state.assembly_since[0] == env.now
    assert not any(e.get('reason') == 'planner_abort' for e in controller.events)


def install_fake_learned_outputs(monkeypatch):
    """Only learned logits are stand-ins; command wrappers and execution are real."""
    class Policy:
        def __init__(self):
            self.memory, self.dwell = NS(present=None), NS(present=None)
        def reset(self):
            pass
        def __call__(self, state, with_lanes=False):
            out = torch.full_like(state['s_m'], -.25)
            return (out, None) if with_lanes else out
    monkeypatch.setattr('wms.control.policy.load_policy_fn', lambda *a, **kw: Policy())
    monkeypatch.setitem(sys.modules, 'zero', NS(ZeroResidual=lambda exe: exe))


@pytest.mark.parametrize('peer', CONTROLLERS)
def test_every_peer_to_wall_and_back_uses_real_owner_and_frozen_safety(peer, monkeypatch):
    install_fake_learned_outputs(monkeypatch)
    env, calls = fake_env()
    h = make_handover(dict(mappo='synthetic_logits', ctrl0='synthetic_logits'))
    peer_controller = h.select(peer, env)
    env.tick(*(peer_controller.command(env) if peer_controller else (None, {})))
    wall = h.select('wall', env)
    assert env.pi_session is None and not env._direct_pi
    assert env.ownership.controller is env._circles
    accel, kw = wall.command(env)
    assert np.isfinite(accel).tolist() == [[True, True, False, False]]
    assert h.select('wall', env) is wall  # same selection preserves assembly
    env.tick(accel, kw)
    rows = list(env.ownership._last.values())
    for row in rows:
        if row.owner == 'wall':
            expected = direct_target(reports(env.frame, row.timestamp_s)[row.vehicle_id],
                                      row.wall_request_mps2, env.config, 2.6)
            assert (row.issued_acceleration_mps2, row.issued_target_speed_mps) == expected
    old = wall.exe
    selected = h.select(peer, env)
    assert not old.running and old.law is None
    assert env.pi_session is None and not env._direct_pi
    env.tick(*(selected.command(env) if selected else (None, {})))
    if peer == 'pi':
        assert env._direct_pi and env.pi_session is not None
    for stamp in {r[0] for r in calls}:
        for vid in env.cav_ids:
            assert sum(r[0] == stamp and r[1] == 'slowDown' and r[2] == vid for r in calls) <= 1
    assert all(row[2] != '3' for row in calls)
    # Actual commands only originate in ownership.py; no mode or lane writers exist in this API.
    assert any(row[1] == 'setSpeed' for row in calls)


def test_no_pair_at_selection_and_no_hidden_timer_or_roster_requirement():
    env, _ = fake_env(413.)
    env.frame.is_cav[:2] = False
    wall = Controller('wall', {})
    assert np.isnan(wall.command(env)[0]).all()
    env.now += 1
    env.frame = telemetry()
    assert np.isfinite(wall.command(env)[0]).sum() == 2
    assert wall.exe.law.executor.state.assembly_since[0] == 414.


@pytest.mark.parametrize('change', ['recycled', 'departed', 'not_cav', 'not_live'])
def test_member_identity_or_cav_role_change_aborts_without_commanding_replacement(change):
    env, _ = fake_env()
    wall = SwitchableWallController()
    wall.command(env)
    if change == 'recycled':
        env.frame.vid_index[0] = 4
    elif change == 'departed':
        env.frame.present[0] = False
    elif change == 'not_cav':
        env.frame.is_cav[0] = False
    else:
        env.api.vehicle.getIDList = lambda: ['1', '2', '3']
    env.now += 1
    assert np.isnan(wall.command(env)[0]).all()
    assert wall.events[-1]['reason'] == 'member_frame_missing'


def test_sumowrapper_wm_executor_and_independent_frozen_session_deterministic_parity():
    env, calls = fake_env()
    wrapper = Controller('wall', {})
    wm = WallRolloutController(Plant(), fallback=fallback, start_s=300.)
    session = WallGroupSessionV22('independent', members=frozenset({'0', '1'}), wall_speed_mps=17.)
    script = [((-1000, -1050), (20, 20)), ((-980, -1000), (18, 19)),
              ((-960, -963), (17, 17)), ((-944, -947), (17, 17)),
              ((0, -3), (17, 17)), ((17, 14), (17, 17))]
    for tick, (positions, speeds) in enumerate(script):
        env.now = 300. + tick
        env.frame.s[:2], env.frame.v[:2] = positions, speeds
        accel, _ = wrapper.command(env)
        state = model_state(env.frame)
        out = wm(state)
        independent = session.tick(reports(env.frame, env.now), env.now)
        expected = np.full(4, np.nan)
        for vid, request in independent.items():
            expected[int(vid)] = request
        np.testing.assert_array_equal(accel[0], expected)
        assert torch.equal(wm.mask, torch.as_tensor(np.isfinite(accel)))
        np.testing.assert_array_equal(wm.held[wm.mask].numpy(), accel[np.isfinite(accel)])
        assert torch.equal(out[..., 3][wm.mask], torch.minimum(
            wm.held, Plant().physics_accel(state))[wm.mask])
        assert torch.equal(out[~wm.mask], fallback(state)[~wm.mask])
        next_state = dict(state, s_m=out[..., 0], v_mps=out[..., 2])
        half = wm(next_state)
        assert torch.equal(half[..., 3][wm.mask], torch.minimum(
            wm.held, (wm.target - next_state['v_mps']) / .5)[wm.mask])
    assert calls == []
    assert session.phase != PHASE_HOLD
    assert [e['event'] for e in wrapper.exe.events].count('merge_release') == 1
    assert [e['event'] for e in wm.controllers[0].events].count('merge_release') == 1


def test_rollout_native_safety_braking_hook_lanes_and_snapshot_isolation():
    class BrakingPlant(Plant):
        def physics_accel(self, state):
            return torch.full_like(state['s_m'], -5.)
    state = model_state(telemetry())
    wm = WallRolloutController(BrakingPlant(), fallback=fallback, start_s=413.)
    result = wm(state)
    assert torch.equal(result[..., 1], state['lane'])
    assert (result[..., 3][wm.mask] == -5).all()  # wall cannot override native emergency braking
    branch = deepcopy(wm)
    before = deepcopy(wm.controllers[0].law.executor.state)
    branch.stop(now_s=414.)
    assert wm.running and wm.controllers[0].law.executor is not None
    np.testing.assert_array_equal(wm.controllers[0].law.executor.state.phase, before.phase)
    assert torch.equal(branch(state), fallback(state))
    received = []
    def execute(st, request, target, mask, base):
        received.append((request.clone(), mask.clone()))
        return base
    custom = WallRolloutController(Plant(), fallback=fallback, execute=execute)
    assert torch.equal(custom(state), fallback(state))
    assert len(received) == 1 and received[0][1].sum() == 2


def test_batched_rollouts_and_recycled_halfstep_never_inherit_commands():
    state = model_state(telemetry())
    state = {k: v.repeat(2, 1) for k, v in state.items()}
    state['is_cav'][1] = False
    wm = WallRolloutController(Plant(), fallback=fallback)
    wm(state)
    assert wm.mask.sum(1).tolist() == [2, 0]
    replaced = {k: v.clone() for k, v in state.items()}
    replaced['vid_index'][0, 0] = 99
    out = wm(replaced)
    assert out[0, 0, 3] == -.25
    wm(replaced)
    assert not wm.mask.any()
    assert wm.controllers[0].events[-1]['reason'] == 'member_frame_missing'


@pytest.mark.parametrize('law', [V21, V22])
def test_numpy_torch_frozen_executors_cpu_parity_and_release(law):
    mask = [[True, True, False]]
    kwargs = dict(prior_controller=[[1, 1, 0]], law=law)
    numpy = NumpyWallExecutor(mask, [17.], **kwargs)
    tensor = TorchWallExecutor(mask, [17.], **kwargs)
    for now, p, v in [(17., [1000, 1050, 2000], [20, 20, 20]),
                      (18., [1100, 1103, 2000], [17, 17, 20]),
                      (19., [MERGE_START_S, MERGE_START_S - 3, 3000], [17, 17, 20])]:
        inputs = dict(now_s=now, lane_index=[[0, 1, 2]])
        a, b = numpy.step([p], [v], **inputs), tensor.step([p], [v], **inputs)
        for key in ('owner', 'acceleration_mps2', 'target_speed_mps', 'phase'):
            np.testing.assert_array_equal(a[key], b[key].numpy())
        assert a['events'] == b['events']
    assert numpy.state.phase[0] == (1 if law == V21 else 2)


def test_existing_formation_and_historical_library_are_exactly_preserved():
    env, _ = fake_env()
    old, wall = WallLaw(4), SwitchableWallController()
    expected = old.commands(env.frame, env.builder, now_s=env.now, live_cavs=env.cav_ids)
    actual = wall.commands(env.frame, env.builder, now_s=env.now, live_cavs=env.cav_ids)
    assert expected == actual
    assert old.members == wall.law.members
    np.testing.assert_array_equal(old.executor.speed, wall.law.executor.speed)
    with pytest.raises(ValueError):
        Handover(lambda name: None).select('wall', env)
    assert SWITCHABLE_CONTROLLERS == (*CONTROLLERS, 'wall')


def test_ownership_stale_frames_allowed_speed_and_mode_checks_still_apply():
    env, calls = fake_env()
    wall = SwitchableWallController()
    accel, _ = wall.command(env)
    frames = reports(env.frame, env.now)
    frames['0'] = replace(frames['0'], allowed_speed_mps=18.)
    row = env.ownership.step(frames, now_s=env.now, live_ids=env.api.vehicle.getIDList(),
                              wall_commands={'0': accel[0, 0], '1': accel[0, 1]})
    assert row[0].issued_target_speed_mps <= 18.
    env.now += 1
    stale = env.ownership.step(frames, now_s=env.now, live_ids=env.api.vehicle.getIDList(),
                                wall_commands={'0': accel[0, 0], '1': accel[0, 1]})
    assert all(r.owner == 'native' and r.released_to_native for r in stale)
    env.api.vehicle.getSpeedMode = lambda vid: 0
    before = list(calls)
    with pytest.raises(ValueError, match='speedMode31'):
        env.ownership.step(reports(env.frame, env.now + 1), now_s=env.now + 1,
                            live_ids=env.api.vehicle.getIDList(), wall_commands={'0': -1.})
    assert calls == before


def test_wm_execution_cap_and_invalid_identity_validation():
    state = model_state(telemetry())
    state['cav_exec_cap_mps2'] = torch.full_like(state['s_m'], -7.)
    wm = WallRolloutController(Plant(), fallback=fallback)
    out = wm(state)
    assert (out[..., 3][wm.mask] == -7).all()
    bad = deepcopy(state)
    bad['vid_index'][0, 1] = bad['vid_index'][0, 0]
    with pytest.raises(ValueError, match='unique'):
        WallRolloutController(Plant(), fallback=fallback)(bad)


def test_real_wms_model_rollout_callback_contract_and_frozen_damping_fallback():
    from wms.model.wms import WmSModel
    from wms.train.cav_drive import TimelineControllers
    torch.manual_seed(20261101)
    model = WmSModel().eval()
    state = {key: value.float() if value.is_floating_point() else value
             for key, value in model_state(telemetry()).items()}
    timeline = dict(controller=torch.ones(1, 4, 4, dtype=torch.long),
                    acceleration_mps2=torch.zeros(1, 4, 4),
                    target_speed_mps=torch.zeros(1, 4, 4))
    damping = TimelineControllers(model, timeline)
    wall = WallRolloutController(model, fallback=damping, start_s=301.)
    with torch.inference_mode():
        result = model.rollout(state, n_steps=4, cav_controller=wall)
    assert torch.isfinite(result['positions']).all()
    assert torch.isfinite(result['speeds']).all()
    assert wall.step_index == damping.step_index == 4
    assert wall.controllers[0].law.n_walls == 1
