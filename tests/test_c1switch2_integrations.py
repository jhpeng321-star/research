"""CPU falsifiers for real WALL six-row integration and peer checkpoint contracts."""
import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.c1switch.design import sha
from tools.c1switch2 import engine, selectors
from tools.c1switch2.design import CONTROLLERS, PROTOCOL
from tools.c1switch2.inference import Predictor, validate_engine
from tools.c1switch2.wall import Wall


def payload(now=300):
    def report(t, vid, lane, s):
        return dict(vehicle_id=vid, timestamp_s=t, s_m=s, lane=lane, v_mps=17., a_mps2=0.,
                    age_s=0., source='cav_self')
    return dict(schema='c1switch_observable/1', history=[dict(now_s=now-60+i*.5,
        self_reports=[report(now-60+i*.5, 'a', 0, -1000), report(now-60+i*.5, 'b', 1, -1010)],
        detections=[], coverage=[], counters=None) for i in range(121)],
        controller_state=dict(incumbent='native', now_s=now, pi={}, commands={}, wall=None))


def attestation():
    return dict(protocol=PROTOCOL, library=list(CONTROLLERS), history_s=60, horizon_s=60, K=8,
        objective='on_road_fleet_delay', evaluation_objective='full_team_delay_fleet_plus_source',
        objective_design_sha256='a'*64, observable_source_admissions=False, entries=False,
        controller_feedback=True, wall_start_stop=True, anonymous_seed=True)


def test_road_objective_is_explicit_and_cannot_claim_full_team_or_drop_binding():
    row = attestation()
    validate_engine(row)
    for key, value in [('objective', 'full_team_delay_fleet_plus_source'), ('entries', True),
                       ('objective_design_sha256', ''), ('observable_source_admissions', True)]:
        with pytest.raises(ValueError):
            validate_engine(dict(row, **{key: value}))


@pytest.mark.parametrize('model_name', ('wm', 'bb'))
def test_six_row_rollout_real_wall_continuation_crn_and_replanning(monkeypatch, model_name):
    from tools.c1switch import observable
    from wms.physics.config import PhysicsConfig
    state = dict(s_m=torch.tensor([[-1000., -1010., 0., 0.]]),
        v_mps=torch.tensor([[17., 17., 0., 0.]]), a_mps2=torch.zeros(1, 4),
        lane=torch.tensor([[0, 1, 0, 0]]), present=torch.tensor([[True, True, False, False]]),
        is_cav=torch.tensor([[True, True, False, False]]), length_m=torch.full((1, 4), 5.))
    def inputs(raw, model):
        tracker = SimpleNamespace(tracks={i: SimpleNamespace(slot=i, cav_id=c, is_cav=True)
                                          for i, c in enumerate(('a', 'b'))})
        return copy.deepcopy(state), {}, tracker
    monkeypatch.setattr(observable, 'rollout_inputs', inputs)
    def motion(row, lanes, acceleration):
        speed = (row['v_mps']+.5*acceleration).clamp_min(0)
        return torch.stack((row['s_m']+.25*(row['v_mps']+speed), lanes, speed, acceleration), -1)
    def fallback(row):
        return motion(row, row['lane'], torch.zeros_like(row['s_m']))
    monkeypatch.setattr(engine.v7.Adapter, 'callback', lambda *args: fallback)
    monkeypatch.setattr(engine, 'circles_callback', lambda *args: fallback)
    streams, batch_shapes = [], []
    class BlackBoxTrajModel:
        training = False
        physics = PhysicsConfig()
        def rollout(self, current, *, n_steps, cav_controller, entries, **kw):
            batch_shapes.append(tuple(current['s_m'].shape))
            for _ in range(n_steps):
                out = cav_controller(current)
                current.update(s_m=out[..., 0], lane=out[..., 1].long(),
                               v_mps=out[..., 2], a_mps2=out[..., 3])
                entries.insert(current)
            return dict(speeds=current['v_mps'])
    class Futures:
        def __init__(self, *args):
            pass
        def plan(self, *args):
            return [[{} for _ in range(8)]]
    instance = engine.Engine.__new__(engine.Engine)
    instance.binding, instance.device, instance.optimized = {'model': model_name}, 'cpu', True
    instance.model = instance.reference = BlackBoxTrajModel()
    instance.cfg = None
    instance.attestation = attestation()
    instance.execution_dependencies = lambda: ('test_native_lanes', 'test_merge', lambda row, accel: accel)
    instance.frozen = SimpleNamespace(native_fn=lambda *a, **k: lambda row: torch.zeros_like(row['s_m']),
        parity_motion=motion, road_end_exit_bound=lambda physics: 500.,
        pair_merge_streams=lambda st, *args: streams.append(args[-1]),
        expand=lambda value, size: value, lane_uniform_draws=lambda *a, **k: None,
        SampledFutures=Futures, FuturesConfig=SimpleNamespace(from_trainer=lambda cfg: None))
    raw = payload(320)
    before = copy.deepcopy(raw)
    fresh = instance.imagine_all(raw, steps=4)
    again = instance.imagine_all(raw, steps=4)
    assert raw == before and fresh == again
    assert all(shape == (6, 4) for shape in batch_shapes)
    assert set(fresh) == set(CONTROLLERS) and all(len(values) == 8 for values in fresh.values())
    assert fresh['wall'][0] > fresh['native'][0]  # Actual wall braking, no picker penalty.
    raw['controller_state'].update(incumbent='wall', wall=dict(schema='c1switch2.wall/1',
        started_s=300., last_s=319., cooldown_s=None, members=['a', 'b'], speed_mps=17.,
        phase=1, assembly_since_s=300., assembled_at_s=301.))
    continued = instance.imagine_all(raw, steps=4)
    assert continued['wall'][0] < fresh['wall'][0]  # Incumbent HOLD isn't restarted.
    # Predictor dispatches ONE six-row batch API, not six independently seeded calls.
    monkeypatch.setattr(instance, 'imagine_all', lambda *a, **k: continued)
    assert Predictor(instance, Wall()).samples(raw) == continued
    assert len(streams) == (24 if model_name == 'wm' else 3)


def test_pending_selectors_fail_without_loading_any_files(monkeypatch):
    monkeypatch.setattr(selectors, 'checked', lambda p: pytest.fail('PENDING opened a path'))
    for cls in (selectors.Fleet, selectors.PerCav):
        with pytest.raises(selectors.DependencyPending):
            cls({'status': 'PENDING'})


def test_runner_model_rpc_plugins_require_both_devices_and_bound_objective(tmp_path):
    from tools.c1switch2 import rpc
    binding = dict(inference_specs={'wm': {'status': 'PENDING'}})
    with pytest.raises(selectors.DependencyPending):
        rpc.wm(binding, tmp_path)
    routes = {}
    for host in ('gpuA', 'gpuR'):
        (tmp_path/host/'wm').mkdir(parents=True)
        routes[host] = host+'/wm'
    spec = dict(status='FROZEN_READY', engine=dict(attestation(), arm='wm'), routes=routes)
    binding.update(inference_specs={'wm': spec}, prediction_objective='on_road_fleet_delay',
                   design_sha256='a'*64)
    assert len(rpc.wm(binding, tmp_path).roots) == 2
    spec['routes'] = {'gpuR': 'gpuR/wm'}
    with pytest.raises(ValueError, match='both'):
        rpc.wm(binding, tmp_path)
    spec['routes'] = routes
    binding['prediction_objective'] = 'full_team_delay_fleet_plus_source'
    with pytest.raises(ValueError, match='objective'):
        rpc.wm(binding, tmp_path)


@pytest.mark.parametrize('variant', ('fleet', 'per_cav'))
def test_real_peer_final169_feature_and_deterministic_inference(tmp_path, variant):
    root = Path(os.environ.get('C1SWITCH2_SELECTOR_SOURCE', 'E:/agent-work/c4-selmappo/tools/selmappo'))
    if not root.is_dir():
        pytest.skip('peer selector source not staged')
    names = ('__init__.py', 'selector.py', 'protocol.py', 'observations.py', 'learning.py')
    spec = dict(status='FROZEN_FINAL', root=str(root), sources={n: sha(root/n) for n in names})
    package = selectors.package(spec, set(names))
    learning = importlib.import_module(package+'.learning')
    model = learning.Selector()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.actor[-1].bias[5] = 1.
    sources = {n[:-3]: spec['sources'][n] for n in names if n != '__init__.py'}
    bindings = dict(integration_sha256='a'*64, design_sha256='b'*64)
    record = dict(model=model.state_dict(), iteration=169, protocol=PROTOCOL,
        library=CONTROLLERS, variant=variant, optimization_seed=1, sources=sources, bindings=bindings)
    path = tmp_path/'synthetic_checkpoint_169.pt'
    torch.save(record, path)
    freeze = tmp_path/'SYNTHETIC_FINAL.json'
    freeze.write_text(json.dumps(dict(iteration=169, checkpoint_sha256=sha(path), sources=sources, bindings=bindings)))
    spec.update(checkpoint=str(path), checkpoint_sha256=sha(path), frozen_receipt=str(freeze),
        frozen_receipt_sha256=sha(freeze), checkpoint_sources=sources, checkpoint_bindings=bindings)
    selector = selectors.Fleet(spec) if variant == 'fleet' else selectors.PerCav(spec)
    for now in range(300, 900, 20):
        raw = payload(now)
        if variant == 'per_cav':
            # Even after WALL was picked, a re-entrant's context is native.
            selector.sync_incumbents({'a': 'native', 'b': 'wall' if now > 300 else 'native'})
            assert selector.incumbents['a'] == 'native'
        result = selector.select(raw)
        assert result == ('wall' if variant == 'fleet' else {'a': 'wall', 'b': 'wall'})
    with pytest.raises(ValueError, match='grid|reused'):
        selector.select(payload(900))
    record['iteration'] = 168
    torch.save(record, path)
    with pytest.raises(ValueError, match='hash'):
        (selectors.Fleet if variant == 'fleet' else selectors.PerCav)(spec)


@pytest.mark.parametrize('residual_hook_exists', (False, True))
def test_grouped_backend_uses_peer_atomic_wall_and_retires_to_exact_fleet_controller(monkeypatch, residual_hook_exists):
    from tools.c1switch2 import runtime
    from tools.c1switch2.core import Handover
    definition = importlib.util.spec_from_file_location('_c1switch2_wall_cpu_fixture',
        Path(__file__).with_name('test_switchable_wall.py'))
    fixture = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(fixture)
    shared_root = Path(os.environ.get('C1SWITCH2_SELECTOR_SOURCE', 'E:/agent-work/c4-selmappo/tools/selmappo'))
    cav_root = Path(os.environ.get('C1SWITCH2_CAV_SOURCE', 'E:/agent-work/c4-selmappo2/tools/selmappo2'))
    if not (shared_root/'runtime.py').is_file() or not (cav_root/'mixed.py').is_file():
        pytest.skip('complete peer execution snapshot not staged')
    def source_package(root):
        return selectors.package(dict(status='FROZEN_FINAL', root=str(root),
            sources={p.name: sha(p) for p in root.glob('*.py')}), {'__init__.py'})
    shared = importlib.import_module(source_package(shared_root)+'.runtime')
    mixed = importlib.import_module(source_package(cav_root)+'.mixed')
    env, calls = fixture.fake_env()
    original_residual = lambda *a, **k: {}
    if residual_hook_exists:
        env.ownership.execute_residual = original_residual
    original_convert = env._accel_by_vehicle
    def base_init(self, *args, **kwargs):
        self.env, self.wall, self.inputs = env, kwargs['wall'], {}
        self.handover = Handover(lambda name: runtime.v7.Controller(name, {}))
        def observed():
            raw = payload(env.now)
            for row in raw['history']:
                row['self_reports'] = [dict(vehicle_id=str(i), timestamp_s=row['now_s'],
                    s_m=float(env.frame.s[i]), lane=int(env.frame.lane[i]), v_mps=float(env.frame.v[i]),
                    a_mps2=0., age_s=0., source='cav_self') for i in (0, 1, 2)]
            return raw
        self.belief = SimpleNamespace(payload=observed)
    monkeypatch.setattr(runtime.Backend, '__init__', base_init)
    backend = runtime.PerCavBackend(shared=shared, mixed_factory=mixed.make_mixed_fleet(shared), wall=Wall())
    assert backend.current_names() == {'0': 'native', '1': 'native', '2': 'native'}
    backend.select({'0': 'wall', '1': 'wall', '2': 'native'})
    backend.mixed.step(env.ownership, fixture.reports(env.frame, env.now), now_s=env.now,
                       live_ids={'0', '1', '2', '3'})
    group = backend.mixed.wall
    assert backend.mixed.members == {'0', '1'} and group.law.n_walls == 1
    assert backend.current_names()['0'] == 'wall'
    backend.select('native')
    assert backend.mixed is None and env.fleet is None and not group.running
    if residual_hook_exists:
        assert env.ownership.execute_residual is original_residual
    else:
        assert not hasattr(env.ownership, 'execute_residual')
    assert env._accel_by_vehicle is original_convert
    assert backend.handover.name == 'native'
    assert all(call[2] != '3' for call in calls)  # No HDV writer through either ownership path.
