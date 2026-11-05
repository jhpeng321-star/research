"""G4 production orchestration on fake physics; real collectors, workers and PPO.

No manufactured G4 captures/envelopes and no replacement of temporal targets.
The only physics implementation installed here is the explicit toy plant.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time
import types

import pytest
import torch

from . import accel_plant, g4_runtime as runtime, launch_training as launch
from . import sparse_synergy_credit as credit, track2_backend as tb
from .test_track2_backend_cpu import cfg, cpu_threads, fake_load, rows  # noqa: F401
from .v2v_intent import IntentPolicy


@pytest.fixture
def toy(cfg, monkeypatch):
    """Install fake physics beneath accel_plant, never fake the G4 adapters."""
    calls, fail_worker = [], [False]
    world = dict(seed=7, cell=0, repeat=0, penetration=.4, demand=2400)
    cfg['worlds_by_update'] = {'1': [world], '2': [world]}
    cfg['g4_synergy_credit'] = dict(enabled=True, registration=credit.registration(),
        phase_id='g4-production-cpu', first_update=1)
    cfg['budget']['phases']['g4_synergy_credit'] = 230400
    cfg['initial_sidecars_required'] = True
    cfg['warm_history'] = False
    # A prospective test input, chosen only from the registered sampling hash.
    cfg['training_seed'] = next(seed for seed in range(1000) if credit._sampled(
        dict(time_s=1100, proposer='c1'), dict(world=world), dict(training_seed=seed))
        and not credit._sampled(dict(time_s=1160, proposer='c1'),
                                dict(world=world), dict(training_seed=seed)))
    fake_load(monkeypatch)
    for arm in ('B', 'R'):
        torch.manual_seed(7)
        policy = tb.EpisodePolicy(IntentPolicy(), cfg, arm)
        with torch.no_grad():
            # A real, fixed checkpoint that strongly prefers feasible walls.
            for parameter in policy.trigger.trigger_head.parameters():
                parameter.zero_()
            hidden = policy.layer.base.actor[-1].in_features
            wall = policy.trigger.families.index('wall')
            policy.trigger.trigger_head[0].weight[0, hidden+wall] = 5.
            policy.trigger.trigger_head[-1].weight[0, 0] = 5.
        torch.save(dict(arm=arm, update=0, worker=policy.worker_state(),
            layer_optimizer=policy.layer.optimizer.state_dict(),
            trigger_optimizer=policy.trigger_optimizer.state_dict()),
            Path(cfg['run_root'])/f'{arm}_track2_000.pt')

    class Physics:
        def __init__(self, path, world, sim_s, checkpoint):
            self.path, self.now, self.actual = Path(path), 300, 300
            self.totals = dict(team_delay_vs=0., road_delay_vs=0., source_wait_vs=0.,
                collisions=0., teleports=0., hard_brake=0., ttc_below_3_vs=0.)
            calls.append(str(path))
            if fail_worker[0] and Path(path).name.startswith('g4_'):
                raise RuntimeError('fake physics worker failure')

        def fingerprint(self):
            return dict(t_s=self.now, metrics=deepcopy(self.totals))

        def close(self):
            pass

    def observations(physics):
        return rows() if physics.now >= 1100 else []

    def tick(physics, controller, trace):
        observed = observations(physics)
        commands = {r['vid']: 0. for r in observed if r['is_cav']}
        requests = controller.apply(observed, commands, physics.now)
        values = [r['requested_raw'] for r in requests.values()]
        # Explicit nonlinear toy physics, not a claimed SUMO measurement.
        delay = 5+sum(v*v for v in values)+3*sum(values)**2
        physics.totals['team_delay_vs'] += delay
        physics.totals['road_delay_vs'] += delay
        vehicles = [dict(r, issued_acceleration_mps2=requests.get(r['vid'], {}).get('requested_raw', 0.),
            command_present=r['is_cav']) for r in observed]
        trace.write(json.dumps(dict(time_s=physics.now, vehicles=vehicles, actions=[]))+'\n')
        physics.now += 1
        physics.actual = physics.now
        return []

    backend = types.ModuleType('tools.leverprobe.backend')
    backend.setup = lambda *args: None
    backend.Backend = Physics
    package = types.ModuleType('tools.leverprobe')
    package.backend = backend
    monkeypatch.setitem(sys.modules, 'tools.leverprobe', package)
    monkeypatch.setitem(sys.modules, 'tools.leverprobe.backend', backend)
    from tools.coopfree import backend as cf
    monkeypatch.setattr(cf, 'tick', tick)
    monkeypatch.setattr(cf, 'observations', observations)
    monkeypatch.setattr(accel_plant, 'install_seed_scope', lambda: None)
    monkeypatch.setattr(accel_plant, 'guard', lambda *args: None)
    from . import accel_protocol, runner, storage
    monkeypatch.setattr(accel_protocol, 'install_seed_scope', lambda: None)
    monkeypatch.setattr(accel_protocol, 'guard', lambda *args: None)
    monkeypatch.setattr(runner, 'admission', lambda cfg: None)
    monkeypatch.setattr(tb, '_approved', lambda cfg: None)
    monkeypatch.setattr(storage, 'ledger', lambda root: [json.loads(line) for line in
        (Path(root)/'CHARGES.jsonl').read_text().splitlines()] if (Path(root)/'CHARGES.jsonl').exists() else [])

    def wave(cfg, policies, worlds, ident, extras, **kwargs):
        result = []
        for arm, policy in policies.items():
            for world in worlds:
                key = f'{ident}_{arm}_{world["seed"]}'
                job = dict(id=key, path=str(Path(cfg['run_root'])/'jobs'/key),
                    world=world, arm=arm, role='train', weights=deepcopy(policy.state_dict()),
                    runtime='fake', checkpoint='fake', noise_seed=9, deterministic=False,
                    sim_s=1200, queue_root=cfg['run_root'], accel_bounds=[-3., 2.6],
                    warm_history=False, guided=False, **extras)
                result.append(accel_plant.episode(job))
        return result

    monkeypatch.setattr(tb, '_wave', wave)
    monkeypatch.setattr(tb.ArmRuntime, 'bank_pairs', lambda self: ([], dict(status='CPU no bank')))
    def safety(self, episodes):
        self.safety_checked = True
        return dict(status='toy physics zero hard events')
    monkeypatch.setattr(tb.ArmRuntime, 'safety', safety)
    monkeypatch.setattr(launch, 'augmentation_receipt', lambda cfg: dict(admitted=dict(B=0, R=0)))
    monkeypatch.setattr(launch, 'health', lambda root: None)
    monkeypatch.setattr(launch.production, 'backend_config', lambda cfg, **kwargs: deepcopy(cfg))
    monkeypatch.setattr(launch.production, 'bank_move_supervision', lambda hooks:
        dict(final_ce=None, source='CPU no bank'))
    launch.write(Path(cfg['run_root'])/'A_UPDATE_001.json', dict(reference=True))
    hooks_by_arm, aborted, errors = {}, threading.Event(), []
    original_build = tb.build_arm_runtime
    def build(cfg, arm):
        hooks = original_build(cfg, arm)
        hooks_by_arm[arm] = hooks
        return hooks
    monkeypatch.setattr(tb, 'build_arm_runtime', build)
    def wait(path, root):
        deadline = time.monotonic()+120
        while not Path(path).exists():
            if aborted.is_set():
                raise RuntimeError('paired CPU owner failed: '+repr(errors[0])) from errors[0]
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(.01)
        return launch.read(path)
    monkeypatch.setattr(launch, 'wait_for', wait)
    def update(arm):
        try:
            torch.set_num_threads(1)
            return launch.update(cfg, arm, 1)
        except BaseException as error:
            errors.append(error)
            aborted.set()
            raise
    return types.SimpleNamespace(cfg=cfg, calls=calls, fail_worker=fail_worker,
        hooks=hooks_by_arm, update=update)


@pytest.mark.parametrize('mode', ['measured', 'no_sample', 'reservation_failure', 'worker_failure'])
def test_real_launcher_capture_worker_paired_ppo_and_restart(toy, mode):
    cfg, root = toy.cfg, Path(toy.cfg['run_root'])
    if mode == 'no_sample':
        # Outcome-blind input whose sole joint boundary is not sampled.
        cfg['training_seed'] = next(seed for seed in range(1000) if all(not credit._sampled(
            dict(time_s=t, proposer='c1'), dict(world=dict(seed=7)), dict(training_seed=seed))
            for t in (1100, 1160)))
    if mode == 'reservation_failure':
        cfg['budget']['phases']['g4_synergy_credit'] = 0
    if mode == 'worker_failure':
        toy.fail_worker[0] = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {a: pool.submit(toy.update, a) for a in ('B', 'R')}
        if mode.endswith('failure'):
            for future in futures.values():
                with pytest.raises((RuntimeError, ValueError)):
                    future.result(timeout=180)
        else:
            receipts = {a: future.result(timeout=180) for a, future in futures.items()}
    inputs = {a: runtime.read(runtime.input_path(cfg, a, 1))['episodes'] for a in ('B', 'R')}
    assert all(e['g4_source'] == 'ordinary_training' for es in inputs.values() for e in es)
    assert all(credit.opportunities(es) for es in inputs.values())
    branch_calls = [p for p in toy.calls if Path(p).name.startswith('g4_')]
    ledger = root/'CHARGES.jsonl'
    if mode == 'reservation_failure':
        assert not ledger.exists() and not branch_calls
        assert all(not h['runtime'].policy.trigger_optimizer.state for h in toy.hooks.values())
        return
    charges = [json.loads(line) for line in ledger.read_text().splitlines()]
    plan = charges[0]['g4_plan']
    assert plan['reservation_after']['starts'] == dict.fromkeys(('B', 'R'), 0 if mode == 'no_sample' else 1)
    assert sum(r['sim_s'] for r in charges) == plan['quote_sim_s']
    before = ledger.read_bytes()
    calls_before = len(toy.calls)
    if mode == 'worker_failure':
        assert len(branch_calls) == 1
        assert not runtime._envelope_path(cfg, 1).exists()
        with pytest.raises(ValueError, match='incomplete; no retry'):
            runtime.Coordinator(cfg).measure(1, inputs)
        assert all(not h['runtime'].policy.trigger_optimizer.state for h in toy.hooks.values())
    else:
        envelope = runtime.read(runtime._envelope_path(cfg, 1))
        assert len(branch_calls) == (0 if mode == 'no_sample' else 16)
        assert runtime.Coordinator(cfg).measure(1, inputs) == envelope
        assert all((root/'jobs'/row['id']/'RESULT.json').is_file() for row in charges)
        assert all(receipts[a]['learning']['g4_synergy_credit']['plan_sha256'] == plan['plan_sha256']
                   for a in ('B', 'R'))
        for arm in ('B', 'R'):
            learner = toy.hooks[arm]['runtime']
            assert learner.policy.trigger_optimizer.state  # actual categorical PPO
            learner.policy.layer._check_frozen()
            assert receipts[arm]['learning']['reward']['temporal']['method'] == 'full_rollout_discounted_return'
        if mode == 'measured':
            assert len(envelope['rows']) == 2
            assert any(row['synergy_veh_s'] != 0 for row in envelope['rows'])
            for job in plan['jobs']:
                payload = runtime.load_capture(job, cfg)
                assert payload['time_s'] == 1100
                assert max(map(int, payload['samples'])) == 1099
                assert max(map(int, payload['prefixes'])) == 1100
            job = plan['jobs'][0]
            file = root/'jobs'/job['start']['episode_id']/'g4_captures'/f'{job["start"]["opportunity_id"]}.pt'
            original = file.read_bytes()
            try:
                file.write_bytes(original+b'tampered')
                with pytest.raises(ValueError, match='immutable capture bytes'):
                    runtime.load_capture(job, cfg)
            finally:
                file.write_bytes(original)
    assert ledger.read_bytes() == before and len(toy.calls) == calls_before
    # New coordinator instance / next update derives cumulative state from the
    # authoritative ledger, including failed attempts. No phase allowance reset.
    for es in inputs.values():
        for episode in es:
            for step in episode['steps']:
                step['sample'].pop('track2_trigger', None)
    later = runtime.Coordinator(cfg).measure(2, inputs)
    assert later['plan']['reservation_before'] == plan['reservation_after']
    assert later['plan']['reservation_after']['reserved_sim_s'] == plan['quote_sim_s']
    assert len(toy.calls) == calls_before


def test_prospective_activation_and_practice_count_binding(cfg, tmp_path):
    cfg['g4_synergy_credit'] = dict(enabled=True, registration=credit.registration(),
        phase_id='later', first_update=4)
    assert not runtime.enabled(cfg, 3)
    assert runtime.enabled(cfg, 4)
    assert not runtime.enabled(cfg, 4, smoke=True)
    assert credit.trigger_terms(cfg, 'B', 3, [], None) == ({}, None)
    runtime.publish(runtime.input_path(cfg, 'B', 4), dict(binding=credit.settings(cfg),
        update=4, arm='B', episodes=[dict(practice=dict(slot='p')), dict(id='ordinary')]))
    assert runtime.paired_practice_count(cfg, 4) == 1
    cfg['g4_synergy_credit']['phase_id'] = 'changed'
    with pytest.raises(ValueError, match='practice count'):
        runtime.paired_practice_count(cfg, 4)


@pytest.mark.parametrize('slot', [None, False])
def test_disabled_capture_requires_no_new_job_fields(slot):
    cfg = {} if slot is None else dict(g4_synergy_credit=slot)
    assert runtime.BoundaryCapture.for_job(dict(arm='B', track2_config=cfg), object(), {}) is None


def test_actor_local_release_mask_and_no_future_triggers(cfg):
    from .acceleration import Controller
    from .v2v_runtime import controller_class
    move = tb.moves.Move('wall', ('c1', 'c2'), 2, 20, .3)
    job = dict(job_sha256='b'*64, retained_members=['c1', 'c2'], verification_seed=1,
        start=dict(time_s=300, move=move.to_dict(), capture=dict(crn_sha256='a'*64)))
    payload = dict(samples={}, memory={}, reports={}, previous_frame=None,
                   intent_board={}, trigger_intent_board={},
                   generator_state=torch.Generator().manual_seed(9).get_state())
    observer = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')
    controller = controller_class(Controller)(IntentPolicy(), torch.Generator().manual_seed(9), False)
    runtime.branch_hook(job, payload, observer)(controller, {}, {})
    observed = rows()
    observed[0]['s_m'] = tb.moves._ZONE_S0-500
    observed[1]['s_m'] = tb.moves._ZONE_S0
    for t in range(300, 303):
        controller.apply(observed, dict(c1=0., c2=0.), t)
    sample = controller.events[-1]['sample']
    # c2's own local position releases c2. c1 cannot see c2's crossing: it must
    # keep its commitment, despite the offline worker having all public rows.
    assert sample['g4_fixed_macro']['offsets'] == dict(c1=-.3, c2=0.)
    assert 'track2_trigger' not in sample and 'coop_layer' not in sample
    assert set(controller.events[-1]['requests']) <= {'c1', 'c2'}
    observed[0]['s_m'] = tb.moves._ZONE_S0-10
    controller.apply(observed, dict(c1=0., c2=0.), 303)
    assert controller.events[-1]['sample']['g4_fixed_macro']['offsets'] == dict(c1=0., c2=0.)


def test_real_backend_translation_owner_bound_allocation_and_smoke(tmp_path):
    from . import rerun_cpus as rc, launch_production as production
    cfg = rc.runtime_config({}, root=str(tmp_path/'run'), reference_root=str(tmp_path/'reference'),
        source_root=str(tmp_path/'source'), common_checkpoint_sha256='a'*64, track2=True)
    cfg['launch_optimization'] = deepcopy(production.OPTIMIZATION)
    cfg['worlds_by_update'] = {'1': [dict(seed=7), dict(seed=8)]}
    cfg['g4_synergy_credit'] = dict(enabled=True, registration=credit.registration(),
        phase_id='successor', first_update=1)
    cfg['g4_budget'] = dict(phase_id='successor', allocation_sim_s=230400,
        debits={}, provenance='CPU test prospective owner-bound allocation')
    original = deepcopy(cfg)
    native = production.backend_config(cfg)
    assert cfg == original
    assert native['budget']['phases']['g4_synergy_credit'] == 230400
    assert native['budget']['phases']['v2v_train'] == cfg['budget']['phases']['v2v_train']
    assert sum(native['budget']['phases'].values()) == sum(cfg['budget']['phases'].values())+230400
    smoke = production.backend_config(cfg, smoke=True)
    assert smoke['g4_synergy_credit'] is False
    assert 'g4_synergy_credit' not in smoke['budget']['phases']
    del cfg['g4_budget']
    with pytest.raises(ValueError, match='owner-bound'):
        production.backend_config(cfg)


@pytest.mark.parametrize('failure', ['owner_remaining', 'atomic_write', 'debited_phase_spent'])
def test_atomic_reservation_refusals_preserve_ledger_and_do_not_dispatch(cfg, monkeypatch, failure):
    from .test_sparse_synergy_credit_cpu import planned, captures
    pure, episodes, plan, _ = planned()
    cfg.update(pure)
    for arm, es in episodes.items():
        es[0]['g4_captures'] = captures(episodes)[arm]
    cfg['budget'] = dict(total=30000, phases=dict(g4_synergy_credit=19200, train=10000))
    root = Path(cfg['run_root'])
    path = root/'CHARGES.jsonl'
    runtime.publish(path, dict(id='old', phase='train', sim_s=10000))
    before = path.read_bytes()
    called = []
    monkeypatch.setattr(tb, '_approved', lambda cfg: None)
    if failure == 'atomic_write':
        cfg['budget']['total'] = 40000
        publish = runtime._publish
        def interrupted(path, data, **kwargs):
            if Path(path).name == 'CHARGES.jsonl':
                raise OSError('simulated interruption before atomic ledger commit')
            return publish(path, data, **kwargs)
        monkeypatch.setattr(runtime, '_publish', interrupted)
    if failure == 'debited_phase_spent':
        cfg['budget']['total'] = 40000
        cfg['budget']['phases']['train'] = 9999
    with pytest.raises((ValueError, OSError)):
        runtime.Coordinator(cfg, worker=lambda *args: called.append(args)).measure(1, episodes)
    assert not called and path.read_bytes() == before
    assert not runtime._envelope_path(cfg, 1).exists()


def test_successful_atomic_batch_preserves_existing_append_only_bytes(cfg, monkeypatch):
    from .test_sparse_synergy_credit_cpu import planned, captures, measured
    pure, episodes, _, _ = planned()
    cfg.update(pure)
    for arm, es in episodes.items():
        es[0]['g4_captures'] = captures(episodes)[arm]
    cfg['budget'] = dict(total=40000, phases=dict(g4_synergy_credit=19200, train=10000))
    path = Path(cfg['run_root'])/'CHARGES.jsonl'
    prefix = b'{ "id" : "old", "phase" : "train", "sim_s" : 10000 }\n'
    path.write_bytes(prefix)
    monkeypatch.setattr(tb, '_approved', lambda cfg: None)
    seen = []
    def worker(job, cfg):
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert sum(r['sim_s'] for r in rows) == 10000+19200
        assert len(rows) == 17  # all paired branches committed before first job
        seen.append(job['job_sha256'])
        return measured(job)
    envelope = runtime.Coordinator(cfg, worker=worker).measure(1, episodes)
    assert path.read_bytes().startswith(prefix) and len(seen) == 16
    assert runtime.Coordinator(cfg, worker=worker).measure(1, episodes) == envelope
    assert len(seen) == 16
