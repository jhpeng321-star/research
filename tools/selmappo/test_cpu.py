"""Mechanical CPU gates; no SUMO import/start, no training/evaluation outcomes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .environment import SelectorEnv, action_name, window_reward
from .handover import FleetHandover
from .integration import evaluation_panel, load, registered
from .observations import ACTOR_DIM, SHARED_DIM, build
from .protocol import (BLOCKS, CAP, LIBRARY, TIMES, approval_line, audit_registration,
                       digest, guard_seed, guard_world, safe_path, training_world)
from .storage import Budget, PrechargedEpisode


def payload(now=300):
    def row(key, lane=0):
        return dict(vehicle_id=key, timestamp_s=now, s_m=-100., lane=lane, v_mps=20., a_mps2=-.5)
    return dict(history=[dict(now_s=now, self_reports=[row('cav1')],
                             detections=[row('anonymousA'), row('anonymousB', 1)],
                              neighbours=[], coverage=[], counters=None)])


def transport_probe(blob):
    """Spawn-safe CPU-only exercise of the actual byte boundary, no backend."""
    from .transport import pack, unpack
    return pack(unpack(blob))


class FakeBackend:
    def __init__(self):
        self.now, self.picks = 300, []
    def observable(self):
        return payload(self.now)
    def select_fleet(self, name):
        self.picks.append((self.now, name))
    def tick(self):
        self.now += 1
        return dict(speed_deficit_vehicle_seconds=2., source_wait_vehicle_seconds=3.,
                    ttc_below_3_vehicle_seconds=.5, braking_vehicle_seconds=.25)


class EnvironmentTests(unittest.TestCase):
    def test_closed_terminal_connection_keeps_final_reward_and_closes_once(self):
        from tools.selmappo.runtime import Backend
        class ClosingEnv:
            def __init__(self):
                self.clock, self.api = 300, object()
            @property
            def now(self):
                if self.api is None:
                    raise AssertionError('closed SUMO clock queried')
                return self.clock
            def step(self, *args, **kwargs):
                self.clock += 1
                done = self.clock == 900
                if done:
                    self.api = None
                return None, dict(done=done, time_s=self.clock, lane_commands={},
                                  interval=dict(speed_deficit_vehicle_seconds=2.,
                                                source_wait_vehicle_seconds=3.))
            close = Mock()
        backend = Backend.__new__(Backend)
        backend.env, backend.mixed = ClosingEnv(), None
        backend._terminal_time, backend._closed = None, False
        backend.timing = dict(controller_wall_s=0., environment_wall_s=0., observable_wall_s=0.)
        backend.handover = SimpleNamespace(name='wall', controller=Mock())
        backend.handover.controller.command.return_value = (None, {})
        backend.lib = SimpleNamespace(Backend=SimpleNamespace(observe=Mock()))
        backend.observable = lambda: payload(backend.now)
        backend.select_fleet = Mock()
        backend.traci = SimpleNamespace(simulationStep=Mock())
        backend.original_step = Mock()
        backend.guard_scope = SimpleNamespace(__exit__=Mock())
        backend.budget, backend.actual = Mock(), 900
        env = SelectorEnv(backend, 'fleet')
        rewards = [env.step({'fleet': 5})[1] for _ in TIMES]
        self.assertEqual(sum(rewards), -3000.)
        self.assertEqual(backend.now, 900)
        self.assertEqual(backend.lib.Backend.observe.call_count, 599)
        backend.close()
        backend.close()
        backend.handover.controller.stop.assert_called_once_with(now_s=900)
        backend.budget.finish.assert_called_once_with(900)
        self.assertIs(backend.traci.simulationStep, backend.original_step)

    def test_thirty_windows_exactly(self):
        backend = FakeBackend()
        env = SelectorEnv(backend, 'fleet')
        total = 0.
        for j in range(30):
            _, reward, done, metrics = env.step({'fleet': j % 6})
            total += reward
            self.assertEqual(done, j == 29)
            self.assertEqual(metrics['ttc_below_3_vs'], 10.)
        self.assertEqual([t for t, _ in backend.picks], list(TIMES))
        self.assertEqual(backend.now, 900)
        self.assertEqual(total, -3000.)
        with self.assertRaises(ValueError):
            env.step({'fleet': 0})

    def test_invalid_action_atomic(self):
        backend = FakeBackend()
        env = SelectorEnv(backend, 'fleet')
        for actions in ({'fleet': True}, {'fleet': 6}, {'fleet': 1.}, {}, {'extra': 0}):
            with self.assertRaises(ValueError):
                env.step(actions)
        self.assertEqual(backend.picks, [])
        self.assertEqual(backend.now, 300)

    def test_off_grid_rejected(self):
        backend = FakeBackend()
        env = SelectorEnv(backend, 'fleet')
        backend.now = 301
        with self.assertRaises(ValueError):
            env.step({'fleet': 0})

    def test_reward_full_team(self):
        self.assertEqual(window_reward([dict(speed_deficit_vehicle_seconds=7.,
                                             source_wait_vehicle_seconds=11.)])[0], -18.)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                window_reward([dict(speed_deficit_vehicle_seconds=value, source_wait_vehicle_seconds=0)])

    def test_safety_stops(self):
        backend = FakeBackend()
        backend.tick = lambda: dict(collision_involvements=1)
        with self.assertRaises(ValueError):
            SelectorEnv(backend, 'fleet').step({'fleet': 0})

    def test_sixth_is_wall(self):
        self.assertEqual(action_name(5), 'wall')


class ObservableTests(unittest.TestCase):
    def test_schema(self):
        observation = build(payload(), variant='fleet', incumbents={'fleet': 'wall'})
        self.assertEqual(len(observation.shared), SHARED_DIM)
        self.assertEqual(len(observation.actors[0]), ACTOR_DIM)
        self.assertEqual(observation.actors[0][SHARED_DIM:SHARED_DIM+6], (0, 0, 0, 0, 0, 1))

    def test_hidden_extra_noninterference(self):
        first = payload()
        second = copy.deepcopy(first)
        second['unseen_global_rows'] = object()
        second['history'][-1].update(true_penetration=.8, demand=99999, backlog_truth=99999)
        for row in second['history'][-1]['detections']:
            row['hidden_driver_type'] = object()
        self.assertEqual(build(first, variant='fleet', incumbents={}),
                         build(second, variant='fleet', incumbents={}))

    def test_anonymous_identity_order_noninterference(self):
        first = payload()
        second = copy.deepcopy(first)
        second['history'][-1]['self_reports'][0]['vehicle_id'] = 'renamedSelf'
        second['history'][-1]['detections'].reverse()
        for j, row in enumerate(second['history'][-1]['detections']):
            row['vehicle_id'] = f'renamed{j}'
        self.assertEqual(build(first, variant='fleet', incumbents={}),
                         build(second, variant='fleet', incumbents={}))

    def test_no_truth_counter_fallback(self):
        first = payload()
        first['history'][0]['requested'] = 123456
        self.assertEqual(build(first, variant='fleet', incumbents={}).shared[-3:], (0, 0, 0))

    def test_observed_flow(self):
        data = payload(320)
        earlier = payload()['history'][0]
        earlier['counters'] = dict(timestamp_s=300, requested=10, entered=8)
        data['history'].insert(0, earlier)
        data['history'][-1]['counters'] = dict(timestamp_s=320, requested=20, entered=13)
        self.assertEqual(build(data, variant='fleet', incumbents={}).shared[-3:], (.5, .25, 1))

    def test_future_sensor_rejected(self):
        data = payload()
        data['history'][0]['detections'][0]['timestamp_s'] = 301
        with self.assertRaises(ValueError):
            build(data, variant='fleet', incumbents={})

    def test_empty_cav_fleet_still_has_action(self):
        data = payload()
        data['history'][0]['self_reports'] = []
        self.assertEqual(build(data, variant='fleet', incumbents={}).keys, ('fleet',))


class HandoverTests(unittest.TestCase):
    def test_wall_lifecycle_and_incumbent_memory(self):
        owner = SimpleNamespace(release_all=Mock(), controller=None)
        env = SimpleNamespace(ownership=owner, api=SimpleNamespace(vehicle=SimpleNamespace(
            getIDList=lambda: ['cav', 'hdv'])), _circles=object(), pi_session=object(), _direct_pi=True,
            now=300)
        created = []
        def factory(name):
            result = SimpleNamespace(name=name, memory=[], start=Mock(), stop=Mock())
            created.append(result)
            return result
        h = FleetHandover(factory)
        wall = h.select('wall', env)
        wall.memory.append('assembly')
        self.assertIs(h.select('wall', env), wall)
        self.assertEqual(wall.memory, ['assembly'])
        wall.start.assert_called_once_with(now_s=300)
        self.assertEqual(owner.release_all.call_count, 1)
        h.select('pi', env)
        wall.stop.assert_called_once_with(now_s=300)
        self.assertEqual(owner.release_all.call_count, 2)
        self.assertIsNone(env.pi_session)
        self.assertFalse(env._direct_pi)
        self.assertEqual(h.switches, 2)
        with self.assertRaises(ValueError):
            h.select('invented', env)
        self.assertEqual(owner.release_all.call_count, 2)


class DiskTests(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ['SELMAPPO_TEST_ROOT'])
        if not parent.is_dir():
            raise ValueError('pre-existing test parent required')
        # Retain test fixtures/receipts; no cleanup/deletion.
        self.root = Path(tempfile.mkdtemp(prefix='selmappo_cpu_', dir=parent))

    def text(self, name, value):
        path = self.root/name
        path.write_text(value, encoding='utf-8')
        return path

    def test_approval_not_quoted_or_negative(self):
        self.assertIsNone(approval_line(self.text('negative.md', "proposal: S-SELMAPPO BUDGET APPROVED\n")))
        self.assertIsNotNone(approval_line(self.text('yes.md',
            '- 23:42 S-SELMAPPO BUDGET APPROVED IN ADVANCE by the owner (chat)')))

    def test_seed_roles_and_forbidden(self):
        for seed in (964000, 964199, 962001, 977123, 18100000, 18007774):
            with self.assertRaises(ValueError):
                guard_seed(seed, 'fleet')
        self.assertEqual(guard_seed(18007998, 'fleet', smoke=True), 18007998)
        with self.assertRaises(ValueError):
            guard_seed(969112, 'fleet', evaluation=True)
        self.assertEqual(CAP, 169*46*900+2*900+16*900)

    def test_seed_cell_schedule_guard(self):
        world = training_world(18000000)
        guard_world(world, 'fleet')
        world['demand'] = 2600
        with self.assertRaises(ValueError):
            guard_world(world, 'fleet')

    def test_runtime_missing_integration_before_library(self):
        from .runtime import Backend
        reg = self.text('runtime_reg.md',
            '- 23:42 S-SELMAPPO BUDGET APPROVED IN ADVANCE by the owner\n'
            '- S-SELMAPPO DESIGN REGISTERED SHA256 '+('a'*64)+
            ' fleet_cap=7012800 protocol=S-C1SWITCH2/v2\n')
        with patch('tools.selmappo.runtime.library') as factory:
            with self.assertRaises(FileNotFoundError):
                Backend(world=training_world(18000000), variant='fleet', registration=reg,
                        c1_root=self.root, episode_root=self.root, budget=Mock(),
                        integration=self.root/'absent.json', design_sha='a'*64)
            factory.assert_not_called()

    def test_registration_umbrella_collision(self):
        reg = self.text('registration.md', '- OTHER TRAIN17900000..18999999\n')
        self.assertFalse(audit_registration(reg)['passed'])
        self.assertFalse(audit_registration(self.text('short.md', '- OTHER 18000000-099\n'))['passed'])

    def test_exact_task_hash_exemption_and_shared_block(self):
        from .protocol import FLEET_DESIGN_SHA
        own = ('- S-SELMAPPO DESIGN REGISTERED SHA256 '+FLEET_DESIGN_SHA+
               ' protocol=S-C1SWITCH2/v2 fleet_cap=7012800 TRAIN18000000..18007999\n')
        self.assertTrue(audit_registration(self.text('own.md', own))['passed'])
        competing = own.replace(FLEET_DESIGN_SHA, 'b'*64)
        self.assertFalse(audit_registration(self.text('competing.md', own+competing))['passed'])
        peer = ('- S-SELMAPPO2 DESIGN REGISTERED SHA256 '+('c'*64)+
                ' protocol=S-C1SWITCH2/v2 per_cav_cap=7012800 TRAIN18100000..18107999\n')
        reg = self.text('peer.md', own+peer)
        self.assertTrue(audit_registration(reg)['passed'])
        self.assertTrue(audit_registration(reg, variant='per_cav', design_sha='c'*64)['passed'])
        self.assertEqual(training_world(18100000, 'per_cav')['seed'], 18100000)

    def test_no_unregistered_integration(self):
        reg = self.text('reg.md', '- 23:42 S-SELMAPPO BUDGET APPROVED IN ADVANCE by the owner\n')
        with self.assertRaises(PermissionError):
            registered(reg, 'a'*64)
        with self.assertRaises(PermissionError):
            load('missing.json', reg, self.root, 'a'*64)

    def test_budget_no_replay_and_no_refund(self):
        budget = Budget(self.root)
        world = dict(seed=18000000, demand=1900, penetration=.05)
        budget.reserve(world, 'fleet')
        with self.assertRaises(ValueError):
            Budget(self.root)
        budget.finish(17.5)
        restored = Budget(self.root)
        self.assertEqual(restored.charged, 900)
        with self.assertRaises(ValueError):
            restored.reserve(world, 'fleet')

    def test_continuation_retains_charge_and_refuses_replay(self):
        from .continuation import inherit, ORIGINAL
        original = self.root/ORIGINAL
        (original/'training').mkdir(parents=True)
        old = Budget(original/'training')
        old.reserve(dict(seed=18007998, demand=2700, penetration=.4), 'fleet', smoke=True)
        old.finish(0)
        ledger_sha = digest(old.path)
        (original/'DISPATCH.json').write_text(json.dumps(dict(pid=999999999)))
        (original/'STOP.json').write_text(json.dumps(dict(own_pid=999999999,
            own_pid_exited=True, status='STOPPED_BEFORE_PHYSICS', charged_sim_s=900,
            actual_sim_s=0, training_episodes=0, checkpoints=0, ledger_sha256=ledger_sha)))
        reg = self.text('ack.md', "- 00:29 MAIN: MAIN ACKNOWLEDGES S-SELMAPPO's no-retry stop under the same cap: run only the original unused second mechanics episode\n")
        out = self.root/'new'
        out.mkdir()
        with patch('tools.selmappo.continuation.LEDGER_SHA', ledger_sha):
            inherit(out, original, reg)
            with self.assertRaises(FileExistsError):
                inherit(out, original, reg)
            with self.assertRaises(PermissionError):
                inherit(out, original, self.text('noack.md', 'budget approved'))
        continued = Budget(out)
        self.assertEqual(continued.charged, 900)
        with self.assertRaises(ValueError):
            continued.reserve(dict(seed=18007998, demand=2700, penetration=.4), 'fleet', smoke=True)
        continued.reserve(dict(seed=18007999, demand=2700, penetration=.4), 'fleet', smoke=True)
        continued.finish(900)
        self.assertEqual(continued.charged, 1800)

    def test_parallel_reservations_close_by_identity(self):
        budget = Budget(self.root)
        a = dict(seed=18000000, demand=1900, penetration=.05)
        b = dict(seed=18000001, demand=1900, penetration=.05)
        budget.reserve(a, 'fleet')
        budget.reserve(b, 'fleet')
        with self.assertRaises(ValueError):
            budget.finish(900)
        budget.finish(900, 'fleet:18000001')
        budget.finish(12, 'fleet:18000000')
        self.assertEqual(Budget(self.root).charged, 1800)

    def test_production_continuation_binds_both_charges_and_skips_all_smoke(self):
        from .continuation import inherit_production, validate_production, TERMINAL
        from .train import mechanics_seeds, worlds
        original = self.root/TERMINAL
        (original/'training').mkdir(parents=True)
        old = Budget(original/'training')
        for seed, actual in ((18007998, 0), (18007999, 900)):
            old.reserve(dict(seed=seed, demand=2700, penetration=.4), 'fleet', smoke=True)
            old.finish(actual)
        ledger_sha = digest(old.path)
        (original/'training.log').write_text('preserved terminal failure')
        (original/'DISPATCH.json').write_text(json.dumps(dict(pid=999999999)))
        (original/'STOP.json').write_text(json.dumps(dict(own_pid=999999999,
            own_pid_exited=True, status='STOPPED_AT_MECHANICS_TERMINAL', charged_sim_s=1800,
            actual_sim_s=900, training_episodes=0, checkpoints=0, ledger_sha256=ledger_sha,
            log_sha256=digest(original/'training.log'), result_complete=False,
            mechanics_slots_remaining=0)))
        reg = self.text('production_ack.md', '- 01:08 MAIN: ACKNOWLEDGES S-SELMAPPO production-only continuation; retain both charges, no additional mechanics episode; 7,012,800\n')
        out = self.root/'production'
        out.mkdir()
        with patch('tools.selmappo.continuation.TERMINAL_LEDGER_SHA', ledger_sha):
            inherit_production(out, original, reg)
            with self.assertRaises(FileExistsError):
                inherit_production(out, original, reg)
            with self.assertRaises(PermissionError):
                validate_production(original, self.text('old_ack.md', '- 00:29 MAIN: budget approved'))
            (original/'training.log').write_text('changed')
            with self.assertRaises(ValueError):
                validate_production(original, reg)
        continued = Budget(out)
        self.assertEqual(continued.charged, 1800)
        for seed in (18007998, 18007999):
            with self.assertRaises(ValueError):
                continued.reserve(dict(seed=seed, demand=2700, penetration=.4), 'fleet', smoke=True)
        self.assertEqual(mechanics_seeds('fleet', production_only=True), ())
        self.assertEqual(mechanics_seeds('fleet', continuation=True), (18007999,))
        all_worlds = [w for iteration in range(169) for w in worlds('fleet', iteration)]
        self.assertEqual(len({w['seed'] for w in all_worlds}), 7774)
        self.assertEqual(1800+len(all_worlds)*900+14400, CAP)

    def test_priority_admission_yields_at_eighty_percent_without_precharge(self):
        from .dispatch import admitted
        memory = 'MemAvailable: 16777216 kB\n'
        with patch('tools.selmappo.dispatch.shutil.disk_usage', return_value=SimpleNamespace(free=4*1024**3)), \
             patch('pathlib.Path.read_text', return_value=memory), \
             patch('tools.selmappo.dispatch.os.cpu_count', return_value=32), \
             patch('tools.selmappo.dispatch.os.getloadavg', create=True) as load:
            load.return_value = (25.6, 20, 20)
            self.assertFalse(admitted(self.root, 1))
            load.return_value = (13, 20, 20)
            self.assertTrue(admitted(self.root, 12))
            self.assertFalse(admitted(self.root, 13))
        self.assertFalse((self.root/'ledger.jsonl').exists())

    def test_remote_launch_templates_compile_without_dispatch(self):
        from . import launch
        captured = []
        def intercept(host, code, **kwargs):
            compile(code, '<remote launch>', 'exec')
            captured.append(code)
            return SimpleNamespace(stdout='')
        with patch.object(launch, 'run', side_effect=intercept), \
             patch.object(launch, 'registered'), patch.object(launch, 'digest', return_value='a'*64), \
             patch.object(launch, 'safe_path') as safe, \
             patch.object(launch, 'bundle', return_value=({}, {}, 'b'*64)):
            safe.return_value.read_text.return_value = 'registration snapshot'
            launch.stage()
            launch.dispatch('build_test', production_only=True)
            launch.inventory()
        self.assertEqual(len(captured), 3)

    def test_cap_no_silent_overallocation(self):
        budget = Budget(self.root)
        budget.charged = CAP
        with self.assertRaises(ValueError):
            budget.reserve(dict(seed=18000000), 'fleet')

    def test_worker_requires_live_supervisor_precharge(self):
        world = training_world(18000000)
        worker = PrechargedEpisode(self.root/'18000000', world, 'fleet')
        with self.assertRaises(FileNotFoundError):
            worker.reserve(world, 'fleet')
        budget = Budget(self.root)
        budget.reserve(world, 'fleet')
        worker.reserve(world, 'fleet')
        budget.finish(900)
        with self.assertRaises(ValueError):
            PrechargedEpisode(self.root/'18000000', world, 'fleet').reserve(world, 'fleet')

    def test_panel_not_old_or_incomplete(self):
        path = self.text('panel.json', json.dumps(dict(protocol='S-C1SWITCH2/v2', worlds=[])))
        with self.assertRaises(ValueError):
            evaluation_panel(path, digest(path))

    def test_forbidden_path_before_open(self):
        with self.assertRaises(ValueError):
            safe_path(self.root/'sealed'/'never_opened')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'existing CPU torch environment required')
class LearningTests(unittest.TestCase):
    def test_full_wave_inline_transport_is_bitwise_and_descriptor_bounded(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        import torch
        from .learning import Selector, Transition, Updater
        from .transport import pack, retain, unpack
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = Selector()
        observation = build(payload(), variant='fleet', incumbents={})
        actions, logp = model.act(observation)
        episode = [Transition(observation, actions, logp.clone(), -float(j+1)) for j in range(30)]
        blob = pack((model.state_dict(), episode))
        descriptors = Path('/proc/self/fd')
        before = len(list(descriptors.iterdir())) if descriptors.is_dir() else None
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            returned = list(pool.map(transport_probe, [blob]*46))
        episodes = []
        for value in returned:
            state, copied = unpack(value)
            self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(torch.equal(t.old_logp, logp) for t in copied))
            episodes.append(copied)
        if before is not None:
            self.assertLess(len(list(descriptors.iterdir())), before+64)
        parent = Path(os.environ['SELMAPPO_TEST_ROOT'])
        root = Path(tempfile.mkdtemp(prefix='selmappo_transport_', dir=parent))
        retained, sha = retain(root, episodes[0])
        self.assertEqual(digest(root/'trajectory.pkl'), sha)
        self.assertTrue(torch.equal(unpack(retained)[0].old_logp, logp))
        with self.assertRaises(FileExistsError):
            retain(root, episodes[0])
        # Exercise the actual worker's wire schema and durable write boundary.
        from .dispatch import job
        world = training_world(18000000)
        supervisor = Budget(root)
        supervisor.reserve(world, 'fleet')
        args = SimpleNamespace(out=str(root), variant='fleet')
        def no_physics(model, world, args, budget):
            budget.root.mkdir()
            budget.reserve(world, 'fleet')
            budget.finish(0)
            return episodes[0]
        with patch('tools.selmappo.train.episode', side_effect=no_physics):
            wire = job(pack(model.state_dict()), world, args)
        self.assertTrue(wire['ok'])
        self.assertIsInstance(wire['transitions_blob'], bytes)
        self.assertEqual(digest(root/str(world['seed'])/'trajectory.pkl'), wire['trajectory_sha256'])
        self.assertTrue(torch.equal(unpack(wire['transitions_blob'])[0].old_logp, logp))
        supervisor.finish(0)
        stats = Updater(model).update(episodes)
        self.assertEqual(stats['frames'], 46*30)

    def test_long_horizon_and_real_optimizer(self):
        import torch
        from .learning import Selector, Transition, Updater, compute_gae
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = Selector()
        observation = build(payload(), variant='fleet', incumbents={})
        generator = torch.Generator().manual_seed(0)
        actions, logp = model.act(observation, generator=generator)
        ep = [Transition(observation, actions, logp, -float(j+1)) for j in range(30)]
        values = torch.linspace(-10, 10, 30)
        rewards = torch.tensor([t.reward for t in ep])
        _, returns = compute_gae(rewards, values, gamma=1., lam=1., bootstrap_value=0.)
        self.assertTrue(torch.allclose(returns, rewards.flip(0).cumsum(0).flip(0), atol=1e-4))
        before = model.actor[-1].weight.detach().clone()
        stats = Updater(model).update([ep])
        self.assertEqual(stats['frames'], 30)
        self.assertFalse(torch.equal(before, model.actor[-1].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_six_way_argmax_tie_and_rng_isolation(self):
        import torch
        from .learning import Selector
        model = Selector()
        for p in model.parameters():
            p.data.zero_()
        obs = build(payload(), variant='fleet', incumbents={})
        self.assertEqual(model.act(obs, deterministic=True)[0], {'fleet': 0})
        g = torch.Generator().manual_seed(37)
        saved = g.get_state()
        first = model.act(obs, generator=g)[0]
        torch.manual_seed(999)
        g.set_state(saved)
        self.assertEqual(first, model.act(obs, generator=g)[0])
        model.actor[-1].bias.data[5] = 1
        self.assertEqual(model.act(obs, deterministic=True)[0], {'fleet': 5})


if __name__ == '__main__':
    unittest.main()
