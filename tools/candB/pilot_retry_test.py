"""Owner03:00 retry contracts: exact B-L fix, zero-progress recovery, S4 binding."""
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.candB import pilot_continue, pilot_retry, pilot_retry_authority
from tools.candB import pilot_lean_authority, pilot_wmfast
from tools.candB.pilot_retry_launch import launch_source


def workspace_tempdir():
    # Workspace-scoped fixtures: safe_path refuses paths outside allowed roots.
    return tempfile.TemporaryDirectory(dir='.')


def synthetic_parent(root, **override):
    frozen = dict(update=191, scoring_phase_position=191, panel=[1, 2, 3, 4],
        independent=dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.))
    prior = dict(label='EXPLORATORY-LEAN', status='failed',
        error="TypeError: 'NoneType' object is not subscriptable", updates=[],
        seed=965140, resumed_from=dict(update_count=191), zero_start=True,
        frozen_scoring_baseline=frozen)
    prior.update(override)
    (root/'RUN_SUMMARY.json').write_text(json.dumps(prior))
    (root/'frozen_scoring_baseline.json').write_text(json.dumps(frozen))
    return prior


class RetryContractTests(unittest.TestCase):
    def test_retry_launcher_source_contract(self):
        source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
        for expected in ('WMS_SPEED_ES_STACK_SIDES=4', 'WMS_SPEED_FORWARD_MEMORY_FRACTION=.75',
                'WMS_SPEED_FUTURE_VJP_SLOTS=2', 'PILOT_RETRY_AUTHORITY.json',
                'tools.candB.pilot_retry_authority', 'tools.candB.pilot_retry ',
                '>= 46000'):
            self.assertIn(expected, source)
        self.assertNotIn('--new-phase', source)
        self.assertNotIn('WMS_SPEED_FORWARD_WORKERS=2', source)

    def test_retry_authority_binds_owner0300_s4_recipe(self):
        with workspace_tempdir() as directory:
            panel = Path(directory)/'panel.json'
            panel.write_text(json.dumps(dict(starts=[f'w{i}' for i in range(16)])))
            doc = dict(inputs=dict(gate_panel=str(panel)),
                input_sha256=dict(gate_panel='a'*64), config_sha256='b'*64)
            recipe = pilot_retry_authority.recipe(doc)
            self.assertFalse(recipe['continuation_only'])
            self.assertEqual(recipe['wmfast_stack_sides'], 4)
            self.assertIn('03:00', recipe['owner_amendment'])
            self.assertIn('no reinitialization or baseline forwards', recipe['initialization'])
            self.assertEqual(recipe['environment']['WMS_SPEED_ES_STACK_SIDES'], '4')
            self.assertEqual(recipe['environment']['WMS_SPEED_FORWARD_WORKERS'], '1')
            self.assertEqual(recipe['environment']['WMS_SPEED_FORWARD_MEMORY_FRACTION'], '.75')
            self.assertEqual(recipe['environment']['WMS_SPEED_FUTURE_VJP_SLOTS'], '2')
            self.assertEqual(recipe['es_future_indices'], [0, 1, 2, 3])
            self.assertEqual(pilot_retry_authority.ENV['WMS_SPEED_ES_STACK_SIDES'], '4')

    def test_prepare_base_and_adapter_replacement_targets_exact(self):
        continue_source = inspect.getsource(pilot_continue)
        for target in ("payload['update_count'] != prior['updates'][-1]['update']",
                "prior['updates'][-1]['update']-origin"):
            self.assertEqual(continue_source.count(target), 1)
        prepare_source = inspect.getsource(pilot_continue.prepare_base)
        for target in ("'policy_latest.pt'", 'validate_parent(parent)'):
            self.assertIn(target, prepare_source)
        lean_source = inspect.getsource(__import__('tools.candB.pilot_lean', fromlist=['run']))
        self.assertEqual(lean_source.count(
            'from tools.candB import pilot_cli as base, pilot_lean_authority as authority'), 1)
        self.assertIn('measured._wms_checkpoint_passthrough = callback', lean_source)
        specialized = pilot_continue.replace_once(lean_source,
            'from tools.candB import pilot_cli as base, pilot_lean_authority as authority',
            'from tools.candB import pilot_retry_authority as authority')
        compile(specialized, 'pilot_lean.py', 'exec')

    def test_zero_progress_score_fallback(self):
        score = dict(J=1.25, raw_dJ_vs_CTRL0_dense=0.)
        self.assertEqual(pilot_wmfast.last_measured_score(
            dict(updates=[], independent_score_before=score)), score)
        row = dict(independent_J=dict(J=2.5))
        self.assertEqual(pilot_wmfast.last_measured_score(
            dict(updates=[row], independent_score_before=score))['J'], 2.5)

    def test_validate_parent_refusals(self):
        with workspace_tempdir() as directory:
            root = Path(directory)
            synthetic_parent(root)
            with patch.object(pilot_retry, 'PARENT', root):
                with self.assertRaisesRegex(ValueError, 'label|retry is restricted'):
                    pilot_retry.validate_parent(root)
                synthetic_parent(root, label='EXPLORATORY')
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    pilot_retry.validate_parent(root)
                synthetic_parent(root, updates=[dict(update=191)])
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    pilot_retry.validate_parent(root)
                synthetic_parent(root, seed=965141)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    pilot_retry.validate_parent(root)
                synthetic_parent(root, zero_start=False)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    pilot_retry.validate_parent(root)
                # Exact schema reaches the artifact-hash stage and refuses drift.
                synthetic_parent(root)
                with self.assertRaisesRegex(ValueError, 'artifact drift'):
                    pilot_retry.validate_parent(root)

    def test_validate_identity_accepts_exactly_the_b_l_repair(self):
        with workspace_tempdir() as directory:
            home = Path(directory)
            panel = home/'panel.json'
            panel.write_text(json.dumps(dict(starts=[f'w{i}' for i in range(16)])))
            doc = dict(inputs=dict(gate_panel=str(panel)),
                input_sha256=dict(gate_panel='a'*64), config_sha256='b'*64)
            old = home/'old_runtime'
            (old/'authority_gen').mkdir(parents=True)
            authority_path = old/'authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'
            authority_path.write_text(json.dumps(dict(input_sha256=doc['input_sha256'],
                config_sha256=doc['config_sha256'], lean_recipe=pilot_lean_authority.recipe(doc))))
            import hashlib
            prior = dict(exploratory_authority_sha256=hashlib.sha256(
                authority_path.read_bytes()).hexdigest())
            marker = "        for cell in fn.__closure__ or ():\n"
            old_memory = "def _controller_cells(controller):\n" + marker + "            pass\n"
            old_recovery = ('def load_continuation(raw, args):\n'
                "    if (prior['status'] != 'failed' or prior['seed'] != args.seed\n"
                "            or not prior['updates'] or prior['updates'][-1]['update'] != saved\n"
                "            or not start < saved < start + args.max_updates):\n"
                '        pass\n')
            (old/'src/wms/control').mkdir(parents=True)
            (old/'src/wms/control/imagination_memory.py').write_text(old_memory)
            (old/'src/wms/control/policy14_recovery.py').write_text(old_recovery)
            (old/'SOURCE_MANIFEST.json').write_text(json.dumps({
                'src/wms/control/imagination_memory.py': 'one', 'src/wms/control/policy14_recovery.py': 'two',
                'src/wms/control/other.py': 'same', 'configs/c4_policy14_candB.toml': 'cfg'}))
            cwd = home/'new_runtime'
            (cwd/'src/wms/control').mkdir(parents=True)
            (cwd/'configs').mkdir()
            (cwd/'configs/c4_policy14_candB.toml').write_text('unchanged')
            (cwd/'src/wms/control/other.py').write_text('peer file')
            expected_memory = pilot_continue.replace_once(old_memory, marker,
                "        # Explicitly observational wrappers keep run-wide telemetry in shared\n"
                "        # closure cells. Snapshot/release ONLY the underlying physical controller;\n"
                "        # clearing wrapper cells would corrupt its factory and the next replay.\n"
                "        passthrough = getattr(fn, '_wms_checkpoint_passthrough', None)\n"
                "        if passthrough is not None:\n"
                "            visit(passthrough)\n"
                "            return\n" + marker)
            expected_recovery = pilot_continue.replace_once(old_recovery,
                'def load_continuation(raw, args):',
                'def load_continuation(raw, args, *, initial_checkpoint=False):')
            expected_recovery = pilot_continue.replace_once(expected_recovery,
                "    if (prior['status'] != 'failed' or prior['seed'] != args.seed\n"
                "            or not prior['updates'] or prior['updates'][-1]['update'] != saved\n"
                "            or not start < saved < start + args.max_updates):",
                "    progress = (not prior['updates'] and saved == start if initial_checkpoint else\n"
                "                bool(prior['updates']) and prior['updates'][-1]['update'] == saved\n"
                "                and start < saved < start + args.max_updates)\n"
                "    if (prior['status'] != 'failed' or prior['seed'] != args.seed or not progress):")
            (cwd/'src/wms/control/imagination_memory.py').write_text(expected_memory)
            (cwd/'src/wms/control/policy14_recovery.py').write_text(expected_recovery)
            (cwd/'SOURCE_MANIFEST.json').write_text(json.dumps({
                'src/wms/control/imagination_memory.py': 'one.new', 'src/wms/control/policy14_recovery.py': 'two.new',
                'src/wms/control/other.py': 'same', 'configs/c4_policy14_candB.toml': 'cfg'}))
            original = os.getcwd()
            os.chdir(cwd)
            try:
                with patch.object(pilot_retry, 'OLD_RUNTIME', old):
                    pilot_retry.validate_identity(doc, prior)
                    (cwd/'src/wms/control/other.py').write_text('drifted peer file')
                    with self.assertRaisesRegex(ValueError, 'unexpected baseline scientific source drift'):
                        pilot_retry.validate_identity(doc, prior)
                    (cwd/'src/wms/control/other.py').write_text('peer file')
                    (cwd/'src/wms/control/policy14_recovery.py').write_text(expected_recovery+'\n')
                    with self.assertRaisesRegex(ValueError, 'exact reviewed B-L repair'):
                        pilot_retry.validate_identity(doc, prior)
            finally:
                os.chdir(original)

    def test_real_zero_progress_initial_checkpoint_recovery(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        import hashlib
        import torch
        from wms.control import policy14_recovery
        from wms.control.dreamer import DreamerConfig
        from wms.control.contract import RewardWeights
        with workspace_tempdir() as directory:
            root = Path(directory)
            checkpoint = root/'policy_native_best.pt'
            checkpoint.write_bytes(b'owned synthetic zero checkpoint')
            panel = root/'panel.json'
            panel.write_bytes(b'owned synthetic panel')
            prior = dict(status='failed', seed=965140, resumed_from=dict(update_count=191),
                updates=[], wm_members=[],
                effective_config_before_updates=asdict(DreamerConfig()),
                effective_reward_before_updates=asdict(RewardWeights()),
                gate_panel=dict(sha256=hashlib.sha256(panel.read_bytes()).hexdigest()),
                native_curve=[dict(update=191, passed=True, mean_delta=0.)])
            (root/'RUN_SUMMARY.json').write_text(json.dumps(prior))
            args = SimpleNamespace(continue_run=root, new_phase=False, resume_allow=[],
                native_init=True, resume=checkpoint, output_dir=root/'new', seed=965140,
                max_updates=150, phase_relative_signal_gate=True, gate_panel=panel,
                wm_ckpt=[], wm_config=[])
            def load(path, **kwargs):
                return dict(update_count=191)
            with patch.object(policy14_recovery, 'checked_path', side_effect=Path), patch.object(torch, 'load', load):
                receipt = policy14_recovery.load_continuation(dict(trainer={}, reward={}),
                    args, initial_checkpoint=True)
                self.assertEqual(args.max_updates, 150)
                self.assertEqual(receipt['phase_start'], 191)
                self.assertEqual(receipt['saved_update'], 191)
                self.assertEqual(receipt['updates'], [])
                args.max_updates = 150
                with self.assertRaisesRegex(ValueError, 'progress/seed mismatch'):
                    policy14_recovery.load_continuation(dict(trainer={}, reward={}),
                        args, initial_checkpoint=False)

    def test_marker_wrapper_survives_release_without_guard(self):
        # The exact observed update1 failure mode: the persistent observational
        # wrapper must survive the replay controller release via its passthrough
        # marker, without any replay guard.
        import torch
        from tests.unit.test_longmem import trainer, observe, assert_tree_close
        from tools.candB.pilot_replay_guard import vjp_guard  # noqa: F401  (NOT used here)

        def core_factory(policy, group, safety, lane_record=None, **kwargs):
            def callback(state):
                obs = observe(state)
                action = policy(obs, group(obs, state))[0]
                accel = safety(state, action)
                speed = (state['v_mps'] + .5*accel).clamp_min(0.)
                return torch.stack((state['s_m']+.5*speed, state['lane'], speed, accel), -1)
            return callback

        def measured_factory():
            state = dict(calls=0)
            def factory(*args, **kwargs):
                callback = core_factory(*args, **kwargs)
                def measured(frame):
                    state['calls'] += 1
                    return callback(frame)
                measured._wms_checkpoint_passthrough = callback
                return measured
            return factory, lambda: state

        options = dict(future_samples=2, future_driver_scale=0., future_accel_noise=True)
        expected = trainer(1, 2, True, horizon=3, batch=2, **options)
        actual = trainer(1, 2, True, horizon=3, batch=2, **options)
        expected.controller_factory = core_factory
        wrapped, state = measured_factory()
        actual.controller_factory = wrapped
        import os as os_module
        with patch.dict(os_module.environ, WMS_SPEED_FUTURE_WORKERS='1', WMS_SPEED_PARALLEL_FORWARD='0'):
            for seed in (41, 42):
                torch.manual_seed(seed)
                left = expected.update()
                rng = torch.get_rng_state()
                torch.manual_seed(seed)
                right = actual.update()
                self.assertGreater(state()['calls'], 0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key in ('actor_loss', 'actor_grad_norm', 'critic_loss', 'critic_grad_norm'):
                    assert_tree_close(left[key], right[key], atol=0, rtol=0)
                for a, b in zip(expected.actor_parameters, actual.actor_parameters):
                    assert_tree_close(a, b, atol=0, rtol=0)
                    assert_tree_close(a.grad, b.grad, atol=0, rtol=0)
                assert_tree_close(expected.actor_optimizer.state_dict(),
                    actual.actor_optimizer.state_dict(), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
