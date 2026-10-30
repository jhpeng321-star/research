"""CPU contract tests for ordinary same-phase continuation and baseline reuse."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.candB import pilot_continue


class ContinuationTest(unittest.TestCase):
    def test_phase_budget_does_not_restart_on_continuation(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        import torch
        from wms.control import policy14_recovery
        from wms.control.dreamer import DreamerConfig
        from wms.control.contract import RewardWeights
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'policy_latest.pt'
            checkpoint.write_bytes(b'owned synthetic checkpoint')
            panel = root / 'panel.json'
            panel.write_bytes(b'owned synthetic panel')
            prior = dict(status='failed', seed=965140, resumed_from=dict(update_count=190),
                updates=[dict(update=191)], wm_members=[],
                effective_config_before_updates=asdict(DreamerConfig()),
                effective_reward_before_updates=asdict(RewardWeights()),
                gate_panel=dict(sha256=hashlib.sha256(panel.read_bytes()).hexdigest()),
                native_curve=[dict(update=190, passed=True, mean_delta=0.)])
            (root / 'RUN_SUMMARY.json').write_text(json.dumps(prior))
            args = SimpleNamespace(continue_run=root, new_phase=False, resume_allow=[], native_init=True,
                resume=checkpoint, output_dir=root/'new', seed=965140, max_updates=150,
                phase_relative_signal_gate=True, gate_panel=panel, wm_ckpt=[], wm_config=[])
            def load(path, **kwargs):
                return dict(update_count=191 if Path(path)==checkpoint else 190)
            with patch.object(policy14_recovery, 'checked_path', side_effect=Path), patch.object(torch, 'load', load):
                receipt = policy14_recovery.load_continuation(dict(trainer={}, reward={}), args)
            self.assertEqual(args.max_updates, 149)
            self.assertEqual(receipt['phase_updates'], 150)
            self.assertEqual(receipt['phase_start'], 190)

    def test_nested_adam_head_state_comparison_detects_drift(self):
        import torch
        original = dict(state={0: dict(step=torch.tensor(1.), exp_avg=torch.tensor([.2]))})
        restored = dict(state={0: dict(step=torch.tensor(1.), exp_avg=torch.tensor([.2]))})
        self.assertTrue(pilot_continue.equal_state(original, restored))
        restored['state'][0]['exp_avg'][0] = .3
        self.assertFalse(pilot_continue.equal_state(original, restored))

    def test_config_allows_only_receipted_entropy_annealing(self):
        from dataclasses import asdict
        from wms.control.dreamer import DreamerConfig
        raw = dict(trainer={}, policy14=dict(independent_rescore=True,
            entropy_floor=.0001, entropy_decay_on_improvement=.9))
        saved = asdict(DreamerConfig())
        pilot_continue.verify_saved_config(saved, raw, dict(updates=[]))
        saved['future_samples'] += 1
        with self.assertRaisesRegex(ValueError, 'bound preset'):
            pilot_continue.verify_saved_config(saved, raw, dict(updates=[]))

    def test_ordinary_resume_not_fresh_phase(self):
        args = pilot_continue.continuation_args(['--resume', 'old', '--new-phase',
            '--resume-allow', 'trainer.foo', '--seed', '965140', '--max-updates', '150',
            '--prep-dry-run-updates', '2'], Path('/owned'))
        self.assertEqual(args, ['--resume', '/owned/policy_latest.pt', '--seed', '965140',
            '--max-updates', '150', '--continue-run', '/owned'])

    def test_exact_source_refusal(self):
        with self.assertRaises(ValueError):
            pilot_continue.replace_once('x x', 'x', 'y')

    def test_real_cli_bootstrap_patch_keeps_update_and_gate(self):
        from wms.control import policy14_cli as cli
        patched = pilot_continue.continued_main(cli)
        self.assertIsNot(patched, cli.main)
        self.assertIs(patched.__globals__, cli.__dict__)
        source = inspect.getsource(cli.main)
        self.assertIn('gate_position = trainer.update_count - native_start_update', source)
        self.assertIn("baseline = recovery['prior']['frozen_scoring_baseline']", source)

    def test_bound_parent_and_adapter_source(self):
        import torch
        frozen = dict(independent=dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.))
        prior = dict(label='EXPLORATORY', status='failed',
            error='RuntimeError: own-run STOP requested; checkpoint saved',
            frozen_scoring_baseline=frozen, resumed_from=dict(update_count=190),
            updates=[dict(label='EXPLORATORY', update=191, phase_update=1)])
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            (parent / 'RUN_SUMMARY.json').write_text(json.dumps(prior))
            (parent / 'frozen_scoring_baseline.json').write_text(json.dumps(frozen))
            with patch.object(torch, 'load', return_value=dict(update_count=191)):
                base, verify, actual = pilot_continue.prepare_base(parent)
                self.assertEqual(actual, prior)
                self.assertTrue(callable(base.run))
                self.assertTrue(callable(verify))
            prior['updates'][0]['phase_update'] = 2
            (parent / 'RUN_SUMMARY.json').write_text(json.dumps(prior))
            with self.assertRaisesRegex(ValueError, 'phase clock'):
                pilot_continue.validate_parent(parent)


if __name__ == '__main__':
    unittest.main()
