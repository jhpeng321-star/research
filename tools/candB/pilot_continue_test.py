"""CPU contract tests for ordinary same-phase continuation and baseline reuse."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.candB import pilot_continue


class ContinuationTest(unittest.TestCase):
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
