"""CPU contracts for bound lean full-resume, original gate and phase budget."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.candB import pilot_continue, pilot_wmfast, pilot_wmfast_authority


class WMFastContinuationTests(unittest.TestCase):
    def test_skipped_rescore_preserves_last_actual_score_and_real_cli(self):
        from wms.control import policy14_cli as cli
        baseline = dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.)
        prior = dict(independent_score_before=baseline, updates=[dict(phase_update=1,
            independent_rescore_skipped=True)])
        self.assertEqual(pilot_wmfast.last_measured_score(prior), baseline)
        measured = dict(raw_dJ_vs_CTRL0_dense=-1., raw_dJ_vs_CTRL0_long=-2.)
        prior['updates'] += [dict(independent_J=measured),dict(independent_rescore_skipped=True)]
        self.assertEqual(pilot_wmfast.last_measured_score(prior), measured)
        patched = pilot_wmfast.continued_main(cli)
        self.assertIs(patched.__globals__, cli.__dict__)
        self.assertIsNot(patched, cli.main)

    def test_saved_initial_score_reused_without_forward(self):
        import os
        from types import SimpleNamespace
        import torch
        from tools.candB import pilot_lean_repaired, pilot_lean, pilot_lean_authority, pilot_authority
        from wms.control import policy14_cli as cli
        from wms.control.dreamer import DreamerConfig
        from wms.control.contract import RewardWeights
        saved = dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0., J=123.)
        doc = dict(input_sha256=dict(current_policy='init'), config_sha256='cfg',
            inputs=dict(current_policy='init.pt'))
        keys = ['src/wms/control/'+n+'.py' for n in ('dreamer', 'eslong', 'parallel_futures')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root/'old_runtime'
            (old/'authority_gen').mkdir(parents=True)
            (old/'authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json').write_text(json.dumps(doc))
            (old/'SOURCE_MANIFEST.json').write_text(json.dumps(dict.fromkeys(keys, 'old')))
            (root/'SOURCE_MANIFEST.json').write_text(json.dumps(dict.fromkeys(keys, 'new')))
            parent = root/'old_run'/'pilot_EXPLORATORY-LEAN'
            parent.mkdir(parents=True)
            (parent.parent/'OWNERSHIP.json').write_text(json.dumps(dict(runtime=str(old))))
            (parent/'RUN_SUMMARY.json').write_text(json.dumps(dict(label='EXPLORATORY-LEAN', status='failed',
                error='ValueError: native gate requires 16 finite paired costs', updates=[], seed=965140,
                exploratory_authority_sha256='digest', resumed_from=dict(sha256='init'), independent_score_before=saved)))
            output = root/'new_EXPLORATORY-LEAN'
            output.mkdir()
            payload = dict(policy={'residual_head.weight':torch.ones(1), 'residual_head.bias':torch.ones(1)})
            trainer = SimpleNamespace(update_count=191, config=DreamerConfig(), reward_weights=RewardWeights(),
                policy=SimpleNamespace(state_dict=lambda: {k:torch.zeros_like(v) for k,v in payload['policy'].items()}))
            def exercise(*args, **kwargs):
                self.assertEqual(cli.independent_score(trainer, None), saved)
                proof = json.loads((output/'pilot_STARTUP_REUSE_PROOF.json').read_text())
                self.assertTrue(proof['saved_independent_reused'])
                return 0
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(pilot_lean_authority, 'verify', return_value=doc), patch.object(
                        pilot_authority, 'sha256', return_value='digest'), patch.object(
                        pilot_authority, 'config', return_value=dict(trainer={}, reward={})), patch.object(
                        torch, 'load', return_value=payload), patch.object(pilot_lean, 'run', side_effect=exercise), patch.object(
                        cli, 'independent_score', side_effect=AssertionError('must not recompute saved initial score')):
                    self.assertEqual(pilot_lean_repaired.run('authority', str(output), exploratory=True,
                        failed_parent=parent), 0)
            finally:
                os.chdir(cwd)

    def test_actual_adapter_accepts_lean_and_preserves_clock(self):
        import torch
        frozen = dict(independent=dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.))
        prior = dict(label='EXPLORATORY-LEAN', status='failed', seed=965140,
            error='RuntimeError: own-run STOP requested; checkpoint saved',
            frozen_scoring_baseline=frozen, resumed_from=dict(update_count=191),
            updates=[dict(label='EXPLORATORY-LEAN', update=192, phase_update=1,
                startup_s=25., independent_J=frozen['independent'])])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save():
                (root/'RUN_SUMMARY.json').write_text(json.dumps(prior))
                (root/'frozen_scoring_baseline.json').write_text(json.dumps(frozen))
            save()
            with patch.object(pilot_continue, 'validate_parent', pilot_wmfast.validate_parent), patch.object(
                    torch, 'load', return_value=dict(update_count=192)):
                base, verify, receipt = pilot_continue.prepare_base(root)
                adapter = pilot_wmfast.lean_module(base, receipt)
            self.assertIs(adapter.base, base)
            self.assertIs(adapter.authority, pilot_wmfast_authority)
            self.assertEqual(pilot_wmfast.validate_parent(root)[1], 191)
            self.assertEqual(pilot_continue.continuation_args(['--new-phase', '--resume', 'old'], root),
                ['--resume', str(root/'policy_latest.pt'), '--continue-run', str(root)])
            prior['updates'][-1]['phase_update'] = 2
            save()
            with self.assertRaisesRegex(ValueError, 'clock'):
                pilot_wmfast.validate_parent(root)
            prior['updates'][-1]['phase_update'] = 1
            prior['updates'][-1]['R4'] = dict(passed=False)
            save()
            with self.assertRaisesRegex(ValueError, 'failed R4'):
                pilot_wmfast.validate_parent(root)

    def test_stack_environment_remains_explicit_and_memory_bounded(self):
        from tools.candB.pilot_lean import memory_bound
        env = pilot_wmfast_authority.ENV
        self.assertEqual(env['WMS_SPEED_ES_STACK_SIDES'], '2')
        self.assertEqual(env['WMS_SPEED_FORWARD_WORKERS'], '1')
        total = 49140*1024**2
        bound = memory_bound(total, 7*1024**3, 6*1024**3, 1, .50)
        self.assertLessEqual(bound, 45*1024**3)
        with self.assertRaises(ValueError):
            memory_bound(total, 7*1024**3, 6*1024**3, 2, .50)


if __name__ == '__main__':
    unittest.main()
