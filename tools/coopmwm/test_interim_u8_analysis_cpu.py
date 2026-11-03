"""Interim analysis leakage/completeness guards and paired-statistic fixtures."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
try:
    import interim_u8_analysis as a
except ModuleNotFoundError:  # collection from the repository root
    from tools.coopmwm import interim_u8_analysis as a


class InterimAnalysisTests(unittest.TestCase):
    def test_incomplete_blocks_before_any_metadata_or_outcome_read(self):
        with tempfile.TemporaryDirectory() as d, patch.object(a, 'read') as read:
            with self.assertRaisesRegex(ValueError, 'not completely certified'):
                a.metadata(Path(d), [])
            read.assert_not_called()

    def test_final_checkpoint_episode_not_in_interim_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            record = dict(arm='B', world=dict(seed=989600), id='fixedfinal_B_989600', files={})
            with patch.object(a, 'sha') as sha, patch.object(a, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'outside fixed interim'):
                    a.load_metrics(Path(d), [record], reader=read)
                sha.assert_not_called();read.assert_not_called()

    def test_changed_fingerprint_blocks_outcome_decode(self):
        record = dict(arm='B', world=dict(seed=989600), id='interim008_B_989600',
                      files={'RESULT.json': dict(sha256='original')})
        with tempfile.TemporaryDirectory() as d, patch.object(a, 'sha', return_value='changed'), patch.object(a, 'read') as read:
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                a.load_metrics(Path(d), [record], reader=read)
            read.assert_not_called()

    def test_paired_scope_stats_and_no_duplicate_worlds(self):
        panel = [dict(seed=989600+i, cell=i//2, demand=2400 if i<2 else 1900,
                      penetration=.2 if i<2 else .05) for i in range(4)]
        records = []
        for arm in a.ARMS:
            for world in panel:
                metrics = {h: {m: (90 if arm=='B' else 100) if m=='team_delay_vs' else 0
                               for m in a.METRICS} for h in ('600', '900')}
                records.append(dict(arm=arm, world=world, metrics=metrics))
        out = a.analyze(records, panel)
        __import__('json').dumps(out, allow_nan=False)
        self.assertIsNone(out['results']['900:all:B-A:collisions']['pct_delay_reduction'])
        self.assertEqual(out['focus_worlds'], 2)
        self.assertEqual(out['worlds'], 4)
        for scope in ('all', 'focus'):
            x = out['results'][f'900:{scope}:B-A:team_delay_vs']
            self.assertEqual(x['mean_diff_vs'], -10)
            self.assertEqual(x['ci95'], [-10, -10])
            self.assertEqual(x['pct_delay_reduction'], 10)
            self.assertTrue(x['holm_within_endpoint_comparator']['reject'])
        with self.assertRaisesRegex(ValueError, 'complete unique'):
            a.analyze(records+[records[0]], panel)
        with self.assertRaisesRegex(ValueError, 'complete unique'):
            a.analyze(records[:-1], panel)


if __name__ == '__main__':unittest.main()
