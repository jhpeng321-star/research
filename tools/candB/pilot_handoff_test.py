"""CPU-only meaningful checks for the one-update handoff and admission gates."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.candB.pilot_handoff import may_parallel, rows, validate_first_update, write
from tools.candB.pilot_recipe import parallel_requested


class HandoffTest(unittest.TestCase):
    def test_exclusive_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'OWNERSHIP.json'
            write(path, dict(pid=123))
            with self.assertRaises(FileExistsError):
                write(path, dict(pid=456))
            self.assertEqual(json.loads(path.read_text())['pid'], 123)

    def test_partial_row_is_not_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'updates.jsonl'
            path.write_text('{"phase_update":1}\n{"phase_update":2')
            self.assertEqual(rows(path), [dict(phase_update=1)])

    def test_zero_peers_and_capacity_required(self):
        sample = dict(total_MiB=49140, free_MiB=44000, processes=[])
        self.assertTrue(may_parallel(sample))
        sample['processes'] = [dict(pid=123)]
        self.assertFalse(may_parallel(sample))
        sample['free_MiB'] = 28000
        self.assertTrue(may_parallel(sample, 123))
        sample['free_MiB'] = 24000
        self.assertFalse(may_parallel(sample, 123))

    def test_default_serial_invalid_flag_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'TWO_FORWARD_WORKERS.json'
            self.assertFalse(parallel_requested(path))
            path.write_text('{"forward_workers":2}')
            with self.assertRaises(ValueError):
                parallel_requested(path)
            path.write_text('{"label":"EXPLORATORY","forward_workers":2}')
            self.assertTrue(parallel_requested(path))

    def test_finite_single_update_and_boundary_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            row = dict(phase_update=1, label='EXPLORATORY', actor_loss=0., critic_loss=0.,
                       actor_grad_norm=1., critic_grad_norm=0., independent_J=dict(
                           raw_dJ_vs_CTRL0_dense=-.1, raw_dJ_vs_CTRL0_long=-.2))
            (stage / 'updates.jsonl').write_text(json.dumps(row) + '\n')
            (stage / 'frozen_scoring_baseline.json').write_text(json.dumps(dict(independent=dict(
                raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.))))
            (stage / 'RUN_SUMMARY.json').write_text(json.dumps(dict(
                error='RuntimeError: own-run STOP requested; checkpoint saved')))
            (stage / 'policy_latest.pt').touch()
            self.assertEqual(validate_first_update(stage), row)
            row['actor_loss'] = float('nan')
            (stage / 'updates.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                validate_first_update(stage)
            row['actor_loss'] = 0.
            (stage / 'updates.jsonl').write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                validate_first_update(stage)


if __name__ == '__main__':
    unittest.main()
