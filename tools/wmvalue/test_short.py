import copy
import unittest

from .core import evaluate
from .evaluate_frozen import predictions
from .evaluate_short import evaluate_short
from .test_evaluate_frozen import fixtures


class ShortGateTests(unittest.TestCase):
    def test_prefix_and_long_checkpoints_compose_identically_without_a(self):
        tasks, data, models = fixtures()
        full = predictions(tasks, data, models)
        for i, d in enumerate(tasks.values()):
            d['source_sha256']['worker'] = 'old' if i % 2 else 'new'
            if i % 2:
                d['end_time'] = d['t0']+90
                d['results']['steps'] = 180
        short = predictions(tasks, data, models, short_only=True, allowed_workers={'old', 'new'})
        self.assertEqual(short, [{k: v for k, v in r.items() if k != 'A'} for r in full])
        self.assertTrue(all('A' not in r for r in short))
        first = next(iter(tasks.values()))
        first['source_sha256']['runtime'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'sources differ'):
            predictions(tasks, data, models, short_only=True, allowed_workers={'old', 'new'})

    def test_incomplete_and_unregistered_worker_rejected(self):
        tasks, data, models = fixtures()
        with self.assertRaisesRegex(ValueError, 'worker source'):
            predictions(tasks, data, models, short_only=True, allowed_workers={'registered'})
        tasks.pop(next(iter(tasks)))
        with self.assertRaisesRegex(ValueError, '320'):
            predictions(tasks, data, models, short_only=True)

    def test_amended_statistics_exactly_match_frozen_complete_evaluator(self):
        rows = predictions(*fixtures())
        for i, row in enumerate(rows):
            row['real'] = (i % 7)*100.+i
            row['B'] = row['real']*.9
            row['C'] = row['real']*1.05
            row['D'] = -row['real']
            row['E'] = i % 3
            row['A'] = row['real']  # A prevents full PASS, necessary condition survives
        full = evaluate(rows)
        short = evaluate_short(rows)
        self.assertEqual(short['verdict'], 'NEEDS_A')
        self.assertEqual(full['verdict'], 'FAIL')
        self.assertEqual(short['metrics'], {m: full['metrics'][m] for m in ('B', 'C', 'D', 'E')})
        self.assertEqual(short['contrasts'], {k: v for k, v in full['contrasts'].items()
                                            if not k.endswith('-A')})
        failed = copy.deepcopy(rows)
        for row in failed:
            row['B'] = row['C'] = row['D']
        self.assertEqual(evaluate_short(failed)['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
