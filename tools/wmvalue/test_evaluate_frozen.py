import unittest

import numpy as np

from .audit import ARMS
from .core import TIMES
from .evaluate_frozen import WM_SHA, predictions


class PredictOnly:
    def predict(self, x):
        return np.zeros(len(x))


def fixtures():
    tasks = {}
    cells = [(s, t, a) for s in range(969432, 969440) for t in TIMES for a in ARMS]
    data = dict(seed=np.array([s for s, _, _ in cells]), time_s=np.array([t for _, t, _ in cells]),
                arm=np.array([a for _, _, a in cells]), x=np.zeros((200, 111)),
                actions=np.zeros((200, 6)), y=np.zeros(200))
    for s in range(969432, 969440):
        for t in TIMES:
            for f in range(8):
                windows = {}
                for h in (60, 90):
                    x = np.zeros((5, 111))
                    x[:, -1] = 900-t-h
                    windows[str(h)] = dict(x=x.tolist(), J=[float(f)]*5)
                tasks[s, t, f] = dict(seed=s, t0=t, future=f, start_time=t, end_time=900,
                    arms=list(ARMS), K=8, mix=True, real_outcomes_opened=False,
                    wm=[dict(checkpoint_sha256=WM_SHA)], source_sha256={'same': 'runtime'},
                    results=dict(J=[float(f)]*5, steps=2*(900-t), windows=windows))
    value = type('Value', (), dict(model=PredictOnly()))()
    models = dict(V=value, D=value, E={(a, t): 1. for a in ARMS for t in TIMES})
    return tasks, data, models


class FrozenGateTests(unittest.TestCase):
    def test_no_partial_or_source_mixed_test(self):
        tasks, data, models = fixtures()
        last = tasks.pop((969439, 540, 7))
        with self.assertRaisesRegex(ValueError, '320'):
            predictions(tasks, data, models)
        tasks[969439, 540, 7] = last
        last['source_sha256'] = {'changed': 'runtime'}
        with self.assertRaisesRegex(ValueError, 'sources differ'):
            predictions(tasks, data, models)

    def test_wrong_endpoint_or_bootstrap_clock_rejected(self):
        tasks, data, models = fixtures()
        last = tasks[969439, 540, 7]
        last['end_time'] = 600
        with self.assertRaisesRegex(ValueError, 'clock mismatch'):
            predictions(tasks, data, models)
        last['end_time'] = 900
        last['results']['windows']['90']['x'][0][-1] = 900-540
        with self.assertRaisesRegex(ValueError, 'remaining-time'):
            predictions(tasks, data, models)

    def test_complete_composition_uses_all_eight_before_differencing(self):
        rows = predictions(*fixtures())
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(r['A'] == r['B'] == r['C'] == 3.5 for r in rows))
        self.assertTrue(all(r['D'] == 0. and r['E'] == 1. for r in rows))


if __name__ == '__main__':
    unittest.main()
