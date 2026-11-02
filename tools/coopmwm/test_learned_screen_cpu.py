"""CPU tests for the learned screen (synthetic receipts; no SUMO/WM/GPU)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import learned_screen as ls
from . import mppi_search as ms
from . import plan_proposer as pp
from .test_mppi_search_cpu import BOUNDS, Fake, group_of

FAST = dict(steps=40, eval_every=20, d_model=32)


class LearnedScreen(unittest.TestCase):
    def test_screen_flag_skips_wm_screen_and_keeps_final(self):
        g = group_of(2); fake = Fake(2)
        rec = dict(state_id='s', rows=g['rows'], policy_residuals={v: 0. for v in g['members']})
        plain = ms.search(rec, g, fake.policy, BOUNDS, Fake(2), 'ordinary', seed=1)
        self.assertNotIn('learned_screen', plain)
        model = ls.ScreenNet(dict(ls.REGISTERED, d_model=32))
        kw = ls.learned_screen_kwargs(model, rec, g, 'ordinary')
        res = ms.search(rec, g, fake.policy, BOUNDS, fake, 'ordinary', seed=1, **kw)
        self.assertEqual(len(fake.calls), 1)                         # only the K8 final stage hit the WM
        self.assertEqual(fake.calls[0][1], ms.REGISTERED['final_K'])
        self.assertEqual(len(res['finalists']), 16); self.assertIn('learned_screen', res)
        self.assertEqual(res['screen'], plain['screen']); self.assertEqual(res['final']['K'], plain['final']['K'])
        with self.assertRaises(ValueError):
            ms.search(rec, g, fake.policy, BOUNDS, fake, 'ordinary', seed=1, screen=lambda r, gg, c: [0.])

    def test_units_training_determinism_and_checkpoint(self):
        rec = pp.synthetic_receipts(16, 4, seed=3, sizes=(2, 3), population=16)
        rec.append(pp.synthetic_receipts(1, 1, seed=4, topk_only=True)[0])
        U = ls.build_units(rec)
        self.assertEqual(len(U), 16)                                 # top-k-only receipt has no pool
        self.assertAlmostEqual(float(U[0]['y'].mean()), 0., places=9)
        m1, _ = ls.train(U, FAST); m2, _ = ls.train(U, FAST)
        self.assertEqual(pp.state_sha(m1), pp.state_sha(m2))
        with tempfile.TemporaryDirectory() as d:
            body = ls.save(m1, dict(note='t'), Path(d)/'ls')
            m3, meta = ls.load(Path(d)/'ls')
            self.assertEqual(meta['pt_sha256'], body['pt_sha256']); self.assertEqual(pp.state_sha(m1), pp.state_sha(m3))
        x = U[0]['x']
        a = ls.predict(m1, x, [np.zeros((U[0]['m'], ms.canonical_knots()))]*2, [300, 60])
        self.assertEqual(a.shape, (2,))

    def test_gold_lists_and_offline_check_structure(self):
        rec = pp.synthetic_receipts(24, 6, seed=5, sizes=(2,), population=16)
        pp.attach_verification(rec, pp.synthetic_verification(rec, seed=1, k=8))
        U = ls.build_units(rec)
        self.assertTrue(any(u['gold'] is not None for u in U))
        tr, va, vw = pp.world_split(U, .34, 1)
        model, _ = ls.train(tr, FAST)
        held = [r for r in rec if str(r['world_id']) in set(vw)][:2]
        res = ls.offline_check(model, held, dict(bootstraps=50))
        self.assertFalse(res['enough_data']); self.assertFalse(res['ready'])
        for k in ('P', 'S1', 'S2', 'coverage', 'diversity', 'wall'):
            self.assertIn(k, res)
        self.assertLess(res['wall']['futures_ls'], res['wall']['futures_plain'])


if __name__ == '__main__':
    unittest.main()
