"""CPU-only tests for the learned plan proposer (synthetic receipts; no SUMO, no WM, no GPU)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import mppi_search as ms
from . import plan_proposer as pp
from .test_mppi_search_cpu import BOUNDS, Fake, group_of

FAST = dict(steps=60, eval_every=20, d_model=32)


def dataset(n=48, worlds=8, seed=3, **kw):
    s, skipped = pp.build_dataset(pp.synthetic_receipts(n, worlds, seed=seed, population=16, **kw))
    assert not skipped, skipped
    return s


class Features(unittest.TestCase):
    def test_variable_sizes_forward_and_samples(self):
        S = dataset(20, 4)
        self.assertEqual(sorted({s['m'] for s in S}), [2, 3, 4, 5, 6])
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        ll, w = pp.log_likelihood(model, S)
        self.assertTrue(torch.isfinite(ll).all())
        for s in S[:5]:
            out = pp.sample(model, s['x'], 6, np.random.default_rng(0))
            self.assertEqual(len(out), 6)
            for z, D in out:
                self.assertEqual(z.shape, (s['m'], ms.canonical_knots()))
                self.assertIn(D, ms.allowed_durations())
                self.assertTrue((z[:, ~pp.knot_mask(D)] == 0).all())

    def test_only_observable_row_fields_are_read(self):
        r = pp.synthetic_receipts(1, 1, seed=5)[0]
        x = pp.member_features(r['rows'], r['members'], r['regime'])
        leaky = [dict(row, driver_params={'tau': 9.}, vtype='hdv_aggr', world_seed=123, demand=2700)
                 for row in r['rows']]
        np.testing.assert_array_equal(x, pp.member_features(leaky, r['members'], r['regime']))
        r2 = dict(r, world_id='OTHER', search_id='zzz')
        s1, _ = pp.build_search(r); s2, _ = pp.build_search(r2)
        np.testing.assert_array_equal(s1['x'], s2['x'])
        self.assertEqual(x.shape[1], pp.FEATURES)


class Permutation(unittest.TestCase):
    def test_likelihood_invariant_and_outputs_equivariant(self):
        torch.manual_seed(0)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        for p in model.parameters():           # non-trivial weights (heads start at zero)
            torch.nn.init.normal_(p, 0, .2)
        model.eval()
        s = next(s for s in dataset(10, 2) if s['m'] == 5 and s['targets'])
        perm = [3, 0, 4, 1, 2]
        sp = dict(s, x=s['x'][perm], targets=[dict(t, z=t['z'][perm]) for t in s['targets']])
        with torch.no_grad():
            a, _ = pp.log_likelihood(model, [s]); b, _ = pp.log_likelihood(model, [sp])
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-4)
            x1, m1 = pp._batch([s]); x2, m2 = pp._batch([sp])
            mu1 = model(x1, m1)[0][0, :5].numpy(); mu2 = model(x2, m2)[0][0, :5].numpy()
            np.testing.assert_allclose(mu1[perm], mu2, rtol=1e-4, atol=1e-5)
        # tokens themselves are a function of the member SET (order only permutes rows)
        r = pp.synthetic_receipts(5, 1, seed=9)[3]          # sizes cycle 2..6 -> 5 members
        self.assertEqual(len(r['members']), 5)
        mem = r['members']; pm = [mem[i] for i in [2, 0, 4, 1, 3]]
        np.testing.assert_allclose(pp.member_features(r['rows'], mem, r['regime'])[[2, 0, 4, 1, 3]],
                                   pp.member_features(r['rows'], pm, r['regime']))


class Learning(unittest.TestCase):
    def test_learns_planted_direction_and_beats_initial_on_heldout_worlds(self):
        rec = pp.synthetic_receipts(160, 16, seed=11, sizes=(2, 3), population=24)
        S, _ = pp.build_dataset(rec)
        tr, va, vw = pp.world_split(S, .25, 1)
        model, info = pp.train(tr, dict(steps=300, eval_every=50, d_model=32))
        res = pp.offline_check(model, va, dict(min_heldout_worlds=4, min_heldout_searches=20, bootstraps=300))
        self.assertGreater(res['loglik_gain']['mean'], 0)
        self.assertLess(res['nearest_elite_distance_change']['mean'], 0)
        self.assertGreater(res['loglik_gain']['ci95'][0], 0)
        self.assertTrue(res['passed'], res)
        # the most likely component points along the planted (braking) direction
        s = va[0]; z, D = pp.sample(model, s['x'], 1, np.random.default_rng(0))[0]
        planted, _ = pp.planted_direction(s['x'], s['regime'])
        self.assertGreater(float((z*planted).sum()), 0)

    def test_untrained_model_does_not_pass(self):
        S = dataset(60, 12)
        _, va, _ = pp.world_split(S, .5, 1)
        res = pp.offline_check(pp.ProposerNet(dict(pp.REGISTERED, d_model=32)), va,
                               dict(min_heldout_worlds=4, min_heldout_searches=10, bootstraps=200))
        self.assertFalse(res['passed'])

    def test_too_little_heldout_data_never_passes(self):
        S = dataset(24, 6)
        _, va, _ = pp.world_split(S, .34, 1)
        model, _ = pp.train([s for s in S if s not in va], FAST)
        res = pp.offline_check(model, va, dict(bootstraps=100))
        self.assertFalse(res['enough_data']); self.assertFalse(res['passed'])


class SplitAndDeterminism(unittest.TestCase):
    def test_world_grouped_split(self):
        S = dataset(60, 10)
        tr, va, vw = pp.world_split(S)
        self.assertFalse({s['world'] for s in tr} & {s['world'] for s in va})
        self.assertEqual(set(vw), {s['world'] for s in va})
        self.assertEqual(len(vw), 2)                        # round(.25 * 10) -> 2 (banker's), deterministic
        self.assertEqual(vw, pp.world_split(list(reversed(S)))[2])
        tr2, va2, _ = pp.world_split(S, seed=99)
        self.assertFalse({s['world'] for s in tr2} & {s['world'] for s in va2})
        self.assertEqual(len(tr2)+len(va2), len(S))

    def test_training_is_deterministic(self):
        S = dataset(30, 6)
        m1, i1 = pp.train(S, FAST); m2, i2 = pp.train(S, FAST)
        self.assertEqual(pp.state_sha(m1), pp.state_sha(m2))
        self.assertEqual(i1['log'], i2['log'])
        a = pp.sample(m1, S[0]['x'], 8, np.random.default_rng(4)); b = pp.sample(m2, S[0]['x'], 8, np.random.default_rng(4))
        for (za, da), (zb, db) in zip(a, b):
            np.testing.assert_array_equal(za, zb); self.assertEqual(da, db)

    def test_checkpoint_sha_roundtrip_and_tamper(self):
        S = dataset(20, 4)
        model, info = pp.train(S, FAST)
        with tempfile.TemporaryDirectory() as d:
            body = pp.save_checkpoint(model, dict(note='t'), Path(d)/'prop')
            m2, meta = pp.load_checkpoint(Path(d)/'prop')
            self.assertEqual(pp.state_sha(model), pp.state_sha(m2)); self.assertEqual(meta['pt_sha256'], body['pt_sha256'])
            raw = bytearray((Path(d)/'prop.pt').read_bytes()); raw[-20] ^= 1
            (Path(d)/'prop.pt').write_bytes(bytes(raw))
            with self.assertRaises(ValueError):
                pp.load_checkpoint(Path(d)/'prop')


class SchemaTolerance(unittest.TestCase):
    def test_missing_and_bad_fields(self):
        base = pp.synthetic_receipts(1, 1, seed=2, sizes=(3,))[0]
        cases = {
            'no_candidates': dict(base, candidates=None),
            'no_initial_or_canonical': {k: v for k, v in base.items() if k not in ('initial', 'canonical', 'durations_s')},
            'no_topk_no_executed': {k: v for k, v in base.items() if k not in ('topk', 'executed')},
            'no_regime': {k: v for k, v in base.items() if k != 'regime'},
            'bad_candidate_rows': dict(base, candidates=[{'id': 'x'}, {'id': 'y', 'perturbation_canonical': [[np.nan]*10]*3,
                                                          'duration_s': 60, 'saving_vs_policy': 1.}]
                                      + base['candidates']),
        }
        for name, r in cases.items():
            s, why = pp.build_search(copy.deepcopy(r))
            self.assertIsNotNone(s, (name, why))
        self.assertEqual(pp.build_search(dict(base, candidates=None))[0]['targets'], [])
        for bad in (dict(base, arm='R'), dict(base, members=base['members'][:1]), dict(base, rows=None),
                    dict(base, canonical=dict(n_knots=8, span_s=180))):
            s, why = pp.build_search(bad)
            self.assertIsNone(s); self.assertTrue(why)

    def test_sumo_weighting(self):
        base = pp.synthetic_receipts(1, 1, seed=4, sizes=(2,))[0]
        pid = base['topk'][0]
        pos, _ = pp.build_search(dict(base, executed=dict(plan_id=pid, sumo_realized_advantage=2.)))
        neg, _ = pp.build_search(dict(base, executed=dict(plan_id=pid, sumo_realized_advantage=-1.)))
        none, _ = pp.build_search(dict(base, executed=None))
        self.assertEqual([t['source'] for t in pos['targets']].count('sumo'), 1)
        self.assertAlmostEqual(sum(t['w'] for t in pos['targets'] if t['source'] == 'wm'), 1.)
        self.assertEqual(len(neg['targets']), len(none['targets'])-1)        # SUMO-losing plan never imitated
        self.assertFalse(any(t['source'] == 'sumo' for t in neg['targets']))
        r = pp.attach_sumo([copy.deepcopy(base)], [dict(plan=pid, sumo_realized_advantage=.7)])[0]
        self.assertEqual(r['executed'], dict(plan_id=pid, sumo_realized_advantage=.7))

    def test_service_adapter_and_queue_loader(self):
        res, fake = _search_with_recorder()[:2]
        g = group_of(2)
        request = dict(state_id='s', world_id=17, rows=g['rows'], group=g['members'],
                       search_learning=dict(covariance=ms.CovarianceModel(.5).to_json(), library={'entries': []}))
        rec, why = pp.receipt_from_service(request, dict(key='s', result=dict(res, record_sha256='abc')))
        self.assertIsNone(why); self.assertEqual(rec['world_id'], 17)
        s, _ = pp.build_search(rec)
        self.assertTrue(s['elites']); self.assertEqual(s['cov0'].shape, (20, 20))
        self.assertIsNone(pp.receipt_from_service(request, dict(accepted=False, plan=None, topk=None))[0])
        self.assertIsNone(pp.receipt_from_service(request, dict(accepted=True, random_match={'path': 'x'}))[0])
        with tempfile.TemporaryDirectory() as d:
            for sub in ('requests', 'responses'):
                (Path(d)/sub).mkdir()
            (Path(d)/'requests'/'a.json').write_text(json.dumps(request))
            (Path(d)/'responses'/'a.json').write_text(json.dumps(dict(res, record_sha256='abc')))
            (Path(d)/'requests'/'b.json').write_text(json.dumps(request))     # no response yet
            got, skipped = pp.load_receipts([d])
            self.assertEqual(len(got), 1); self.assertEqual(len(skipped), 1)


def _search_with_recorder(extra_seeds=None):
    fake = Fake(2); g = group_of(2)
    recorder = pp.RecordingEvaluator(fake)
    res = ms.search(dict(state_id='s', world_id=1, policy_residuals={v: 0. for v in g['members']}), g,
                    fake.policy, BOUNDS, recorder, 'ordinary', mappo_traj=fake.mappo, seed=1, extra_seeds=extra_seeds)
    return res, fake, recorder, g


class Integration(unittest.TestCase):
    def test_recording_evaluator_keeps_full_pool_and_search_unchanged(self):
        res, fake, recorder, g = _search_with_recorder()
        plain = ms.search(dict(state_id='s', world_id=1, policy_residuals={v: 0. for v in g['members']}), g,
                          fake.policy, BOUNDS, Fake(2), 'ordinary', mappo_traj=fake.mappo, seed=1)
        self.assertEqual(json.dumps(res, sort_keys=True), json.dumps(plain, sort_keys=True))
        self.assertNotIn('extra_seeds', plain)
        rec = pp.receipt_from_search(dict(state_id='s', world_id=1, rows=g['rows'], group=g['members']), res, recorder)
        self.assertEqual(len([c for c in rec['candidates'] if c['stage'] == 'screen']), res['candidates_searched'])
        s, why = pp.build_search(rec)
        self.assertIsNone(why); self.assertTrue(s['targets']); self.assertTrue(s['elites'])

    def test_proposal_seeds_respect_bounds_knots_and_feed_search(self):
        g = group_of(3); fake = Fake(3)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        with torch.no_grad():                            # extreme proposals must still be clipped
            model.member.bias.fill_(8.)
        record = dict(state_id='q', rows=g['rows'], group=g['members'])
        seeds = pp.proposal_seeds(record, g, fake.policy, 6, model, BOUNDS, regime='ordinary')
        again = pp.proposal_seeds(record, g, fake.policy, 6, model, BOUNDS, regime='ordinary')
        self.assertEqual([p['id'] for p in seeds], [p['id'] for p in again])
        self.assertEqual(len(seeds), 6)
        for p in seeds:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
            self.assertEqual(p['origin'], 'proposer')
            self.assertLessEqual(len(p['knot_times_s']), ms.REGISTERED['long_max_knots'])
            a = np.asarray(p['profile'])
            self.assertTrue((a >= BOUNDS[0]-1e-9).all() and (a <= BOUNDS[1]+1e-9).all())
            self.assertEqual(np.asarray(p['perturbation_canonical']).shape, (3, ms.canonical_knots()))
        res = ms.search(dict(state_id='q', policy_residuals={v: 0. for v in g['members']}), g, fake.policy, BOUNDS,
                        fake, 'ordinary', mappo_traj=fake.mappo, seed=2, extra_seeds=seeds)
        self.assertIn('proposer', res['origins_searched']); self.assertEqual(res['extra_seeds'], 6)
        scored = [a for a in fake.seen if a['origin'] == 'proposer']
        self.assertEqual(len(scored), 6)                  # every seed scored by the (fake) WM
        self.assertEqual(pp.proposal_seeds(record, g, fake.policy, 0, model, BOUNDS), [])


if __name__ == '__main__':
    unittest.main()


class WarmStart(unittest.TestCase):
    def _run(self, **kw):
        fake = Fake(3); g = group_of(3)
        res = ms.search(dict(state_id='w', policy_residuals={v: 0. for v in g['members']}), g, fake.policy, BOUNDS,
                        fake, 'ordinary', mappo_traj=fake.mappo, seed=5, **kw)
        return res, fake, g

    def test_flag_off_is_byte_identical_and_prior_warm_start_reproduces_search(self):
        plain, _, g = self._run()
        self.assertNotIn('warm_start', plain)
        prior = dict(mean=np.zeros(3*ms.canonical_knots()).tolist(),
                     cov=ms.isotropic_covariance(3, ms.default_sigma(BOUNDS)).tolist(),
                     duration_probs=[1.]*len(ms.allowed_durations()), generic_share=0.)
        warm, _, _ = self._run(warm_start=prior)
        self.assertIn('warm_start', warm)
        warm.pop('warm_start')
        self.assertEqual(json.dumps(plain, sort_keys=True), json.dumps(warm, sort_keys=True))

    def test_proposer_warm_start_runs_and_validates(self):
        g = group_of(3); fake = Fake(3)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        ws = pp.warm_start(dict(state_id='w', rows=g['rows']), g, model, 'ordinary')
        self.assertEqual(len(ws['mean']), 3*ms.canonical_knots())
        self.assertAlmostEqual(sum(ws['duration_probs']), 1.)
        res, fake, _ = self._run(warm_start=ws)
        self.assertEqual(res['warm_start']['source'], 'plan_proposer')
        for p in fake.seen:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
        with self.assertRaises(ValueError):
            self._run(warm_start=dict(ws, mean=ws['mean'][:-1]))
        with self.assertRaises(ValueError):
            self._run(warm_start=dict(ws, cov=(-np.eye(30)).tolist()))

    def test_fully_defensive_moments_equal_the_prior(self):
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32, defensive=1.))
        x = dataset(5, 1)[2]['x']; m = x.shape[0]
        mean, cov, pd = pp.mixture_moments(model, x)
        np.testing.assert_allclose(mean, 0, atol=1e-12)
        np.testing.assert_allclose(cov, ms.isotropic_covariance(m, pp.Z_SCALE), atol=1e-5)
        np.testing.assert_allclose(pd, 1/len(ms.allowed_durations()))


class NoveltyAndEntropy(unittest.TestCase):
    def test_novelty_slice_and_generic_share_counts(self):
        g = group_of(3); fake = Fake(3)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        rec = dict(state_id='n', rows=g['rows'], policy_residuals={v: 0. for v in g['members']})
        ws = pp.warm_start(rec, g, model, 'ordinary')
        seen = []
        inner = ws['novelty_score']

        def spy(Z, D):
            seen.append(len(Z))
            return inner(Z, D)
        ws['novelty_score'] = spy
        res = ms.search(rec, g, fake.policy, BOUNDS, fake, 'ordinary', seed=3, warm_start=ws)
        N = ms.scaling(3, ms.canonical_knots())['population']
        self.assertEqual(res['warm_start']['generic_per_iteration'], int(np.ceil(.2*N)))
        self.assertEqual(res['warm_start']['novelty_per_iteration'], int(np.ceil(.1*N)))
        self.assertEqual(seen, [int(np.ceil(.1*N))*4]*ms.REGISTERED['iterations'])
        self.assertEqual(res['candidates_searched'], N*ms.REGISTERED['iterations'])
        with self.assertRaises(ValueError):
            ms.search(rec, g, fake.policy, BOUNDS, fake, 'ordinary', seed=3, warm_start=dict(ws, generic_share=.95))

    def test_novelty_scorer_ranks_proposer_modes_as_least_novel(self):
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        n = ms.canonical_knots()
        with torch.no_grad():
            b = model.member.bias.view(4, 2, n)
            b[:, 0].fill_(-2.)        # every component mean = -2 (Z units) = -1 m/s^2
            b[:, 1].fill_(-3.)        # tight
        s = dataset(5, 1)[0]; m = s['m']
        score = pp.novelty_scorer(model, s['x'])
        mode = np.full((m, n), -1.); far = np.full((m, n), 1.)
        a, b = score([mode, far], [300, 300])
        self.assertGreater(a, b)                                     # the far draw is "more novel"

    def test_entropy_floor_penalises_collapsed_mixture(self):
        S = [s for s in dataset(10, 2) if s['targets']]
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        with torch.no_grad():
            model.mix.bias.copy_(torch.tensor([30., 0., 0., 0.]))
            model.mix.weight.zero_()
            plain = float(pp._weighted_nll(model, S)); floored = float(pp._weighted_nll(model, S, entropy=True))
        self.assertAlmostEqual(floored-plain, pp.REGISTERED['entropy_floor'], places=3)
        model2 = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        with torch.no_grad():
            model2.mix.weight.zero_(); model2.mix.bias.zero_()          # uniform mixture: H = log 4 > floor
            self.assertAlmostEqual(float(pp._weighted_nll(model2, S, entropy=True)),
                                   float(pp._weighted_nll(model2, S)), places=5)

    def test_unaccepted_full_pool_searches_are_training_data(self):
        r = pp.synthetic_receipts(1, 1, seed=8, sizes=(3,), null_frac=1.)[0]
        s, why = pp.build_search(r)
        self.assertIsNone(why); self.assertTrue(s['full_pool'])
        self.assertTrue(s['targets']); self.assertTrue(all(t['w'] >= 0 for t in s['targets']))
        self.assertAlmostEqual(sum(t['w'] for t in s['targets'] if t['source'] == 'wm'), 1.)

    def test_plan_novelty_and_report(self):
        g = group_of(2); fake = Fake(2)
        model = pp.ProposerNet(dict(pp.REGISTERED, d_model=32))
        res = ms.search(dict(state_id='p', policy_residuals={v: 0. for v in g['members']}), g, fake.policy, BOUNDS,
                        fake, 'ordinary', mappo_traj=fake.mappo, seed=1)
        lib = ms.PlanLibrary(); lib.add(ms.describe(g, 'ordinary'), res['plan'], 1.)
        row = pp.plan_novelty(res['plan'], model, g['rows'], 'ordinary', lib, dict(label='new'))
        self.assertAlmostEqual(row['library_distance'], 0.)
        self.assertGreaterEqual(row['proposer_mode_distance'], 0.)
        rep = pp.novelty_report([row, dict(row, posthoc_label='O1', origin='proposer')])
        self.assertEqual(rep['label_counts'], {'O1': 1, 'new': 1}); self.assertEqual(rep['new_fraction'], .5)
