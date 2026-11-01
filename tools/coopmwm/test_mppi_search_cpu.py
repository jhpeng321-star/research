"""CPU-only tests for the policy-centred MPPI/CEM group search (no SUMO, no WM)."""
import inspect
import json
import unittest
from unittest.mock import patch
import numpy as np
from . import mppi_search as ms

BOUNDS = (-3., 1.5)
T = 300


def row(vid, s, lane, cav=True, v=20.):
    return dict(vid=vid, s_m=float(s), v_mps=v, lane=lane, is_cav=cav, length_m=5.)


def target(m):
    """Known smooth optimum away from the policy (0.2 m/s^2 constant)."""
    t = np.arange(T, dtype=float)
    out = np.full((m, T), .2)
    out[0] += -1.2*np.sin(np.pi*np.clip(t, 0, 120)/120)
    out[1] += .8*np.sin(np.pi*np.clip(t, 0, 120)/120)
    return out


class Fake:
    """Quadratic team cost of the group's acceleration path; paired future noise."""

    def __init__(self, m, H_cost=T):
        self.m, self.calls, self.seen = m, [], []
        self.policy = np.full((m, T), .2); self.mappo = np.zeros((m, T)); self.target = target(m)

    def cost(self, path):
        return float(np.square(path-self.target).sum())

    def __call__(self, record, actions, K, H):
        ids = [a['id'] for a in actions]
        self.calls.append((ids, K, H))
        rng = np.random.default_rng(7)
        common = rng.normal(0., 20., 64)  # identical futures for every action
        out = {}
        for a in actions:
            if a['id'] == 'policy':
                path = self.policy
            elif a['id'] == 'MAPPO':
                path = self.mappo
            else:
                self.seen.append(a)
                path = self.policy.copy(); D = a['duration_s']
                path[:, :D] = np.asarray(a['profile'])
            own = np.random.default_rng(int(ms._digest(a['id'])[:8], 16)).normal(0, 1., K)
            out[a['id']] = [dict(team_delay=self.cost(path)+common[k]+own[k]) for k in range(K)]
        return out


def group_of(m=2):
    rows = [row(f'c{i}', -200-40*i, i % 2) for i in range(m)]
    return ms.group_components(rows)[0]


def run(seed=1, m=2, **kw):
    fake = Fake(m)
    g = group_of(m)
    res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g,
                    fake.policy, BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=seed, **kw)
    return res, fake


class Groups(unittest.TestCase):
    def test_chain_all_lanes_ramp_and_hdvs_ignored(self):
        rows = [row('a', -500, 0), row('b', -450, 0), row('c', -380, 1), row('r', -300, 2),
                row('h', -200, 0, cav=False), row('d', -120, 2), row('e', -60, 2), row('lone', 200, 1)]
        gs = ms.group_components(rows)
        self.assertEqual([g['members'] for g in gs], [['e', 'd'], ['r', 'c', 'b', 'a']])
        self.assertEqual(gs[1]['lanes'], [0, 1, 2])
        self.assertEqual(ms.group_components(rows, exclude={'d'})[0]['members'], ['r', 'c', 'b', 'a'])
        json.dumps(gs)

    def test_more_than_six_prefers_lanes_then_tight_then_rechains(self):
        rows = [row(f'm{i}', -20*i, 0) for i in range(10)]   # 0..-180, one lane, 20 m apart
        rows[6]['lane'] = 1; rows[7]['lane'] = 2                # lanes vary only around m6/m7
        gs = ms.group_components(rows)
        first = next(g for g in gs if g['size'] == 6)
        self.assertEqual(first['lanes'], [0, 1, 2])
        self.assertTrue({'m6', 'm7'} <= set(first['members']))
        self.assertEqual(sorted(sum((g['members'] for g in gs), [])), sorted(set(sum((g['members'] for g in gs), []))))
        for g in gs:
            self.assertTrue(2 <= g['size'] <= 6)
        # Contiguous windows only: chosen members stay chained (gaps <= 100 m).
        s = sorted(r['s_m'] for r in first['rows'])
        self.assertTrue(max(np.diff(s)) <= 100)
        # Tightness tie-break when lanes are equal.
        rows = [row(f't{i}', x, 0) for i, x in enumerate([0, -90, -180, -190, -200, -210, -220, -230])]
        top = [g for g in ms.group_components(rows) if g['size'] == 6][0]
        self.assertEqual(top['members'], ['t2', 't3', 't4', 't5', 't6', 't7'])


class PlanSpace(unittest.TestCase):
    def test_knots_and_scaling(self):
        self.assertEqual([ms.knot_count(d) for d in (30, 60, 90, 120, 180, 210, 240, 270, 300)], [4, 7, 8, 8, 8, 8, 9, 10, 10])
        self.assertEqual(ms.scaling(2, 8), dict(dim=16, population=32, elites=8, finalists=8))
        self.assertEqual(ms.scaling(6, 10), dict(dim=60, population=120, elites=30, finalists=8))
        self.assertEqual(ms.allowed_durations(max_duration_s=120), [30, 60, 90, 120])
        self.assertEqual(ms.allowed_durations(), [30, 60, 90, 120, 150, 180, 240, 300])
        self.assertEqual(ms.canonical_knots(), 10)
        self.assertEqual([ms.final_horizon(d) for d in (30, 180, 240, 300, 600)], [300, 300, 360, 420, 600])
        for D in ms.allowed_durations():
            kt = ms.knot_times(D)
            self.assertLessEqual(len(kt), 10)
            self.assertEqual((kt[0], kt[-1]), (0., float(D)))

    def test_bounds_knots_comparators_and_no_family_seeds(self):
        with patch('tools.coopmwm.plans.family_seeds', side_effect=AssertionError('family seed used')):
            res, fake = run(m=3)
        self.assertTrue(fake.seen)
        for p in fake.seen:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
            self.assertLessEqual(len(p['knot_times_s']), 10)
            self.assertIn(p['origin'], ('centre', 'mappo', 'sampled'))
            self.assertEqual(p['family'], ms.PLAN_TYPE)
        for ids, K, H in fake.calls:
            self.assertEqual(ids[:2], ['policy', 'MAPPO'])
        Dfin = max(f['duration_s'] for f in res['finalists'])
        self.assertEqual([(K, H) for _, K, H in fake.calls], [(4, 300)]*3+[(8, min(600, max(300, Dfin+120)))])
        self.assertEqual(res['final']['H'], ms.final_horizon(Dfin))
        src = inspect.getsource(ms.search)+inspect.getsource(ms.sample_unscored)
        self.assertNotIn('family_seeds', src); self.assertNotIn('wmteach', inspect.getsource(ms))

    def test_profile_plan_executes_through_inherited_accessor(self):
        from .plans import acceleration_at
        res, _ = run()
        p = res['plan']
        self.assertEqual(acceleration_at(p, 5), {v: p['profile'][i][5] for i, v in enumerate(p['members'])})
        self.assertEqual(acceleration_at(p, p['duration_s']), {})


class Search(unittest.TestCase):
    def test_improves_over_policy_and_reports_topk(self):
        res, fake = run()
        self.assertTrue(res['accepted'])
        pol = res['comparator_means']['policy']
        best = min(f['saving_vs_policy'] for f in res['finalists'])
        self.assertGreater(max(f['saving_vs_policy'] for f in res['finalists']), .5*fake.cost(fake.policy)-fake.cost(fake.target))
        self.assertGreater(res['plan']['saving_vs_policy'], 0)
        self.assertTrue(np.isfinite(pol) and np.isfinite(best))
        self.assertAlmostEqual(res['centre_open_loop_gap'], 0., delta=3.)  # fake: centre == policy
        self.assertTrue(1 <= len(res['topk']) <= 4)
        for p in res['topk']:
            self.assertGreater(p['saving_vs_mappo'], 0)
        self.assertAlmostEqual(sum(res['probabilities'].values()), 1., places=12)
        self.assertLess(res['iterations'][-1]['mean_rel_cost'], res['iterations'][0]['mean_rel_cost'])
        json.dumps(res)

    def test_six_member_group_scales_and_improves(self):
        res, fake = run(m=6)
        self.assertEqual(res['scaling'], dict(dim=60, population=120, elites=30, finalists=8))
        self.assertEqual(res['candidates_searched'], 3*120)
        self.assertEqual(res['imagined_futures'], 3*122*4+10*8)
        self.assertTrue(res['accepted'])
        self.assertGreater(res['plan']['saving_vs_policy'], 0)
        self.assertEqual(res['plan']['origin'], 'sampled')

    def test_long_hold_library_reused_and_release_scored(self):
        fake = Fake(2)
        fake.target = fake.policy.copy()
        fake.target[0, :270] += -1.                     # a 270 s hold is optimal
        fake.target[1, :270] += .6
        g = group_of(2)
        # A finite stochastic search need not discover the global optimum on
        # every LAPACK implementation. Pin a previously verified long plan to
        # test the actual duration/library/release contract deterministically.
        verified=ms.build_plan(g,fake.policy,None,300,BOUNDS,'sampled','ordinary',absolute=fake.target)
        library=ms.PlanLibrary()
        library.add(ms.describe(g,'ordinary'),verified,fake.cost(fake.policy))
        res = ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS, fake, 'ordinary', seed=2,library=library)
        self.assertTrue(res['accepted'])
        self.assertGreaterEqual(res['plan']['duration_s'], 240,
            [(p['duration_s'],p['saving_vs_policy']) for p in res['finalists']])
        self.assertEqual(res['final']['H'], ms.final_horizon(max(f['duration_s'] for f in res['finalists'])))
        self.assertGreaterEqual(res['final']['H'], 360)
        self.assertGreater(res['plan']['saving_vs_policy'], .3*fake.cost(fake.policy))
        self.assertTrue(res['library_seeds'])
        with self.assertRaises(ValueError):           # trajectory must cover the 300 s maximum
            ms.search(dict(policy_residuals={}), g, fake.policy[:, :200], BOUNDS, fake, 'ordinary', seed=2)
        short = ms.search(dict(policy_residuals={}), g, fake.policy[:, :180], BOUNDS, fake, 'ordinary',
                          seed=2, config=dict(max_duration_s=180))
        self.assertEqual(short['scaling']['dim'], 16)
        self.assertEqual(short['screen']['H'], 180)

    def test_cem_variant(self):
        res, fake = run(config=dict(method='cem'))
        self.assertTrue(res['accepted'])
        self.assertGreater(res['plan']['saving_vs_policy'], 0)

    def test_determinism(self):
        a, _ = run(seed=3); b, _ = run(seed=3); c, _ = run(seed=4)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertNotEqual(json.dumps(a, sort_keys=True), json.dumps(c, sort_keys=True))

    def test_rejects_when_nothing_beats_mappo(self):
        fake = Fake(2); fake.mappo = fake.target.copy()  # MAPPO is already optimal
        g = group_of(2)
        res = ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS, fake, 'ordinary', seed=1)
        self.assertFalse(res['accepted']); self.assertIsNone(res['plan'])
        json.dumps(res)


class Thompson(unittest.TestCase):
    def test_probabilities_sum_to_one_and_favour_best(self):
        rng = np.random.default_rng(0); K = 8
        noise = rng.normal(0, 3, K)
        costs = dict(policy=40+noise, MAPPO=50+noise, A=10+noise+rng.normal(0, 1, K),
                     B=13+noise+rng.normal(0, 1, K), C=14+noise+rng.normal(0, 1, K), Z=60+noise)
        plans = [dict(id=i) for i in 'ABCZ']
        t = ms.thompson_topk(plans, costs, 3, np.random.default_rng(1))
        self.assertTrue(t['accepted'])
        self.assertEqual([c['id'] for c in t['candidates']], ['A', 'B', 'C'])
        self.assertNotIn('Z', t['probabilities'])
        self.assertAlmostEqual(sum(t['probabilities'].values()), 1., places=12)
        self.assertEqual(max(t['probabilities'], key=t['probabilities'].get), 'A')
        self.assertGreater(t['probabilities']['A'], .5)
        self.assertEqual(t['candidates'][0]['saving_vs_policy'], float((costs['policy']-costs['A']).mean()))


class Learned(unittest.TestCase):
    def test_covariance_update_shifts_sampling_toward_rewarded_direction(self):
        cm = ms.CovarianceModel(ms.default_sigma(BOUNDS))
        v = np.stack([np.ones(10), -np.ones(10)])          # brake/accelerate anti-correlated pair
        u = np.stack([np.ones(10), np.ones(10)])            # orthogonal joint direction
        def ratio(cov):
            a, b = v.reshape(-1), u.reshape(-1)
            return (a@cov@a/(a@a))/(b@cov@b/(b@b))
        before = ratio(cm.get('ordinary', 2))
        self.assertAlmostEqual(before, 1., places=9)       # isotropic across members at start
        for w in (1., 2., 1., 3.)*10:
            cm.update('ordinary', 1.5*v, w)
        after = ratio(cm.get('ordinary', 2))
        self.assertGreater(after, 3*before)
        self.assertAlmostEqual(ratio(cm.get('merge-dense', 2)), 1., places=9)  # per regime
        with self.assertRaises(ValueError):
            cm.update('ordinary', v, 0.)
        # Reward-unit invariance (Kish count).
        cm2 = ms.CovarianceModel(ms.default_sigma(BOUNDS))
        for w in (1., 2., 1., 3.)*10:
            cm2.update('ordinary', 1.5*v, 100*w)
        np.testing.assert_allclose(cm.get('ordinary', 2), cm2.get('ordinary', 2))
        # Samples follow: projection on v grows relative to u.
        g = group_of(2); P = np.zeros((2, T))
        def proj(model):
            a = []
            for s in range(200):
                p = ms.sample_unscored(g, P, (-50., 50.), 'ordinary', s, model)
                z = np.asarray(p['perturbation_canonical'])
                a.append((abs((z*v).sum()), abs((z*u).sum())))
            a = np.array(a); return a[:, 0].mean()/a[:, 1].mean()
        self.assertGreater(proj(cm), 1.5*proj(None))
        rt = ms.CovarianceModel.from_json(json.loads(json.dumps(cm.to_json())))
        np.testing.assert_allclose(rt.get('ordinary', 2), cm.get('ordinary', 2))
        self.assertGreater(cm.favoured_directions('ordinary', 2, 1)[0]['variance_ratio'], 1.)

    def test_library_seeding(self):
        fake = Fake(2); g = group_of(2); desc = ms.describe(g, 'ordinary')
        best = ms.build_plan(g, fake.policy, None, 120, BOUNDS, 'mappo', 'ordinary', absolute=fake.target)
        lib = ms.PlanLibrary()
        other = dict(desc, rel_s=[0., -400.], leader_s=900.)
        lib.add(other, ms.build_plan(g, fake.policy, np.zeros((2, 10)), 60, BOUNDS, 'centre', 'ordinary'), 5.)
        eid = lib.add(desc, best, 40.)
        lib.add(dict(desc, size=3), best, 9.)                     # different shape key: never matched
        with self.assertRaises(ValueError):
            lib.add(desc, best, -1.)
        self.assertEqual(lib.nearest(desc, 1)[0]['id'], eid)
        lib = ms.PlanLibrary.from_json(json.loads(json.dumps(lib.to_json())))
        res = ms.search(dict(policy_residuals={}), g, fake.policy, BOUNDS, fake, 'ordinary',
                        mappo_traj=fake.mappo, library=lib, seed=1, config=dict(library_seeds=1))
        self.assertEqual(res['library_seeds'], [eid])
        self.assertIn('library', res['origins_searched'])
        seeded = [p for p in fake.seen if p['origin'] == 'library']
        self.assertEqual(len({p['id'] for p in seeded}), 1)
        lib_cost = fake.cost(np.concatenate([np.asarray(seeded[0]['profile']), fake.policy[:, 120:]], 1))
        self.assertLess(lib_cost, .2*fake.cost(fake.policy))
        self.assertEqual(res['plan']['origin'], 'library')
        # learn() records a SUMO-positive plan; non-positive is ignored.
        cm = ms.CovarianceModel(1.); lib2 = ms.PlanLibrary()
        self.assertTrue(ms.learn(res['plan'], 12., desc, cm, lib2)['learned'])
        self.assertFalse(ms.learn(res['plan'], 0., desc, cm, lib2)['learned'])
        self.assertEqual(len(lib2.entries), 1)

    def test_posthoc_label_only(self):
        g = group_of(2); P = np.zeros((2, T))
        brake = np.zeros((2, T)); brake[0, :60] = -2.
        plan = ms.build_plan(g, P, None, 60, BOUNDS, 'mappo', 'ordinary', absolute=brake)
        protos = [dict(id='O1_s0', family='O1', members=g['members'], knots=[[-2.]*7, [0.]*7]),
                  dict(id='O2_s0', family='O2', members=g['members'], knots=[[1.]*7, [1.]*7])]
        self.assertEqual(ms.posthoc_label(plan, protos, P)['label'], 'O1')
        centre = ms.build_plan(g, P, np.zeros((2, 10)), 60, BOUNDS, 'centre', 'ordinary')
        self.assertEqual(ms.posthoc_label(centre, protos, P)['label'], 'new')


if __name__ == '__main__':
    unittest.main()
