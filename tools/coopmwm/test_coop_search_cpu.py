"""CPU tests for option ``coop_objective`` (synergy score, diversity, novelty, cost cap).

Real torch on CPU with a synthetic WM stub; no SUMO, no WM checkpoint, no GPU job.
"""
import json
import unittest
import numpy as np
import torch
from . import mppi_search as ms
from . import coop_search as cs
from .test_mppi_search_cpu import BOUNDS, Fake, T, row

LIVE = dict(iterations=2, screen_K=2, top_k=8, finalists_per_k=1)   # live label-list8 shape


def group(m, lanes=None):
    lanes = lanes or [i % 2 for i in range(m)]
    return ms.group_components([row(f'c{i}', -200-40*i, lanes[i]) for i in range(m)])[0]


class JointStub:
    """Synthetic WM in torch. Team cost of an action = function of each member's mean deviation
    from the policy over the first ``window`` s. Non-listed members follow the policy, so a solo
    plan moves exactly one member (the 'policy continues' rule). Futures: one common noise per k
    (paired, cancels in every gain). ``solo_w``: gain when member 0 brakes alone (the single-CAV
    trap); ``joint_w``: gain only when member 0 brakes AND member 1 accelerates."""

    def __init__(self, g, solo_w=0., joint_w=40., window=60, solo_member=0, solo_sign=-1.):
        self.members = list(g['members']); m = len(self.members)
        self.policy = np.full((m, T), .2); self.mappo = np.zeros((m, T))
        self.solo_w, self.joint_w, self.window = solo_w, joint_w, window
        self.solo_member, self.solo_sign = solo_member, solo_sign
        self.calls = []

    def path(self, a):
        if a['id'] == 'MAPPO':
            return self.mappo.copy()
        out = self.policy.copy()
        if a['id'] == 'policy':
            return out
        idx = {v: i for i, v in enumerate(self.members)}
        for v, r in zip(a['members'], a['profile']):
            out[idx[v], :a['duration_s']] = r
        return out

    def cost(self, paths):
        x = torch.as_tensor(np.stack(paths), dtype=torch.float64)
        dev = (x-torch.as_tensor(self.policy))[:, :, :self.window].mean(-1)
        brake0, acc1 = torch.relu(-dev[:, 0]), torch.relu(dev[:, 1])
        solo = torch.relu(self.solo_sign*dev[:, self.solo_member])
        return 100.-self.solo_w*solo-self.joint_w*brake0*acc1

    def __call__(self, record, actions, K, H):
        self.calls.append(dict(ids=[a['id'] for a in actions], K=K, H=H,
                               solos=[a for a in actions if 'solo_of' in a]))
        c = self.cost([self.path(a) for a in actions])
        common = np.random.default_rng(3).normal(0., 5., K)
        return {a['id']: [dict(team_delay=float(c[i])+float(common[k])) for k in range(K)]
                for i, a in enumerate(actions)}


def coop_search_run(stub, g, seed=1, **cfg):
    rec = dict(state_id='s', policy_residuals={v: 0. for v in g['members']})
    return ms.search(rec, g, stub.policy, BOUNDS, stub, 'ordinary', mappo_traj=stub.mappo, seed=seed, config=cfg)


def manual_plan(g, P, delta_rows, duration=60, tag='x'):
    """Plan whose per-second deviation is constant ``delta_rows[i]`` for member i."""
    m = len(g['members']); nc = ms.canonical_knots()
    z = np.repeat(np.asarray(delta_rows, float)[:, None], nc, 1)
    return ms.build_plan(g, P, z, duration, BOUNDS, 'sampled', 'ordinary', tag)


class OffIdentical(unittest.TestCase):
    def _run(self, config, niches=1, m=2):
        fake = Fake(m); g = group(m); cfg = dict(config or {})
        if niches > 1:
            cfg['niches'] = niches
        res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g, fake.policy,
                        BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=5, config=cfg)
        return json.dumps(res, sort_keys=True), fake.calls

    def test_off_is_byte_identical_single_and_niche(self):
        for niches in (1, 4):
            ref, calls = self._run(None, niches)
            for off in (dict(coop_objective=False), dict(coop_objective=False, coop_lambda=3., coop_solo_cap=2)):
                out, calls2 = self._run(off, niches)
                self.assertEqual(out, ref)
                self.assertEqual(calls2, calls)
            self.assertNotIn('coop', ref)

    def test_unregistered_coop_key_rejected(self):
        with self.assertRaises(ValueError):
            self._run(dict(coop_objective=True, coop_bogus=1))


class Synergy(unittest.TestCase):
    def test_only_joint_action_helps(self):
        g = group(2); stub = JointStub(g)
        P = stub.policy
        joint = manual_plan(g, P, [-1., 1.], tag='J')     # member0 brakes, member1 accelerates
        single = manual_plan(g, P, [-1., 0.], tag='S')    # only member0 brakes
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)))
        base = ms.comparators(dict(policy_residuals={}))
        solos = cs.solos_for([joint, single])
        self.assertEqual([s['members'] for s in solos], [['c0'], ['c1'], ['c0'], ['c1']])
        scores = stub(dict(), base+[joint, single]+solos, 4, 300)
        rel, extra = cs.screen(dict(), stub, base, [joint, single], scores, 4, 300, ctx, 'screen0')
        self.assertEqual(extra, 0)                         # cap 16 >= 2: solos batched in the call
        info = ctx.info['screen0']
        expected_g = 40.*1.*1.                             # product of the two deviations
        self.assertAlmostEqual(info[joint['id']]['group_gain'], expected_g, places=9)
        self.assertEqual(info[joint['id']]['solo_gains'], {'c0': 0., 'c1': 0.})
        self.assertAlmostEqual(info[joint['id']]['synergy'], expected_g, places=9)
        self.assertAlmostEqual(info[joint['id']]['score'], 2*expected_g, places=9)   # lambda 1
        self.assertAlmostEqual(info[single['id']]['group_gain'], 0., places=9)
        self.assertAlmostEqual(info[single['id']]['synergy'], 0., places=9)
        np.testing.assert_allclose(rel, [-2*expected_g, 0.], atol=1e-9)

    def test_single_cav_trap_scores_zero_synergy_and_lambda_scales(self):
        g = group(2); stub = JointStub(g, solo_w=30., joint_w=0.)
        single = manual_plan(g, stub.policy, [-1., 0.], tag='S')
        both = manual_plan(g, stub.policy, [-1., .5], tag='B')
        for lam in (0., 1., 2.5):
            ctx = cs.Context(g, cs.split_config(dict(coop_objective=True, coop_lambda=lam)))
            base = ms.comparators(dict(policy_residuals={}))
            scores = stub({}, base+[single, both]+cs.solos_for([single, both]), 2, 300)
            cs.screen({}, stub, base, [single, both], scores, 2, 300, ctx, 's')
            for p in (single, both):
                i = ctx.info['s'][p['id']]
                self.assertAlmostEqual(i['group_gain'], 30., places=9)
                self.assertAlmostEqual(i['best_solo_gain'], 30., places=9)
                self.assertAlmostEqual(i['synergy'], 0., places=9)
                self.assertAlmostEqual(i['score'], 30., places=9)

    def test_search_reports_and_prefers_synergy(self):
        g = group(2)
        # Trap: member 1 BRAKING alone helps (single-CAV win) but conflicts with the joint move
        # (member 0 brakes AND member 1 accelerates).
        trap = dict(solo_w=20., joint_w=60., solo_member=1)
        def true_synergy(stub, plans, reduce=np.mean):
            out = []
            for p in plans:
                c = stub.cost([stub.path(dict(id='policy')), stub.path(p)]+[stub.path(s) for s in cs.solos_for([p])])
                gain = float(c[0]-c[1]); out.append(gain-max(float(c[0]-x) for x in c[2:]))
            return float(reduce(out))
        def finals(stub, res):
            plans = [p for p in _scored_plans(stub) if p['id'] in {f['id'] for f in res['finalists']}]
            self.assertEqual(len(plans), len(res['finalists']))
            return plans
        for seed in (1, 2):
            stub_off = JointStub(g, **trap); off = coop_search_run(stub_off, g, seed=seed, **LIVE)
            stub_on = JointStub(g, **trap); on = coop_search_run(stub_on, g, seed=seed, coop_objective=True, **LIVE)
            stub_nd = JointStub(g, **trap)
            nodiv = coop_search_run(stub_nd, g, seed=seed, coop_objective=True, coop_diversity=False, **LIVE)
            # Synergy scoring (diversity off): the eight finalists carry more true cooperation.
            self.assertGreater(true_synergy(stub_nd, finals(stub_nd, nodiv)),
                               true_synergy(stub_off, finals(stub_off, off)))
            # Diversity keeps the top-2 coop plans (same search pool) and spreads the rest.
            self.assertEqual(true_synergy(stub_on, finals(stub_on, on), max),
                             true_synergy(stub_nd, finals(stub_nd, nodiv), max))
            self.assertGreater(on['coop_objective']['finalists_report']['mean_pairwise_descriptor_distance'],
                               nodiv['coop_objective']['finalists_report']['mean_pairwise_descriptor_distance'])
        self.assertNotIn('coop_objective', off)
        rep = on['coop_objective']['finalists_report']
        self.assertEqual(rep['n'], len(on['finalists']))
        for f in on['finalists']:
            c = f['coop']
            self.assertAlmostEqual(c['synergy'], c['group_gain']-c['best_solo_gain'], places=9)
            self.assertAlmostEqual(c['score'], c['group_gain']+c['synergy'], places=9)
        for p in rep['plans']:
            self.assertIsNotNone(p['synergy'])
        self.assertGreaterEqual(rep['distinct_sign_patterns'], 1)
        self.assertGreater(rep['mean_pairwise_descriptor_distance'], 0.)
        json.dumps(on)


def _scored_plans(stub):
    return list({a['id']: a for a in stub.seen}.values())


_orig_call = JointStub.__call__


def _recording_call(self, record, actions, K, H):
    self.__dict__.setdefault('seen', []).extend(a for a in actions if 'profile' in a and 'solo_of' not in a)
    return _orig_call(self, record, actions, K, H)


JointStub.__call__ = _recording_call


class Diversity(unittest.TestCase):
    def test_spreads_and_keeps_top2(self):
        g = group(3); P = np.full((3, T), .2)
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        items = []
        # The best scores: a tight cluster of near-duplicates (all members slower) ...
        for i in range(3):
            p = manual_plan(g, P, [-.8+.001*i, -.8, -.8], tag=f'k{i}'); items.append((p['id'], 10.-.01*i, p))
        # ... distinct ones at the median score: different participants / magnitudes ...
        for i, d in enumerate(([-.8, 0., 0.], [0., 0., -1.2], [0., -.5, 0.], [-1.5, -1.5, 0.])):
            p = manual_plan(g, P, d, tag=f'd{i}'); items.append((p['id'], 9., p))
        # ... and below-median plans that are even more distinct.
        for i, d in enumerate(([1., 1., 1.], [1., -2., 1.])):
            p = manual_plan(g, P, d, tag=f'b{i}'); items.append((p['id'], 1.-.1*i, p))
        top = sorted(items, key=lambda t: (-t[1], t[0]))
        sel = cs.select_diverse(items, 5, ctx)
        self.assertEqual(sel[:2], [top[0][0], top[1][0]])          # top-2 by score kept
        self.assertEqual(len(sel), 5)
        self.assertFalse(any(i.startswith('b') for i in sel))      # only >= median score
        self.assertEqual(sum(i.startswith('d') for i in sel), 3)   # spread wins over the cluster
        byid = {t[0]: t[2] for t in items}
        spread = cs.report([byid[i] for i in sel], ctx, 'x')
        greedy = cs.report([t[2] for t in top[:5]], ctx, 'x')
        self.assertGreater(spread['mean_pairwise_descriptor_distance'], greedy['mean_pairwise_descriptor_distance'])
        self.assertGreater(spread['distinct_participation_onset_orders'], greedy['distinct_participation_onset_orders'])
        off = cs.Context(g, cs.split_config(dict(coop_objective=True, coop_diversity=False)), P, BOUNDS)
        self.assertEqual(cs.select_diverse(items, 5, off), [t[0] for t in top[:5]])

    def test_descriptor_uses_issued_commands_after_clamp(self):
        """A plan that only differs from the policy where the actuator clamp removes it has
        zero issued deviation (no participation), unlike its raw perturbation."""
        g = group(2); P = np.full((2, T), 1.4)             # policy near the 1.5 ceiling
        feas = dict(members=g['members'], lo=np.full((2, T), -3.), hi=np.full((2, T), 1.))
        plan = ms.build_plan(g, P, np.full((2, ms.canonical_knots()), -.3), 60, BOUNDS, 'sampled',
                             'ordinary', 'c', feasibility=feas)
        self.assertTrue(plan.get('policy_duplicate'))      # issued 1.0 == issued policy 1.0
        coop = cs.split_config(dict(coop_objective=True))
        raw = cs.member_features(plan['perturbation_profile'], coop)
        self.assertTrue((raw[:, 3] < 0).all())              # raw perturbation says 'slower'
        F = cs.Context(g, coop, P, BOUNDS, feas).features(plan)
        np.testing.assert_allclose(F[:, [0, 2, 3, 4]], 0.)  # issued: identical to the policy
        np.testing.assert_allclose(F[:, 1], 60.)            # never starts

    def test_onset_order_and_participation(self):
        coop = cs.split_config(dict(coop_objective=True))
        dev = np.zeros((3, 40)); dev[0, 12:30] = -.8; dev[1, 2:20] = -.8; dev[2, 5:8] = -2.
        F = cs.member_features(dev, coop)
        np.testing.assert_allclose(F[:, 0], [1, 1, 0])       # 3 s active < 5 s: not participating
        np.testing.assert_allclose(F[:, 1], [12, 2, 5])

    def test_search_finalists_spread_more_than_score_order(self):
        g = group(3)
        on = coop_search_run(JointStub(g, solo_w=10.), g, coop_objective=True, **LIVE)
        nodiv = coop_search_run(JointStub(g, solo_w=10.), g, coop_objective=True, coop_diversity=False, **LIVE)
        a = on['coop_objective']['finalists_report']['mean_pairwise_descriptor_distance']
        b = nodiv['coop_objective']['finalists_report']['mean_pairwise_descriptor_distance']
        self.assertGreater(a, b)
        self.assertEqual(len(on['finalists']), 8)


class Novelty(unittest.TestCase):
    def test_bank_neighbours_penalised(self):
        g = group(2, lanes=[2, 0]); stub = JointStub(g, solo_w=1., joint_w=0.)
        P = stub.policy
        near = manual_plan(g, P, [-1., -1.], tag='N'); far = manual_plan(g, P, [0., -.6], tag='F')
        mid = manual_plan(g, P, [-.5, 0.], tag='M')
        plans = [near, far, mid]
        probe = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        bank = [list(probe.role_descriptor(near)+d) for d in (0., .01, -.01)]
        out = {}
        for flag in (False, True):
            ctx = cs.Context(g, cs.split_config(dict(coop_objective=True, coop_novelty=flag,
                                                     coop_novelty_bank=bank)), P, BOUNDS)
            base = ms.comparators(dict(policy_residuals={}))
            scores = stub({}, base+plans+cs.solos_for(plans), 2, 300)
            cs.screen({}, stub, base, plans, scores, 2, 300, ctx, 's')
            out[flag] = {p['id']: ctx.info['s'][p['id']] for p in plans}
        for p in plans:
            self.assertEqual(out[False][p['id']]['novelty'], 0.)
        self.assertLess(out[True][near['id']]['novelty'], 0.)
        self.assertGreater(out[True][far['id']]['novelty'], 0.)
        lift = {i: out[True][i]['score']-out[False][i]['score'] for i in out[True]}
        self.assertLess(lift[near['id']], lift[far['id']])

    def test_bank_descriptor_and_validation(self):
        dev_b = np.zeros(20); dev_b[3:13] = .5
        issued = np.stack([np.full(20, -1.), dev_b, np.full(20, .2)])
        plan = dict(members=['a', 'b', 'c'], duration_s=20, issued=issued.tolist())
        d = cs.plan_role_descriptor(plan, {'a': 2, 'b': 0, 'c': 1}, np.zeros((3, 20)))
        self.assertEqual(len(d), cs.ROLE_DIM)
        np.testing.assert_allclose(d[:7], [1, 0, -1, -1, -1, -1, 1])                 # ramp member a
        np.testing.assert_allclose(d[7:14], [.5, 3, .275, .225, .2, .25, .225])       # mainline b, c
        self.assertEqual(d[14], -3.)                                                 # ramp acts first
        with self.assertRaises(ValueError):
            cs.split_config(dict(coop_objective=True, coop_novelty_bank=[[0.]*5]))


class CostCap(unittest.TestCase):
    def test_cap_respected_and_futures_counted(self):
        g = group(6); stub = JointStub(g, solo_w=5.)
        res = coop_search_run(stub, g, coop_objective=True, coop_solo_cap=16, **LIVE)
        N = res['scaling']['population']; self.assertEqual(N, 120)
        screen_solo_calls = [c for c in stub.calls if c['solos'] and c['K'] == 2]
        self.assertEqual(len(screen_solo_calls), 2)                   # one capped extra call per iteration
        for c in screen_solo_calls:
            self.assertEqual(len(c['solos']), 16*6)
            self.assertEqual(len({s['solo_of'] for s in c['solos']}), 16)
            self.assertEqual(len(c['ids']), 16*6+2)                    # only solos + comparators
        final = stub.calls[-1]
        self.assertEqual(final['K'], 8)
        self.assertEqual(len(final['solos']), 8*6)
        self.assertEqual(res['imagined_futures'], sum(len(c['ids'])*c['K'] for c in stub.calls))
        self.assertEqual(res['coop_objective']['solo_rollouts'], dict(screen=2*16*6, final=8*6))
        imputed = [s['candidates']-s['solo_computed'] for s in res['coop_objective']['screens']]
        self.assertEqual(imputed, [N-16, N-16])
        est = res['coop_objective']['cost_estimate']
        self.assertEqual(est['coop_action_futures'], res['imagined_futures'])

    def test_uncapped_batches_solos_into_the_candidate_call(self):
        g = group(2); stub = JointStub(g)
        res = coop_search_run(stub, g, coop_objective=True, coop_solo_cap=None, **LIVE)
        self.assertEqual(len(stub.calls), 3)                           # 2 screens + final, no extra call
        N = res['scaling']['population']
        self.assertEqual(len(stub.calls[0]['solos']), N*2)
        self.assertEqual(res['coop_objective']['extra_wm_calls'], 0)

    def test_cost_multiplier_bounded(self):
        for m in range(2, 7):
            est = cs.cost_multiplier(m, dict(LIVE, coop_solo_cap=16))
            self.assertLess(est['multiplier_action_futures'], 2.6)
            self.assertGreater(est['multiplier_action_futures'], 1.5)


class LiveWrappers(unittest.TestCase):
    def test_capture_excludes_solos_and_carries_coop(self):
        from .ls_gpu import capture_search
        g = group(3); stub = JointStub(g)
        rec = dict(state_id='s', policy_residuals={v: 0. for v in g['members']}, group=g['members'],
                   rows=[dict(vid=r['vid'], s_m=r['s_m'], v_mps=r['v_mps'], lane=r['lane']) for r in g['rows']])
        cfg = dict(iterations=2, screen_K=2, niches=4, coop_objective=True)
        out = capture_search(ms.search, label_list8=True)(rec, g, stub.policy, BOUNDS, stub, 'ordinary',
                                                           mappo_traj=stub.mappo, seed=2, config=cfg)
        top8 = out['collection_top8']
        self.assertEqual(len(top8), 8)
        self.assertFalse(any('solo_of' in p for p in top8))
        self.assertTrue(all('coop' in p for p in top8))
        self.assertEqual({p['id'] for p in top8}, {f['id'] for f in out['finalists']})

    def test_collector_config_validation(self):
        from .niche_collect import search_config, validate
        coop = dict(coop_objective=True, coop_lambda=1., coop_solo_cap=16)
        self.assertEqual(search_config(3), dict(iterations=2, screen_K=2, niches=4))
        self.assertEqual(search_config(3, coop), dict(iterations=2, screen_K=2, niches=4, **coop))
        self.assertEqual([('coop_objective' in search_config(i, coop, True)) for i in range(4)],
                         [True, True, False, False])
        self.assertEqual(validate(search_config(1, coop)), search_config(1, coop))
        with self.assertRaises(ValueError):
            validate(dict(search_config(1), coop_lambda=1.))
        with self.assertRaises(ValueError):
            validate(dict(search_config(1, coop), coop_solo_cap=0))


if __name__ == '__main__':
    unittest.main()
