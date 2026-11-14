"""CPU tests for option ``coop_objective`` (synergy score, diversity, novelty, cost cap).

Real torch on CPU with a synthetic WM stub; no SUMO, no WM checkpoint, no GPU job.
"""
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
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


GOLDEN = Path(__file__).resolve().parent/'fixtures'/'coop_off_golden.json'
OFF_CASES = (('n1_m2', 1, 2, {}), ('n4_m2', 4, 2, {}), ('n1_m3_live', 1, 3, LIVE), ('n4_m3_live', 4, 3, LIVE))
OFF_VARIANTS = (None, dict(coop_objective=False), dict(coop_objective=False, coop_lambda=3., coop_solo_cap=2))


def off_digests(module, config_extra=None):
    """sha256 of the serialized search output and of the WM call sequence per off case."""
    out = {}
    for name, niches, m, cfg in OFF_CASES:
        fake = Fake(m); g = group(m); config = dict(cfg, **(config_extra or {}))
        if niches > 1:
            config['niches'] = niches
        res = module.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g, fake.policy,
                            BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=5, config=config)
        out[name] = dict(result=hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest(),
                         calls=hashlib.sha256(json.dumps(fake.calls).encode()).hexdigest(),
                         coop_key=any('coop' in k for k in res) or any(k.startswith('coop') for k in res['config']))
    return out


def golden_reference():
    """Base b20bec2f outputs. Linux qualification sets COOPMWM_COOP_BASELINE to the immutable
    pre-change mppi_search.py (same interpreter/BLAS); otherwise the committed Windows fixture
    generated from that base module is used. Never regenerated from the new implementation."""
    path = os.environ.get('COOPMWM_COOP_BASELINE')
    if not path:
        return json.loads(GOLDEN.read_text())['digests']
    spec = importlib.util.spec_from_file_location('tools.coopmwm._pre_coop_reference', path)
    old = importlib.util.module_from_spec(spec); sys.modules[spec.name] = old; spec.loader.exec_module(old)
    return off_digests(old)


class OffIdentical(unittest.TestCase):
    def _run(self, config, niches=1, m=2):
        fake = Fake(m); g = group(m); cfg = dict(config or {})
        if niches > 1:
            cfg['niches'] = niches
        res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g, fake.policy,
                        BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo, seed=5, config=cfg)
        return json.dumps(res, sort_keys=True), fake.calls

    def test_off_is_byte_identical_to_base_b20bec2f(self):
        ref = golden_reference()
        self.assertEqual(sorted(ref), sorted(c[0] for c in OFF_CASES))
        for variant in OFF_VARIANTS:
            got = off_digests(ms, variant)
            self.assertEqual(got, ref, variant)
            self.assertFalse(any(v['coop_key'] for v in got.values()))

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


class ReviewFixes(unittest.TestCase):
    def test_pessimistic_imputation_and_measured_finalists(self):
        g = group(6); stub = JointStub(g, solo_w=5.)
        res = coop_search_run(stub, g, coop_objective=True, coop_solo_cap=16, **LIVE)
        co = res['coop_objective']
        for sc in co['screens']:
            self.assertAlmostEqual(sc['imputed_share'], (sc['candidates']-sc['solo_computed'])/sc['candidates'])
            self.assertGreater(sc['imputed_share'], .5)
        # Recompute: every imputed synergy equals the MIN computed synergy of its call.
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)))
        base = ms.comparators(dict(policy_residuals={}))
        plans = [manual_plan(g, stub.policy, [(-1.)**i*.2*j for i in range(6)], tag=f'p{j}') for j in range(5)]
        ctx.coop['coop_solo_cap'] = 2
        scores = stub({}, base+plans, 2, 300)
        cs.screen({}, stub, base, plans, scores, 2, 300, ctx, 's')
        info = ctx.info['s']
        computed = [x['synergy'] for x in info.values() if not x['synergy_imputed']]
        self.assertEqual(len(computed), 2)
        for x in info.values():
            if x['synergy_imputed']:
                self.assertEqual(x['synergy'], min(computed))
        # Finalists only from candidates with measured solos.
        for f in res['finalists']:
            self.assertFalse(f['coop']['synergy_imputed'])
        self.assertEqual(co['extended_candidates'], 0)        # 2 x 16 measured >= 8 finalists

    def test_extension_call_when_too_few_measured(self):
        g = group(2); stub = JointStub(g, solo_w=5.)
        res = coop_search_run(stub, g, coop_objective=True, coop_solo_cap=2, **LIVE)
        co = res['coop_objective']
        self.assertEqual(co['extended_candidates'], 8-2*2)       # 4 measured in screens, 8 finalists
        ext = [c for c in stub.calls if c['solos'] and c['K'] == 2][-1]
        self.assertEqual(len({s['solo_of'] for s in ext['solos']}), 4)
        self.assertEqual(co['solo_rollouts']['extension'], 8)
        self.assertEqual(sum(sc['solo_extended'] for sc in co['screens']), 4)
        self.assertEqual(res['imagined_futures'], sum(len(c['ids'])*c['K'] for c in stub.calls))
        self.assertEqual(len(res['finalists']), 8)

    def test_niche_best_first_and_plain_log(self):
        g = group(3); stub = JointStub(g, solo_w=10.)
        res = coop_search_run(stub, g, coop_objective=True, niches=4, **LIVE)
        fin = res['finalists']
        self.assertEqual(len({f['niche'] for f in fin}), 4)
        firsts = fin[:4]
        self.assertEqual(len({f['niche'] for f in firsts}), 4)  # best of each niche first
        for it, sc in zip(res['iterations'], res['coop_objective']['screens']):
            self.assertAlmostEqual(it['mean_rel_cost'], -sc['mean_group_gain'], places=9)
            for n in it['niches']:
                self.assertIn('mean_coop_rel_cost', n)

    def test_bank_by_digest(self):
        import tempfile
        from .niche_collect import search_config, validate
        g = group(2, lanes=[2, 0]); P = np.full((2, T), .2)
        probe = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        bank = [[float(x) for x in probe.role_descriptor(manual_plan(g, P, [-1., d], tag='q'))] for d in (0., .5)]
        sha = cs.bank_digest(bank)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/cs.bank_file_name(sha)).write_text(json.dumps(dict(descriptors=bank)))
            cs.BANK_DIRS.append(tmp)
            try:
                cfg = dict(coop_objective=True, coop_novelty=True, coop_novelty_bank_sha256=sha)
                coop = cs.split_config(cfg)
                self.assertEqual(coop['coop_novelty_bank'], bank)
                self.assertEqual(cfg['coop_novelty_bank'], dict(n=2, sha256=sha))
                self.assertNotIn('coop_novelty_bank_sha256', cfg)
            finally:
                cs.BANK_DIRS.remove(tmp)
        with self.assertRaises(ValueError):
            cs.split_config(dict(coop_objective=True, coop_novelty=True,
                                 coop_novelty_bank_sha256='0'*64))   # not found (novelty on resolves)
        self.assertIsNone(cs.split_config(dict(coop_objective=True, coop_novelty=False,
                                               coop_novelty_bank_sha256='0'*64))['coop_novelty_bank'])
        sc = search_config(0, dict(coop_objective=True, coop_novelty=True, coop_novelty_bank_sha256=sha))
        self.assertEqual(validate(sc), sc)                       # digest only: validated without the file
        with self.assertRaises(ValueError):
            validate(dict(sc, coop_novelty_bank=bank))           # embedding is rejected
        with self.assertRaises(ValueError):
            validate(dict(sc, coop_novelty_bank_sha256='xyz'))


class Fix2Review(unittest.TestCase):
    """Second review round: bank preflight (CLI + supervisor startup), novelty-off resolution
    skip, extension counting over finalist-ELIGIBLE measured candidates only."""

    def _bank_file(self, tmp, sha=None, descriptors=None):
        g = group(2, lanes=[2, 0]); P = np.full((2, T), .2)
        probe = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        bank = descriptors if descriptors is not None else \
            [[float(x) for x in probe.role_descriptor(manual_plan(g, P, [-1., d], tag='b'))] for d in (0., .4)]
        sha = sha or cs.bank_digest(bank)
        (Path(tmp)/cs.bank_file_name(sha)).write_text(json.dumps(dict(descriptors=bank)))
        return sha

    def test_preflight_cli_and_supervisor_startup(self):
        """`python -m tools.coopmwm.coop_search preflight --sha` (GPU host, COOPMWM_COOP_BANK_DIR)
        exits 0 only when the bank resolves and matches its digest; the collector supervisor's
        startup runs the same check and fails loudly before admitting work."""
        import subprocess
        import tempfile
        from unittest.mock import patch
        from .ls_collection import preflight_coop
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            sha = self._bank_file(tmp)
            env = dict(os.environ, PYTHONPATH=str(root), COOPMWM_COOP_BANK_DIR=tmp)
            r = subprocess.run([sys.executable, '-m', 'tools.coopmwm.coop_search', 'preflight',
                                '--sha', sha], capture_output=True, text=True, cwd=str(root), env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)['n'], 2)
            # Missing bank file: non-zero exit, JSON failure on stderr.
            r = subprocess.run([sys.executable, '-m', 'tools.coopmwm.coop_search', 'preflight',
                                '--sha', '0'*64], capture_output=True, text=True, cwd=str(root), env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse(json.loads(r.stderr.strip().splitlines()[-1])['ok'])
            # Present file that does not match its digest: non-zero exit.
            self._bank_file(tmp, sha='f'*64, descriptors=[[0.]*cs.ROLE_DIM, [1.]*cs.ROLE_DIM])
            self.assertNotEqual(cs.bank_digest([[0.]*cs.ROLE_DIM, [1.]*cs.ROLE_DIM]), 'f'*64)
            r = subprocess.run([sys.executable, '-m', 'tools.coopmwm.coop_search', 'preflight',
                                '--sha', 'f'*64], capture_output=True, text=True, cwd=str(root), env=env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('digest', r.stderr)
            # Supervisor startup helper: same check, raising on missing/mismatched bank.
            coop = dict(coop_objective=True, coop_novelty=True, coop_novelty_bank_sha256=sha)
            with patch.dict(os.environ, {'COOPMWM_COOP_BANK_DIR': tmp}):
                self.assertEqual(len(preflight_coop(dict(coop_search=coop))), 2)
                with self.assertRaises(ValueError):
                    preflight_coop(dict(coop_search=dict(coop, coop_novelty_bank_sha256='0'*64)))
                with self.assertRaises(ValueError):
                    preflight_coop(dict(coop_search=dict(coop, coop_novelty_bank_sha256='f'*64)))
                # Novelty off or no coop config: no check, nothing resolved.
                self.assertIsNone(preflight_coop({}))
                self.assertIsNone(preflight_coop(dict(
                    coop_search=dict(coop_objective=True, coop_novelty=False, coop_novelty_bank_sha256='0'*64))))

    def test_novelty_off_skips_bank_resolution(self):
        """A present sha with coop_novelty False must not touch the bank at all (no resolution,
        no failure when the file is absent); novelty on resolves and fails loudly."""
        from unittest.mock import patch
        sha = '0'*64                                    # no file anywhere
        calls = []
        real = cs.resolve_bank
        def spy(sha_, dirs=None):
            calls.append(sha_); return real(sha_, dirs)
        with patch.object(cs, 'resolve_bank', spy), patch.dict(os.environ, {'COOPMWM_COOP_BANK_DIR': 'no_such_dir'}):
            cfg = dict(coop_objective=True, coop_novelty=False, coop_novelty_bank_sha256=sha)
            coop = cs.split_config(cfg)
            self.assertEqual(calls, [])
            self.assertIsNone(coop['coop_novelty_bank'])
            self.assertIsNone(cfg['coop_novelty_bank'])
            self.assertNotIn('coop_novelty_bank_sha256', cfg)
            with self.assertRaises(ValueError):          # same sha, novelty on: resolves
                cs.split_config(dict(coop_objective=True, coop_novelty=True, coop_novelty_bank_sha256=sha))
            self.assertEqual(calls, [sha])

    def test_extension_counts_finalist_eligible_only(self):
        g = group(2); P = np.full((2, T), .2)
        def evaluate(record, actions, K, H):
            return {a['id']: [dict(team_delay=100.) for _ in range(K)] for a in actions}
        def setup(ctx, spec):
            pool, ids = {}, {}
            for i, (dup, imp, gg, niche) in enumerate(spec):
                p = manual_plan(g, P, [-1., .5], tag=f'e{i}')
                if dup:
                    p['policy_duplicate'] = True
                ctx.info.setdefault('screen0', {})[p['id']] = dict(
                    group_gain=gg, novelty=0., synergy=0., synergy_imputed=imp,
                    best_solo_gain=None, solo_gains=None)
                ctx.stage_of[p['id']] = 'screen0'
                pool[p['id']] = (-float(gg), p); ids[i] = p['id']
            return pool, ids
        base = ms.comparators(dict(policy_residuals={}))
        # 4 candidates with computed solos but 3 of them clipped policy duplicates (finalist-
        # ineligible); 2 imputed; n=4 finalists: the extension MUST run (the old count of all
        # measured candidates saw 4 >= 4 and skipped it).
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        spec = [(True, False, 5., 0), (True, False, 4., 0), (True, False, 3., 0), (False, False, 2., 0),
                (False, True, 1., 0), (False, True, .5, 0)]
        pool, ids = setup(ctx, spec)
        extra = cs.extend_solos(dict(), evaluate, base, pool, None, 4, 2, 300, ctx)
        self.assertEqual(extra, (len(base)+4)*2)             # the 2 imputed plans x 2 member solos
        self.assertEqual(ctx.extra_calls, 1)
        self.assertEqual(ctx.extended, 2)
        ext = {pid for pid, x in ctx.info['screen0'].items() if x.get('solo_extension')}
        self.assertEqual(ext, {ids[4], ids[5]})
        for pid in ext:
            x = ctx.info['screen0'][pid]
            self.assertFalse(x['synergy_imputed'])
            self.assertEqual(x['solo_gains'], {'c0': 0., 'c1': 0.})   # stub: solo delay = policy delay
            self.assertAlmostEqual(x['score'], 2*x['group_gain'], places=9)   # synergy == group_gain, lambda 1
            self.assertAlmostEqual(pool[pid][0], -x['score'], places=9)
        self.assertEqual(ctx.solo_rollouts['extension'], 4)
        # All 4 measured candidates eligible: no extension (measured >= n), unchanged rule.
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        pool, _ = setup(ctx, [(False, False, 5., 0), (False, False, 4., 0), (False, False, 3., 0),
                              (False, False, 2., 0), (False, True, 1., 0)])
        self.assertEqual(cs.extend_solos(dict(), evaluate, base, pool, None, 4, 2, 300, ctx), 0)
        self.assertEqual(ctx.extra_calls, 0)
        # Niches: a niche covered only by a duplicate measured candidate is UNcovered, so its
        # best imputed candidate gets solos (n=3; the old rule saw covered-by-dup + 3 >= 3 and
        # skipped the call, leaving niche 0 without one finalist-eligible plan).
        ctx = cs.Context(g, cs.split_config(dict(coop_objective=True)), P, BOUNDS)
        spec = [(True, False, 5., 0), (False, True, 4., 0), (False, False, 3., 1), (False, False, 2., 2)]
        pool, ids = setup(ctx, spec)
        niche_of = {pid: spec[i][3] for i, pid in ids.items()}
        extra = cs.extend_solos(dict(), evaluate, base, pool, niche_of, 3, 2, 300, ctx)
        self.assertEqual(extra, (len(base)+2)*2)             # niche 0's best imputed plan x 2 solos
        self.assertEqual(ctx.extended, 1)
        self.assertEqual({pid for pid, x in ctx.info['screen0'].items() if x.get('solo_extension')},
                         {ids[1]})


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
