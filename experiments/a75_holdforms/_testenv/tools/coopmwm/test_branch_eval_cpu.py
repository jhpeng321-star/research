"""CPU-only tests for the paired branch-test harness. Fake deterministic runner; no SUMO."""
import copy
import hashlib
import unittest

import numpy as np

from . import branch_eval as be

KEYS = ('team_delay_vs', 'road_delay_vs', 'source_wait_vs', 'collisions', 'teleports',
        'ttc_below_3_vs', 'hard_brake')
# Per-second team-delay change caused by each arm while it controls (veh-s per s).
EFFECT = dict(MAPPO=0., A=-.5, B=-1., R=-.25)
FOCUS_BONUS_B = -1.  # extra B effect in focus-stratum worlds


def small_panel():
    """Two worlds per cell for four cells: two focus cells and two non-focus cells."""
    cells = [(.05, 1900), (.2, 2400), (.4, 2700), (.1, 2700)]
    return [dict(seed=900000 + 10 * i + r, cell=i, repeat=r, penetration=p, demand=d)
            for i, (p, d) in enumerate(cells) for r in range(2)]


def base_rate(world, t):
    return 5. + (world['seed'] % 7) + .01 * (t % 97)


def effect(arm, world):
    e = EFFECT[arm]
    if arm == 'B' and be.in_focus(world):
        e += FOCUS_BONUS_B
    return e


class FakeRunner:
    """Deterministic plant: the fingerprint hashes the realised physics, not the arm label."""

    def __init__(self, leak=False):
        self.leak, self.sim_s, self.calls = leak, 0, []

    def _episode(self, world, stop, controller_at):
        totals, h, prefixes = dict.fromkeys(KEYS, 0.), hashlib.sha256(str(world['seed']).encode()), {}
        for t in range(stop):
            if t >= be.WARMUP_S:
                prefixes[str(t)] = dict(t_s=float(t), tape=h.hexdigest(),
                                        metrics={k: v.hex() for k, v in totals.items()})
            arm = controller_at(t) if t >= be.WARMUP_S else 'MAPPO'
            inc = base_rate(world, t) + effect(arm, world)
            totals['team_delay_vs'] += inc
            totals['road_delay_vs'] += .75 * inc
            totals['source_wait_vs'] += .25 * inc
            totals['hard_brake'] += .1 if arm == 'B' else 0.
            h.update(repr(inc).encode())
        self.sim_s += stop
        final = dict(t_s=float(stop), tape=h.hexdigest(), metrics={k: v.hex() for k, v in totals.items()})
        return dict(prefixes=prefixes, final=final)

    def reference(self, job):
        self.calls.append(job['id'])
        return self._episode(job['world'], job['stop_s'], lambda t: 'MAPPO')

    def run(self, job, reference):
        self.calls.append(job['id'])
        T, arm = job['branch_time_s'], job['arm']
        start = min(T, be.WARMUP_S + 60) if self.leak else T  # leak: arm acts inside the prefix
        return self._episode(job['world'], job['stop_s'], lambda t: arm if t >= start else 'MAPPO')


class PlanTests(unittest.TestCase):
    def test_branch_times_valid(self):
        times = be.branch_times()
        self.assertEqual(times, [300, 420, 540, 660, 780, 900])
        self.assertTrue(be.validate_times(times))
        self.assertTrue(all(t >= 300 and t + 300 <= 1200 for t in times))
        for bad in ([299], [901], [420, 300], [300, 300], [300.0]):
            with self.assertRaises(ValueError):
                be.validate_times(bad)
        self.assertEqual(be.branch_times(4), [300, 500, 700, 900])

    def test_plan_hash_stable_and_tamper_evident(self):
        a, b = be.build_plan(), be.build_plan()
        self.assertEqual(a['plan_sha256'], b['plan_sha256'])
        self.assertEqual(len(a['plan_sha256']), 64)
        self.assertTrue(be.verify_plan(a))
        reordered = dict(reversed(list(a.items())))
        self.assertEqual(be.plan_sha(reordered), a['plan_sha256'])
        t = copy.deepcopy(a)
        t['branches']['times_s'][0] = 360
        with self.assertRaises(ValueError):
            be.verify_plan(t)
        self.assertNotEqual(be.build_plan(n_branches=4)['plan_sha256'], a['plan_sha256'])

    def test_final_panel_identity_and_focus_subset(self):
        plan = be.build_plan()
        ws = plan['panel']['worlds']
        self.assertEqual(len(ws), 96)
        self.assertEqual(plan['panel']['panel_sha256'], be.FINAL_PANEL_SHA)
        self.assertEqual({w['seed'] for w in ws}, set(range(989600, 989636)) | set(range(990600, 990660)))
        focus = plan['analysis']['focus']
        self.assertEqual(focus['n_worlds'], 32)
        chosen = [w for w in ws if w['seed'] in set(focus['seeds'])]
        self.assertEqual({(w['penetration'], w['demand']) for w in chosen},
                         {(.2, 2400), (.2, 2700), (.4, 2400), (.4, 2700)})
        self.assertTrue(all(sum(1 for w in chosen if w['cell'] == c) == 8 for c in {w['cell'] for w in chosen}))
        self.assertFalse(be.in_focus(dict(penetration=.1, demand=2700)))
        self.assertFalse(be.in_focus(dict(penetration=.2, demand=1900)))
        self.assertTrue(be.in_focus(dict(penetration=.4, demand=2400)))

    def test_holm_family_is_exactly_four(self):
        fam = be.build_plan()['analysis']['co_primary']['family']
        self.assertEqual(fam, ['branch:B-MAPPO', 'branch:B-A', 'focus:B-MAPPO', 'focus:B-A'])

    def test_budget_arithmetic(self):
        times = be.branch_times()
        q = be.budget_quote(96, times, ['MAPPO', 'A', 'B'])
        self.assertEqual(q['per_world_per_arm_sim_s'], 5400)
        self.assertEqual(q['per_arm_sim_s'], 518400)
        self.assertEqual(q['total_sim_s'], 1036800)
        self.assertEqual(q['naive_every_arm_rerun_sim_s'], 1555200)
        self.assertEqual(q['saved_state_restore_alternative_sim_s'], 518400)
        q2 = be.budget_quote(96, times, ['MAPPO', 'A', 'B'], reuse_mappo_reference=False)
        self.assertEqual(q2['total_sim_s'], 1036800 + 115200)
        q3 = be.budget_quote(96, times, ['MAPPO', 'A', 'B', 'R', 'native'])
        self.assertEqual(q3['total_sim_s'], 2073600)
        plan = be.build_plan()
        refs, js = be.jobs(plan)
        self.assertEqual(len(js), 96 * 6 * 3)
        self.assertEqual(len({j['id'] for j in js}), len(js))
        self.assertEqual(be.job_sim_s(refs, js), plan['budget']['total_sim_s'])
        refs, js = be.jobs(plan, ['MAPPO', 'A', 'B', 'R', 'native'])
        self.assertEqual(be.job_sim_s(refs, js), plan['budget_with_later_arms']['total_sim_s'])
        with self.assertRaises(ValueError):
            be.jobs(plan, ['Z'])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.plan = be.build_plan(worlds=small_panel(), reuse_mappo_reference=False, n_boot=2000)

    def test_identical_prefix_across_arms_and_known_effect(self):
        runner = FakeRunner()
        records = be.execute(self.plan, runner)
        self.assertEqual(len(records), 8 * 6 * 3)
        # Fake actual sim-s equals the quoted exact-rerun budget.
        self.assertEqual(runner.sim_s, self.plan['budget']['total_sim_s'])
        by = {(r['seed'], r['branch_time_s'], r['arm']): r for r in records}
        for w in small_panel():
            for t in self.plan['branches']['times_s']:
                self.assertAlmostEqual(by[w['seed'], t, 'A']['J'] - by[w['seed'], t, 'MAPPO']['J'], -150., 6)
                expect = -300. + (-300. if be.in_focus(w) else 0.)
                self.assertAlmostEqual(by[w['seed'], t, 'B']['J'] - by[w['seed'], t, 'MAPPO']['J'], expect, 6)
        out = be.analyze(records, self.plan)
        res = out['results']
        self.assertAlmostEqual(res['branch:B-MAPPO']['mean_diff_vs'], -450., 6)
        self.assertAlmostEqual(res['focus:B-MAPPO']['mean_diff_vs'], -600., 6)
        self.assertAlmostEqual(res['focus:B-A']['mean_diff_vs'], -450., 6)
        self.assertAlmostEqual(res['branch:A-MAPPO']['mean_diff_vs'], -150., 6)
        self.assertEqual(out['n_focus_worlds'], 4)
        for k in out['family']:
            self.assertTrue(res[k]['holm']['reject'])
            self.assertGreater(res[k]['pct_delay_reduction'], 0)
        self.assertNotIn('holm', res['branch:A-MAPPO'])

    def test_prefix_leak_is_rejected(self):
        with self.assertRaises(be.PrefixMismatch):
            be.execute(self.plan, FakeRunner(leak=True), arms=['MAPPO', 'A'])

    def test_prefix_check_each_second(self):
        runner = FakeRunner()
        ref = runner.reference(dict(id='r', world=small_panel()[0], stop_s=1200))
        run = runner.run(dict(id='a', world=small_panel()[0], arm='B', branch_time_s=540, stop_s=840), ref)
        self.assertTrue(be.check_prefix(run, ref, 540))
        self.assertNotEqual(run['prefixes']['541'], ref['prefixes']['541'])
        bad = copy.deepcopy(run)
        bad['prefixes']['400']['tape'] = '0' * 64
        with self.assertRaises(be.PrefixMismatch):
            be.check_prefix(bad, ref, 540)

    def test_incomplete_or_duplicate_panel_rejected(self):
        records = be.execute(self.plan, FakeRunner())
        with self.assertRaises(ValueError):
            be.analyze(records[:-1], self.plan)
        with self.assertRaises(ValueError):
            be.analyze(records + records[:1], self.plan)


class AnalysisTests(unittest.TestCase):
    def test_holm(self):
        out = be.holm({'a': .01, 'b': .04, 'c': .03, 'd': .005}, .05)
        self.assertEqual({k: v['reject'] for k, v in out.items()}, dict(a=True, b=False, c=False, d=True))
        self.assertAlmostEqual(out['d']['p_holm'], .02)
        self.assertAlmostEqual(out['a']['p_holm'], .03)
        self.assertAlmostEqual(out['c']['p_holm'], .06)
        self.assertAlmostEqual(out['b']['p_holm'], .06)
        # Step-down stops at the first non-rejection even if a later raw p is small enough.
        out = be.holm({'x': .02, 'y': .026}, .05)
        self.assertTrue(out['x']['reject'] and out['y']['reject'])
        out = be.holm({'x': .03, 'y': .04}, .05)
        self.assertFalse(out['x']['reject'] or out['y']['reject'])

    def test_world_cluster_bootstrap(self):
        rng = np.random.default_rng(1)
        n, k = 40, 6
        comp = 1000 + rng.normal(0, 50, (n, k))
        world_effect = rng.normal(-30, 40, (n, 1))
        treat = comp + world_effect + rng.normal(0, 1, (n, k))
        r = be.cluster_bootstrap(treat, comp, 4000, 7)
        d = (treat - comp).mean(1)
        idx = np.random.default_rng(7).integers(0, n, (4000, n))
        manual = np.quantile(d[idx].mean(1), [.025, .975])
        np.testing.assert_allclose(r['ci95'], manual, rtol=0, atol=1e-9)
        self.assertLess(r['ci95'][0], (treat - comp).mean())
        self.assertLess((treat - comp).mean(), r['ci95'][1])
        # Branch-level iid resampling would ignore the shared world effect and be much narrower.
        flat = (treat - comp).ravel()
        iid = np.quantile(flat[np.random.default_rng(7).integers(0, flat.size, (4000, flat.size))].mean(1),
                          [.025, .975])
        self.assertGreater(r['ci95'][1] - r['ci95'][0], 2 * (iid[1] - iid[0]))
        # Fixed RNG -> identical output.
        self.assertEqual(r, be.cluster_bootstrap(treat, comp, 4000, 7))
        self.assertEqual(r['n_worlds'], n)
        self.assertEqual(r['n_branches'], n * k)

    def test_known_effect_and_null(self):
        rng = np.random.default_rng(3)
        comp = 2000 + rng.normal(0, 100, (30, 6))
        strong = be.cluster_bootstrap(comp * .9, comp, 2000, 5)
        self.assertAlmostEqual(strong['pct_delay_reduction'], 10., 9)
        self.assertLess(strong['p_two_sided'], .01)
        np.testing.assert_allclose(strong['pct_ci95'], [10., 10.], atol=1e-9)
        noise = rng.normal(0, 30, (30, 1)) * np.ones((1, 6))
        null = be.cluster_bootstrap(comp + noise - noise.mean(), comp, 2000, 5)
        self.assertAlmostEqual(null['mean_diff_vs'], 0., 9)
        self.assertGreater(null['p_two_sided'], .5)
        worse = be.cluster_bootstrap(comp * 1.05, comp, 2000, 5)
        self.assertLess(worse['pct_delay_reduction'], 0)
        self.assertGreater(worse['mean_diff_vs'], 0)


class TakeoverTests(unittest.TestCase):
    def test_mixin_suppresses_before_takeover(self):
        class Base:
            def __init__(self, branch):
                self.branch, self.events = branch, []

            def apply(self, rows, commands, now):
                req = {'cav': dict(requested_raw=1.)}
                self.events.append(dict(time_s=int(now), requests=req))
                return req

        C = type('C', (be.TakeoverControllerMixin, Base), {})
        c = C(dict(takeover_s=310, time_s=310, revert=[]))
        self.assertEqual(c.apply([], {}, 309.), {})
        self.assertTrue(c.events[-1]['takeover_suppressed'])
        self.assertEqual(c.events[-1]['suppressed_requests'], {'cav': dict(requested_raw=1.)})
        self.assertEqual(c.apply([], {}, 310.), {'cav': dict(requested_raw=1.)})
        self.assertEqual(len(c.events), 2)

    def test_real_controller_takeover(self):
        try:
            import torch
            from .acceleration import Policy
            from tools.coopnative.test_cpu import rows
        except ImportError:
            self.skipTest('torch unavailable in this interpreter')
        policy = Policy()
        with torch.no_grad():
            policy.actor[-1].bias.fill_(.3)
        r = rows()
        commands = {x['vid']: .1 for x in r if x['is_cav']}
        from . import accel_plant
        original = accel_plant.Controller
        with be.takeover_controller() as cls:
            self.assertIsNot(accel_plant.Controller, original)
            c = cls(policy, torch.Generator().manual_seed(0), True,
                    branch=dict(time_s=310, takeover_s=310, revert=[], prefixes={}))
            self.assertEqual(c.apply(r, commands, 309.), {})
            after = c.apply(r, commands, 310.)
        self.assertIs(accel_plant.Controller, original)
        self.assertEqual(set(after), set(commands))
        for v in after.values():
            self.assertAlmostEqual(v['requested_raw'], .4, 5)
        self.assertEqual(len(c.events), 2)
        # The zero-residual MAPPO adapter requests nothing at any time (bitwise MAPPO path).
        z = be.TakeoverControllerMixin
        self.assertTrue(issubclass(cls, z))
        zero = original(Policy(), torch.Generator().manual_seed(0), True)
        self.assertEqual(zero.apply(r, commands, 400.), {})


if __name__ == '__main__':
    unittest.main()
