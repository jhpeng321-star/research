"""CPU tests for the exact SUMO plan grader. Fake deterministic plant behind the REAL
accel_plant.episode loop and the real acceleration.Controller/Policy; no SUMO, no GPU."""
from contextlib import contextmanager
import copy
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from . import accel_plant, exploration, plan_client
from . import exact_grade as xg
from .acceleration import Policy
from .protocol import digest, sha

WORLD = dict(seed=986001, cell=0, demand=2400, penetration=.2, repeat=0)
STOP = 520
OFFERS = {320: 'c1', 340: 'c1', 400: 'c4', 480: 'c1'}  # time -> anchor CAV of the offered group


class Plant:
    """Deterministic toy road. A CAV below 1 m/s leaves the sensor (membership can change)."""

    def __init__(self, world):
        r = world['seed'] % 7
        self.veh = {'c1': [0., 20., 0, True], 'c2': [-30., 19.5, 0, True], 'c3': [-70., 18.+.1*r, 1, True],
                    'h1': [40., 18., 0, False], 'h2': [-200., 15., 1, False],
                    'c4': [-1500., 20., 0, True], 'c5': [-1530., 19.5, 1, True]}
        self.totals = dict(team_delay_vs=0., road_delay_vs=0., source_wait_vs=0., collisions=0., teleports=0.,
                           hard_brake=0.)
        self.h = hashlib.sha256(str(world['seed']).encode())

    def rows(self):
        return [dict(vid=v, s_m=s, v_mps=sp, lane=l, is_cav=c, length_m=5.)
                for v, (s, sp, l, c) in sorted(self.veh.items()) if not (c and sp < 1.)]

    @staticmethod
    def commands(rows):
        return {r['vid']: .2*(20.-r['v_mps']) for r in rows if r['is_cav']}

    def step(self, desired, t, perturb=False):
        for v, x in sorted(self.veh.items()):
            s, sp, _, c = x
            a = (desired[v]['requested_raw'] if v in desired else .2*(20.-sp)) if c else .1*(18.-sp)
            a = min(2., max(-3., a))
            self.totals['hard_brake'] += a < -2.5
            x[1] = max(0., sp+a);x[0] = s+x[1]
        road = sum((30.-x[1])/30. for x in self.veh.values())+(1e-9 if perturb else 0.)
        source = .5+.001*(t % 50)
        self.totals['road_delay_vs'] += road;self.totals['source_wait_vs'] += source
        self.totals['team_delay_vs'] += road+source
        self.h.update(repr((sorted(self.veh.items()), sorted(self.totals.items()))).encode())

    def fp(self, t):
        return dict(t_s=float(t), tape=self.h.hexdigest(), metrics={k: float(v).hex() for k, v in self.totals.items()})


@contextmanager
def fake_sumo(perturb_at=None):
    lp = types.ModuleType('tools.leverprobe');be = types.ModuleType('tools.leverprobe.backend')
    be.setup = lambda runtime, checkpoint: None

    class Backend:
        def __init__(self, path, world, sim_s, checkpoint):
            self.plant = Plant(world);self.actual = 300;self.totals = self.plant.totals

        def fingerprint(self):
            return self.plant.fp(self.actual)

        def close(self):
            pass
    be.Backend = Backend;lp.backend = be

    def tick(b, controller, trace):
        rows = b.plant.rows()
        desired = controller.apply(rows, Plant.commands(rows), b.actual)
        b.plant.step(desired, b.actual, perturb=perturb_at == b.actual);b.actual += 1
        return []
    with patch.dict(sys.modules, {'tools.leverprobe': lp, 'tools.leverprobe.backend': be}), \
            patch('tools.coopfree.backend.tick', tick), patch.object(accel_plant, 'install_seed_scope'), \
            patch.object(accel_plant, 'guard'), patch.object(exploration, 'regime', lambda rows: 'ordinary'):
        yield


class FakeGuide:
    """Same generator consumption as plan_client.guide (rand per eligible query; R: randint)."""

    def __init__(self, generator, arm, accel, offers=None):
        self.g, self.arm, self.accel, self.offers = generator, arm, accel, offers or OFFERS

    def __call__(self, rows, commands, now, sample):
        from .mppi_search import group_components
        if int(now) % 10 or now > 900:
            return None
        groups = group_components(rows)
        if not groups:
            return None
        p = .95;queried = float(torch.rand((), generator=self.g)) < p
        b = dict(route='wm' if queried else 'policy', sample_probability=p, mixture_mass=p, regime='ordinary',
                 executed=False, beta=0., q=0., eligible_query=True)
        sample['behaviour'] = b
        if not queried:
            return None
        group = next((g for g in groups if self.offers.get(int(now)) in g['members']), None)
        if group is None:
            result = dict(accepted=False)
        else:
            members = group['members']
            plan = dict(id=f'P{int(now)}', members=members, duration_s=30, knot_s=10, family='MPPI',
                        knots=[[self.accel]*4 for _ in members])
            result = dict(accepted=True, plan=plan, option='MPPI', expected_saving=1.)
            if self.arm == 'R':
                result['random_match'] = dict(path='x', key=str(now))
                result['seed'] = int(torch.randint(2**31, (), generator=self.g))
        b.update(proposal_accepted=result['accepted'])
        result['behaviour'] = b
        return result


class ScriptedGuide:
    """Independent oracle: the original's draw schedule; from t0 on, proposals are never executed."""

    def __init__(self, generator, original, t0, arm):
        self.g, self.t0, self.arm = generator, t0, arm
        self.by_time = {int(s['time_s']): s for s in original['steps']}

    def __call__(self, rows, commands, now, sample):
        s = self.by_time.get(int(now))
        if s is None:
            return None
        b = s['sample'].get('behaviour') or {}
        if not b.get('eligible_query'):
            return None
        torch.rand((), generator=self.g)
        sample['behaviour'] = copy.deepcopy(b)
        p = s.get('proposal')
        if p is None or not p.get('accepted'):
            return None
        if p.get('random_match') is not None:
            torch.randint(2**31, (), generator=self.g)
        out = copy.deepcopy(p);out['behaviour'] = sample['behaviour']
        if int(now) == self.t0:      # only the graded plan is removed; later plans execute as recorded
            out['option'] = 'NOEXEC'
        return out


def policy():
    torch.manual_seed(5);p = Policy()
    torch.nn.init.normal_(p.actor[-1].weight, std=.05);torch.nn.init.constant_(p.actor[-1].bias, .05)
    torch.nn.init.constant_(p.log_std, math.log(.1))
    return p


def run_episode(root, ident, weights, noise_seed, guide_factory, arm='B', stop=STOP, beta=1.):
    from .accel_plant import episode
    from .accel_storage import read
    job = dict(id=ident, path=str(Path(root)/'jobs'/ident), world=WORLD, role='train', arm=arm, weights=weights,
               runtime='', checkpoint='', noise_seed=noise_seed, deterministic=False, sim_s=stop, stop=stop,
               queue_root=str(root), accel_bounds=[-3., 2.6], gpu_deadline=0, guided=False, beta=beta,
               warm_history=False)
    with patch.object(plan_client, 'guide', guide_factory):
        episode(job)
    return read(Path(root)/'jobs'/ident/'RESULT.json')


class Fixture:
    def __init__(self, tmp, arm='B', accel=-3., offers=None):
        from .accel_learner import save
        self.tmp, self.arm = Path(tmp), arm
        self.policy = policy();self.weights = self.policy.state_dict()
        self.ckpt = self.tmp/'run'/f'{arm}_checkpoint_000.pt';self.ckpt.parent.mkdir(parents=True)
        save(self.ckpt, self.policy, torch.optim.Adam(self.policy.parameters()), torch.Generator().manual_seed(3), 0, 0)
        self.noise_seed = xg.train_noise_seed(WORLD['seed'], 1, 20261101)
        self.ident = f'warm_u001_{arm}_{WORLD["seed"]}'
        with fake_sumo():
            self.original = run_episode(self.tmp/'run', self.ident, self.weights, self.noise_seed,
                                        lambda job, g: FakeGuide(g, arm, accel, offers), arm)
        self.meta = dict(run_root=str(self.tmp/'run'), result_path=str(self.tmp/'run'/'jobs'/self.ident/'RESULT.json'),
                         result_sha256=digest(self.original), update=1, arm=arm, checkpoint_path=str(self.ckpt),
                         checkpoint_sha256=sha(self.ckpt), checkpoint_update=0, noise_seed=self.noise_seed,
                         runtime='', sumo_checkpoint='', beta=1., warm_history=False, accel_bounds=[-3., 2.6])

    def specs(self, **kw):
        return xg.job_specs(self.original, self.meta, **kw)

    def queue(self, **kw):
        return xg.GradeQueue(self.tmp/'grader', **kw)

    def oracle(self, t0, name):
        beta = dict(beta=1., buckets={'NOEXEC|ordinary': dict(beta=0.)})
        with fake_sumo():
            return run_episode(self.tmp/'oracle', name, self.weights, self.noise_seed,
                               lambda job, g: ScriptedGuide(g, self.original, t0, self.arm), self.arm, beta=beta)


def receipt(gid, g, world=1, status='graded', option='MPPI', arm='B', finished=0.):
    body = dict(schema=xg.SCHEMA, grade_id=gid, status=status, grade_veh_s=g if status == 'graded' else None,
                arm=arm, finished_unix=finished,
                world_seed=world, episode_id='e'+gid, plan=dict(id='p'+gid, start_s=300, duration_s=2, option=option,
                                                               regime='ordinary'))
    return xg._seal(body, 'receipt_sha256')


class ExactGradeCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self._tmp = tempfile.TemporaryDirectory();self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def grade_all(self, fx, **kw):
        q = fx.queue(**kw);ids = [s['grade_id'] for s in fx.specs() if q.enqueue(s)]
        with fake_sumo():
            out = q.drain(xg.AccelPlantGradeRunner(fx.tmp/'grader'), xg.load_original_from_disk)
        return q, ids, out

    def check_against_oracle(self, arm):
        fx = Fixture(self.tmp, arm)
        starts = xg.executed_plan_steps(fx.original)
        self.assertGreaterEqual(len(starts), 2, 'fixture must execute plans')
        q, ids, receipts = self.grade_all(fx)
        self.assertEqual(len(receipts), len(starts))
        for r in receipts:
            self.assertEqual(r['status'], 'graded', r.get('reason'))
            xg.verify_sealed(r, 'receipt_sha256')
            t0, t_end = r['window']['start_s'], r['window']['end_s']
            self.assertTrue(r['noise']['crn_verified_t0']);self.assertEqual(r['noise']['noise_mode_used'], 'crn')
            oracle = fx.oracle(t0, f'oracle_{arm}_{t0}')
            for t in range(300, t_end):
                self.assertEqual(oracle['prefixes'][str(t)]['metrics'],
                                 q_branch(fx, r)['prefixes'][str(t)]['metrics'], f't={t}')
            J = lambda res: xg.totals_at(res, t_end)['team_delay_vs']-xg.totals_at(res, t0)['team_delay_vs']
            self.assertEqual(r['J_without_veh_s'], J(oracle));self.assertEqual(r['J_with_veh_s'], J(fx.original))
            self.assertEqual(r['grade_veh_s'], J(oracle)-J(fx.original))
            self.assertEqual(r['charged_sim_s'], t_end)
            self.assertEqual(r['noise']['replayed_plans'], {str(o['id']): o['seconds'] for o in r['replayed_plans']})
        self.assertTrue(any(r['replayed_plans'] for r in receipts), 'fixture must replay another plan')
        return fx, q, receipts

    def test_prefix_identical_grade_equals_independent_oracle_arm_B(self):
        fx, q, receipts = self.check_against_oracle('B')
        self.assertTrue(any(r['grade_veh_s'] != 0 for r in receipts))
        last = max(receipts, key=lambda r: r['window']['start_s'])
        self.assertTrue(last['window']['truncated']);self.assertEqual(last['window']['end_s'], STOP)
        lines = (fx.tmp/'grader'/'CHARGES.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), len(receipts))

    def test_prefix_identical_grade_equals_independent_oracle_arm_R_randint_draws(self):
        fx, _, _ = self.check_against_oracle('R')
        self.assertTrue(any('random_match' in (s.get('proposal') or {}) for s in fx.original['steps']))

    def test_prefix_mismatch_is_a_logged_stop_never_imputed_or_retried(self):
        fx = Fixture(self.tmp);q = fx.queue();spec = fx.specs()[0];q.enqueue(spec)
        calls = []
        class Spy(xg.AccelPlantGradeRunner):
            def run(self, s, o):
                calls.append(1);return super().run(s, o)
        with fake_sumo(perturb_at=spec['window']['start_s']-5):
            r = q.run_one(spec['grade_id'], Spy(fx.tmp/'grader'), xg.load_original_from_disk)
            again = q.run_one(spec['grade_id'], Spy(fx.tmp/'grader'), xg.load_original_from_disk)
        self.assertEqual(r['status'], 'stopped');self.assertIsNone(r['grade_veh_s'])
        self.assertIn('PrefixMismatch', r['reason']);self.assertFalse(r['prefix']['pass_'])
        self.assertEqual(again, r);self.assertEqual(len(calls), 1)
        w = xg.imitation_weights([r], 'B', dict(temperature_veh_s=10.))
        self.assertEqual(w['weights'], {});self.assertEqual(w['counts']['stopped'], 1)

    def test_post_hoc_fingerprint_check_catches_runner_without_in_loop_check(self):
        fx = Fixture(self.tmp);spec = fx.specs()[0]
        bad = copy.deepcopy(fx.original);bad['prefixes'][str(spec['window']['start_s']-1)]['tape'] = 'x'
        runner = types.SimpleNamespace(run=lambda s, o: bad)
        with self.assertRaises(xg.PrefixMismatch):
            xg.grade(spec, fx.original, runner)

    def test_without_plan_branch_follows_policy_own_actions_with_shared_noise(self):
        fx = Fixture(self.tmp, accel=-.5)
        q, _, receipts = self.grade_all(fx)
        for r in receipts:
            br = q_branch(fx, r);t0, t_end = r['window']['start_s'], r['window']['end_s']
            orig = {int(s['time_s']): s for s in fx.original['steps']}
            self.assertEqual(r['noise']['aligned_until_s'], t_end)
            for s in br['steps']:
                t = int(s['time_s'])
                if t < t0:
                    self.assertEqual(s['requests'], orig[t]['requests']);continue
                smp = s['sample'];self.assertFalse(smp['behaviour']['executed'])
                replay = smp.get('absolute_accels', {})
                if t < t0+r['plan']['duration_s']:
                    self.assertEqual(replay, {})
                else:                       # another plan: replayed exactly as recorded
                    self.assertEqual(replay, orig[t]['sample'].get('absolute_accels', {}))
                    self.assertFalse(set(replay) & set(r['plan']['members']))
                for v, (a,) in zip(smp['context']['members'], smp['own_action']):
                    want = replay[v] if v in replay else s['commands'][v]+a
                    self.assertEqual(s['requests'][v]['requested_raw'], want)
                    if v in r['plan']['members']:
                        self.assertNotIn(v, replay)
                with torch.no_grad():
                    mb = fx.policy(smp['context'])[0];mo = fx.policy(orig[t]['sample']['context'])[0]
                nb = [a-float(m) for (a,), m in zip(smp['own_action'], mb[:, 0])]
                no = [a-float(m) for (a,), m in zip(orig[t]['sample']['own_action'], mo[:, 0])]
                for x, y in zip(nb, no):
                    self.assertAlmostEqual(x, y, places=5)
            plan_t = [t for t in range(t0, t0+r['plan']['duration_s'])]
            self.assertTrue(all('absolute_accels' in orig[t]['sample'] for t in plan_t))

    def test_membership_divergence_recorded_and_crn_failure_falls_back_to_mean(self):
        fx = Fixture(self.tmp, accel=-3.)
        q, _, receipts = self.grade_all(fx)
        self.assertTrue(any(r['noise']['aligned_until_s'] < r['window']['end_s'] for r in receipts))
        fx.meta['noise_seed'] += 1
        spec = fx.specs()[0]
        with fake_sumo():
            body = xg.grade(spec, fx.original, xg.AccelPlantGradeRunner(fx.tmp/'grader2'))
        self.assertEqual(body['status'], 'graded');self.assertFalse(body['noise']['crn_verified_t0'])
        self.assertEqual(body['noise']['noise_mode_used'], 'mean')

    def test_window_and_J_arithmetic_and_budget(self):
        steps = []
        for t in range(300, 1200):
            b = dict(executed=t in range(880, 1180) or t in range(400, 460), plan_start=t in (400, 880), plan_id='a' if t < 700 else 'b',
                     option='MPPI', regime='ordinary')
            prop = dict(plan=dict(id=b['plan_id'], members=['c1'], duration_s=300 if t >= 700 else 60)) if b['plan_start'] else None
            steps.append(dict(time_s=t, sample=dict(behaviour=b, context=dict(members=['c1'])), proposal=prop))
        fps = lambda rate: {str(t): dict(t_s=float(t), metrics=dict(team_delay_vs=float(rate*t).hex())) for t in range(300, 1200)}
        orig = dict(id='ep', world=WORLD, steps=steps, prefixes=fps(2.),
                    final=dict(t_s=1200., metrics=dict(team_delay_vs=float(2400.).hex())))
        meta = dict(update=3, arm='B', checkpoint_path='c', checkpoint_sha256='s', noise_seed=1)
        a, b = xg.job_specs(orig, meta)
        self.assertEqual((a['window']['start_s'], a['window']['end_s'], a['window']['truncated'], a['sim_s']), (400, 580, False, 580))
        self.assertEqual((b['window']['end_s'], b['window']['nominal_end_s'], b['window']['truncated'], b['sim_s']), (1200, 1300, True, 1200))
        self.assertEqual(a['policy']['checkpoint_update'], 2);self.assertEqual(a['overlapping_plan_starts'], [])
        def branch(t0):
            val = lambda t: 2.*t if t <= t0 else 2.*t0+3.*(t-t0)
            return dict(prefixes={k: (v if int(k) <= t0 else dict(v, metrics=dict(team_delay_vs=val(int(k)).hex())))
                                  for k, v in orig['prefixes'].items()},
                        final=dict(t_s=1200., metrics=dict(team_delay_vs=val(1200).hex())))
        run = types.SimpleNamespace(run=lambda s, o: branch(s['window']['start_s']))
        body = xg.grade(a, orig, run)
        self.assertEqual((body['J_with_veh_s'], body['J_without_veh_s'], body['grade_veh_s']), (360., 540., 180.))
        body = xg.grade(b, orig, run)
        self.assertEqual((body['J_with_veh_s'], body['J_without_veh_s'], body['grade_veh_s']), (640., 960., 320.))
        with self.assertRaises(xg.PrefixMismatch):
            xg.grade(b, orig, types.SimpleNamespace(run=lambda s, o: branch(400)))
        self.assertEqual(xg.quote([(300, 60), (900, 300), (850, 100)]), 480+1200+1070)
        self.assertEqual(xg.budget_sim_s(a), 580)
        self.assertEqual(xg.remap('/gpu/run/jobs/x', {'/gpu': '/a', '/gpu/run': '/b'}), '/b/jobs/x')
        with self.assertRaises(ValueError):
            xg.job_specs(orig, dict(meta, arm='A'))
        tampered = dict(a, sim_s=1)
        with self.assertRaises(ValueError):
            xg.grade(tampered, orig, run)

    def test_only_exact_positive_grades_imitated_uniform_then_frozen_temperature_whole_buffer(self):
        rs = [receipt('a', 5.), receipt('b', 0.), receipt('c', -4.), receipt('d', None, status='stopped'),
              receipt('e', None, status='failed'), receipt('f', 1000.), receipt('g', None, status='overlap_unresolved'),
              receipt('h', 9., arm='R')]
        T = xg.temperature(rs, 'B')
        self.assertEqual((T['rule'], T['temperature_veh_s']), ('uniform_before_freeze', None))
        w = xg.imitation_weights(rs, 'B', T)
        self.assertEqual({k: v['weight'] for k, v in w['weights'].items()}, {'a': 1., 'f': 1.})
        self.assertEqual(w['counts'], dict(graded=4, positive=2, nonpositive=2, stopped=1, failed=1,
                                           overlap_unresolved=1, other_arm=1))
        many = [receipt(f'm{i:02}', float(i-5), world=i % 3, finished=100.-i) for i in range(25)]
        T2 = xg.temperature(many, 'B')       # first 20 by completion time = i in 24..5
        first = [float(i-5) for i in range(24, 4, -1)]
        self.assertEqual(T2['grade_ids'][0], 'm24');self.assertEqual(len(T2['grade_ids']), 20)
        self.assertAlmostEqual(T2['temperature_veh_s'], math.sqrt(sum(g*g for g in first)/20))
        self.assertEqual(xg.temperature(many[:19], 'B')['temperature_veh_s'], None)
        self.assertEqual(xg.temperature([receipt(f'z{i}', .1, finished=i) for i in range(20)], 'B')['temperature_veh_s'], 1.)
        w2 = xg.imitation_weights(many+rs, 'B', T2)  # whole positive buffer, every update
        self.assertEqual(set(w2['weights']), {f'm{i:02}' for i in range(6, 25)} | {'a', 'f'})
        self.assertEqual(w2['weights']['f']['weight'], 20.)
        self.assertAlmostEqual(w2['weights']['m06']['weight'], math.exp(1./T2['temperature_veh_s']))
        with self.assertRaises(ValueError):
            xg.imitation_weights([dict(rs[0], grade_veh_s=99.)], 'B', T)
        # Durable freeze: first write wins; later grades never move T.
        q = xg.GradeQueue(Path(self.tmp)/'gq')
        q.receipts = lambda: many[:19]
        self.assertIsNone(q.frozen_temperature('B')['temperature_veh_s'])
        q.receipts = lambda: many
        frozen = q.frozen_temperature('B')
        q.receipts = lambda: many+[receipt('late', 500., finished=-1.)]
        self.assertEqual(q.frozen_temperature('B'), frozen)
        self.assertTrue((Path(self.tmp)/'gq'/'TEMPERATURE_B_None.json').exists())
        with self.assertRaises(ValueError):
            xg.temperature([], 'R', frozen=frozen)

    def test_exact_update_imitates_graded_pairs_only_no_plan_ratios(self):
        fx = Fixture(self.tmp)
        q, _, receipts = self.grade_all(fx)
        r = dict(receipts[0], status='graded', grade_veh_s=7.);r = xg._seal({k: v for k, v in r.items() if k != 'receipt_sha256'}, 'receipt_sha256')
        entry = xg.imitation_weights([r], r['arm'], dict(temperature_veh_s=7.))['weights'][r['grade_id']]
        pairs = xg.imitation_pairs(fx.original, entry)
        self.assertEqual(len(pairs), r['plan']['duration_s']);self.assertTrue(all(w == math.e for _, w in pairs))
        p = policy();orig_score = p.score;scored = []
        def score(s):
            self.assertFalse(s.get('behaviour', {}).get('executed'));scored.append(1);return orig_score(s)
        current = dict(steps=[s for s in fx.original['steps'][:40]])
        with patch.object(p, 'score', side_effect=score):
            m = xg.exact_update(p, torch.optim.Adam(p.parameters(), lr=3e-4), [current], torch.Generator().manual_seed(1),
                                pairs+[(pairs[0][0], 0.)], epochs=1)
        own = sum(not (s['sample'].get('behaviour') or {}).get('executed') for s in current['steps'])
        self.assertEqual(sum(x['own_policy_steps'] for x in m), own);self.assertEqual(len(scored), own)
        self.assertEqual(sum(x['critic_steps'] for x in m), 40)
        self.assertEqual(sum(x['exact_imitation_steps'] for x in m), len(pairs))
        self.assertNotEqual(sum(x['imitation'] for x in m), 0.)
        p2 = policy();m2 = xg.exact_update(p2, torch.optim.Adam(p2.parameters(), lr=3e-4), [current],
                                            torch.Generator().manual_seed(1), [], epochs=1)
        self.assertEqual(sum(x['imitation'] for x in m2), 0.)

    def test_queue_resume_idempotence_and_orphan_recovery(self):
        fx = Fixture(self.tmp);specs = fx.specs()
        alive = {'v': True}
        q = fx.queue(alive=lambda c: alive['v'])
        self.assertTrue(q.enqueue(specs[0]));self.assertFalse(q.enqueue(specs[0]))
        with self.assertRaises(ValueError):
            q.enqueue(xg._seal(dict({k: v for k, v in specs[0].items() if k != 'spec_sha256'}, sim_s=1), 'spec_sha256'))
        gid = specs[0]['grade_id']
        claim = q.claim(gid)                       # a live worker holds it
        self.assertEqual(q.status(gid), 'running');self.assertEqual(q.pending(), []);self.assertEqual(q.recover(), [])
        with self.assertRaises(FileExistsError):
            q.claim(gid)
        alive['v'] = False                         # worker died (e.g. spot reclaim)
        self.assertEqual(q.status(gid), 'orphaned');self.assertEqual(q.recover(), [gid])
        self.assertEqual(q.pending(), [gid]);self.assertEqual(len(q.attempts(gid)), 1)
        charged = []
        q.charge = lambda s, a: charged.append((s['grade_id'], a))
        with fake_sumo():
            r = q.run_one(gid, xg.AccelPlantGradeRunner(fx.tmp/'grader'), xg.load_original_from_disk)
        self.assertEqual((r['status'], r['attempt']), ('graded', 2));self.assertEqual(charged, [(gid, 2)])
        # Resume: a fresh queue object on the same root sees the durable receipt; nothing reruns.
        q2 = fx.queue(alive=lambda c: False);q2.charge = lambda s, a: charged.append('again')
        self.assertEqual(q2.run_one(gid, None, None), r);self.assertEqual(q2.recover(), [])
        self.assertEqual(charged, [(gid, 2)]);self.assertEqual(q2.receipts(), [r])
        del claim

    def test_attempts_exhausted_and_executor_error_are_failed_not_retried(self):
        fx = Fixture(self.tmp);specs = fx.specs();q = fx.queue(alive=lambda c: False)
        for s in specs[:2]:
            q.enqueue(s)
        g0, g1 = specs[0]['grade_id'], specs[1]['grade_id']
        for _ in range(xg.MAX_ATTEMPTS):
            q.claim(g0);q.recover()
        self.assertEqual(q.status(g0), 'failed');self.assertIsNone(q.receipt(g0)['grade_veh_s'])
        boom = types.SimpleNamespace(run=lambda s, o: (_ for _ in ()).throw(RuntimeError('executor fault')))
        r = q.run_one(g1, boom, lambda s: fx.original)
        self.assertEqual(r['status'], 'failed');self.assertIn('executor fault', r['reason'])
        self.assertEqual(q.run_one(g1, boom, lambda s: fx.original), r)
        lines = (fx.tmp/'grader'/'CHARGES.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 1)            # charged before physics; exhausted claims were never run

    def test_budget_blocked_does_not_run_or_impute(self):
        fx = Fixture(self.tmp);spec = fx.specs()[0]
        def refuse(s, a):
            raise xg.BudgetBlocked('whole-wave budget blocked')
        q = fx.queue(charge=refuse);q.enqueue(spec)
        with self.assertRaises(xg.BudgetBlocked):
            q.run_one(spec['grade_id'], types.SimpleNamespace(run=lambda s, o: self.fail('ran')), lambda s: fx.original)
        self.assertIsNone(q.receipt(spec['grade_id']));self.assertTrue((fx.tmp/'grader'/'GRADER_BLOCKED.json').exists())
        self.assertEqual(q.status(spec['grade_id']), 'pending')

    def test_overlap_with_shared_members_is_unresolved_never_run_or_imitated(self):
        fx = Fixture(self.tmp, accel=-.5, offers={320: 'c1', 400: 'c2'})
        specs = {s['window']['start_s']: s for s in fx.specs()}
        self.assertTrue(specs[320]['overlap_unresolved']);self.assertEqual(specs[320]['sim_s'], 0)
        self.assertTrue(specs[320]['overlap_clash_members'])
        self.assertFalse(specs[400]['overlap_unresolved'])
        q = fx.queue(charge=lambda s, a: self.fail('charged'))
        q.enqueue(specs[320])
        r = q.run_one(specs[320]['grade_id'], types.SimpleNamespace(run=lambda s, o: self.fail('ran')), None)
        self.assertEqual((r['status'], r['grade_veh_s']), ('overlap_unresolved', None))
        self.assertEqual(xg.imitation_weights([r], 'B', dict(temperature_veh_s=None))['weights'], {})
        self.assertEqual(xg.trust_evidence([r], 1.)['overlap_unresolved'], 1)
        with self.assertRaises(ValueError):
            xg.grade(specs[320], fx.original, None)

    def test_smoke_grades_one_real_episode_plan_once(self):
        fx = Fixture(self.tmp, accel=-.5)
        binding = fx.tmp/'binding.json'
        binding.write_text('{"runtime": "", "checkpoint": "", "accel_bounds": [-3.0, 2.6], "warm_history": false}')
        with fake_sumo():
            r = xg.smoke(fx.tmp/'run', fx.ident, fx.tmp/'smoke', binding, training_seed=20261101, charge='local')
            again = xg.smoke(fx.tmp/'run', fx.ident, fx.tmp/'smoke', binding, training_seed=20261101, charge='local')
        self.assertEqual(r['status'], 'graded');self.assertEqual(again, r)
        out = xg.summary(r)
        self.assertTrue(out['prefix_pass']);self.assertTrue(out['crn_verified_t0'])
        self.assertEqual(len(list((fx.tmp/'smoke'/'receipts').glob('*.json'))), 1)
        with self.assertRaises(ValueError):
            xg.smoke(fx.tmp/'run', 'final_B_1', fx.tmp/'smoke', binding, training_seed=20261101, charge='local')

    def test_trust_evidence_and_harm_guard(self):
        good = [receipt(f'g{i}', 3.+i % 2, world=i % 5) for i in range(10)]
        t = xg.trust_evidence(good, 1.)
        self.assertGreater(t['mean'], 0);self.assertEqual(t['beta'], 1.);self.assertFalse(t['harm_guard']['pause'])
        bad = [receipt(f'b{i}', -5.-i % 3, world=i % 6) for i in range(30)]
        t = xg.trust_evidence(bad+[receipt('s', None, status='stopped')], 1.)
        self.assertLess(t['ci95'][1], 0);self.assertTrue(t['harm_guard']['pause']);self.assertEqual(t['beta'], 0.)
        self.assertEqual(t['stopped'], 1);self.assertEqual(t['buckets']['MPPI|ordinary']['beta'], 0.)
        self.assertFalse(xg.trust_evidence(bad[:29], 1.)['harm_guard']['pause'])
        self.assertIsNone(xg.trust_evidence(good[:1], .7)['ci95'])

    def test_rng_schedule_matches_controller_and_guide(self):
        step = dict(sample=dict(behaviour=dict(eligible_query=True)), proposal=dict(accepted=True, random_match={}))
        self.assertEqual(xg.guide_draws(step, 'R'), ['rand', 'randint', 'rand'])
        self.assertEqual(xg.guide_draws(dict(step, proposal=dict(accepted=False)), 'B'), ['rand'])
        self.assertEqual(xg.guide_draws(dict(sample=dict(behaviour=dict(route='plan')), proposal=None), 'B'), [])


def q_branch(fx, r):
    from .accel_storage import read
    return read(fx.tmp/'grader'/'jobs'/r['grade_id']/f'attempt{r["attempt"]}'/'RESULT.json')


if __name__ == '__main__':
    unittest.main()
