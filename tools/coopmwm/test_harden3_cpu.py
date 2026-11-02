"""HARDEN-3 regressions on the integrated branch (CPU only, no SUMO/GPU)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import capacity_broker as cb
from .protocol import digest
from .query_errors import plan_shape


def record(group=('a', 'b')):
    r = dict(world_id=1, time_s=300, rows=[], group=list(group), commands={}, policy_residuals={})
    r['state_id'] = digest(dict(record=r, job='j'))
    return r


class PlanShapeTests(unittest.TestCase):
    def good(self, r):
        return dict(accepted=True, record_sha256=digest(r),
                    topk=[dict(id='c', members=['a', 'b'], duration_s=60)],
                    plan=dict(id='p', members=['a', 'b'], duration_s=60, profile=[[0.]*60]*2))

    def test_accepted_plan_shape(self):
        r = record()
        self.assertIsNone(plan_shape(r, self.good(r)))
        self.assertIsNone(plan_shape(r, dict(accepted=False, record_sha256=digest(r))))
        for bad in (dict(self.good(r), plan=dict(id='broken')), dict(self.good(r), topk=[dict(members=['z'])]),
                    dict(self.good(r), plan=dict(self.good(r)['plan'], duration_s=60.5))):
            self.assertIsNotNone(plan_shape(r, bad))

    def test_broker_declines_shape_fault_instead_of_publishing_it(self):
        r = record()
        broken = dict(self.good(r), plan=dict(id='broken'))
        result, problem = cb.forecast_result('plan', r, dict(result=broken, wall_s=1.))
        self.assertIsNotNone(problem)
        self.assertFalse(result['accepted']);self.assertEqual(result['record_sha256'], digest(r))
        ok, problem = cb.forecast_result('plan', r, dict(result=self.good(r), wall_s=1.))
        self.assertIsNone(problem);self.assertTrue(ok['accepted'])


class ResumeConfigTests(unittest.TestCase):
    def test_parallel_fanout_bounds_submissions(self):
        from . import parallel_groups as pg
        import inspect
        src = inspect.getsource(pg.query)
        self.assertIn("groups[:max(1,int(factor*len(workers))-pending)]", src)


class RehearsalSmoke(unittest.TestCase):
    """Real orchestration end to end in a subprocess (module patches stay isolated)."""

    def test_two_updates_complete_without_stop(self):
        with tempfile.TemporaryDirectory() as work:
            out = Path(work)/'result.json'
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            top = Path(__file__).resolve().parents[2]
            subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.coopmwm.rehearsal_cpu', 'a_clean',
                            '--workdir', work, '--out', str(out), '--set', 'episodes_per_update=2',
                            '--set', 'N_exec=1', '--set', 'verify_workers=3', '--set', 'latency_s=0.2',
                            '--set', 'timeout_s=2400'],
                           cwd=top, env=env, check=True, capture_output=True, timeout=2700)
            v = json.loads(out.read_text())
            self.assertEqual(v['outcome'].get('B'), 'complete', v['failures'])
            self.assertEqual(v['outcome'].get('R'), 'complete', v['failures'])
            self.assertFalse(v['deadlock_or_idle']);self.assertEqual(v['stops'], [])
            for key in ('B4', 'B5', 'R4', 'R5'):
                self.assertTrue(v['updates'][key]['checkpoint'])
            self.assertTrue(v['verifications']['all_charges_equal_k_plus_1_branches'])
            self.assertTrue(v['verifications']['all_within_quote'])



class LabelList8Tests(unittest.TestCase):
    def test_top8_unfiltered_from_the_original_final_k8_calls(self):
        from .mppi_runtime import label_list8_search
        seen = {}

        def fake_search(record, group, policy, bounds, evaluate, regime, **kw):
            seen['config'] = kw['config']
            plans = [dict(id=f'p{i}', profile=[[0.]]) for i in range(9)]
            acts = plans+[dict(id='policy'), dict(id='MAPPO')]
            evaluate(record, acts, 4, 300)            # screen call: ignored
            evaluate(record, acts, 8, 420)            # final K8 call: labelled
            return dict(accepted=False, plan=plans[0], topk=[])

        def evaluate(record, actions, k, h):
            # p_i costs 100+i; policy 50 (every candidate LOSES vs policy): still kept.
            return {a['id']: [dict(team_delay=(50. if a['id'] == 'policy' else 60. if a['id'] == 'MAPPO'
                                               else 100.+int(a['id'][1:])))]*k for a in actions}
        out = label_list8_search(fake_search)({}, {}, None, None, evaluate, 'ordinary', config=dict(x=1))
        self.assertEqual(seen['config'], dict(x=1, top_k=8, finalists_per_k=1))
        self.assertEqual([p['id'] for p in out['topk']], [f'p{i}' for i in range(8)])
        self.assertEqual(out['available_k'], 8)
        self.assertTrue(all(p['saving_vs_policy'] < 0 for p in out['topk']))


if __name__ == '__main__':
    unittest.main()
