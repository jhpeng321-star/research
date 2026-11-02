"""HARDEN-3 regressions (CPU only, no SUMO/GPU): malformed results, dead-slot centres, rehearsal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from . import capacity_broker as cb
from . import centre_client
from .protocol import digest
from .query_errors import declined, malformed
from .warm_spool import immutable


def plan_record(group=('a', 'b')):
    r = dict(world_id=1, time_s=300, rows=[], group=list(group), commands={}, policy_residuals={})
    r['state_id'] = digest(dict(record=r, job='j'))
    return r


class MalformedTests(unittest.TestCase):
    def test_plan_shapes(self):
        r = plan_record()
        good = dict(accepted=True, record_sha256=digest(r), topk=[dict(members=['a', 'b'], duration_s=60)],
                    plan=dict(id='p', members=['a', 'b'], duration_s=60, profile=[[0.]*60]*2))
        self.assertIsNone(malformed('plan', r, good))
        self.assertIsNone(malformed('plan', r, dict(accepted=False, record_sha256=digest(r))))
        for bad in ([], dict(good, record_sha256='x'), dict(good, accepted='yes'),
                    dict(good, plan=dict(id='broken')), dict(good, topk=[dict(members=['z'])]),
                    dict(accepted=True, plan=dict(id='broken'), note='rehearsal')):
            self.assertIsNotNone(malformed('plan', r, bad))

    def test_declined_plan_is_policy_continues(self):
        r = plan_record();d = declined('plan', r, 'malformed GPU response: x')
        self.assertFalse(d['accepted']);self.assertEqual(d['record_sha256'], digest(r))
        self.assertIsNone(malformed('plan', r, d))

    def test_centre_shapes(self):
        r = plan_record()
        self.assertIsNone(malformed('centre', r, dict(record_sha256=digest(r), candidate_scores=0,
                                                      policy_traj=[[0.], [0.]], receipt={})))
        self.assertIsNotNone(malformed('centre', r, dict(record_sha256=digest(r), candidate_scores=3,
                                                         policy_traj=[[0.], [0.]])))
        self.assertIsNone(malformed('centre', r, declined('centre', r, 'dead')))


class DeadSlotCentreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for d in ('requests', 'responses', 'centre_requests', 'centre_responses', 'broker_assignments'):
            (self.root/d).mkdir()

    def test_centre_claimed_by_dead_slot_is_settled_and_client_continues(self):
        r = plan_record()
        immutable(self.root/'centre_requests'/(r['state_id']+'.json'), r)
        immutable(self.root/'broker_assignments'/(r['state_id']+'.json'), dict(slot='gpuA_second', kind='centre'))
        settled = cb.settle_unavailable(self.root, 'gpuA_second', [('centre', r)])
        self.assertEqual(settled, [r['state_id']])
        value = json.loads((self.root/'centre_responses'/(r['state_id']+'.json')).read_text())
        self.assertTrue(value['declined'])
        out = {}
        t = threading.Thread(target=lambda: out.update(v=centre_client.trajectory(self.root, dict(r), time.time()+5)))
        # the request already exists byte-identically; the client must return without waiting
        (self.root/'centre_requests'/(r['state_id']+'.json')).unlink()
        t.start();t.join(5)
        self.assertFalse(t.is_alive())
        self.assertIsNone(out['v'][0])

    def test_other_slot_and_answered_centres_untouched(self):
        r = plan_record()
        immutable(self.root/'broker_assignments'/(r['state_id']+'.json'), dict(slot='gpuR_primary', kind='centre'))
        self.assertEqual(cb.settle_unavailable(self.root, 'gpuA_second', [('centre', r)]), [])


class RehearsalSmoke(unittest.TestCase):
    """Real orchestration end to end in a subprocess (module patches stay isolated)."""

    def test_two_updates_complete_without_stop(self):
        with tempfile.TemporaryDirectory() as work:
            out = Path(work)/'result.json'
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            top = Path(__file__).resolve().parents[2]
            subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tools.coopmwm.rehearsal_cpu', 'a_clean',
                            '--workdir', work, '--out', str(out), '--set', 'episodes_per_update=2',
                            '--set', 'N_exec=2', '--set', 'latency_s=0.2', '--set', 'timeout_s=1500'],
                           cwd=top, env=env, check=True, capture_output=True, timeout=1800)
            v = json.loads(out.read_text())
            self.assertEqual(v['outcome'].get('B'), 'complete', v['failures'])
            self.assertEqual(v['outcome'].get('R'), 'complete', v['failures'])
            self.assertFalse(v['deadlock_or_idle']);self.assertEqual(v['stops'], [])
            for key in ('B4', 'B5', 'R4', 'R5'):
                self.assertTrue(v['updates'][key]['checkpoint'])
            self.assertTrue(v['verifications']['all_charges_equal_k_plus_1_branches'])
            self.assertTrue(v['verifications']['all_within_quote'])
            self.assertEqual(v['verifications']['charged_sim_s'], v['verifications']['ledger_verification_sim_s'])


if __name__ == '__main__':
    unittest.main()
