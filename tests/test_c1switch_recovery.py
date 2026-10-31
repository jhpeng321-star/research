"""Regression gates on the actual interrupted panel; never start SUMO or GPU."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.c1switch.recovery import expected_jobs, plan


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'tools/c1switch/evidence/V7_INTERRUPTION_AUDIT.json'
        cls.audit = json.loads(path.read_text())

    def test_registered_whole_panel_not_only_started_jobs(self):
        jobs = expected_jobs()
        self.assertEqual(len(jobs), 128+480)
        self.assertEqual(sum(x['sim_s'] for x in jobs.values()),431400)

    def test_actual_interruption_blocks_without_starting_a_process(self):
        with patch('subprocess.Popen',side_effect=AssertionError('CPU-only audit')):
            result = plan(self.audit)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertFalse(result['dispatch_authorized'])
        self.assertEqual(result['charged_sim_s'],227310)
        self.assertEqual(result['actual_closed_sim_s'],227310)
        self.assertEqual(result['projected_total_sim_s'],501960)
        self.assertEqual(len(result['complete_jobs']),241)
        self.assertEqual(len(result['partial_jobs']),23)
        self.assertEqual(result['gpu']['bb']['uncommitted_picks'],12)
        self.assertEqual(result['gpu']['wm']['uncommitted_picks'],143)
        self.assertEqual(len(result['reasons']),2)
        self.assertEqual(result['gpu']['bb']['remaining_gpu_s'],0)
        self.assertEqual(sum(x['abandoned_sim_s'] for x in result['per_arm'].values()),11100)
        self.assertEqual(result['new_sim_s'],0)
        self.assertEqual(result['new_gpu_s'],0)

    def test_completed_physics_never_enters_partial_recovery(self):
        result = plan(self.audit)
        self.assertTrue(set(result['complete_jobs']).isdisjoint(result['partial_jobs']))
        self.assertEqual(result['per_arm']['bb']['projected_total_sim_s'],180540)
        self.assertEqual(result['per_arm']['wm']['projected_total_sim_s'],175560)
        for arm in ('random','native','ctrl0','mappo','pi','circles'):
            self.assertEqual(result['per_arm'][arm]['remaining_sim_s'],14400)

    def check_rejected(self, mutate, message):
        audit = copy.deepcopy(self.audit)
        mutate(audit)
        with self.assertRaisesRegex(ValueError,message):
            plan(audit)

    def test_live_owned_worker_blocks_recovery(self):
        self.check_rejected(lambda a:a['cpuQ']['own_pids'].update({'123':True}),'still alive')

    def test_live_owned_gpu_blocks_recovery(self):
        self.check_rejected(lambda a:a['gpuR']['arms']['bb'].update(owned_pid_alive=True),'still alive')

    def test_unclosed_job_cannot_be_assumed_zero_cost(self):
        self.check_rejected(lambda a:a['cpuQ']['jobs'][0].update(close=None),'unclosed')

    def test_duplicate_charge_cannot_receive_double_credit(self):
        self.check_rejected(lambda a:a['cpuQ']['accounting'].append(a['cpuQ']['accounting'][0]),'duplicate ledger')

    def test_mismatched_actual_debit_rejected(self):
        self.check_rejected(lambda a:a['cpuQ']['jobs'][0]['close'].update(actual_sim_s=1),'actual/charged')

    def test_partial_result_cannot_be_declared_completed(self):
        def mutate(a):
            next(j for j in a['cpuQ']['jobs'] if not j['result_present'] and j['close']['actual_sim_s']>0)['result_present']=True
        self.check_rejected(mutate,'incomplete accounting')

    def test_missing_gpu_terminal_receipt_blocks_recovery(self):
        self.check_rejected(lambda a:a['gpuR']['arms']['bb'].pop('GPU_FINISHED.json'),'terminal receipt')


if __name__ == '__main__':
    unittest.main()
