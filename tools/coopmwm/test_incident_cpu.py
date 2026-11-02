"""MAIN09:44 regression: actual runtime integration, no WM/SUMO launches."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import mppi_runtime as runtime
from .query_errors import execute,validate_response


class IncidentTests(unittest.TestCase):
    def test_accepted_four_five_six_member_plans_survive_runtime(self):
        for n in (4,5,6):
            with self.subTest(n=n):
                rows=[dict(vid=str(i),s_m=-i*10.,v_mps=10.,lane=i%3,
                    is_cav=True,length_m=5.) for i in range(n)]
                record=dict(rows=rows,group=[r['vid'] for r in rows],commands={str(i):0. for i in range(n)})
                plan=dict(id='winner',members=record['group'],duration_s=30,profile=[[0.]*30]*n)
                chosen=dict(accepted=True,plan=plan,topk=[plan])
                rt=SimpleNamespace(mp=SimpleNamespace(policy=SimpleNamespace(accel_min=-3.,accel_max=2.6)))
                with patch.object(runtime,'imagined_centres',return_value=([[0.]*300]*n,[[0.]*300]*n,{})),patch.object(
                        runtime.search,'search',return_value=copy.deepcopy(chosen)),patch(
                        'tools.coopmwm.plan_proposer.receipt_from_search',return_value={}),patch(
                        'tools.coopmwm.overlap.recheck',side_effect=lambda r,v,s:v),patch(
                        'tools.coopmwm.plans.family_seeds',side_effect=AssertionError('unsupported prototypes must not run')):
                    result=runtime.plan(rt,record,None,{})
                self.assertTrue(result['accepted']);self.assertEqual(result['plan'],plan)
                self.assertFalse(result['known_family']);self.assertEqual(result['posthoc']['label'],'new')

    def test_supported_label_error_is_logged_without_changing_plan(self):
        record=dict(group=['a','b'],rows=[],commands={})
        with patch('tools.coopmwm.plans.family_seeds',side_effect=ValueError('broken label')):
            label=runtime.posthoc({},record,(-3.,2.6),[])
        self.assertEqual(label['label'],'unlabelled');self.assertIn('broken label',label['traceback'])

    def test_code_fault_declines_once_and_next_query_works(self):
        record=dict(state_id='a'*64)
        def fault():raise KeyError('bad code')
        value=execute(record,fault);validate_response(record,value)
        self.assertFalse(value['accepted']);self.assertIn('KeyError',value['query_error']['traceback'])
        self.assertTrue(value['compute_counts_unknown']);self.assertNotIn('candidates_searched',value)
        self.assertEqual(execute(record,lambda:dict(accepted=True)),dict(accepted=True))

    def test_physics_parity_accounting_validation_faults_remain_fatal(self):
        # MAIN17:26 (HARDEN-1): plain per-query validation errors now decline;
        # named run invariants and non-OOM RuntimeErrors still propagate.
        for cls,text in ((ValueError,'parity failed'),(AssertionError,'accounting'),(ValueError,'physics'),
                         (AssertionError,'safety'),(ValueError,'budget'),(RuntimeError,'fatal'),(OSError,'fatal')):
            def fault():raise cls(text)
            with self.subTest(cls=cls,text=text),self.assertRaises(cls):execute({},fault)
        for cls in (RuntimeError,OSError):
            def fault():raise cls('fatal')
            with self.subTest(cls=cls),self.assertRaises(cls):execute({},fault)
        # harden-2 (owner 17:26 'any per-query error is a decline'): a ValueError
        # stays fatal only when it names an invariant.
        for message in ('persistent worker batch parity failed','conflicting immutable spool receipt x',
                        'registered GPU deadline','accounting drift','plan response identity mismatch'):
            def fault():raise ValueError(message)
            with self.subTest(message=message),self.assertRaises(ValueError):execute({},fault)

    def test_error_response_requires_original_record_identity(self):
        def fault():raise KeyError('bad')
        value=execute({'a':1},fault)
        with self.assertRaises(ValueError):validate_response({'a':2},value)

    def test_dead_worker_requests_only_its_replacement_without_stop_or_retry(self):
        import tempfile
        import json
        from pathlib import Path
        from .capacity_broker import check_worker,WorkerUnavailable
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);endpoint=dict(slot='gpuA@0')
            with self.assertRaises(WorkerUnavailable):
                check_worker(root,endpoint,dict(alive=False,failed=True),'a'*64)
            values=list((root/'worker_restart_requests').glob('*.json'))
            self.assertEqual(len(values),1)
            self.assertEqual(json.loads(values[0].read_text())['slot'],'gpuA@0')
            self.assertFalse(list(root.glob('*STOP*')))
            check_worker(root,dict(slot='gpuR@0'),dict(alive=True,failed=False))
            with self.assertRaises(ValueError):
                check_worker(root,endpoint,dict(alive=False,failed=True,
                    receipts={'SPOOL_FAILED.json':dict(error='parity failed')}))


if __name__=='__main__':unittest.main()
