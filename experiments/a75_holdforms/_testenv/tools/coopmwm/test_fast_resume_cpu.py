"""Regression of the observed u4 wait-all deadlock and admission publish race."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch
from . import parallel_groups as pg, quota_admission as q
from .warm_spool import immutable
from .warm_history import ObservableHistory
from .protocol import digest


class FastResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('requests','responses','centre_requests','centre_responses'):
            (self.root/name).mkdir()
        self.job=dict(queue_root=str(self.root),id='fresh_B_986000',update=4,
            N_exec=10,arm='B',world=dict(seed=986000),gpu_deadline=time.time()+30,
            bounded_fanout=True)

    def test_unbound_reservations_close_global_publish_race_and_wait_consumes_no_rng(self):
        draw=lambda:0.
        first=q.bounded_batch(self.root,self.job,300,draw,2,6)
        self.assertEqual(first[2]['query_count'],4)
        other=dict(self.job,id='fresh_B_986001')
        def forbidden():raise AssertionError('capacity wait must not consume RNG')
        self.assertEqual(q.bounded_batch(self.root,other,300,forbidden,2,6),'wait')
        records=[dict(state_id=str(i)) for i in range(4)]
        q.bind_group_batch(first[3],records)
        for r in records:immutable(self.root/'requests'/(r['state_id']+'.json'),r)
        self.assertEqual(q.bounded_batch(self.root,other,300,forbidden,2,6),'wait')
        immutable(self.root/'responses/0.json',dict(accepted=False))
        next_=q.bounded_batch(self.root,other,300,draw,2,6)
        self.assertEqual(next_[2]['query_count'],1)

    def test_accepted_first_response_does_not_wait_for_slower_groups(self):
        groups=[dict(members=['a','b']),dict(members=['c','d'])]
        sample=dict(context=dict(members=['a','b','c','d']),own_action=[[0.]]*4)
        seen=[]
        def publish(path,value):
            immutable(path,value)
            if path.parent.name=='requests':
                seen.append(value)
                if len(seen)==1:
                    immutable(self.root/'responses'/path.name,dict(record_sha256=digest(value),
                        accepted=True,plan=dict(id='first',members=value['group'],duration_s=60),
                        option='MPPI',expected_saving=1.))
        with patch.object(pg,'immutable',side_effect=publish),patch.object(pg,'assert_running'),\
             patch('tools.coopmwm.plan_client.worker_slots',return_value=[1,2]),\
             patch.object(pg.time,'sleep',side_effect=AssertionError('wait-all regression')):
            result=pg.query(self.job,torch.Generator().manual_seed(0),ObservableHistory(),[],{},300,
                sample,groups,'ordinary',[1,2])
        self.assertEqual(result['plan']['id'],'first')
        self.assertEqual(len(seen),2)
        self.assertEqual(result['parallel_unconsumed_queries'],[seen[1]['state_id']])
        self.assertTrue((self.root/'query_cancellations'/(seen[1]['state_id']+'.json')).exists())
        self.assertFalse((self.root/'responses'/(seen[1]['state_id']+'.json')).exists())

    def test_concurrent_R_pins_latest_completed_quota_without_current_B(self):
        from .warm_runner import random_reference
        for u in (2,3):immutable(self.root/f'B_RANDOM_QUOTA_{u:03}.json',dict(tickets=[]))
        p=random_reference(self.root,4,True)
        self.assertEqual(p.name,'B_RANDOM_QUOTA_003.json')
        immutable(self.root/'B_RANDOM_QUOTA_004.json',dict(tickets=[]))
        self.assertEqual(random_reference(self.root,4,True),p)
        self.assertEqual(random_reference(self.root,5,True).name,'B_RANDOM_QUOTA_004.json')

    def test_closed_wave_does_not_consume_quota_or_admission_capacity(self):
        (self.root/'quota_requests/u004').mkdir(parents=True)
        immutable(self.root/'FAST_RETIRED_JOBS_1844.json',dict(jobs=[self.job['id']],query_keys=['old']))
        immutable(self.root/'requests/old.json',dict(state_id='old'))
        immutable(self.root/'quota_requests/u004/old_RESERVE.json',dict(token='old',job=self.job['id'],query_count=8))
        other=dict(self.job,id='fresh_B_986001')
        result=q.bounded_batch(self.root,other,300,lambda:0.,2,6)
        self.assertEqual(result[2]['pending'],0)
        self.assertEqual(result[2]['query_count'],4)


if __name__=='__main__':unittest.main()
