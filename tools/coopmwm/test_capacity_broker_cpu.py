import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .capacity_broker import capacity_declines,decline,pending
from .protocol import digest


def request(key,**extras):return ('plan',dict(state_id=key,**extras))


class CapacityTests(unittest.TestCase):
    def test_breadth_first_group_index_before_reservation_time(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requests').mkdir();(root/'responses').mkdir()
            q=root/'quota_requests/u004';q.mkdir(parents=True)
            for name,stamp,keys in [('early',1.,['z0','a1']),('late',2.,['y0','b1'])]:
                (q/(name+'_REQUEST.json')).write_text(json.dumps(dict(records=[dict(state_id=k) for k in keys])))
                (q/(name+'_RESERVE.json')).write_text(json.dumps(dict(unix=stamp)))
                for key in keys:
                    (root/'requests'/(key+'.json')).write_text(json.dumps(dict(state_id=key,rows=[],time_s=300)))
            with patch('tools.coopmwm.exploration.regime',return_value='ordinary'):
                self.assertEqual([r['state_id'] for _,r in pending(root)],['z0','y0','a1','b1'])

    def test_legacy_partial_request_is_not_a_fatal_read_race(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requests').mkdir();(root/'responses').mkdir()
            p=root/'requests/a.json';p.write_text('')
            self.assertEqual(pending(root),[])
            p.write_text(json.dumps(dict(state_id='a',rows=[],time_s=300)))
            with patch('tools.coopmwm.exploration.regime',return_value='ordinary'):
                self.assertEqual(len(pending(root)),1)

    def test_stale_malformed_request_is_skipped_with_receipt_not_stop(self):
        # harden-2: a writer that died mid-write (full disk) has no waiting
        # client; one malformed file must not STOP every arm (was fatal).
        import os,time
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requests').mkdir();(root/'responses').mkdir()
            p=root/'requests/a.json';p.write_text('{"trunc');os.utime(p,(time.time()-60,)*2)
            self.assertEqual(pending(root),[])
            receipt=json.loads((root/'malformed_requests/requests__a.json').read_text())
            self.assertEqual(receipt['bytes'],7);self.assertFalse((root/'responses/a.json').exists())
            self.assertEqual(pending(root),[])  # idempotent receipt

    def test_busy_declines_only_unassigned(self):
        a=request('a');b=request('b');c=request('c')
        rejected,busy=capacity_declines([a,b,c],{'a':{'slot':'gpuA@0'},'b':{'slot':'gpuR@0'}},['gpuA@0','gpuR@0'])
        self.assertEqual(rejected,[c[1]]);self.assertEqual(len(busy),2)

    def test_free_slots_keep_priority_prefix(self):
        items=[request(str(i)) for i in range(5)]
        rejected,_=capacity_declines(items,{},['a','b','c'])
        self.assertEqual(rejected,[v[1] for v in items[3:]])

    def test_assigned_never_declined_even_over_capacity(self):
        items=[request(str(i)) for i in range(5)]
        assigned={str(i):{'slot':'a'} for i in range(4)}
        rejected,_=capacity_declines(items,assigned,['a'])
        self.assertEqual(rejected,[items[-1][1]])

    def test_diagnostic_and_centre_not_fabricated(self):
        items=[request('a',mode='diagnostic'),('centre',dict(state_id='b')),request('c')]
        rejected,_=capacity_declines(items,{},[])
        self.assertEqual(rejected,[items[-1][1]])

    def test_response_is_explicit_zero_compute_policy_route(self):
        with tempfile.TemporaryDirectory() as d,patch('subprocess.Popen',side_effect=AssertionError('no compute')):
            root=Path(d);(root/'responses').mkdir();(root/'capacity_declines').mkdir()
            record=dict(state_id='a',world_id=986000,time_s=300)
            decline(root,record,['a'],{'a'},'registered')
            result=json.loads((root/'responses/a.json').read_text())
            self.assertFalse(result['accepted']);self.assertEqual(result['record_sha256'],digest(record))
            self.assertEqual(result['imagined_futures'],0)
            self.assertEqual(result['query_admission']['wm_admission_probability'],0)
            self.assertTrue(result['query_admission']['legacy_query_probability_unchanged'])

    def test_client_busy_window_never_waits_for_capacity(self):
        import torch
        from . import plan_client as client
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'requests').mkdir();(root/'responses').mkdir()
            (root/'requests/busy.json').write_text('{}')
            job=dict(queue_root=str(root),accel_bounds=[-4.,2.6],arm='B')
            call=client.guide(job,torch.Generator().manual_seed(7));sample={}
            with patch.object(client,'group_components',return_value=[dict(members=['a','b'],size=2)]), \
                 patch.object(client,'regime',return_value='ordinary'), \
                 patch.object(client,'worker_slots',return_value=['gpuA@0']), \
                 patch.object(client.time,'sleep',side_effect=AssertionError('busy window waited')):
                self.assertIsNone(call([],{},300,sample))
            self.assertEqual(sample['behaviour']['sample_probability'],0.)
            self.assertTrue(sample['behaviour']['capacity_admission']['nonblocking'])


if __name__=='__main__':unittest.main()
