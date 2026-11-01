import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from .hotadd import GATES,binding,qualified
from .warm_service import NUMERICAL


class HotaddTests(unittest.TestCase):
    def test_restricted_rpc_exposes_background_gates_without_changing_forecast(self):
        from .hotadd_rpc import handle
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in GATES[-2:]:(root/name).write_text(json.dumps(dict(pass_=True,max_cost_error=0.)))
            payload=dict(forecast={'key':'same','result':{'plan':'same'}},status={'receipts':{}})
            with patch('tools.coopmwm.hotadd_rpc.warm_rpc.handle',return_value=payload):
                result=handle(root,dict(action='forecast',key='same'))
            self.assertEqual(result['forecast'],{'key':'same','result':{'plan':'same'}})
            self.assertEqual(set(result['status']['receipts']),set(GATES[-2:]))

    def test_waits_for_fresh_exact_gates_and_pinned_owner(self):
        owner={'pid':123,'start_ticks':'456','boot_id':'same'}
        status=dict(alive=True,ready=False,failed=False,receipts={'SPOOL_OWNER.json':owner})
        self.assertFalse(qualified(status,owner))
        status['ready']=True
        with self.assertRaises(ValueError):qualified(status,owner)
        status['receipts'].update({n:dict(pass_=True,max_cost_error=0.) for n in GATES})
        self.assertTrue(qualified(status,owner))
        with self.assertRaises(ValueError):qualified(status,dict(owner,pid=999))
        status['receipts'][GATES[-1]]['max_cost_error']=1e-12
        with self.assertRaises(ValueError):qualified(status,owner)

    def test_failures_never_admit(self):
        for alive,failed in ((False,False),(True,True)):
            with self.assertRaises(ValueError):qualified(dict(alive=alive,failed=failed,ready=True,receipts={}))

    def test_batch_change_cannot_change_search_or_adopt_gates(self):
        hashes={name:'same-'+name for name in NUMERICAL}
        original=dict(hashes=hashes,deadline_unix=123456,memory_mode='warm',
            startup_qualification=True,adopted_startup_gates={'old':'gates'},batch_widths={'4':34})
        for host in ('gpuA','gpuR'):
            value=binding(original,dict(hashes),host)
            self.assertEqual(value['hashes'],hashes)
            self.assertEqual(value['deadline_unix'],123456)
            self.assertNotIn('adopted_startup_gates',value)
            self.assertNotIn('batch_widths',value)
        changed=dict(hashes);changed[NUMERICAL[0]]='changed'
        with self.assertRaises(ValueError):binding(original,changed,'gpuA')


if __name__=='__main__':unittest.main()
