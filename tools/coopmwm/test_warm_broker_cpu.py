"""Durable remote broker identity, retry and shared-centre scheduling contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .warm_broker import Remote,available
from .warm_rpc import GATES,handle


class BrokerCPU(unittest.TestCase):
    def status(self):
        receipts={name:dict(pass_=True,max_cost_error=0.) for name in GATES}
        receipts.update({'SPOOL_OWNER.json':dict(pid=1,start_ticks='2',boot_id='b'),'SERVICE_OWNER.json':dict(pid=1)})
        return dict(alive=True,ready=True,failed=False,receipts=receipts)

    def test_reconnect_verifies_original_owner_before_replaying_same_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=Remote(tmp,dict(slot='gpuA@0'));remote.raw=lambda v:self.status();remote.verify()
            values=[]
            def raw(value):
                values.append(value)
                if len(values)==1:raise RuntimeError('Connection closed by host port 22')
                return self.status() if value['action']=='status' else dict(key='done')
            with patch.object(remote,'raw',side_effect=raw),patch('tools.coopmwm.warm_transport.time.sleep'):
                # Inject zero delays; the real registration retains15/30/60/120.
                from .warm_transport import retry
                result=retry(lambda:remote.raw(dict(action='submit')),remote.verify,remote.event,delays=(0,),sleep=lambda n:None)
            self.assertEqual(result,dict(key='done'))
            self.assertEqual([v['action'] for v in values],['submit','status','submit'])

    def test_changed_worker_or_gate_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=Remote(tmp,{});remote.raw=lambda v:self.status();remote.verify()
            value=self.status();value['receipts']['SPOOL_OWNER.json']['pid']=9
            remote.raw=lambda v:value
            with self.assertRaises(ValueError):remote.verify()

    def test_centres_wait_for_B_pause_then_have_lower_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('requests','responses','centre_requests','centre_responses'):(root/name).mkdir()
            record=dict(rows=[],time_s=300,state_id='a')
            (root/'requests/a.json').write_text(json.dumps(record))
            (root/'centre_requests/b.json').write_text(json.dumps(dict(record,state_id='b')))
            self.assertEqual([k for k,r in available(root)],['plan'])
            (root/'WARM_B_PAUSED.json').write_text('{}')
            self.assertEqual([k for k,r in available(root)],['plan','centre'])

    def test_RPC_rejects_arbitrary_commands_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'GPU_BINDING.json').write_text('{}')
            with self.assertRaises(ValueError):handle(root,dict(action='execute',code='anything'))
            with self.assertRaises(ValueError):handle(root,dict(action='forecast',key='../escape'))


if __name__=='__main__':unittest.main()
