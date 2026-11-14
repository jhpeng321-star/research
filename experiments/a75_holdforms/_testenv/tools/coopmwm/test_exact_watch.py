import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import exact_watch as w


class LaunchReceipts(unittest.TestCase):
    def test_remote_dispatch_is_adopted_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w.ops,'EVIDENCE',Path(tmp)):
            owners={name:dict(owner=dict(pid=i,start_ticks=str(i)),state='S') for i,name in enumerate(w.DISPATCHES.values())}
            cpu=dict(owners=owners,stops={})
            gpus={h:dict(alive=True,ready=True,failed=False,receipts={f'STARTUP_{i}':{} for i in range(8)}) for h in ('A','R')}
            (Path(tmp)/'EXACT_RUNTIME_REGISTRATION.json').write_text('{}')
            with patch.object(w.exact_ops,'dispatch') as dispatch,patch.object(w.exact_ops,'register') as register:
                self.assertTrue(w.launch_ready(gpus,cpu));self.assertTrue(w.launch_ready(gpus,cpu))
                dispatch.assert_not_called();register.assert_not_called()
            self.assertEqual(json.loads((Path(tmp)/w.DISPATCHES['B']).read_text()),owners[w.DISPATCHES['B']]['owner'])

    def test_no_launch_before_all_gates_and_no_restart_after_pause(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(w.ops,'EVIDENCE',Path(tmp)):
            gpu=dict(alive=True,ready=True,failed=False,receipts={f'STARTUP_{i}':{} for i in range(6)})
            with patch.object(w.exact_ops,'dispatch') as dispatch,patch.object(w.exact_ops,'register') as register:
                self.assertFalse(w.launch_ready({'A':gpu,'R':gpu},dict(owners={},stops={})))
                with self.assertRaises(ValueError):w.launch_ready({'A':gpu,'R':gpu},dict(owners={},stops={'PAUSE_SAFETY.json':{}}))
                dispatch.assert_not_called();register.assert_not_called()


if __name__=='__main__':unittest.main()
