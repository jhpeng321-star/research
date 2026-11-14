import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from . import exact_ops as e
from .warm_deploy import ROOT


class ObserverStatusTests(unittest.TestCase):
    def test_actual_remote_update_index_ignores_timing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'requests').mkdir()
            for name in ('A_UPDATE_048.json','B_UPDATE_001.json','B_UPDATE_001_TIMING.json','B_UPDATE_other.json'):
                (root/name).write_text('metadata does not need to be parsed')
            evidence=root/'evidence';evidence.mkdir()
            def remote(host,source,timeout):
                compile(source,'<actual status transport payload>','exec')
                if host!='cpuQ':return json.dumps(dict(alive=True,ready=True,failed=False,receipts={}))
                fake=types.ModuleType('tools.coopmwm.accel_storage');fake.read=lambda p:json.loads(p.read_text());fake.charged=lambda r:0
                paths=types.ModuleType('pathlib');paths.Path=lambda *args:root if args==(ROOT,) else Path(*args)
                out=io.StringIO();saved=sys.path[:]
                try:
                    with patch.dict(sys.modules,{'pathlib':paths,'tools.coopmwm.accel_storage':fake}),patch.object(os,'getloadavg',return_value=(0.,0.,0.),create=True),contextlib.redirect_stdout(out):exec(source,{})
                finally:sys.path[:]=saved
                return out.getvalue()
            with patch.object(e.ops.ssh,'remote',side_effect=remote),patch.object(e.ops,'EVIDENCE',evidence),contextlib.redirect_stdout(io.StringIO()):
                _,cpu=e.status()
            self.assertEqual(cpu['updates'],dict(A=48,B=1,R=0))


if __name__=='__main__':unittest.main()
