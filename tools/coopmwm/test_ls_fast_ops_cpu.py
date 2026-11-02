"""Compile the exact remote status/deployment payloads without executing them."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import ls_validity_ops as v, ls_fast_ops as f


class Captured(Exception):pass


class PayloadTests(unittest.TestCase):
    def check(self, call, reads):
        captured=[]
        def remote(host,code,**kwargs):
            compile(code,'<remote-'+host+'>','exec')
            captured.append(code)
            raise Captured
        with tempfile.TemporaryDirectory() as tmp, patch.object(v.ops,'EVIDENCE',Path(tmp)), \
             patch.object(v,'remote',remote), patch.object(f,'remote',remote), \
             patch.object(v,'read',side_effect=reads), patch.object(f,'read',side_effect=reads), \
             patch.object(f,'sha',return_value='a'*64):
            with self.assertRaises(Captured):call()
        self.assertEqual(len(captured),1)

    def test_gpu_status_payload(self):
        self.check(lambda:v.gpu_status('gpuA'),lambda p:dict(owners=[dict(root='/root/example',pid=1)]))

    def test_cpu_status_payload(self):
        self.check(v.status,lambda p:{})

    def test_broker_handover_payload(self):
        self.check(f.broker,lambda p:dict(cpu=dict(source_hashes={})))

    def test_continuation_dispatch_adopts_existing_owner(self):
        with patch('pathlib.Path.exists',return_value=True), \
             patch.object(f,'read',return_value=dict(pid=123,unix=100.)), \
             patch('subprocess.Popen') as launch:
            f.continue_dispatch()
        launch.assert_not_called()


if __name__=='__main__':unittest.main()
