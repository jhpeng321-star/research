import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import orchestrate as o


class DispatchOwnershipTests(unittest.TestCase):
    def root(self):
        # Preserve new fixtures as receipts: no deletion/cleanup, even of test files.
        parent = Path('E:/Temp/opencode')
        self.assertTrue(parent.is_dir())
        root = Path(tempfile.mkdtemp(prefix='wmvalue_dispatch_test_', dir=parent))
        for name in ('claims', 'tasks', 'logs'):
            (root/name).mkdir()
        return root

    def claim(self, root, pid):
        task = (969439, 540, 0)
        key = o.task_key(task)
        d = dict(task=list(task), host='gpuR', device='cpu', dispatcher_pid=pid,
                 remote='/root/wmvalue20261031/cells_v1/'+key+'.json')
        o.save_new(root/'claims'/(key+'.json'), d)
        return task, key

    def test_existing_claim_cannot_spawn_second_worker(self):
        root = self.root()
        task, _ = self.claim(root, os.getpid())
        with patch.object(o.subprocess, 'Popen') as spawn:
            self.assertFalse(o.run_task(root, 'test', task, 'gpuR', 'cpu', 'gate'))
            spawn.assert_not_called()

    def test_reservation_cannot_take_other_owners_or_started_ssh(self):
        for already_started in (False, True):
            root = self.root()
            pid = os.getpid() if already_started else os.getpid()+1
            task, key = self.claim(root, pid)
            if already_started:
                o.save_new(root/'claims'/(key+'.ssh.json'), dict(pid=123))
            with patch.object(o.subprocess, 'Popen') as spawn:
                with self.assertRaises(AssertionError):
                    o.run_task(root, 'test', task, 'gpuR', 'cpu', 'gate', reservation_pid=os.getpid())
                spawn.assert_not_called()

    def test_peer_gpu_pid_blocks_admission(self):
        reply = type('Reply', (), dict(returncode=0, stdout='42\n43\n'))()
        with patch.object(o.subprocess, 'run', return_value=reply):
            with patch.object(o, 'OWN_GPU_PIDS', {42}):
                self.assertFalse(o.gpu_free())
            with patch.object(o, 'OWN_GPU_PIDS', {42, 43}):
                self.assertTrue(o.gpu_free())

    def test_cpu_assist_reserves_all_new_lane_headroom(self):
        reply = type('Reply', (), dict(returncode=0, stdout=json.dumps(dict(
            cores=16., quota=25., free=16*2**30))))()
        with patch.object(o.subprocess, 'run', return_value=reply):
            self.assertFalse(o.cpu_free('gpuR', reserve_cores=10))
            self.assertTrue(o.cpu_free('gpuR', reserve_cores=2))
            self.assertFalse(o.cpu_free('gpuR', reserve_cores=2, reserve_gib=40))

    def test_partial_transfer_preserved_and_hash_bound_without_inference(self):
        from .recover_completed import recover
        root = self.root()
        task, key = self.claim(root, os.getpid())
        payload = json.dumps(dict(seed=task[0], t0=task[1], future=task[2],
                                  end_time=630, results=dict(steps=180))).encode()
        partial = root/'tasks'/(key+'.json')
        partial.write_bytes(b'{"incomplete":')
        (root/'logs'/(key+'.log')).write_text(json.dumps(dict(status='DONE',
            sha256=hashlib.sha256(payload).hexdigest()))+'\n')
        def transfer(host, remote, target):
            self.assertFalse(target.exists())
            target.write_bytes(payload)
        with patch('tools.wmvalue.recover_completed.fetch', side_effect=transfer) as copy:
            output = recover(root, key)
            self.assertEqual(copy.call_count, 1)
        self.assertEqual(partial.read_bytes(), b'{"incomplete":')
        self.assertEqual(o.task_path(root, key), output)
        with output.open('ab') as stream:
            stream.write(b' ')
        with self.assertRaisesRegex(AssertionError, 'changed'):
            o.task_path(root, key)


if __name__ == '__main__':
    unittest.main()
