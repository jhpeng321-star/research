"""Observer ownership guards; no real process control or remote calls."""
import unittest
from unittest.mock import Mock, patch
from . import ls_validity_ops as ops


class ObserverTests(unittest.TestCase):
    def test_fast_observer_reused_pid_is_never_signalled(self):
        owner=Mock();owner.create_time.return_value=200.
        exists=lambda p:p.name!='LS_VALIDITY_OBSERVER_DISPATCH_V7.json'
        with patch('pathlib.Path.exists',exists),patch.object(ops,'read',return_value=dict(pid=123,unix=100.)), \
             patch('psutil.Process',return_value=owner),patch('subprocess.Popen') as launch:
            with self.assertRaisesRegex(AssertionError,'creation identity'):ops.fast_watch_dispatch()
        owner.terminate.assert_not_called();launch.assert_not_called()

    def test_fast_observer_existing_dispatch_is_adopted(self):
        with patch('pathlib.Path.exists',return_value=True),patch.object(ops,'read',return_value=dict(pid=123,unix=100.)), \
             patch('psutil.Process') as lookup,patch('subprocess.Popen') as launch:
            ops.fast_watch_dispatch()
        lookup.assert_not_called();launch.assert_not_called()

    def test_reused_observer_pid_is_never_signalled(self):
        owner=Mock();owner.create_time.return_value=200.
        old=dict(pid=123,unix=100.)
        exists=lambda p: p.name!='LS_VALIDITY_OBSERVER_DISPATCH_V6.json'
        with patch('pathlib.Path.exists',exists),patch.object(ops,'read',return_value=old), \
             patch('psutil.Process',return_value=owner),patch('subprocess.Popen') as launch:
            with self.assertRaisesRegex(AssertionError,'creation identity'):
                ops.focus_watch_dispatch()
        owner.terminate.assert_not_called();launch.assert_not_called()

    def test_changed_argv_is_never_signalled(self):
        owner=Mock();owner.create_time.return_value=100.
        owner.cmdline.return_value=['python','another_task']
        exists=lambda p: p.name!='LS_VALIDITY_OBSERVER_DISPATCH_V6.json'
        with patch('pathlib.Path.exists',exists),patch.object(ops,'read',return_value=dict(pid=123,unix=100.)), \
             patch('psutil.Process',return_value=owner),patch('subprocess.Popen') as launch:
            with self.assertRaises(AssertionError):ops.focus_watch_dispatch()
        owner.terminate.assert_not_called();launch.assert_not_called()

    def test_existing_dispatch_is_adopted_without_new_process(self):
        with patch('pathlib.Path.exists',return_value=True), \
             patch.object(ops,'read',return_value=dict(pid=123,unix=100.)), \
             patch('psutil.Process') as lookup,patch('subprocess.Popen') as launch:
            ops.focus_watch_dispatch()
        lookup.assert_not_called();launch.assert_not_called()


if __name__=='__main__':unittest.main()
