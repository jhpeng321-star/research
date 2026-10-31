"""Regression gates for the observed singleton-publisher SSH outage; SUMO forbidden."""
import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

from tools import coopfree_hold_publish as publisher


class HoldPublisherRecoveryTests(unittest.TestCase):
    def test_observed_disconnect_then_success_preserves_observer(self):
        output = io.StringIO()
        with patch.object(publisher, 'once', side_effect=[
                RuntimeError('Connection closed by 1.15.42.207 port 22\n'), True
        ]) as once, patch('subprocess.Popen', side_effect=AssertionError('no dispatch')), contextlib.redirect_stdout(output):
            self.assertFalse(publisher.poll_once(True))
            self.assertTrue(publisher.poll_once(True))
        self.assertEqual(once.call_count, 2)
        self.assertTrue(json.loads(output.getvalue())['no_physics_retry'])

    def test_remote_validation_traceback_is_not_a_transport_retry(self):
        for error in (
            RuntimeError('Traceback (most recent call last):\nAssertionError: checkpoint hash'),
            RuntimeError('Traceback: Connection closed by 1.15.42.207 port 22'),
            RuntimeError('Permission denied (publickey).'),
            ValueError('existing archive hash mismatch'),
        ):
            with self.subTest(error=str(error)), patch.object(publisher, 'once', side_effect=error):
                with self.assertRaises(type(error)):
                    publisher.poll_once(False)

    def test_archive_ssh_disconnect_is_recoverable_but_tar_failure_is_not(self):
        for code, command, stderr, expected in (
            (255, ['ssh', 'cpuQ', 'tar'], b'Connection reset by 1.15.42.207 port 22\n', True),
            (2, ['ssh', 'cpuQ', 'tar'], b'tar: Cannot open: No such file', False),
            (255, ['tar'], b'Connection reset by 1.15.42.207 port 22', False),
            (255, ['ssh', 'cpuQ'], b'Permission denied (publickey).', False),
        ):
            with self.subTest(code=code, command=command, stderr=stderr):
                self.assertEqual(publisher.transport_failure(
                    subprocess.CalledProcessError(code, command, stderr=stderr)), expected)

    def test_timeout_is_recoverable_without_physics_dispatch(self):
        with patch.object(publisher, 'once', side_effect=subprocess.TimeoutExpired(['ssh'], 25)), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(publisher.poll_once(False))

    def test_success_passes_terminal_state_through(self):
        with patch.object(publisher, 'once', return_value=True):
            self.assertTrue(publisher.poll_once(True))


if __name__ == '__main__':
    unittest.main()
