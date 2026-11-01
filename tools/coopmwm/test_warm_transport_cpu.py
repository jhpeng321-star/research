"""Meaningful fail-closed transport tests; no torch, SSH, GPU or simulator."""
import subprocess
import unittest
from unittest.mock import Mock
from .warm_transport import connection_error,retry,validate_snapshot
from .resume_gate import check_budget


class Transport(unittest.TestCase):
    def test_only_explicit_connection_errors(self):
        for text in ('Connection closed by 1.15.42.207 port 22',
            'ssh: connect to host cpuQ port 22: Connection timed out',
            'kex_exchange_identification: read: Connection reset by peer',
            'Read from remote host 1.15.42.207: Connection reset\nclient_loop: send disconnect: Connection reset'):
            self.assertTrue(connection_error(RuntimeError(text)))
        for error in (RuntimeError('Traceback\nConnection closed by 1.15.42.207 port 22'),
            RuntimeError('parity failed'),RuntimeError('executor timeout'),RuntimeError('Permission denied'),
            subprocess.TimeoutExpired('ssh',30),ValueError('Connection closed by host port 22')):
            self.assertFalse(connection_error(error))

    def test_reverify_precedes_identical_operation_replay(self):
        events=[]
        def operation():
            events.append('operation')
            if events.count('operation')==1:raise RuntimeError('Connection closed by host port 22')
            return 'persisted'
        result=retry(operation,lambda:events.append('verify'),lambda _:events.append('record'),
            sleep=lambda _:events.append('sleep'))
        self.assertEqual(result,'persisted')
        self.assertEqual(events,['operation','record','sleep','verify','operation'])

    def test_identity_failure_blocks_replay(self):
        op=Mock(side_effect=RuntimeError('Connection closed by host port 22'))
        with self.assertRaisesRegex(ValueError,'PID'):
            retry(op,Mock(side_effect=ValueError('PID changed')),Mock(),sleep=lambda _:None)
        self.assertEqual(op.call_count,1)

    def test_five_attempt_bound(self):
        op=Mock(side_effect=RuntimeError('Connection closed by host port 22'));verify=Mock();sleeps=[]
        with self.assertRaises(RuntimeError):retry(op,verify,Mock(),sleep=sleeps.append)
        self.assertEqual(op.call_count,5);self.assertEqual(verify.call_count,4)
        self.assertEqual(sleeps,[15,30,60,120])

    def test_executor_failure_never_retried(self):
        op=Mock(side_effect=RuntimeError('Traceback: executor failed'));verify=Mock()
        with self.assertRaises(RuntimeError):retry(op,verify,Mock(),sleep=lambda _:self.fail('retry'))
        self.assertEqual(op.call_count,1);verify.assert_not_called()

    def test_reconnect_verification_consumes_deadline(self):
        op=Mock(side_effect=RuntimeError('Connection closed by host port 22'))
        with self.assertRaises(TimeoutError):
            retry(op,Mock(),Mock(),sleep=lambda _:None,clock=Mock(side_effect=[0,0,0,301]))
        self.assertEqual(op.call_count,1)

    def test_changed_or_missing_receipt_and_reused_pid_stop(self):
        before=dict(boot_id='boot',receipts={'job/CLAIM.json':'sha'},live_processes={'7':{'ticks':1}})
        after=dict(boot_id='boot',receipts={'job/CLAIM.json':'sha','job/CLOSE.json':'done'},live_processes={})
        self.assertEqual(validate_snapshot(before,after),after)
        for changed in (dict(after,boot_id='new'),dict(after,receipts={}),
            dict(after,live_processes={'7':{'ticks':2}})):
            with self.assertRaises(ValueError):validate_snapshot(before,changed)

    def test_explicit_resume_budget_cap(self):
        check_budget(dict(allocated=6000000))
        check_budget(dict(allocated=6473440))
        check_budget(dict(allocated=6500000))
        with self.assertRaisesRegex(ValueError,'1'):check_budget(dict(allocated=6500001))


if __name__=='__main__':unittest.main()
