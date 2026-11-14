import base64
import hashlib
import unittest
from unittest.mock import patch
from tools import coopfree_continue_publish as p


class PublisherCPU(unittest.TestCase):
    def test_collect_is_read_only_and_source_compiles(self):
        def inspect(host,source,timeout):
            self.assertEqual(host,'cpuQ')
            compile(source,'<read-only continuation observer>','exec')
            self.assertNotIn('subprocess.Popen',source)
            self.assertNotIn('os.kill',source)
            self.assertNotIn('.write_text(',source)
            return '{}'
        with patch.object(p,'remote',side_effect=inspect):self.assertEqual(p.collect(),{})

    def test_digest_and_name_gate(self):
        data=b'checkpoint'
        item=dict(base64=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(p.decoded({'policy_hold_u002.pt':item}),{'policy_hold_u002.pt':data})
        for name in ('../secret.json','x/y.json','checkpoint.exe'):
            with self.assertRaises(ValueError):p.decoded({name:item})
        item['sha256']='0'*64
        with self.assertRaises(ValueError):p.decoded({'HOLD_UPDATE_002.json':item})

    def test_report_budget_and_no_unpublished_gain(self):
        snapshot=dict(unix=0,alive=True,charged=2384073,actual_closed=1952433,plans=[],updates=[],panels=[],checks=[],fix=None,stop=False,complete=False)
        text=p.report(snapshot)
        self.assertIn('2,384,073 precharged / 1,952,433',text)
        self.assertIn('TRAIN effects are not DEV gains',text)
        self.assertNotIn('Proposal checkpoint',text)

    def test_fix_noop_without_trigger(self):
        with patch.object(p,'remote',side_effect=AssertionError('no remote control without trigger')):
            p.authorize_fix(dict(fix=None,fix_authorized=False))
            p.authorize_fix(dict(fix={'phase':{}},fix_authorized=True))

    def test_transport_retry_is_narrow(self):
        self.assertTrue(p.transport_failure(RuntimeError('Connection closed by 1.2.3.4 port 22')))
        self.assertFalse(p.transport_failure(RuntimeError('ValueError: checksum mismatch')))


if __name__=='__main__':unittest.main()
