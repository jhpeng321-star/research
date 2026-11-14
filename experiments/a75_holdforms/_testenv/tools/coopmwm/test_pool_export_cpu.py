"""Export correctness, full-pool guards and byte parity without WM/SUMO."""
import json
from pathlib import Path
import tempfile
import unittest
from . import pool_export as pe


class PoolExportTests(unittest.TestCase):
    def fixture(self, root, accepted=False):
        (root/'forecasts').mkdir()
        key='a'*64
        rec=dict(schema='coopmwm.plan_proposer.receipt/1',arm='B',candidates=[
            dict(id='one',stage='screen'),dict(id='two',stage='screen'),dict(id='one',stage='final')])
        payload=dict(key=key,result=dict(accepted=accepted,candidates_searched=2,proposer_receipt=rec))
        path=root/'forecasts'/(key+'.json');path.write_text(json.dumps(payload))
        return path,payload

    def test_accepted_and_rejected_full_pool_byte_parity(self):
        for accepted in (False,True):
            with self.subTest(accepted=accepted),tempfile.TemporaryDirectory() as d:
                root=Path(d);path,payload=self.fixture(root,accepted);before=path.read_bytes()
                first=pe.scan(root);second=pe.scan(root)
                self.assertTrue(first[0]['created']);self.assertFalse(second[0]['created'])
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(json.loads((root/'proposer_receipts'/path.name).read_bytes()),payload['result']['proposer_receipt'])
                self.assertEqual(first[0]['new_wm_calls'],0)

    def test_partial_pool_stops_before_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,payload=self.fixture(root)
            payload['result']['candidates_searched']=3;path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'incomplete'):pe.scan(root)
            self.assertFalse((root/'proposer_receipts').exists())

    def test_conflict_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,payload=self.fixture(root);pe.scan(root)
            dest=root/'proposer_receipts'/path.name;before=dest.read_bytes()
            payload['result']['proposer_receipt']['extra']='different';path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'conflicting'):pe.scan(root)
            self.assertEqual(dest.read_bytes(),before)

    def test_non_search_is_skipped_but_missing_search_pool_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,payload=self.fixture(root)
            del payload['result']['proposer_receipt'];path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'lacks full pool'):pe.scan(root)
            del payload['result']['candidates_searched'];path.write_text(json.dumps(payload))
            self.assertEqual(pe.scan(root),[])


if __name__=='__main__':unittest.main()
