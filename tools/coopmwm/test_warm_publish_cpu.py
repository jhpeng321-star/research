import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import warm_publish as p
from .storage import write


class PublishTests(unittest.TestCase):
    def test_metadata_and_final_outcomes_are_never_training_reports(self):
        for n in ('EXACT_VIABILITY_REPORT_CONFIG.json','EXACT_VIABILITY_REPORT_DISPATCH.json',
                  'EXACT_VIABILITY_REPORT_FAILED.json','FINAL_TEST_COMPLETE.json','A_CHECK_RECORDS_048.json'):
            self.assertFalse(p.publishable(n),n)

    def test_actual_reports_and_health_receipts_remain_publishable(self):
        for n in ('A_UPDATE_048.json','AB_CHECK_004.json','A_CHECK_PRESERVED_004.json',
                  'B_EXACT_VIABILITY_'+'a'*64+'.json','WARM_A_PAUSED.json','PAUSE_HARMFUL_PLANS.json'):
            self.assertTrue(p.publishable(n),n)

    def test_cached_but_unpublished_receipt_is_adopted_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=root/'docs/paper_results/coopmwm';e.mkdir(parents=True)
            doc=e.parent/'COOPMWM.md';doc.write_text('# Results\n')
            local=e/'warm_live';local.mkdir()
            value=dict(arm='A',update=48,proposal_routing={},trust=dict(beta=0.),wall_s=1.)
            name='A_UPDATE_048.json';write(local/name,value)
            original=(local/name).read_bytes()
            result=json.dumps(dict(receipts={name:value,'EXACT_VIABILITY_REPORT_CONFIG.json':dict(private='metadata')},charged_sim_s=100))
            with patch.object(p.ops,'LOCAL',root),patch.object(p.ops,'EVIDENCE',e),patch.object(p.ops.ssh,'remote',return_value=result):
                p.publish();first=doc.read_text();p.publish()
            self.assertIn('warm-receipt:A_UPDATE_048.json:',first)
            self.assertEqual(first,doc.read_text())
            self.assertEqual(original,(local/name).read_bytes())
            self.assertFalse((local/'EXACT_VIABILITY_REPORT_CONFIG.json').exists())

    def test_changed_cached_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);e=root/'docs/paper_results/coopmwm';e.mkdir(parents=True)
            (e.parent/'COOPMWM.md').write_text('# Results\n');(e/'warm_live').mkdir()
            write(e/'warm_live/A_UPDATE_048.json',dict(value=1))
            result=json.dumps(dict(receipts={'A_UPDATE_048.json':dict(value=2)},charged_sim_s=100))
            with patch.object(p.ops,'LOCAL',root),patch.object(p.ops,'EVIDENCE',e),patch.object(p.ops.ssh,'remote',return_value=result):
                with self.assertRaisesRegex(AssertionError,'immutable cached'):p.publish()
            self.assertEqual(json.loads((e/'warm_live/A_UPDATE_048.json').read_text()),dict(value=1))


if __name__=='__main__':unittest.main()
