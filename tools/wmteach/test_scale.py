"""CPU contracts for source exclusion, distinct states and the helpful-state gate."""
import copy
import unittest

from tools.wmteach.labels import helpful_quality
from tools.wmteach.scale import WORLDS,TIMES,TARGET,validate_record,roster_key


class ScaleContracts(unittest.TestCase):
    def record(self):
        return dict(role='TRAIN',world_id=969400,lineage=0,time_s=307.5,setting=0,
            state_id='w969400_line0_t615',rows=[dict(vid='1',s_m=10.,v_mps=20.,lane=0,
                length_m=5.,is_cav=True)],backlog=[0,0],request_total=[30,20])

    def test_distinct_schedule_inside_documented_phase(self):
        ids={(w,l,t) for w in WORLDS for l in (0,1) for t in TIMES}
        self.assertEqual(len(ids),TARGET)
        self.assertEqual(TARGET,5120)
        self.assertLessEqual(max(TIMES)+300,1200)
        self.assertEqual(set(WORLDS),set(range(969400,969432)))
        self.assertEqual(len(TIMES),80)

    def test_quarantine_and_diagnostic_exclusion(self):
        for world in (969432,969550,981002,984000,964000,962000,977000):
            r=self.record();r['world_id']=world
            with self.assertRaises(ValueError):validate_record(r)
        r=self.record();r['role']='QUALITY_ONLY'
        with self.assertRaises(ValueError):validate_record(r)

    def test_observable_identity_not_setting_or_hidden_latents(self):
        r=self.record();validate_record(r)
        second=copy.deepcopy(r);second['setting']=3;second['hidden_draw']='must_not_count'
        self.assertEqual(roster_key(r),roster_key(second))
        second['rows'][0]['s_m']+=1
        self.assertNotEqual(roster_key(r),roster_key(second))

    def test_gate_rejects_all_o0_and_tiny_lucky_sample(self):
        rows=[dict(oracle_saving=10.,agreement=False,exact_saving=0.) for _ in range(12)]
        self.assertFalse(helpful_quality(rows)['pass_'])
        self.assertEqual(helpful_quality(rows)['helpful_choice_agreement'],0.)
        self.assertFalse(helpful_quality([dict(oracle_saving=10.,agreement=True,exact_saving=10.)])['pass_'])
        self.assertFalse(helpful_quality([])['pass_'])

    def test_gate_requires_positive_exact_usefulness(self):
        rows=[dict(oracle_saving=10.,agreement=True,exact_saving=10.) for _ in range(3)]
        self.assertTrue(helpful_quality(rows)['pass_'])
        rows.extend(dict(oracle_saving=10.,agreement=False,exact_saving=-100.) for _ in range(3))
        self.assertFalse(helpful_quality(rows)['pass_'])


if __name__=='__main__':unittest.main()
