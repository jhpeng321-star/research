import json
import unittest
from .noop_report import compare,by_update


class NoopReportTests(unittest.TestCase):
    @staticmethod
    def tape(changed=False):
        return [json.dumps(dict(time_s=t,vehicles=[dict(vid='v',s_m=t+(int(changed) if t==1 else 0),v_mps=1.,lane=0)])) for t in range(3)]

    def test_exact_motion_identity(self):
        r=compare(self.tape(),self.tape(),0,3)
        self.assertEqual(r,dict(seconds=3,roster_differences=0,motion_differences=0))

    def test_zero_grade_does_not_imply_motion_null(self):
        self.assertEqual(compare(self.tape(),self.tape(True),0,3)['motion_differences'],1)

    def test_incomplete_tape_fails(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            compare(self.tape(),self.tape()[:2],0,3)

    def test_counts_include_losses_and_zeros(self):
        rows=[dict(update=2,null_classification=dict(exact_zero=x,observed_physical_null=y)) for x,y in ((True,True),(True,False),(False,False))]
        r=by_update(rows)['2']
        self.assertEqual((r['exact_graded'],r['exact_zero'],r['confirmed_physical_null']),(3,2,1))
        self.assertEqual(r['confirmed_null_fraction'],1/3)
