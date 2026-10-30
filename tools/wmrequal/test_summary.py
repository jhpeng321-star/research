import json
from pathlib import Path
import unittest
import numpy as np
from summarize import calls,stats


class SummaryTest(unittest.TestCase):
    def test_strict_gain_threshold_q10_and_zero_real_rule(self):
        rows=[dict(world='a'),dict(world='b'),dict(world='c')]
        d=np.array([[-2.]*8,[-4.]*8,[-4.]*7+[12.]])
        r=np.array([-3.,0.,-1.])
        result=calls(rows,d,r,1.)
        # Equal-to-margin is not confident. Zero real still contributes coverage.
        # Third row q10(-dJ) is negative, so median gain alone does not open gate.
        self.assertEqual(result['gain'],1)
        self.assertIsNone(result['gain_sign'])
        self.assertFalse(result['CCT2_style_pass'])

    def test_component_attribution_adds_and_is_not_sum_of_medians(self):
        source=np.array([0.,0.,0.,1.,10.,20.,30.,40.])
        road=np.array([20.,30.,40.,0.,0.,0.,0.,0.])
        total=source+road
        order=np.argsort(total,kind='stable')
        weights=np.zeros(8);weights[order[3:5]]=.5
        self.assertEqual(weights@source+weights@road,np.quantile(total,.5))
        self.assertNotEqual(np.quantile(source,.5)+np.quantile(road,.5),np.quantile(total,.5))

    def test_archived_validation_real_targets_match_exactly(self):
        p=Path(__file__).parent/'evidence'
        canonical=json.loads((p/'LEGACY_REAL.json').read_text())['validation']
        lookup={(r['pair_key'],r['arm']):r for r in canonical}
        doc=json.loads((p/'WM2mix_validation.json').read_text())
        for row in doc['rows']:
            for h in ('60','90','120','180','300'):
                for arm,pair in zip(row['arm_names'],row['dJ'][h]):
                    got=lookup[row['pair_key'],arm]['values'][h]-lookup[row['pair_key'],'native']['values'][h]
                    self.assertEqual(got,pair[1])


if __name__=='__main__':
    unittest.main()
