import unittest
from copy import deepcopy
from tools.wmteach.live import approval,budget,admit_event,PANEL_SHA,parent
from tools.coopbandit.protocol import digest


class LiveTests(unittest.TestCase):
    def test_shared_panel_and_exact_upper_budget(self):
        self.assertEqual(digest(parent().panel()),PANEL_SHA)
        self.assertEqual(len(parent().panel()),24)
        self.assertEqual(budget()['total'],24*2*1200+1200)
        self.assertLessEqual(budget()['total'],100000)

    def test_efficacy_only_never_prunes_or_substitutes_a_real_move(self):
        for t in range(300,961,60):
            event=dict(time_s=t,executed_action=dict(id='O3',members=['a','b'],window=60),group=['a','b'])
            original=deepcopy(event)
            self.assertEqual(admit_event(event,0),0)
            self.assertEqual(event['executed_action'],original['executed_action'])
            self.assertEqual(event['group'],['a','b'])
            self.assertFalse(event['budget_pruned'])

    def test_exact_wm_authority_required(self):
        with self.assertRaises(PermissionError):approval('S-DISTILL BUDGET APPROVED 200000')
        with self.assertRaises(PermissionError):approval('S-WMTEACH BUDGET APPROVED 100000 pending')
        self.assertIn('S-WMTEACH',approval('MAIN S-WMTEACH BUDGET APPROVED 100,000'))


if __name__=='__main__':unittest.main()
