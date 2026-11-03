import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from solo_seed2 import eligible,cost


class Seed2Tests(unittest.TestCase):
    def test_cost_includes_whole_prefix_and_one_shared_control(self):
        self.assertEqual(cost(300,300,6),7*720)
        self.assertEqual(cost(900,300,6),7*1200)

    def test_incomplete_or_nonpositive_solo_evidence_never_qualifies(self):
        c=dict(cooperation=dict(complete=True,cooperation_gain_veh_s=10),
            checks=dict(replication=True,no_later_harm=True))
        self.assertTrue(eligible(c))
        c['cooperation']['complete']=False;self.assertFalse(eligible(c))
        c['cooperation']['complete']=True;c['checks']['replication']=False
        self.assertFalse(eligible(c))
        self.assertFalse(eligible({}))


if __name__=='__main__':unittest.main()
