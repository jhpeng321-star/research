import unittest
from .warm_protocol import budget,batch_worlds
from .accel_protocol import worlds
from .branch_eval import build_plan


class LatestScope(unittest.TestCase):
    def test_exact_owner_budget_and_protected_liabilities(self):
        b=budget();self.assertEqual((b['total'],b['allocated'],b['unallocated']),(6500000,6473440,26560))
        self.assertEqual(sum(b['phases'].values()),b['allocated'])
        self.assertEqual(b['phases']['paired_branch'],691200)
        self.assertEqual(b['phases']['discovery'],200000)
        self.assertEqual(b['phases']['cold_B_retained'],28800)
        self.assertEqual(b['phases']['failed_A_retained'],14400)

    def test_all_batches_use_existing_worlds_and_exact_shared_focus_fraction(self):
        observed=set()
        for u in range(1,49):
            ws=batch_worlds(u)
            self.assertEqual(len({w['seed'] for w in ws}),24)
            self.assertTrue(all(w in worlds('train') for w in ws))
            self.assertEqual(sum(w['demand']>=2400 and w['penetration']>=.2 for w in ws),12)
            self.assertEqual(ws,batch_worlds(u));observed.update(w['seed'] for w in ws)
        self.assertEqual(observed,{w['seed'] for w in worlds('train')})

    def test_two_comparator_specific_holm_families(self):
        cp=build_plan(times=[300,500,700,900],later_arms=())['analysis']['co_primary']
        self.assertEqual(cp['families'],{'MAPPO':['branch:B-MAPPO','focus:B-MAPPO'],'A':['branch:B-A','focus:B-A']})


if __name__=='__main__':unittest.main()
