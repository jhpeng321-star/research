"""Meaningful synthetic counterfactual and control-context contracts; zero SUMO."""
import contextlib
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.coopaudit.design import interaction, select_group, worlds, SEEDS
from tools.coopaudit.controllers import WallControl, wall_roster
from tools.coopaudit.analyze import bootstrap
from tools.coopaudit.accounting import Ledger


def row(vid,s,lane=0,**kw):
    return dict(vid=vid,s=s,lane=lane,v=12.,junction=False,is_cav=True,**kw)


class AuditContracts(unittest.TestCase):
    def test_additive_has_zero_interaction(self):
        self.assertEqual(interaction(100,70,[90,80])['C_G'],0)

    def test_synergy_is_signed_saved_cost(self):
        self.assertEqual(interaction(100,60,[90,80])['C_G'],10)
        self.assertEqual(interaction(100,80,[90,80])['C_G'],-10)

    def test_singleton_refused(self):
        with self.assertRaises(ValueError):
            interaction(100,90,[90])

    def test_group_uses_geometry_and_not_hidden_inputs(self):
        rows=[row('b',-2),row('a',2),row('c',-140),row('outside',-151)]
        self.assertEqual(select_group(rows,2),['a','b'])
        for r in rows:
            r.update(hidden_type='irrelevant',gain=1e99,v=-100)
        self.assertEqual(select_group(list(reversed(rows)),2),['a','b'])
        self.assertEqual(select_group(rows),['a','b','c'])

    def test_all_group_not_capped_at_four(self):
        self.assertEqual(len(select_group([row(str(i),-i) for i in range(15)])),15)

    def test_missing_is_not_singleton(self):
        self.assertEqual(select_group([row('a',0)]),[])

    def test_wall_solos_retain_full_coordinated_context(self):
        rows=[row('front',-10,0),row('rear',-20,1)]
        group=['front','rear']
        self.assertEqual(wall_roster(rows,group),['front','rear'])
        together=WallControl(group)
        solo=WallControl(group)
        all_commands=together.commands(rows,300,frozenset(group))
        only_commands=solo.commands(rows,300,frozenset(['rear']))
        self.assertEqual(all_commands,{'front':0.,'rear':1.})
        self.assertEqual(only_commands,{'rear':1.})
        self.assertEqual(together.session.members,solo.session.members)

    def test_wall_abort_retained_and_no_commands_after_release(self):
        wall=WallControl(['a','b'])
        rows=[row('a',-10),row('b',-20)]  # duplicate lane -> original safety abort
        self.assertEqual(wall.commands(rows,300,{'a','b'}),{})
        self.assertEqual(wall.session.abort_reason,'lane_coverage_violation')
        self.assertEqual(wall.commands(rows,301,set()),{})

    def test_bootstrap_world_cluster_and_missing_stratum(self):
        estimates=bootstrap([[1,2],[1,2],[3,6],[3,6]],[0,0,1,1],100)
        self.assertEqual(estimates[0]['mean'],2.)
        self.assertEqual(estimates[1]['lo'],2*estimates[0]['lo'])
        self.assertIsNone(bootstrap([[None],[1]],[0,1],100)[0]['mean'])

    def test_seed_mapping_disjoint_and_high_share_count(self):
        self.assertEqual(len(worlds()),28)
        self.assertEqual(sum(w['cell']<4 for w in worlds()),24)
        self.assertFalse(SEEDS & (set(range(969200,969625))|set(range(981200,981300))))

    @unittest.skipIf(os.name=='nt','fcntl ledger is exercised on Linux host')
    def test_precharge_no_refund_duplicate_or_partial_panel(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(d,cap=100)
            ledger.charge('one',60)
            with self.assertRaises(ValueError):
                ledger.charge('one',1)
            with self.assertRaises(ValueError):
                ledger.admit_panel([20,21])
            with self.assertRaises(ValueError):
                ledger.charge('two',41)
            self.assertEqual(sum(r['sim_s'] for r in ledger.rows()),60)


if __name__=='__main__':
    unittest.main(verbosity=2)
