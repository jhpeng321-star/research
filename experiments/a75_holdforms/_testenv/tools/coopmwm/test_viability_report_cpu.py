import tempfile
import unittest
from pathlib import Path
from . import viability_report as v
from .accel_storage import write
from .warm_protocol import SEED


class ViabilityTests(unittest.TestCase):
    def row(self,world,gain,rank=1):
        return dict(world=world,exact_saving=gain,type='MPPI',regime='ordinary',predicted_rank=rank,
                    known_label='new',group_size=4,update=1)

    def test_losing_plans_are_retained_in_exact_evidence(self):
        r=v.summarize([self.row(1,100),self.row(2,-300,2)],{})
        self.assertEqual(r['overall']['mean'],-100)
        self.assertEqual(r['overall']['positive'],1)
        self.assertEqual(r['by']['predicted_rank']['2']['mean'],-300)

    def test_one_world_cannot_manufacture_precision(self):
        r=v.bootstrap([self.row(1,100),self.row(1,-300)])
        self.assertIsNone(r['ci95'])
        self.assertEqual(r['worlds'],1)

    def test_world_clusters_keep_unequal_plan_counts_together(self):
        rows=[self.row(1,1)]*9+[self.row(2,11)]
        r=v.bootstrap(rows)
        self.assertEqual(r['mean'],2)
        self.assertEqual(r['ci95'],[1.,11.])

    def test_pending_unresolved_receipts_never_impute_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root/'exact_grades/receipts/x.json',dict(arm='B',training_seed=SEED,
                world_seed=986000,status='overlap_unresolved'))
            _,r=v.collect(root)
            self.assertEqual(r['overall']['plans'],0)
            self.assertEqual(r['excluded'],dict(overlap_unresolved=1))

    def test_non_train_grade_is_rejected_before_episode_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root/'exact_grades/receipts/x.json',dict(arm='B',training_seed=SEED,
                world_seed=989600,status='graded'))
            with self.assertRaisesRegex(ValueError,'outside registered TRAIN'):v.collect(root)


if __name__=='__main__':unittest.main()
