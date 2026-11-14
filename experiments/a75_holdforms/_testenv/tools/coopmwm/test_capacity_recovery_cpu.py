import copy
import unittest
from .capacity_recovery import replacement_plan,repair_budget
from .warm_protocol import batch_worlds,budget,worlds


class RepairTests(unittest.TestCase):
    def test_real_operational_binding_merge_and_u0_preflight(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from . import capacity_recovery as recovery
        from . import warm_runner, accel_storage
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,value in [('EXACT_OPERATIONAL_BINDING.json',dict(workers=4)),
                ('WARM_B_RUNTIME_BINDING_V4.json',dict(operational=dict(workers=8,health_guards=True))),
                ('AWR_BINDING.json',dict(config=dict(coefficient=1)))]:
                (root/name).write_text(json.dumps(value))
            b=dict(budget=budget(),repair_plan=dict(worlds=[],precharged_unattempted=['one']))
            with patch.object(recovery,'binding',return_value=b),patch('tools.coopmwm.warm_health.assert_running'), \
                    patch.multiple(warm_runner,verify=warm_runner.verify,budget=warm_runner.budget,
                                   batch_worlds=warm_runner.batch_worlds), \
                    patch.object(accel_storage,'_reserve',accel_storage._reserve):
                wr,_=recovery.install(root)
                self.assertEqual(wr.verify(root,'B')['workers'],24)
                self.assertEqual(wr.verify(root,'R')['workers'],4)
                self.assertEqual(wr.verify(root,'B')['resume_update'],0)

    def test_exact_six_precharges_and_eighteen_fresh_same_cell_worlds(self):
        old=batch_worlds(1);claimed={w['seed'] for w in old[:-6]}
        plan=replacement_plan(old,worlds('train'),claimed)
        self.assertEqual(plan['additional_sim_s'],21600)
        self.assertEqual(len(plan['precharged_unattempted']),6)
        self.assertEqual(len(plan['worlds']),24)
        self.assertEqual(len({w['seed'] for w in plan['worlds']}),24)
        self.assertFalse({w['seed'] for w in plan['worlds']}&claimed)
        self.assertEqual([sum(w['cell']==c for w in old) for c in range(12)],
            [sum(w['cell']==c for w in plan['worlds']) for c in range(12)])
        self.assertEqual(sum(w['demand']>=2400 and w['penetration']>=.2 for w in plan['worlds']),12)

    def test_budget_keeps_every_failed_charge_and_approved_cap(self):
        original=budget();old=copy.deepcopy(original);new=repair_budget(original,21600)
        self.assertEqual(original,old)
        self.assertEqual(new['allocated'],7495040);self.assertEqual(new['unallocated'],4960)
        self.assertEqual(new['total'],7500000)
        for phase,amount in old['phases'].items():
            if phase!='train':self.assertEqual(new['phases'][phase],amount)

    def test_insufficient_room_is_a_stop_not_partial_wave(self):
        with self.assertRaises(ValueError):repair_budget(budget(),30000)

    def test_replacement_never_selects_failed_world_even_if_lower_seed(self):
        original=batch_worlds(1)
        plan=replacement_plan(original,worlds('train'),{original[0]['seed']})
        self.assertNotIn(original[0],plan['worlds'])
        self.assertEqual(plan['additional_sim_s'],1200)


if __name__=='__main__':unittest.main()
