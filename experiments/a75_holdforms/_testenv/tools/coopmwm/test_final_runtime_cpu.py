"""Outcome-blind fixed-endpoint orchestration and accounting contracts."""
import tempfile
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from . import final_runtime as f
from .accel_storage import write
from .branch_eval import build_plan,verify_plan
from .warm_protocol import budget,worlds


class FinalRuntimeTests(unittest.TestCase):
    def test_exact_registered_panel_and_full_prefix_liability(self):
        episodes=f.specs(('mappo','native','A','B','R'))
        branches=f.specs(('A','B'),branches=True)
        self.assertEqual(len(episodes),480)
        self.assertEqual(len(branches),768)
        self.assertEqual(len({s['id'] for s in episodes+branches}),1248)
        phases={}
        for s in episodes+branches:phases[s['phase']]=phases.get(s['phase'],0)+s['sim_s']
        for k in phases:self.assertEqual(phases[k],budget()['phases'][k])
        self.assertEqual(phases['paired_branch'],691200)
        self.assertEqual({s['time_s'] for s in branches},set(f.TIMES))
        self.assertEqual({s['world']['seed'] for s in episodes},{w['seed'] for w in worlds('final_test')})

    def test_precompute_is_only_already_frozen_arms(self):
        items=f.specs(('mappo','native','A'))+f.specs(('A',),branches=True)
        self.assertEqual(len(items)+1,673)
        self.assertEqual(sum(s['sim_s'] for s in items)+600,691800)
        self.assertFalse(any(s['arm'] in ('B','R') for s in items))
        self.assertEqual(budget()['phases']['branch_reuse_equivalence'],600)

    def test_no_outcome_opening_before_all_fixed_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arm,u in (('A',48),('B',48)):
                write(root/f'WARM_{arm}_PAUSED.json',dict(update=u))
                (root/f'{arm}_checkpoint_{u:03}.pt').touch()
            self.assertTrue(f.endpoints_ready(root,('A',)))
            self.assertFalse(f.endpoints_ready(root))
            with patch.object(f,'read',side_effect=AssertionError('outcome read forbidden')):
                with patch.object(f,'endpoints_ready',return_value=False):
                    with self.assertRaisesRegex(AssertionError,'all fixed endpoints'):
                        f.analyze_complete(root,{},None)
            self.assertFalse((root/'FINAL_OUTCOMES_OPENED.json').exists())

    def test_wrong_endpoint_never_substitutes_an_earlier_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root/'WARM_A_PAUSED.json',dict(update=24))
            (root/'A_checkpoint_048.pt').touch()
            with self.assertRaisesRegex(ValueError,'endpoint receipt mismatch'):
                f.endpoints_ready(root,('A',))

    def test_public_receipts_are_explicit_hash_only_not_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'RESULT.json').write_text('{"secret_metric":123}')
            (root/'CLOSE.json').write_text('{"secret_metric":456}')
            (root/'scenarios').mkdir()
            spec=f.specs(('A',),panel=[worlds('final_test')[0]])[0]
            with patch.object(f,'read',side_effect=AssertionError('no result deserialisation')):
                meta=f.public_completion(spec,root)
            self.assertEqual(set(meta['files']),{'RESULT.json','CLOSE.json'})
            self.assertTrue(meta['outcomes_closed'])
            self.assertNotIn('secret_metric',str(meta))
            self.assertNotIn('metrics',meta)

    def test_duplicate_reservation_stops_before_worker_submission(self):
        spec=f.specs(('A',),panel=[worlds('final_test')[0]])[0]
        with patch.object(f,'ledger',return_value=[dict(id=spec['id'])]),patch.object(
                f,'ProcessPoolExecutor',side_effect=AssertionError('no worker submission')):
            with self.assertRaisesRegex(AssertionError,'already reserved'):
                f.run_stage(Path('unused'),{},[spec],1)

    def test_registered_branch_plan_has_two_holm_families(self):
        plan=build_plan(times=list(f.TIMES),later_arms=[])
        self.assertTrue(verify_plan(plan))
        self.assertEqual(plan['budget']['total_sim_s'],691200)
        families=plan['analysis']['co_primary']['families']
        self.assertEqual(set(families),{'MAPPO','A'})
        self.assertTrue(all(len(v)==2 for v in families.values()))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable in this interpreter')
    def test_reuse_checks_continuation_not_only_prefix(self):
        import copy
        spec=dict(id='fixture',arm='mappo',world=dict(seed=1),kind='reuse',time_s=300,
                  sim_s=600,phase='branch_reuse_equivalence')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=dict(prefixes={str(t):dict(t_s=t,metrics=dict(team_delay_vs=float(t).hex()))
                                     for t in range(300,601)})
            write(root/'jobs/fixedfinal_mappo_1/RESULT.json',reference)
            result=copy.deepcopy(reference);result['final']=reference['prefixes']['600']
            result['prefixes']['400']=dict(t_s=400,metrics=dict(team_delay_vs='0x0.0p+0'))
            with patch.object(f,'make_job',return_value=dict(path=str(root/'jobs/fixture'))),patch(
                    'tools.coopmwm.accel_plant.episode',return_value=result),patch.object(
                    f.os,'sched_setscheduler',create=True),patch.object(f.os,'SCHED_IDLE',5,create=True),patch.object(
                    f.os,'sched_param',create=True),patch.object(f.os,'getpriority',return_value=19,create=True),patch.object(
                    f.os,'PRIO_PROCESS',0,create=True),patch.object(f.os,'nice',create=True):
                with self.assertRaisesRegex(ValueError,'reuse equivalence failed'):
                    f.execute_one(root,{},spec)
            self.assertFalse((root/'final_receipts/fixture.json').exists())


if __name__=='__main__':unittest.main()
