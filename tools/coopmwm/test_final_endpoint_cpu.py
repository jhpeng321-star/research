import tempfile
import unittest
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from . import final_runtime as f
from .final_endpoint import ready
from .storage import write


class EndpointTests(unittest.TestCase):
    def test_resource_handover_hashes_opaque_artifacts_without_decoding_outcomes(self):
        from .final_endpoint import adopt_completed_stage
        from .protocol import sha
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'final_receipts').mkdir();p=r/'jobs/fixedfinal_A24v2_989600';p.mkdir(parents=True)
            (r/'A_checkpoint_024.pt').write_bytes(b'fixed24')
            # Deliberately invalid JSON: admission must hash, never decode it.
            (p/'RESULT.json').write_bytes(b'opaque outcome, not JSON')
            write(r/'final_receipts'/f'{p.name}.json',dict(id=p.name,outcomes_closed=True,
                checkpoint_update=24,checkpoint_sha256=sha(r/'A_checkpoint_024.pt'),
                files={'RESULT.json':dict(sha256=sha(p/'RESULT.json'))}))
            specs=[dict(id=p.name,checkpoint_update=24)]
            self.assertEqual(adopt_completed_stage(r,specs),1)
            (p/'RESULT.json').write_bytes(b'changed')
            with self.assertRaises(AssertionError):adopt_completed_stage(r,specs)

    def test_final_phase_caps_use_approved_config_not_older_training_binding(self):
        from .warm_protocol import install_accounting,budget
        from . import accel_storage
        from .accel_protocol import SUPERSEDED_CHARGE
        b=budget();b['phases']=dict(b['phases'],final_test=691200,paired_branch=1297400)
        b['allocated']=sum(b['phases'].values());b['total']=9500000
        with patch.object(accel_storage,'_reserve'):
            install_accounting(budget_override=b)
            globals_=accel_storage._reserve.__globals__
            self.assertEqual(globals_['CAPS']['final_test'],691200)
            self.assertEqual(globals_['CAPS']['paired_branch'],1297400)
            self.assertEqual(globals_['CAP'],9500000-SUPERSEDED_CHARGE)

    def test_spawn_worker_uses_serialized_checkpoint_even_when_import_budget_is_48(self):
        # Serialize the parent's fixed-u24 spec, then simulate the fresh worker
        # import's original u48 budget. No checkpoint/outcome is selected by data.
        from .warm_protocol import budget
        b=dict(runtime='runtime',checkpoint='mappo',accel_bounds=[-3.,2.6],gpu_deadline=1.)
        primary=dict(budget(),updates=dict(A=24,B=24,R=24))
        with patch.object(f,'budget',return_value=primary):
            spec=json.loads(json.dumps(f.specs(('A',),panel=[dict(seed=989600)])[0]))
        self.assertEqual(spec['checkpoint_update'],24)
        policy=SimpleNamespace(state_dict=lambda:{})
        with patch.object(f,'budget',return_value=dict(primary,updates=dict(A=48,B=48,R=24))),patch(
                'tools.coopmwm.accel_learner.load',return_value=(policy,None,None,dict(update=24))) as load:
            f.make_job(Path('run'),b,spec)
        load.assert_called_once_with(Path('run/A_checkpoint_024.pt'))
        spec.pop('checkpoint_update')
        with self.assertRaises(KeyError):f.make_job(Path('run'),b,spec)

    def test_A24_jobs_have_fresh_ids_without_changing_arm(self):
        with patch.object(f,'ID_ALIASES',{'A':'A24'}):
            jobs=f.specs(('A',),panel=[dict(seed=989600)])
            branches=f.specs(('A',),branches=True,panel=[dict(seed=989600)])
            self.assertEqual(jobs[0]['id'],'fixedfinal_A24_989600')
            self.assertEqual(jobs[0]['arm'],'A')
            self.assertEqual(branches[0]['id'],'fixedbranch_A24_989600_t300')
            self.assertEqual(f.job_id('B',989600,300),'fixedbranch_B_989600_t300')

    def test_all_primary_checkpoints_and_certificates_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for a,u in [('A',48),('B',24)]:
                write(r/f'WARM_{a}_PAUSED.json',dict(update=u));(r/f'{a}_checkpoint_024.pt').touch()
            self.assertFalse(ready(r))
            write(r/'WARM_R_PAUSED.json',dict(update=3));(r/'R_checkpoint_024.pt').touch()
            with self.assertRaises(ValueError):ready(r)
            (r/'WARM_R_PAUSED.json').write_text('{"update":24}')
            self.assertTrue(ready(r))

    def test_replacement_liability_and_no_duplicate_old_jobs(self):
        with patch.object(f,'ID_ALIASES',{'A':'A24'}):
            new=f.specs(('A',))+f.specs(('A',),branches=True)
            self.assertEqual(sum(s['sim_s'] for s in new),460800)
            self.assertTrue(all('_A24_' in s['id'] for s in new))
            self.assertFalse(any('mappo' in s['id'] or 'native' in s['id'] for s in new))


if __name__=='__main__':unittest.main()
