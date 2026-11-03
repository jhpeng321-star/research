"""Preparation/launch isolation: no torch, model, simulator or network required."""
import types
import unittest
from unittest.mock import Mock
from . import v2v_runner as v


class RunnerTests(unittest.TestCase):
    def binding(self):
        return v.prepare(dict(v2v=False, sumo_verify={'k':4}, resume_progress={'old':True},budget=dict(total=17421760)),
            root='/new', legacy_root='/old', source_root='/qualified',
            common_checkpoint_sha256='a'*64, v2v_dim=24)

    def test_prepare_is_separate_and_unlaunched(self):
        b = self.binding()
        self.assertEqual(b['arm_names'], dict(A='A-v2v',B='B-v2v',R='R-v2v'))
        self.assertEqual(b['final_updates'],dict(A=24,B=24,R=24))
        self.assertEqual(b['A_resume_update'],0)
        self.assertEqual(set(b['start_checkpoint_sha256'].values()),{'a'*64})
        self.assertEqual(b['A_stop_after_update'],24)
        self.assertEqual(b['accounting_root'],'/old')
        self.assertFalse(b['launch_authorized'])
        self.assertEqual(b['physics_reservation_sim_s'],0)
        self.assertNotIn('sumo_verify',b)
        self.assertNotIn('resume_progress',b)
        self.assertEqual(b['bank_arms'],dict(B='B',R='R'))

    def test_admission_blocks_before_callbacks(self):
        hooks={n:Mock() for n in v.REQUIRED_HOOKS}
        with self.assertRaisesRegex(ValueError,'not authorised'):
            v.run_arm('/new','B',self.binding(),{},hooks)
        for hook in hooks.values():hook.assert_not_called()

    def test_readiness_and_scope_fail_closed(self):
        b=self.binding();b.update(launch_authorized=True,phase_funding_authorized=True,
            launch_amendment_sha256='l'*64,funding_amendment_sha256='f'*64)
        hooks={n:Mock() for n in v.REQUIRED_HOOKS}
        with self.assertRaisesRegex(ValueError,'non-stub'):
            v.admit(b,dict(qualified=True),hooks)
        release=dict(qualified=True,linux_cpu_pass=True,zero_init_parity=True,
            default_off_byte_identical=True,stub=False,v2v_dim=24,rehearsal_pass=True)
        v.admit(b,release,hooks)
        b['accounting_root']='/new'
        with self.assertRaisesRegex(ValueError,'accounting'):
            v.admit(b,release,hooks)

    def test_patch_restore_after_failure(self):
        m=types.SimpleNamespace(wave='original')
        with self.assertRaises(RuntimeError):
            with v.patched(m,dict(wave='temporary')):
                self.assertEqual(m.wave,'temporary');raise RuntimeError('test')
        self.assertEqual(m.wave,'original')

    def test_budget_fallback_and_common_start(self):
        b=self.binding()
        self.assertEqual(b['active_arms'],['A','B','R'])
        self.assertEqual(b['diagnostic_schedule']['B']['lessons'],[6,12,18,24])
        self.assertEqual(b['diagnostic_schedule']['R']['isolated'],[12,24])
        self.assertEqual(b['diagnostic_schedule']['A'],dict(lessons=[],isolated=[]))
        self.assertFalse(b['final_paired_branches'])
        self.assertEqual(b['phase_map']['train'],'v2v_train')
        with self.assertRaisesRegex(ValueError,'all three'):
            v.prepare({},root='/new',legacy_root='/old',source_root='/qualified',
                common_checkpoint_sha256='b'*64,v2v_dim=24,active_arms=('A','B'))

    def test_budget_and_common_checkpoint_guards(self):
        b=self.binding();b.update(launch_authorized=True,phase_funding_authorized=True,
            launch_amendment_sha256='l'*64,funding_amendment_sha256='f'*64,
            A_diagnostic_slate_authorized=True,reporting_scope_authorized=True)
        release=dict(qualified=True,linux_cpu_pass=True,zero_init_parity=True,
            default_off_byte_identical=True,stub=False,v2v_dim=24,rehearsal_pass=True)
        hooks={n:Mock() for n in v.REQUIRED_HOOKS}
        b['budget']['total']=17500001
        with self.assertRaisesRegex(ValueError,'ceiling'):
            v.admit(b,release,hooks)
        b['budget']['total']=17421760;b['start_checkpoint_sha256']['B']='b'*64
        with self.assertRaisesRegex(ValueError,'same MAPPO u0'):
            v.admit(b,release,hooks)
        for hook in hooks.values():hook.assert_not_called()

    def test_rehearsal_fallback_requires_all_real_smokes(self):
        b=self.binding();b.update(launch_authorized=True,phase_funding_authorized=True,
            launch_amendment_sha256='l'*64,funding_amendment_sha256='f'*64,
            A_diagnostic_slate_authorized=True,reporting_scope_authorized=True)
        release=dict(qualified=True,linux_cpu_pass=True,zero_init_parity=True,
            default_off_byte_identical=True,stub=False,v2v_dim=24,
            real_update_smokes_pass=True,smoke_arms=['A','B'])
        hooks={n:Mock() for n in v.REQUIRED_HOOKS}
        with self.assertRaisesRegex(ValueError,'every active arm'):
            v.admit(b,release,hooks)
        release['smoke_arms']=['A','B','R'];v.admit(b,release,hooks)


if __name__=='__main__':unittest.main()
