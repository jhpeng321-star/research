"""Registered pause boundaries, using realistic update receipt fields."""
import copy
import unittest
from .v2v_learning_watch import inspect_update, complete_update


def receipt():
    return dict(arm='B', update=2, bank_imitation=dict(pairs_total=226,
        epochs_completed=5, requested_epochs=5, pass1_effective=True,
        pass1_loss_drop=.0085, imitation_failed=False, kl=.0014, kl_limit=.045,
        v2v_intent=dict(before=dict(intent_loss=2.4), after=dict(intent_loss=2.535))),
        exact_imitation=dict(imitation_filter=dict(empty_bank=False, imitated_cooperative_examples=8)),
        ppo=[dict(own_policy_steps=128, kl=0., loss=8., gradient_norm=7.),
             dict(own_policy_steps=128, kl=.046, loss=4., gradient_norm=2.)])


class LearningWatchTests(unittest.TestCase):
    def test_incomplete_publication_is_observed_again_without_pause_or_exit(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'B_UPDATE_003.json'
            self.assertEqual(complete_update(p),(None,'FileNotFoundError'))
            p.write_text('{"arm":')
            self.assertEqual(complete_update(p),(None,'JSONDecodeError'))
            p.write_text(json.dumps(receipt()))
            value,pending=complete_update(p)
            self.assertIsNone(pending)
            self.assertTrue(inspect_update(value)['learning_mechanics_ok'])

    def test_valid_update_and_rising_intent_are_reported_without_pause(self):
        value=receipt();original=copy.deepcopy(value);report=inspect_update(value,.045)
        self.assertTrue(report['learning_mechanics_ok'])
        self.assertTrue(report['bank_intent_loss_increased'])
        self.assertEqual(report['runtime_kl_limit'],.045)
        self.assertIsNone(report['total_kl_above_ppo_stop_plus_limit'])
        self.assertEqual(report['first_minibatch_ppo_kl'],0.)
        self.assertEqual(report['ppo_kl_max'],.046)  # standard PPO stopping bound, not bank KL
        self.assertEqual(value,original)

    def test_receipt_kl_limit_must_equal_the_pinned_runtime_limit(self):
        # fix round c4/imit-kl-option: the receipt's own kl_limit is evidence, never authority.
        value=receipt();value['bank_imitation'].update(kl=.0014,kl_limit=.06)
        self.assertIn('imitation_kl_limit_mismatch',inspect_update(value,.045)['failures'])
        # the same receipt against ITS runtime pin is fine, and its bound is the runtime's:
        self.assertTrue(inspect_update(value,.06)['learning_mechanics_ok'])
        wide=receipt();wide['bank_imitation'].update(kl=.055,kl_limit=1.)
        self.assertIn('imitation_kl_out_of_limit',inspect_update(wide,.045)['failures'])
        self.assertIn('imitation_kl_limit_mismatch',inspect_update(wide,.06)['failures'])
        self.assertTrue(inspect_update(dict(wide,bank_imitation=dict(
            wide['bank_imitation'],kl=.055,kl_limit=.06)),.06)['learning_mechanics_ok'])

    def test_runtime_kl_limit_reader_pins_the_runtime_value(self):
        from .v2v_learning_watch import runtime_kl_limit
        self.assertEqual(runtime_kl_limit(dict(config=dict(imitation_kl_limit=.09))),.09)
        self.assertEqual(runtime_kl_limit(dict(config=dict())),.045)  # pre-option runtime
        for bad in (dict(config=dict(imitation_kl_limit=.05)),dict(config=None),{},'runtime'):
            with self.assertRaises(ValueError):
                runtime_kl_limit(bad)

    def test_total_kl_above_ppo_stop_plus_limit_is_reported_only(self):
        value=receipt();value['bank_imitation'].update(kl_limit=.06,total_kl_vs_pre_update=.2)
        report=inspect_update(value,.06)   # .2 > PPO stop .045 + limit .06
        self.assertTrue(report['learning_mechanics_ok'])
        self.assertTrue(report['total_kl_above_ppo_stop_plus_limit'])
        self.assertEqual(report['reported_only'],['total_kl_above_ppo_stop_plus_limit'])
        below=receipt();below['bank_imitation'].update(kl_limit=.06,total_kl_vs_pre_update=.1)
        report=inspect_update(below,.06)   # .1 <= .105
        self.assertFalse(report['total_kl_above_ppo_stop_plus_limit'])
        self.assertEqual(report['reported_only'],[])
        default=receipt()                  # a .045 runtime never derives the flag
        self.assertIsNone(inspect_update(default,.045)['total_kl_above_ppo_stop_plus_limit'])

    def test_empty_lag_one_bank_is_allowed(self):
        value=receipt();value.update(arm='R',bank_imitation=dict(pairs_total=0,
            epochs_completed=0,pass1_effective=False,imitation_failed=False),
            exact_imitation=dict(imitation_filter=dict(empty_bank=True,imitated_cooperative_examples=0)))
        self.assertTrue(inspect_update(value)['learning_mechanics_ok'])

    def test_each_registered_mechanics_failure_pauses(self):
        for change,expected in [({'imitation_failed':True},'imitation_failed'),
            ({'pass1_loss_change':dict(n=226,mean=.01,ci95=[.002,.02],significantly_worse=True)},
             'pass1_loss_significantly_worse'),
            ({'epochs_completed':0},'no_complete_imitation_pass'),
            ({'kl':float('nan')},'nonfinite_imitation_kl'),
            ({'kl':.04500001},'imitation_kl_out_of_limit')]:
            with self.subTest(change=change):
                value=receipt();value['bank_imitation'].update(change)
                self.assertIn(expected,inspect_update(value)['failures'])

    def test_non_significant_pass1_change_is_reported_only_for_B_and_R(self):
        # MAIN 09:00: R u3 (3 cooperative examples, loss +0.0017 of 19.35) must not pause
        for arm in ('B','R'):
            for change,label in [(dict(n=102,mean=.00002,ci95=[-.0004,.0005],significantly_worse=False),
                                  'pass1_not_effective_not_significant'),
                                 ({},'pass1_not_effective_no_ci_reported_only')]:
                with self.subTest(arm=arm,change=change):
                    value=receipt();value['arm']=arm
                    value['bank_imitation'].update(pass1_effective=False,pass1_loss_drop=-.0017)
                    if change:value['bank_imitation']['pass1_loss_change']=change
                    report=inspect_update(value)
                    self.assertTrue(report['learning_mechanics_ok'])
                    self.assertEqual(report['reported_only'],[label])
        value=receipt();value['bank_imitation']['pass1_loss_change']=dict(
            n=226,mean=.01,ci95=[.002,.02],significantly_worse=True)
        self.assertIn('pass1_loss_significantly_worse',inspect_update(value)['failures'])

    def test_nonfinite_ppo_or_intent_is_a_failure(self):
        value=receipt();value['ppo'][0]['loss']=float('inf')
        self.assertIn('nonfinite_ppo_loss_record_0',inspect_update(value)['failures'])
        value=receipt();value['bank_imitation']['v2v_intent']['after']['intent_loss']=float('nan')
        self.assertIn('nonfinite_bank_intent_after',inspect_update(value)['failures'])

    def test_A_has_no_bank_and_no_cooperation_gate_is_applied(self):
        value=dict(arm='A',update=2,ppo=[dict(own_policy_steps=128,kl=0.,loss=1.)],
            bank_imitation=None,cooperation_gain_lb95=-1.)
        self.assertTrue(inspect_update(value)['learning_mechanics_ok'])


if __name__=='__main__':unittest.main()
