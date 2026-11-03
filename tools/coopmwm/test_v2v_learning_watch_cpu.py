"""Registered pause boundaries, using realistic update receipt fields."""
import copy
import unittest
from .v2v_learning_watch import inspect_update


def receipt():
    return dict(arm='B', update=2, bank_imitation=dict(pairs_total=226,
        epochs_completed=5, requested_epochs=5, pass1_effective=True,
        pass1_loss_drop=.0085, imitation_failed=False, kl=.0014, kl_limit=.045,
        v2v_intent=dict(before=dict(intent_loss=2.4), after=dict(intent_loss=2.535))),
        exact_imitation=dict(imitation_filter=dict(empty_bank=False, imitated_cooperative_examples=8)),
        ppo=[dict(own_policy_steps=128, kl=0., loss=8., gradient_norm=7.),
             dict(own_policy_steps=128, kl=.046, loss=4., gradient_norm=2.)])


class LearningWatchTests(unittest.TestCase):
    def test_valid_update_and_rising_intent_are_reported_without_pause(self):
        value=receipt();original=copy.deepcopy(value);report=inspect_update(value)
        self.assertTrue(report['learning_mechanics_ok'])
        self.assertTrue(report['bank_intent_loss_increased'])
        self.assertEqual(report['first_minibatch_ppo_kl'],0.)
        self.assertEqual(report['ppo_kl_max'],.046)  # standard PPO stopping bound, not bank KL
        self.assertEqual(value,original)

    def test_empty_lag_one_bank_is_allowed(self):
        value=receipt();value.update(arm='R',bank_imitation=dict(pairs_total=0,
            epochs_completed=0,pass1_effective=False,imitation_failed=False),
            exact_imitation=dict(imitation_filter=dict(empty_bank=True,imitated_cooperative_examples=0)))
        self.assertTrue(inspect_update(value)['learning_mechanics_ok'])

    def test_each_registered_mechanics_failure_pauses(self):
        for change,expected in [({'imitation_failed':True},'imitation_failed'),
            ({'pass1_effective':False},'ineffective_pass1'),
            ({'epochs_completed':0},'no_complete_imitation_pass'),
            ({'kl':float('nan')},'nonfinite_imitation_kl'),
            ({'kl':.04500001},'imitation_kl_out_of_limit')]:
            with self.subTest(change=change):
                value=receipt();value['bank_imitation'].update(change)
                self.assertIn(expected,inspect_update(value)['failures'])

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
