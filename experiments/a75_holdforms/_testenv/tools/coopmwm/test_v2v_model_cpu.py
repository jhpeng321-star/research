"""CPU gates: v2v zero-init warm start equivalence, gradients, optimizer state."""
import tempfile
import unittest
from pathlib import Path

import torch

from tools.coopnative.test_cpu import rows
from .accel_learner import load, save
from .acceleration import Policy, encode
from .v2v_model import V2V_DIM, extend_for_v2v, forward_with_v2v, load_checkpoint, v2v_extra

BASE_ACTOR_INPUT = 29+4
BASE_CRITIC_INPUT = 29+12


class V2vModelCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)

    def random_context(self, generator, extra_dim):
        """Whitelisted sensors plus random v2v intent channels."""
        context = encode(rows())
        x = torch.tensor(context['x'], dtype=torch.float32)
        rsu = torch.tensor(context['rsu'], dtype=torch.float32)
        extra = torch.randn((len(x), extra_dim), generator=generator)
        return context, x, rsu, torch.cat((x, extra), -1)

    def test_zero_init_warm_start_is_equivalent_on_random_observations(self):
        """Zero-init extension reproduces actor and critic outputs exactly."""
        for extra_dim in (1, 3, V2V_DIM):
            generator = torch.Generator().manual_seed(11)
            base, extended = Policy(), Policy()
            extended.load_state_dict(base.state_dict())
            extend_for_v2v(extended, extra_dim)
            context, x, rsu, wide = self.random_context(generator, extra_dim)
            base_mean, base_value = base(context)
            mean, value = forward_with_v2v(extended, wide, rsu)
            self.assertLessEqual(float((base_mean-mean).abs().max().detach()), 1e-7)
            self.assertLessEqual(float((base_value-value).abs().max().detach()), 1e-7)
            # Equivalence holds for the original feature block under any extra values.
            wide2 = wide.clone()
            wide2[:, 29:] = torch.randn(wide2[:, 29:].shape, generator=generator)*17.
            mean2, value2 = forward_with_v2v(extended, wide2, rsu)
            self.assertLessEqual(float((base_mean-mean2).abs().max().detach()), 1e-7)
            self.assertLessEqual(float((base_value-value2).abs().max().detach()), 1e-7)

    def test_widened_layers_copy_weights_and_zero_new_inputs(self):
        reference = Policy()
        extended = Policy()
        extended.load_state_dict(reference.state_dict())
        extend_for_v2v(extended, 5)
        self.assertEqual(tuple(extended.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+5))
        self.assertEqual(tuple(extended.critic[0].weight.shape), (128, BASE_CRITIC_INPUT+5))
        # The zero gap joins after the x/pooled block; latent/rsu tails shift right.
        self.assertTrue(torch.equal(extended.actor[0].weight[:, :29], reference.actor[0].weight[:, :29]))
        self.assertTrue(torch.equal(extended.actor[0].weight[:, 34:], reference.actor[0].weight[:, 29:]))
        self.assertTrue(torch.equal(extended.critic[0].weight[:, :29], reference.critic[0].weight[:, :29]))
        self.assertTrue(torch.equal(extended.critic[0].weight[:, 34:], reference.critic[0].weight[:, 29:]))
        self.assertTrue(torch.equal(extended.actor[0].weight[:, 29:34], torch.zeros(64, 5)))
        self.assertTrue(torch.equal(extended.critic[0].weight[:, 29:34], torch.zeros(128, 5)))
        self.assertTrue(torch.equal(extended.actor[0].bias, reference.actor[0].bias))

    def test_gradients_flow_to_new_weights(self):
        extended = Policy()
        extend_for_v2v(extended, 2)
        with torch.no_grad():  # a warm-started trained head, not the fresh zero head
            extended.actor[-1].weight.normal_()
            extended.actor[-1].bias.normal_()
        x = torch.randn((4, 29))
        extra = torch.randn((4, 2), requires_grad=True)
        rsu = torch.randn(12)
        mean, value = forward_with_v2v(extended, torch.cat((x, extra), -1), rsu)
        (mean.sum()+value.sum()).backward()
        new_slice = slice(29, 29+2)  # widened block sits before latent (actor) / rsu (critic)
        actor_grad = extended.actor[0].weight.grad[:, new_slice]
        critic_grad = extended.critic[0].weight.grad[:, new_slice]
        self.assertIsNotNone(actor_grad)
        self.assertIsNotNone(critic_grad)
        self.assertGreater(float(actor_grad.abs().sum()), 0.)
        self.assertGreater(float(critic_grad.abs().sum()), 0.)
        # Zero init also blocks backward signal into the new channels at load.
        self.assertIsNotNone(extra.grad)
        self.assertEqual(float(extra.grad.abs().sum()), 0.)

    def test_optimizer_state_survives_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(3)
            policy = Policy()
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            mean, value = policy(encode(rows()))
            (mean.sum()+value.sum()).backward()
            optimizer.step()
            save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {})
            extended, remapped, generator, payload = load_checkpoint(path, v2v_dim=2)
            self.assertEqual(tuple(extended.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+2))
            self.assertEqual(tuple(extended.critic[0].weight.shape), (128, BASE_CRITIC_INPUT+2))
            self.assertEqual(remapped.param_groups[0]['lr'], 3e-4)
            self.assertEqual(len(remapped.state), len(optimizer.state))
            self.assertEqual(len(list(extended.parameters())), len(list(policy.parameters())))
            old_moment = optimizer.state[policy.critic[0].weight]['exp_avg']
            critic_moment = remapped.state[extended.critic[0].weight]['exp_avg']
            self.assertTrue(torch.equal(critic_moment[:, :29], old_moment[:, :29]))
            self.assertTrue(torch.equal(critic_moment[:, 31:], old_moment[:, 29:]))
            self.assertTrue(torch.equal(critic_moment[:, 29:31], torch.zeros(128, 2)))
            self.assertTrue(torch.equal(remapped.state[extended.critic[0].weight]['step'],
                optimizer.state[policy.critic[0].weight]['step']))
            self.assertTrue(torch.equal(extended.critic[0].weight[:, 29:31], torch.zeros(128, 2)))
            self.assertTrue(torch.equal(extended.critic[0].weight[:, :29], policy.critic[0].weight[:, :29]))
            self.assertTrue(torch.equal(extended.critic[0].weight[:, 31:], policy.critic[0].weight[:, 29:]))
            loss = sum((forward_with_v2v(extended, torch.cat((torch.randn((3, 29)),
                torch.randn((3, 2))), -1), torch.randn(12))[1]) for _ in range(2))
            remapped.zero_grad(); loss.backward(); remapped.step()  # extended optimizer still trains

    def test_flag_off_is_byte_identical_to_registered_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            generator = torch.Generator().manual_seed(9)
            save(path, policy, optimizer, generator, 7, {'charged': True})
            base = load(path)
            rng_after_base = torch.get_rng_state()
            for config in (None, {}, {'v2v': False}, {'other': 1}):
                extended = load_checkpoint(path, config)
                self.assertEqual([tuple(p.shape) for p in base[0].parameters()],
                    [tuple(p.shape) for p in extended[0].parameters()])
                for a, b in zip(base[0].state_dict().values(), extended[0].state_dict().values()):
                    self.assertTrue(torch.equal(a, b))
                self.assertEqual(base[1].state_dict()['param_groups'], extended[1].state_dict()['param_groups'])
                for key in base[1].state_dict()['state']:
                    for field in base[1].state_dict()['state'][key]:
                        self.assertTrue(torch.equal(base[1].state_dict()['state'][key][field],
                            extended[1].state_dict()['state'][key][field]))
                self.assertTrue(torch.equal(base[2].get_state(), extended[2].get_state()))
                self.assertEqual(base[3]['update'], extended[3]['update'])
                self.assertTrue(torch.equal(base[3]['torch_rng'], extended[3]['torch_rng']))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_base))

    def test_flag_on_widens_and_extended_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            save(path, policy, optimizer, torch.Generator().manual_seed(9), 0, {})
            extended, remapped, _, _ = load_checkpoint(path, {'v2v': True})
            self.assertEqual(tuple(extended.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+V2V_DIM))
            self.assertEqual(tuple(extended.critic[0].weight.shape), (128, BASE_CRITIC_INPUT+V2V_DIM))
            again = Path(tmp)/'A_checkpoint_001.pt'
            save(again, extended, remapped, torch.Generator().manual_seed(9), 1, {})
            reloaded, _, _, payload = load_checkpoint(again, v2v_dim=V2V_DIM)
            self.assertEqual(tuple(reloaded.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+V2V_DIM))
            for a, b in zip(extended.state_dict().values(), reloaded.state_dict().values()):
                self.assertTrue(torch.equal(a, b))

    def test_width_mismatch_is_rejected_and_extra_dim_is_a_parameter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            save(path, policy, torch.optim.Adam(policy.parameters(), lr=3e-4),
                torch.Generator().manual_seed(9), 0, {})
            extended = Path(tmp)/'wide.pt'
            widened = Policy(); extend_for_v2v(widened, 3)
            save(extended, widened, torch.optim.Adam(widened.parameters(), lr=3e-4),
                torch.Generator().manual_seed(9), 0, {})
            with self.assertRaises(ValueError):
                load_checkpoint(extended, v2v_dim=2)
        self.assertEqual(v2v_extra(None), 0)
        self.assertEqual(v2v_extra({}), 0)
        self.assertEqual(v2v_extra({'v2v': False}), 0)
        self.assertEqual(v2v_extra({'v2v': True}), V2V_DIM)
        self.assertEqual(v2v_extra({'v2v': True, 'v2v_dim': 7}), 7)
        self.assertEqual(v2v_extra({'v2v': True}, v2v_dim=2), 2)
        model = Policy()
        before = [tuple(p.shape) for p in model.parameters()]
        self.assertEqual(extend_for_v2v(model, 0), [])
        self.assertEqual([tuple(p.shape) for p in model.parameters()], before)
