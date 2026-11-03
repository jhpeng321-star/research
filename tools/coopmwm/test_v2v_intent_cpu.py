"""CPU gates: v2v_intent head (part B, branch c4/intent-head).

Gates (all CPU, no SUMO/GPU):
  1. option resolution: v2v_intent DEFAULT OFF; the intent v2v block is the plain
     6-feature partner block + H intents + valid bit per partner (K*(6+H+1)=48);
  2. executed action IDENTICAL at load: zero-init widened inputs and a
     zero-init intent head that never feeds the actor;
  3. checkpoint save/load round trip with the intent head (and plain->intent);
  4. intent head trains: supervised planned-vs-realised next-H error falls on a
     synthetic sequence, through accel_learner.update (PPO, all arms identically);
  5. masking at episode end / despawn in intent_examples;
  6. option-off byte-identical: loader, update metric schema, bank pairs;
  7. bank plan-future-command intent targets (bank_v2v/bank_imitation).
"""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.coopnative.test_cpu import rows
from . import bank_imitation
from . import bank_v2v
from .accel_learner import load as base_load, save, update
from .acceleration import Policy, encode
from .plans import acceleration_at
from .v2v_integrate import widen_context
from .v2v_model import (DEFAULT_INTENT_WEIGHT, INTENT_BOUNDS, V2V_INTENT_DIM, V2V_INTENT_H,
    add_intent_head, forward_with_intent, intent_config, intent_error, intent_examples,
    intent_loss, intent_options, intent_v2v_dim, load_checkpoint, planned_commands, v2v_extra)

BASE_ACTOR_INPUT = 29+4
MEMBERS = [r['vid'] for r in rows() if r['is_cav']]  # sorted: downstream, ramp, upstream


def sensor(i):
    """The whitelisted sensors with per-second speeds (targets stay learnable)."""
    return [dict(r, v_mps=r['v_mps']+i*.1) for r in rows()]


def realised(sensor_rows, vid, t):
    """Synthetic truth: realised absolute command is a linear function of the sensed
    speed (the policy observation carries v/30, so the head CAN learn it)."""
    v = next(r['v_mps'] for r in sensor_rows if r['vid'] == vid)
    return min(max((v-10.)/6., INTENT_BOUNDS[0]), INTENT_BOUNDS[1])


def synthetic_episode(length=12):
    """Steps with whitelisted contexts plus commands/requests (the rollout tape fields)."""
    generator = torch.Generator().manual_seed(23)
    policy = Policy()
    steps = []
    for i in range(length):
        sensor_rows = sensor(i)
        context = encode(sensor_rows)
        sample, _ = policy.sample(context, generator)
        commands = {r['vid']: .3 for r in sensor_rows if r['is_cav']}
        requests = {vid: dict(requested_raw=realised(sensor_rows, vid, i)) for vid in context['members']}
        steps.append(dict(time_s=300+i, sample=sample, reward=-1., commands=commands, requests=requests))
    return dict(steps=steps, world=dict(seed=1, cell=0))


def intent_kwargs(**overrides):
    cfg = dict(horizon=V2V_INTENT_H, weight=1., bounds=INTENT_BOUNDS)
    cfg.update(overrides)
    return cfg


def frozen_trunk(policy):
    """Freeze every non-intent parameter so the registered KL early-stop cannot
    fire during the head-training gates; the head's trainability is isolated."""
    for name, param in policy.named_parameters():
        if not name.startswith('intent.'):
            param.requires_grad_(False)
    return policy


class V2vIntentCPU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)

    # ---- 1. option resolution ------------------------------------------------
    def test_intent_option_default_off_and_block_width(self):
        self.assertEqual(intent_options(None), (0, 0.))
        self.assertEqual(intent_options({}), (0, 0.))
        self.assertEqual(intent_options({'v2v': True}), (0, 0.))
        self.assertEqual(intent_options({'v2v_intent': False}), (0, 0.))
        self.assertEqual(intent_options({'v2v_intent': True}), (V2V_INTENT_H, DEFAULT_INTENT_WEIGHT))
        self.assertEqual(intent_options({'v2v_intent': True, 'v2v_intent_horizon': 3, 'v2v_intent_weight': .5}),
                         (3, .5))
        # block = existing 6 partner features + H intents + valid bit, K partners
        self.assertEqual(V2V_INTENT_DIM, 4*(6+V2V_INTENT_H+1))
        self.assertEqual(intent_v2v_dim(V2V_INTENT_H), V2V_INTENT_DIM)
        self.assertEqual(intent_v2v_dim(3), 4*10)
        self.assertEqual(v2v_extra(None), 0)
        self.assertEqual(v2v_extra({'v2v_intent': True}), V2V_INTENT_DIM)  # supersedes plain v2v
        self.assertEqual(v2v_extra({'v2v_intent': True, 'v2v_intent_horizon': 3}), 40)
        self.assertEqual(v2v_extra({'v2v_intent': True}, v2v_dim=7), 7)
        self.assertIsNone(intent_config(None))
        self.assertIsNone(intent_config({}))
        cfg = intent_config({'v2v_intent': True, 'accel_bounds': [-4., 2.]})
        self.assertEqual(cfg, dict(horizon=5, weight=DEFAULT_INTENT_WEIGHT, bounds=(-4., 2.)))

    # ---- 2. executed action identical at load --------------------------------
    def test_action_identical_at_load_with_intent_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            save(path, policy, torch.optim.Adam(policy.parameters(), lr=3e-4),
                 torch.Generator().manual_seed(9), 0, {})
            extended, _, _, _ = load_checkpoint(path, config={'v2v_intent': True, 'accel_bounds': INTENT_BOUNDS})
            self.assertEqual(tuple(extended.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+V2V_INTENT_DIM))
            self.assertEqual(tuple(extended.critic[0].weight.shape), (128, 29+12+V2V_INTENT_DIM))
            self.assertTrue(torch.equal(extended.actor[0].weight[:, 29:29+V2V_INTENT_DIM], torch.zeros(64, V2V_INTENT_DIM)))
            self.assertTrue(torch.equal(extended.intent.weight, torch.zeros(V2V_INTENT_H, 64)))
            self.assertTrue(torch.equal(extended.intent.bias, torch.zeros(V2V_INTENT_H)))
            # executed action identical on the plain context...
            context = encode(rows())
            base_mean, base_value = policy(context)
            mean, value = forward_with_intent(extended, torch.tensor(context['x']), context['rsu'])
            self.assertLessEqual(float((base_mean-mean).abs().max()), 1e-7)
            self.assertLessEqual(float((base_value-value).abs().max()), 1e-7)
            # ...and under ANY (random) intent/v2v channel values: zero-init columns.
            generator = torch.Generator().manual_seed(3)
            wide = torch.randn((len(context['x']), 29+V2V_INTENT_DIM), generator=generator)
            mean2, value2 = forward_with_intent(extended, wide, context['rsu'])
            self.assertLessEqual(float((base_mean-mean2).abs().max()), 1e-7)
            self.assertLessEqual(float((base_value-value2).abs().max()), 1e-7)
            # the intent plan is clipped to the action clip and starts neutral
            planned = forward_with_intent(extended, wide, context['rsu'])[2]
            self.assertTrue(torch.equal(planned, torch.zeros(len(wide), V2V_INTENT_H)))

    # ---- 3. checkpoint round trip --------------------------------------------
    def test_checkpoint_round_trip_with_intent_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            add_intent_head(policy, V2V_INTENT_H, INTENT_BOUNDS)
            with torch.no_grad():
                policy.intent.weight.normal_()
                policy.intent.bias.fill_(.25)
            optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
            save(path, policy, optimizer, torch.Generator().manual_seed(9), 4, {})
            # intent checkpoints only reload with the option on
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)
            reloaded, _, _, _ = load_checkpoint(path, config={'v2v_intent': True, 'accel_bounds': INTENT_BOUNDS})
            self.assertEqual(tuple(reloaded.actor[0].weight.shape), (64, BASE_ACTOR_INPUT+V2V_INTENT_DIM))
            for a, b in zip(policy.state_dict().values(), reloaded.state_dict().values()):
                self.assertTrue(torch.equal(a, b))
            # a plain checkpoint loads with the option on: zero-init intent head
            plain = Path(tmp)/'A_checkpoint_001.pt'
            base = Policy()
            save(plain, base, torch.optim.Adam(base.parameters(), lr=3e-4),
                 torch.Generator().manual_seed(9), 0, {})
            mixed, _, _, _ = load_checkpoint(plain, config={'v2v_intent': True, 'accel_bounds': INTENT_BOUNDS})
            self.assertTrue(torch.equal(mixed.intent.weight, torch.zeros(V2V_INTENT_H, 64)))
            self.assertTrue(torch.equal(mixed.actor[0].weight[:, :29], base.actor[0].weight[:, :29]))

    # ---- 4. the intent head trains on its own rollout commands ---------------
    def test_intent_head_trains_error_falls_on_synthetic_sequence(self):
        episode = synthetic_episode()
        before_pack = intent_examples([episode], V2V_INTENT_H, INTENT_BOUNDS)
        samples = [s['sample'] for s in episode['steps']]
        policy = frozen_trunk(Policy())
        add_intent_head(policy, V2V_INTENT_H, INTENT_BOUNDS)
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=.02)
        before = intent_error(policy, samples, *before_pack)
        self.assertIsNotNone(before)
        metrics = update(policy, optimizer, [episode], torch.Generator().manual_seed(11),
                         epochs=120, intent=intent_kwargs())
        after = metrics[-1]['intent']['error_after_mps2']
        self.assertEqual(metrics[-1]['intent']['error_before_mps2'], before)
        self.assertIsNotNone(after)
        self.assertLess(after, before)          # the head really trains through update()
        self.assertLess(after, .2*before)       # to near zero on this learnable sequence

    def test_intent_loss_reaches_the_shared_actor_trunk(self):
        episode = synthetic_episode(4)
        samples = [s['sample'] for s in episode['steps']]
        pack = intent_examples([episode], V2V_INTENT_H, INTENT_BOUNDS)
        policy = Policy()
        add_intent_head(policy, V2V_INTENT_H, INTENT_BOUNDS)
        with torch.no_grad():  # a warmed head (zero init blocks the trunk path at load)
            policy.intent.weight.fill_(.01)
        loss = intent_loss(policy, samples[:2], pack[0][:2], pack[1][:2])
        loss.backward()
        self.assertGreater(float(policy.actor[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(policy.intent.weight.grad.abs().sum()), 0.)

    def test_intent_training_reaches_the_head_through_wide_v2v_contexts(self):
        class WideIntentPolicy(Policy):
            def __init__(self):
                super().__init__()
                from .v2v_model import extend_for_v2v, forward_with_v2v

                def forward(context):
                    x = torch.as_tensor(context['x'], dtype=torch.float32)
                    if x.shape[1] == 29:
                        x = torch.cat((x, torch.zeros(len(x), V2V_INTENT_DIM)), -1)
                    return forward_with_v2v(self, x, torch.as_tensor(context['rsu'], dtype=torch.float32))
                self.forward = forward
                extend_for_v2v(self, V2V_INTENT_DIM, V2V_INTENT_H, INTENT_BOUNDS)
        episode = synthetic_episode()
        for s in episode['steps']:
            s['sample']['context'] = widen_context(s['sample']['context'],
                {vid: [0.]*V2V_INTENT_DIM for vid in s['sample']['context']['members']})
        policy = frozen_trunk(WideIntentPolicy())
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=.02)
        before = intent_error(policy, [s['sample'] for s in episode['steps']],
                              *intent_examples([episode], V2V_INTENT_H, INTENT_BOUNDS))
        metrics = update(policy, optimizer, [episode], torch.Generator().manual_seed(11),
                         epochs=120, intent=intent_kwargs())
        after = metrics[-1]['intent']['error_after_mps2']
        self.assertIsNotNone(after)
        self.assertLess(after, .2*before)

    # ---- 5. masking at episode end / despawn ---------------------------------
    def test_targets_masked_at_episode_end_and_despawn(self):
        episode = synthetic_episode(3)
        steps = episode['steps']
        del steps[2]['commands']['downstream']           # despawned after step 1
        del steps[2]['requests']['downstream']
        targets, masks = intent_examples([episode], V2V_INTENT_H, INTENT_BOUNDS)
        self.assertEqual(len(targets), 3)
        members = steps[0]['sample']['context']['members']
        row = members.index('downstream')
        other = members.index('ramp')
        # step 0: horizon 5, steps 1-2 exist, but downstream despawned at step 2
        self.assertEqual(float(masks[0][row].sum()), 1.)
        self.assertEqual(float(masks[0][row][2:].sum()), 0.)
        self.assertAlmostEqual(float(targets[0][row][0]), realised(sensor(1), 'downstream', 1), places=6)
        # step 2 (last): everything masked at episode end
        self.assertEqual(float(masks[2].sum()), 0.)
        # despawn: downstream gone from step 2's commands -> masked there
        self.assertEqual(float(masks[1][row].sum()), 0.)
        self.assertEqual(float(masks[1][row][0]), 0.)
        self.assertEqual(float(masks[1][other].sum()), 1.)  # episode end masks j>=2
        # realised commands are clipped to the registered action clip
        steps[1]['requests']['ramp'] = dict(requested_raw=99.)
        targets, masks = intent_examples([episode], 1, INTENT_BOUNDS)
        self.assertAlmostEqual(float(targets[0][other][0]), INTENT_BOUNDS[1], places=6)
        # the baseline command is the fallback where no request row exists
        del steps[1]['requests']['ramp']
        targets, _ = intent_examples([episode], 1, INTENT_BOUNDS)
        self.assertAlmostEqual(float(targets[0][other][0]), .3, places=6)

    # ---- 6. option-off byte-identical ----------------------------------------
    def test_option_off_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'A_checkpoint_000.pt'
            torch.manual_seed(5)
            policy = Policy()
            generator = torch.Generator().manual_seed(9)
            save(path, policy, torch.optim.Adam(policy.parameters(), lr=3e-4), generator, 7, {'charged': 1})
            base = base_load(path)
            rng_after = torch.get_rng_state()
            for config in (None, {}, {'v2v_intent': False}):
                extended = load_checkpoint(path, config)
                self.assertEqual([tuple(p.shape) for p in base[0].parameters()],
                                 [tuple(p.shape) for p in extended[0].parameters()])
                self.assertFalse(hasattr(extended[0], 'intent'))
                for a, b in zip(base[0].state_dict().values(), extended[0].state_dict().values()):
                    self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_after))
            # plain v2v (v2v_intent off) widens by the 24-wide partner block, no intent head
            widened = load_checkpoint(path, {'v2v': True})[0]
            self.assertEqual(tuple(widened.actor[0].weight.shape), (64, 29+4+24))
            self.assertFalse(hasattr(widened, 'intent'))
        # update(): registered metric schema exactly, no intent fields
        episode = synthetic_episode(4)
        policy = Policy()
        metrics = update(policy, torch.optim.Adam(policy.parameters(), lr=3e-4), [episode],
                         torch.Generator().manual_seed(13), epochs=1)
        self.assertEqual(set(metrics[0]), {'loss', 'kl', 'gradient_norm', 'imitation', 'own_policy_steps',
                                           'critic_steps', 'awr_positive_steps', 'loss_rule'})
        self.assertEqual(len(metrics), 1)  # no intent summary appended
        # bank pairs without the option carry no intent targets
        pairs = [(dict(context=dict(members=['cav0'], x=[[0.]*29], rsu=[0.]*12),
                       behaviour=dict(proposal_members=['cav0']), action=[[0.]]), 1.)]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp)/'observable.jsonl.gz'
            frames = [dict(time_s=9, vehicles=[dict(vid='cav0', s_m=0., v_mps=5., lane=0, is_cav=True,
                                                    length_m=5., issued_acceleration_mps2=.1)]),
                      dict(time_s=10, vehicles=[dict(vid='cav0', s_m=5., v_mps=5., lane=0, is_cav=True,
                                                     length_m=5., issued_acceleration_mps2=.2)])]
            with gzip.open(trace, 'wt', encoding='utf-8') as stream:
                for frame in frames:
                    stream.write(json.dumps(frame)+'\n')
            plan = dict(id='p', members=['cav0'], duration_s=30, knots=[[0., 1., 2.]])
            kept_off, _ = bank_v2v.attach_pairs([(dict(s['sample']), w) for s, w in pairs], trace, 10, plan=plan)
            self.assertEqual(len(kept_off), 1)
            self.assertNotIn('intent_target', kept_off[0][0]['context'])
            kept_on, _ = bank_v2v.attach_pairs([(dict(s['sample']), w) for s, w in pairs], trace, 10,
                                               plan=plan, intent=True, intent_horizon=3)
            self.assertIn('intent_target', kept_on[0][0]['context'])
            off = {k: v for k, v in kept_off[0][0]['context'].items()}
            on = {k: v for k, v in kept_on[0][0]['context'].items() if k != 'intent_target'}
            self.assertEqual(off, on)

    # ---- 7. bank plan-future-command intent targets --------------------------
    def test_bank_plan_future_intent_targets_and_training(self):
        plan = dict(id='p', members=['downstream', 'ramp'], duration_s=30,
                    knots=[[0., 1., 2.], [0., -1., -2.]])  # 10 s knots, plans.acceleration_at
        targets = bank_v2v.plan_intent_targets(plan, 24, ['downstream', 'ramp', 'upstream'], 5)
        self.assertNotIn('upstream', targets)               # member outside the plan: no target
        at26 = acceleration_at(plan, 26)
        for vid in ('downstream', 'ramp'):
            self.assertEqual(targets[vid]['mask'], [1., 0., 0., 0., 0.])  # plan ends at 30
            self.assertAlmostEqual(targets[vid]['target'][0], at26[vid], places=7)
        # bank_imitation trains the head on those pairs and reports before/after
        policy = frozen_trunk(Policy())
        add_intent_head(policy, V2V_INTENT_H, INTENT_BOUNDS)
        reference = Policy()
        reference.load_state_dict(policy.state_dict())
        add_intent_head(reference, V2V_INTENT_H, INTENT_BOUNDS)
        pairs = []
        for i in range(8):
            context = encode(sensor(i))
            sample = dict(context=dict(context, intent_target=bank_v2v.plan_intent_targets(
                plan, 0, context['members'], V2V_INTENT_H)),
                behaviour=dict(proposal_members=list(context['members'])),
                action=[[0.] for _ in context['members']])
            pairs.append((sample, 1.))
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=.02)
        report = bank_imitation.run_separate(policy, optimizer, pairs, torch.Generator().manual_seed(7),
                                             reference, passes=5, intent=dict(weight=1., horizon=V2V_INTENT_H))
        self.assertIn('intent', report)
        self.assertEqual(report['intent']['weight'], 5.)
        self.assertIsNotNone(report['intent']['error_before_mps2'])
        self.assertIsNotNone(report['intent']['error_after_mps2'])
        self.assertLess(report['intent']['error_after_mps2'], report['intent']['error_before_mps2'])
        # the head changed on the intent loss; legacy run() stays intent-free
        self.assertGreater(float(policy.intent.weight.abs().sum()), 0.)
        legacy = bank_imitation.run(policy, optimizer, pairs, torch.Generator().manual_seed(7),
                                    reference, passes=5)
        self.assertNotIn('intent', legacy)
        # option-off run_separate reports no intent block at all
        plain = bank_imitation.run_separate(policy, optimizer, pairs, torch.Generator().manual_seed(7),
                                            reference, passes=5)
        self.assertNotIn('intent', plain)


if __name__ == '__main__':
    unittest.main()
