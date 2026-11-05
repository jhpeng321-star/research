"""F1: real trigger/executor/controller command contracts, entirely on CPU."""
from copy import deepcopy
import math

import pytest
import torch

from . import track2_backend as tb
from .acceleration import Controller, PHI
from .feasible import issued_target
from .test_track2_backend_cpu import cfg, cpu_threads, rows, policy
from .v2v_runtime import controller_class


class PreferMove(torch.nn.Module):
    """Test-only deterministic preference; candidate/decision/execution are real."""
    def __init__(self, descriptor_width):
        super().__init__()
        self.descriptor_width = descriptor_width

    def forward(self, inputs):
        # The first descriptor column is the 'none' family indicator.
        return -1000. * inputs[..., -self.descriptor_width, None]


def selected_policy(cfg, monkeypatch, family, arm='B', *, onset=2):
    p = policy(cfg)
    p.arm = arm
    with torch.no_grad():
        p.layer.base.actor[-1].weight.zero_()
        p.layer.base.actor[-1].bias.fill_(.125)
        # Keep this command-equality fixture strictly inside the plant's clips.
        p.layer.base.log_std.fill_(math.log(.125))
    p.trigger.trigger_head = PreferMove(len(tb.moves.FAMILIES)+cfg['trigger_partner_slots']+3)

    def candidates(state, rng, n):
        members = (state['self'],) if family == 'ease_off' else ('c1', 'c2')
        return [tb.TriggerMove(tb.moves.Move(family, members, onset, 20, .5))]
    monkeypatch.setattr(tb.MoveAPI, 'candidates', staticmethod(candidates))
    return p


@pytest.mark.parametrize('arm', ['B', 'R'])
@pytest.mark.parametrize('deterministic', [False, True])
@pytest.mark.parametrize('family', tb.moves.STRUCTURED)
def test_selected_move_changes_commands_by_exact_library_offsets_from_u0(
        cfg, monkeypatch, arm, deterministic, family):
    p = selected_policy(cfg, monkeypatch, family, arm)
    assert p.layer.trigger_gain.item() == 0.
    assert torch.count_nonzero(p.layer.residual[-1].weight) == 0
    assert torch.count_nonzero(p.layer.residual[-1].bias) == 0
    frozen = deepcopy(p.layer.base.state_dict())
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), deterministic)
    base_controller = controller_class(Controller)(p.layer.base,
        torch.Generator().manual_seed(9), deterministic)
    commands = {'c1': .25, 'c2': .25}
    observed = rows()
    # Full onset, hold and expiry; the same common-base AR noise is used at each tick.
    saw_nonzero = False
    for now in range(300, 323):
        p.observe(observed, now)
        actual = controller.apply(observed, commands, now)
        base = base_controller.apply(observed, commands, now)
        sample = controller.events[-1]['sample']
        if now == 300:
            trigger = sample['track2_trigger']
            assert trigger['actions'] == [1, 1]
            assert trigger['accepted']['c1']
            if family != 'ease_off':
                assert not trigger['accepted']['c2']  # one owner of the joint move
            accepted_moves = [e['move'] for e in p.executor.active]
        expected = {'c1': 0., 'c2': 0.}
        for move in accepted_moves:
            expected.update(tb.moves.offsets(move.move, now-300,
                state={'s_m': {r['vid']: r['s_m'] for r in observed}}))
        assert sample['context']['move_offsets'] == expected
        for vid, offset in expected.items():
            raw = actual.get(vid, {}).get('requested_raw', commands[vid])
            base_raw = base.get(vid, {}).get('requested_raw', commands[vid])
            assert raw-base_raw == pytest.approx(offset, abs=2e-7)
            # Inside existing feasibility limits, the physical target changes by
            # exactly the library offset too (one-second plant, no simulation).
            target = issued_target(10., raw, 31.29, accel_min=-3., accel_max=1.2)
            base_target = issued_target(10., base_raw, 31.29, accel_min=-3., accel_max=1.2)
            assert target-base_target == pytest.approx(offset, abs=2e-7)
            saw_nonzero |= offset != 0.
        assert set(actual) <= {'c1', 'c2'}  # never commands the sensed HDV
        assert controller.memory == base_controller.memory
        assert torch.equal(controller.generator.get_state(), base_controller.generator.get_state())
        if not deterministic:
            assert float(p.score(sample)[0].detach()) == pytest.approx(sample['logp'])
        assert float(p.layer.joint_kl(sample).detach()) == pytest.approx(0., abs=1e-7)
    assert saw_nonzero
    assert all(torch.equal(v, p.layer.base.state_dict()[k]) for k, v in frozen.items())


@pytest.mark.parametrize('legacy_gain', [0., -10., 10.])
def test_residual_is_additive_and_forward_scores_kl_share_production_mean(cfg, monkeypatch, legacy_gain):
    p = selected_policy(cfg, monkeypatch, 'wall', onset=0)
    with torch.no_grad():
        p.layer.trigger_gain.fill_(legacy_gain)
        p.layer.residual[-1].bias.fill_(.25)
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
    p.observe(rows(), 300)
    controller.apply(rows(), {'c1': 0., 'c2': 0.}, 300)
    sample = controller.events[-1]['sample']
    context = sample['context']
    mean, _ = p(context)
    base_mean = p.layer.base(context)[0]
    assert mean.detach().numpy() == pytest.approx((base_mean + .25 - .5).numpy())
    assert sample['coop_layer']['old_gain'] == 1.
    assert sample['coop_layer']['old_residual'] == [-.25, -.25]
    sigma = p.log_std.detach().exp().reshape(()) * (1-PHI**2)**.5
    conditional = mean.reshape(-1) + PHI*torch.tensor(sample['previous_noise']).reshape(-1)
    expected_scores = torch.distributions.Normal(conditional, sigma).log_prob(
        torch.tensor(sample['own_action']).reshape(-1))
    assert p.layer.member_scores(sample).detach().numpy() == pytest.approx(expected_scores.detach().numpy())
    assert p.layer.member_scores(sample, old=True).numpy() == pytest.approx(expected_scores.detach().numpy())
    assert float(p.score(sample)[0].detach()) == pytest.approx(sample['logp'])
    p(context)[0].sum().backward()
    assert p.layer.trigger_gain.grad is None
    assert p.layer.residual[-1].bias.grad.item() == 2.
    assert all(v.grad is None for v in p.layer.base.parameters())
    with torch.no_grad():
        p.layer.residual[-1].bias.add_(.01)
    from .bank_imitation import joint_kl
    reference = deepcopy(p)
    with torch.no_grad():
        reference.layer.residual[-1].bias.sub_(.01)
    assert float(p.layer.joint_kl(sample).detach()) > 0.
    assert float(p.layer.joint_kl(sample).detach()) == pytest.approx(
        joint_kl(p, reference, [sample]), rel=1e-4)


def test_b_r_actuation_and_worker_restore_identical_even_with_legacy_gain(cfg):
    original = policy(cfg)
    # Simulate an old Adam sidecar with moments for its formerly learned gain.
    original.layer.trigger_gain.grad = torch.ones(())
    original.layer.optimizer.step()
    original.layer.optimizer.zero_grad(set_to_none=True)
    original.layer.trigger_gain.data.fill_(-.007765)  # old scalar cannot invert the move
    results = []
    for arm in ('B', 'R'):
        restored = tb.EpisodePolicy(original.layer.base, cfg, arm)
        restored.restore_worker(original.worker_state())
        restored.layer.optimizer.load_state_dict(original.layer.optimizer.state_dict())
        restored.last_decision = 300
        move = tb.TriggerMove(tb.moves.Move('ramp_assist', ('c1', 'c2'), 0, 20, .5))
        assert restored.executor.start(move, ego='c1', observable_cav_ids=['c1', 'c2'], now=300, until=320)
        controller = controller_class(Controller)(restored, torch.Generator().manual_seed(9), False)
        restored.observe(rows(), 301)
        commands = controller.apply(rows(), {'c1': .25, 'c2': .25}, 301)
        sample = deepcopy(controller.events[-1]['sample'])
        assert sample['context']['move_offsets'] == dict(c1=-.5, c2=-.5)
        assert sample['coop_layer']['old_residual'] == [-.5, -.5]
        gain = restored.layer.trigger_gain.detach().clone()
        moments = deepcopy(restored.layer.optimizer.state[restored.layer.trigger_gain])
        report = restored.layer.update_rewards([dict(sample=sample, returns=[1., -1.])], epochs=1)
        assert report[0]['accepted']
        assert torch.equal(restored.layer.trigger_gain, gain)
        assert all(torch.equal(v, restored.layer.optimizer.state[restored.layer.trigger_gain][k])
                   for k, v in moments.items())
        assert restored.layer.trigger_gain.grad is None
        sample['track2'].pop('arm')
        results.append((commands, sample, controller.memory, report))
    assert results[0] == results[1]


def test_a_episode_hook_and_commands_unaffected(cfg):
    base = policy(cfg).layer.base
    out, factory = tb.episode_components(dict(track2_backend=True, track2_config=cfg, arm='A'),
        base, controller_class(Controller))
    assert out is base
    # Use the same controller class object to check the exact A passthrough.
    cls = controller_class(Controller)
    assert tb.episode_components(dict(track2_backend=True, track2_config=cfg, arm='A'), base, cls) == (base, cls)
    actual = factory(out, torch.Generator().manual_seed(9), False)
    expected = cls(base, torch.Generator().manual_seed(9), False)
    for now in range(300, 303):
        assert actual.apply(rows(), {'c1': .25, 'c2': .25}, now) == expected.apply(
            rows(), {'c1': .25, 'c2': .25}, now)
        assert actual.events[-1]['sample'] == expected.events[-1]['sample']
        assert 'coop_layer' not in actual.events[-1]['sample']
