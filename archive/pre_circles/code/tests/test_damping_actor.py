import io

import torch

from env.native_damping import DampingContract
from idm.damping_actor import SharedDampingActor


def test_zero_initial_policy_respects_asymmetric_physical_bounds():
    actor = SharedDampingActor()
    observations = torch.randn(24, DampingContract().observation_dim)
    torch.testing.assert_close(actor(observations), torch.zeros(24, 1), atol=1e-8, rtol=0)


def test_bounded_samples_correct_log_probability_and_finite_learning_gradient():
    torch.manual_seed(7)
    actor = SharedDampingActor()
    obs = torch.randn(32, DampingContract().observation_dim)
    actions, old_logp = actor.sample(obs)
    assert (actions >= -0.5).all() and (actions <= 0.3).all()
    logp = actor.log_prob(obs, actions.detach())
    torch.testing.assert_close(logp, old_logp, atol=1e-5, rtol=1e-5)
    # A synthetic preference checks gradient plumbing, not traffic efficacy.
    loss = -(logp * (-actions.detach().squeeze(-1))).mean()
    loss.backward()
    assert actor.mean_head.bias.grad.abs().item() > 0
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


def test_actor_checkpoint_roundtrip_retains_predictions():
    actor = SharedDampingActor()
    with torch.no_grad():
        actor.mean_head.weight.fill_(0.01)
    buffer = io.BytesIO()
    torch.save(actor.state_dict(), buffer)
    restored = SharedDampingActor()
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    obs = torch.randn(4, DampingContract().observation_dim)
    torch.testing.assert_close(actor(obs), restored(obs), rtol=0, atol=0)
