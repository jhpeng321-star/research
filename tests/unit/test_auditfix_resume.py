"""New-phase configuration migration keeps state, but never silently inherits settings."""
from dataclasses import asdict, replace

import pytest
import torch

from tests.unit.test_longmem import trainer as make_trainer
from wms.control.phase_resume import resume_diff


def test_new_phase_refuses_before_mutating_and_keeps_toml_and_adam(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make_trainer(horizon=5, checkpoint_every=100, actor_lr=5e-5)
    # Real Adam moments/step, not an empty-state load test.
    sum(p.square().sum() for p in old.actor_parameters).backward()
    old.actor_optimizer.step()
    old.update_count = 180
    path = tmp_path / 'c1.pt'
    old.save_checkpoint(path)
    new = make_trainer(horizon=7, checkpoint_every=1, actor_lr=1e-3, critic_lr=2e-3)
    new.reward_weights = replace(new.reward_weights, delay=2.)
    requested = asdict(new.config)
    before = [p.clone() for p in new.actor_parameters]
    payload = torch.load(path, weights_only=True)
    diff = resume_diff(new.config, new.reward_weights, payload)
    with pytest.raises(ValueError, match='unapproved'):
        new.load_checkpoint(path, new_phase=True)
    for a, b in zip(before, new.actor_parameters):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='unused'):
        new.load_checkpoint(path, new_phase=True, resume_allow=[*diff, 'trainer.typo'])
    new.load_checkpoint(path, new_phase=True, resume_allow=list(diff))
    assert asdict(new.config) == requested
    assert new.reward_weights.delay == 2.
    assert new.update_count == 180
    assert new.actor_optimizer.param_groups[0]['lr'] == 1e-3
    assert new.critic_optimizer.param_groups[0]['lr'] == 2e-3
    for p, q in zip(old.actor_parameters, new.actor_parameters):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(old.actor_optimizer.state[p][key],
                                       new.actor_optimizer.state[q][key], rtol=0, atol=0)


def test_new_phase_writes_requested_checkpoint_each_update(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    old = make_trainer(checkpoint_every=100)
    path = tmp_path / 'c1.pt'
    old.save_checkpoint(path)
    new = make_trainer(checkpoint_every=1)
    new.load_checkpoint(path, new_phase=True, resume_allow=['trainer.checkpoint_every'])
    new.output_dir = tmp_path
    new.update()
    assert (tmp_path / 'policy_000001.pt').is_file()
