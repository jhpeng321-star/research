"""Numerical/CRN proofs on CPU, including actual WM matched scoring and ES workers."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from tests.unit.test_eslong import make, capture
from tests.unit.test_longmem import assert_tree_close
from wms.control import speedprof, eslong
from wms.control.dreamer import _rng_state, _restore_rng


def test_cadence_gate_final_resume_and_no_rng(monkeypatch):
    trainer = SimpleNamespace(policy=SimpleNamespace(residual_on_native=True),
                              config=SimpleNamespace(entropy_weight=.001), update_count=180)
    switches = dict(entropy_decay_on_improvement=.99, signal_gate_updates=10)
    rng = torch.get_rng_state()
    assert speedprof.rescore_due(trainer, switches, 1)
    monkeypatch.setenv('WMS_SPEED_RESCORE_EVERY', '3')
    due = [i for i in range(1, 14) if speedprof.rescore_due(trainer, switches, i, final=i == 13)]
    assert due == [3, 6, 9, 10, 12, 13]
    assert [i for i in range(7, 14) if speedprof.rescore_due(trainer, switches, i, final=i == 13)] == due[2:]
    trainer.update_count = 10  # Absolute-clock signal gates also force a rescore.
    assert speedprof.rescore_due(trainer, switches, 1)
    assert torch.equal(rng, torch.get_rng_state())
    trainer.policy.residual_on_native = False
    with pytest.raises(ValueError, match='diagnostic-only'):
        speedprof.rescore_due(trainer, switches, 2)
    for bad in ('0', '-1', '1.5', 'abc'):
        monkeypatch.setenv('WMS_SPEED_RESCORE_EVERY', bad)
        with pytest.raises(ValueError, match='positive integer'):
            speedprof.rescore_every()


def real_trainer(tmp_path, monkeypatch):
    from functools import partial
    from tests.unit.test_gated300 import cct2_authority, enable_gain_only
    from tests.unit.test_longmem import Starts, Critic, queue_reward
    from wms.control import gated300
    from wms.control.bridge import make_cav_controller
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel
    from pathlib import Path
    monkeypatch.setenv('C4_EXECUTION_PARITY', '2')
    monkeypatch.setenv('C4_SUMO_LEADER_RULE', '1')
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(Path(__file__).resolve().parents[2] /
                                               'configs/execparity_cav_idm_v1.json'))
    monkeypatch.setattr(gated300, 'STEP_S', 60.)
    torch.manual_seed(72)
    model = WmSModel(ModelSpec(history_hidden=8, k_hidden=8, rollout_mode='hybrid',
        lane_model='hazard_v5', merge_decision='first_passage', lane_decision='sample',
        merge_passage_law='hybrid'))
    policy = CavPolicy(hidden_dim=8, n_heads=2, lane_head=False, residual_on_native=True,
                       residual_reference='ctrl0')
    with torch.no_grad():
        policy.residual_head.bias.fill_(-.1)  # Nonzero treatment, not only zero identity.
    def reward(*args):
        value = queue_reward(*args)
        return dict(value, hdv_hard_brake_seconds=torch.zeros_like(value['reward']))
    trainer = Policy14Trainer([model], policy, RsuGroupEncoder(hidden_dim=8, n_heads=2),
        Critic(), Starts(1, 4), observation_builder=build_observations, reward_fn=reward,
        safety=lambda state, command, **kw: command,
        controller_factory=partial(make_cav_controller, native_lane_change=True,
            native_ramp_merge='wm_law', symmetric_exec_cap=True),
        config=DreamerConfig(batch_size=1, horizon=1, imagine_entries=False, entrant_control=True, cav_exec_cap='idm',
            native_residual_train_group=True, native_residual_train_trunk=True,
            future_samples=8, pessimistic_credit=True, deterministic_crn=True),
        counterfactual=CounterfactualConfig(enabled=True, max_starts=1, n_individual=0,
            n_group=1, n_flip=0, native_cav='head', w_team=[0., 0.]))
    enable_gain_only(trainer, cct2_authority(tmp_path, hg=300, entrant_control=True),
                     ctrl0_reference=True, score_training_horizon=True)
    return trainer


def test_actual_wm_panel_reuse_bitwise_scores_and_rng(tmp_path, monkeypatch):
    from wms.control import policy14_cli as cli
    trainer = real_trainer(tmp_path, monkeypatch)
    val = SimpleNamespace(relabel=False, _load=lambda index: index,
                          _batch=lambda rows, device: copy.deepcopy(trainer.start_states.batch))
    panel = dict(starts=[dict(index=index) for index in range(16)])
    before = _rng_state(trainer.device)
    native = cli.score_native_panel(trainer, val, panel, 'cpu')
    expected = cli.score_gate_panel(trainer, val, panel, 'cpu')
    receipt = copy.deepcopy(trainer._gatemeas_panel_scores)
    assert_tree_close(before, _rng_state(trainer.device), 0, 0)
    monkeypatch.setenv('WMS_SPEED_REUSE_PANEL', '1')
    assert cli.score_native_panel(trainer, val, panel, 'cpu') == native
    monkeypatch.setattr(cli, 'independent_score', lambda *a, **k: pytest.fail('duplicate rollout'))
    assert cli.score_gate_panel(trainer, val, panel, 'cpu') == expected
    assert trainer._gatemeas_panel_scores == receipt
    assert '_speedprof_native_panel' not in trainer.__dict__  # Single-consumer cache.
    assert_tree_close(before, _rng_state(trainer.device), 0, 0)


def test_native_cadence_preserves_next_update_weights_adam_and_rng(tmp_path, monkeypatch):
    from wms.control.policy14 import independent_score
    trainer = real_trainer(tmp_path, monkeypatch)
    trainer.counterfactual = replace(trainer.counterfactual, n_individual=4)
    before = independent_score(trainer, trainer.start_states.batch)
    other = copy.deepcopy(trainer)
    switches = dict(entropy_decay_on_improvement=.99, signal_gate_updates=10)
    monkeypatch.setenv('WMS_SPEED_RESCORE_EVERY', '3')
    other_before = before
    for index in range(1, 3):
        torch.manual_seed(190+index)
        expected = trainer.update()
        after = independent_score(trainer, trainer.start_states.batch)
        if after['J'] < before['J']:
            trainer.config = replace(trainer.config, entropy_weight=trainer.config.entropy_weight*.99)
        before = after
        rng = _rng_state(trainer.device)
        torch.manual_seed(190+index)
        actual = other.update()
        if speedprof.rescore_due(other, switches, index, final=index == 2):
            other_after = independent_score(other, other.start_states.batch)
            assert_tree_close(other_after, after, 0, 0)
            if other_after['J'] < other_before['J']:
                other.config = replace(other.config, entropy_weight=other.config.entropy_weight*.99)
            other_before = other_after
        assert expected['actor_loss'] == actual['actor_loss']
        for left, right in zip(trainer.actor_parameters, other.actor_parameters):
            assert torch.equal(left, right) and torch.equal(left.grad, right.grad)
        assert_tree_close(trainer.actor_optimizer.state_dict(), other.actor_optimizer.state_dict(), 0, 0)
        assert_tree_close(rng, _rng_state(other.device), 0, 0)
@pytest.mark.parametrize('change', ['weight', 'wm', 'update', 'config', 'phase', 'relabel'])
def test_panel_reuse_invalidates_live_identity(tmp_path, monkeypatch, change):
    trainer = make(tmp_path, monkeypatch)
    trainer.gated300['config']['score_training_horizon'] = True
    val, panel = SimpleNamespace(relabel=False), dict(starts=[dict(index=0)])
    monkeypatch.setenv('WMS_SPEED_REUSE_PANEL', '1')
    old = speedprof.panel_key(trainer, val, panel, 'cpu')
    if change == 'weight':
        with torch.no_grad():
            next(trainer.policy.parameters()).add_(.1)
    elif change == 'wm':
        trainer.world_models = copy.deepcopy(trainer.world_models)
    elif change == 'update':
        trainer.update_count += 1
    elif change == 'config':
        trainer.config = replace(trainer.config, horizon=trainer.config.horizon+1)
    elif change == 'phase':
        trainer._gatemeas_score_position = 99
    else:
        val.relabel = True
    assert speedprof.panel_key(trainer, val, panel, 'cpu') != old


def test_logged_vjp_skips_and_profiler_preserve_update_bitwise(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    serial = make(tmp_path, monkeypatch)
    serial.config = replace(serial.config, actor_gradient_audit=True)
    serial.counterfactual = replace(serial.counterfactual, term_grad_audit_every=1, n_individual=3)
    other = copy.deepcopy(serial)
    other.output_dir = tmp_path
    torch.manual_seed(131)
    expected = serial.update()
    rng = _rng_state(serial.device)
    monkeypatch.setenv('WMS_SPEED_SKIP_GRAD_DIAGNOSTICS', '1')
    monkeypatch.setenv('WMS_SPEED_PROFILE', '1')
    torch.manual_seed(131)
    actual = other.update()
    for key in ('actor_loss', 'critic_loss', 'actor_grad_norm', 'grad_norm_dense',
                'grad_norm_long_es', 'eslong_pairs', 'eslong_loss_differences'):
        assert actual[key] == expected[key]
    assert 'actor_grad_norms' in expected and 'actor_grad_norms' not in actual
    assert 'cf_term_grad_norms' in expected and 'cf_term_grad_norms' not in actual
    for left, right in zip(serial.actor_parameters, other.actor_parameters):
        assert torch.equal(left, right) and torch.equal(left.grad, right.grad)
    assert_tree_close(serial.actor_optimizer.state_dict(), other.actor_optimizer.state_dict(), 0, 0)
    assert_tree_close(rng, _rng_state(other.device), 0, 0)
    records = [json.loads(line) for line in (tmp_path/'speedprof_execution.jsonl').read_text().splitlines()]
    assert len([row for row in records if row['phase'] == 'es_forward']) == 32
    other.save_checkpoint(tmp_path/'profile.pt')
    assert json.loads((tmp_path/'speedprof_execution.jsonl').read_text().splitlines()[-1])['phase'] == 'checkpoint'


@pytest.mark.parametrize('workers', [3, 4])
def test_es_worker_counts_preserve_ordered_crn_samples(tmp_path, monkeypatch, workers):
    from wms.control.parallel_futures import FuturePool
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    trainer = make(tmp_path, monkeypatch)
    trainer.output_dir = tmp_path/'workers'
    _, context = capture(trainer)
    common = {**context, 'kind': 'forward', 'eslong': True,
              'directions': torch.ones(2, 3), 'sigma': .08}
    tasks = [(direction, sign, future) for direction in range(2) for sign in (1, -1) for future in range(8)]
    before = _rng_state(trainer.device)
    expected = [eslong.forward_task(trainer, common, task) for task in tasks]
    monkeypatch.setenv('WMS_SPEED_FORWARD_WORKERS', str(workers))
    with FuturePool(trainer, common) as pool:
        actual = list(pool.map(tasks))
    _restore_rng(before, trainer.device)
    assert_tree_close(expected, actual, 0, 0)
