"""Same-phase continuation retains the phase origin, curriculum and optimizer."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit.test_longmem import assert_tree_close
from wms.control import policy14_recovery as recovery
from wms.control.contract import RewardWeights
from wms.control.dreamer import DreamerConfig

pytest_plugins = ['tests.unit.test_policy14_recovery']


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'checked_path', lambda p: Path(p).resolve())
    root = tmp_path / 'prior'
    root.mkdir()
    files = {}
    for key in ('gate_panel', 'wm', 'wm_config'):
        path = tmp_path / key
        path.write_text(key)
        files[key] = str(path)
    checkpoint = root / 'policy_000189.pt'
    torch.save({'update_count': 189}, checkpoint)
    torch.save({'update_count': 182}, root / 'policy_native_best.pt')
    def digest(key):
        return hashlib.sha256(Path(files[key]).read_bytes()).hexdigest()
    raw = {'trainer': asdict(DreamerConfig()), 'reward': asdict(RewardWeights())}
    prior = {'status': 'failed', 'error': "KeyError: 'cav_exec_cap_mps2'", 'seed': 965140,
        'resumed_from': {'update_count': 182},
        'updates': [{'update': u} for u in range(183, 190)],
        'effective_config_before_updates': raw['trainer'],
        'effective_reward_before_updates': raw['reward'],
        'native_curve': [{'update': 182, 'passed': True, 'mean_delta': -3.,
                          'panel_sha256': recovery.PANEL_SHA256}],
        'gate_panel': {'sha256': digest('gate_panel'), 'J_before': [10., 20.]},
        'wm_members': [{'checkpoint_sha256': digest('wm'), 'config_sha256': digest('wm_config')}]}
    (root / 'RUN_SUMMARY.json').write_text(json.dumps(prior))
    args = SimpleNamespace(continue_run=str(root), new_phase=False, resume_allow=[],
        native_init=True, resume=str(checkpoint), seed=965140, max_updates=150,
        phase_relative_signal_gate=True, gate_panel=files['gate_panel'],
        wm_ckpt=[files['wm']], wm_config=[files['wm_config']], output_dir=str(tmp_path / 'fresh'))
    return raw, args, prior


def test_gate_192_and_remaining_original_phase_budget(continuation):
    from tools.candA.authority import MANIFEST_FILES
    assert {'src/wms/model/wms.py', 'src/wms/control/policy14_recovery.py',
            'tools/candA/stepK.sh'} <= set(MANIFEST_FILES)
    raw, args, prior = continuation
    result = recovery.load_continuation(raw, args)
    assert args.max_updates == 143 and args.critic_warmup_updates == 0
    assert result['phase_start'] == 182
    assert 192 - result['phase_start'] == 10
    assert result['prior']['gate_panel']['J_before'] == prior['gate_panel']['J_before']
    assert result['updates'][-1]['update'] == result['saved_update'] == 189
    assert result['phase_updates'] == 150


def test_cli_continuation_validates_original_budget_before_remaining(tmp_path, monkeypatch):
    from wms.control import nativeinit, policy14_cli as cli
    panel = tmp_path / 'panel.json'
    panel.write_text('{}')
    monkeypatch.setattr(cli, 'checked_path', Path)
    monkeypatch.setattr(nativeinit, 'PANEL_SHA256', hashlib.sha256(panel.read_bytes()).hexdigest())
    monkeypatch.setattr(cli, '_read_config', lambda _: {'policy14': {}, 'trainer': {}})
    monkeypatch.setattr(recovery, 'load_recovery', lambda *args: None)
    seen = []

    def load(raw, args):
        seen.append(args.continue_run)
        args.max_updates = 143
        return {'phase_updates': 150}

    monkeypatch.setattr(recovery, 'load_continuation', load)
    def reached_config(**kwargs):
        raise ValueError('reached trainer after same-phase budget validation')
    monkeypatch.setattr(cli, 'DreamerConfig', reached_config)
    printed = []
    monkeypatch.setattr('builtins.print', lambda text: printed.append(json.loads(text)))
    assert cli.main(['--config', 'unused.toml', '--wm-ckpt', 'unused.pt', '--wm-config', 'unused.toml',
        '--real-returns', 'unused.npz', '--output-dir', str(tmp_path / 'out'),
        '--seed', '965140', '--max-updates', '150', '--native-init', '--resume', 'policy_000189.pt',
        '--continue-run', 'prior-run', '--gate-panel', str(panel), '--compact']) == 1
    assert seen == ['prior-run']
    assert 'reached trainer after same-phase' in printed[-1]['error']


@pytest.mark.parametrize('change', ['new_phase', 'phase_relative_signal_gate', 'seed',
                                   'checkpoint', 'config', 'panel', 'wm', 'best'])
def test_continuation_rejects_changed_phase_inputs(continuation, change):
    raw, args, _ = continuation
    if change == 'new_phase':
        args.new_phase = True
    elif change == 'phase_relative_signal_gate':
        args.phase_relative_signal_gate = False
    elif change == 'seed':
        args.seed += 1
    elif change == 'checkpoint':
        torch.save({'update_count': 188}, args.resume)
    elif change == 'config':
        raw['trainer']['actor_lr'] *= 2
    elif change in ('panel', 'wm'):
        Path(args.gate_panel if change == 'panel' else args.wm_ckpt[0]).write_text('changed')
    else:
        torch.save({'update_count': 180}, Path(args.continue_run) / 'policy_native_best.pt')
    with pytest.raises(ValueError, match='continuation'):
        recovery.load_continuation(raw, args)


def test_gain_only_continuation_restores_curriculum_without_cqr_refusal(long_resume):
    trainer, boundary, receipt, output, payload, _ = long_resume
    receipt['prior']['looplong_long_credit'] = None
    proof = recovery.restore_looplong(trainer, boundary, receipt, output)
    assert proof['horizon'] == payload['looplong']['horizon']
    assert proof['credit_refusal_sha256'] is None
    assert trainer.horizon_curriculum.stage(514) == 2
    assert torch.equal(torch.get_rng_state(), payload['rng']['torch'])


def test_ordinary_resume_preserves_trained_head_adam_and_longgrad_reset_flag(tmp_path, monkeypatch):
    from tests.unit.test_keyerr import make_trainer
    trainer = make_trainer(monkeypatch, exits=False)
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    driver = tmp_path / 'driver.json'
    driver.write_text(json.dumps({'schema': 'execparity_learned_idm/1',
        'fit_controller': 'direct', 'evaluation_data_used': False,
        'fit_worlds': ['synthetic-train'], 'score_worlds': ['synthetic-check'],
        'parameters': {'accel': 1.1, 'decel': 2., 'time_headway': 1.2, 'min_gap': 2., 'v0': 28.}}))
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(driver))
    trainer.config = replace(trainer.config, long_grad_balance=True, long_grad_reset_actor_adam=True)
    sum(p.square().sum() for p in trainer.actor_parameters).backward()
    trainer.actor_optimizer.step()
    trainer.update_count = 189
    checkpoint = tmp_path / 'policy_000189.pt'
    trainer.save_checkpoint(checkpoint)
    restored = make_trainer(monkeypatch, exits=False)
    monkeypatch.setenv('C4_EXECUTION_DRIVER', str(driver))
    restored.load_checkpoint(checkpoint)
    assert restored.update_count == 189
    assert restored.config.long_grad_balance and restored.config.long_grad_reset_actor_adam
    assert restored.actor_optimizer.state
    assert 'actor_adam_reset' not in restored.resume_config_receipt
    assert_tree_close(trainer.policy.state_dict(), restored.policy.state_dict(), 0, 0)
    assert_tree_close(trainer.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict(), 0, 0)
