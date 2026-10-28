"""C4 preparation refuses missing scientific bindings and bounds scratch updates."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/c4prep'))
import common
import launch


def test_unresolved_config_cannot_dispatch(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'ready': False, 'blockers': ['WM3 gate pending']}))
    def forbidden(*args, **kwargs):
        raise AssertionError('must refuse before spawning')
    monkeypatch.setattr(launch.subprocess, 'Popen', forbidden)
    with pytest.raises(ValueError, match='WM3 gate pending'):
        launch.run(path, dry=True)
    with pytest.raises(ValueError, match='WM3 gate pending'):
        launch.run(path, dry=False)
    with pytest.raises(ValueError, match='WM3 gate pending'):
        launch.start(path)


@pytest.mark.parametrize('token', ['sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vtype'])
def test_prep_path_guard(token, tmp_path):
    with pytest.raises(ValueError, match='prohibited'):
        common.safe(tmp_path / token / 'data.json')


def test_configs_share_flags_except_entrant_and_its_artifact():
    from wms.control.train_cli import _read_config
    from wms.control.dreamer import DreamerConfig
    from wms.control.counterfactual import CounterfactualConfig
    a = _read_config(ROOT / 'configs/c4_policy14_c4prep_off.toml')
    b = _read_config(ROOT / 'configs/c4_policy14_c4prep_on.toml')
    assert not a['trainer']['entrant_control'] and b['trainer']['entrant_control']
    for key in ('entrant_control', 'native_anchor_calibration'):
        a['trainer'].pop(key)
        b['trainer'].pop(key)
    assert a == b
    # Supplying an unresolved artifact remains valid configuration, never readiness.
    cfg = DreamerConfig(**a['trainer'], native_anchor_calibration='UNRESOLVED')
    CounterfactualConfig(**a['counterfactual'])
    assert cfg.future_samples == 8 and cfg.pessimistic_credit and cfg.native_anchor
    assert a['run']['gpu_memory_fraction'] == .45
    assert cfg.imagination_microbatch == 1 and cfg.imagination_checkpoint_segment == 8
    assert cfg.imagination_offload_futures and cfg.imagination_offload_saved_tensors
    assert not a['uncertainty']['enabled'] and not a['probe']['enabled']


@pytest.mark.parametrize('bounded,expected', [(True, 2), (False, 3)])
def test_cli_scratch_bound_and_normal_path(tmp_path, monkeypatch, bounded, expected):
    import wms.control.policy14_cli as cli
    import wms.control.paircrit as paircrit
    raw = cli._read_config(ROOT / 'configs/c4_policy14.toml')
    raw['policy14'].update(fd_gate=False, free_road_gate=False, independent_rescore=False,
                           critic_warmup_updates=0)
    for section in ('paircrit', 'longcritic', 'real_share', 'breakdown'):
        raw[section] = {'enabled': False}
    for section in ('uncertainty', 'exploration', 'horizon'):
        raw.pop(section, None)
    raw['trainer']['real_return_mix'] = 0.
    raw['trainer']['imagine_entries'] = False
    saved = []
    class Trainer:
        def __init__(self, *args, config, **kwargs):
            self.config = config
            self.update_count = 0
            self.paircrit = self.longcritic = None
        def update(self):
            self.update_count += 1
            return dict(update=self.update_count, actor_loss=1., critic_loss=1.,
                        actor_grad_norm=1., critic_grad_norm=1.)
        def save_checkpoint(self, path):
            saved.append(str(path))
    monkeypatch.setattr(cli, '_read_config', lambda path: raw)
    monkeypatch.setattr(cli, 'load_members', lambda *args: ([SimpleNamespace()],
                        [{'checkpoint_sha256': '0' * 64, 'config_sha256': '0' * 64}]))
    monkeypatch.setattr(cli, 'Policy14Trainer', Trainer)
    monkeypatch.setattr(cli, 'NPZRealReturnProvider', lambda *args, **kwargs:
                        SimpleNamespace(path='fake.npz', provenance={'source_kind': 'test'}, indices=np.arange(1)))
    monkeypatch.setattr(cli, 'SmokeSampler', lambda model: SimpleNamespace(sample=lambda *a, **kw: None))
    monkeypatch.setattr(cli, 'independent_score', lambda *args: {'J': 1.})
    monkeypatch.setattr(cli, 'warm_start_real_critic', lambda *args, **kw: {})
    monkeypatch.setattr(paircrit, 'install', lambda *args: None)
    output = tmp_path / 'out'
    args = ['--smoke', '--max-updates', '3', '--wm-ckpt', 'fake.pt', '--wm-config', 'fake.toml',
            '--real-returns', 'fake.npz', '--output-dir', str(output), '--compact']
    if bounded:
        args += ['--prep-dry-run-updates', '2']
    assert cli.main(args) == 0
    summary = json.loads((output / 'RUN_SUMMARY.json').read_text())
    assert len(summary['updates']) == expected
    if bounded:
        assert summary['status'] == 'completed_dry_run'
        assert summary['deployment_authorized'] is False
        assert saved == []
        assert not (output / 'native_gate_result.json').exists()
    else:
        assert summary['status'] == 'completed'
        assert len(saved) == 1
