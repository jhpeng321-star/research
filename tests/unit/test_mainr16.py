"""Round-16 continuation must retain learned adaptation and registered frozen core."""
import copy
import importlib.util
from pathlib import Path

import pytest

from wms.config.load import load_train_config


def test_round16_loop_ft_retains_selected_model_and_physics():
    overrides = ['model.init_from="unused.pt"']
    selected = load_train_config("configs/c4_wm_v10round16_stage_b.toml")
    loop = load_train_config("configs/c4_wm_loop14_r16_ft.toml", overrides)
    assert loop.model.ramp_speed_adapt_learnable
    assert loop.physics == selected.physics
    assert "ramp_speed_adapt" in loop.optim.freeze_prefixes
    assert "merge_yield" not in loop.optim.freeze_prefixes
    assert sum(loop.curriculum.stage_steps) == 40
    assert not loop.model.follower_courtesy_init_zero
    original = load_train_config("configs/c4_wm_loop14_ft.toml", overrides).to_legacy_flat()
    adapted = loop.to_legacy_flat()
    for key in set(original) | set(adapted):
        if key not in {"name", "out_dir", "ramp_speed_adapt_learnable", "freeze_prefixes"}:
            assert adapted.get(key) == original.get(key), key


def test_restart_preserves_charges_and_attempt_paths_without_adopting_old_policy():
    path = Path(__file__).resolve().parents[2] / 'scripts/c4/mainr16_restart.py'
    spec = importlib.util.spec_from_file_location('mainr16_restart', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    phases = ('policy', 'collect', 'convert', 'wm_ft', 'check')
    state = {'cycles': [{'cycle': 0, 'phases': {p: {'status': s, 'attempt': a}
             for p, s, a in zip(phases, ('done', 'failed', 'pending', 'pending', 'pending'), (1, 2, 0, 0, 0))}}],
             'policy': {'checkpoint': 'old/policy.pt'}, 'sim_s_charged': 58266,
             'seeds_used': list(range(968103, 968129)), 'stopped': None}
    before = copy.deepcopy(state)
    new = module.restarted_state(state, registration='reg', registration_sha256='regsha',
        base={'sha256': 'r16sha'}, initial_returns='native.npz', phases=phases)
    assert state == before
    assert new['policy'] is None and new['superseded_policies'] == [state['policy']]
    assert new['cycles'][0]['phases']['policy'] == {'status': 'pending', 'attempt': 1}
    assert new['sim_s_charged'] == 58266 and new['seeds_used'] == state['seeds_used']
    state['stopped'] = 'scientific stop'
    with pytest.raises(ValueError):
        module.restarted_state(state, registration='reg', registration_sha256='sha',
            base={}, initial_returns='native.npz', phases=phases)


def test_horizon_loader_accepts_resolved_config_without_changing_model(tmp_path):
    import json
    import torch
    from tools.r10_attribution import load_model, build_model, build_physics
    from wms.config.schema import to_mapping

    preset = 'configs/c4_wm_v10round16_stage_b.toml'
    cfg = load_train_config(preset, ['runtime.device=cpu', 'data.entries_v2=false'])
    model = build_model(cfg, build_physics(cfg))
    checkpoint = tmp_path / 'wm.pt'
    torch.save({'model': model.state_dict()}, checkpoint)
    resolved = tmp_path / 'config.resolved.json'
    resolved.write_text(json.dumps({'config': to_mapping(cfg)}))
    a, _settings, loaded, report = load_model(str(resolved), str(checkpoint))
    assert not report['new'] and not report['unused']
    assert a.model.ramp_speed_adapt_learnable and a.physics == cfg.physics
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
