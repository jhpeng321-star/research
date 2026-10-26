"""Round-16 continuation must retain learned adaptation and registered frozen core."""
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
