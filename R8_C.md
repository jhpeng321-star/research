# R8_C — config plumbing for merge-core v2

## Files changed
- `src/wms/config/train.py` — added `version`, `z_tf`, `z_on`, `sigma_reg_weight`,
  `sigma_fit` to `MergeCoreLossConfig` (defaults 1 / 1.0 / 1.0 / 0.0 / 0.0) with validation
  (version in {1, 2}; z_tf / z_on finite > 0; weights >= 0; sigma_fit finite >= 0).
- `src/wms/train/settings.py` — flat `merge_core_version`, `merge_core_z_tf`,
  `merge_core_z_on`, `merge_core_sigma_reg_weight`, `merge_core_sigma_fit`, same validation.
- `src/wms/config/legacy_flat.py` — flat <-> dotted mappings and `OMIT_WHEN_DEFAULT` entries,
  so every existing config keeps its previous legacy flat form (round 7b SHA-256 unchanged).
- `src/wms/train/losses/compose.py` — passes the five fields to `MergeCoreConfig(...)`.
- `configs/c4_wm_v10round8.toml` — round-8 preset exactly per `R8_INTERFACE.md`.
- `tests/test_merge_core_v2_config.py` — plumbing, round-7b flat regression, freeze /
  residual-zero, lr-scale longest-prefix and full-model unmatched-prefix checks.

## Notes
- `merge_core.temperature_raw` and the `residual_head` / `residual_scale_raw` are frozen by the
  round-8 preset; the residual head is zero-initialised, so its margin contribution is exactly
  `0.0` at init (verified bitwise in the test), no model switch is needed.
- On this branch `MergeCoreConfig` in `merge_core.py` still lacks agent A's R8 kwargs, so the
  composer-kwargs test is `xfail(strict=False)`; the settings-level plumbing is fully tested.
  The coordinator merge with agent A activates that test.
- Resolved optimizer report for the round-8 config: 15 groups, no unmatched freeze or
  lr_scale prefixes; `merge_core.{s0,t,b,tau}_raw` at 100x, `merge_core` (alpha_net),
  `alpha_bias`, `alpha_log_sigma_raw`, `yield_scale_raw` at 20x, 6 frozen tensors / 323
  parameters.
