# WM-S-R4-20261011 implementation record

## Changes

- Added `cumsum` (the unchanged default) and learned `saturating` residual gates.
  The saturating gate is `sigmoid(a) * (1 - exp(-k / softplus(b)))`, initialized
  to `g_max=0.30`, `tau=50` control steps. Cumsum checkpoints keep their original
  `gate_increments` state-dict layout; checkpoint loading infers old checkpoints
  as `cumsum` when the selector is absent.
- Added R4's fully annotated config. It matches the R3 receipt's 911-window
  pack mix, C1/R3 curriculum, 60 s driver history and R3 guards; disables all
  three braking losses and merge calibration; and selects the saturating gate.
- LANECAL's deterministic six-start-draw audit for the exact R3/R4 LPWO list
  gives `92,873 / 74,700,490 = 0.001243271630` true lane changes per valid,
  alive HDV target step (0.124327%). It forward-fills the prior valid lane over
  junction `-1` observations, then counts only actual lane transitions. The
  target excludes junction toggles and is recorded in `configs/r4.toml`.
- Saturating-gate training records now log `g(k)` at steps 10, 20, 60, 120 and
  240. Default cumsum training records and numerical outputs remain unchanged.

## Files

- `src/wms/model/gate.py`
- `src/wms/model/spec.py`
- `src/wms/model/wms.py`
- `src/wms/config/train.py`
- `src/wms/config/legacy_flat.py`
- `src/wms/train/settings.py`
- `src/wms/train/unroll.py`
- `tests/unit/test_model_gate.py`
- `configs/r4.toml`
- `runs/tmp_preflight/r4_smoke/wm_s.pt` and `train_log.jsonl` (local smoke artifacts)
- `runs/active/WM-S-R4-20261011/IMPLEMENTATION.md`

## Verification

- `python -m wms preflight --config configs/r4.toml`: passed; resolved the
  registered paths and confirmed 10/30/60/120 s with 2500/2500/1875/1250 steps.
- Three-update CPU smoke using a small synthetic batch and R4's saturating
  model: passed. `g_max` moved from 0.30000001 to 0.30009258 and `tau_steps`
  from 50.000000 to 49.999558. The saved checkpoint reloaded with matching
  learned parameters and identical gate outputs. `train_log.jsonl` contains
  the requested `gate_g_by_k` values.
- Focused gate/component/checkpoint tests: 11 passed.
- `pytest tests/unit tests/equivalence tests/golden -q -p no:cacheprovider`
  excluding `tests/unit/cli/test_launch_script.py`: 680 passed, 2 skipped.
  The skipped tests are the opt-in slow full eval and the Python-version-specific
  TOML test. The excluded Windows launch-script tests cannot start their mocked
  interpreter because `shutil.which("echo")` returns `None` on this host; three
  tests fail at process creation with a non-string `PY` environment value. The
  remaining launch-script LF test passes when run independently.
- `ruff check src tests`: passed.
- `git diff --check`: passed.

## gpuA launch command (record only; not launched)

From the staged R4 repository root on gpuA:

```bash
PYTHONPATH=src python -m wms train --config configs/r4.toml --mode run --out /root/autodl-tmp/wms_train_r4/r4_main_a1
```
