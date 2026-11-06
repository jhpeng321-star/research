# V10 HORIZON — making long-horizon accuracy learnable in WM-S

Branch `c4/v10horizon` (from `c4/v10entzone` `b75475ad`), 2026-10-19.
Two opt-in mechanisms, **both default OFF = the fixed 10/30/60/120 s recipe is bit-identical**:

1. **Plateau-advancing horizon curriculum** — a stage trains until its own (shorter) validation
   error stops improving, then the curriculum advances to the next horizon.
2. **Prioritised mixed replay** — loop data + original native/random TRAIN sources keep their
   mix ratios; merge-zone, CAV-follower and high-error windows are drawn more often.

No SUMO, no GPU, no evaluation seeds, no sealed/holdout data were used.  Driver settings stay
learned from prediction error only; no pushforward / noise-injection / self-recovery trick was
added.

## Why (context)

The pulse/HDV diagnostics show braking response is accurate to ~10 s but collapses by 30 s
(HDV |ds| 4/15/31/79 m at 10/30/60/120 s), and 91 % of native excess delay accrues after 120 s.
The existing trainer already had 10/30/60/120 s stages and chunked truncated BPTT, but every
stage was a fixed step budget, so the long stages were under-trained and the data mix was static.

## Design

### 1. Plateau-advancing curriculum (`src/wms/train/curriculum.py`, `trainer.py`)

`PlateauRule.observe(step, value)` is fed the stage-horizon validation `mean_position`
(lower is better) at the configured `val_every` cadence:

* an improvement of more than `plateau_min_delta` *relative* to the best resets the stale
  counter; otherwise the counter grows;
* the stage advances when `stale >= plateau_patience` and at least `plateau_min_steps`
  observations were seen (`reason = "plateau"`), or when `plateau_max_steps` observations
  were seen (`reason = "max_steps"`, 0 = no cap);
* `stage_steps` remains the hard cap, so the run cannot exceed the planned budget;
* the decision (`history`, best, stale, advanced, reason) is written to `val_log.jsonl`
  (`plateau`, `curriculum_advance`, `curriculum_advance_reason`) and to the stage record.

Validation is forward-only and uses the existing deterministic evaluation path.  The rule only
reads validation values; it never changes loss, gradients or RNG.

### 2. Prioritised mixed replay (`src/wms/data/replay_flags.py`, `src/wms/train/replay.py`)

`PriorityReplayDataset` wraps the **already assembled** mixed TRAIN dataset (loop episodes +
`c4_sources` native/genrand; the source ratios are unchanged) and supplies the existing
`stage_weights(stage_s)` sampler hook.  Per window:

```
weight = base_stage_weight(stage) x merge_zone_mult x cav_follower_mult x high_error_mult
```

* **merge-zone** (V10ROUND1): at some step a present recorded HDV in the ramp/merge lane
  (lane index 2 of the `states[..., 1:4]` one-hot) is within `data.replay.merge_zone_end_m`
  (default 40 m) of the acceleration-lane end `physics.merge_zone_s1_m` (75.44 m); the old
  rule flagged any lane-2 HDV upstream, i.e. ~every window;
* **CAV-follower**: at some step a present HDV is the nearest predecessor of a present CAV in
  the same lane (`states[..., 0]` ordering);
* **high-error**: the trainer forward-probes an evenly spaced subset of
  `replay_probe_windows` windows at the stage horizon
  (`UnrollRunner.window_errors`, per-window positional Huber), updates an EMA of the window
  error, and multiplies by `clip((error / mean_seen_error) ** alpha, 1, error_max_boost)`.
  Probes run at every stage boundary and every `replay_refresh_every` optimizer steps; the
  loader is rebuilt so the new priorities take effect.

The two static flags are computed once (mmap-window scan, like `wms.data.jam`).  All
multipliers default to 1, and the wrapper is only built when `data.replay.enabled = true`.

### 3. Truncated BPTT, checkpointing, memory

The trainer already supports chunked backward (`curriculum.truncation_steps_per_stage`).
V10 adds `optim.batch_size_per_stage` so the long stages use smaller batches, and the recipe
enables `model.checkpoint_step` / `model.checkpoint_residual` (re-materialise step / residual
activations in backward).  Credit profiling (`loss.loss_share_logging`) stays **off** — the
earlier gpuR OOMs came from it.

Analytic activation upper bound (no checkpointing), production WM dimensions
`K=192 slots, d_model=128, n_layers=3, n_heads=4, ffn=512, fp32`:
`K*(3d + ffn) + n_heads*K^2 ~= 3.8 MB` per (window, rollout step).

| stage | chunk steps | batch | upper bound (no ckpt) | with step ckpt (approx) |
|---|---|---|---|---|
| 10 s | 20 | 8 | 0.61 GB | ~0.05–0.10 GB + state |
| 30 s | 40 | 6 | 0.92 GB | ~0.05–0.10 GB + state |
| 60 s | 60 | 4 | 0.92 GB | ~0.05–0.10 GB + state |
| 120 s | 80 | 2 | 0.61 GB | ~0.05–0.10 GB + state |

Add the WM parameters/optimizer state (well under 1 GB) and the CUDA context; this leaves ample
headroom on a 24–48 GB gpuR card.  The table is analytic (no GPU on this host) and deliberately
conservative.

## Config keys (all default OFF / 1.0)

| key | default | meaning |
|---|---|---|
| `curriculum.plateau_advance` | `false` | enable plateau-driven stage advancement |
| `curriculum.plateau_patience` | `3` | consecutive non-improving validations before advancing |
| `curriculum.plateau_min_delta` | `0.01` | relative improvement that resets patience |
| `curriculum.plateau_min_steps` | `2` | minimum validations before a plateau advance |
| `curriculum.plateau_max_steps` | `0` | observation cap (0 = none; `stage_steps` is the cap) |
| `optim.batch_size_per_stage` | unset | one batch size per curriculum stage |
| `model.checkpoint_step` | `false` | per-step activation checkpointing |
| `model.checkpoint_residual` | `false` | residual-net activation checkpointing |
| `data.replay.enabled` | `false` | build the priority-replay wrapper |
| `data.replay.merge_oversample` | `1.0` | multiplier for merge-zone windows |
| `data.replay.merge_zone_end_m` | `40.0` | lane-2 HDV distance-to-end band (m) of a merge-zone window |
| `data.replay.cav_follower_oversample` | `1.0` | multiplier for CAV-follower windows |
| `data.replay.error_max_boost` | `8.0` | cap of the high-error multiplier |
| `data.replay.error_alpha` | `1.0` | exponent of the error ratio |
| `data.replay.error_decay` | `0.9` | EMA decay of the per-window error |
| `data.replay.probe_windows` | `0` | windows forward-probed per refresh (0 = static only) |
| `data.replay.refresh_every` | `0` | optimizer steps between online refreshes (0 = stage boundary) |

Recipes: `configs/c4_wm_v10horizon.toml` (full) and
`configs/c4_wm_v10horizon_smoke.toml` (tiny budgets).  The full recipe keeps
`10 -> 30 -> 60 -> 120 s`, `batch_size_per_stage = [8, 6, 4, 2]`,
`truncation_steps_per_stage = [20, 40, 60, 80]`, plateau on, replay on, checkpointing on.

### Default-off = bit-identical

Every new config key equals its default in the legacy recipe and is omitted from
`to_legacy_flat()` when default (`OMIT_WHEN_DEFAULT`).  In code, the stage loop, loader and
records are unchanged when `plateau_advance` / `replay_enabled` are false; `_window_probe`
adds a record key only when explicitly requested.  The existing train-layer equivalence and
golden tests were re-run (see Tests).

## Smoke evidence (CPU, synthetic TRAIN-val windows)

Runner: `python tools/v10horizon_smoke.py` (4 deterministic windows, 240 steps each, batch 2,
CPU, plateau `patience=1 min_steps=1 min_delta=1.0`, `val_every=3`, replay on with
`probe_windows=4`).  Output: `runs/active/c4_wm_v10horizon_smoke/SMOKE_EVIDENCE.json`
(committed copy: `docs/v10horizon_smoke_evidence.json`).

The curriculum advanced at **every** stage via `reason = "plateau"` (10 → 30 → 60 → 120 s),
and the multi-step rollout loss trend is negative at every horizon:

| horizon | steps | first loss | last loss | min loss | slope / step | advanced |
|---|---|---|---|---|---|---|
| 10 s | 6 | 142.47 | 141.09 | 139.17 | −0.49 | plateau |
| 30 s | 6 | 710.83 | 714.97 | 709.97 | −0.07 | plateau |
| 60 s | 6 | 1533.30 | 1531.75 | 1525.80 | −1.67 | plateau |
| 120 s | 6 | 2582.26 | 2583.76 | 2576.27 | −0.33 | plateau |

(The 30 s last step is 0.6 % above the first — batch-of-2 noise; its best value is 0.7 %
below the first and the fitted slope is negative.  The per-step trace is in
`train_log.jsonl`.)

Replay scan on the same windows: `n_merge_zone = 4/4`, `n_cav_follower = 4/4`;
the per-stage high-error snapshots (mean/max error and boost) are in `val_log.jsonl`
(`replay` field), and the EMA boost updates at every stage.

## Tests

* `pytest tests/unit/train/test_v10horizon.py` — 15 passed (plateau rule and trainer
  advancement, batch-size plumbing, replay flags/weights, per-window probe, trainer replay
  refresh, config files).
* `pytest tests/unit/train tests/unit/data tests/unit/config tests/unit/test_sched.py
  tests/unit/test_pairloss.py tests/unit/test_v9entry.py tests/unit/test_v8wm*.py` —
  355 passed, 1 failed.  The failure is pre-existing and unrelated:
  `tests/unit/data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`
  (the committed `tools/data_registry.json` description differs from the offline generator;
  this branch does not touch it).
* `pytest tests/equivalence/test_train_equivalence.py tests/golden/test_golden_new.py` —
  15 passed, 10 skipped (local packs absent), 1 failed.  The failure is pre-existing:
  `test_fit_matches_legacy_fit` compares log keys with the archived legacy trainer, which does
  not emit `grad_norm` / `grad_clipped` / `lr` added by round-5b logging on this branch (present
  at `b75475ad`, untouched here).
* `ruff check` on every changed/new file — clean.

## Files

* `src/wms/train/curriculum.py` — `PlateauRule`
* `src/wms/train/trainer.py` — plateau stage loop, per-stage batch size, replay probe/refresh
* `src/wms/train/unroll.py` — forward-only `window_errors` probe
* `src/wms/train/replay.py` — `PriorityReplayDataset`
* `src/wms/train/settings.py` — new flat settings + validation
* `src/wms/data/replay_flags.py` — merge-zone / CAV-follower window scan
* `src/wms/config/{train,data,legacy_flat}.py` — config plumbing
* `src/wms/cli/assemble.py` — build the replay wrapper
* `configs/c4_wm_v10horizon*.toml`, `tests/unit/train/test_v10horizon.py`,
  `tools/v10horizon_smoke.py`

## Not done / next

* No GPU/SUMO run on real converted episodes (this host has no `data_cache`); the full recipe
  is ready for the coordinator to launch on gpuR with `model.init_from=<v5m checkpoint>`.
* The real-run acceptance should read `HORIZON.md`-style loss-by-horizon from `train_log.jsonl`
  and check that 30/60/120 s validation keeps improving across plateau advances.
