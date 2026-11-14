# C4 Dreamer data loop — notes (branch c4/loop, 2026-10-12)

Hook that feeds SUMO episodes collected with the learned policy back into WM-S training
(docs/WM_C4_DESIGN.md 2.4/2.7): collect -> convert -> refresh-fine-tune the C4 WM.

## 1. CLI

```
# 1. collect (SUMO; exposed dev seeds 4107-4148 only, see SUMO_NOTES.md section 4)
python -m traffic_shepherd.c4.collect --policy <ckpt|random|native> --penetrations ... \
    --demands ... --seeds ... --warmup 300 --horizon 600 --workers N --out runs/active/<ID>/collect
# 2. convert (no SUMO, ~1.5 s per 480 s episode)
python -m wms.data.c4_episodes --episodes runs/active/<ID>/collect [more dirs...] \
    --out runs/active/<ID>/c4_episodes [--val-frac 0.2] [--split-seed 0] [--allow-designed]
# 3. refresh the C4 WM on original data + collected episodes
WMS_DATA_C4_EPISODES=runs/active/<ID>/c4_episodes WMS_DATA_CKPT_C4_WM=<C4 WM wm_s_best.pt> \
    python -m wms train --config configs/c4_wm_refresh.toml --mode run   # default mode = preflight
```

Converter (`src/wms/data/c4_episodes.py`): takes every `manifest.status == complete` episode
under the given roots (`.incomplete-N` skipped), writes one mmap-able dir per episode
`episodes/<world>__<arm>/` (pack-format `veh_kin/veh_mask/target_mask/counter/vid/is_cav`
+ the per-CAV action timeline `controller/issued_acceleration_mps2/issued_target_speed_mps`)
and `C4_EPISODES_MANIFEST.json` (`wms_c4_episode_source/1`, split TRAIN, source sha256 per
episode). Write-once (refuses a non-empty output, duplicate world/arm, restricted tokens).
* World ids carry the pack pattern `w<seed>m<demand>p<pen %:02d>t<warmup s>`
  (`w4107m2700p20t300`; parsed by `wms.eval.states.world_meta`); arm = collector policy tag.
* TRAIN/val split by episode, **grouped by world** (`ceil(frac * n_worlds)` smallest
  `sha256(seed:world)` are val; none with < 2 worlds), so one demand/driver draw never
  straddles the split.
* Designed controllers (damping / PI in the action timeline) are refused unless
  `--allow-designed` (owner: WM training sees native + random/learned direct actions only).

Loader (`build_c4_datasets`): windows cut at load time with the trainer's horizon
(max curriculum unroll), t0 on the 1 s control grid every `window_stride_s` from each
episode's warmup end; records are key-for-key the pack window loader's
(`_PackWindowLoader`): alive-since-t0 presence (`target_mask` cumprod: mid-window entrants
never added, reused slots never revive — the pack entrant handling), junction steps excluded
from the loss, CAV teacher, short `hist` + 60 s `driver_hist` via `prior_history`,
`entry_feat` via `entry_features` when `data.entries_v2` (counter rebuilt from first
appearances, `DepartureLog.counter_at` binning), and the per-CAV `action_timeline` on the t0
slot columns (native once the t0 vehicle has left). Per-window switch/direct events feed the
mix's `event_oversample`.

## 2. Config knobs

```toml
[data.c4_episodes]        # src/wms/config/data.py C4EpisodesConfig; off when path unset
path = "c4_episodes"      # logical name (WMS_DATA_C4_EPISODES / configs/data_registry.toml) or abs path
mix_ratio = 0.3           # sampling share of collected windows; other sources keep relative shares
window_stride_s = 10.0    # whole seconds
start_s = 300             # optional; default = each episode's warmup_s
use_val = true            # collected val worlds join the monitoring set
```

`assemble.build_datasets` mixes the source last (`add_c4_source`): with the C4 mix
original 0.49 / switching direct-mix 0.21 / collected 0.30; it checks K against the packs,
applies the switching zero-shot held-out-world guard to the collected worlds, composes with
jam over-sampling, and records `receipt["c4_episodes"]` + per-stage sampling.
`declared_data_names` lists it (preflight existence check).
`configs/c4_wm_refresh.toml` extends `c4_wm.toml`: `init_from = "ckpt_c4_wm"`
(`WMS_DATA_CKPT_C4_WM`, fails closed when unset), `[data.c4_episodes]` as above,
`stage_steps = [300, 300, 300]`, `lr_per_stage = 1e-4` each, `max_hours = 1.5`,
out `runs/active/c4_wm_refresh`. The 60 s stage needs >= 60 s of episode after warmup.

## 3. Smoke evidence (CPU, this laptop; raw data under runs/active/C4-LOOP-SMOKE-20261012/, git-ignored)

* Collect: `--policy random --penetrations 0.2 --demands 2700 --seeds 4107,4108 --warmup 300
  --horizon 180 --workers 2`: 2/2 complete, 27 s wall, 0 collisions/teleports. Control:
  same with `--policy native` (16 s). SUMO total < 1 min.
* Convert (`--val-frac 0.5`): w4107 -> train, w4108 -> val; 4060 / 5097 direct CAV steps;
  13 + 13 windows of 60 s; 7.5 MB per 480 s episode (f32 npy).
* Refresh dry run (`python tools/c4_loop_smoke.py --c4-source ... --out ...`, config
  c4_wm_refresh, R5b best as the `WMS_DATA_CKPT_C4_WM` stand-in: "loaded 105 tensors,
  shape-mismatch 0"): mix original 583 (0.7) + c4_episodes 13 (0.3), val original 99 +
  c4 13; batch of 4 = 2 pack windows (native, 0 direct steps) + 2 collected windows
  (20/21 CAVs at t0, 1078/1201 direct CAV control steps, |issued a| <= 2.47, direct only on
  CAV slots, not synthetic); 2 updates finite (loss 7.90 -> 7.84); 10 s / 30 s fwd+bwd
  10.5 s / 26.8 s. Summary: `_out/c4_loop/SMOKE_SUMMARY_20261012.json`.
  Ran WITHOUT the switching source: its guard opens the zero-shot held-out-world list,
  which this task's rules forbid opening; the 3-source mix is unit-tested instead.
* Sanity (forward only, R5b weights, 10 s, `_out/c4_loop/PER_SOURCE_LOSS_20261012.log`):
  pack windows m2600-2700 p20-40 loss 7.83 (pos 7.20, lane 0.14); collected NATIVE
  windows 4.40 (pos 3.96, lane 0.12) — conversion consistent with the packs; collected
  RANDOM-policy windows 18.5 (pos 16.5, lane 0.84): the WM is clearly worse on traffic
  driven by random direct CAV actions — the gap the refresh loop is meant to close.
* Tests: `tests/unit/test_c4_loop.py` 9 passed (synthetic: world ids, convert/split/
  counter, designed-controller + restricted-name refusal, pack-contract windows incl.
  leaver/entrant/junction/timeline masking + entries_v2, cross-check vs
  `recorder.episode_windows`, 2- and 3-source mixing + collate, refresh config, assemble
  wiring). `tests/unit/test_c4_*.py tests/unit/test_import_graph.py tests/unit/{cli,config,data}`:
  583 passed, 1 failed (pre-existing `test_drive_registry` registry-text mismatch). ruff clean.

## 4. Limitations / open

* No real refresh run yet (no GPU in this task); no checkpoint named `ckpt_c4_wm` exists in
  the registry — set `WMS_DATA_CKPT_C4_WM` to the C4 WM best checkpoint per iteration.
* `python -m wms train --config configs/c4_wm_refresh.toml --mode preflight` not run here
  (it stats the switching held-out list); the smoke used assemble + dry run directly.
* The counter is rebuilt from first appearance on the network (= insertion; the packs use
  SUMO `departed_ids`): identical unless the K = 192 slot pool overflows.
* `vehicle_ids` are source-local (`(episode + 1) * 1e6 + vid`), not pack identity sidecar ids
  (the trainer does not use them). `controller_id` = episode controller (`direct`).
* Converted arrays are f32 (~15 MB per 900 s episode); a 1000-episode loop is ~15 GB —
  switch to f16 like the packs if disk matters.
* Mixing share is static per run; per-iteration weighting/recency of older loop data is a
  config choice (pass several collect dirs to one convert, or convert per iteration).
* Collected windows start at the warmup end (all-native warmup); the policy's effect on the
  60 s driver history only appears for t0 > warmup + 60 s.
