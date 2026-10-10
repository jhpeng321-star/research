# RUN1_DIAG - two WM-S run-1 weaknesses (TRAIN, 20 states, offline; no SUMO/GPU)
Status: complete. Checkpoint wm_s.pt sha256 97d8f998...; TRAIN only, native arm. Scripts `_out/run1_diag.py` -> `_out/run1_diag.json`, `_out/run1_mergecal_cmp.py`; merge_cal eval `_out/wms_eval2/wms_run1_mergecal/`.

## (1) Matched fraction: entries + a permanent junction-transit drop
Pooled over 20 states, recorded-present vehicles at the horizon (reproduces RUN1_EVAL_FAIR 0.59/0.75 @30 s, 0.38/0.53 @60 s):
- @30 s wms matched 1180/1990 = 0.593; keep_speed 1500/1990 = 0.754. wms-unmatched = entrants 490 (24.6%; 435 HDV+55 CAV) + junction-transit drops 318 (16.0%; 268 HDV+50 CAV) + t0-on-junction 2.
- @60 s wms 791/2047 = 0.386; keep_speed 1066/2047 = 0.521. Unmatched = entrants 981 (47.9%; 877+104) + junction-transit 274 (13.4%; 245+29) + t0-on-junction 1.
- keep_speed drops entrants only (490/981), so the entire wms-specific deficit (320 @30 s, 275 @60 s) is the junction-transit permanent drop: the teacher presence mask ANDs every step and marks a slot absent whenever its recorded row is a junction internal lane (`:`-prefixed, lane<0); once dropped it never returns. Zero drops were "missing row"; all 592 are junction transits.
- Wrong lane at the merge end is negligible: lane disagreement among matched 68/1180 (5.8%) @30 s and 90/791 (11.4%) @60 s, of which only 10 and 2 are merge-end lane-1/2 swaps (<=0.8%). Identity matching by id is exact (no slot mismatch).
- "Premature exits" are this same false drop: those vehicles are still present in the recorded horizon set.
Fixes: (a) never permanently drop a slot for one junction step - carry the last state through the junction (or re-acquire identity) instead of cumulative AND; (b) entrants need an entry model to ever be matchable.

## (2) Merge congestion: not the merge flag - the eval has no arrivals
- Run-1 training flags (`physics.json` / TRAIN_RECEIPT: open_road, sumo_fidelity, sumo_idm) have NO merge_cal, and `PhysicsConfig` had no such field. Calibrated gap acceptance existed only in numpy `rollout2/merge_cal.py` (branch wms/merge, f131347/6d0edee) and was exported as `_out/merge_params.json` "for training run 2"; it was never ported to the torch base.
- I ported it into `wms/model.py` (flag `merge_cal`, default OFF, byte-identical when off; test added) and re-ran the run-1 checkpoint on the same 20 states. Merge section, mean over bins/states: queue 0.55->0.52, stopped 0.13->0.15, delay 19.5->19.9 (recorded native 8.51 / 1.31 / 197.4). It does NOT restore the queue.
- Reason: the eval is t0-only (no arrival model) and recorded merge congestion is entry-driven. Merge-section vehicles at 270-300 s: 101/10480 are t0 (99.0% post-t0 entrants); stopped vehicle-ticks there 1528, only 24 t0. t0 merge population falls 9280 (0-30 s) -> 101 (270-300 s). Predicted queue drains to ~0 by bin 5 (~150 s) and stays 0; recorded stays 5-14 m across all bins. No base-physics flag can replenish traffic.
Fixes for run 2 / C1: (1) add the merge branch's observed-counter arrival/entry model to the WM-S rollout so the bottleneck is fed; (2) keep merge_cal in `PhysicsConfig` so run 2's residual trains on the calibrated acceptance, but treat entries as the queue lever; (3) scope t0-only eval claims to horizons <~150 s.
