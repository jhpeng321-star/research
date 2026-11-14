# MERGE_CAL — calibrated SUMO-like merge physics (flag `merge_cal`, default OFF)

STATUS: DONE (fitB kept+committed; fitted on TRAIN chunk-4 only; validated on the 20 HORIZON_ERROR states, pure physics + A11 residual; 64 tests pass, ruff clean)

## What changed
- `rollout2/merge_cal.py` (new core, wired via `Rollout2Config.merge_cal` / `batched_rollout(merge_cal=False)` default):
  strategic LC2013-style mandatory merge (urgency-relaxed lead time `t_lead_base->t_lead_end`,
  follower/lead required-decel ceiling `b_accept_base->b_accept_end`, gap floor), cooperative
  mainline yielding (bounded `b_yield_max`, `yield_lookahead`, yields only to mergers with
  `v >= yield_min_merger_speed_mps`), lane-2 end obstacle only within `end_obstacle_range_m`.
- Params: `rollout2/merge_cal_params.json` (fitB: b_yield_max 1.75, yield_min_speed 2.5,
  merger-speed floor 3.5). Export for torch training run 2: `_out/merge_params.json`.

## Fit / validation (TRAIN chunk-4 recorded stats `_out/merge_cal_recorded.json`)
- 8 TRAIN states, merge stopped vehicle-ticks: recorded 1849; pre-cal v2g rule 13269 (~7.2x
  over-queue); fitB **1516 (0.82x)**; merge rate 7.19 vs rec 6.81 veh/min.
- 20 states (HORIZON_ERROR set): recorded 6293; cal_phys (pure physics) **3918 (0.62x)**;
  cal (A11 residual) **3944 (0.63x)**; rate 6.92 / 7.18 vs rec 6.5; per-30 s merge-section
  speed [10.7, 8.5, 9.7, 11.3] phys, [11.9, 9.5, 10.8, 13.2] residual vs rec [9.9, 8.6, 8.8, 8.4].
- Position error p50 @30/60/120 s: phys 25.3 / 52.7 / 140.9 m; residual 30.2 / 62.7 / 208.1 m.
- Matched fraction @60 s: 0.94 phys / 0.92 residual vs rec 0.53 (rollout retains more t0
  vehicles than the recording; recording attrition is route exits, not tracking loss).

## Caveats
- Aggregate stop ratio <1 hides state spread (e.g. w8623 516 vs 262 rec; w8677 227 vs 172).
- Accepted lead gaps (p50 ~3.5 m) are tighter than recorded (~10.9 m); bounded by the
  safety ceilings, not tunable without re-fit.
- Residual arm degrades dpos (known A11 over-speed bias); pure physics is the better base.

## Artifacts / tests
- Evals: `_out/mc_fitA.json`, `_out/mc_fitB.json` (20 states, cal_phys), `_out/mc_fitB_res.json`
  (20 states, cal residual); harness `_out/merge_cal_eval.py`, report `_out/merge_cal_fit_report.py`.
- Tests: `output/wd01/wm_n1/rollout2/tests/test_merge_cal.py` (6) + full rollout2 suite 64 passed
  (`python -m pytest output/wd01/wm_n1/rollout2/tests` from repo root); ruff clean on touched files.
