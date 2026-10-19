# WM-vs-SUMO fidelity suite (C1-C6)

- schema: `c4_wm_fidelity_suite/1`
- WM: `wm_s_best_ap.pt`; arm `native`; k=1; horizon 300.000 s; entries `queue`
- 2 TRAIN-val starts; road end 553.33 m; workers 2
- tolerance: <= 25 % relative deviation (per-metric absolute floors in the JSON); C3 flags on any overrun
- Local proof run: 2 TRAIN-val starts (`v5_native` + `genrand_plain`), v5m `wm_s_best_ap.pt`, k=1, 300 s horizon, entries queue, 2 workers, wall 1503 s. JSON/MD: `_out/fidelity_suite/fidelity_laptop_2state.{json,md}` in the repo and this directory.

## Bottom line (2-state local proof)

- **C1 FLAG** - CAV accelerations are teacher-forced and match by construction, but the WM's
  CAV gap and time headway to the leader are too large (median 70.6 vs 53.3 m, +32 %; THW 3.51
  vs 2.63 s, +34 %). Hard-brake share is close (0.45 % vs 0.43 %).
- **C2 FLAG** - the WM produces more near misses: TTC < 3 s on 2.53 % of all roster pair-steps
  (SUMO 0.94 %) and < 1.5 s on 0.88 % (SUMO 0.11 %); CAV-follower near-miss rates are 2.2-9.4x
  the SUMO rates.
- **C3 FLAG** - no road-end removal: an average 96.5 WM vehicles per start cross 553.33 m while
  still present (65 and 128 per state), accumulating 7,286 vehicle-seconds beyond the end
  (max dwell 195.5 s). SUMO has zero. The code path confirms the model has an `exit_s_m` gate
  that the predictor never supplies.
- **C4 FLAG** - lane shares are close at 0/60 s, then diverge at 120 s (lane-0 HDV share 0.449
  vs 0.336) and lane speeds differ on the ramp / lane 1.
- **C5 PASS** - per-lane free-flow mean/median/p90 are within 25 % on lanes 0/1/2.
- **C6 FLAG** - the WM brakes in many shallow pulses: 1,037 onset events vs SUMO's 48, with a
  median leader drop of 0.37 vs 5.02 m/s and follower drops 0.46-0.81 m/s vs 0.54-7.64 m/s.


## Checks and verdicts

| check | definition | tolerance | verdict |
|---|---|---|---|
| C1 native CAV behaviour | Native t0 CAVs: acceleration distribution, hard-brake share (a < -2.5 m/s^2), jerk, gap and time headway to the same-lane leader. | <= 25 % relative deviation on every reported scalar (metric floors in JSON). | **FLAG** |
| C2 TTC distribution and near-miss rate | Leader-follower TTC (gap / closing speed) distribution and the share of pair-steps with TTC < 3 s and < 1.5 s; all roster pairs and CAV-follower pairs. | <= 25 % relative deviation on TTC quantiles and near-miss rates. | **FLAG** |
| C3 road-end exits | Vehicles present with s > 553.33 m (SUMO road end): unique vehicles, vehicle-seconds beyond the end and dwell; plus the rollout code-path finding. | PASS only when no WM vehicle is present beyond the road end (SUMO = 0). | **FLAG** |
| C4 lane share over time | HDV lane share (lane 0 / 1 of mainline HDVs, ramp share) and per-lane mean speed at 0/60/120/300 s. | <= 25 % relative deviation on lane shares (floor 0.05) and mean speeds (1 m/s). | **FLAG** |
| C5 free-flow speed distribution | Per-lane speed distribution of roster vehicles with no same-lane leader within 100 m. | <= 25 % relative deviation on per-lane free-flow mean/median/p90 (lanes with >= 20 samples on both sides). | PASS |
| C6 wave amplitude along chains | For leader decel onsets a < -1.5 m/s^2: follower 1..5 own speed drop and its ratio to the leader's drop over the wave window. | <= 25 % relative deviation on follower drops and drop ratios (ranks with >= 5 events on both sides). | **FLAG** |

## C3 road-end code-path finding

**Rollouts do NOT remove vehicles at the road end. wms.model.wms.step gates removal on state['exit_s_m'] (src/wms/model/wms.py: `next_present = next_present & (next_s <= exit_s)`), but WmsPredictor.predict_states (src/wms/eval/predictor.py) never puts exit_s_m into the initial state. Only the recorded presence sequence (roster exits) and the entry masks remove slots, so an over-running WM vehicle stays present past 553.33 m.**

- model exit gate present: yes; predictor supplies `exit_s_m`: no; rollouts remove at road end: no

## Per-state summary

| world | set | arm | t0 s | roster | CAV | C3 WM veh beyond | C3 WM veh-s | C3 SUMO veh beyond |
|---|---|---|---|---|---|---|---|---|
| w963004m1900p40t300 | v5_native | native | 300.000 | 87 | 34 | 65.000 | 5265.500 | 0 |
| w963956m2700p20t300 | genrand_plain | generic_random | 900.000 | 91 | 16 | 128.000 | 9306.500 | 0 |

## C1 - native CAV behaviour (FLAG)

Native t0 CAVs: acceleration distribution, hard-brake share (a < -2.5 m/s^2), jerk, gap and time headway to the same-lane leader.

Tolerance: <= 25 % relative deviation on every reported scalar (metric floors in JSON).

| metric | WM | SUMO | relative deviation | verdict |
|---|---|---|---|---|
| C1 / accel_mean | 0.051 | 0.051 | 0.000 | ok |
| C1 / accel_p50 | 0.009 | 0.010 | 0.005 | ok |
| C1 / accel_p95 | 0.905 | 0.918 | 0.015 | ok |
| C1 / accel_min | -3.265 | -2.893 | 0.129 | ok |
| C1 / hard_brake_share | 0.004 | 0.004 | 0.014 | ok |
| C1 / jerk_mean | 0.002 | 0.002 | 0.000 | ok |
| C1 / jerk_median | -0.001 | -0.001 | 0.000 | ok |
| C1 / jerk_p95_abs | 0.459 | 0.413 | 0.092 | ok |
| C1 / gap_median | 70.575 | 53.345 | 0.323 | **FLAG** |
| C1 / gap_p10 | 14.096 | 17.079 | 0.175 | ok |
| C1 / thw_median | 3.509 | 2.628 | 0.335 | **FLAG** |
| C1 / speed_mean | 19.851 | 19.854 | 0.000 | ok |

## C2 - TTC distribution and near-miss rate (FLAG)

Leader-follower TTC (gap / closing speed) distribution and the share of pair-steps with TTC < 3 s and < 1.5 s; all roster pairs and CAV-follower pairs.

Tolerance: <= 25 % relative deviation on TTC quantiles and near-miss rates.

| metric | WM | SUMO | relative deviation | verdict |
|---|---|---|---|---|
| C2/all_pairs / ttc_p05 | 3.079 | 5.941 | 0.482 | **FLAG** |
| C2/all_pairs / ttc_p10 | 6.631 | 10.142 | 0.346 | **FLAG** |
| C2/all_pairs / ttc_p50 | 65.502 | 84.134 | 0.221 | ok |
| C2/all_pairs / ttc_median | 65.502 | 84.134 | 0.221 | ok |
| C2/all_pairs / near_miss_3_rate | 0.025 | 0.009 | 1.682 | **FLAG** |
| C2/all_pairs / near_miss_15_rate | 0.009 | 0.001 | 1.531 | **FLAG** |
| C2/cav_follower_pairs / ttc_p05 | 3.480 | 8.751 | 0.602 | **FLAG** |
| C2/cav_follower_pairs / ttc_p10 | 7.706 | 12.876 | 0.402 | **FLAG** |
| C2/cav_follower_pairs / ttc_p50 | 51.331 | 89.449 | 0.426 | **FLAG** |
| C2/cav_follower_pairs / ttc_median | 51.331 | 89.449 | 0.426 | **FLAG** |
| C2/cav_follower_pairs / near_miss_3_rate | 0.022 | 0.006 | 2.799 | **FLAG** |
| C2/cav_follower_pairs / near_miss_15_rate | 0.010 | 0.001 | 1.798 | **FLAG** |

## C3 - road-end exits (FLAG)

Vehicles present with s > 553.33 m (SUMO road end): unique vehicles, vehicle-seconds beyond the end and dwell; plus the rollout code-path finding.

Tolerance: PASS only when no WM vehicle is present beyond the road end (SUMO = 0).

| metric | WM | SUMO |
|---|---|---|
| beyond_vehicle_seconds | 7286.000 | 0.000 |
| beyond_vehicle_steps | 14572.000 | 0.000 |
| dt_s | 0.500 | 0.500 |
| max_dwell_s | 195.500 | 0.000 |
| mean_dwell_s | 76.857 | n/a |
| n_vehicle_steps | 76974.500 | 14501.500 |
| n_vehicles_beyond | 96.500 | 0.000 |
| share_of_vehicle_steps | 0.190 | 0.000 |

## C4 - lane share over time (FLAG)

HDV lane share (lane 0 / 1 of mainline HDVs, ramp share) and per-lane mean speed at 0/60/120/300 s.

Tolerance: <= 25 % relative deviation on lane shares (floor 0.05) and mean speeds (1 m/s).

| metric | WM | SUMO | relative deviation | verdict |
|---|---|---|---|---|
| C4@0 / lane0_hdv_share | 0.606 | 0.606 | 0.000 | ok |
| C4@0 / lane1_hdv_share | 0.394 | 0.394 | 0.000 | ok |
| C4@0 / ramp_hdv_share | 0.113 | 0.113 | 0.000 | ok |
| C4@0 / lane0_speed | 24.401 | 24.401 | 0.000 | ok |
| C4@0 / lane1_speed | 20.532 | 20.532 | 0.000 | ok |
| C4@0 / lane2_speed | 9.416 | 9.416 | 0.000 | ok |
| C4@60 / lane0_hdv_share | 0.526 | 0.527 | 0.002 | ok |
| C4@60 / lane1_hdv_share | 0.474 | 0.473 | 0.003 | ok |
| C4@60 / ramp_hdv_share | 0.029 | 0.057 | 0.500 | **FLAG** |
| C4@60 / lane0_speed | 27.511 | 27.723 | 0.008 | ok |
| C4@60 / lane1_speed | 14.345 | 18.147 | 0.210 | ok |
| C4@60 / lane2_speed | 0.167 | 0.453 | 0.286 | **FLAG** |
| C4@120 / lane0_hdv_share | 0.449 | 0.336 | 0.337 | **FLAG** |
| C4@120 / lane1_hdv_share | 0.551 | 0.664 | 0.170 | ok |
| C4@120 / ramp_hdv_share | 0.000 | 0.045 | 0.909 | **FLAG** |
| C4@120 / lane0_speed | 27.969 | 28.444 | 0.017 | ok |
| C4@120 / lane1_speed | 14.056 | 11.098 | 0.267 | **FLAG** |
| C4@120 / lane2_speed | 0.000 | 0.000 | 0.000 | ok |
| C4@300 / lane0_hdv_share | 0.000 | 0.000 | 0.000 | ok |
| C4@300 / lane1_hdv_share | 0.500 | 0.500 | 0.000 | ok |
| C4@300 / ramp_hdv_share | 0.000 | 0.000 | 0.000 | ok |
| C4@300 / lane0_speed | n/a | n/a | n/a | ok |
| C4@300 / lane1_speed | 16.622 | 15.826 | 0.050 | ok |
| C4@300 / lane2_speed | n/a | n/a | n/a | ok |

## C5 - free-flow speed distribution (PASS)

Per-lane speed distribution of roster vehicles with no same-lane leader within 100 m.

Tolerance: <= 25 % relative deviation on per-lane free-flow mean/median/p90 (lanes with >= 20 samples on both sides).

| metric | WM | SUMO | relative deviation | verdict |
|---|---|---|---|---|
| C5/lane0 / ff_mean | 27.785 | 28.374 | 0.021 | ok |
| C5/lane0 / ff_median | 28.276 | 28.811 | 0.019 | ok |
| C5/lane0 / ff_p90 | 30.266 | 30.324 | 0.002 | ok |
| C5/lane1 / ff_mean | 21.814 | 22.616 | 0.035 | ok |
| C5/lane1 / ff_median | 24.503 | 25.099 | 0.024 | ok |
| C5/lane1 / ff_p90 | 28.699 | 28.394 | 0.011 | ok |
| C5/lane2 / ff_mean | 2.844 | 2.464 | 0.154 | ok |
| C5/lane2 / ff_median | 0.560 | 0.522 | 0.038 | ok |
| C5/lane2 / ff_p90 | 10.354 | 8.631 | 0.200 | ok |

## C6 - wave amplitude along chains (FLAG)

For leader decel onsets a < -1.5 m/s^2: follower 1..5 own speed drop and its ratio to the leader's drop over the wave window.

Tolerance: <= 25 % relative deviation on follower drops and drop ratios (ranks with >= 5 events on both sides).

| metric | WM | SUMO |
|---|---|---|
| event_threshold_mps2 | -1.500 | -1.500 |
| leader_drop_median | 0.370 | 5.016 |
| n_events | 1037.500 | 48.000 |
| n_events_with_followers | 936.500 | 46.500 |
| window_s | 10.000 | 10.000 |

| rank | WM drop | SUMO drop | WM ratio | SUMO ratio |
|---|---|---|---|---|
| 1 | 0.807 | 7.638 | 0.987 | 1.198 |
| 2 | 0.463 | 5.246 | 0.536 | 0.909 |
| 3 | 0.456 | 2.254 | 0.555 | 0.489 |
| 4 | 0.516 | 1.289 | 0.629 | 0.143 |
| 5 | 0.538 | 0.537 | 0.539 | 0.032 |

## Command

Local (this proof run; laptop, <= 3 workers):

```sh
python scripts/c4/wm_fidelity_suite.py --wm E:/research-dense/runs/active/C4-V7FT-20261017/v5m_ref/wm_s_best_ap.pt --states E:/research-dense/runs/active/C4-V7FT-20261017/v10delaycheck/inputs/delay_states.json --select 2 --horizon 300 --targets 0,60,120,300 --wave-window-s 10 --k 1 --entries queue --device cpu --workers 2 --out _out/fidelity_suite/fidelity_laptop_2state.json --md _out/fidelity_suite/fidelity_laptop_2state.md
```

cpuN production command (24 starts, k=2, 300 s, 3 workers):

```sh
cd ~/c4
OMP_NUM_THREADS=1 ~/miniconda3/bin/python scripts/c4/wm_fidelity_suite.py \
  --wm ~/c4/delay_check/wm_s_best_ap.pt \
  --selection ~/c4/delay_check/sel24_states.json \
  --horizon 300 --targets 0,60,120,300 --wave-window-s 10 \
  --k 2 --entries queue --device cpu --workers 3 \
  --out ~/c4/fidelity_suite/fidelity_suite.json \
  --md ~/c4/fidelity_suite/FIDELITY_SUITE.md
```

## Notes

- Per-vehicle metrics (C1, C2, C4-C6) restrict BOTH sides to the t0 roster, because the recorded delay-check stream carries no post-t0 entrants.
- C3 deliberately uses all WM vehicles (entry slots included).
- CAV motion is teacher-forced from the recorded arm, so C1's CAV acceleration and jerk are expected to match SUMO; the gap / time-headway part is the non-trivial part of C1.
- C6 events are onset rows (a < -1.5 m/s^2 after a row at or above it), not every braking event-second.

