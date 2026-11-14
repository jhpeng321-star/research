# FIDELITY — open-road A11 rollout speed/queue fidelity (TRAIN-only, no SUMO)

Model member_1.pt sha256 41221852…f21e8; TRAIN chunk-4 states. Diag: `_out/fidelity_diag.py` -> `_out/fidelity_diag.json`.

## Diagnosis (8 TRAIN chunk-4 states; recorded `__pi` vs rollout, per 10 s bin, segments in the shared merge-relative `s_down`: up<0, merge [0,75.44], down>75.44)
- **Root cause: geometric frame mismatch.** `WorldState.s_m` is the model `s_down` coordinate (`geometry.s_down_of` = physical s − lane offset), but the frozen `partA_v2g/params_nominal.json` geometry is PHYSICAL. The mandatory-merge zone `s0z..s1z` = 378.66..454.1 therefore lies beyond lane-2 `s_down` (max `llen−off` = 75.44): **the merge rule NEVER fires** and the accel-lane-end obstacle never binds. Confirmed: nominal merge-segment speed MAE vs recorded is 13.5 m/s (no queue); recorded merge mean_v ~6–9 m/s with a stopped queue.
- Residual net is small: mean |accel_delta| = 0.28–0.37 m/s² (not the cause).
- SUMO vTypes are inline `hdv_mix` in `osm.demand.*.rou.xml`: Krauss, speedFactor mean **0.992** (0.10/0.70/0.20 aggressive/normal/cautious), tau mean 1.165, accel mean 2.52, decel mean 4.2, minGap mean 2.6. So `v0`=31.29 is not the driver of the gap; congestion/bottleneck is.
- Entries use the observed counters (main 0.75, ramp 0.125 veh/s) — correct magnitude. Exit threshold fixed previously (EXIT_FIX).

## Fix (flag-gated, default OFF = byte-identical)
`Rollout2Config.sumo_fidelity` (+`sumo_idm`, default True; `sumo_speed_factor`, default None) -> `baseline.model_frame_params`: subtracts each lane offset from segment starts/lane lengths, shifts the accel merge zone by `−off[2]` (->[0,75.44]), zeroes offsets so `d=−s_down` is the shared cross-lane axis, and replaces IDM by the `hdv_mix`-weighted vType means. Wired through `batched_rollout` / `A11Predictor`. Tests: `rollout2/tests/test_fidelity.py` (7).

## Gap vs recorded (8-state mean of per-bin |Δ|, m/s; and merge stopped vehicle-ticks total 8 states)
| variant | merge MAE | up MAE | down MAE | merge stopped | head dV MAE | corr |
|---|---|---|---|---|---|---|
| closed | 13.6 | 5.3 | 6.1 | 4 | 3564 | 0.447 |
| exit | 13.6 | 5.9 | 1.8 | 4 | 961 | 0.348 |
| open | 13.5 | 2.7 | 1.5 | 34 | **947** | 0.372 |
| open+fid(geom only) | **3.9** | **2.0** | 1.4 | 13269 | 1206 | −0.381 |
| open+fid(full) | 4.2 | 3.5 | 2.4 | 13629 | 1048 | −0.217 |
| recorded | — | — | — | 1849 | — | — |

## Findings
1. The frame fix restores the merge bottleneck: merge mean-speed error 13.5 -> 3.9 m/s and upstream 2.7 -> 2.0; the queue now exists (nominal had ~0 stopped vehicle-ticks).
2. It **over-queues 7x** vs recorded (13269 vs 1849 merge stopped vehicle-ticks): the mandatory-merge gap-acceptance/yielding rule lets few ramp vehicles in, so lane-2 piles up at the lane end. Caps/n_cav bin11 remain over-dropped (w8676 recorded 8.6 vs open_fid 1) — mainline capacity drop is still not reproduced.
3. **Head dV does not improve** (open MAE 947/corr 0.37 vs full fid 1048/corr −0.22): the paired-effect head is trained on recorded plans; the fidelity rollout's slower/queued plans are not recalibrated, and the over-queue moves negative-TRUE states the wrong way. Fixing the head is a training change (out of F2 scope).
4. Mainline upstream remains ~24–27 vs recorded ~20–22: an equilibrium fundamental-diagram difference, not the merge.

## Enable / next
`A11Predictor(bundle, sumo_fidelity=True, open_road=True)` or `batched_rollout(..., sumo_fidelity=True, open_entries=True, physical_exits=True)`. Next: calibrate the on-ramp merge gap acceptance to the recorded merge rate (remove the 7x over-queue), then consider retraining the plan head on fidelity-rollout plans.
