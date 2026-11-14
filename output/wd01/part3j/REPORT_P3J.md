# WD-01 part 3j — numeric decomposition of the structured emulator's delay under-prediction

Deliverable: a NUMERIC decomposition of the delay under-prediction of the
structured native Part A emulator (L0 = plain IDM `idm_normal`,
L3 = structured sampled IDM `idm_sampled`, member 0), as measured by the
part-3h gate. ZERO simulation (no SUMO/TraCI anywhere); train+validation
roots only — the holdout block inside `root_arrays` stays authoritative
and was never bypassed.

## Method (every number below is pasted from `decompose.py` stdout)

- Rollouts: `part3/engine.py Rollout` imported READ-ONLY from the
  populated copy `D:/rd-wt/wd01/output/wd01/part3`; truth is the p2C
  model view (`root_arrays`); 0.5 s snapshots, H = 60 s (K = 120 steps).
- Delay per vehicle-step: `dt - ds/limit(lane, position)` with
  `ds` = forward advance = DECREASE of `d = OFFSETS[lane] - s` (d
  decreases downstream and is continuous across lane changes), `limit`
  = per-segment SUMO limit from `part4/delay/geometry` via
  `engine.lane_speed_limits` (main 31.29; accel 22.22, merge 31.29),
  evaluated at the START of the step. Birth steps use the logged
  insertion `(lane, s)` as start state on both sides.
- Attribution cells: lane (0/1/2) x segment-by-start-of-step d
  (d<0 downstream; 0<=d<75 merge_edge; 75<=d<800 rsu_window; d>=800
  upstream) x class (t0-present vs born) x lc_affected (lane change
  within the previous 5 s of the same trajectory, change step included).
- Exit mismatch (one side left the section while the other remains:
  the remaining side's delay is counted) and entry waiting (expected 0;
  births are inserted at logged times) are accounted separately.
- Reconciliation: the p3h "delay" metric is pure exposure
  (0.5 s x vehicle count per tick), so per-window
  `0.5*(n_steps_pred - n_steps_true)` must equal the p3h per-window
  `delay_pred - delay_true` exactly (identical rollouts, L0 and L3
  member 0; p3h's published L3 mean is over 5 members), and the window
  mean must match the published gate_v2 values within member spread.
- Identity used throughout: per-vehicle-step physics delay =
  exposure (0.5 s) - free-flow usage (ds/limit), so
  `delay_deficit = expo_deficit + ff_component` where
  `ff_component = ff_true - ff_pred` and `ff = 0.5*n_steps - delay`.
- Files: `results/decomposition.csv` (per-window cells),
  `results/speed_profile.csv` (matched-speed profile),
  `results/stdout_last.txt` (verbatim stdout).

## Full stdout of `python decompose.py` (this run)

```text
priority below-normal: False
=== WD-01 part 3j: delay under-prediction decomposition ===
ZERO simulation; holdout blocked in root_arrays (never passed).
device=cpu torch_threads=2
variants: L0 = kind 'idm_normal'; L3 = kind 'idm_sampled' member_seed=1 lc_scale=0.003
H=60 s (K=120 steps of 0.5 s); validation roots ['cv20_7005m2000', 'cv20_7006m2000', 'cv30_7305m2700', 'seed_7535m2400']; train check roots ['cv17', 'cv19_4107', 'cv19_4108', 'cv20_7001m2000']
grids: main t0 every 60 s (120, 180, 240, 300, 360, 420, 480); reconciliation t0 every 30 s (120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480)
d = OFFSETS[lane] - s, OFFSETS=(2645.67, 2644.22, 378.66); segments [('downstream', -inf, 0.0), ('merge_edge', 0.0, 75.0), ('rsu_window', 75.0, 800.0), ('upstream', 800.0, inf)]
limits via engine.lane_speed_limits: e.g. limit(0,100)=31.29 limit(2,0)=22.22 limit(2,440)=31.29

== A. reconciliation with p3h (H=60, validation, t0 every 30 s: 4 roots x 13 t0 = 52 windows/variant) ==
[   4.8s] L0 cv20_7005m2000 t0=120: n_p=7988 n_t=8744 expo_def=-378.0 delay_def=-483.1
[  11.4s] L3 cv20_7005m2000 t0=120: n_p=8165 n_t=8744 expo_def=-289.5 delay_def=-171.6
[  14.3s] L0 cv20_7005m2000 t0=150: n_p=8203 n_t=8712 expo_def=-254.5 delay_def=-459.0
[  20.3s] L3 cv20_7005m2000 t0=150: n_p=8388 n_t=8712 expo_def=-162.0 delay_def=-78.6
[  23.7s] L0 cv20_7005m2000 t0=180: n_p=8365 n_t=8869 expo_def=-252.0 delay_def=-459.0
[  30.2s] L3 cv20_7005m2000 t0=180: n_p=8538 n_t=8869 expo_def=-165.5 delay_def=-121.0
[  33.4s] L0 cv20_7005m2000 t0=210: n_p=8415 n_t=9012 expo_def=-298.5 delay_def=-443.7
[  39.9s] L3 cv20_7005m2000 t0=210: n_p=8554 n_t=9012 expo_def=-229.0 delay_def=-65.4
[  44.8s] L0 cv20_7005m2000 t0=240: n_p=8060 n_t=8816 expo_def=-378.0 delay_def=-529.2
[  54.5s] L3 cv20_7005m2000 t0=240: n_p=8316 n_t=8816 expo_def=-250.0 delay_def=-173.5
[  58.9s] L0 cv20_7005m2000 t0=270: n_p=8060 n_t=9079 expo_def=-509.5 delay_def=-694.4
[  65.8s] L3 cv20_7005m2000 t0=270: n_p=8243 n_t=9079 expo_def=-418.0 delay_def=-362.4
[  69.7s] L0 cv20_7005m2000 t0=300: n_p=8731 n_t=9811 expo_def=-540.0 delay_def=-763.6
[  76.9s] L3 cv20_7005m2000 t0=300: n_p=8937 n_t=9811 expo_def=-437.0 delay_def=-399.5
[  80.6s] L0 cv20_7005m2000 t0=330: n_p=9652 n_t=10533 expo_def=-440.5 delay_def=-627.8
[  87.5s] L3 cv20_7005m2000 t0=330: n_p=9906 n_t=10533 expo_def=-313.5 delay_def=-202.0
[  91.0s] L0 cv20_7005m2000 t0=360: n_p=10221 n_t=10955 expo_def=-367.0 delay_def=-524.8
[  98.0s] L3 cv20_7005m2000 t0=360: n_p=10483 n_t=10955 expo_def=-236.0 delay_def=-11.2
[ 101.5s] L0 cv20_7005m2000 t0=390: n_p=9668 n_t=10730 expo_def=-531.0 delay_def=-695.7
[ 111.9s] L3 cv20_7005m2000 t0=390: n_p=9902 n_t=10730 expo_def=-414.0 delay_def=-230.4
[ 116.5s] L0 cv20_7005m2000 t0=420: n_p=8892 n_t=10091 expo_def=-599.5 delay_def=-790.6
[ 124.0s] L3 cv20_7005m2000 t0=420: n_p=9159 n_t=10091 expo_def=-466.0 delay_def=-434.4
[ 127.1s] L0 cv20_7005m2000 t0=450: n_p=7914 n_t=9321 expo_def=-703.5 delay_def=-838.0
[ 133.4s] L3 cv20_7005m2000 t0=450: n_p=8068 n_t=9321 expo_def=-626.5 delay_def=-551.6
[ 136.2s] L0 cv20_7005m2000 t0=480: n_p=7304 n_t=8799 expo_def=-747.5 delay_def=-789.9
[ 141.7s] L3 cv20_7005m2000 t0=480: n_p=7522 n_t=8799 expo_def=-638.5 delay_def=-492.5
[ 145.2s] L0 cv20_7006m2000 t0=120: n_p=6685 n_t=7267 expo_def=-291.0 delay_def=-283.9
[ 151.0s] L3 cv20_7006m2000 t0=120: n_p=6871 n_t=7267 expo_def=-198.0 delay_def=-21.7
[ 154.2s] L0 cv20_7006m2000 t0=150: n_p=6939 n_t=7414 expo_def=-237.5 delay_def=-212.8
[ 160.2s] L3 cv20_7006m2000 t0=150: n_p=7080 n_t=7414 expo_def=-167.0 delay_def=92.6
[ 163.8s] L0 cv20_7006m2000 t0=180: n_p=7265 n_t=7792 expo_def=-263.5 delay_def=-286.5
[ 173.0s] L3 cv20_7006m2000 t0=180: n_p=7385 n_t=7792 expo_def=-203.5 delay_def=-37.8
[ 176.9s] L0 cv20_7006m2000 t0=210: n_p=7092 n_t=8042 expo_def=-475.0 delay_def=-542.8
[ 182.5s] L3 cv20_7006m2000 t0=210: n_p=7317 n_t=8042 expo_def=-362.5 delay_def=-251.1
[ 185.3s] L0 cv20_7006m2000 t0=240: n_p=6542 n_t=7967 expo_def=-712.5 delay_def=-724.4
[ 190.4s] L3 cv20_7006m2000 t0=240: n_p=6698 n_t=7967 expo_def=-634.5 delay_def=-486.9
[ 193.2s] L0 cv20_7006m2000 t0=270: n_p=6642 n_t=8030 expo_def=-694.0 delay_def=-702.5
[ 198.6s] L3 cv20_7006m2000 t0=270: n_p=6774 n_t=8030 expo_def=-628.0 delay_def=-461.2
[ 201.4s] L0 cv20_7006m2000 t0=300: n_p=6988 n_t=7936 expo_def=-474.0 delay_def=-545.2
[ 207.4s] L3 cv20_7006m2000 t0=300: n_p=7189 n_t=7936 expo_def=-373.5 delay_def=-265.3
[ 210.3s] L0 cv20_7006m2000 t0=330: n_p=7130 n_t=7728 expo_def=-299.0 delay_def=-372.8
[ 215.8s] L3 cv20_7006m2000 t0=330: n_p=7286 n_t=7728 expo_def=-221.0 delay_def=-92.2
[ 218.5s] L0 cv20_7006m2000 t0=360: n_p=7416 n_t=7818 expo_def=-201.0 delay_def=-309.4
[ 227.4s] L3 cv20_7006m2000 t0=360: n_p=7599 n_t=7818 expo_def=-109.5 delay_def=-5.7
[ 232.3s] L0 cv20_7006m2000 t0=390: n_p=7228 n_t=7902 expo_def=-337.0 delay_def=-498.4
[ 240.7s] L3 cv20_7006m2000 t0=390: n_p=7394 n_t=7902 expo_def=-254.0 delay_def=-201.7
[ 244.2s] L0 cv20_7006m2000 t0=420: n_p=7342 n_t=8408 expo_def=-533.0 delay_def=-674.8
[ 250.2s] L3 cv20_7006m2000 t0=420: n_p=7602 n_t=8408 expo_def=-403.0 delay_def=-370.6
[ 253.3s] L0 cv20_7006m2000 t0=450: n_p=7970 n_t=8958 expo_def=-494.0 delay_def=-578.5
[ 259.3s] L3 cv20_7006m2000 t0=450: n_p=8239 n_t=8958 expo_def=-359.5 delay_def=-252.1
[ 262.3s] L0 cv20_7006m2000 t0=480: n_p=7643 n_t=8582 expo_def=-469.5 delay_def=-496.9
[ 268.1s] L3 cv20_7006m2000 t0=480: n_p=7845 n_t=8582 expo_def=-368.5 delay_def=-154.7
[ 273.9s] L0 cv30_7305m2700 t0=120: n_p=10267 n_t=10789 expo_def=-261.0 delay_def=-514.0
[ 282.1s] L3 cv30_7305m2700 t0=120: n_p=10453 n_t=10789 expo_def=-168.0 delay_def=-49.9
[ 288.2s] L0 cv30_7305m2700 t0=150: n_p=10479 n_t=11230 expo_def=-375.5 delay_def=-650.0
[ 300.1s] L3 cv30_7305m2700 t0=150: n_p=10667 n_t=11230 expo_def=-281.5 delay_def=-144.5
[ 304.3s] L0 cv30_7305m2700 t0=180: n_p=10435 n_t=11374 expo_def=-469.5 delay_def=-754.1
[ 312.6s] L3 cv30_7305m2700 t0=180: n_p=10674 n_t=11374 expo_def=-350.0 delay_def=-294.0
[ 316.7s] L0 cv30_7305m2700 t0=210: n_p=10418 n_t=11516 expo_def=-549.0 delay_def=-830.9
[ 324.8s] L3 cv30_7305m2700 t0=210: n_p=10619 n_t=11516 expo_def=-448.5 delay_def=-330.1
[ 329.2s] L0 cv30_7305m2700 t0=240: n_p=10827 n_t=11903 expo_def=-538.0 delay_def=-975.4
[ 337.3s] L3 cv30_7305m2700 t0=240: n_p=11108 n_t=11903 expo_def=-397.5 delay_def=-444.7
[ 341.8s] L0 cv30_7305m2700 t0=270: n_p=10985 n_t=12245 expo_def=-630.0 delay_def=-1088.5
[ 353.5s] L3 cv30_7305m2700 t0=270: n_p=11318 n_t=12245 expo_def=-463.5 delay_def=-563.8
[ 359.5s] L0 cv30_7305m2700 t0=300: n_p=11107 n_t=12347 expo_def=-620.0 delay_def=-1139.7
[ 368.1s] L3 cv30_7305m2700 t0=300: n_p=11574 n_t=12347 expo_def=-386.5 delay_def=-550.5
[ 372.6s] L0 cv30_7305m2700 t0=330: n_p=11628 n_t=12819 expo_def=-595.5 delay_def=-1253.2
[ 381.2s] L3 cv30_7305m2700 t0=330: n_p=12005 n_t=12819 expo_def=-407.0 delay_def=-673.3
[ 385.7s] L0 cv30_7305m2700 t0=360: n_p=12272 n_t=13618 expo_def=-673.0 delay_def=-1434.3
[ 394.8s] L3 cv30_7305m2700 t0=360: n_p=12739 n_t=13618 expo_def=-439.5 delay_def=-800.8
[ 399.5s] L0 cv30_7305m2700 t0=390: n_p=12765 n_t=14189 expo_def=-712.0 delay_def=-1508.2
[ 412.4s] L3 cv30_7305m2700 t0=390: n_p=13250 n_t=14189 expo_def=-469.5 delay_def=-739.0
[ 420.4s] L0 cv30_7305m2700 t0=420: n_p=13657 n_t=14743 expo_def=-543.0 delay_def=-1396.1
[ 431.1s] L3 cv30_7305m2700 t0=420: n_p=14058 n_t=14743 expo_def=-342.5 delay_def=-739.9
[ 436.4s] L0 cv30_7305m2700 t0=450: n_p=14086 n_t=15128 expo_def=-521.0 delay_def=-1337.0
[ 447.2s] L3 cv30_7305m2700 t0=450: n_p=14563 n_t=15128 expo_def=-282.5 delay_def=-635.3
[ 452.5s] L0 cv30_7305m2700 t0=480: n_p=14126 n_t=15280 expo_def=-577.0 delay_def=-1444.4
[ 464.4s] L3 cv30_7305m2700 t0=480: n_p=14485 n_t=15280 expo_def=-397.5 delay_def=-765.7
[ 471.4s] L0 seed_7535m2400 t0=120: n_p=9533 n_t=9938 expo_def=-202.5 delay_def=-332.6
[ 481.5s] L3 seed_7535m2400 t0=120: n_p=9736 n_t=9938 expo_def=-101.0 delay_def=79.7
[ 485.9s] L0 seed_7535m2400 t0=150: n_p=9927 n_t=10456 expo_def=-264.5 delay_def=-451.4
[ 493.7s] L3 seed_7535m2400 t0=150: n_p=10209 n_t=10456 expo_def=-123.5 delay_def=38.2
[ 497.7s] L0 seed_7535m2400 t0=180: n_p=9741 n_t=10489 expo_def=-374.0 delay_def=-544.1
[ 505.8s] L3 seed_7535m2400 t0=180: n_p=9962 n_t=10489 expo_def=-263.5 delay_def=-76.8
[ 509.7s] L0 seed_7535m2400 t0=210: n_p=9864 n_t=10546 expo_def=-341.0 delay_def=-583.9
[ 517.8s] L3 seed_7535m2400 t0=210: n_p=10041 n_t=10546 expo_def=-252.5 delay_def=-139.0
[ 522.4s] L0 seed_7535m2400 t0=240: n_p=10225 n_t=10730 expo_def=-252.5 delay_def=-528.5
[ 533.3s] L3 seed_7535m2400 t0=240: n_p=10460 n_t=10730 expo_def=-135.0 delay_def=-37.2
[ 537.8s] L0 seed_7535m2400 t0=270: n_p=10666 n_t=11332 expo_def=-333.0 delay_def=-528.2
[ 544.9s] L3 seed_7535m2400 t0=270: n_p=10851 n_t=11332 expo_def=-240.5 delay_def=-47.3
[ 549.1s] L0 seed_7535m2400 t0=300: n_p=11237 n_t=12072 expo_def=-417.5 delay_def=-638.6
[ 558.0s] L3 seed_7535m2400 t0=300: n_p=11495 n_t=12072 expo_def=-288.5 delay_def=-110.8
[ 562.5s] L0 seed_7535m2400 t0=330: n_p=11181 n_t=12366 expo_def=-592.5 delay_def=-833.0
[ 571.9s] L3 seed_7535m2400 t0=330: n_p=11400 n_t=12366 expo_def=-483.0 delay_def=-340.7
[ 576.4s] L0 seed_7535m2400 t0=360: n_p=11455 n_t=12362 expo_def=-453.5 delay_def=-819.6
[ 585.2s] L3 seed_7535m2400 t0=360: n_p=11857 n_t=12362 expo_def=-252.5 delay_def=-255.6
[ 591.1s] L0 seed_7535m2400 t0=390: n_p=11538 n_t=12379 expo_def=-420.5 delay_def=-974.6
[ 601.7s] L3 seed_7535m2400 t0=390: n_p=11927 n_t=12379 expo_def=-226.0 delay_def=-326.7
[ 606.2s] L0 seed_7535m2400 t0=420: n_p=11447 n_t=12652 expo_def=-602.5 delay_def=-1177.2
[ 615.1s] L3 seed_7535m2400 t0=420: n_p=11799 n_t=12652 expo_def=-426.5 delay_def=-616.1
[ 619.5s] L0 seed_7535m2400 t0=450: n_p=12038 n_t=13305 expo_def=-633.5 delay_def=-1230.8
[ 628.1s] L3 seed_7535m2400 t0=450: n_p=12446 n_t=13305 expo_def=-429.5 delay_def=-643.9
[ 632.9s] L0 seed_7535m2400 t0=480: n_p=12959 n_t=14050 expo_def=-545.5 delay_def=-1119.5
[ 641.6s] L3 seed_7535m2400 t0=480: n_p=13418 n_t=14050 expo_def=-316.0 delay_def=-429.3
  L0: mean exposure_pred=4800.17  mean exposure_true=5261.29  mean deficit=-461.12 veh s  | p3h gate_v2 delay_bias=-461.11 (delay_pred=4800.17 delay_true=5261.29)  deviation=-0.00%
  L3: mean exposure_pred=4928.60  mean exposure_true=5261.29  mean deficit=-332.69 veh s  | p3h gate_v2 delay_bias=-328.05 (delay_pred=4933.24 delay_true=5261.29)  deviation=-1.42%
  L0: per-window |diff| vs p3h raw member-0 rows: max=0.000000 veh s over 52 windows (worst ('L0', 0, 'cv20_7005m2000', 120) diff=+0.000000)
  L3: per-window |diff| vs p3h raw member-0 rows: max=0.000000 veh s over 52 windows (worst ('L3', 0, 'cv20_7005m2000', 120) diff=+0.000000)

== B. per-window totals (validation, t0 every 60 s) ==
  root             t0  variant  n_pred n_true expo_deficit delay_pred delay_true delay_deficit
  cv20_7005m2000   120       L0   7988   8744      -378.00     257.54     740.68       -483.14
  cv20_7005m2000   120       L3   8165   8744      -289.50     569.12     740.68       -171.57
  cv20_7005m2000   180       L0   8365   8869      -252.00     295.16     754.19       -459.03
  cv20_7005m2000   180       L3   8538   8869      -165.50     633.20     754.19       -120.99
  cv20_7005m2000   240       L0   8060   8816      -378.00     210.21     739.41       -529.20
  cv20_7005m2000   240       L3   8316   8816      -250.00     565.90     739.41       -173.51
  cv20_7005m2000   300       L0   8731   9811      -540.00     345.85    1109.50       -763.65
  cv20_7005m2000   300       L3   8937   9811      -437.00     710.03    1109.50       -399.47
  cv20_7005m2000   360       L0  10221  10955      -367.00     458.43     983.25       -524.83
  cv20_7005m2000   360       L3  10483  10955      -236.00     972.02     983.25        -11.23
  cv20_7005m2000   420       L0   8892  10091      -599.50     332.47    1123.04       -790.57
  cv20_7005m2000   420       L3   9159  10091      -466.00     688.67    1123.04       -434.37
  cv20_7005m2000   480       L0   7304   8799      -747.50     238.89    1028.77       -789.88
  cv20_7005m2000   480       L3   7522   8799      -638.50     536.30    1028.77       -492.47
  cv20_7006m2000   120       L0   6685   7267      -291.00     139.46     423.36       -283.90
  cv20_7006m2000   120       L3   6871   7267      -198.00     401.68     423.36        -21.68
  cv20_7006m2000   180       L0   7265   7792      -263.50     146.90     433.38       -286.48
  cv20_7006m2000   180       L3   7385   7792      -203.50     395.61     433.38        -37.77
  cv20_7006m2000   240       L0   6542   7967      -712.50     159.52     883.94       -724.42
  cv20_7006m2000   240       L3   6698   7967      -634.50     397.02     883.94       -486.92
  cv20_7006m2000   300       L0   6988   7936      -474.00     235.10     780.28       -545.18
  cv20_7006m2000   300       L3   7189   7936      -373.50     514.94     780.28       -265.34
  cv20_7006m2000   360       L0   7416   7818      -201.00     177.77     487.14       -309.36
  cv20_7006m2000   360       L3   7599   7818      -109.50     481.46     487.14         -5.68
  cv20_7006m2000   420       L0   7342   8408      -533.00     194.61     869.41       -674.80
  cv20_7006m2000   420       L3   7602   8408      -403.00     498.78     869.41       -370.63
  cv20_7006m2000   480       L0   7643   8582      -469.50     243.47     740.40       -496.93
  cv20_7006m2000   480       L3   7845   8582      -368.50     585.69     740.40       -154.71
  cv30_7305m2700   120       L0  10267  10789      -261.00     554.06    1068.03       -513.96
  cv30_7305m2700   120       L3  10453  10789      -168.00    1018.09    1068.03        -49.94
  cv30_7305m2700   180       L0  10435  11374      -469.50     587.01    1341.12       -754.11
  cv30_7305m2700   180       L3  10674  11374      -350.00    1047.08    1341.12       -294.04
  cv30_7305m2700   240       L0  10827  11903      -538.00     619.06    1594.43       -975.37
  cv30_7305m2700   240       L3  11108  11903      -397.50    1149.71    1594.43       -444.71
  cv30_7305m2700   300       L0  11107  12347      -620.00     829.71    1969.44      -1139.74
  cv30_7305m2700   300       L3  11574  12347      -386.50    1418.94    1969.44       -550.50
  cv30_7305m2700   360       L0  12272  13618      -673.00    1133.20    2567.47      -1434.28
  cv30_7305m2700   360       L3  12739  13618      -439.50    1766.64    2567.47       -800.83
  cv30_7305m2700   420       L0  13657  14743      -543.00    1678.70    3074.85      -1396.15
  cv30_7305m2700   420       L3  14058  14743      -342.50    2334.90    3074.85       -739.95
  cv30_7305m2700   480       L0  14126  15280      -577.00    1872.73    3317.11      -1444.39
  cv30_7305m2700   480       L3  14485  15280      -397.50    2551.40    3317.11       -765.72
  seed_7535m2400   120       L0   9533   9938      -202.50     386.82     719.41       -332.59
  seed_7535m2400   120       L3   9736   9938      -101.00     799.08     719.41         79.67
  seed_7535m2400   180       L0   9741  10489      -374.00     513.31    1057.39       -544.08
  seed_7535m2400   180       L3   9962  10489      -263.50     980.61    1057.39        -76.78
  seed_7535m2400   240       L0  10225  10730      -252.50     593.05    1121.57       -528.52
  seed_7535m2400   240       L3  10460  10730      -135.00    1084.32    1121.57        -37.25
  seed_7535m2400   300       L0  11237  12072      -417.50     653.66    1292.28       -638.62
  seed_7535m2400   300       L3  11495  12072      -288.50    1181.47    1292.28       -110.81
  seed_7535m2400   360       L0  11455  12362      -453.50     793.86    1613.50       -819.64
  seed_7535m2400   360       L3  11857  12362      -252.50    1357.95    1613.50       -255.55
  seed_7535m2400   420       L0  11447  12652      -602.50     939.66    2116.88      -1177.22
  seed_7535m2400   420       L3  11799  12652      -426.50    1500.77    2116.88       -616.11
  seed_7535m2400   480       L0  12959  14050      -545.50    1351.11    2470.59      -1119.48
  seed_7535m2400   480       L3  13418  14050      -316.00    2041.29    2470.59       -429.29
  L0 60s-grid means over 28 windows: expo_def=-454.86  delay_pred=569.33  delay_true=1300.74  delay_def=-731.41 veh s
  L3 60s-grid means over 28 windows: expo_def=-322.77  delay_pred=1006.52  delay_true=1300.74  delay_def=-294.22 veh s

== C. deficit by attribute (validation, 60 s grid; units veh s pooled over 28 windows/variant; expo_deficit = 0.5*(n_pred - n_true); delay_deficit = delay_pred - delay_true (physics delay); ff_component = ff_true - ff_pred; identity: expo + ff = delay deficit) ==
  -- by lane (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lane=0                128048       136645     -4298.50       320.41      -3978.09
  lane=1                131892       141719     -4913.50     -7817.95     -12731.45
  lane=2                  8790        15838     -3524.00      -245.97      -3769.97
  -- by segment (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  segment=downstream        53234        65335     -6050.50     -2148.68      -8199.18
  segment=merge_edge         9185        13055     -1935.00      -645.01      -2580.01
  segment=rsu_window        61708        68014     -3153.00     -1822.01      -4975.01
  segment=upstream         144603       147798     -1597.50     -3127.81      -4725.31
  -- by class (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  class=born              68253        72766     -2256.50     -2437.57      -4694.07
  class=t0               200477       221436    -10479.50     -5305.94     -15785.44
  -- by lc_affected (L0) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lc_affected=no               268730       287263     -9266.50    -10172.00     -19438.50
  lc_affected=yes                   0         6939     -3469.50      2428.49      -1041.01
  -- by lane (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lane=0                131114       136645     -2765.50      5067.17       2301.67
  lane=1                136044       141719     -2837.50     -4231.56      -7069.06
  lane=2                  8969        15838     -3434.50       -36.26      -3470.76
  -- by segment (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  segment=downstream        53816        65335     -5759.50      -178.56      -5938.06
  segment=merge_edge         9876        13055     -1589.50      -361.26      -1950.76
  segment=rsu_window        62704        68014     -2655.00       485.34      -2169.66
  segment=upstream         149731       147798       966.50       853.84       1820.34
  -- by class (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  class=born              68284        72766     -2241.00       759.10      -1481.90
  class=t0               207843       221436     -6796.50        40.25      -6756.25
  -- by lc_affected (L3) --
  value             n_steps_pred n_steps_true expo_deficit ff_component delay_deficit
  lc_affected=no               268468       287263     -9397.50      1303.86      -8093.64
  lc_affected=yes                7659         6939       360.00      -504.50       -144.50

== D. top-10 cells by |delay_deficit| (validation, 60 s grid, per variant) ==
  -- L0 --
  cell                                                           delay_deficit n_pred n_true
  lane=1 seg=rsu_window cls=t0   lc=no                              -4179.30  27169  32557
  lane=1 seg=downstream cls=t0   lc=no                              -2904.26  29197  28841
  lane=2 seg=downstream cls=t0   lc=no                              -2182.13    927   5149
  lane=1 seg=merge_edge cls=t0   lc=no                              -2044.82   4747   7655
  lane=0 seg=upstream   cls=born lc=no                              -1368.45  29715  32244
  lane=1 seg=upstream   cls=t0   lc=no                              -1273.41  38810  39904
  lane=1 seg=upstream   cls=born lc=no                              -1270.83  31969  28159
  lane=2 seg=downstream cls=born lc=no                              -1120.68    689   2718
  lane=0 seg=downstream cls=t0   lc=no                               -839.75  22421  24526
  lane=0 seg=upstream   cls=t0   lc=no                               -837.28  44109  44996
  -- L3 --
  cell                                                           delay_deficit n_pred n_true
  lane=1 seg=rsu_window cls=t0   lc=no                              -2989.42  27401  32557
  lane=1 seg=downstream cls=t0   lc=no                              -2222.79  27869  28841
  lane=2 seg=downstream cls=t0   lc=no                              -2123.02   1037   5149
  lane=1 seg=merge_edge cls=t0   lc=no                              -1695.37   5199   7655
  lane=2 seg=downstream cls=born lc=no                              -1104.70    655   2718
  lane=0 seg=upstream   cls=t0   lc=no                                750.74  43872  44996
  lane=0 seg=rsu_window cls=t0   lc=no                                723.43  27628  28284
  lane=1 seg=downstream cls=t0   lc=yes                              -533.71    594   1721
  lane=0 seg=downstream cls=t0   lc=no                                472.09  22871  24526
  lane=1 seg=upstream   cls=t0   lc=no                                414.43  41473  39904

== E. speed profile, bins with |dv| > 1 m/s (matched vehicle-steps pooled over validation 60 s windows, binned by TRUTH (lane, 50 m d-bin) at tick j; dv = v_pred_mean - v_true_mean m/s) ==
  variant lane  d_bin  v_pred_mean v_true_mean      n     dv
  L0    0   -500       29.388       28.211   1669  +1.177
  L0    0   -450       29.200       27.717   2073  +1.483
  L0    0   -400       29.070       27.531   2233  +1.539
  L0    0   -350       29.010       27.483   2297  +1.527
  L0    0   -300       28.979       27.454   2258  +1.525
  L0    0   -250       28.930       27.455   2308  +1.475
  L0    0   -200       28.825       27.413   2277  +1.411
  L0    0   -150       28.785       27.409   2268  +1.376
  L0    0   -100       28.619       27.200   2700  +1.419
  L0    0    -50       28.918       27.420   2313  +1.498
  L0    0      0       29.161       27.606   2377  +1.555
  L0    0     50       29.192       27.762   2150  +1.430
  L0    0    100       29.273       27.892   2099  +1.381
  L0    0    150       29.326       28.108   2011  +1.218
  L0    0    200       29.330       28.198   2011  +1.132
  L0    0    250       29.336       28.267   1989  +1.069
  L0    0    300       29.334       28.284   1966  +1.050
  L0    0    350       29.317       28.247   1965  +1.070
  L0    0    400       29.334       28.209   1983  +1.126
  L0    0    450       29.406       28.240   1945  +1.166
  L0    0    500       29.425       28.248   1974  +1.177
  L0    0    550       29.411       28.282   1974  +1.129
  L0    0    600       29.408       28.322   1980  +1.086
  L0    0    650       29.409       28.324   1994  +1.085
  L0    0    700       29.381       28.298   2004  +1.083
  L0    0    750       29.378       28.293   2011  +1.085
  L0    0    800       29.392       28.285   1987  +1.107
  L0    0    850       29.378       28.292   2011  +1.086
  L0    0    900       29.377       28.249   2022  +1.128
  L0    0    950       29.388       28.247   1999  +1.141
  L0    0   1000       29.419       28.251   2009  +1.168
  L0    0   1050       29.409       28.205   2030  +1.204
  L0    0   1100       29.419       28.230   2029  +1.189
  L0    0   1150       29.412       28.260   1967  +1.153
  L0    0   1200       29.398       28.245   2002  +1.153
  L0    0   1250       29.387       28.251   2019  +1.136
  L0    0   1300       29.394       28.285   1974  +1.110
  L0    0   1350       29.383       28.273   2010  +1.110
  L0    0   1400       29.377       28.254   1993  +1.122
  L0    0   1450       29.389       28.277   1995  +1.113
  L0    0   1500       29.374       28.258   1973  +1.116
  L0    0   1550       29.387       28.255   1969  +1.132
  L0    0   1600       29.394       28.251   2003  +1.143
  L0    0   1650       29.413       28.254   1979  +1.158
  L0    0   1700       29.414       28.233   1995  +1.181
  L0    0   1750       29.443       28.230   2001  +1.213
  L0    0   1800       29.444       28.203   2011  +1.241
  L0    0   1850       29.444       28.186   1999  +1.258
  L0    0   1900       29.460       28.178   2008  +1.281
  L0    0   1950       29.449       28.126   2001  +1.323
  L0    0   2000       29.462       28.111   2001  +1.351
  L0    0   2050       29.496       28.085   2001  +1.411
  L0    0   2100       29.504       27.963   2014  +1.542
  L0    0   2150       29.468       27.836   2029  +1.632
  L0    0   2200       29.462       27.700   2021  +1.763
  L0    0   2250       29.395       27.482   2055  +1.913
  L0    0   2300       29.288       27.224   2079  +2.065
  L0    0   2350       29.084       26.827   2107  +2.257
  L0    0   2400       28.760       26.110   2143  +2.650
  L0    0   2450       28.047       24.864   2250  +3.183
  L0    0   2500       26.603       22.997   2406  +3.606
  L0    0   2550       23.072       19.351   2780  +3.721
  L0    0   2600       12.147        9.481   4431  +2.666
  L0    1   -550       29.747       28.423    409  +1.323
  L0    1   -500       29.530       27.257    738  +2.272
  L0    1   -450       29.254       26.697    954  +2.557
  L0    1   -400       29.119       26.374   1113  +2.745
  L0    1   -350       28.881       26.026   1173  +2.855
  L0    1   -300       28.535       25.552   1222  +2.982
  L0    1   -250       28.084       24.738   1242  +3.346
  L0    1   -200       27.332       23.454   1251  +3.878
  L0    1   -150       26.041       21.341   1353  +4.700
  L0    1   -100       23.406       11.565   2782  +11.841
  L0    1    -50       21.845        6.142   5299  +15.703
  L0    1      0       20.918        7.617   4919  +13.301
  L0    1     50       20.616       10.131   3906  +10.484
  L0    1    100       20.833       12.459   3365  +8.373
  L0    1    150       21.815       14.218   3118  +7.597
  L0    1    200       22.782       16.410   2755  +6.372
  L0    1    250       22.894       17.986   2552  +4.908
  L0    1    300       23.603       19.147   2425  +4.456
  L0    1    350       24.897       20.281   2296  +4.616
  L0    1    400       25.611       21.602   2187  +4.009
  L0    1    450       28.116       23.688   2013  +4.428
  L0    1    500       29.660       26.812   1775  +2.848
  L0    1    550       29.672       27.471   1728  +2.201
  L0    1    600       29.683       27.611   1713  +2.072
  L0    1    650       29.685       27.668   1716  +2.017
  L0    1    700       29.692       27.673   1691  +2.019
  L0    1    750       29.653       27.657   1724  +1.996
  L0    1    800       29.651       27.663   1737  +1.988
  L0    1    850       29.654       27.654   1708  +2.000
  L0    1    900       29.671       27.672   1727  +1.999
  L0    1    950       29.664       27.660   1719  +2.004
  L0    1   1000       29.661       27.644   1710  +2.017
  L0    1   1050       29.679       27.654   1703  +2.025
  L0    1   1100       29.665       27.635   1707  +2.030
  L0    1   1150       29.665       27.644   1741  +2.021
  L0    1   1200       29.653       27.641   1699  +2.011
  L0    1   1250       29.641       27.664   1726  +1.977
  L0    1   1300       29.632       27.644   1738  +1.987
  L0    1   1350       29.609       27.658   1707  +1.950
  L0    1   1400       29.606       27.660   1754  +1.946
  L0    1   1450       29.589       27.649   1733  +1.940
  L0    1   1500       29.572       27.642   1726  +1.931
  L0    1   1550       29.575       27.654   1747  +1.922
  L0    1   1600       29.562       27.632   1713  +1.929
  L0    1   1650       29.559       27.628   1735  +1.931
  L0    1   1700       29.560       27.621   1731  +1.939
  L0    1   1750       29.571       27.628   1719  +1.943
  L0    1   1800       29.539       27.602   1724  +1.937
  L0    1   1850       29.534       27.610   1715  +1.924
  L0    1   1900       29.533       27.595   1736  +1.938
  L0    1   1950       29.521       27.566   1717  +1.955
  L0    1   2000       29.506       27.519   1731  +1.987
  L0    1   2050       29.516       27.491   1740  +2.025
  L0    1   2100       29.512       27.436   1712  +2.076
  L0    1   2150       29.488       27.359   1728  +2.128
  L0    1   2200       29.432       27.232   1744  +2.200
  L0    1   2250       29.391       27.064   1753  +2.328
  L0    1   2300       29.274       26.822   1780  +2.452
  L0    1   2350       29.084       26.394   1807  +2.690
  L0    1   2400       28.669       25.586   1819  +3.083
  L0    1   2450       27.909       24.257   1959  +3.653
  L0    1   2500       26.241       22.134   2201  +4.107
  L0    1   2550       22.664       18.746   2683  +3.919
  L0    1   2600       12.339        9.773   4423  +2.566
  L0    2   -100        6.340        1.958    152  +4.382
  L0    2    -50       17.540        9.758    791  +7.782
  L0    2      0       21.420       16.390   1289  +5.030
  L0    2     50       21.637       19.802    996  +1.835
  L0    2    100       21.633       20.555   1002  +1.078
  L3    0   -600       26.217       29.677    107  -3.459
  L3    0   -550       26.063       28.907   1928  -2.844
  L3    0   -500       25.907       27.948   2199  -2.041
  L3    0   -450       25.704       27.628   2256  -1.923
  L3    0   -400       25.515       27.511   2275  -1.996
  L3    0   -350       25.314       27.483   2299  -2.168
  L3    0   -300       25.125       27.454   2258  -2.329
  L3    0   -250       24.874       27.452   2310  -2.579
  L3    0   -200       24.730       27.397   2283  -2.667
  L3    0   -150       24.629       27.383   2274  -2.753
  L3    0   -100       24.644       27.186   2706  -2.542
  L3    0    -50       25.140       27.355   2324  -2.214
  L3    0      0       25.697       27.606   2377  -1.909
  L3    0     50       26.022       27.762   2150  -1.740
  L3    0    100       26.258       27.892   2099  -1.634
  L3    0    150       26.341       28.108   2011  -1.767
  L3    0    200       26.464       28.198   2011  -1.735
  L3    0    250       26.562       28.267   1989  -1.705
  L3    0    300       26.649       28.284   1966  -1.635
  L3    0    350       26.692       28.247   1965  -1.555
  L3    0    400       26.757       28.209   1983  -1.451
  L3    0    450       26.859       28.240   1945  -1.381
  L3    0    500       26.927       28.248   1974  -1.321
  L3    0    550       26.969       28.282   1974  -1.313
  L3    0    600       26.969       28.322   1980  -1.353
  L3    0    650       26.947       28.324   1994  -1.378
  L3    0    700       26.941       28.298   2004  -1.356
  L3    0    750       26.943       28.293   2011  -1.349
  L3    0    800       26.966       28.285   1987  -1.319
  L3    0    850       26.954       28.292   2011  -1.338
  L3    0    900       26.958       28.249   2022  -1.292
  L3    0    950       26.968       28.247   1999  -1.278
  L3    0   1000       26.989       28.251   2009  -1.262
  L3    0   1050       27.000       28.205   2030  -1.205
  L3    0   1100       27.032       28.230   2029  -1.198
  L3    0   1150       27.021       28.260   1967  -1.238
  L3    0   1200       26.997       28.245   2002  -1.248
  L3    0   1250       26.983       28.251   2019  -1.267
  L3    0   1300       26.967       28.285   1974  -1.318
  L3    0   1350       26.937       28.273   2010  -1.336
  L3    0   1400       26.924       28.254   1993  -1.330
  L3    0   1450       26.920       28.277   1995  -1.357
  L3    0   1500       26.910       28.258   1973  -1.348
  L3    0   1550       26.886       28.255   1969  -1.369
  L3    0   1600       26.873       28.251   2003  -1.378
  L3    0   1650       26.813       28.254   1979  -1.442
  L3    0   1700       26.819       28.233   1995  -1.414
  L3    0   1750       26.840       28.230   2001  -1.390
  L3    0   1800       26.883       28.203   2011  -1.320
  L3    0   1850       26.885       28.186   1999  -1.301
  L3    0   1900       26.924       28.178   2008  -1.254
  L3    0   1950       26.890       28.126   2001  -1.236
  L3    0   2000       26.925       28.111   2001  -1.186
  L3    0   2050       26.907       28.085   2001  -1.178
  L3    0   2100       26.897       27.963   2014  -1.066
  L3    0   2150       26.803       27.836   2029  -1.033
  L3    0   2200       26.688       27.700   2021  -1.011
  L3    1   -600       27.019       29.025     42  -2.006
  L3    1   -550       27.106       28.119    941  -1.013
  L3    1   -200       24.450       23.357   1294  +1.093
  L3    1   -150       23.334       20.860   1460  +2.474
  L3    1   -100       20.880       11.024   3099  +9.856
  L3    1    -50       19.773        5.849   5906  +13.925
  L3    1      0       18.935        7.285   5197  +11.650
  L3    1     50       18.225        9.886   4016  +8.339
  L3    1    100       18.371       12.549   3334  +5.822
  L3    1    150       20.046       14.365   3076  +5.682
  L3    1    200       21.374       16.665   2705  +4.709
  L3    1    250       21.255       18.051   2541  +3.204
  L3    1    300       22.317       19.147   2425  +3.170
  L3    1    350       23.118       20.262   2299  +2.856
  L3    1    400       23.494       21.602   2187  +1.892
  L3    1    450       25.522       23.688   2013  +1.834
  L3    2   -100        6.571        2.359    182  +4.212
  L3    2    -50       16.836        9.457   1136  +7.379
  L3    2      0       19.456       16.117   1220  +3.339
  (208 bins with |dv| > 1 m/s)

== F. exit mismatch & entry waiting (validation, 60 s grid) ==
  L0: pred-exited-early: 1304 vehicles, truth remained 25634 steps, remaining truth delay=7399.80 s | truth-exited-early: 219 vehicles, pred remained 162 steps, remaining pred delay=26.86 s
  L0: entry waiting=0 steps over 1243 matched births (expected 0); pred-only births=0; truth-only births=0; t0 vehicles missing in pred=0 (expected 0); truth presence gaps=0
  L3: pred-exited-early: 713 vehicles, truth remained 20797 steps, remaining truth delay=6560.65 s | truth-exited-early: 736 vehicles, pred remained 2776 steps, remaining pred delay=317.58 s
  L3: entry waiting=0 steps over 1243 matched births (expected 0); pred-only births=0; truth-only births=0; t0 vehicles missing in pred=0 (expected 0); truth presence gaps=0

== G. train-root check (first 4 sorted train roots, 60 s grid) ==
[ 647.6s] train L0 cv17 t0=120: expo_def=-328.5 delay_def=-434.7
[ 657.8s] train L3 cv17 t0=120: expo_def=-235.5 delay_def=-14.6
[ 663.0s] train L0 cv17 t0=180: expo_def=-306.0 delay_def=-620.7
[ 671.8s] train L3 cv17 t0=180: expo_def=-148.0 delay_def=-149.1
[ 675.3s] train L0 cv17 t0=240: expo_def=-534.5 delay_def=-970.3
[ 682.7s] train L3 cv17 t0=240: expo_def=-353.0 delay_def=-435.9
[ 686.7s] train L0 cv17 t0=300: expo_def=-645.5 delay_def=-1177.7
[ 694.7s] train L3 cv17 t0=300: expo_def=-443.5 delay_def=-662.8
[ 699.7s] train L0 cv17 t0=360: expo_def=-465.0 delay_def=-976.6
[ 709.9s] train L3 cv17 t0=360: expo_def=-151.0 delay_def=-261.7
[ 715.9s] train L0 cv17 t0=420: expo_def=-579.0 delay_def=-1119.9
[ 723.7s] train L3 cv17 t0=420: expo_def=-362.5 delay_def=-599.1
[ 727.9s] train L0 cv17 t0=480: expo_def=-419.5 delay_def=-1029.5
[ 736.5s] train L3 cv17 t0=480: expo_def=-175.0 delay_def=-328.7
[ 740.0s] train L0 cv19_4107 t0=120: expo_def=-153.0 delay_def=-260.7
[ 745.4s] train L3 cv19_4107 t0=120: expo_def=-110.5 delay_def=0.8
[ 748.2s] train L0 cv19_4107 t0=180: expo_def=-265.5 delay_def=-306.2
[ 755.2s] train L3 cv19_4107 t0=180: expo_def=-197.0 delay_def=-36.4
[ 759.6s] train L0 cv19_4107 t0=240: expo_def=-168.0 delay_def=-279.8
[ 769.1s] train L3 cv19_4107 t0=240: expo_def=-94.5 delay_def=85.0
[ 773.9s] train L0 cv19_4107 t0=300: expo_def=-352.0 delay_def=-469.7
[ 781.9s] train L3 cv19_4107 t0=300: expo_def=-248.5 delay_def=-77.2
[ 785.1s] train L0 cv19_4107 t0=360: expo_def=-520.5 delay_def=-689.3
[ 790.8s] train L3 cv19_4107 t0=360: expo_def=-399.0 delay_def=-308.2
[ 793.7s] train L0 cv19_4107 t0=420: expo_def=-165.0 delay_def=-274.2
[ 799.2s] train L3 cv19_4107 t0=420: expo_def=-64.0 delay_def=59.2
[ 802.1s] train L0 cv19_4107 t0=480: expo_def=-156.5 delay_def=-361.4
[ 808.0s] train L3 cv19_4107 t0=480: expo_def=-56.0 delay_def=-31.2
[ 811.0s] train L0 cv19_4108 t0=120: expo_def=-232.0 delay_def=-314.7
[ 816.1s] train L3 cv19_4108 t0=120: expo_def=-149.0 delay_def=-50.9
[ 819.2s] train L0 cv19_4108 t0=180: expo_def=-127.5 delay_def=-175.7
[ 826.4s] train L3 cv19_4108 t0=180: expo_def=-54.0 delay_def=90.4
[ 830.5s] train L0 cv19_4108 t0=240: expo_def=-248.0 delay_def=-291.9
[ 837.4s] train L3 cv19_4108 t0=240: expo_def=-174.0 delay_def=11.6
[ 840.3s] train L0 cv19_4108 t0=300: expo_def=-172.5 delay_def=-234.2
[ 845.4s] train L3 cv19_4108 t0=300: expo_def=-109.0 delay_def=47.8
[ 848.0s] train L0 cv19_4108 t0=360: expo_def=-197.0 delay_def=-251.8
[ 853.2s] train L3 cv19_4108 t0=360: expo_def=-109.5 delay_def=54.2
[ 855.9s] train L0 cv19_4108 t0=420: expo_def=-226.5 delay_def=-329.2
[ 860.6s] train L3 cv19_4108 t0=420: expo_def=-143.0 delay_def=-81.1
[ 863.0s] train L0 cv19_4108 t0=480: expo_def=-209.5 delay_def=-281.1
[ 868.0s] train L3 cv19_4108 t0=480: expo_def=-169.5 delay_def=33.2
[ 871.4s] train L0 cv20_7001m2000 t0=120: expo_def=-237.5 delay_def=-312.7
[ 877.5s] train L3 cv20_7001m2000 t0=120: expo_def=-117.5 delay_def=21.3
[ 880.3s] train L0 cv20_7001m2000 t0=180: expo_def=-434.5 delay_def=-544.2
[ 888.3s] train L3 cv20_7001m2000 t0=180: expo_def=-365.5 delay_def=-225.7
[ 892.4s] train L0 cv20_7001m2000 t0=240: expo_def=-460.0 delay_def=-596.5
[ 898.9s] train L3 cv20_7001m2000 t0=240: expo_def=-298.0 delay_def=-260.5
[ 901.8s] train L0 cv20_7001m2000 t0=300: expo_def=-468.5 delay_def=-574.3
[ 907.3s] train L3 cv20_7001m2000 t0=300: expo_def=-382.5 delay_def=-247.3
[ 910.3s] train L0 cv20_7001m2000 t0=360: expo_def=-230.0 delay_def=-420.4
[ 916.0s] train L3 cv20_7001m2000 t0=360: expo_def=-112.0 delay_def=-22.3
[ 918.9s] train L0 cv20_7001m2000 t0=420: expo_def=-234.0 delay_def=-380.1
[ 924.6s] train L3 cv20_7001m2000 t0=420: expo_def=-157.0 delay_def=-41.4
[ 927.4s] train L0 cv20_7001m2000 t0=480: expo_def=-421.5 delay_def=-539.2
[ 933.0s] train L3 cv20_7001m2000 t0=480: expo_def=-323.0 delay_def=-215.0
  L0: mean expo_def=-319.91  mean delay_def=-507.74 veh s/window over 28 train windows
  L3: mean expo_def=-203.62  mean delay_def=-130.20 veh s/window over 28 train windows
[ 933.0s] wrote D:\rd-wt\wd01d\output\wd01\part3j\results\decomposition.csv (3021 data rows), D:\rd-wt\wd01d\output\wd01\part3j\results\speed_profile.csv (280 data rows), D:\rd-wt\wd01d\output\wd01\part3j\results\stdout_last.txt, REPORT_P3J.md

DONE part 3j decomposition.
```
