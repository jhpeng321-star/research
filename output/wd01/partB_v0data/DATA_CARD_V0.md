# WD-01 Part B round 0 - V0 (native-policy value) dataset v0.2 - DATA_CARD

CONSOLIDATED v0.2: chunk 1 (WM-44, pre-fix fleet) copied from the
wd01n worktree with every file sha256-verified against the old
MANIFEST before and after the copy, plus chunk 2 (WM-47, post-fix
fleet) newly built with the chunk-agnostic builder.  ZERO
SIMULATION: only the campaign run logs, the plans and the frozen
network XML are read.  HOLDOUT worlds are never opened (listed in
MANIFEST skipped_holdout; guarded in code).

Labels: contract §4.1 total delay over [t, t+tau] (Monte-Carlo on
the logged trajectory, part4 delay library conventions,
equivalence tested), tau in {30, 60, 120, 300, 570} s, complete
horizons only (tau_available recorded; truncated horizons
dropped).  Decision times every 10 s: base source over [120,
t0+570], native source over (t0, t0+570].  TD tuples (s_t,
D[t,t+30], s_{t+30}, tau) at commit 30 s.  Every row carries
chunk_id, cav_speed_mode, pre_fix = (cav_speed_mode == 31), p40 =
(penetration == 0.4), plus TTC<3 exposure (veh·s), hard-brake
seconds (a < -3 m/s²) and collision/teleport flags over the same
window (V_exp / V_brake / P_coll anchors).

## Provenance and caveats (read before training)

1. **Two sources per world.** `source=base` is the campaign
   incumbent arm: fleet-uniform **damping** (WM-44/WM-47
   DATA_CONTRACT; no controller internals logged).
   `source=native` is the native-IDM branch.  DESIGN PartB v2
   §4 Step 0 includes both; `policy` and `source` let downstream
   training filter (e.g. native-only).
2. **CRN overlap.** The native branch restores the base state at
   t0: post-t0 base and native windows are strongly overlapping
   realizations of the same traffic.  Treat cross-source samples
   as correlated, not independent.
3. **Two fleets.** Chunk 1 is PRE-FIX: cav_speed_mode = 31 on
   every chunk-1 row.  Chunk 2 is POST-FIX: cav_speed_mode = 27
   on every chunk-2 row (WM-47 INDEX; per-run resolution
   receipt -> observed physics values -> per-chunk config
   default).  Do not mix fleets without the pre_fix flag.
4. **Entry waiting dominates at high demand.** Labels include
   entry waiting from the scheduled departures (contract §4.1);
   at m2600-2700 the mainline queue is the main delay term.
5. HOLDOUT worlds are NOT built and their run files are NEVER
   opened (chunk 1: 14 listed, chunk 2: 14 listed).  Worlds with
   incomplete receipts are listed in skipped_incomplete and
   built on their available sources only.
6. vType is supervision-only and is not featurised (tested).

## Split x chunk x source counts

| chunk | split | worlds | sources | decision times | label rows | TD rows |
| --- | --- | --- | --- | --- | --- | --- |
| chunk1 | TRAIN | 49 | 96 | 6841 | 25277 | 34205 |
| chunk1 | CALIBRATION | 7 | 14 | 955 | 3473 | 4775 |
| chunk2 | TRAIN | 49 | 98 | 6877 | 25271 | 34385 |
| chunk2 | CALIBRATION | 7 | 14 | 1027 | 3833 | 5135 |

## Label counts by tau (complete-horizon rows)

| chunk | source | split | tau=30 | tau=60 | tau=120 | tau=300 | tau=570 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| chunk1 | base | TRAIN | 4162 | 4015 | 3721 | 2839 | 1516 | 16253 |
| chunk1 | base | CALIBRATION | 556 | 535 | 493 | 367 | 178 | 2129 |
| chunk1 | native | TRAIN | 2679 | 2538 | 2256 | 1410 | 141 | 9024 |
| chunk1 | native | CALIBRATION | 399 | 378 | 336 | 210 | 21 | 1344 |
| chunk2 | base | TRAIN | 4084 | 3937 | 3643 | 2761 | 1438 | 15863 |
| chunk2 | base | CALIBRATION | 628 | 607 | 565 | 439 | 250 | 2489 |
| chunk2 | native | TRAIN | 2793 | 2646 | 2352 | 1470 | 147 | 9408 |
| chunk2 | native | CALIBRATION | 399 | 378 | 336 | 210 | 21 | 1344 |

### Native tau=570 label rows on TRAIN (headroom check, flag if < 150)

- chunk chunk1: **141** native TRAIN rows at tau=570  **FLAG: < 150**
- chunk chunk2: **147** native TRAIN rows at tau=570  **FLAG: < 150**

## cav_speed_mode resolution per chunk (receipt -> observed -> config fallback)

| chunk | receipt | observed | config (fallback) | n/a (pre-chunk entries) |
| --- | --- | --- | --- | --- |
| chunk1 | 0 | 110 | 0 | 0 |
| chunk2 | 0 | 0 | 112 | 0 |

Chunk 2 records cav_speed_mode = 27 in the run receipts'
config only as null; the physics rows carry mixed per-row
values (31 pre-restore history + 27 post-fix fleet), so the
per-chunk config default 27 (WM-47 INDEX cav_speed_mode) is
the resolved value on every chunk-2 source.  Chunk-1
entries carry 'observed' from the flag backfill (single
physics-row value 31 = the pre-fix fleet mode).

## Counts per demand and penetration (label rows)

| chunk | split | source | demand (veh/h) | penetration | worlds | label rows | mean delay (veh·s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chunk1 | CALIBRATION | base | 2400 | 0.05 | 1 | 257 | 7488.1 |
| chunk1 | CALIBRATION | base | 2600 | 0.1 | 1 | 347 | 10490.9 |
| chunk1 | CALIBRATION | base | 2600 | 0.2 | 1 | 287 | 12017.0 |
| chunk1 | CALIBRATION | base | 2700 | 0.02 | 1 | 377 | 9086.0 |
| chunk1 | CALIBRATION | base | 2700 | 0.2 | 1 | 317 | 9818.5 |
| chunk1 | CALIBRATION | base | 2700 | 0.4 | 2 | 544 | 10752.5 |
| chunk1 | CALIBRATION | native | 2400 | 0.05 | 1 | 192 | 5409.9 |
| chunk1 | CALIBRATION | native | 2600 | 0.1 | 1 | 192 | 9003.6 |
| chunk1 | CALIBRATION | native | 2600 | 0.2 | 1 | 192 | 10097.8 |
| chunk1 | CALIBRATION | native | 2700 | 0.02 | 1 | 192 | 6996.6 |
| chunk1 | CALIBRATION | native | 2700 | 0.2 | 1 | 192 | 7277.5 |
| chunk1 | CALIBRATION | native | 2700 | 0.4 | 2 | 384 | 9059.1 |
| chunk1 | TRAIN | base | 2000 | 0.02 | 3 | 951 | 2133.1 |
| chunk1 | TRAIN | base | 2000 | 0.4 | 2 | 574 | 2946.4 |
| chunk1 | TRAIN | base | 2400 | 0.05 | 1 | 407 | 5697.5 |
| chunk1 | TRAIN | base | 2400 | 0.1 | 1 | 287 | 4842.4 |
| chunk1 | TRAIN | base | 2400 | 0.4 | 1 | 347 | 8706.0 |
| chunk1 | TRAIN | base | 2600 | 0.02 | 3 | 891 | 6353.3 |
| chunk1 | TRAIN | base | 2600 | 0.05 | 3 | 951 | 7735.1 |
| chunk1 | TRAIN | base | 2600 | 0.1 | 4 | 1508 | 7861.2 |
| chunk1 | TRAIN | base | 2600 | 0.2 | 2 | 634 | 6391.5 |
| chunk1 | TRAIN | base | 2600 | 0.4 | 4 | 1238 | 10141.2 |
| chunk1 | TRAIN | base | 2700 | 0.02 | 4 | 1388 | 9067.8 |
| chunk1 | TRAIN | base | 2700 | 0.05 | 6 | 2022 | 9569.1 |
| chunk1 | TRAIN | base | 2700 | 0.1 | 5 | 1555 | 9892.0 |
| chunk1 | TRAIN | base | 2700 | 0.2 | 7 | 2399 | 11330.5 |
| chunk1 | TRAIN | base | 2700 | 0.4 | 3 | 1101 | 16133.0 |
| chunk1 | TRAIN | native | 2000 | 0.02 | 3 | 576 | 1535.8 |
| chunk1 | TRAIN | native | 2000 | 0.4 | 2 | 384 | 1642.4 |
| chunk1 | TRAIN | native | 2400 | 0.05 | 1 | 192 | 5092.0 |
| chunk1 | TRAIN | native | 2400 | 0.1 | 1 | 192 | 2514.2 |
| chunk1 | TRAIN | native | 2600 | 0.02 | 3 | 576 | 4337.6 |
| chunk1 | TRAIN | native | 2600 | 0.05 | 3 | 576 | 6863.4 |
| chunk1 | TRAIN | native | 2600 | 0.1 | 4 | 768 | 6769.4 |
| chunk1 | TRAIN | native | 2600 | 0.2 | 2 | 384 | 6344.3 |
| chunk1 | TRAIN | native | 2600 | 0.4 | 4 | 768 | 7545.1 |
| chunk1 | TRAIN | native | 2700 | 0.02 | 4 | 768 | 7516.1 |
| chunk1 | TRAIN | native | 2700 | 0.05 | 6 | 1152 | 8659.9 |
| chunk1 | TRAIN | native | 2700 | 0.1 | 5 | 960 | 8874.7 |
| chunk1 | TRAIN | native | 2700 | 0.2 | 7 | 1344 | 9100.8 |
| chunk1 | TRAIN | native | 2700 | 0.4 | 2 | 384 | 10235.1 |
| chunk2 | CALIBRATION | base | 2000 | 0.2 | 1 | 407 | 3082.9 |
| chunk2 | CALIBRATION | base | 2400 | 0.4 | 2 | 694 | 7400.2 |
| chunk2 | CALIBRATION | base | 2600 | 0.02 | 1 | 287 | 7964.1 |
| chunk2 | CALIBRATION | base | 2600 | 0.05 | 1 | 407 | 10021.7 |
| chunk2 | CALIBRATION | base | 2600 | 0.2 | 1 | 407 | 13217.8 |
| chunk2 | CALIBRATION | base | 2700 | 0.1 | 1 | 287 | 9086.3 |
| chunk2 | CALIBRATION | native | 2000 | 0.2 | 1 | 192 | 1206.8 |
| chunk2 | CALIBRATION | native | 2400 | 0.4 | 2 | 384 | 5784.8 |
| chunk2 | CALIBRATION | native | 2600 | 0.02 | 1 | 192 | 7637.8 |
| chunk2 | CALIBRATION | native | 2600 | 0.05 | 1 | 192 | 7740.8 |
| chunk2 | CALIBRATION | native | 2600 | 0.2 | 1 | 192 | 10021.4 |
| chunk2 | CALIBRATION | native | 2700 | 0.1 | 1 | 192 | 6956.7 |
| chunk2 | TRAIN | base | 2000 | 0.02 | 1 | 287 | 1436.6 |
| chunk2 | TRAIN | base | 2000 | 0.1 | 1 | 257 | 1199.3 |
| chunk2 | TRAIN | base | 2000 | 0.4 | 1 | 407 | 4705.7 |
| chunk2 | TRAIN | base | 2400 | 0.02 | 2 | 694 | 6650.3 |
| chunk2 | TRAIN | base | 2400 | 0.05 | 1 | 347 | 4381.4 |
| chunk2 | TRAIN | base | 2400 | 0.1 | 1 | 347 | 10878.8 |
| chunk2 | TRAIN | base | 2400 | 0.2 | 1 | 347 | 3336.1 |
| chunk2 | TRAIN | base | 2600 | 0.02 | 1 | 257 | 4872.5 |
| chunk2 | TRAIN | base | 2600 | 0.05 | 3 | 1041 | 6984.4 |
| chunk2 | TRAIN | base | 2600 | 0.1 | 3 | 891 | 7372.1 |
| chunk2 | TRAIN | base | 2600 | 0.2 | 4 | 1418 | 8107.6 |
| chunk2 | TRAIN | base | 2600 | 0.4 | 2 | 574 | 8716.1 |
| chunk2 | TRAIN | base | 2700 | 0.02 | 6 | 2052 | 9353.0 |
| chunk2 | TRAIN | base | 2700 | 0.05 | 6 | 1812 | 8513.1 |
| chunk2 | TRAIN | base | 2700 | 0.1 | 5 | 1885 | 11821.9 |
| chunk2 | TRAIN | base | 2700 | 0.2 | 4 | 1118 | 10114.4 |
| chunk2 | TRAIN | base | 2700 | 0.4 | 7 | 2129 | 10405.2 |
| chunk2 | TRAIN | native | 2000 | 0.02 | 1 | 192 | 1106.5 |
| chunk2 | TRAIN | native | 2000 | 0.1 | 1 | 192 | 858.9 |
| chunk2 | TRAIN | native | 2000 | 0.4 | 1 | 192 | 2713.7 |
| chunk2 | TRAIN | native | 2400 | 0.02 | 2 | 384 | 5909.3 |
| chunk2 | TRAIN | native | 2400 | 0.05 | 1 | 192 | 2628.4 |
| chunk2 | TRAIN | native | 2400 | 0.1 | 1 | 192 | 7128.9 |
| chunk2 | TRAIN | native | 2400 | 0.2 | 1 | 192 | 1995.7 |
| chunk2 | TRAIN | native | 2600 | 0.02 | 1 | 192 | 4446.1 |
| chunk2 | TRAIN | native | 2600 | 0.05 | 3 | 576 | 6609.3 |
| chunk2 | TRAIN | native | 2600 | 0.1 | 3 | 576 | 5668.5 |
| chunk2 | TRAIN | native | 2600 | 0.2 | 4 | 768 | 5914.9 |
| chunk2 | TRAIN | native | 2600 | 0.4 | 2 | 384 | 6130.7 |
| chunk2 | TRAIN | native | 2700 | 0.02 | 6 | 1152 | 8174.1 |
| chunk2 | TRAIN | native | 2700 | 0.05 | 6 | 1152 | 6775.0 |
| chunk2 | TRAIN | native | 2700 | 0.1 | 5 | 960 | 10385.8 |
| chunk2 | TRAIN | native | 2700 | 0.2 | 4 | 768 | 8904.5 |
| chunk2 | TRAIN | native | 2700 | 0.4 | 7 | 1344 | 8152.3 |

## Chunk and fix-flag counts

| chunk | sources | label rows | TD rows | pre_fix rows (cav_speed_mode==31) | p40 rows (penetration==0.4) | cav_speed_mode values |
| --- | --- | --- | --- | --- | --- | --- |
| chunk1 | 110 | 28750 | 38980 | 28750 | 5724 | [31] |
| chunk2 | 112 | 29104 | 39520 | 0 | 6108 | [27] |
pre_fix = (cav_speed_mode == 31): chunk-1 rows are all
pre-fix; chunk-2 rows are all post-fix (27).  p40 flags the
40% penetration block.

## Label distributions per tau (total delay, veh·s)

| chunk | source | split | tau (s) | n | mean | std | p50 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chunk1 | base | CALIBRATION | 30 | 556 | 1993.5 | 936.5 | 2035.3 | 3076.4 | 4911.1 |
| chunk1 | base | CALIBRATION | 60 | 535 | 3998.6 | 1824.3 | 4072.5 | 6093.5 | 9720.8 |
| chunk1 | base | CALIBRATION | 120 | 493 | 8044.3 | 3449.8 | 8109.5 | 12187.4 | 19010.2 |
| chunk1 | base | CALIBRATION | 300 | 367 | 20316.3 | 7070.8 | 20297.5 | 29774.4 | 41700.6 |
| chunk1 | base | CALIBRATION | 570 | 178 | 37816.9 | 9386.9 | 36120.7 | 50910.5 | 66114.4 |
| chunk1 | base | TRAIN | 30 | 4162 | 1680.2 | 1092.8 | 1435.7 | 3165.2 | 6887.2 |
| chunk1 | base | TRAIN | 60 | 4015 | 3366.5 | 2150.9 | 2901.9 | 6267.9 | 13658.8 |
| chunk1 | base | TRAIN | 120 | 3721 | 6755.2 | 4168.9 | 5849.5 | 12499.3 | 26581.7 |
| chunk1 | base | TRAIN | 300 | 2839 | 17085.0 | 9530.5 | 15146.7 | 30059.0 | 59914.1 |
| chunk1 | base | TRAIN | 570 | 1516 | 33007.5 | 16363.8 | 30556.7 | 54460.4 | 95916.5 |
| chunk1 | native | CALIBRATION | 30 | 399 | 2182.5 | 781.5 | 2028.3 | 3335.5 | 4417.9 |
| chunk1 | native | CALIBRATION | 60 | 378 | 4367.0 | 1511.8 | 4054.6 | 6615.5 | 8725.9 |
| chunk1 | native | CALIBRATION | 120 | 336 | 8745.8 | 2827.4 | 8117.6 | 12701.3 | 17078.8 |
| chunk1 | native | CALIBRATION | 300 | 210 | 21878.9 | 6008.9 | 19829.2 | 31151.6 | 38121.1 |
| chunk1 | native | CALIBRATION | 570 | 21 | 41466.8 | 11059.3 | 37967.3 | 58836.1 | 60676.7 |
| chunk1 | native | TRAIN | 30 | 2679 | 1890.0 | 1127.5 | 1741.1 | 3292.2 | 6348.1 |
| chunk1 | native | TRAIN | 60 | 2538 | 3778.7 | 2237.1 | 3515.9 | 6561.7 | 12660.2 |
| chunk1 | native | TRAIN | 120 | 2256 | 7549.3 | 4411.7 | 7150.7 | 12964.5 | 25135.4 |
| chunk1 | native | TRAIN | 300 | 1410 | 18822.1 | 10639.4 | 18182.8 | 31339.7 | 60092.4 |
| chunk1 | native | TRAIN | 570 | 141 | 35910.5 | 19284.8 | 35328.1 | 58289.4 | 99998.1 |
| chunk2 | base | CALIBRATION | 30 | 628 | 1565.8 | 882.1 | 1645.3 | 2779.4 | 3622.6 |
| chunk2 | base | CALIBRATION | 60 | 607 | 3135.8 | 1726.0 | 3333.9 | 5519.0 | 7164.9 |
| chunk2 | base | CALIBRATION | 120 | 565 | 6277.7 | 3294.3 | 6673.7 | 10595.3 | 13850.3 |
| chunk2 | base | CALIBRATION | 300 | 439 | 15685.1 | 7356.0 | 16535.2 | 25725.7 | 31520.2 |
| chunk2 | base | CALIBRATION | 570 | 250 | 29692.1 | 12471.6 | 31507.6 | 45150.7 | 54576.4 |
| chunk2 | base | TRAIN | 30 | 4084 | 1660.7 | 961.2 | 1511.9 | 2974.5 | 5761.3 |
| chunk2 | base | TRAIN | 60 | 3937 | 3321.3 | 1878.7 | 3028.8 | 5880.0 | 11154.2 |
| chunk2 | base | TRAIN | 120 | 3643 | 6643.1 | 3585.0 | 6096.9 | 11537.3 | 21124.4 |
| chunk2 | base | TRAIN | 300 | 2761 | 16608.6 | 7773.6 | 15743.0 | 27059.0 | 48777.9 |
| chunk2 | base | TRAIN | 570 | 1438 | 31809.9 | 12186.6 | 31628.4 | 47083.9 | 74275.8 |
| chunk2 | native | CALIBRATION | 30 | 399 | 1726.6 | 882.8 | 1872.5 | 2840.5 | 3370.6 |
| chunk2 | native | CALIBRATION | 60 | 378 | 3462.0 | 1744.5 | 3774.9 | 5696.4 | 6638.9 |
| chunk2 | native | CALIBRATION | 120 | 336 | 6936.6 | 3384.9 | 7591.3 | 11153.2 | 12983.2 |
| chunk2 | native | CALIBRATION | 300 | 210 | 17373.5 | 7854.7 | 19100.2 | 26476.2 | 29667.9 |
| chunk2 | native | CALIBRATION | 570 | 21 | 32804.7 | 14883.9 | 39160.4 | 51542.4 | 52158.7 |
| chunk2 | native | TRAIN | 30 | 2793 | 1873.1 | 890.3 | 1797.7 | 3023.3 | 4876.0 |
| chunk2 | native | TRAIN | 60 | 2646 | 3741.1 | 1751.3 | 3601.2 | 6029.3 | 9499.7 |
| chunk2 | native | TRAIN | 120 | 2352 | 7463.9 | 3392.2 | 7267.1 | 11967.2 | 17739.3 |
| chunk2 | native | TRAIN | 300 | 1470 | 18583.0 | 7762.0 | 18834.8 | 28683.6 | 40329.3 |
| chunk2 | native | TRAIN | 570 | 147 | 35588.0 | 13684.8 | 36473.7 | 51502.5 | 64551.2 |

## TD-tuple consistency (label[t,tau] = D30 + label[t+30, tau-30])

Checked 42154 complete tuples across 222 sources; violations beyond tol 1e-06: **0**; worst |diff| 5.821e-11 veh·s (identity holds to float noise).
Per-source numbers in results/td_consistency.json.

## Window exposure per tau (all sources, sums)

| tau (s) | rows | TTC<3 exposure (veh·s) | hard brake (s) | rows w/ collision |
| --- | --- | --- | --- | --- |
| 30 | 15700 | 116838.0 | 435470.5 | 42 |
| 60 | 15034 | 224510.0 | 837596.0 | 78 |
| 120 | 13702 | 411365.5 | 1537936.5 | 136 |
| 300 | 9706 | 735676.5 | 2762536.5 | 244 |
| 570 | 3712 | 528434.0 | 2011342.0 | 196 |

## Build quality

- physics rows off the 0.5 s grid: 0 (all counted, none silently dropped)
- unmapped/unresolvable lane records (s would be NaN): 0
- lane-change-at-boundary steps: 1257; implausible advances: 0
- realised insertions cross-checked vs insertions.jsonl: 115134 exact, 0 mismatches

## Skipped

- [chunk1] HOLDOUT worlds (never built or opened): w8407m2400p02t360, w8408m2600p02t300, w8409m2600p02t360, w8421m2600p05t420, w8422m2700p05t240, w8423m2400p05t300, w8435m2700p10t240, w8436m2700p10t240, w8437m2600p10t360, w8449m2400p20t240, w8450m2700p20t300, w8451m2700p20t420, w8463m2700p40t420, w8464m2600p40t480
- [chunk1] incomplete receipts (source skipped): w8469m2400p40t420__native (RUN_RECEIPT.json missing); w8473m2700p40t540__native (RUN_RECEIPT.json missing)
- [chunk2] HOLDOUT worlds (never built or opened): w8477m2600p02t480, w8478m2700p02t360, w8479m2700p02t240, w8491m2700p05t420, w8492m2600p05t360, w8493m2600p05t240, w8505m2700p10t300, w8506m2600p10t240, w8507m2000p10t360, w8519m2600p20t240, w8520m2700p20t360, w8521m2700p20t240, w8533m2600p40t420, w8534m2600p40t480

Machine-readable copies: MANIFEST.json, results/td_consistency.json, results/sanity_v02.json, results/verify_v0data.json.
