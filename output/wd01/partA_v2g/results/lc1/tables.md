# WD-01-PA2G-LC1 decomposition tables (machine-generated)

## Headline v2g LC counts (post-LC1-fixes eval)

| model|H | n | merge p/t | disc p/t |
|---|---|---|---|
| v2g|H10 | 576 | 398/607 | 2034/442 |
| v2g|H120 | 512 | 1951/2120 | 7247/1389 |
| v2g|H30 | 576 | 647/1749 | 6226/1132 |
| v2g|H60 | 576 | 1737/2320 | 8364/1397 |

## Lane pair, events in (w, w+120], tracked cohort

| class | truth | pred | pred/truth |
|---|---|---|---|
| merge | 2406 | 2213 | 0.92x |
| disc_left | 1012 | 6791 | 6.71x |
| disc_right | 510 | 1878 | 3.68x |

## Merge position, 10 m bins on the merge edge [378.66, 454.10) m

| s bin (m) | truth | pred |
|---|---|---|
| 379-389 | 62 | 59 |
| 389-399 | 178 | 21 |
| 399-409 | 152 | 39 |
| 409-419 | 178 | 95 |
| 419-429 | 236 | 134 |
| 429-439 | 324 | 437 |
| 439-449 | 510 | 510 |
| 449-459 | 618 | 790 |

## Discretionary events, 100 m s-bins (from-lane s)

| s bin (m) | truth | pred |
|---|---|---|
| 1846-1946 | 4 | 160 |
| 1946-2046 | 22 | 254 |
| 2046-2146 | 41 | 361 |
| 2146-2246 | 55 | 570 |
| 2246-2346 | 88 | 847 |
| 2346-2446 | 123 | 1157 |
| 2446-2546 | 181 | 1493 |
| 2546-2646 | 242 | 1340 |
| 2646-2746 | 224 | 1425 |
| 2746-2846 | 252 | 522 |
| 2846-2946 | 15 | 124 |

## Local regime at the event (same-lane neighbours within +-50 m | their mean speed)

| regime | truth | pred |
|---|---|---|
| n0|v_undef | 764 | 2468 |
| n1-2|v10-20 | 192 | 635 |
| n1-2|v<10 | 25 | 892 |
| n1-2|v>20 | 310 | 678 |
| n3-5|v10-20 | 35 | 39 |
| n3-5|v<10 | 72 | 1824 |
| n6+|v<10 | 124 | 2327 |

## CAV vs HDV

| side | truth events | pred events | truth /1000 v-s | pred /1000 v-s |
|---|---|---|---|---|
| cav | 583 | 1725 | 0.562 | 1.663 |
| hdv | 3345 | 9351 | 0.585 | 1.634 |

## Time since window start

| bin (s) | truth | pred |
|---|---|---|
| 0-10 | 1049 | 2436 |
| 10-30 | 1832 | 4451 |
| 30-60 | 836 | 3329 |
| 60-120 | 211 | 860 |

## Repeats per vehicle (120 s rollout)

| metric | truth | pred |
|---|---|---|
| vehicles with 1 events | 3690 | 8423 |
| vehicles with 2-3 events | 119 | 1161 |
| vehicles with 4-9 events | 0 | 42 |
| vehicles with a change | 3809 | 9626 |
| changed back (opp. disc) within 10 s | 69 (0.018) | 544 (0.057) |

## Merge outcomes, accel-lane vehicles at w

| metric | truth | pred |
|---|---|---|
| n | 1163 | 1121 |
| merged | 1024 | 1097 |
| stopped | 48 | 0 |
| never_H120 | 139 | 24 |
| dwell_from_w_mean | 16.89 | 34.18 |
| dwell_from_w_median | 12.0 | 34.5 |
| dwell_from_edge_entry_mean | 16.89 | None |
| dwell_from_edge_entry_median | 12.0 | None |

## Merge position stats

| side | n | mean s (m) | share below 415 m |
|---|---|---|---|
| truth | 2406 | 432.31 | 0.224 |
| pred | 2213 | 440.32 | 0.075 |

## Worst windows by disc surplus (H120 slice)

| world | w | disc pred | disc true | surplus | merge p/t |
|---|---|---|---|---|---|
| w8694m2600p40t240 | 690.0 | 49 | 2 | 47 | 2/2 |
| w8651m2400p05t420 | 900.0 | 46 | 2 | 44 | 7/8 |
| w8694m2600p40t240 | 630.0 | 42 | 1 | 41 | 9/10 |
| w8623m2600p02t540 | 990.0 | 40 | 0 | 40 | 4/4 |
| w8623m2600p02t540 | 930.0 | 39 | 0 | 39 | 5/5 |
| w8623m2600p02t540 | 870.0 | 40 | 2 | 38 | 5/5 |
| w8676m2700p20t360 | 810.0 | 39 | 1 | 38 | 5/5 |
| w8680m2600p20t300 | 690.0 | 38 | 1 | 37 | 7/8 |
| w8693m2600p40t420 | 870.0 | 38 | 1 | 37 | 4/5 |
| w8676m2700p20t360 | 690.0 | 36 | 0 | 36 | 10/11 |
| w8686m2600p20t240 | 720.0 | 37 | 2 | 35 | 7/7 |
| w8694m2600p40t240 | 600.0 | 38 | 3 | 35 | 12/12 |

## Rejected decision tallies (pred, all windows)

| kind|reason | total |
|---|---|
| disc|b_safe | 1431099 |
| disc|threshold | 989767 |
| disc|gap_floor | 311110 |
| merge|threshold | 76507 |
| merge|foll_bsafe | 53008 |
| merge|lead_headway | 28761 |
| merge|foll_s0 | 14345 |
| merge|lead_bsafe | 2 |

## Descriptive ablations (LABELLED: NOT ADOPTED, NOT SELECTED; params_nominal.json untouched)

| variant | disc/window (H120) | merge/window (H120) | disc change-back rate | s_RMSE H10 | s_RMSE H120 | delay_err H10 |
|---|---|---|---|---|---|---|
| nominal | 14.154 | 3.811 | 0.057 | 9.98 | 137.91 | 3.38 |
| politeness_0.2 | 40.537 | 3.9 | 0.417 | 10.09 | 134.39 | 3.62 |
| politeness_1.0 | 13.943 | 3.777 | 0.038 | 11.31 | 139.02 | 4.85 |
| bias0_symmetric | 23.496 | 3.678 | 0.288 | 9.99 | 137.88 | 3.34 |
| threshold_0.3 | 11.252 | 3.859 | 0.029 | 9.97 | 138.39 | 3.43 |
| urgency_late | 14.08 | 3.811 | 0.056 | 9.96 | 137.98 | 3.37 |

Delay-scope note (owner decision (c)): Part A is tracked-scope; Part B carries untracked arrivals.
