# WD-01 Part A v0 NATIVE dataset - DATA_CARD

Window [60, 600] s at the 0.5 s physics step on the WM-30 physical
lanes (0 main_left, 1 main_right, 2 accel; -1 junction with
in_junction=1).  Deduplicated to distinct traffic roots by window
content hash; split by root (holdout iff the root's runs are
holdout in the v3 build_tracks_manifest).  ZERO SIMULATION.

vType (`type_label`: 0 aggressive, 1 normal, 2 cautious) is
SUPERVISION ONLY - ideal sensing does not reveal vType - and is
excluded from every input-feature builder (enforced by
tests/test_inputs.py::test_vtype_not_in_features).
`is_cav` is derived from telemetry cav_report distinct ids (the
controller's polled fleet): CAVs alive shorter than one poll
interval (~5 s) could in principle be missed; designated counts
vs observed counts are in MANIFEST.json per root.

## Split-level counts

| split | roots | vehicle-steps | distinct vehicles | lane changes | births (in-window) | exits (arrived) | hard-brake /1000 | TTC<3s /1000 | TTC<1.5s /1000 | CAV step frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21 | 1934055 | 9260 | 4159 | 8369 | 7245 | 6.254 | 8.961 | 1.223 | 0.2067 |
| holdout | 5 | 539367 | 2344 | 815 | 2135 | 1731 | 7.52 | 10.434 | 1.075 | 0.2029 |
| overall | 26 | 2473422 | 11604 | 4974 | 10504 | 8976 | 6.53 | 9.283 | 1.191 | 0.2059 |

Lane changes by (from -> to), overall: 1253 (0->1), 2221 (1->0), 1500 (2->1 accel merge).  Births by entrance: 4375 mainline, 6129 ramp.
Junction vehicle-steps: 14259 (0.576% of all vehicle-steps).

## Per-root counts

| root | split | veh-steps | vehicles | LC 0->1 | LC 1->0 | LC 2->1 | births | exits | junction | hard-brake | TTC<3 | resid max (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cv17 | train | 105481 | 465 | 31 | 62 | 49 | 419 | 351 | 666 | 858 | 917 | 8.083 |
| cv19_4107 | train | 74294 | 397 | 72 | 123 | 56 | 365 | 326 | 342 | 257 | 223 | 8.101 |
| cv19_4108 | train | 65519 | 360 | 80 | 116 | 48 | 329 | 299 | 282 | 194 | 238 | 8.768 |
| cv20_7001m2000 | train | 77797 | 408 | 61 | 105 | 60 | 368 | 328 | 439 | 385 | 304 | 8.106 |
| cv20_7002m2000 | train | 70895 | 390 | 68 | 108 | 57 | 353 | 309 | 309 | 207 | 274 | 8.477 |
| cv20_7003m2000 | train | 80050 | 416 | 65 | 113 | 65 | 376 | 340 | 469 | 439 | 359 | 8.141 |
| cv20_7004m2000 | train | 81219 | 424 | 62 | 97 | 50 | 385 | 335 | 410 | 358 | 379 | 8.224 |
| cv20_7005m2000 | train | 81216 | 414 | 60 | 115 | 68 | 374 | 344 | 486 | 451 | 355 | 8.191 |
| cv20_7006m2000 | train | 68671 | 375 | 110 | 168 | 76 | 337 | 317 | 452 | 463 | 327 | 8.252 |
| cv20_7007m2700 | train | 114808 | 485 | 27 | 52 | 60 | 445 | 346 | 693 | 892 | 1639 | 8.214 |
| cv21_7008m2700 | train | 110602 | 485 | 19 | 46 | 45 | 433 | 369 | 780 | 843 | 1526 | 8.134 |
| cv23_7010m2700 | train | 103627 | 492 | 37 | 72 | 54 | 443 | 382 | 483 | 425 | 1602 | 8.641 |
| cv28_7101m2000 | holdout | 68721 | 369 | 68 | 115 | 46 | 341 | 296 | 317 | 189 | 231 | 8.140 |
| cv30_7302m2000 | train | 81837 | 422 | 51 | 100 | 52 | 385 | 336 | 451 | 497 | 467 | 8.210 |
| cv30_7303m2000 | train | 66555 | 387 | 89 | 141 | 71 | 353 | 308 | 330 | 249 | 328 | 8.178 |
| cv30_7304m2700 | train | 112657 | 503 | 23 | 55 | 60 | 446 | 374 | 684 | 866 | 1673 | 8.240 |
| cv30_7305m2700 | train | 114133 | 476 | 32 | 57 | 54 | 431 | 346 | 645 | 864 | 1580 | 8.080 |
| cv30_7306m2700 | train | 110953 | 505 | 22 | 55 | 50 | 458 | 401 | 664 | 774 | 799 | 8.322 |
| seed_7011m2700 | holdout | 113047 | 485 | 24 | 48 | 53 | 444 | 354 | 683 | 993 | 1734 | 8.124 |
| seed_7012m2700 | holdout | 124005 | 504 | 35 | 57 | 69 | 459 | 354 | 791 | 1059 | 1507 | 8.020 |
| seed_7301m2000 | train | 71727 | 388 | 77 | 125 | 67 | 343 | 327 | 382 | 322 | 302 | 8.293 |
| seed_7531m2700 | train | 118466 | 493 | 27 | 53 | 63 | 445 | 369 | 745 | 973 | 900 | 8.028 |
| seed_7532m2700 | train | 119709 | 499 | 19 | 51 | 62 | 448 | 378 | 783 | 1058 | 1731 | 8.181 |
| seed_7534m2700 | holdout | 131073 | 514 | 19 | 49 | 65 | 465 | 367 | 800 | 1027 | 1469 | 8.086 |
| seed_7535m2400 | train | 103839 | 476 | 33 | 63 | 50 | 433 | 360 | 569 | 720 | 1409 | 8.182 |
| seed_7536m2400 | holdout | 102521 | 472 | 42 | 75 | 50 | 426 | 360 | 604 | 788 | 687 | 8.130 |

s-advance residual (s[t+1]-s[t] - v*0.5, same-lane non-junction steps): per-root |resid| p99/max in the table above; full percentiles in results/per_root_counts.csv and MANIFEST.json.

machine-readable copies: results/data_card.json, results/per_root_counts.csv, MANIFEST.json


## ERRATUM (2026-09-26, part 3)

The TTC columns of this card (ttc_lt3/ttc_lt1p5 per 1000, computed by
build.py::ttc_rate) used an inverted front/behind comparison: they counted
small-gap pairs where the FRONT vehicle pulls away, not follower TTC.
Corrected follower->leader exposures (computed directly from the npz
files, MANIFEST and arrays untouched) are in results/ttc_erratum.json:
overall TTC<3 s = 2.31 per 1000 vehicle-steps, TTC<1.5 s = 0.0 per 1000
(at the 0.5 s sampling the true TTC<1.5 s rate is below one event per
~800 vehicle-steps).  All other counts in this card are unaffected.
build.py::ttc_rate itself is corrected as of this erratum for future
re-runs (results committed with part 3).
