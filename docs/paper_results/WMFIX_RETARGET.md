# WMFIX retarget decision — zero-head entry path on the corrected validation panel
(addendum 03:55 item 4; addendum 05:55 gate (c)) — FINAL, 2026-10-31 ~09:10.
ZERO_HEAD_COMPARE.json sha256 c5210190369d58d957f82cc53cf15dc17aed5fb962c2d8dcfb5d833e7fd84301.

## Design of the comparison
- Entry path: conserve_queue + learned admission ON, merge-response head at zero
  (WM2 weights b8797242, no head), K8, native lanes, mix ON, fixed source clock.
  36 groups x 32 rows complete + DONE.json. gpuA PID 100400 (groups 0-27, 32-35);
  gpuR shard PIDs 482999/484326 (groups 28-31, identical code/inputs/CRN,
  WMFIX_SHARD_STAGING.md). Mixed-provenance validated: groups 33/34 computed on
  BOTH boxes -> max single-cell |dJ| 225/272 veh-s but group-level draw-medians
  identical (rel diff 0.0); every comparator metric consumes medians only.
- WM2 base: WMREQUAL3 validation batches (36 groups) on the labels' left-boundary
  source clock (J_fixed = J_old - .5*(q_n - q_0)).
- Rule (threshold-free, pre-registered): SE = world-bootstrap SE of WM2 upstream
  paired MAE (2,000 draws, seed 0) = 90.67. gone iff min(wm2,entry) <= 2*SE;
  changed iff |entry-wm2| > 2*SE; else REMAINS.

## Result
- Upstream paired MAE 120-300 s: WM2 (clock-corrected) 796.3, entry 771.3;
  |delta| 25.0 < 2*SE 181.3; min > 2*SE -> gone=false, changed=false,
  **residual_remains=true**.
- Robustness: gpuA-native groups only (28-31 excluded): WM2 908.5, entry 865.9
  (120/300 s recompute) -> same flags.
- Attribution (the reviewer's isolation): entry path nearly removes the SOURCE
  component error (60/90/120/180/300 s: 26.6->2.5, 59.9->6.1, 104.1->13.0,
  219.5->31.6, 584.0->82.5 veh-s MAE) while UPSTREAM is unchanged within noise
  (1456.9->1393.0 at 300 s) and discharge ~0 throughout. The upstream/merge-road
  residual is NOT an entry-path/source artefact.
- Consequence (recorded honestly): paired J MAE worsens under the entry path
  (pooled 512.7->546.1; J300 1449.1->1536.2; bias +225->+401) and confident-gain
  coverage collapses (103->3 calls): WM2's source error was partially CANCELLING
  its road error (negative source bias vs positive upstream bias, shares -7%/96%).
  Removing the source error unmasked road error. The head must therefore beat the
  ENTRY baseline (step_000000 zero-head reference), and the candidate must still
  beat WM2 end-to-end on J in the verdict panel to be worth adopting.

## Three-way attribution (WMREQUAL3 reference numbers, one-tick clock)
WMREQUAL3 matched upstream (bias/MAE veh-s): 120 s +238.1/322.2 (111.4% share);
180 s +394.4/638.3 (103.9%); 300 s +1031.1/1497.6 (96.0%). Matched J MAE
120/300 s: 280.0/1376.6. Third leg (trained head) is produced post-training by
verdict panel A on the same corrected harness.

| leg | clock | 120 s upstream MAE | 300 s upstream MAE |
|---|---|---|---|
| WM2 as scored (WMREQUAL3) | one-tick | 322.2 | 1497.6 |
| WM2 clock-corrected | labels left | 294.9 | 1456.9 |
| zero-head entry path | labels left | 293.8 | 1393.0 |

## Decision
REMAINS -> proceed: harness_contract(residual_remains=1) -> real TRAIN feasibility
(cuda, <=12 GB slice) -> REVIEW (MAIN/Claude in Astra's place) -> head fine-tune
launch by MAIN. No retarget amendment; the merge-response head trains against the
entry-path baseline with paired road/crossing gradients (source reported
separately), per the anchor-rebalance already implemented.
