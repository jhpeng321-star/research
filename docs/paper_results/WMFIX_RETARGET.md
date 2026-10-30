# WMFIX retarget decision — zero-head entry path on the corrected validation panel
(addendum 03:55 item 4; addendum 05:55 gate (c)) — written 2026-10-31, numbers
filled from ZERO_HEAD_COMPARE.json when all 36 validation groups exist.

## Design of the comparison
- Entry path: conserve_queue + learned admission ON, merge-response head at zero
  (WM2 weights b8797242, no head), K8, native lanes, mix ON, fixed source clock.
  Batches: validation_zero_head/ (gpuA PID 100400 0-15.. + shard if used; 36 groups
  x 32 rows; identical code/inputs/CRN hash-verified per WMFIX_SHARD_STAGING.md).
- WM2 base: WMREQUAL3 validation batches (36 groups), compared on the labels'
  left-boundary source clock: per row/horizon J_fixed = J_old - .5*(q_n - q_0).
- Rule (threshold-free): SE = world-bootstrap SE of WM2's upstream paired MAE
  (2,000 draws, seed 0). residual GONE if min(wm2, entry) upstream MAE <= 2*SE;
  CHANGED if |entry - wm2| > 2*SE; otherwise REMAINS. Train the head ONLY if it
  REMAINS. No hand-set behavioural thresholds anywhere in this rule.

## Three-way attribution (WMREQUAL3 reference numbers, one-tick clock)
WMREQUAL3 matched native-lane upstream component (bias/MAE veh-s, share of J
squared-error covariance): 120 s +238.1/322.2 (111.4%); 180 s +394.4/638.3
(103.9%); 300 s +1031.1/1497.6 (96.0%). Matched J MAE 120/300 s: 280.0/1376.6.

| leg | clock | 120 s upstream MAE | 300 s upstream MAE |
|---|---|---|---|
| WM2 as scored (WMREQUAL3) | one-tick | 322.2 | 1497.6 |
| WM2 clock-corrected | labels left | PENDING | PENDING |
| zero-head entry path | labels left | PENDING | PENDING |

## Result (PENDING all 36 groups)
- wm2 upstream paired MAE 120-300 s: PENDING; entry: PENDING; bootstrap SE: PENDING.
- gone: PENDING; changed: PENDING; residual_remains: PENDING.
- Paired-J pooled accuracy (sign/r/MAE) wm2 vs entry: PENDING (report tables).
- Component check that the delta is where the fix aims (upstream/source): PENDING.

## Decision
PENDING. If REMAINS: proceed to harness_contract(residual_remains=1) -> real TRAIN
feasibility (cuda, <=12 GB slice) -> REVIEW -> training launch (head fine-tune).
If GONE or CHANGED: no head training; record retarget amendment (the entry path
alone changes the residual; the merge-response head design must be re-argued
against the new error decomposition before any GPU training is requested).
