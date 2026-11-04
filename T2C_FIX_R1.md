# T2C independent-review fix receipts

Recovered HEAD: `588bc68b`. Preserved `REVIEW_T2C_r1.md` and the previous receipt
`T2C_READY.blocked1.json`. No matching T2C process was running; no dispatch started.

## Step 1 — calibrated dense labels and realised monitoring

Consumed teacher protocol now carries target-specific calibrated probability,
verification provenance, and eligibility. Runtime applies the registered 0.8 rule
to both arms, permits verified bypass and masks only dense imitation. Empty eligible
sets retain group/policy/bank learning and write diagnostic receipts. Teacher spot
checks produce realised accuracy with source row IDs, independently of model CE.
Registration/criterion/approval hashes change with this filter.

CPU verification (requested interpreter, GPU hidden, one OMP/MKL thread,
bytecode/cache writing disabled): runtime + monitor suites **32 passed** in
233.82 s, `.t2c-r1-label2`. An initial collection failed on indentation and was
corrected before this passing run. New regressions cover both arms at/below 0.8,
verified bypass, null targets, empty imitation updates, mask forgery and realised
accuracy/filtered-share receipts.

## Step 2 — shared paired final evaluation

u24 now invokes the explicit `planner_eval` T1D boundary, requiring all 288 paired
full-world jobs, complete checkpoint file identity, echoed result identities, TTC,
shared paired bootstrap analysis and a self-hashed final receipt. Isolated tests
remain separate and now require TTC. The common-world list and changed evaluation
semantics bind config/criterion/budget request/approval; the 345,600 sim-s final
phase still covers 96 full episodes per arm (no horizon or cost reduction).

Sibling `planner_eval.py` is absent here; the documented consumed protocol is
exercised by a test-only module. Tests prove the full run routes into that module
and reject unpaired worlds, missing jobs/components, wrong checkpoints, shortened
windows, result-identity mismatch, missing TTC and incomplete CIs before readiness.

CPU verification: focused evaluation cases **10 passed** in 53.86 s,
`.t2c-r1-final-eval`; config/approval/evaluation selection **54 passed, 1 skipped,
15 deselected** in 118.75 s, `.t2c-r1-final-bind2`. An earlier selection had two
stale fixture assertions (new quote fields/TTC); both were updated. The skip is the
inherited Windows file-symlink privilege case; hard-link assertions execute.

## Step 3 — state-conditioned bank and effective supervision

The learner now receives a validated source-bound batch: own observation/features,
move target, original plan, source/seed identity, unchanged per-seed weight and
provenance, plus the complete effective sample and binding limits. Bank options
are registration/approval-bound. Source-context/plan associations and own-arm
identities are validated before dispatch; learner consumption gets a separate
batch-bound receipt, refusing missing effective rows.

The CPU substitute now uses a state-conditioned head for actual choices and a
separate full bank pass with the reviewed `bank_imitation.pair_loss` effective
Tobit objective. Swapping labels across distinct states with identical class
frequencies reverses gradients and subsequent choices. Tests check weight/limit
sensitivity, dead-zone invariance, layer-only effective gradients and frozen base.
An initial new gradient test exposed an overly wide no-grad scope in the CPU
substitute; only base inference is now detached, preserving layer gradients.

CPU verification: bank + complete runtime suites **40 passed** in 163.48 s,
`.t2c-r1-bank3` (supersedes a 39-pass run before the missing-effective-receipt case).
This includes full smoke/monitor CLI execution and shared final-harness routing.
