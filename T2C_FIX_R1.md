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
