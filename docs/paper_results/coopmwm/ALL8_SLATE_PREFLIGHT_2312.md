# Actual-slate all8 verification planning — Nov3 23:12 UTC+8

Read-only receipt: `ALL8_SLATE_QUOTE_1793718753675224300.json`.
No simulator, inference, bank mutation, reservation or phase transfer.

The qualified first-seed receipt reader finds129 unique B decisions and120 R
decisions. Re-running their full candidate lists on seed2 alone would require
784,290 + 762,220 = **1,546,510 sim-s**, before any extended or solo branches.
The remaining bank_verification_ext allowance is137,180, leaving a **minimum
1,409,330sim-s shortfall**. MAIN is notified before new builder activation.
The global17,471,840cap and all restricted phase allocations are preserved.

88 B decisions lack the candidate-spec files required by the all8 reader in
their original LS label_attempts paths. Present specs are verified against both
the sealed branch-spec hash and candidate-plan hash. Missing inputs must be
recovered from verified archives; no search replay or omitted/zero-cost inputs.
R's specs are complete. Its all-positive conditional-confirmation upper bound
is6,323,870sim-s; B's present-spec subset upper bound is1,786,380 and is not a
complete B quote. The minimum is therefore not a full migration allocation.

These are cost/input checks, not training or outcome-selection results. All
completed accepted and rejected decisions were included, deduplicated by the
verification identity. No outcome-dependent selection or scope change.
