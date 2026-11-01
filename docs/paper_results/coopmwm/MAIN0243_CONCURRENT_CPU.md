# MAIN02:43 concurrent-plan engineering qualification

CPU3: 164 PASS, zero skips, 9.16 s, zero SUMO. Receipt
`CPU_coopmwm20261102_concurrentcpu3.json` pins the immutable source.

The plant now retains several disjoint commitments, excludes their members
from new groups and logs all active plans per step. A new WM request carries
the other plans' remaining profiles. Both reference and transfer-batched
decoders apply these absolute profiles to EVERY candidate/comparator until
their original expiry, then revert those members to the policy continuation.
Tests cover concurrent execution, independent expiry, overlap rejection,
420-second tables and exact reference/transfer-table equality.

The first CPU dispatch had a launcher indentation error before tests/physics;
CPU2 retained two fixture failures (old unchained group fixture and an incorrect
fixture member count), repaired in CPU3. No failed or completed physics repeated.

GPU source-bound parity and prospective runtime registration remain required;
the original READY GPU workers and A continuation are untouched in this step.
