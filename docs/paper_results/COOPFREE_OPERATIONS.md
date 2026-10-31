# S-COOPFREE bounded same-host batch acceleration

The recovered one-worker u1 plan costs1,174,560sim-s in1125 joint/solo arms.
At roughly40–60sim-s/wall-s serial execution takes several hours before the first
learned proposal exists. cpuQ has32 host CPUs, load10.58, about57GiB available RAM
and4.27GiB available disk at the prospective check. COOPBANDIT continues concurrently.

`tools/coopfree_batch.py` is a one-shot operational sidecar using8 nice19/single-thread
cpuQ workers. Every worker calls the **original bound** `runner.branch`; original
load<=20, memory>=8GiB, disk>=2GiB, bitwise prefix and safety gates remain in force.
No change to policy, action samples, credit, seeds, budget, optimizer or scientific sources.

To prevent duplicate physics, the sidecar identity-checks and temporarily SIGSTOPs only
our exact supervisor PID339173/start47835664. Its active branch/SUMO socket remains
untouched. After the pause, it freezes an immutable allocation of only plan arms with
no existing job directory; completed arms and the supervisor's one in-flight arm are
excluded. All assigned IDs must match the existing whole-batch charge rows exactly.
Workers each own a unique job directory, so incomplete attempts cannot be retried.
No new charge, new sample or new simulation arm is introduced.

The sidecar waits for its bounded worker pool, validates unchanged charge rows, and
SIGCONTs only the same identity-checked supervisor in `finally`, including on failure.
The original supervisor completes its original active arm, reuses all completed arms,
then performs the single original optimizer step/dev/discovery sequence. On any failure
a durable STOP is written; no automatic physics retry occurs. No peer PID is signalled.
`BATCH_PAUSE`, `BATCH_ALLOCATION`, `BATCH_COMPLETE`, `BATCH_RESUMED` JSON receipts
retain the exact partition/identities. A second sidecar invocation is forbidden.

CPU gates cover process-identity reuse rejection, whole-group/all-solo allocation,
exclusion of started/completed arms, uncharged-arm rejection and failure-finally resume.
Publication observer is separate/read-only and exports completed optimizer checkpoints
plus evaluation receipts; it never treats partial-window credits as a learned policy.
