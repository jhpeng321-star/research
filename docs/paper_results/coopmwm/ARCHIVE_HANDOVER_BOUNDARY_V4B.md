# Archive handover liveness correction

The v4 waiter correctly retained an active old transfer, but its whole-loop
sleep requirement could starve while collection writes prevented the old8GiB
maintenance target from being reached. A SHA-bound engineering amendment adds
one safe boundary: no active transfer children AND either a completed batch or
a pre-proof/source-hashing phase. The parent is stopped briefly, then state,
children and the newest batch proof are rechecked. An incomplete SHA-certified
removal phase is always resumed unchanged; transfers are never interrupted.

Amendment807d6863b8916935988b6282eaf5b7b3908964a9e5cc844357e9050857922732
is append-only in the canonical registry and new control. Four operator tests
PASS, including generated programs, proof-boundary guards and waiting rather
than launch during an active transfer. The existing9Linux-qualified mover and
its source hashes are unchanged.

Next: retire only the identity-proven idle LOCAL waiter43764, after confirming
the new remote mover never started, then dispatch generationV4B once. Original
watcher/transfer/manifest receipts remain; no duplicate mover or physics.
