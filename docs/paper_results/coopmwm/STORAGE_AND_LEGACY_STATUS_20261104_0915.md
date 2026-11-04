# Storage and legacy-stop verification — Nov 4, 09:15 +08

## Permanent legacy stop is complete
MAIN 09:10 is satisfied by the existing permanent owner-specific stop and a new
direct verification, recorded in canonical REGISTRATION.md. The actual legacy
endpoints are B21/R20 (not the rounded u20 description); checkpoint hashes,
receipts and bank remain intact. All old arm/supervisor/grader PIDs are absent.
No additional termination was necessary. Shared collection/bank services stay live.

## Approved SHA-gated archival is healthy
The V2V companion 2213098 now reports **1,035,251,575 bytes moved**, up from
499,403,300 bytes at the previous handoff. Eight certified batches have matching
verified/removed file counts; each retains its original per-file SHA manifest.
No eligible files remain at this scan. Both movers are live; old transport
timeout receipts are retained and followed by successful certified transfers.

cpuQ free: **6,902,763,520 bytes (~6.43 GiB)** at 09:13. This is **2,760,912,896
bytes short of 9 GiB** before further writes. gpuA free at the worker scan:
20,212,305,920 bytes. The destination floor and all source reference/FD/age guards
are preserved. Backfill remains correctly paused; it is not redispatched.

The all-three-completed-update rule protects an important real consumer:
`v2v_live.hooks.safety` reads each **same-world, same-update A RESULT** when B/R
reach that update. A7 being ahead of B3/R3 does not make those A safety-reference
payloads disposable. No archive-scope broadening is made on that basis.

## Collection
Current collection graders are alive but have exhausted their saved snapshot
queue. The known V2V capture and imagined-actor/intent wiring gap still prevents
fresh current-policy searches; it is not a dead-worker condition. Existing
config-routing fixes are CPU-qualified but not activated as a cosmetic restart.
The launch operator's mechanics/stronger-imitation/practice-starts work remains
separate and is not duplicated here.

Evidence: `V2V_ARCHIVE_SCOPE_0712_V2_STATUS_1793754825606042500.json`,
`RESUME_SERVICES_1793754627419389500.json`, and the direct adoption receipt
`RESUME_ADOPTION_0915_1793754745255121600.json`. Zero new SUMO or inference.
