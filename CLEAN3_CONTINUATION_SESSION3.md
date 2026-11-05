# S-LAUNCH recovery — 2026-11-06 00:38 +08

Recovered clean `c4/rerun-launch` at `6d8046e7`; read git status/history,
the full tracks specification, the last 300 registration lines and preserved
receipts before action. Completed setup, integration, qualification, smoke,
gate, launch and diagnostics remain adopted.

## Verified live state

`CLEAN3_RESUME_AUDIT_SESSION3.json` (commit `76306ebd`) is the accepted-F6
read-only cpuV audit at **00:31:38 +08**. All 538 deployed source hashes,
immutable configuration/approval/staging, final smoke/fix/reporting/gate/launch
bindings, clean1 stop/prefix and effective phase caps passed. Original owners
A301793/B301794/R301795/P3monitor301796 (start6682640) and resident chain296173
(start6536748) are alive. Combined monitors **u1..u16** are seal-valid,
CONTINUE/no DIAGNOSE, with **zero collisions and teleports**. Deployed source
remains `285663a1`.

Audit clean2 charges: **1,705,200 sim-s**; cumulative actual charges:
**2,954,640 sim-s**, within clean2's original registered authority. The existing
local follower9504 was independently checked by executable, command and creation
time18:49:32. Its state includes u1..u16/no faults, and the five-line u16 status
was already appended at00:28:15. Existing resident supervision remains active.

## Acceptance gate

- **G1 accepted**: `G1_ACCEPTED.json`, independent `REVIEW_G1_r1.md` PASS,
  reviewed HEAD `4718f00041d71b9f81310c3c87d772975105d941`.
- **G2 accepted**: existing `G2_ACCEPTED.json` / `REVIEW_G2_r0.md` retained.
- **G4 not yet accepted**: round1 fix READY is published with production commit
  `989e0422d6717a191a17e9655d863c109d652d19` and 309 passed/1 platform skip.
  Former fix worker35672 has exited. Existing autoreview records build round1;
  no G4 round1 review or acceptance receipt was present when checked. Preserve
  its independent review workflow; do not launch another builder/reviewer.

All-three acceptance and integrated Windows/Linux qualification remain required
before the owner-authorized clean2 sealed-boundary stop. G1's passing review
explicitly leaves the genuine native certification executor to launch integration;
synthetic CPU certificates are not physics evidence. G5 is separately being built
under the00:27 registration and is not claimed accepted or active.

## Full-design affordability remains blocked

`CLEAN3_QUOTE_SESSION3.json` (commit `e8bd3c8f`) is the complete zero-physics
native-manifest quote at **00:35:23 +08**:

- Prior charged **2,981,040** = clean1 **1,248,480** (includes plan_fix33,600)
  + clean2 **1,731,600** + parity960.
- Protected unspent plan_fix **566,400**, counted once.
- Complete prospective clean3 allocation **3,241,440**, preserving 24 updates,
  both evaluations, the full probe/replicated schedule, full windows, smoke20000,
  single gate15160/no retry, and G1/G4 caps230400 each.
- Cumulative bound **6,788,880**, shortfall **788,880 sim-s** against the
  clean3 scope's registered6M. Even excluding the protected plan_fix reservation
  leaves **222,480** deficit; this calculation cancels no reservation.

The later owner registration at00:27 raises the overall cap to6.5M by adding
**500,000 for separate mining with its own ledger**. Retaining that allocation
gives an overall prospective bound **7,288,880**, still **788,880 over6.5M**.
The original `BUDGET_CLEAN3.json` is unchanged and is not an unallocated6.5M
approval. No separate mining ledger was inspected here. Its actual charges and
remaining approved reservation must be included exactly once in final global
admission. MAIN/owner must resolve the cumulative shortfall before clean3 charges.

This is a running-prefix bound, not final stopped-ledger admission. Accepted
production adapters and complete G smoke expansion still require qualification.
No clean3 physics/chain or clean2 stop was dispatched in this continuation.
Verification consisted of successful existing read-only audit/quote execution;
no production implementation changed, and completed tests were reused.
