# S-SELMAPPO2 — per-CAV MAPPO controller picker

## Status (2026-11-01)

MAIN01:48 has now **acknowledged and launched** the same-cap continuation: unused second
smoke18107999 only, then all originally registered production worlds. The explicit
one-use continuation implementation passes93/93 local CPU tests and Ruff. It
retains the hash-pinned failed ledger byte-for-byte,900reserved/300actual, and
never replays18107998. A later failure still stops dispatch. The report validates
this historical exception separately and retains the600s reserved/actual gap in
every update. cpuQ93/93CPU/zero-skips and real frozen integration pass;
safe-filename exposure audit199667/zero conflicts/errors. Immutable current build:
`c141d5715f26af4e9b50d0866cdf971ef92d55dd1223c4312c4f25bdec5a8b29`.
Owned supervisor606968 (UID1000/startticks48374316) was launched once, nice19,
CUDAhidden/single-thread workers,<=12 after successful second smoke. Dispatch
disk2.601GB/load7.08/32cores; each reservation yields below2GiB disk/8GiB memory
or above spare-CPU admission. Initial production verification at Unix1793469726 records
**second smoke900s PASS and20production worlds complete**, supervisor still live.
Totals19,200actual/26,100reserved (including original failed900/300);7production
reservations still unresolved at that observation,0complete PPO updates yet.
The21complete episode receipts have0collisions/teleports in recorded control
windows. CollectionSHA25699d8a03dc01102ef052f1944f1ffaeafef828ade5c160ab2cb5bd40891fb9935.

### Preserved first attempt

**Stopped during the first mechanics canary.**
Own cpuQ supervisor428649 exited. The first smoke reserved900 sim-s and ran300
native-warmup seconds; no controlled window or gradient update completed.
Checkpoint000 is retained; final169 is absent. Learning curves and final
evaluation are pending, not zero-valued results.

The failure was `ModuleNotFoundError: No module named 'wms.control.execution'`
in the inherited per-CAV CTRL0 route. All original receipts, failed reservation,
logs, source snapshot and dispatch lock are retained. The remaining original
smoke and all production seeds are unconsumed. A same-cap continuation request
is registered; no automatic relaunch is armed.

## Registered experiment

`SELMAPPO2_DESIGN.md` SHA256
`09499f2af60311b3690a940dca8d94605d710465afb9c9eb05d5b0a79b23900c`.
Shared REGISTRATION.md contains the exact advance-approved per-CAV cap of
**7,012,800 sim-s**:6,996,600 production +1,800 mechanics +14,400 evaluation
reservation. The evaluation reservation remains unstarted because S-C1SWITCH2
owns the panel physics and additional J900 tail. Training is169x46 fresh900s
worlds; optimization seed1; one training replicate; no retries or top-ups.

Thirty decisions at300..880 every20s, six-controller categorical library including
WALL; full-team road+source delay reward, gamma=lambda1. Shared environment,
features, PPO and runtime are imported from S-SELMAPPO; local code only owns
per-CAV dispatch/accounting and the atomic consenting WALL roster.
One frozen WallLaw group selects its eligible cross-lane pair; consenting
nonmembers use CIRCLES, other CAVs keep their selected controller. Active-member
withdrawal/departure/re-entry interrupts the group atomically. No new behavioural
thresholds. WALL receives current observable receipts; missing junction flags
and traffic outside sensor coverage are not filled from truth.

## Verification and source binding

- Local86/86 CPU tests pass after repair, Ruff passes.
- cpuQ56/56 CPU tests pass, zero skips; real CTRL0/MAPPO checkpoint constructors,
  frozen WALL formation/preservation/stop, per-CAV frozen seed aliases,
  30x20s reward windows and actual shared PPO learning pass with SUMO intercepted.
- Registry allocation audit passes. Pruned metadata-only exposure census:
  cpuQ185,978 safe filenames; gpuR303,475; no conflicts or traversal errors.
- Immutable staged binding:
  `b42c257b09813c175fcb7261e341c7599d0bb5c4c971529e2856890d2fc8038c`.
  Parent:`/home/ubuntu/selmappo2_20261101/`; build name is `build_` plus this hash.
- Earlier CPU-only stage35514f91 failed because its private test import omitted
  the frozen baseline search path. Original stage/evidence retained; no physics,
  budget reservation or training occurred there. Corrected56-test stage passed.

## Diagnosed repair and accounting

The local per-CAV adapter now hands CTRL0 through its real command owner as native
at the one-second ownership boundary; the unchanged frozen typed residual path,
under original `zero_scope()`, recomputes the **forced native-base CTRL0 command**
before BOTH half-second physics steps. It affects only CTRL0-selected CAVs and
avoids an extra CIRCLES/slowDown writer. This preserves the registered CTRL0
controller; it is not a substitution of raw native-zero execution.

Ten new CPU regressions execute the actual hash-checked frozen control-tick body
with real command owners, PI/CIRCLES, synthetic policy outputs and fake transport.
They verify selected-only half-step writes, non-CTRL0 routing, NaN-slot filtering,
current handover clocks and zero-write WALL cleanup after terminal connection close.
Shared owner terminal-clock/reward fixes are also available. No repair simulation
has been dispatched.

Repaired immutable cpuQ stage:
`build_b299a4b31582d7bfc57b9486dbdf89287044be79a66d045fbd3bd6a0e3fe2e0d`.
All86/86 tests pass there (zero skips,7.99s), including10 actual-frozen-control-tick
regressions and20 collection/report checks. Real frozen checkpoint/WALL/seed-alias/
shared-PPO integration also passes. Its receipt explicitly marks CPU-only repair,
continuation unauthorized and0newSUMO. Source/audit/CPU/bundle receipts are retained.
An intervening CPU-only stagecddb11be caught two stale4GiB test expectations after
raising memory admission to8GiB for MAIN01:08 priority; corrected boundary tests
now pass. It incurred no simulation or reservation and remains preserved.

Verified closed accounting: **900 reserved /300 actual sim-s**, no refund,
one failed smoke, zero successful smoke/production episodes, zero optimizer updates.
Ledger SHA256:`eb53747c54b045b8a4c3447969d14ac4fc6f24e3719cdc5b18aff31d358417a9`.
Full collection SHA256:
`87cebb19f9bbd7ec44458916de6299bbaef631dfb8b4af93f4f1d947f66616f1`.
The proposed continuation retains this900 charge, uses ONLY the original unused
second smoke900, and retains production6,996,600 plus unstarted evaluation14,400:
the original7,012,800 cap is unchanged. MAIN01:48 acknowledgement is now recorded
under the amended no-retry protocol; no failed world is replayed or replaced.

## Learning curves and evaluation

Production is running. No learning curve can be measured from zero completed updates.
Supervisor records per-update J600 mean/std/range, episode
return, PPO loss/KL-stop, sim-s/hour, actual/reserved charge and checkpoint SHA256
in `training/training.jsonl`. Initial000 and every001..169 checkpoint are immutable;
`FROZEN_FINAL.json` binds the final policy and sources. One worker completes both
smoke worlds before expansion to at most12spare nice19/single-thread CPU workers.
The first attempt stopped before this expansion. Any acknowledged continuation
must explicitly account for the already consumed failed smoke slot.

`scripts/selmappo2_watch.py` is a separate read-only collector (4CPU checks/Ruff
pass), not a training dispatcher. Its exclusive watcher lock prevents duplication;
own localPID12432 was launched once at Unix1793469838 and is live.
300s read spacing, immutable full collections/curve CSV+SVG/report snapshots, and
a live section below keep this report current without changing staged training.
Only verified COMPLETE final169 produces `SELMAPPO2_FINAL.json` for S-C1SWITCH2.
It can ingest the peer's explicit byte-pinned evaluation manifest at
`selmappo2/PEER_EVALUATION_MANIFEST.json` with schema`selmappo2.peer_manifest/1`,
absolute safe `path` and `sha256`; underlying receipt must be complete schema
`selmappo2.evaluation/1` on the exact PANEL_v2 contract. No local evaluation physics.

S-C1SWITCH2 panel969136..969151 is disjoint from production18100000..18107773 and
smoke18107998/18107999. The peer's complete hash-bound J600 receipts will supply
evaluation and paired WM/BB/six-fixed contrasts (2000world bootstraps, seed0,
95%CI, best-fixed reselected per draw). No partial-checkpoint efficacy claim;
single-training-seed uncertainty and richer per-CAV action space will be disclosed.
Frozen speed-MAPPO library pretraining and WM/BB data/compute are separate costs.

Evidence is under `docs/paper_results/selmappo2/`.
