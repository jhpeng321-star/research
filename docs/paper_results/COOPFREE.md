# S-COOPFREE — implementation and execution receipt

## Current state
Build and CPU gates complete. MAIN's explicit 23:58 budget approval was observed.
Frozen design: `COOPFREE_DESIGN.md` (62 lines), SHA256
`05c729eac9b45c2128b7d8bf0db01f350730fb381d1956709a6a7b7fae9c91e3`.
Approved SUMO pipeline is running on cpuQ, PID339173/start_ticks47835664, nice19/one thread.
Registered operational sidecar436993/start48005147 now drains1082 disjoint precharged
u1 arms on8 nice19 CPU workers, excluding all started work and the one in-flight arm.
It resumes the intentionally paused original supervisor for the unchanged optimizer step.
Original scientific sources, samples,2M budget and prefix/safety gates remain hash-bound.
First1140s baseline and full repeat both closed at charged=actual1140, bitwise-identical
RESULT SHA256 `a5419ebb2219b3cc765a020e6515c4e38bebcc5aba800b876375fb9d4e328030`.
Fresh-session status: owned process identity matches, all24 screens/all12 repeats and
the zero-update canary complete;50,820closed actual/1,216,740precharged, no STOP.
The whole156-window first update is executing; optimizer/dev/discovery remain pending.

## Implemented
- `tools/coopfree/policy.py`: observable per-CAV continuous speed/log-gap policy,
  four-dimensional zero-init RSU broadcast, shared+individual Gaussian exploration,
  learned categorical 10/20/30 s window, exact correlated joint search density.
- `execution.py` / `backend.py`: fixed absolute targets, whole-zone/any-lane CAV roster,
  unchanged original safety executor, command memory and CommandOwnership actuator path.
- `credit.py`: full team delay, exact D_G/D_i/C_G, equal individual/cooperation weights
  from update1, shared team credit, objective algebraically 2D_G (anti-gaming).
- `runner.py`: bitwise baseline repeats/full-prefix gates, one owned nice19 CPU worker,
  whole156-window batch precharge, >=12-world floor, exact Adam/checkpoint resume,
  measured execution-share exploration adjustment, no automatic physics retries.
- `evaluation.py`: fixed24-world/all12-cell paired J600/J900 bootstrap/safety checks;
  frozen peer comparator required rather than substituting MAPPO under its name.
- `prior.py`: hash-bound peer TRAIN-only signed findings bias state sampling only.
  The pipeline checks public peer snapshots before updates; it never starts peer jobs.
- `analysis.py`: original S-COOPBEHAV detectors plus silhouette-selected clustering;
  exact learned-group discovery reruns include every member solo with the joint targets.

## CPU and provenance
Final existing cpuQ environment: 18/18 tests PASS, 0 skips, 1.21 s, SUMO0 sim-s.
`tools/coopfree/evidence/CPU_BUILD5.json` binds all25 staged Python source files.
Coverage: real frozen MAPPO first-command parity, off-zone bitwise command/memory parity,
original safety and CommandOwnership execution, exact multivariate-normal density/gradient,
credit/anti-gaming, action expiry/exit/solo context, whole-batch caps and no-repeat resume,
real frozen env seed-guard aliases, fake-plant training/checkpoint orchestration,
paired bootstrap arithmetic, peer detector schema and TRAIN-prior leakage rejection.
Build2's detector-return-schema test failure is retained; the existing detector was not changed.
Fresh TRAIN982000–982023 / DEV982024–982047: both guards48/48; exact registry/range and
safe-name audits PASS on cpuQ223601/gpuR359876/gpuA837771, no candidate exposures.
Frozen E3 b3c031ad, MAPPO169 d9042cc5 and SUMO1.27.1 checked in the existing runtime.

## Budget and interpretation
Hard reservation2,000,000 sim-s: screen27,360 + repeat13,680 + TRAIN1,650,000
+ DEV172,800 + discovery91,200 + UNALLOCATED44,960; failed/prefix/warmup physics retained.
Whole-zone groups can make only one156-window update fit; the pipeline stops before
an unaffordable full batch rather than shrinking groups or calling a smaller update complete.
Absolute held targets only approximate changing MAPPO commands. CPU first-command parity
 is not rollout parity; a TRAIN-charged zero-update physical canary measures the drift.
Score-function search reuses the mandatory joint+solo rollouts for gradients. ES/population
search has not been efficiency-qualified or run; no empirical ES superiority is claimed.
No learned cooperation, performance or new mechanism is claimed before the paired receipts.

### Recovered zero-update canary (one TRAIN world, seed982007)
First-command error is exactly0, but269/271 compared motion frames differ from MAPPO
(99.26%). Thus sustained zero-update holds do **not** establish rollout-level MAPPO parity.
At absolute t900 (600 after activation), total delay is34,858.27 vs39,518.72vehicle-s;
at t1140 (840 after activation),65,412.28 vs67,278.12. These are cumulative totals,
not a J900 endpoint, and this single-world drift is not a learned improvement claim.
Collisions/teleports are0 for both; hard-brake totals18vs14.5 and23vs18 respectively.
Keep this structural hold effect separate from any subsequent learning attribution.

### Peer partial findings after the u1 plan was frozen
COOPBANDIT partial135 is now public:16/135 helpful states, but its registered
uncertainty-gated policy selectsO0 throughout; positive C_G alone did not imply D_G>0.
Read-only CPU preflight of our prior fit and real pickled comparator succeeded
(`PEER_PARTIAL135_CPU.json`,0sim-s). The signed state-only prior uses28 available
states/13TRAIN worlds; no option pattern enters the free action space. This snapshot
arrived after u1 sampling: u1 retains its original observable-only priorities, not a
retrospectively changed sample. The pipeline adopts the newest complete public peer
snapshot before its next update/budget check and freezes that comparator for dev.

## Reproducible operations
Immutable cpuQ root: `/home/ubuntu/coopfree20261101_build5`.
Build4's launch preflight rejected the correct binary's banner because it omitted the word
"Version"; no Popen/charge/physics occurred. The exact1.27.1 banner check is now regression-tested;
build4 remains intact and build5 has a fresh source/CPU binding, same scientific plan and budget.
Existing peer sklearn1.7.2/scipy1.15.3/joblib/threadpoolctl target is linked read-only into
build5's import path for comparator/clustering; no install/newvenv or physics restart.
Additional CPU model-pickle/clustering QA PASS, SUMO0, binding SHA unchanged
`987a7ae0bd0c056bd06f8c6b4a099f00852ca2ba98961659efd6bb5a1481e05f`.
`python -m tools.coopfree.ops status` reads only that owned dispatch/accounting state.
`ops bind` requires the actual MAIN approval, CPU/source certificate and fresh exposure audits.
`ops launch` is one-shot, binary-version/PATH preflighted; it cannot duplicate an existing dispatch.
Pipeline: screens/repeats -> zero-update canary -> whole-batch updates -> u1/final dev -> discovery.
Checkpoints `policy_uNNN.pt`; updates `UPDATE_NNN.json`; ledger `CHARGES.jsonl`;
terminal failure `STOP.json` (never cleared automatically), budget exit `BUDGET_STOP.json`.

<!-- COOPFREE PUBLICATIONS BEGIN -->
## Automatically published proposal checkpoints
Consumer guide: `docs/paper_results/COOPFREE_PROPOSALS.md`.
Public artifacts: `tools/coopfree/publications/`; verify `MANIFEST.json` hashes.
Frozen remote root: `/home/ubuntu/coopfree20261101_build5` (cpuQ).
No learned checkpoint yet; the full156-window/all-member-solo first update is running.
Latest milestone: 136 complete jobs; 1216740 charged/146760.0 closed-actual sim-s.
Training-window credits describe exploration; they are not paired dev-policy performance.
The owned supervisor is paused while eight same-host workers drain only disjoint precharged arms; the sidecar resumes it for the optimizer step (see COOPFREE_OPERATIONS.md).
<!-- COOPFREE PUBLICATIONS END -->
