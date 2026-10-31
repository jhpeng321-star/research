# S-COOPFREE — implementation and execution receipt

## Current state
Build and CPU gates complete. MAIN's explicit 23:58 budget approval was observed.
Frozen design: `COOPFREE_DESIGN.md` (62 lines), SHA256
`05c729eac9b45c2128b7d8bf0db01f350730fb381d1956709a6a7b7fae9c91e3`.
SUMO dispatch follows the source/approval/registration binding; no learned result yet.

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

## Reproducible operations
Immutable cpuQ root: `/home/ubuntu/coopfree20261101_build5`.
Build4's launch preflight rejected the correct binary's banner because it omitted the word
"Version"; no Popen/charge/physics occurred. The exact1.27.1 banner check is now regression-tested;
build4 remains intact and build5 has a fresh source/CPU binding, same scientific plan and budget.
`python -m tools.coopfree.ops status` reads only that owned dispatch/accounting state.
`ops bind` requires the actual MAIN approval, CPU/source certificate and fresh exposure audits.
`ops launch` is one-shot, binary-version/PATH preflighted; it cannot duplicate an existing dispatch.
Pipeline: screens/repeats -> zero-update canary -> whole-batch updates -> u1/final dev -> discovery.
Checkpoints `policy_uNNN.pt`; updates `UPDATE_NNN.json`; ledger `CHARGES.jsonl`;
terminal failure `STOP.json` (never cleared automatically), budget exit `BUDGET_STOP.json`.
