# S-RB-UNC — READY; bounded observer LIVE through09:00

Branch `c4/rb-unc`; resumed clean `d3231c74`. All prior code/results preserved;
full earlier handoff: `docs/RBUNC_HANDOFF_ARCHIVE_0230.md`.

## Integrator intake
- Default-OFF `future_sampling='hidden_state'` in
  `wms.control.hidden_state.plan_hidden_state_futures`. No shared model edits.
- Preferred `FittedHiddenState.from_directory(shrinkage_v1)` samples FIT's exact
  joint support/weights once per observable world/VID/future. Shrinkage ALREADY
  applied. Unknown drivers/newcomers use fitted population; preserve CAV settings.
- Pass numeric observable `vehicle_ids=`, numeric world `row_keys`, `t0_s=300`,
  `arrival_fit=LearnedArrivalFit.from_file(observed_arrivals.json)` plus causal
  entrance histories/queues. Never use packed slot IDs as observed VIDs.
- Rebuild mutable futures per action, keeping world/VID/seed/future and starts
  equal: incumbent settings, demand-rate/request and newborn draws share CRN.
- New-world causal posteriors: FIT `condition_motion`, then
  `fit.with_motion_posteriors({(world,vid):dist},t0_s=t0)` clears saved-ID/LC
  records, binds exact finite weights and hashes the causal snapshot. Caller
  supplies faithful motion predictions/support order; missing IDs use population.
- Existing INT FULL-v1 already includes fitted IDM/learned arrivals. New commits
  for NEXT immutable candidate: `51ae6345` optional LC source-domain reader;
  `073e3527` additive causal arrival-observation receipts. Earlier adapter commits
  `e4365034`, `0d04093b`, `5a6cb332` cover VID CRN and causal-motion bridge.
- LC remains opt-in/not adopted: first4motivations≥0, cooperative≤1, assertive>0;
  no clipping. Old unadopted cooperative>1 supports now fail closed. Historical
  preferred32 receipts retain original source/inputs. Await FIT's refit gate.

## Learned arrivals / calibration evidence
- FIT969400–969431 counters: request=inserted+queue. Documented constant-Poisson
  pre1200s phase wins BIC against temporal profile/extra NB dispersion. Retain
  time-bin diagnostics; do not extrapolate unsupported trends. Per-future demand
  rate Gamma(count+.5,exposure), exact request clocks, population newcomer draws.
- NEW `d850d5b9`: frozen causal0–150→150–300s count check on existing FIT32 only.
  Gamma–Poisson minus SAME-MEAN Poisson NLL−0.3137 [−0.5846,−0.0732]; CRPS
  −0.1957 [−0.4242,+0.0132]counts unresolved.90%coverage85.94% vs73.44%, exact
  discrete nominal91.24%/91.57%. Supports rate uncertainty, not full calibration.
- NEW `learned_arrival_observations` binds actual causal counters/family/start
  separately from the file SHA. Reconditioned starts must match BOTH hashes.
- Completed preferred32/K8/H120 LEGACY reference: team within-K SD748.35
  [668.54,828.51] vs real across-world4200.50veh-s; drivers164.22/arrivals874.61/
  both850.71/fixed0.00 on969400. All280futures source/conservation/RNG PASS.
  Team CRPS980.79 vs raw980.68, paired change+0.11 [−24.19,+24.32]: NO
  demonstrated traffic improvement. Do not tune conditional SD to marginal SD.

## Owned ongoing work / resume safely
- ONE CPU-only nice19 observer: gpuA PID348253, source `4d860683`, exact boot/
  start/source/target hashes in `rbunc/interim_watch_start.json`. Root
  `/root/rbeval20261031/rbunc_support_v1`; poll1200s; deadline09:00+08 Nov1.
- Startup LIVE, load12.90/208CPUs, no complete FIT world yet. Opens explicit
  FIT969420–969431 task paths only; requires K8/source/native-start/CRN/ledger
  consistency, emits native120s uncertainty diagnostics as complete sets change.
  Missing new provenance in the OLD runtime is recorded unavailable. NO automatic
  physical scale change, model inference, peer process action or full EVAL scoring.
- `py -3.13 -B tools/rbunc/interim_ops.py status` is read-only. Fetch with
  `... fetch --out docs/paper_results/rbunc/NEW_SNAPSHOT.json`. NEVER redispatch.
- INT's333463..333465 +335983 own FULL inference/04/08/full scoring; preserve.
  Prior UNC32-world workers all terminal0; count analysis executed once/rc0.

## Verification / receipts
- Final combined local CPU64PASS/1optional SciPy-module skip/32.06s; offline
  numerical3PASS/1.74s in existing cpuQ SciPy. Observer tests included locally.
  No install. Owned observer incarnation rechecked LIVE after final tests.
- Reports/receipts under `docs/paper_results/rbunc/`: `preferred_summary.json`,
  `arrival_prefix_v1.json`, `arrival_prefix_cpu.json`, `arrival_snapshot_cpu.json`,
  `lc_domain_cpu.json`, `interim_watch_cpu.json`, `interim_watch_start.json`.
- API/commands: `tools/rbunc/README.md`; frozen designs `RBUNC_{ADDENDUM,
  PREFERRED,ARRIVAL_PREFIX,INTERIM}_DESIGN.md`. No integrated efficacy score yet.
