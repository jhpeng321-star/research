# S-RB2-BREAK — GPU PAUSE requested, 2026-11-01 ~17:38 +08

Branch `c4/rb2-break`, physical baseFULL-v4/22cc32f6. Plan `docs/rb2break/PLAN.md`.
**The major congestion/breakdown defect is NOT solved. No qualification claimed.**
Prior notes preserved in `docs/rb2break/HANDOFF_{1318,1720}_ARCHIVE.md`.

## Corrected code / integration
- `8addc19e`: default-OFF physics.sumo_cf_lane_speed_limits (network-local CF limit).
- `e164b309`: native_model='sumo_uncommanded' delegates native CAV CF/merge/LC
  to the same physical stream as identical-setting HDVs; dispatcher still once/tick.
  Requires match_cav_execution_settings=true; commanded action equations unchanged.
- **Use5cf4b153, NOT116b2147's mode2 binding interpretation.** Actual mode2
  alwayscmd uses shared sumo_idm law and ignores the retired learned mirror.
  Correct binder reads actual registered CAV design/allowed speed (known action
  definition, not hidden HDV truth). Legacy modes retain their prior fit path.
- `5947c801`: bind before population/sampled-LC replay; known-actuator fallback
  if no observed CAV donor. Unit proof: same sampled vectors/arrivals as late
  binding without replay, HDVs and encoder tensors untouched.
- Flags remain default OFF in base recipe. No controller/black-box retraining,
  neural residual, outcome offset or hand-set jam threshold.
- Latest combined local QA134PASS/2CUDA skips; exact source hashes in receipts.

## Complete dev12 K1/all5arms/H600 evidence
All results below matched future0, NOT independent K8 qualification.
Baseline600s bias14,979.67veh-s/CRPS16,725.57. New actual-mode2 CAV binding:
bias14,802.50/CRPS16,646.09. Add true native-CF ownership:
bias14,666.50/CRPS16,552.48; upstream600s event rate still100% vs real90%.
Source waiting still+12,206.98veh-s. All4commanded arms EXACTLY unchanged vs
corrected CAV parent (`NATIVE_ACTUAL2_COMPLETE.json`, guard-enforced).
Corrected evidence `CAV_ACTUAL2_COMPLETE.json`; old CAV/NATIVE_COMPLETE.json
are preserved old-vector diagnostics, not current plant-consistency evidence.

## Attribution (conditional K1 screens, not additive causal shares)
600s delay differences vs same-FI0 baseline: LC execution held+11,219;
merge advice removed+3,436; departLane first+29,045; departSpeed last−30,888
but worsens CRPS/cancels source−45,934 with road+15,046. Reject these rule swaps.
Population-only current drivers−327; tau homogenized−241; v0 homogenized+135;
zero-shrinkage+422; fixed conditional rate+2,803. No major source-bias remedy.
LC execution screen RETAINS wishes/advice; don't infer whole LC-family causality.
Frozen dev K8 baseline bias13,485 =road2,154 +source11,331 (84% accounting).
Native mainline admissions−50.65vehicles/600s; mean intensity excess only0.713
requests. Source positions180..300s are100%SUMO base/5m bodies, not a boundary
inflow or length mistake (`source_observables/SOURCE_OBSERVABLE_AUDIT.json`).

## GPU priority / continuation / exposure
- MAIN17:35 directs checkpoint of own GPU workers501141/500921 and no new GPU
  jobs until MAIN appends `S-RB2-BREAK RESUME`. Cooperative STOP files installed
  after exact owner/PID/boot/start-tick/command checks; workers finish the in-flight
  task, retain its receipt, then exit before another task. No process signals.
  At request: shard0 had3tasks; shard1 had2. These are PARTIAL K8, no scored worlds.
- Corrected candidate FULL development K8 source5947c801: two disjoint6world
  shards `/root/rbeval20261031/rb2break/devk8_native2_s{0,1}`, all96new tasks;
  own workers501141/500921 at launch,5GiB cap/one CPU thread/nice19 each.
  Max2BREAKCUDA workers; no peer PID action. Complete K8 worlds only for scores.
- Three motion-only population/support extensions failed pre-t0 CRPS gates and
  got no traffic inference. Startup-only10s v3 improves birth-disjoint pre-t0
  acceleration CRPS0.206464→0.188588m/s² (10/12worlds); packaged entrant-only
  artifact SHA256fdd816e24c85f6f4b65b67b5f992b2879c404c4c3fe75cc8cb07a5bce0397e39.
  Traffic/probe parity remains UNTESTED; not deployable or part of frozen K8.
  Recovered probe/binding code retained; no new inference launched in recovery.
- Recovery local QA25PASS using existing Miniconda; default Python lacks torch
  (collection failed before execution). No environment created or modified.
- New fitting/selection ONLY969420–969431. Frozen priors inheritFIT32; no new
  other TRAIN tapes. No reserved raw tasks/starts or WMDATA2 used. Historical
  owner-required aggregates read; mixed JSON parsed only to extract DEV rows.
- All learned corrections OFF. INT owns final source/recipe freeze, extended
  recorded-world exposure audit, replicated gate/CIs and teacher authorization.
