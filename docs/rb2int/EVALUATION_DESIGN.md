# WM-PHYS round-2 comparison — prospective design

2026-11-01. Owner direction: compare fixed FULL-v5 and FULL-v5-LP against frozen
FULL-v4 and WM2 by 2026-11-02 ~10:00 +08. WM inference only; zero new SUMO.
This design precedes new final-panel outcome inspection and candidate selection.
Exact identities/cells are in `PANEL_METADATA.json`, SHA256
`422eb8d11ec6b78c504ca45ea53c52c3fe4630d6dc419730ca746750a2b5e501`.

## Plain-language scope

The recorded data can support a larger, replicated comparison, but cannot support
a precise all-traffic qualification claim. Use all eight reserved WMDATA worlds
plus all 50 WMDATA2 worlds. Four of the 12 traffic cells still have one world.
The predeclared primary comparison therefore uses the **54 worlds in the eight
replicated cells**. The other four worlds are descriptive. This is a scoped
comparison, not a relaxation or PASS of the original 12-cell distributional gate.
Even the replicated cells are small; rate precision can independently block PASS.

## Exposure / release requirements

- Fit/diagnose/select only on 969400–969431 and declared previously exposed TRAIN
  tapes. All workers must attest to exact fit/selection worlds and inherited
  artifact/checkpoint lineage. Audit the actual delivered artifacts, not just
  workers' intent. No final execution is authorized by the metadata audit alone.
- Prospective quarantine: 969432–969439 and 969550–969599 cannot enter round-2
  fit/diagnosis/selection. Any disclosed overlap invalidates affected primary
  results; retain it as an explicit exposure finding, not an outcome-based swap.
- Historical aggregates and reports were already exposed. WMDATA2 was collected
  as TRAIN/FIT and used by other research streams. These are not historically
  untouched data. The audit must establish no fitting/selection overlap in the
  **actual four model lineages**, and disclose broader historical exposure.
- Freeze one FULL-v5 and one FULL-v5-LP using development evidence, exact source
  commits, recipe/artifact hashes, learned-correction-OFF checks and CPU/smoke
  receipts before viewing new final scores. Both candidates are reported; no
  choice between them based on final results. WM2 checkpoint b8797242… and
  FULL-v4 source22cc32f6 remain immutable.

## Panel / estimands / execution

- All 58 worlds, all five original arms: native, CTRL0, speed-only MAPPO iter169,
  PI external-U, CIRCLES damping. Generic-random is excluded prospectively.
- t0=300 s, authentic causal histories and observed entrance counters/queues.
  Forecasts open-loop to 600 s; score H120/300/600 with the unchanged registered
  BIC downward-step+trend detector, regions and 10 s bin definitions.
- K=8 complete hidden-driver/arrival futures per world, shared across arms.
  Same recorded observation and controller/source contracts for all models.
  More futures require a prospective amendment before final scores; no
  outcome-adaptive K, replacement worlds, retries or model selection.
- Primary54 estimand: equal-world average at the frozen recorded cell mixture.
  Each world's five arms are equally weighted; each world remains one cluster.
  Fixed-cell summaries accompany the average. No claim about uniform12-cell
  populations. The full58 and reserved8 results are secondary descriptive views;
  singleton fixed-cell intervals are NA, never conditional bootstrap intervals
  described as full-panel CIs.
- Preserve completed, identity-bound reserved8 WM2/FULL-v4 predictions and
  endpoints. Run only the missing extended tasks in separate immutable roots.
  Reuse requires exact source/start/controller/RNG/observer binding; any interface
  parity bridge is verified on development worlds before final scoring. No
  duplicate process or task dispatch. Record attrition and failed artifacts.
- Score only complete K8/all-five-arm matched worlds. Report every scheduled
  and missing task. Complete primary54 is required for qualification. If a
  partial diagnostic is published, rebuild strata using ONLY its scored worlds;
  a missing/unreplicated contributing cell prevents full-primary CIs.

## Uncertainty / multiplicity (frozen)

- 10,000 paired whole-world bootstrap resamples, RNG seed20261101. Fixed
  demand×penetration strata; **do not split by collection** in primary54 (target
  is the frozen combined recorded mixture). Report collection-specific points
  separately; that secondary partition does not create replication.
- Keep all arms, horizons, bins, models and driver/arrival future identities
  together inside each world. For paired saved-ensemble CRPS and HDV60 errors,
  intervals condition on the saved K8 predictions; label this explicitly. Rate,
  onset-CDF and expected-shift forecast means also report finite-K uncertainty
  using paired future resampling within world, preserving arm coupling. Do not
  count K futures as factual worlds.
- Centered standardized max-stat simultaneous95% intervals per frozen metric
  family. Primary model comparisons: v5−WM2, v5−v4, LP−WM2, LP−v4, LP−v5, and
  v4−WM2, jointly across the three horizons (18 delay-CRPS contrasts). Report
  fair-K8 CRPS as a declared sensitivity, not a substitute selected after seeing
  results. Breakdown-distance comparisons form their own18-contrast family.
- Only scored-world replication can support bootstrap intervals. Every
  contributing stratum needs >=2 distinct factual worlds. Nonfinite/missing
  support, zero estimable variance, absent metrics or insufficient replication
  yield NA/insufficient evidence. Bootstrap CIs are finite-sample approximations;
  two-world strata are not a precision claim. Wilson rate intervals remain the
  registered distributional gate's real-outcome uncertainty calculation.

## Distributional gate (unchanged full12-cell standard)

Retain the gate in S-WMBREAKDOWN lines32–40: primary upstream cell×arm×horizon
event-rate calibration with simultaneous Wilson95% real intervals and <=10pp
half-width (Bonferroni180); include MC uncertainty; onset CDF with separate
pre-existing/no-onset mass, integrated Brier versus development climatology;
team-delay and regional-speed CRPS versus pooled AND cell×arm/time-bin
climatology at every horizon; exchangeability coverage/PIT; expected controller
delay/event shifts versus native AND CTRL0, resolved signs and arm-order skill.
Climatology uses **only original development969400–969431**, with entire target
world excluded where applicable. No WMDATA2 target enters a fitted baseline.
Rate, onset, traffic-distribution, coverage, action-shift and order families keep
their registered multiplicity corrections. Source/road/ramp delay, speed support,
LC diagnostics and clearing are explicit diagnostics, not omitted endpoints.

The full12-cell gate is NOT QUALIFIABLE from this recorded panel because four
cells are singletons and the rate precision is inadequate. Compute its available
components and state FAIL versus INSUFFICIENT separately. A favorable scoped
comparison does not override any missing full-gate condition.

## Switching eligibility / teacher handoff

For each v5 variant versus WM2: lower delay CRPS at ALL120/300/600, HDV60 position
error no worse, and300s upstream breakdown rate closer to recorded. Report point
conditions and simultaneous interval support separately. Missing endpoint or
uncertainty evidence means NOT ESTABLISHED. A point failure means FAIL. No
automatic controller adoption follows from an engineering freeze.

Also report strict paired P(beats incumbent)>=.8 + lower expected delay calls,
observed versus forecast win probability, selection counts, ties, direction and
50/80/90%/range coverage; use the existing credibility definitions and all worlds
as clusters. Calibration failure is not repaired by selecting a different final
threshold. S-WMTEACH receives FULL-v5 READY only after an immutable source/recipe
and successful development smoke receipt exist; readiness is not qualification.
