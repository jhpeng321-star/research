# WM-PHYS morning report — interim, 2026-11-01 03:55 +08
Four matched reserved worlds favor FULL-v1 on delay distributions; the switching rule is not satisfied yet.
Breakdown-rate closeness has not improved on this subset, and60s position-error comparison is pending.

## Candidate and engineering evidence
- Interim FULL-v1 source4015b641: SUMO IDM, LC2013 highway subset, merge/insertion;
  all learned motion/lane/merge/courtesy/admission/noise corrections OFF.
- A: motion encoder + hidden settings/Poisson arrivals; B: population defaults;
  C: motion-fitted pooled point; FULL: fitted joint IDM posterior + learned arrivals.
- FULL uses shrinkage_v1 (learned strength19.35568) and observable
  entrance-counter Poisson family, with Gamma rate uncertainty and action-paired draws.
- FULL-v1's saved-ID crosswalk remains unverified. NEW FULL-v3 sourcef378ac60 conditions
  each start's causal motion directly, removes that lookup, and uses physics-native CAVs.
  Fitted LC remains OFF; no population refit, future-state inputs or neural training.
- Latest touched-module CPU gate270PASS/2CUDA skips; staged FIT tests31PASS on cpuQ;
  CUDA suites6+4PASS, with bit-exact recorded60s dense/fast motion tapes.
  A/B/C/FULL each passed60s on FIT969420/969421: finite, conserved, no learned corrections.
- FULL-v1 canary206s reused in its panel. FULL-v3 GPU canary is live; candidate-only
  lane arbitration is17.6x faster in a synthetic kernel benchmark, not a full-rollout claim.

## Distribution comparison and interim checkpoints
- Registered20worlds969420..969439, five arms/K8/H120/300/600; reserved8 separate.
  FULL processes reserved8 first, without changing membership, seeds or predictions.
### ~04:00 interim (snapshot03:52)
- Four complete paired K8 worlds:969432/433/434/436, all five arms. Other worlds are pending;
  incomplete ensembles are excluded. Delay CRPS improves at all three horizons (CIs below).
- Upstream300s FULL-v1/WM2/real rates91.875/93.125/100%; closeness is worse in point estimate.
  This is an available-case interim on4of8reserved worlds, not complete-panel adoption evidence.
### ~08:00 interim
- Pending. Final candidate freeze09:30, full evaluation09:30–11:30,
  final report11:45 per revised owner schedule; this is not the final candidate freeze.
- Matched WM2 vs A/B/C/FULL breakdown rate/onset, delay CRPS, clearing/speed fields,
  ramp delay and expected-controller shifts: pending; no win/CI is fabricated.
- S-SUMOCORE is complete on a DIFFERENT8world/3arm/K3 contract: inferred all-arm
  delay CRPS120/300/600 =647/4635/16728veh-s, population1087/5452/19343;
  cell climatology1433/4375/9131. These are context, not matched WM-PHYS contrasts.
- SUMOCORE300s upstream rate inferred84.7%, defaults88.9%, real62.5%; inferred
 600s MAPPO/PI effects vs native +3450/+825veh-s vs real−2064/−3858. Broad CIs;
 18/144 initialization collision flags mean this is not a clean information ceiling.
- Latest source LC diagnostic precision2.848%, recall34.163%; improves recall over
  paired historical MOBIL7.50%, but too many proposals remain. Not full traffic parity.

## Switching-eligibility rule
- FULL-v1−WM2 delay CRPS(veh-s):120s −785[−1264,−306];300s −3512[−5825,−1200];
  600s −11041[−16268,−5815]. Paired95%simultaneous world CIs, conditional on saved K8, n=4.
- Breakdown closeness300s: |FULL-v1−real|−|WM2−real|=+1.25pp[−4.17,+6.67], same95%CI.
- HDV60s position error: candidate−WM2, paired-world95%CI: PENDING.
  Old tasks lack endpoints; new forecasts save observed-ID endpoints; baseline/scoring gate next.
- All three registered conditions must be satisfied before S-C1SWITCH changes its WM arm.
  Current eligibility: NOT ESTABLISHED; missing evidence is not a measured regression.

## Credibility and next steps
- S-RB-DECIDE integrated (15CPU tests): paired P(strict beats incumbent)>=0.8 plus improved mean delay;
  calibration/reliability and coverage must be reported separately from this decision rule.
- Physics is still a highway subset: synchronous LC negotiation and cold initial lane
  memory limit parity. Finish five-arm canary, release new A/B/C/FULL sources, collect
  complete K8 worlds, add paired endpoint CIs, and assess expected controller shifts.
- Evidence: docs/rbint/full_v1/reports/paired_0400/COMPARISON.json; HANDOFF.md; peer reports.
