# WM-PHYS morning report — interim, 2026-11-01 05:45 +08
Four matched reserved worlds favor FULL-v3 on delay distributions AND60s positions; switching is not eligible.
Its300s breakdown rate is farther from real on this subset. Full-panel evidence is still pending.

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
- Latest source22cc32f6 integrates final LANE advice and explicit move→LC→emit insertion;
  touched-module287CPU PASS/4explicit skips. Prior CUDA/recorded dense-fast parity retained.
  A/B/C/FULL each passed60s on FIT969420/969421: finite, conserved, no learned corrections.
- FULL-v1/v3 panels remain live, canaries reused. NEW FULL-v4 five-arm canary live;
  source22cc32f6 has8x60s recorded smoke PASS; its new insertion phase is not yet panel-validated.

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
- ~05:45 FULL-v3 snapshot:4paired K8 worlds969432/433/434/436; WM2 baseline160/160tasks.
  Native600s upstream speed CRPS3.23 vs cell1.96m/s: clearing/traffic-field parity still fails.
- S-SUMOCORE is complete on a DIFFERENT8world/3arm/K3 contract: inferred all-arm
  delay CRPS120/300/600 =647/4635/16728veh-s, population1087/5452/19343;
  cell climatology1433/4375/9131. These are context, not matched WM-PHYS contrasts.
- SUMOCORE300s upstream rate inferred84.7%, defaults88.9%, real62.5%; inferred
 600s MAPPO/PI effects vs native +3450/+825veh-s vs real−2064/−3858. Broad CIs;
 18/144 initialization collision flags mean this is not a clean information ceiling.
- Final conditional FIT32 LC diagnostic: default precision3.006%/recall37.942%,
  encoder2.400%/40.492%; non-urgent keep-right false positives dominate. Not traffic parity.

## Switching-eligibility rule
- SAME FULL-v3−WM2 delay CRPS(veh-s):120s −573[−979,−166];300s −3359[−5620,−1099];
  600s −11150[−16387,−5913]. Paired95%simultaneous world CIs, conditional on saved K8, n=4.
- Breakdown closeness300s: |FULL-v3−real|−|WM2−real|=+2.50pp[−1.67,+6.67]; rates90.625/93.125/100%.
- HDV60s position error FULL-v3−WM2: −33.94m[−43.91,−23.97], paired-world95%CI;
  mean-world-median error on574common HDV/arm endpoints;550factual exits/5junctions/41predicted exits censored.
- All three registered conditions must be satisfied before S-C1SWITCH changes its WM arm.
  Current eligibility: NOT ESTABLISHED; missing evidence is not a measured regression.

## Credibility and next steps
- DECIDE05:05 matched3worlds: FULL-v3 selected0.8calls win5/9,5/9,9/10 at120/300/600s,
  vs forecast93.1/93.1/98.8%; improved delay means do not establish calibrated action confidence.
- Physics is still a highway subset: synchronous LC negotiation and cold initial lane
  memory limit parity. Finish five-arm canary, release new A/B/C/FULL sources, collect
  complete K8 worlds, add paired endpoint CIs, and assess expected controller shifts.
- Evidence: docs/rbint/full_v3_gpu/{paired_0545,distribution_0545}/; docs/rbint/hdv60/paired_v3_0545/;
  peer c4-rb-decide/docs/paper_results/wmdecide/INTERIM_0505.md; HANDOFF.md. Pilot only, not final freeze.
