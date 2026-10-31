# WM-PHYS morning report — interim, 2026-11-01 05:25 +08
Four matched reserved worlds favor FULL-v3 on delay AND60s-position POINT estimates; switching is not eligible.
Its300s breakdown rate is farther from real on this subset. Full-panel evidence is still pending.

## Candidate and engineering evidence
- Interim FULL-v1 source4015b641: SUMO IDM, LC2013 highway subset, merge/insertion;
  all learned motion/lane/merge/courtesy/admission/noise corrections OFF.
- A2(encoder+UNC,source4015):6worlds;ΔCRPS120/300/600−501/−2591/−9243;Δrate-distance300−3.75pp.
- B2(defaults,source4015):1world;ΔCRPS−2492/−8183/−19790;Δrate-distance300−2.50pp; not comparable to A's set.
- C2(fitted population point) pending. A/B HDV60 unavailable; their95%world CIs NA; no adoption claim.
- FULL uses shrinkage_v1 (learned strength19.35568) and observable
  entrance-counter Poisson family, with Gamma rate uncertainty and action-paired draws.
- FULL-v1's saved-ID crosswalk remains unverified. NEW FULL-v3 sourcef378ac60 conditions
  each start's causal motion directly, removes that lookup, and uses physics-native CAVs.
  Fitted LC remains OFF; no population refit, future-state inputs or neural training.
- Latest source22cc32f6 integrates final LANE advice and explicit move→LC→emit insertion;
  touched-module287CPU PASS/4explicit skips. Prior CUDA/recorded dense-fast parity retained.
  A/B/C/FULL each passed60s on FIT969420/969421: finite, conserved, no learned corrections.
- FULL-v1/v3 panels retained. FULL-v4 five-arm600s canary PASS370.48s, reused once;
  workers398268/269/270 +monitor398303 now live on gpuA; source22cc32f6, no efficacy claim yet.

## Distribution comparison and interim checkpoints
- Registered20worlds969420..969439, five arms/K8/H120/300/600; reserved8 separate.
  FULL processes reserved8 first, without changing membership, seeds or predictions.
### ~04:00 interim (snapshot03:52)
- Four complete paired K8 worlds:969432/433/434/436, all five arms. Other worlds are pending;
  incomplete ensembles excluded. FULL-v1 delay CRPS differences120/300/600 =−785/−3512/−11041veh-s.
- Upstream300s FULL-v1/WM2/real rates91.875/93.125/100%; closeness is worse in point estimate.
  CORRECTION05:25: old04:00 CIs were invalid; bootstrap included unscored same-cell peers. Retracted.
### ~08:00 interim
- Pending. Final candidate freeze09:30, full evaluation09:30–11:30,
  final report11:45 per revised owner schedule; this is not the final candidate freeze.
- Actual05:15:55 FULL-v3 snapshot:4paired K8 worlds969432/433/434/436; WM2 baseline160/160tasks.
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
- SAME FULL-v3−WM2 delay CRPS(veh-s):120/300/600 =−573/−3359/−11150;95%world CIs NA.
  Corrected registered bootstrap uses ONLY scored worlds;4singleton strata have no replication.
- Breakdown closeness300s: |FULL-v3−real|−|WM2−real|=+2.50pp;95%CI NA; rates90.625/93.125/100%.
- HDV60s position error FULL-v3−WM2: −33.94m;paired-world95%CI NA (singleton strata);
  mean-world-median error on574common HDV/arm endpoints;550factual exits/5junctions/41predicted exits censored.
- All three registered conditions must be satisfied before S-C1SWITCH changes its WM arm.
  Current eligibility: NOT ESTABLISHED; missing evidence is not a measured regression.

## Credibility and next steps
- DECIDE SAME4worlds: selected0.8calls win6/11,6/10,9/11 at120/300/600s vs94.3/92.5/97.7%forecast;
  delay min-max coverage20/40/25%. Confidence is not qualified; credibility world CIs also NA.
- Physics is still a highway subset: synchronous LC negotiation and cold initial lane
  memory limit parity. MERGE nativeFIT32 latestH300 ramp2188 vs2273real is a separate1future diagnostic.
  Complete paired panels, evaluate source phase and controller shifts, then assess the09:30 freeze.
- Evidence: docs/rbint/full_v3_gpu/{paired_0525_corrected,distribution_0525_corrected,credibility_0516}/;
  docs/rbint/hdv60/paired_v3_0525_corrected/; HANDOFF.md. Historical0545-labelled outputs superseded.
