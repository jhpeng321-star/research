# WM-PHYS morning report — live integration receipt, 2026-11-01 02:52 +08
The rebuilt physics runs, but it has not yet shown that it predicts traffic better than WM2.
Keep the switching decision pending until matched distribution and60s-error evidence exists.

## Candidate and engineering evidence
- Interim FULL-v1 source4015b641: SUMO IDM, LC2013 highway subset, merge/insertion;
  all learned motion/lane/merge/courtesy/admission/noise corrections OFF.
- A: motion encoder + hidden settings/Poisson arrivals; B: population defaults;
  C: motion-fitted pooled point; FULL: fitted joint IDM posterior + learned arrivals.
- FULL uses shrinkage_v1 (learned strength19.35568, already applied) and observable
  entrance-counter Poisson family, with Gamma rate uncertainty and action-paired draws.
- Unknown fitted-driver identities use the learned population prior; no reserved-world
  causal posterior refit yet. Fitted LC remains OFF because it loses its baseline.
- Latest touched-module CPU gate155PASS; exact staged FIT tests17PASS on cpuQ.
  A/B/C/FULL each passed60s on FIT969420/969421: finite, conserved, no learned corrections.
- RUNA's original A/C snapshot stopped before rollouts on a missing MAPPO native-lane
  ownership option. Failures retained; default-OFF source port fixes the constructor.
- FULL-v1 five-arm600s GPU canary PASSED206.03s; reused in the live three-shard panel.

## Distribution comparison and interim checkpoints
- Registered20worlds969420..969439, five arms/K8/H120/300/600; reserved8 separate.
  FULL processes reserved8 first, without changing membership, seeds or predictions.
- Early partial02:52: FULL-v1 has3complete reserved worlds; WM2 shares only969432.
  On that ONE paired world, CRPS120/300/600 differences are +60/+635/−1672veh-s;
  upstream300s rate95% vs WM287.5% vs real100%. World-sampling CIs unavailable.
- 04:00 interim: pending complete paired K8 worlds; partial ensembles will not be scored.
- 08:00 interim: pending. Final candidate freeze09:30, full evaluation09:30–11:30,
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
- Delay CRPS: early FULL-v1−WM2 +60/+635/−1672 at120/300/600s;95%CIs unavailable(n=1).
- Breakdown closeness: early300s |FULL-v1−real|−|WM2−real|=−7.5pp;95%CI unavailable(n=1).
- HDV60s position error: candidate−WM2, paired-world95%CI: PENDING.
  Existing saved bin/clock tasks lack endpoints; a separate paired60s gate is requested.
- All three registered conditions must be satisfied before S-C1SWITCH changes its WM arm.
  Current eligibility: NOT ESTABLISHED; missing evidence is not a measured regression.

## Credibility and next steps
- S-RB-DECIDE integrated (15CPU tests): paired P(strict beats incumbent)>=0.8 plus improved mean delay;
  calibration/reliability and coverage must be reported separately from this decision rule.
- Physics is still a highway subset: synchronous LC negotiation and cold initial lane
  memory limit parity. Finish five-arm canary, release new A/B/C/FULL sources, collect
  complete K8 worlds, add paired endpoint CIs, and assess expected controller shifts.
- Evidence: HANDOFF.md; WMPHYS_SMOKE_0ec484ab.json; SUMOCORE peer report; RB_LANE.md.
