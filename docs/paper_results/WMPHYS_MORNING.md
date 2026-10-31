# WM-PHYS morning report — interim, 2026-11-01T07:34:44+08:00
Physics candidates improve several point scores; adoption and calibrated action confidence remain unestablished.
Latest engineering source is FULL-v4/22cc32f6; current paired diagnostic below is FULL-v4.

## Model and engineering
- SUMO IDM +LC2013 highway subset +merge/insertion; all learned corrections OFF; no neural training.
- FULL-v4 uses final LANE advice and move→LC→emit, coupled entrant settings, no birth-tick motion/control.
- FULL posterior: frozen FIT shrinkage_v1 joint support, strength19.35568 already applied, causal own-start motion;
  unknown drivers/entrants use learned population; Gamma-Poisson arrivals use causal counters. Fitted LC OFF.
- A encoder+hidden settings; B population default; C motion-fitted population point; FULL fitted posterior.
- Touched modules287CPU PASS/4explicit skips; later endpoint/bootstrap21PASS/1skip and credibility18PASS.
- A/B/C/FULL each passed60s on FIT969420/969421 (8rollouts): finite, conserved, no learned corrections.
- FULL-v4 five-arm600s canary PASS370.48s/reused once; live sources/jobs retained, own-PID11:30 deadline.

## Distributional evidence (per-variant available cases; not a cross-variant ranking)
- WM2 baseline full20worlds/fivearms/K8; all candidates share registered969420..969439, H120/300/600.
- Development12/reserved8 remain separate; incomplete ensembles never count as K8. Contrasts below reserved only.
- A2:n=8, ΔCRPS120/300/600=-526/-3175/-12100veh-s; Δ300s rate-distance=2.81pp.
- B2:n=8, ΔCRPS120/300/600=-75/58/-2643veh-s; Δ300s rate-distance=4.06pp.
- C2:n=1, ΔCRPS120/300/600=-225/-1501/-9030veh-s; Δ300s rate-distance=-2.50pp.
- FULL-v3:n=8, ΔCRPS120/300/600=-390/-3047/-11811veh-s; Δ300s rate-distance=5.94pp.
- FULL-v4:n=6, ΔCRPS120/300/600=-357/-2569/-8827veh-s; Δ300s rate-distance=-5.83pp.
- Same-world absolute delay CRPS120/300/600:WM2 1709/6994/24727; FULL-v4 1352/4425/15900veh-s (all five arms).
- Same-world FULL-v4600s mean delay bias=17171veh-s; severity alone does not establish clearing fidelity.
- Same-world upstream rates120/300/600 (%):real 46.7/90.0/93.3; WM2 64.6/83.7/90.8; FULL-v4 80.0/89.6/98.3.
- FULL-v4 native600s upstream speed CRPS=3.02 vs cell=2.42m/s; clearing not qualified.
- Same native onset-distribution IBS600=0.270 vs cell=0.281; onset-CDF gate retained.
- Expected600s mappo−native delay=-1702 vs recorded=-2477veh-s (same available worlds).
- Expected600s pi−native delay=-1230 vs recorded=-1805veh-s (same available worlds).
- Integrated ramp-delay target remains unavailable; separate nativeFIT32 H300 diagnostic2188 vs2273real is not this panel.
- Final conditional LC FIT32:default precision3.006%/recall37.942%; encoder2.400%/40.492%; keep-right false positives dominate.
- SUMOCORE DIFFERENT8world/3arm/K3:inferred CRPS647/4635/16728 vs defaults1087/5452/19343;
  cell1433/4375/9131.300s rates inferred84.7/default88.9/real62.5%;18/144 startup collisions, validity FAIL.
  Inferred600s MAPPO/PI effects+3450/+825 vs real−2064/−3858veh-s; not a clean information ceiling.

## Switching-eligibility rule
- FULL-v4−WM2 delay CRPS differences (veh-s),120/300/600:-357;95%CI NA / -2569;95%CI NA / -8827;95%CI NA.
- 300s breakdown-distance difference (pp): -5.83;95%CI NA.
- HDV60 mean-world-median error difference (m): -14.08;95%CI NA; common HDV/arm endpoints=997.
- Full-world95%CIs require replication in EVERY scored demand×penetration×split cell; any singleton gives NA here.
- Detailed mixed-cell bootstrap intervals condition on singleton worlds; they are not full-panel confidence claims.
- Earlier04:00/first0545-labelled numeric CIs RETRACTED (unscored-peer bootstrap bug); point estimates unchanged.
- ALL three conditions are required; no automatic S-C1SWITCH adoption. Sparse cells also prevent distributional qualification.

## Credibility / schedule
- Strict paired P(beats incumbent)>=0.8 plus lower expected delay is only a decision rule; calibration must also hold.
- Same-source selected-switch reliability and50/80/90%/range coverage: E:\agent-work\c4-rb-int\docs\rbint\noon\20261101T073444+0800\FULL-v4\credibility/CREDIBILITY.md.
- Same-source selected0.8calls observed win%=42.9/50.0/70.6 vs forecast%=92.0/92.7/92.6; range coverage%=26.7/53.3/43.3.
- ~04:00 interim retained:FULL-v1 n4 ΔCRPS−785/−3512/−11041; rate-distance+1.25pp; historical CIs retracted.
- ~08:00 interim: PENDING; final engineering freeze09:30: PENDING.
- Final evaluation receipts through11:30, final report11:45; frozen source is not selected by reserved outcome tuning.
- Remaining gaps:serial LC negotiation/internal junction state/cold memory; finish matched panels and validate action confidence.
- Audit/source manifests, per-world attrition, timing/speed/shift/gate/credibility files: docs/rbint/noon/20261101T073444+0800.
