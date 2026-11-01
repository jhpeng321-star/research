# WM-PHYS v5 — integration progress; final comparison pending

**INTERIM16:46: on the first two matched worlds in one replicated traffic cell,
v5 has worse delay CRPS than v4 (+206/+2,333/+11,428veh-s at120/300/600s), while
HDV60 position error improves by8.7m.** This small available slice cannot select a
model or authorize a planner/cooperation switch. Complete reserved8 and the final
registered comparison remain pending. Constant-column equal-endpoint CIs in the
first snapshot are superseded by NA after a numerical-zero variance correction;
point estimates are unchanged. See the detailed INTERIM section below.

**The four fixes are integrated, but round-2 performance is not yet established.**
FULL-v5 and FULL-v5-LP pass the local physical integration tests and two-world
60s recorded-start smoke. FULL-v5 also passes the all-five-arm600s development
canary; LP passes its separate full canary too. Both are engineering-ready for the
teacher-quality rerun, frozen before new final outcomes.
The matched final evaluation is now running on both GPUs; distributional scores,
confidence intervals and switching eligibility are pending. This is not the
completed Nov2 delivery.

## What exists

- Immutable engineering model stage: sourceb2df7458 on gpuA; exact roots, owned
  PIDs, resource admission and completion receipts: `../rb2int/CURRENT_JOBS.md`.
- FULL-v5 combines causal fitted LC continuation, physical latent-spread fitting,
  current-lane SUMO IDM speed limits and consistent CAV admission/execution settings.
- FULL-v5-LP differs only by the learned physical departure-parameter mixture.
  Neither candidate adds a neural residual or learned acceleration/lane/merge/
  admission correction. The original WM2 checkpoint/encoder is preserved.
- Integration repaired an actual composition defect: attaching LC parameters
  rebuilt the base hidden state and silently removed CAL's entrant-mixture wrapper.
  The joint object now preserves both behaviors; regression gate passes.
- All-four-source integration135CPU PASS; infrastructure56PASS/1historical-canary
  skip; LANE intake92PASS/1CUDA skip; no successful final forecast repeated.

## Registered comparison and limits

- WM2 versus FULL-v4 versus FULL-v5 versus FULL-v5-LP; all five original arms,
  K8, H120/300/600. Reserved8 plus all50 recorded WMDATA2 worlds, metadata-selected
  before new final outcomes. Exact design/registration: `../rb2int/`.
- Primary54 worlds occupy eight replicated traffic cells. Four remaining cells
  are singletons and descriptive. They cannot supply full-panel fixed-stratum CIs.
  The original full12-cell gate and its rate-precision standard remain unchanged;
  this recorded panel cannot establish full qualification.
- Paired whole-world fixed-cell uncertainty; finite-K uncertainty for forecast
  rate means; conservative simultaneous interval propagation for absolute rate
  distances. No arm/future pseudoreplication or unscored-peer bootstrap variation.
- Staged parameter-artifact declarations and inherited hashes show no direct
  final-panel fit/selection overlap. Historical aggregate exposure is disclosed;
  these are not historically untouched research data. Final release remains gated.

## Remaining work / teacher status

Baseline/source/input parity and final release are complete. Execute only the
scheduled missing identity-bound tasks; score all models on matched
complete K8 worlds, distributional components, credibility and switching rule.
Keep source-bound v4 decision maps separate from v5. The FULL-v5 READY signal and
immutable source/recipe/smoke binding in HANDOFF.md are now published for
S-WMTEACH's engineering/quality rerun; final qualification is pending.

<!-- RB2-INTERIM:BEGIN -->
## INTERIM v5 versus v4 — 20261101T164650+0800 (available_replicated_cells)

**2 matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.
Worlds: 969436, 969590.
CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.
Strata: [{'demand': 1900, 'penetration': 0.05, 'worlds': 2}].

### Delay, breakdown and HDV60
- FULL-v4 delay CRPS120/300/600=1,224/6,425/20,056veh-s; mean bias=1,476/7,175/23,910veh-s.
- FULL-v5 delay CRPS120/300/600=1,429/8,758/31,484veh-s; mean bias=1,727/10,101/37,800veh-s.
- 120s v5−v4 delay CRPS: 205.6 [95% CI -17.1, 428.2]veh-s; breakdown absolute rate-distance: -22.50 [95% CI -75.00, 44.98]pp.
- 300s v5−v4 delay CRPS: 2,333.2 [95% CI 333.2, 4,333.1]veh-s; breakdown absolute rate-distance: 7.50 [95% CI -21.34, 41.67]pp.
- 600s v5−v4 delay CRPS: 11,427.9 [95% CI 6,815.6, 16,040.1]veh-s; breakdown absolute rate-distance: 0.00; CI NA (insufficient support/variance)pp.
- v5−v4 HDV60 mean-world-median position-error difference: -8.7 [95% CI -12.2, -5.1]m.

### Controller-effect signs and credibility
- FULL-v4/120s:0.8calls n=8, observed win 25.0; CI NA (insufficient support/variance)% vs forecast 95.3 [95% CI 90.6, 100.0]%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 40.0 [95% CI 40.0, 40.0]%.
- FULL-v4/300s:0.8calls n=2, observed win 50.0; CI NA (insufficient support/variance)% vs forecast 87.5; CI NA (insufficient support/variance)%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 75.0 [95% CI 60.0, 90.0]%.
- FULL-v4/600s:0.8calls n=2, observed win 50.0; CI NA (insufficient support/variance)% vs forecast 87.5; CI NA (insufficient support/variance)%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 50.0 [95% CI 40.0, 60.0]%.
- FULL-v4 ctrl0−native expected delay effect120/300/600: -0.9 [95% CI -7.0, 5.2] / -175.2 [95% CI -771.9, 421.6] / 222.2 [95% CI -3,329.4, 3,773.8]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -3.3 [95% CI -26.2, 19.6] / 2,560.0 [95% CI -4,278.0, 9,398.0]veh-s. Negative means lower delay.
- FULL-v4 mappo−native expected delay effect120/300/600: -0.3 [95% CI -7.0, 6.4] / -217.1 [95% CI -698.8, 264.5] / -838.9 [95% CI -3,605.0, 1,927.2]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -550.1 [95% CI -2,025.8, 925.6] / 186.6 [95% CI -2,669.4, 3,042.5]veh-s. Negative means lower delay.
- FULL-v4 pi−native expected delay effect120/300/600: -5.0 [95% CI -14.0, 4.0] / -95.6 [95% CI -378.6, 187.3] / -694.2 [95% CI -3,082.1, 1,693.7]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -7.1 [95% CI -22.5, 8.3] / 3,442.7 [95% CI -418.3, 7,303.8]veh-s. Negative means lower delay.
- FULL-v4 circles−native expected delay effect120/300/600: 11.4 [95% CI -7.3, 30.0] / 213.6 [95% CI -450.7, 877.9] / 311.7 [95% CI -3,477.4, 4,100.8]veh-s; recorded 3.2 [95% CI -1.5, 7.8] / 54.4 [95% CI -36.5, 145.3] / 3,174.1 [95% CI 1,927.1, 4,421.0]veh-s. Negative means lower delay.
- FULL-v4 native−ctrl0 expected delay effect120/300/600: 0.9 [95% CI -5.2, 7.0] / 175.2 [95% CI -421.6, 771.9] / -222.2 [95% CI -3,773.8, 3,329.4]veh-s; recorded 0.0; CI NA (insufficient support/variance) / 3.3 [95% CI -19.6, 26.2] / -2,560.0 [95% CI -9,398.0, 4,278.0]veh-s. Negative means lower delay.
- FULL-v4 mappo−ctrl0 expected delay effect120/300/600: 0.6 [95% CI -2.9, 4.0] / -42.0 [95% CI -260.5, 176.6] / -1,061.1 [95% CI -4,728.9, 2,606.6]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -546.8 [95% CI -1,999.6, 906.0] / -2,373.4 [95% CI -6,355.5, 1,608.6]veh-s. Negative means lower delay.
- FULL-v4 pi−ctrl0 expected delay effect120/300/600: -4.1 [95% CI -10.0, 1.7] / 79.5 [95% CI -449.4, 608.4] / -916.4 [95% CI -4,488.0, 2,655.2]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -3.7 [95% CI -11.3, 3.8] / 882.7 [95% CI -2,094.2, 3,859.7]veh-s. Negative means lower delay.
- FULL-v4 circles−ctrl0 expected delay effect120/300/600: 12.3 [95% CI -7.5, 32.1] / 388.7 [95% CI -567.7, 1,345.1] / 89.5 [95% CI -3,878.9, 4,057.9]veh-s; recorded 3.2 [95% CI -1.5, 7.8] / 57.8 [95% CI -56.0, 171.6] / 614.1 [95% CI -4,977.0, 6,205.1]veh-s. Negative means lower delay.
- FULL-v5/120s:0.8calls n=2, observed win 100.0; CI NA (insufficient support/variance)% vs forecast 93.8 [95% CI 87.5, 100.0]%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 40.0 [95% CI 40.0, 40.0]%.
- FULL-v5/300s:0.8calls n=2, observed win 50.0; CI NA (insufficient support/variance)% vs forecast 93.8; CI NA (insufficient support/variance)%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 80.0 [95% CI 70.0, 90.0]%.
- FULL-v5/600s:0.8calls n=1, observed win 100.0; CI NA (insufficient support/variance)% vs forecast 87.5; CI NA (insufficient support/variance)%; range coverage 50.0 [95% CI 0.0, 100.0]%; all10pair direction accuracy 70.0 [95% CI 70.0, 70.0]%.
- FULL-v5 ctrl0−native expected delay effect120/300/600: -0.2 [95% CI -11.2, 10.8] / 55.5 [95% CI -465.4, 576.3] / -6.4 [95% CI -4,159.9, 4,147.0]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -3.3 [95% CI -26.2, 19.6] / 2,560.0 [95% CI -4,278.0, 9,398.0]veh-s. Negative means lower delay.
- FULL-v5 mappo−native expected delay effect120/300/600: 1.1 [95% CI -10.3, 12.4] / 23.3 [95% CI -479.7, 526.3] / -482.3 [95% CI -3,614.9, 2,650.3]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -550.1 [95% CI -2,025.8, 925.6] / 186.6 [95% CI -2,669.4, 3,042.5]veh-s. Negative means lower delay.
- FULL-v5 pi−native expected delay effect120/300/600: 9.5 [95% CI -52.4, 71.4] / -80.8 [95% CI -377.3, 215.6] / 229.9 [95% CI -2,396.6, 2,856.5]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -7.1 [95% CI -22.5, 8.3] / 3,442.7 [95% CI -418.3, 7,303.8]veh-s. Negative means lower delay.
- FULL-v5 circles−native expected delay effect120/300/600: 27.2 [95% CI -46.8, 101.2] / 269.9 [95% CI -502.5, 1,042.3] / 817.8 [95% CI -2,775.2, 4,410.9]veh-s; recorded 3.2 [95% CI -1.5, 7.8] / 54.4 [95% CI -36.5, 145.3] / 3,174.1 [95% CI 1,927.1, 4,421.0]veh-s. Negative means lower delay.
- FULL-v5 native−ctrl0 expected delay effect120/300/600: 0.2 [95% CI -10.8, 11.2] / -55.5 [95% CI -576.3, 465.4] / 6.4 [95% CI -4,147.0, 4,159.9]veh-s; recorded 0.0; CI NA (insufficient support/variance) / 3.3 [95% CI -19.6, 26.2] / -2,560.0 [95% CI -9,398.0, 4,278.0]veh-s. Negative means lower delay.
- FULL-v5 mappo−ctrl0 expected delay effect120/300/600: 1.3 [95% CI -1.8, 4.4] / -32.2 [95% CI -213.0, 148.6] / -475.8 [95% CI -3,054.1, 2,102.5]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -546.8 [95% CI -1,999.6, 906.0] / -2,373.4 [95% CI -6,355.5, 1,608.6]veh-s. Negative means lower delay.
- FULL-v5 pi−ctrl0 expected delay effect120/300/600: 9.7 [95% CI -46.9, 66.3] / -136.3 [95% CI -653.3, 380.6] / 236.4 [95% CI -4,390.3, 4,863.0]veh-s; recorded 0.0; CI NA (insufficient support/variance) / -3.7 [95% CI -11.3, 3.8] / 882.7 [95% CI -2,094.2, 3,859.7]veh-s. Negative means lower delay.
- FULL-v5 circles−ctrl0 expected delay effect120/300/600: 27.4 [95% CI -41.0, 95.9] / 214.4 [95% CI -433.5, 862.2] / 824.3 [95% CI -2,412.0, 4,060.5]veh-s; recorded 3.2 [95% CI -1.5, 7.8] / 57.8 [95% CI -56.0, 171.6] / 614.1 [95% CI -4,977.0, 6,205.1]veh-s. Negative means lower delay.

### Short-horizon decision quality:30/60/90s
Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm's recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.
These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.
- FULL-v4/30s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 45.0 [95% CI 40.0, 50.0]%.
- FULL-v4/60s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 40.0 [95% CI 40.0, 40.0]%.
- FULL-v4/90s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 40.0 [95% CI 40.0, 40.0]%.
- FULL-v5/30s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 45.0 [95% CI 40.0, 50.0]%.
- FULL-v5/60s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 40.0 [95% CI 40.0, 40.0]%.
- FULL-v5/90s: team direction accuracy identification lower 0.0; CI NA (insufficient support/variance)%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 0.0; CI NA (insufficient support/variance)%; resolved-only accuracy NA; CI NA (insufficient support/variance)%; road-only direction accuracy 40.0 [95% CI 40.0, 40.0]%.

Strict paired P(beats incumbent)>=0.8 plus lower expected delay; ties are non-wins. K8 min–max range is not a90% interval (continuous exchangeability reference7/9). Zero-selected worlds remain in resampling. Full gate/switching: NOT ESTABLISHED at this interim.


## Frozen-candidate engineering limits (later worker development evidence)

LANE's later free-running development diagnostic rejects fitted-LC adoption on
lane-frequency/recall evidence. BREAK reports that frozen116b2147 targets a retired
mode2 learned mirror, not the actual shared CAV actuator. Therefore the frozen v5
switch must not be described as verified mode2 actuator consistency. These later
findings do not change sourceb2df7458 or the running candidates; all outcomes remain
reported. LP remains a narrow learned departure-parameter variant. No integrated
performance or cooperation/planner authorization follows from engineering readiness.
<!-- RB2-INTERIM:END -->
