# WM-PHYS v5 — INTERIM; final comparison pending

**INTERIM 20261101T183853+0800: on 10 matched complete K8 worlds
(available_replicated_cells), v5−v4 delay CRPS is +167/+1,046/+3,618 veh-s at120/300/600s.**
See the interval-supported detailed results below. This is an available-data
diagnostic, not model selection or a planner/cooperation switching authorization.

**Owner17:55 GPU hold: ZERO new evaluation admissions on either GPU until MAIN
appends `S-RB2-INT RESUME`.** Our current admission coordinators and late standbys
are paused; existing evaluation tasks are now collected.
31/216 new registered world jobs are collected. The frozen full comparison
remains incomplete; this is not final delivery. Cache-only interim publication
continues about every3h, with an additional snapshot when all hold-time tasks are
collected and immediately on complete reserved8. No inference is redispatched.

FULL-v5 and FULL-v5-LP remain engineering-ready at frozen sourceb2df7458.
All four fixes and the all-five-arm600s development canaries are preserved.
Teacher readiness does not override the owner's GPU hold or establish quality.

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
## INTERIM v5 versus v4 — 20261101T183853+0800 (available_replicated_cells)

**10 matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.
Worlds: 969433, 969435, 969436, 969437, 969438, 969439, 969590, 969592, 969595, 969599.
CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.
Strata: [{'demand': 1900, 'penetration': 0.05, 'worlds': 2}, {'demand': 1900, 'penetration': 0.4, 'worlds': 2}, {'demand': 2700, 'penetration': 0.05, 'worlds': 2}, {'demand': 2700, 'penetration': 0.1, 'worlds': 2}, {'demand': 2700, 'penetration': 0.4, 'worlds': 2}].

### Delay, breakdown and HDV60
- FULL-v4 delay CRPS120/300/600=952/3,321/10,003veh-s; mean bias=1,118/3,805/11,589veh-s.
- FULL-v5 delay CRPS120/300/600=1,119/4,367/13,622veh-s; mean bias=1,399/5,485/17,537veh-s.
- 120s v5−v4 delay CRPS: 166.9 [95% CI 64.6, 269.3]veh-s; breakdown absolute rate-distance: 10.50 [95% CI -20.33, 41.33]pp.
- 300s v5−v4 delay CRPS: 1,046.1 [95% CI 356.5, 1,735.7]veh-s; breakdown absolute rate-distance: 2.25 [95% CI -23.77, 27.31]pp.
- 600s v5−v4 delay CRPS: 3,618.5 [95% CI 2,051.0, 5,185.9]veh-s; breakdown absolute rate-distance: 0.00 [95% CI -17.70, 17.14]pp.
- v5−v4 HDV60 mean-world-median position-error difference: 0.7 [95% CI -3.3, 4.8]m.

### Controller-effect signs and credibility
- FULL-v4/120s:0.8calls n=21, observed win 47.6 [95% CI 29.7, 65.5]% vs forecast 92.9 [95% CI 89.3, 96.5]%; range coverage 46.0 [95% CI 13.2, 78.8]%; all10pair direction accuracy 51.0 [95% CI 42.7, 59.3]%.
- FULL-v4/300s:0.8calls n=18, observed win 55.6 [95% CI 32.6, 78.5]% vs forecast 91.0 [95% CI 89.4, 92.5]%; range coverage 62.0 [95% CI 29.0, 95.0]%; all10pair direction accuracy 64.0 [95% CI 53.4, 74.6]%.
- FULL-v4/600s:0.8calls n=16, observed win 56.2 [95% CI 43.4, 69.1]% vs forecast 94.5 [95% CI 90.3, 98.8]%; range coverage 76.0 [95% CI 47.8, 100.0]%; all10pair direction accuracy 46.0 [95% CI 36.5, 55.5]%.
- FULL-v4 ctrl0−native expected delay effect120/300/600: -17.9 [95% CI -100.1, 64.3] / -103.8 [95% CI -671.6, 464.1] / 71.0 [95% CI -1,859.3, 2,001.3]veh-s; recorded -106.2 [95% CI -256.9, 44.5] / -658.3 [95% CI -1,468.0, 151.4] / -980.4 [95% CI -5,399.6, 3,438.7]veh-s. Negative means lower delay.
- FULL-v4 mappo−native expected delay effect120/300/600: 16.4 [95% CI -84.3, 117.0] / -219.1 [95% CI -851.9, 413.6] / -861.9 [95% CI -2,617.9, 894.1]veh-s; recorded 2.1 [95% CI -114.8, 118.9] / -352.2 [95% CI -1,083.3, 379.0] / -1,276.1 [95% CI -3,751.7, 1,199.5]veh-s. Negative means lower delay.
- FULL-v4 pi−native expected delay effect120/300/600: -50.1 [95% CI -169.4, 69.3] / -244.5 [95% CI -968.1, 479.0] / -864.2 [95% CI -3,005.8, 1,277.4]veh-s; recorded 4.4 [95% CI -103.5, 112.3] / -207.9 [95% CI -1,047.4, 631.6] / 57.4 [95% CI -4,512.8, 4,627.7]veh-s. Negative means lower delay.
- FULL-v4 circles−native expected delay effect120/300/600: 245.5 [95% CI 132.8, 358.2] / 1,028.4 [95% CI 401.8, 1,655.0] / 3,279.6 [95% CI 1,554.7, 5,004.6]veh-s; recorded 216.1 [95% CI -41.9, 474.1] / 914.8 [95% CI 202.7, 1,626.8] / 3,541.6 [95% CI 587.7, 6,495.6]veh-s. Negative means lower delay.
- FULL-v4 native−ctrl0 expected delay effect120/300/600: 17.9 [95% CI -64.3, 100.1] / 103.8 [95% CI -464.1, 671.6] / -71.0 [95% CI -2,001.3, 1,859.3]veh-s; recorded 106.2 [95% CI -44.5, 256.9] / 658.3 [95% CI -151.4, 1,468.0] / 980.4 [95% CI -3,438.7, 5,399.6]veh-s. Negative means lower delay.
- FULL-v4 mappo−ctrl0 expected delay effect120/300/600: 34.3 [95% CI -65.4, 133.9] / -115.3 [95% CI -620.3, 389.6] / -932.9 [95% CI -2,821.6, 955.7]veh-s; recorded 108.2 [95% CI -4.3, 220.8] / 306.1 [95% CI -503.8, 1,116.1] / -295.6 [95% CI -2,931.7, 2,340.4]veh-s. Negative means lower delay.
- FULL-v4 pi−ctrl0 expected delay effect120/300/600: -32.1 [95% CI -157.8, 93.5] / -140.8 [95% CI -844.3, 562.8] / -935.2 [95% CI -3,073.2, 1,202.8]veh-s; recorded 110.5 [95% CI -35.6, 256.7] / 450.4 [95% CI -196.6, 1,097.4] / 1,037.9 [95% CI -1,577.4, 3,653.2]veh-s. Negative means lower delay.
- FULL-v4 circles−ctrl0 expected delay effect120/300/600: 263.4 [95% CI 133.6, 393.2] / 1,132.1 [95% CI 537.3, 1,727.0] / 3,208.6 [95% CI 1,465.2, 4,952.0]veh-s; recorded 322.3 [95% CI 106.3, 538.3] / 1,573.1 [95% CI 588.9, 2,557.2] / 4,522.1 [95% CI 392.7, 8,651.5]veh-s. Negative means lower delay.
- FULL-v5/120s:0.8calls n=12, observed win 66.7 [95% CI 32.7, 100.0]% vs forecast 92.7 [95% CI 89.0, 96.5]%; range coverage 40.0 [95% CI 12.6, 67.4]%; all10pair direction accuracy 54.0 [95% CI 40.8, 67.2]%.
- FULL-v5/300s:0.8calls n=12, observed win 75.0 [95% CI 57.4, 92.6]% vs forecast 93.8 [95% CI 91.1, 96.4]%; range coverage 60.0 [95% CI 24.7, 95.3]%; all10pair direction accuracy 61.0 [95% CI 56.0, 66.0]%.
- FULL-v5/600s:0.8calls n=13, observed win 69.2 [95% CI 35.6, 100.0]% vs forecast 94.2 [95% CI 91.6, 96.8]%; range coverage 74.0 [95% CI 45.1, 100.0]%; all10pair direction accuracy 61.0 [95% CI 53.8, 68.2]%.
- FULL-v5 ctrl0−native expected delay effect120/300/600: 25.5 [95% CI -86.4, 137.3] / 236.9 [95% CI -302.9, 776.6] / 564.5 [95% CI -1,209.9, 2,338.9]veh-s; recorded -106.2 [95% CI -256.9, 44.5] / -658.3 [95% CI -1,468.0, 151.4] / -980.4 [95% CI -5,399.6, 3,438.7]veh-s. Negative means lower delay.
- FULL-v5 mappo−native expected delay effect120/300/600: 26.3 [95% CI -50.4, 103.0] / 59.8 [95% CI -376.1, 495.8] / -292.8 [95% CI -1,692.1, 1,106.4]veh-s; recorded 2.1 [95% CI -114.8, 118.9] / -352.2 [95% CI -1,083.3, 379.0] / -1,276.1 [95% CI -3,751.7, 1,199.5]veh-s. Negative means lower delay.
- FULL-v5 pi−native expected delay effect120/300/600: -15.2 [95% CI -99.8, 69.3] / 65.5 [95% CI -423.3, 554.4] / 167.8 [95% CI -1,488.2, 1,823.8]veh-s; recorded 4.4 [95% CI -103.5, 112.3] / -207.9 [95% CI -1,047.4, 631.6] / 57.4 [95% CI -4,512.8, 4,627.7]veh-s. Negative means lower delay.
- FULL-v5 circles−native expected delay effect120/300/600: 247.7 [95% CI 148.3, 347.1] / 1,318.0 [95% CI 682.8, 1,953.1] / 4,737.9 [95% CI 2,712.5, 6,763.3]veh-s; recorded 216.1 [95% CI -41.9, 474.1] / 914.8 [95% CI 202.7, 1,626.8] / 3,541.6 [95% CI 587.7, 6,495.6]veh-s. Negative means lower delay.
- FULL-v5 native−ctrl0 expected delay effect120/300/600: -25.5 [95% CI -137.3, 86.4] / -236.9 [95% CI -776.6, 302.9] / -564.5 [95% CI -2,338.9, 1,209.9]veh-s; recorded 106.2 [95% CI -44.5, 256.9] / 658.3 [95% CI -151.4, 1,468.0] / 980.4 [95% CI -3,438.7, 5,399.6]veh-s. Negative means lower delay.
- FULL-v5 mappo−ctrl0 expected delay effect120/300/600: 0.9 [95% CI -96.9, 98.6] / -177.0 [95% CI -781.2, 427.2] / -857.3 [95% CI -2,783.2, 1,068.5]veh-s; recorded 108.2 [95% CI -4.3, 220.8] / 306.1 [95% CI -503.8, 1,116.1] / -295.6 [95% CI -2,931.7, 2,340.4]veh-s. Negative means lower delay.
- FULL-v5 pi−ctrl0 expected delay effect120/300/600: -40.7 [95% CI -137.6, 56.2] / -171.4 [95% CI -655.3, 312.6] / -396.7 [95% CI -1,994.8, 1,201.4]veh-s; recorded 110.5 [95% CI -35.6, 256.7] / 450.4 [95% CI -196.6, 1,097.4] / 1,037.9 [95% CI -1,577.4, 3,653.2]veh-s. Negative means lower delay.
- FULL-v5 circles−ctrl0 expected delay effect120/300/600: 222.3 [95% CI 82.8, 361.8] / 1,081.1 [95% CI 477.6, 1,684.6] / 4,173.4 [95% CI 2,429.9, 5,917.0]veh-s; recorded 322.3 [95% CI 106.3, 538.3] / 1,573.1 [95% CI 588.9, 2,557.2] / 4,522.1 [95% CI 392.7, 8,651.5]veh-s. Negative means lower delay.

### Short-horizon decision quality:30/60/90s
Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm's recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.
These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.
- FULL-v4/30s: team direction accuracy identification lower 4.0 [95% CI 0.0, 10.3]%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 4.0 [95% CI 0.0, 10.3]%; resolved-only accuracy 100.0; CI NA (insufficient support/variance)%; road-only direction accuracy 63.0 [95% CI 55.9, 70.1]%.
- FULL-v4/60s: team direction accuracy identification lower 10.0 [95% CI 6.9, 13.1]%, upper 98.0 [95% CI 94.9, 100.0]%; resolved-pair fraction 12.0; CI NA (insufficient support/variance)%; resolved-only accuracy 83.3 [95% CI 57.2, 100.0]%; road-only direction accuracy 57.0 [95% CI 49.8, 64.2]%.
- FULL-v4/90s: team direction accuracy identification lower 11.0 [95% CI 6.3, 15.7]%, upper 97.0 [95% CI 92.3, 100.0]%; resolved-pair fraction 14.0; CI NA (insufficient support/variance)%; resolved-only accuracy 78.6 [95% CI 45.0, 100.0]%; road-only direction accuracy 52.0 [95% CI 44.0, 60.0]%.
- FULL-v5/30s: team direction accuracy identification lower 4.0 [95% CI 0.0, 10.3]%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 4.0 [95% CI 0.0, 10.3]%; resolved-only accuracy 100.0; CI NA (insufficient support/variance)%; road-only direction accuracy 64.0 [95% CI 58.6, 69.4]%.
- FULL-v5/60s: team direction accuracy identification lower 10.0 [95% CI 6.9, 13.1]%, upper 98.0 [95% CI 94.9, 100.0]%; resolved-pair fraction 12.0; CI NA (insufficient support/variance)%; resolved-only accuracy 83.3 [95% CI 57.2, 100.0]%; road-only direction accuracy 56.0 [95% CI 46.3, 65.7]%.
- FULL-v5/90s: team direction accuracy identification lower 13.0 [95% CI 11.4, 14.6]%, upper 99.0 [95% CI 97.4, 100.0]%; resolved-pair fraction 14.0; CI NA (insufficient support/variance)%; resolved-only accuracy 92.9 [95% CI 81.7, 100.0]%; road-only direction accuracy 59.0 [95% CI 47.2, 70.8]%.

Strict paired P(beats incumbent)>=0.8 plus lower expected delay; ties are non-wins. K8 min–max range is not a90% interval (continuous exchangeability reference7/9). Zero-selected worlds remain in resampling. Full gate/switching: NOT ESTABLISHED at this interim.

## INTERIM v5 versus v4 — 20261101T183839+0800 (reserved8)

**8 matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.
Worlds: 969432, 969433, 969434, 969435, 969436, 969437, 969438, 969439.
CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.
Strata: [{'demand': 1900, 'penetration': 0.05, 'worlds': 1}, {'demand': 1900, 'penetration': 0.4, 'worlds': 2}, {'demand': 2400, 'penetration': 0.4, 'worlds': 1}, {'demand': 2700, 'penetration': 0.05, 'worlds': 1}, {'demand': 2700, 'penetration': 0.2, 'worlds': 1}, {'demand': 2700, 'penetration': 0.4, 'worlds': 2}].

### Delay, breakdown and HDV60
- FULL-v4 delay CRPS120/300/600=1,105/3,777/13,105veh-s; mean bias=1,263/4,451/14,648veh-s.
- FULL-v5 delay CRPS120/300/600=1,274/4,887/16,101veh-s; mean bias=1,536/6,136/20,502veh-s.
- 120s v5−v4 delay CRPS: 169.0; CI NA (singleton)veh-s; breakdown absolute rate-distance: 8.44; CI NA (singleton)pp.
- 300s v5−v4 delay CRPS: 1,110.3; CI NA (singleton)veh-s; breakdown absolute rate-distance: 2.81; CI NA (singleton)pp.
- 600s v5−v4 delay CRPS: 2,996.2; CI NA (singleton)veh-s; breakdown absolute rate-distance: -0.94; CI NA (singleton)pp.
- v5−v4 HDV60 mean-world-median position-error difference: 1.9; CI NA (singleton)m.

### Controller-effect signs and credibility
- FULL-v4/120s:0.8calls n=17, observed win 52.9; CI NA (singleton)% vs forecast 92.6; CI NA (singleton)%; range coverage 45.0; CI NA (singleton)%; all10pair direction accuracy 56.2; CI NA (singleton)%.
- FULL-v4/300s:0.8calls n=18, observed win 55.6; CI NA (singleton)% vs forecast 92.4; CI NA (singleton)%; range coverage 65.0; CI NA (singleton)%; all10pair direction accuracy 60.0; CI NA (singleton)%.
- FULL-v4/600s:0.8calls n=20, observed win 70.0; CI NA (singleton)% vs forecast 93.1; CI NA (singleton)%; range coverage 57.5; CI NA (singleton)%; all10pair direction accuracy 65.0; CI NA (singleton)%.
- FULL-v4 ctrl0−native expected delay effect120/300/600: -32.5; CI NA (singleton) / -212.7; CI NA (singleton) / -49.5; CI NA (singleton)veh-s; recorded -92.4; CI NA (singleton) / -1,241.8; CI NA (singleton) / -2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 mappo−native expected delay effect120/300/600: -12.8; CI NA (singleton) / -316.7; CI NA (singleton) / -1,246.4; CI NA (singleton)veh-s; recorded 57.0; CI NA (singleton) / -661.8; CI NA (singleton) / -2,013.6; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 pi−native expected delay effect120/300/600: -64.3; CI NA (singleton) / -427.8; CI NA (singleton) / -1,674.2; CI NA (singleton)veh-s; recorded 10.9; CI NA (singleton) / -572.7; CI NA (singleton) / -1,130.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 circles−native expected delay effect120/300/600: 349.2; CI NA (singleton) / 1,417.7; CI NA (singleton) / 4,583.1; CI NA (singleton)veh-s; recorded 305.5; CI NA (singleton) / 717.0; CI NA (singleton) / 3,746.1; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 native−ctrl0 expected delay effect120/300/600: 32.5; CI NA (singleton) / 212.7; CI NA (singleton) / 49.5; CI NA (singleton)veh-s; recorded 92.4; CI NA (singleton) / 1,241.8; CI NA (singleton) / 2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 mappo−ctrl0 expected delay effect120/300/600: 19.8; CI NA (singleton) / -104.0; CI NA (singleton) / -1,196.9; CI NA (singleton)veh-s; recorded 149.5; CI NA (singleton) / 580.0; CI NA (singleton) / 539.0; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 pi−ctrl0 expected delay effect120/300/600: -31.7; CI NA (singleton) / -215.1; CI NA (singleton) / -1,624.7; CI NA (singleton)veh-s; recorded 103.3; CI NA (singleton) / 669.1; CI NA (singleton) / 1,421.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v4 circles−ctrl0 expected delay effect120/300/600: 381.8; CI NA (singleton) / 1,630.4; CI NA (singleton) / 4,632.6; CI NA (singleton)veh-s; recorded 397.9; CI NA (singleton) / 1,958.8; CI NA (singleton) / 6,298.7; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5/120s:0.8calls n=14, observed win 71.4; CI NA (singleton)% vs forecast 95.5; CI NA (singleton)%; range coverage 37.5; CI NA (singleton)%; all10pair direction accuracy 57.5; CI NA (singleton)%.
- FULL-v5/300s:0.8calls n=10, observed win 70.0; CI NA (singleton)% vs forecast 93.8; CI NA (singleton)%; range coverage 52.5; CI NA (singleton)%; all10pair direction accuracy 56.2; CI NA (singleton)%.
- FULL-v5/600s:0.8calls n=13, observed win 69.2; CI NA (singleton)% vs forecast 97.1; CI NA (singleton)%; range coverage 62.5; CI NA (singleton)%; all10pair direction accuracy 68.7; CI NA (singleton)%.
- FULL-v5 ctrl0−native expected delay effect120/300/600: 44.3; CI NA (singleton) / 350.8; CI NA (singleton) / 787.6; CI NA (singleton)veh-s; recorded -92.4; CI NA (singleton) / -1,241.8; CI NA (singleton) / -2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 mappo−native expected delay effect120/300/600: 29.6; CI NA (singleton) / 84.7; CI NA (singleton) / -252.2; CI NA (singleton)veh-s; recorded 57.0; CI NA (singleton) / -661.8; CI NA (singleton) / -2,013.6; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 pi−native expected delay effect120/300/600: -30.8; CI NA (singleton) / 67.9; CI NA (singleton) / 67.7; CI NA (singleton)veh-s; recorded 10.9; CI NA (singleton) / -572.7; CI NA (singleton) / -1,130.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 circles−native expected delay effect120/300/600: 354.4; CI NA (singleton) / 1,832.9; CI NA (singleton) / 6,398.6; CI NA (singleton)veh-s; recorded 305.5; CI NA (singleton) / 717.0; CI NA (singleton) / 3,746.1; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 native−ctrl0 expected delay effect120/300/600: -44.3; CI NA (singleton) / -350.8; CI NA (singleton) / -787.6; CI NA (singleton)veh-s; recorded 92.4; CI NA (singleton) / 1,241.8; CI NA (singleton) / 2,552.5; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 mappo−ctrl0 expected delay effect120/300/600: -14.7; CI NA (singleton) / -266.1; CI NA (singleton) / -1,039.8; CI NA (singleton)veh-s; recorded 149.5; CI NA (singleton) / 580.0; CI NA (singleton) / 539.0; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 pi−ctrl0 expected delay effect120/300/600: -75.1; CI NA (singleton) / -282.8; CI NA (singleton) / -719.9; CI NA (singleton)veh-s; recorded 103.3; CI NA (singleton) / 669.1; CI NA (singleton) / 1,421.8; CI NA (singleton)veh-s. Negative means lower delay.
- FULL-v5 circles−ctrl0 expected delay effect120/300/600: 310.1; CI NA (singleton) / 1,482.1; CI NA (singleton) / 5,611.0; CI NA (singleton)veh-s; recorded 397.9; CI NA (singleton) / 1,958.8; CI NA (singleton) / 6,298.7; CI NA (singleton)veh-s. Negative means lower delay.

### Short-horizon decision quality:30/60/90s
Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm's recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.
These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.
- FULL-v4/30s: team direction accuracy identification lower 5.0; CI NA (singleton)%, upper 100.0; CI NA (singleton)%; resolved-pair fraction 5.0; CI NA (singleton)%; resolved-only accuracy 100.0; CI NA (singleton)%; road-only direction accuracy 66.2; CI NA (singleton)%.
- FULL-v4/60s: team direction accuracy identification lower 12.5; CI NA (singleton)%, upper 97.5; CI NA (singleton)%; resolved-pair fraction 15.0; CI NA (singleton)%; resolved-only accuracy 83.3; CI NA (singleton)%; road-only direction accuracy 60.0; CI NA (singleton)%.
- FULL-v4/90s: team direction accuracy identification lower 13.8; CI NA (singleton)%, upper 96.2; CI NA (singleton)%; resolved-pair fraction 17.5; CI NA (singleton)%; resolved-only accuracy 78.6; CI NA (singleton)%; road-only direction accuracy 61.3; CI NA (singleton)%.
- FULL-v5/30s: team direction accuracy identification lower 5.0; CI NA (singleton)%, upper 100.0; CI NA (singleton)%; resolved-pair fraction 5.0; CI NA (singleton)%; resolved-only accuracy 100.0; CI NA (singleton)%; road-only direction accuracy 67.5; CI NA (singleton)%.
- FULL-v5/60s: team direction accuracy identification lower 12.5; CI NA (singleton)%, upper 97.5; CI NA (singleton)%; resolved-pair fraction 15.0; CI NA (singleton)%; resolved-only accuracy 83.3; CI NA (singleton)%; road-only direction accuracy 65.0; CI NA (singleton)%.
- FULL-v5/90s: team direction accuracy identification lower 16.2; CI NA (singleton)%, upper 98.8; CI NA (singleton)%; resolved-pair fraction 17.5; CI NA (singleton)%; resolved-only accuracy 92.9; CI NA (singleton)%; road-only direction accuracy 70.0; CI NA (singleton)%.

Strict paired P(beats incumbent)>=0.8 plus lower expected delay; ties are non-wins. K8 min–max range is not a90% interval (continuous exchangeability reference7/9). Zero-selected worlds remain in resampling. Full gate/switching: NOT ESTABLISHED at this interim.


### Interim CI numerical correction

The16:46snapshot is preserved. Its intervals printed as equal endpoints for
constant direction-accuracy columns are superseded by NA (zero estimable variance).
Centered variance reduction removes floating-point accumulation artifacts; all
point estimates and forecasts are unchanged. No model selection or forecast replay.

## Frozen-candidate engineering limits (later worker development evidence)

LANE's later free-running development diagnostic rejects fitted-LC adoption on
lane-frequency/recall evidence. BREAK reports that frozen116b2147 targets a retired
mode2 learned mirror, not the actual shared CAV actuator. Therefore the frozen v5
switch must not be described as verified mode2 actuator consistency. These later
findings do not change sourceb2df7458 or the running candidates; all outcomes remain
reported. LP remains a narrow learned departure-parameter variant. No integrated
performance or cooperation/planner authorization follows from engineering readiness.
<!-- RB2-INTERIM:END -->
