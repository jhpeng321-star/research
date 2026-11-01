# WM-PHYS v5 — INTERIM; final comparison pending

**INTERIM 20261101T183839+0800: on 8 matched complete K8 worlds
(reserved8), v5−v4 delay CRPS is +169/+1,110/+2,996 veh-s at120/300/600s.**
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
## INTERIM v5 versus v4 — 20261101T182025+0800 (available_replicated_cells)

**8 matched complete K8 worlds; all five arms.** This is an interim diagnostic, not model selection or switching authorization.
Worlds: 969433, 969436, 969437, 969438, 969590, 969592, 969595, 969599.
CIs use10,000 paired fixed-cell whole-world resamples; delay/HDV/accuracy condition on stored K8. Rate/effect means include paired future resampling. Each metric family is simultaneous across its displayed horizons/models; this interim two-model family is separate from the final registered six-contrast family. No sequential/repeated-look confidence claim.
Strata: [{'demand': 1900, 'penetration': 0.05, 'worlds': 2}, {'demand': 1900, 'penetration': 0.4, 'worlds': 2}, {'demand': 2700, 'penetration': 0.05, 'worlds': 2}, {'demand': 2700, 'penetration': 0.1, 'worlds': 2}].

### Delay, breakdown and HDV60
- FULL-v4 delay CRPS120/300/600=1,047/3,570/11,039veh-s; mean bias=1,382/4,683/14,811veh-s.
- FULL-v5 delay CRPS120/300/600=1,287/4,920/15,500veh-s; mean bias=1,676/6,359/20,621veh-s.
- 120s v5−v4 delay CRPS: 240.0 [95% CI 118.5, 361.4]veh-s; breakdown absolute rate-distance: 12.50 [95% CI -22.23, 47.23]pp.
- 300s v5−v4 delay CRPS: 1,350.0 [95% CI 533.0, 2,167.0]veh-s; breakdown absolute rate-distance: 2.50 [95% CI -22.54, 21.42]pp.
- 600s v5−v4 delay CRPS: 4,460.5 [95% CI 2,613.3, 6,307.7]veh-s; breakdown absolute rate-distance: 0.00 [95% CI -18.29, 17.60]pp.
- v5−v4 HDV60 mean-world-median position-error difference: 0.5 [95% CI -4.5, 5.5]m.

### Controller-effect signs and credibility
- FULL-v4/120s:0.8calls n=18, observed win 44.4 [95% CI 25.5, 63.4]% vs forecast 92.4 [95% CI 88.4, 96.3]%; range coverage 37.5 [95% CI 0.0, 75.4]%; all10pair direction accuracy 48.8 [95% CI 40.0, 57.5]%.
- FULL-v4/300s:0.8calls n=14, observed win 57.1 [95% CI 32.0, 82.3]% vs forecast 90.2 [95% CI 88.6, 91.7]%; range coverage 57.5 [95% CI 19.4, 95.6]%; all10pair direction accuracy 65.0 [95% CI 55.4, 74.6]%.
- FULL-v4/600s:0.8calls n=12, observed win 58.3 [95% CI 41.7, 75.0]% vs forecast 93.8 [95% CI 88.8, 98.7]%; range coverage 72.5 [95% CI 39.4, 100.0]%; all10pair direction accuracy 43.8 [95% CI 33.1, 54.4]%.
- FULL-v4 ctrl0−native expected delay effect120/300/600: -41.3 [95% CI -107.6, 25.1] / -181.6 [95% CI -642.1, 279.0] / -184.5 [95% CI -1,930.4, 1,561.4]veh-s; recorded -40.4 [95% CI -146.1, 65.2] / 38.3 [95% CI -416.8, 493.4] / 1,093.2 [95% CI -1,271.0, 3,457.5]veh-s. Negative means lower delay.
- FULL-v4 mappo−native expected delay effect120/300/600: -2.1 [95% CI -125.5, 121.3] / -332.3 [95% CI -1,029.9, 365.2] / -871.7 [95% CI -3,063.9, 1,320.6]veh-s; recorded 60.8 [95% CI -32.7, 154.3] / -15.7 [95% CI -952.1, 920.7] / -684.5 [95% CI -2,821.8, 1,452.9]veh-s. Negative means lower delay.
- FULL-v4 pi−native expected delay effect120/300/600: -45.4 [95% CI -184.5, 93.7] / -254.7 [95% CI -981.3, 471.8] / -843.1 [95% CI -3,080.3, 1,394.2]veh-s; recorded 39.3 [95% CI -80.0, 158.7] / 206.5 [95% CI -659.7, 1,072.6] / 1,194.5 [95% CI -2,477.6, 4,866.6]veh-s. Negative means lower delay.
- FULL-v4 circles−native expected delay effect120/300/600: 82.7 [95% CI -21.5, 186.9] / 417.6 [95% CI -220.0, 1,055.3] / 1,426.9 [95% CI -386.3, 3,240.0]veh-s; recorded 147.3 [95% CI -128.7, 423.3] / 1,107.5 [95% CI 197.9, 2,017.2] / 3,608.0 [95% CI -86.6, 7,302.5]veh-s. Negative means lower delay.
- FULL-v4 native−ctrl0 expected delay effect120/300/600: 41.3 [95% CI -25.1, 107.6] / 181.6 [95% CI -279.0, 642.1] / 184.5 [95% CI -1,561.4, 1,930.4]veh-s; recorded 40.4 [95% CI -65.2, 146.1] / -38.3 [95% CI -493.4, 416.8] / -1,093.2 [95% CI -3,457.5, 1,271.0]veh-s. Negative means lower delay.
- FULL-v4 mappo−ctrl0 expected delay effect120/300/600: 39.2 [95% CI -77.8, 156.2] / -150.8 [95% CI -706.7, 405.2] / -687.1 [95% CI -2,410.0, 1,035.8]veh-s; recorded 101.2 [95% CI -37.4, 239.9] / -54.0 [95% CI -741.0, 633.1] / -1,777.7 [95% CI -3,685.2, 129.8]veh-s. Negative means lower delay.
- FULL-v4 pi−ctrl0 expected delay effect120/300/600: -4.1 [95% CI -155.5, 147.2] / -73.2 [95% CI -925.1, 778.8] / -658.5 [95% CI -3,242.0, 1,925.0]veh-s; recorded 79.7 [95% CI -85.6, 245.1] / 168.2 [95% CI -619.1, 955.5] / 101.3 [95% CI -3,225.3, 3,427.8]veh-s. Negative means lower delay.
- FULL-v4 circles−ctrl0 expected delay effect120/300/600: 124.0 [95% CI 11.1, 236.8] / 599.2 [95% CI -147.6, 1,346.0] / 1,611.4 [95% CI -183.7, 3,406.5]veh-s; recorded 187.7 [95% CI -90.7, 466.2] / 1,069.3 [95% CI 65.7, 2,072.8] / 2,514.7 [95% CI -662.3, 5,691.8]veh-s. Negative means lower delay.
- FULL-v5/120s:0.8calls n=8, observed win 62.5 [95% CI 19.6, 100.0]% vs forecast 90.6 [95% CI 86.2, 95.0]%; range coverage 25.0 [95% CI 0.0, 58.0]%; all10pair direction accuracy 51.2 [95% CI 37.3, 65.2]%.
- FULL-v5/300s:0.8calls n=10, observed win 70.0 [95% CI 51.3, 88.7]% vs forecast 92.5 [95% CI 89.3, 95.7]%; range coverage 50.0 [95% CI 7.6, 92.4]%; all10pair direction accuracy 65.0 [95% CI 59.6, 70.4]%.
- FULL-v5/600s:0.8calls n=9, observed win 77.8 [95% CI 42.5, 100.0]% vs forecast 93.1 [95% CI 89.5, 96.6]%; range coverage 67.5 [95% CI 32.8, 100.0]%; all10pair direction accuracy 60.0 [95% CI 54.0, 66.0]%.
- FULL-v5 ctrl0−native expected delay effect120/300/600: 7.9 [95% CI -74.1, 89.9] / 181.2 [95% CI -440.5, 802.8] / 337.1 [95% CI -1,376.3, 2,050.5]veh-s; recorded -40.4 [95% CI -146.1, 65.2] / 38.3 [95% CI -416.8, 493.4] / 1,093.2 [95% CI -1,271.0, 3,457.5]veh-s. Negative means lower delay.
- FULL-v5 mappo−native expected delay effect120/300/600: 27.9 [95% CI -51.5, 107.3] / 105.4 [95% CI -340.0, 550.8] / -179.7 [95% CI -1,611.2, 1,251.8]veh-s; recorded 60.8 [95% CI -32.7, 154.3] / -15.7 [95% CI -952.1, 920.7] / -684.5 [95% CI -2,821.8, 1,452.9]veh-s. Negative means lower delay.
- FULL-v5 pi−native expected delay effect120/300/600: -1.0 [95% CI -91.8, 89.8] / 68.8 [95% CI -495.2, 632.9] / 131.5 [95% CI -1,722.9, 1,986.0]veh-s; recorded 39.3 [95% CI -80.0, 158.7] / 206.5 [95% CI -659.7, 1,072.6] / 1,194.5 [95% CI -2,477.6, 4,866.6]veh-s. Negative means lower delay.
- FULL-v5 circles−native expected delay effect120/300/600: 107.9 [95% CI -9.4, 225.1] / 677.0 [95% CI -27.4, 1,381.4] / 2,380.2 [95% CI 358.4, 4,402.0]veh-s; recorded 147.3 [95% CI -128.7, 423.3] / 1,107.5 [95% CI 197.9, 2,017.2] / 3,608.0 [95% CI -86.6, 7,302.5]veh-s. Negative means lower delay.
- FULL-v5 native−ctrl0 expected delay effect120/300/600: -7.9 [95% CI -89.9, 74.1] / -181.2 [95% CI -802.8, 440.5] / -337.1 [95% CI -2,050.5, 1,376.3]veh-s; recorded 40.4 [95% CI -65.2, 146.1] / -38.3 [95% CI -493.4, 416.8] / -1,093.2 [95% CI -3,457.5, 1,271.0]veh-s. Negative means lower delay.
- FULL-v5 mappo−ctrl0 expected delay effect120/300/600: 20.0 [95% CI -49.0, 89.0] / -75.7 [95% CI -707.8, 556.4] / -516.8 [95% CI -2,187.9, 1,154.3]veh-s; recorded 101.2 [95% CI -37.4, 239.9] / -54.0 [95% CI -741.0, 633.1] / -1,777.7 [95% CI -3,685.2, 129.8]veh-s. Negative means lower delay.
- FULL-v5 pi−ctrl0 expected delay effect120/300/600: -8.8 [95% CI -96.7, 79.1] / -112.3 [95% CI -648.8, 424.1] / -205.6 [95% CI -2,024.2, 1,613.0]veh-s; recorded 79.7 [95% CI -85.6, 245.1] / 168.2 [95% CI -619.1, 955.5] / 101.3 [95% CI -3,225.3, 3,427.8]veh-s. Negative means lower delay.
- FULL-v5 circles−ctrl0 expected delay effect120/300/600: 100.0 [95% CI -7.5, 207.5] / 495.8 [95% CI -209.4, 1,201.0] / 2,043.1 [95% CI 227.5, 3,858.7]veh-s; recorded 187.7 [95% CI -90.7, 466.2] / 1,069.3 [95% CI 65.7, 2,072.8] / 2,514.7 [95% CI -662.3, 5,691.8]veh-s. Negative means lower delay.

### Short-horizon decision quality:30/60/90s
Exact total-delay accuracy is **partially identified**: the recorded data saved total/source clocks only at120/300/600s. Reconstructed road clocks match recorded120s road delay; unknown short source waiting is bounded between0 and each arm's recorded120s source waiting. The intervals below bound all-pair team-effect accuracy without inventing a source trajectory. Road-only accuracy is a separate diagnostic.
These are five recorded controller options, not unrecorded cooperation membership/group interventions. Cooperation/planner short-horizon qualification remains NOT ESTABLISHED.
- FULL-v4/30s: team direction accuracy identification lower 5.0 [95% CI 0.0, 13.6]%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 5.0 [95% CI 0.0, 13.6]%; resolved-only accuracy 100.0; CI NA (insufficient support/variance)%; road-only direction accuracy 60.0 [95% CI 52.5, 67.5]%.
- FULL-v4/60s: team direction accuracy identification lower 12.5 [95% CI 8.2, 16.8]%, upper 97.5 [95% CI 93.2, 100.0]%; resolved-pair fraction 15.0; CI NA (insufficient support/variance)%; resolved-only accuracy 83.3 [95% CI 54.6, 100.0]%; road-only direction accuracy 55.0 [95% CI 47.5, 62.5]%.
- FULL-v4/90s: team direction accuracy identification lower 13.8 [95% CI 7.3, 20.2]%, upper 96.2 [95% CI 89.8, 100.0]%; resolved-pair fraction 17.5; CI NA (insufficient support/variance)%; resolved-only accuracy 78.6 [95% CI 41.6, 100.0]%; road-only direction accuracy 50.0 [95% CI 39.9, 60.1]%.
- FULL-v5/30s: team direction accuracy identification lower 5.0 [95% CI 0.0, 13.6]%, upper 100.0; CI NA (insufficient support/variance)%; resolved-pair fraction 5.0 [95% CI 0.0, 13.6]%; resolved-only accuracy 100.0; CI NA (insufficient support/variance)%; road-only direction accuracy 61.3 [95% CI 57.5, 65.0]%.
- FULL-v5/60s: team direction accuracy identification lower 12.5 [95% CI 8.2, 16.8]%, upper 97.5 [95% CI 93.2, 100.0]%; resolved-pair fraction 15.0; CI NA (insufficient support/variance)%; resolved-only accuracy 83.3 [95% CI 54.6, 100.0]%; road-only direction accuracy 53.8 [95% CI 42.2, 65.3]%.
- FULL-v5/90s: team direction accuracy identification lower 16.2 [95% CI 14.1, 18.4]%, upper 98.8 [95% CI 96.6, 100.0]%; resolved-pair fraction 17.5; CI NA (insufficient support/variance)%; resolved-only accuracy 92.9 [95% CI 80.5, 100.0]%; road-only direction accuracy 56.2 [95% CI 42.5, 70.0]%.

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
