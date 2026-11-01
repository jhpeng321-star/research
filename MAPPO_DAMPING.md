# S-MAPPODAMP: is speed-only MAPPO (iter169) a damper or a cooperator? (analysis only, 2026-11-02)
**Verdict: PARTLY SUPPORTED.** "Rarely learns cooperative maneuvers" is **supported**. "Its delay gains come mainly from damping-like smoothing" is **not supported**. MAPPO's delay gains over native are not statistically resolved in any split. MAPPO is also *less* smooth than native SUMO by every acceleration measure. CIRCLES, the true damper, smooths CAVs and loses delay.

## Plain language
- MAPPO hardly ever opens a gap for a merging car, and it shows no measurable together-versus-alone synergy. Its behaviour is individual, not cooperative.
- But it does not damp. Its CAVs accelerate harder and jerk more than native SUMO CAVs, and their HDV followers accelerate slightly more. Upstream speed oscillation is not consistently reduced.
- In worlds where MAPPO saves time, vehicles keep speed more steadily and CAVs drive faster. These are near-mechanical restatements of lower delay, not evidence of a damping mechanism.

## Key numbers: MAPPO vs native, paired by world; dev32 / eval8 / rep50 reported separately; % change [95% CI]
- **Delay gain** (native−MAPPO team delay incl. source waiting, veh-s): J300 **+338 [−183, 911] / +662 [−76, 1,371] / +158 [−336, 666]**; J600 **+1,139 [−409, 2,842] / +2,014 [−302, 4,294] / +629 [−800, 2,072]**.
- **CAV acceleration RMS: +11.0% [8.5, 13.6] / +8.4% [4.8, 14.9] / +6.8% [4.8, 8.8]**; CAV jerk RMS **+21.8% [9.6, 36.0] / +26.1% [11.9, 45.8] / +18.0% [8.2, 28.5]**.
- HDVs directly behind a CAV, acceleration RMS: **+3.4% [0.8, 5.8] / +3.1% [−2.9, 9.9] / +2.3% [0.9, 3.8]**. All vehicles: **+2.7% [1.0, 4.3] / +3.5% [1.4, 7.4] / +2.0% [0.7, 3.2]**.
- Within-vehicle speed variance, all vehicles: −2.7% [−9.4, 4.2] / −7.7% [−17.0, −0.7] / −1.5% [−6.0, 3.4]. CAV mean speed: +1.1% [−0.8, 3.3] / +2.4% [−1.0, 5.9] / +1.0% [−0.7, 2.9].
- Upstream oscillation (RMS change between successive registered 10 s upstream mean-speed bins, 600 s): +4.1% [−2.6, 11.4] / **−14.7% [−24.7, −1.6]** / −1.9% [−7.9, 4.3]. Mean depth of the registered LRDIAG BIC step: −18.7% [−36.8, 3.3] / −24.6% [−48.5, 14.2] / −10.4% [−26.6, 9.1]. The evaluation-only reduction does not replicate.
- **CIRCLES reference:** CAV jerk −25.0% [−31.7, −17.5] / −18.2% [−23.0, −13.3] / −20.7% [−25.9, −15.0], and CAV speed −3% to −6%. Yet its J600 gain is **−1,735 [−3,457, −46] / −3,746 [−5,467, −2,142] / −4,547 [−5,961, −3,134]**: damping here costs delay.

## Cooperation indicators
- COOPAUDIT (separate TRAIN 969650–969677 panel, 21 eligible worlds, fixed CAV pair at t0=300 s): MAPPO C_G J300 **+117 [−198, +441]**, J600 **+68 [−1,179, +1,154]** veh-s. The share of worlds with C_G>0 is 0.33/0.38. Point estimates of the summed solo effects are 69%/92% of the group effect D_G.
- Unchanged COOPBEHAV detectors were run (code SHA-identical to the c4-coopbehav final). This task ran them on native and CIRCLES; MAPPO and CTRL0 counts come from the existing outputs. Rates are per CAV-hour; ratio = MAPPO/native.
  - **Yield-to-merge**: native 1.85/2.36/2.28 vs MAPPO **0.17/0.32/0.11**, ratio **0.09 [0.03, 0.18] / 0.14 [0.02, 0.29] / 0.05 [0.02, 0.08]**. CIRCLES ratio: 0.88/0.94/0.84.
  - Cross-lane holds: 0.65 [0.43, 0.95] / 0.53 [0.44, 1.11] / 0.67 [0.54, 0.84]. Follower-attenuation ("wave-damping") episodes: 0.75 [0.66, 0.86] / 0.68 [0.49, 0.86] / 0.67 [0.59, 0.75].
  - These are observational pattern counts, not causal credit. Rows in coopbehav_rates.csv.

## Attribution (descriptive within-split regression of each world's gain on its paired indicator reduction; R² [95% CI])
- J300 on the acceleration-RMS reduction for all vehicles: **0.27 [0.05, 0.63] / 0.01 [0.00, 0.84] / 0.16 [0.01, 0.42]**. CAVs: 0.05 [0.00, 0.26] / 0.01 [0.00, 0.80] / 0.10 [0.00, 0.30]. HDV followers: 0.04 / 0.70 [0.12, 0.95] / 0.16 [0.02, 0.39].
- At J600, every acceleration-based R² is ≤0.16. A joint fit on the four acceleration/roughness indicators gives J300 R² 0.46 [0.27, 0.83] in dev and 0.29 [0.09, 0.62] in rep; with n=8, eval cannot support a joint fit.
- Within-vehicle speed variance gives J300 R² 0.77 [0.65, 0.92] / 0.76 [0.44, 0.99] / 0.83 [0.63, 0.94]; CAV mean speed gives 0.70 / 0.78 / 0.88. Delay is the integral of the speed deficit, so these outcome correlates are not a smoothing mechanism.
- On average MAPPO is less smooth, so the fitted gain at zero acceleration change exceeds the mean gain (dev intercept share 2.4, CI spans 0). Its gain is not explained by smoothing.

## Suggested paper sentence
"Speed-only MAPPO learns little cooperation: its measured together-versus-alone synergy is indistinguishable from zero (C_G = +68 veh-s at 600 s, 95% CI [−1,179, 1,154]), and it yields to merging vehicles at only 5–14% of the native-SUMO rate. Nor is it a wave damper: relative to native driving it raises CAV acceleration RMS by 7–11% and jerk by 18–26%, and its delay savings over native are not statistically resolved, whereas the damping controller CIRCLES lowers CAV jerk by about 20% but increases delay."

## Contract, caveats and sources (SHA256)
- Zero SUMO/training/GPU. The remote reducer was read-only at nice 19 and read only time_s/veh_kin/veh_mask/vid/is_cav/episode.json of the recorded views. It covered 450 episodes (generic_random excluded) over the 300–600 s branch window at 0.5 s.
- Realised acceleration is the identity-continuous Δv/Δt and matches the recorded column within 7.6e-6 m/s². No behavioural thresholds were used. The HDV-follower group is defined geometrically: the same-lane immediate leader is a CAV.
- All three splits were previously used (WMCALIB/WMBREAKDOWN/BDRANK/COOPBEHAV). Eval8 has only 8 worlds. CIs use 10,000 whole-world bootstrap draws within split, seed 20261102.
- The MAPPO policy is d9042cc5 in every MAPPO episode, matching COOPAUDIT iter169. Gains reproduce WMBREAKDOWN (dev J600 MAPPO/PI/CIRCLES +1,139/+2,152/−1,735).
- Inputs:
  - `E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown/wmdata_observables.json` 01d0d2b7027aad9910496c7bc5b9681c0d1ccf08f46b4e16a1113310a67362e9
  - `.../wmdata2_observables.json` a2b3ecf7a6ea9d6904fafa44d57a502e77781024b42b9b9469cbddaf44779fb4
  - `E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md` 4988011101f185fb175179d28564541a4ed4f2d94e438f73bfa7401e5c6b274b
  - `E:/agent-work/c4-coopbehav/tools/coopbehav/evidence/WMDATA_final/episodes.json` 9d7d0136c06103f703c4352e4f43cf33a76b261883a7bc797d3735b2deb234a5
  - `.../WMDATA2_final/episodes.json` 9f81627b3ce159c207d4bf3ca93a60106389facd797e69cdb8ce147f9d31de48
  - Detector code: detectors.py 9c5cdecd3bbff09de9e65e5bb85a5e1915deaeaee62567593f90b02d973906ff
  - Remote views: cpuQ `/home/ubuntu/wmdata20261031/views`, gpuR `/root/wmdata2/views`; per-array SHA256 values are in the evidence JSONs.
- Outputs: `evidence/` holds the vehicle metrics, paired_indicators.csv, attribution.csv, coopbehav_rates.csv, the native/CIRCLES detector outputs, and manifest.json (hashes of all inputs, code and outputs). To reproduce, run `python tools/mappo_damping/collect.py WMDATA|WMDATA2`, then `python tools/mappo_damping/analyze.py`.
