# Claim-evidence report: claim validation 20261113_0231 (Amendment 72) and post hoc amendments 69-75

Assembled 2026-11-13 from existing records only. No simulation, no SUMO, no new statistics. Every number below is
copied from the cited JSON file or REGISTRATION.md line (`REG` = `runs/active/C4-V7FT-20261017/REGISTRATION.md`,
current sha256 in MANIFESTS/SHA256SUMS.txt; the file is append-only, so
later appends change this hash, not the cited lines).

Setting: SUMO map wz3to2, demand 2,800 veh/h, 60% CAVs (cell D2800_p60), 1,200 s episodes, unless stated.
Convention: diff = treatment - reference; negative total delay = improvement; units veh-s per world (episode).
Intervals: paired percentile bootstrap, 10,000 resamples, seed 20261104, 95%; p = 0.0002 is the bootstrap floor.
Panels are never pooled. "Not separated" = the 95% CI includes 0.

Section 1 = the REGISTERED study (A72). Section 2 = POST HOC / exploratory items, kept separate.

---

## 1. REGISTERED (Amendment 72, registered 2026-11-13 before any outcome of this round; `REG:11478-11534`)

Reference U0 = frozen speed-only base MAPPO d9042cc5... Confirmation panel: 96 worlds 990900-990995, shared by
E1 and E2 (U0 mean 42,288.3 veh-s). Multiplicity: Holm over the two primaries, two-sided, alpha 0.05.

### E1. World-model (WM) screening order vs Random order of the same fixed pool

**Claim tested:** ranking a fixed pool of 120 hold-only plans with the WM yields more admissible plans among the first six
than a seeded random order of the same pool (6 search seeds 11, 23, 42, 67, 89, 103).

**Verdict: PRIMARY NOT SUPPORTED.** Delay secondary NOT SUPPORTED.

| Quantity | Value | Source |
|---|---|---|
| Admitted of first six, WM, seeds 11/23/42/67/89/103 | 5 / 4 / 0 / 4 / 6 / 1 (total 20) | FROZEN_BANKS.json `primary_admitted_count`; `REG:11605` |
| Admitted of first six, Random | 4 / 5 / 1 / 5 / 3 / 2 (total 20) | same |
| WM - Random per seed | +1 / -1 / -1 / -1 / +3 / -1; mean 0.0 | same |
| PRIMARY exact sign-flip permutation (64 assignments), two-sided | p = 1.0 (p_less 0.578125; minimum attainable p 0.03125) | same |
| Secondary: WM bank - Random bank total delay, seed x world crossed bootstrap | +1,036.6 [-1,865.5, +4,525.4]; sign-flip p two-sided 0.656 (0.65625); not separated | E1_ANALYSIS.json `by_budget.6`; `REG:11620` |
| Same, per seed 11/23/42/67/89/103 (point only; no per-seed CI recorded) | -354.9, +2,309.5, +8,390.5, -1,820.2, -3,908.2, +1,602.7 | same |
| Secondary: WM bank - U0 (%) | s11 -24.1, s23 -17.6, s42 0 (bank is U0), s67 -28.9, s89 -26.0, s103 -12.8 | E1_ANALYSIS.json `bank_minus_u0`; `REG:11621` |
| Secondary: Random bank - U0 (%) | s11 -23.3, s23 -23.1, s42 -19.8, s67 -24.6, s89 -16.8, s103 -16.6 (all CIs below 0) | same |

All bank - U0 CIs (veh-s and %) are in `E1_SUMMARY.csv`. Budgets 2 and 4 are descriptive only (also in the CSV).

Evidence:
- `e1_conf/E1_ANALYSIS.json` a82616dd22ca692bcd931fcfc7ead68b272620d1ba4194def4055c69d29c7783 (doc 37d5b100...48dc)
- `e1_conf/CONFIRM_RESULTS.json` b4b0bfb90672dc7439316caf89b126a701758124e0e06cf710f97f1ff7dd4e80
- `FROZEN_BANKS.json` f27f2e5a4fbdf0f59b02352719e21a7cead58dd8746278e592d8450c2c944299 (doc c9b271b5...31c7), extracted
  unchanged from `e1_bank_box4.tgz` a69cc94b0796625828bf9e1255116e70dce56fe02efffdecd1b20a89b40365b1
- `e1_verify/VERIFY_RESULTS_seed*.json` (shas in MANIFESTS; equal to the prefixes at `REG:11604`)

Caveats / disclosures as recorded:
- Ranking artefact, disclosed 06:27 before any E1 SUMO outcome (`REG:11598`): under the registered two-stage rule a plan the
  WM never activated scores 0.0 and outranks activated plans with negative WM score. WM top-six plans never activated on a
  final state: s11 5/6, s23 6/6, s42 5/6, s67 3/6, s89 0/6, s103 6/6. "In those seeds the WM arm is close to uninformed."
  The registered primary was run and reported unchanged.
- `REG:11605`: seed 89 (the only seed whose WM top six were all activated) is +3; "consistent with the ranking artefact
  disclosed 06:27, not evidence."
- Protocol risk stated in advance: no bank-level activation cap, so holds of a multi-plan bank stack (A68).
- Holm: minimum attainable E1 p (0.03125) means E1 could pass only if E2 p <= 0.025 (stated before results).
- Flag (file labelling): `E1_ANALYSIS.json` sets `by_budget.6.primary = true` while its `role` field reads "secondary
  (delay); the registered E1 primary is FROZEN_BANKS.primary_admitted_count". The registered primary is the admitted count.
- Flag (build notes vs implementation): `BUILD_NOTES.md` section 2 item 5 says "never-activated descriptors rank last";
  the registered scorer as run scored them 0.0 (the 06:27 artefact). Recorded here, not resolved.

### E2. Original four-plan bank F vs tuned uniform hold H* vs synchronised copy S

**Claim tested:** F has lower total delay than a resource-matched uniform hold H* chosen on Development worlds.

Tuning (24 Development worlds 998200-998223, already-viewed panel; `REG:11559`): mean total delay (minus U0) H18 33,866.4
(-7,405.3); H19 31,469.0 (-9,802.7); H20 30,419.2 (-10,852.5); H21 31,939.2 (-9,332.4); H22 30,451.9 (-10,819.8).
H* = H20 by the registered rule (lowest mean; no tie).

**Verdict: PRIMARY NOT SUPPORTED** (F nominally more delay than H20; not separated).

| Contrast (96 confirmation worlds) | Role | Diff [95% CI] | % [95% CI] | p | worlds lower / higher |
|---|---|---|---|---|---|
| F - H20 | PRIMARY | +686.9 [-205.5, +1,656.5] | +2.2 [-0.7, +5.3] | 0.142 | 48 / 48 |
| F - S | secondary | +58.2 [-886.4, +1,063.7] | +0.2 [-2.7, +3.3] | 0.903 | 50 / 46 |
| F - U0 | secondary | -10,215.1 [-12,374.6, -8,108.9] | -24.2 [-27.7, -20.3] | 0.0002 | 82 / 14 |
| H20 - U0 | secondary | -10,902.0 [-13,044.2, -8,805.5] | -25.8 [-29.2, -22.1] | 0.0002 | 77 / 19 |
| S - U0 | secondary | -10,273.4 [-12,588.1, -8,057.8] | -24.3 [-28.1, -20.3] | 0.0002 | 76 / 20 |

Arm means: U0 42,288.3; F 32,073.2; H20 31,386.3; S 32,015.0. Executed control effort per world (secondary,
descriptive): activations F 15.98, H20 16.04, S 16.01; hold vehicle-seconds F 5,429.0, H20 6,015.7, S 6,067.2;
truncated groups 0 in all arms. As recorded in `e2_secondary/E2_SECONDARY_REPORT.md`: activations matched within 0.4%;
hold vehicle-seconds are not matched (F about 10% fewer than H20 and S).

Holm over the two registered primaries (`e2_secondary/box4/E2_HOLM.json`): raw p E2 0.141986 -> adjusted 0.283972;
E1 1.0 -> 1.0; neither rejected. Computed 13:13 by the e2_secondary agent; recorded at `REG:11671` (closes the 04:51 "Holm step
pending the E1 p-value").

Same-state timing replay (registered secondary, <= 24 Development worlds): **NOT RUN** (0 sim-s). Prepared; zero-SUMO
dry run only (E2_SECONDARY_REPORT.md).

Evidence:
- `box4_out/e2/E2_ANALYSIS.json` f1dab0f6f72827ace8d6585d24d6b8ad5b4e6d0a6185eecdad54524eb56a4ac0; reproduced
  byte-identically on Box 4 (`e2_secondary/box4/E2_ANALYSIS_REPRO.json`, same sha)
- `box4_out/e2/E2C_RESULTS.json` f698c87fbf86882deb8f0117de3a8ba5a36e95c333c2dab1c24cb160840f372a
- `box4_out/e2/E2T_RESULTS.json` 9b8c47a42dd03fe51ffecba2b188bb6a009bb5e2cc3440518bad233e3cf5b477
- `box4_out/e2/E2_SELECTION.json` 426837d455d87461b55c93664bd01a998beaf41805371de703ecaf0cd093ccb1
- `e2_secondary/box4/E2_HOLM.json` 19a32e8b937e4cd1b252384a1fff2ce99d8998fee89f6f7d229b133daeeebec0
- `e2_secondary/E2_SECONDARY_REPORT.md` b82de449dcd599d940c7b872ee1a8cc0bbf304dc9f308ffff69178440656cef7
- Identity check before E2 SUMO: PASS, max_abs_diff 0.0 (`box4_out/ident/IDENT_RAW.json` 97061afb...; `REG:11548`)

Caveats as recorded: tuning used an already-viewed Development panel (by design); S and H keep F's cooldowns, so S/H plans
can finish up to 20 s earlier within the same cooldown (BUILD_NOTES section 4).

---

## 2. POST HOC / EXPLORATORY (not part of the registered A72 family; never pooled with it or with each other)

Note the different references: A69/A70/A71 use MAPPO60C (the 84-update 60% MAPPO) on their own panels; A72/A73 use U0.

### A69 (S-LCSIDE60C) lane-change plans at 60% - exploratory side track of A68

Verdict: lane-change bank **not separated from MAPPO60C**, and does not beat the hand-written hold.
- Admission (24 worlds 992500-992523, vs MAPPO60C): 3 of 6 admitted; CIs of plan - MAPPO60C: b3c709d3 [-9,157.7, -1,254.4];
  d101ef57 [-8,364.2, -912.1]; 297db1f2 [-6,179.9, -269.2]; not admitted 0248e5ef [-7,119.1, +318.6], 6106ac9d
  [-6,037.3, +1,485.3], 73961dad [-5,967.8, +1,437.5] (`REG:11567`).
- Confirmation (96 worlds 996200-996295, paired with stored A68 arms; `REG:11570-11572`): LC_BANK_60C - MAPPO60C -1,221.7
  [-2,754.8, +284.4] (not separated); LC_BANK_60C - DUO_HOLD_60C +619.8 [-589.2, +1,784.4] (not separated);
  LC_BANK_60C - MAIN_BANK_60C -1,408.3 [-2,322.3, -445.8]. Lane changes 2,205 requested, 1,219 completed, 968 gave up;
  0 collision difference. "No follow-up at 60%."
- Evidence: `a69_lcside_confirm/A69_RESULTS_D2800_p60.json` d9001f6a8bdd1ce47acb14069cfbc4700cbbfccc2b7125b7a718005b0679d54c;
  `A69_ADMISSION_D2800_p60.json` ee81e530dc5642bbe4bcde68ce407a043491c7f8a8bdd1c11493f22978d04719.
- Caveats: SUMO stages moved from Box 3 to Box 4 (cross-host identity H1 PASS, `REG:11460`); empty-plan half of the A69
  identity was not run (recorded as not run, `REG:11427`).

### A70 (S-SMALLBANK60C) small banks - POST HOC, motivated by the failed A68 confirmation

Verdict: **primary SUPPORTED (post hoc)** (`REG:11553-11556`), 96 fresh worlds 996300-996395:
- SMALL1_60C - MAPPO60C -3,186.1 [-4,650.7, -1,780.9], p 0.0002; group rule passes (cav -3.2 s, hdv -5.0, origin L0 -11.9,
  L1 -6.5, L2 +6.8).
- SMALL2_60C - MAPPO60C -3,354.4 [-5,068.4, -1,773.0] (L2 +8.3 s); DUO_HOLD_60C - MAPPO60C -2,898.2 [-4,434.4, -1,477.2].
- Versus the hand-written hold: SMALL1 - DUO_HOLD -287.8 [-1,397.6, +817.2]; SMALL2 - DUO_HOLD -456.1 [-1,608.3, +645.6];
  SMALL2 - SMALL1 -168.3 [-1,091.4, +756.1]. Neither small bank is separated from DUO_HOLD on total delay.
- Most-delayed vehicles (mean delay of the slowest tenth, s): SMALL1 - MAPPO60C -42.3 [-50.6, -34.4]; SMALL2 - MAPPO60C
  -47.7 [-56.6, -39.4]; DUO - MAPPO60C -33.3 [-41.8, -25.7]; SMALL1 - DUO -8.9 [-14.2, -3.7]; SMALL2 - DUO -14.3 [-19.3, -9.5].
- Hard brakes per world MAPPO60C 10.16 / DUO 8.99 / SMALL1 7.84 / SMALL2 8.70 (SMALL1 - MAPPO60C -2.31 [-3.91, -0.72]).
  Collisions 0, teleports 0.
- Limits as recorded (`REG:11557`): post hoc; plan ranking used selection-biased admission bounds; one panel, D2800 p60 only.
- Evidence: `a70_smallbank/root/A70_RESULTS_D2800_p60.json` 18c47f74ed6ca5763e34ef04e4367b59a0c4dec36b413039a3575e242c0d781b.

### A71 (S-WIDEADMIT60C) wider admission - POST HOC

Verdict: **primary SUPPORTED (post hoc)** (`REG:11585-11589`):
- Admission screen (63 uncorrected tests on 24 worlds): 3-CAV 14/37, 4-CAV 3/12, 5-CAV 0/8, 6-CAV 0/6; total 17/63 (2 of the
  14 are per-world duplicates of cf289eb6).
- WIDE_BANK (mut_fce1add0 3-CAV + mut_99eff680 4-CAV) - MAPPO60C -2,970.4 [-4,565.5, -1,410.0], p 0.0006; group rule passes
  (L2 +9.0 s, close to the 10 s limit).
- WIDE_BANK - DUO_HOLD_60C -72.2 [-1,204.6, +1,068.3]: not separated. Most-delayed vehicles (slowest tenth): -48.7 s
  [-57.3, -40.6] vs MAPPO60C; -15.4 s [-20.5, -10.3] vs DUO_HOLD. Collisions 0, teleports 0.
- Limits as recorded: post hoc; the A70 panel had already been viewed for SMALL1/SMALL2; admission is an uncorrected screen.
- Descriptive follow-up, zero simulation, not registered (`REG:11662`): search-produced plans admitted 17/53, random plans
  0/10; one-sided hypergeometric P = 0.032 (sizes 3-4 only: 0.042). "Descriptive, not a claim" (10 random plans only, size
  mix differs, multiplicity).
- Evidence: `a71_wideadmit/root/A71_RESULTS_D2800_p60.json` 2ba84e7487b53d6742d07d9668db51fa59a33e212d95aef08fe1817f227e544c;
  `A71_ADMISSION_D2800_p60.json` 230911f434f1febe1ad08944930ef5a923e3057b86b9abdc8368aae24d148a22.

### A73 (S-LCSIDE20) lane-change plans at 20% CAVs (D2800_p20, reference U0)

Verdict: **0 of 6 admitted; confirmation not run** (nothing to confirm). Reported as a limitation at 20% (owner instruction).
- Search found only two-CAV lane-change candidates for selection (6 plans, `REG:11635`).
- Mechanics PASS (24 of 34 commanded lane changes completed, 0 collisions; mechanics only, not efficacy; `REG:11655`).
- Admission, plan - U0 on 24 worlds (`REG:11666`): 6afbc041 +1,059.5 [-2,767.1, +4,844.9]; 3e350d1e -71.0 [-3,989.3,
  +3,410.9]; 0a904151 -1,720.1 [-5,603.5, +2,183.4]; cbac5555 +664.7 [-2,940.8, +4,261.7]; 6dd3eac3 -1,842.6 [-5,844.0,
  +2,255.9]; a1a110da -1,874.2 [-5,263.9, +1,478.9]. Group rule passed for all six; delay test failed for all six.
- Caveat as recorded: search harm calibration fitted at p60, applied at p20.
- Evidence: `a73_lcside20_out/root/A73_ADMISSION_D2800_p20.json` be5559f4f4ee2d0d17476c6080c8da69370ec488859296376d43e0f83960428d.

### WM screening weaknesses and score non-reproducibility (A72 diagnostics; zero SUMO)

- Zero-score ranking artefact: see E1 caveats (`REG:11598`).
- Batching probe (`REG:11614`): one registered final-scored plan re-scored per seed was NOT identical on all five probed
  seeds (registered vs rescored: s11 -52.31/-171.51, s23 -128.51/-380.56, s42 -54.88/-398.13, s67 -218.25/-202.71,
  s103 -82.42/-186.17). Cause not determined (batch-composition dependence suspected, unproven).
- Determinism check (`REG:11626`): s89, same 12 plans, order and states as the registered final call: 0 of 12 summaries
  identical; max |score diff| 196.05 (e.g. 229.47 -> 33.42, -55.31 -> +95.80). "The WM final score is NOT reproducible run
  to run on the same call." The registered primary WM order is "one noisy WM draw per plan".
- Evidence: `wm_scores/`, `wm_act_scores_reuse/`, `wm_act_scores_all/` (shas in MANIFESTS; `.json` shas equal the prefixes
  at `REG:11596`, `REG:11614`, `REG:11625`).

### WM-ACT sensitivity arm (post-registration, pre-outcome for its candidates; never replaces the primary) - RUNNING

Rule: never-activated plans ranked last at screen and final stage (`REG:11599`, `REG:11609`); final scores from one mode-`all`
call (`REG:11615`). Orders and plan: 26 new candidates (s11 5, s23 5, s42 6, s67 4, s89 1, s103 5), 10 reused
(`REG:11627`). Verification launched once on Box 3 at 12:42:07 (`REG:11639`). **No result exists in the record.**

### E1-FIX (Amendment 74) - POST HOC REPAIR, registered 12:58; GPU scoring RUNNING since 13:21, no SUMO, no result

Three fixes together (never-activated last; common arrivals per (seed, state, draw); final draws = 2 x K_final); gate G0
(two final-stage invocations with different chunk sizes must give the same first six in the same order on all six seeds,
else STOP with verdict "WM cannot rank these plans reproducibly", no SUMO). Outside the A72 Holm family; Random arm reused
(`REG:11642-11652`). Implementation hashes and two clarifications appended 13:23 (`REG:11675-11681`: ranking statistic stays
summary.score; Fix 2 also seeds merge-noise keys, entrant CAV share fixed at 0.0, disclosed). Scoring launched on gpuR
13:21 (`REG:11682`); no score, G0 verdict or SUMO in the record yet.

### A75 (intent only, `REG:11668-11669`) - wider hold forms; not registered with hashes; nothing run.

---

## 3. Number conflicts

No disagreement found between REG lines and the cited files: every sha256 and every number checked (E1 counts, per-seed
contrasts, bank - U0 %, all E2 contrasts and effort, Development means, laptop CHARGES totals for A69 identity/mechanics,
A70, A71, A73) matches. Labelling/record flags are listed in `README.md`. Clock note: `REG:11673` records that the header times of the 11-13
entries labelled 13:20, 13:35, 14:10, 14:10 and 14:50 were estimates running ahead of the real clock (all written before
13:17 real time, in the order shown); this report quotes header times as written.

## 4. Suggestions (not computed; would need registration as post hoc descriptives)

- E2 most-delayed vehicles (CVaR90), five group means and hard braking per arm (fields exist in `E2C_RESULTS.json` `compare`).
- Interval for the executed hold vehicle-seconds gap (F - H20, F - S).
- E1: per-seed CIs of WM bank - Random bank (only point estimates are recorded).
