# LCMOVES_REPORT: CAV moves that combine lane changes with speed changes on wz3to2 (S-LCMOVES, phase lcmoves, 2026-11-06 18:20-19:15 +08)

## Verdict
**Lane-change + speed moves did not add verified cooperation beyond the static walls and pairs in this probe. The probe is underpowered.**

1. **Exactly one lane-change move was admitted** in 8 moments.
   - It is a lane_redistribute at D3200_p60 s989021 t450: three lane-1 CAVs move into closing lane 0 and hold it until 300 m before the drop.
   - Per replicate, S = +1,726 / +4,987 / +1,371 and T = +1,671 / +2,924 / +1,393 veh-s per 300 s.
   - **At that same moment, the static wall_g was better in 2 of 3 replicates.** J(wall_g) − J(LC) = −967 / −2,034 / +73 veh-s, so it is not "beyond static".
2. **Admitted moments by cell (all three replicates pass, later harm passes).**

   | cell | B lane-change | references (wall_g / wall_h / pair_t) | R (random programs) |
   |---|---|---|---|
   | p60 | 1/4 (Wilson 0.05–0.70) | 0/4 (0–0.49) | 0/4 (0–0.49) |
   | p30 | 0/4 (0–0.49) | 0/4 (0–0.49) | 0/4 (0–0.49) |

   - **The counts cannot rank cells or families.** The budget ran out before r2: 9 r1 survivors and 3 r0 candidates are untested (§4).
3. **Average gain per strategy.** This is r0 T: one undithered 300-s run per moment, 8 moments, pooled cells, mean [t95].

   | strategy | r0 T, veh-s per 300 s |
   |---|---|
   | rolling_wall | +3,453 [−34, +6,939] |
   | free_program #1 | +3,397 [+399, +6,396] |
   | wall_h (static) | +2,869 [+580, +5,158] |
   | wall_fill | +2,141 [+198, +4,085] |
   | wall_g (static) | +1,787 [+347, +3,228] |
   | R random program (n = 51) | +1,746 [+1,064, +2,427] |
   | free_program #0 | +1,605 [−394, +3,603] |
   | gom_multi | +305 [−371, +981] |
   | pair_t (static) | +181 [−564, +926] |
   | lane_redistribute | −23 [−1,125, +1,080] |
   | gom_1 | −236 [−928, +456] |

   - **Paired against wall_h on the same moments:**
     - rolling_wall: +584 [−1,526, +2,693]
     - free_program #1: +529 [−2,496, +3,554]
     - wall_fill: −728 [−2,580, +1,125]
     - lane_redistribute: −2,891 [−4,610, −1,173]
     - gom_1: −3,171 [−5,263, −1,079]
   - Lane-change walls (fill, rolling) are about as good as static walls on average, and not measurably better.
   - Lane redistribution and gap-open merges are worse on average.
   - The one admitted move belongs to the family with the worst average gain.
4. **Where lane changes add opportunity: low CAV share.**
   - At p30, a wall_fill configuration existed with no static wall or pair at 27% of window ticks. At p60 that never happened (0%).
   - Static references existed at 73% of p30 ticks.
   - No p30 move was admitted, so this opportunity is unverified.
5. **Mechanics.**
   - Changes INTO the closing lane execute reliably: wall_fill 8/8, rolling_wall 7/7, lane_redistribute 18/20, R 25/32.
   - They need a persistent lane request. Without it, members returned to lane 1 within about 2–20 s (smoke; amendment 1).
   - **Commanded merges out of lane 0 into a made gap almost never executed.**
     - gom_1: 4/4 mergers hit the fallback after 36–155 s of requests under lcMode 513.
     - gom_multi: 4/7 mergers merged on their own before the early trigger, 2/7 hit the fallback, and 1/7 merged after 3 s of requests.
   - Gap making fails here, as in NEWMAP (0/12).
6. **Safety.** 553 jobs: 0 collisions, 0 teleports, 0 emergency steps (a < −4.5 m/s²). The lowest group acceleration was −3.76 m/s².

   | cell | hard-brake veh-s, group / base (screened arms, mean) | TTC < 3 s veh-s, group / base |
   |---|---|---|
   | p60 | 9.6 / 15.9 | 1,297 / 1,410 |
   | p30 | 15.2 / 18.8 | 1,193 / 1,228 |

   - The admitted move: hard brake 0.5 / 0.0 / 0.5 vs base 2.5 / 5.0 / 2.5; TTC < 3 s 980 / 435 / 1,023 vs base 1,281 / 748 / 1,115; min a −3.0.
7. **Knobs used** (MAIN 17:3x tags, in every row).
   - Admitted move: lane change into SUMO lane 0 (the closing lane); persistent lane request with hold; negative speed target (speed matching before the change); lcMode 513; no tau.
   - The 30 r0 synergy candidates: lane change 24, of which 16 into the closing lane and 19 executed; own-tau change 19; positive speed target 28; negative 23.
   - **The interface needs** lane-change targets that include the closing lane, a persistent hold, and signed speed targets.
   - **Tau** appears in many candidates but in no admitted move, so its need is untested.
8. **Budget.** 149,720 / 150,000 sim-s. No top-up: MAIN chose option (b) at 18:5x.

## 1. Authority, question, registration
- **Owner (verbatim, logged by MAIN 17:14):** "I think the search should also include moves like this where it actively uses the lane change and accelerate or slow down to cooperate."
  - Phase lcmoves ≤ 150,000 sim-s, gpuA CPU only, TRAIN worlds 989020–989099.
- **MAIN instructions:**
  - 17:3x: do not restrict the lane space (closing lane allowed) and tag the knobs.
  - 18:3x: ≤ 8 gpuA workers (REGIME priority).
  - 18:5x: the one-candidate fallback, then stop.
- **Registration.** REGISTRATION.md "S-LCMOVES registration" at 18:22, before any physics.
  - Amendment 1: persistent lane hold, merger acceleration at amax, dither-probe comparison fix. Registered after smoke, before the funnel.
  - Amendment 2: r2 for one candidate.
  - Status lines are prefixed LCMOVES in CLEAN1_STATUS.md.

## 2. Design (as executed)
- **Base and conventions.**
  - Native CAVs: cav_typical, speedMode 27, LC2013. HDVs are never commanded; the core asserts that every member and dither id is a CAV.
  - 0.5 s physics, 1 s control.
  - J = road speed deficit + source waiting, in veh-s.
  - T = J(base) − J(group) and S = J(best solo) − J(group). Positive means improvement.
- **Cells and worlds.** D3200 × {0.6, 0.3}, worlds 989020 and 989021, with CRN across cells (same seed, nested CAV sets).
  - 8 moments: per world-cell, PLANGEN2 windows [450, 550] and [800, 900].
  - t0 is the earliest save time with a wall_fill configuration and a static reference.
- **B lane-change families.** Every number comes from the NEWMAP2 / COOPKINDS / PLANGEN2 / WMDATA registrations listed in the registration block.
  - **wall_fill:** A filler moves into the empty adjacent lane (closing lane allowed) and slot-tracks the two-member partial wall (kp 0.15, kv 0.6, −1.5…+1.0 m/s²). It settles within 25 m, then all three hold wall_h (15 m/s, 90 s, dmax 1.5).
  - **rolling_wall:** The front pair holds 15 m/s. A rear CAV 200–500 m behind catches up and changes into the uncovered lane, then holds. Hold 120 s.
  - **gap_open_merge:**
    - gom_1: a lane-1 gap maker tracks 25 m behind the merger. The merger accelerates and requests lane 1 from t0.
    - gom_multi: 2–3 gap makers at 25 / 50 / 75 m behind an upstream merger, which requests lane 1 from X_B − 1,200 m.
  - **lane_redistribute:** 2–3 CAVs of the locally most loaded lane (counts within 100 m) move to the least-loaded adjacent lane, closing lane included, with speed matching. They hold that lane until X_B − 300 m.
  - **free_program:** 2 seeded structured programs. A V2V-local group of 2–4 CAVs gets a shared speed action from the R grid, a shared start offset {0, 20, 40, 60} s, a shared tau {none, 0.6, 2.0}, and lane spreading.
- **References at the same moments:** wall_g, wall_h, pair_t.
- **R (matched effort):** as many random programs per moment as B had lane-change variants (6–7). Members are random within 300 m; each member's lane, timing, speed and tau are drawn independently.
- **Funnel (NEWMAP2 / PLANGEN2):**
  1. Screen r0 (H = 300 s).
  2. Solos: B best-T per family (≤ 5), references (≤ 2), R top-5.
  3. r0 candidates (LB95(T) > 0, LB95(S) > 0, S ≥ 1, safe), all replicated with a cap of 5 per class.
  4. r1, then r2 with 120-s extensions; later harm checked over [t0, t0 + 420).
  - Statistics: 30-s circular-block bootstrap, B = 2,000, RNG 20261103.
- **New dither-integrity rule.**
  - Ordered registered dithers are each probed for 10 s.
  - r1 and r2 use the first two dithers that change the dither vehicle's state.
  - Result: 34 of 35 probes changed state. One seeded random_1km dither was inert: that vehicle was already braking harder on its own.
  - Caveat: at p60 s989020 t450 the r1 dither (the CAV nearest the drop, leaving the road) changed only its own state. J of every arm was unchanged, so it was J-inert. No second effective dither existed there, so its 3 candidates are untested.
  - Rows carry `dither_integrity` and a per-replicate `dither_dJ` / `dither_J_inert`.

## 3. Funnel per family (8 moments: p60 4, p30 4)

| family | variants | LB95(T) > 0 | solos run | r0 synergy candidates | r1 pass | admitted |
|---|---|---|---|---|---|---|
| wall_fill | 8 | 6 | 6 | 2 | 0 | 0 |
| rolling_wall | 8 | 7 | 7 | 3 | 1 (p30; r2 not run) | 0 |
| gap_open_merge | 11 | 4 | 4 | 1 | 0 | 0 |
| lane_redistribute | 8 | 4 | 4 | 1 | 1 | **1** |
| free_program | 16 | 11 | 6 | 3 | 3 (r2 not run or dither-inert) | 0 |
| abreast_wall (ref) | 16 | 14 | 7 | 3 | 1 (wall_g p60; r2 not run) | 0 |
| closing_pair (ref) | 8 | 5 | 5 | 1 | 0 | 0 |
| R random programs | 51 | 37 | 33 | 16 | 5 (r2 not run or dither-inert) + 3 r1 not run | 0 |

- **r0 → r1 attrition:** 16 of 27 replicated candidates flipped or failed at r1 (11 passed, 3 of them on a J-inert dither). This is the same near-chaotic pattern as NEWMAP2 and PLANGEN2. Examples:
  - wall_fill p60 s989021 t450: S +132 → −7,658.
  - rolling_wall same moment: S +478 → −7,042.
  - wall_h p30 s989021 t470: S +4,208 → −2,546.
- **Screen level.** At 8 of 8 moments the best B lane-change variant had a higher r0 T than the best reference. This is a best-of-6–7 versus best-of-3 comparison, so it is inflated by selection; use the paired means in the verdict instead.

## 4. Admitted move and untested candidates
**Admitted (3 rows in LCMOVES_MOVES_B.jsonl, PLANGEN2 schema plus knobs, commands, member_lane_tracks, dither_integrity, beyond_static).**

| cell, t0 | family | members (lanes → target) | S per rep (LB95) | T per rep | T over 420 s | solo gains r1 / r2 / r3 | span t0 / held | 4-slot | RSU |
|---|---|---|---|---|---|---|---|---|---|
| p60 s989021 t450 | lane_redistribute | 273, 275, 279 (1, 1, 1 → 0; executed at el 1) | +1,726 / +4,987 / +1,371 (≥ +1,035) | +1,671 / +2,924 / +1,393 | +2,825 | −324/−226/−56; −5,008/−2,063/−2,507; −348/−135/+21 | 117 / 153 m | no | not all in field at t0; inside while held |

- Mechanism: one CAV moving into the closing lane alone is harmful. Three together help, so this is a pure group effect.
- Dithers: r2 was non-inert (ΔJ base −6,707). r3 had a small effect (ΔJ group +294).

**Untested candidates (r1 passed, r2 not run for budget unless noted).** S / T at r1, veh-s per 300 s.

| moment | variant | r1 S (LB95) | r1 T | lane change / knobs | observable (4-slot / RSU field held) |
|---|---|---|---|---|---|
| p60 s989021 t450 | wall_g (ref) | +3,387 (+1,056) | +4,958 | none | yes / yes |
| p60 s989021 t450 | R1 | +2,723 (+1,640) | +7,671 | lane-1 request not executed; tau 0.6 | yes / yes |
| p60 s989020 t800 | free_program #1 | +527 (+225) | +2,851 | no lane change; tau 0.6 | yes / yes |
| p60 s989020 t800 | R5 | +3,121 (+2,100) | +4,239 | into lane 2, executed | no / yes |
| p60 s989020 t800 | R4 | +2,379 (+1,234) | +3,657 | into lanes 0 and 2, executed | no / yes |
| p30 s989021 t470 | free_program #1 | +1,182 (+525) | +1,317 | no lane change; tau 0.6 | yes / yes |
| p30 s989021 t470 | rolling_wall | +973 (+404) | +1,072 | rear into lane 0, executed | no (279 m) / yes |
| p60 s989020 t450 | free_program #1, R3, R4 | r1 = r0 (dither J-inert): +4,711 / +6,274 / +298 | +7,286 / +8,859 / +2,728 | | free_1 yes / yes |
| p30 s989021 t800 | R3, R2, R5 (r0 only) | r0 S +3,850 / +943 / +516 | r0 T +5,298 / +1,810 / +5,330 | | no / yes |

## 5. Budget, hosts, compliance
- **Ledger.** gpuA `/root/autodl-tmp/lcmoves_20261106/ledger/CHARGES.jsonl`: 553 rows, 149,720 / 150,000 sim-s.

  | component | sim-s |
  |---|---|
  | natives | 3,600 |
  | smoke (≤ 1,000 registered) | 930 |
  | screen r0 (base + groups) | 40,200 |
  | solos r0 | 66,000 |
  | dither probes | 350 |
  | r1 | 36,600 |
  | r2 + extensions | 2,040 |

  - By class: shared 9,800, B 53,520, references 22,500, R 63,900.
  - 280 sim-s are unspent.
- **My planning error.** Solos cost about 8.3k per moment, against about 5.4k estimated. Wave 1 put all 8 moments through screen and solos without reserving budget for r2. Four funnels were stranded (BUDGET_SKIPS.jsonl): r2 of 3 moments (12,720) and r1 of p30 s989021 t800 (4,200). A future run should reserve about 25% of the budget for replicates.
- **Hosts.** gpuA only: ≤ 8 workers from 18:3x (MAIN), nice 19, own PIDs (`code/PIDS.txt`, driver 417842). No lcmoves process remains. cpuZ was not used. Wall time was about 2 minutes of physics in total (libsumo about 0.3–1 s per 300-s branch).
- **Compliance.**
  - Seeds 989020–989021 only.
  - No sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path was opened. No hetdata_truth, vType or driver truth file was read; HDV vTypes are drawn only into the route file.
  - HDVs were never commanded.
  - No git reset, clean, stash or push. Nothing was killed. Nothing was deleted.

## 6. Deliverables
- **Run dir:**
  - `LCMOVES_MOVES_B.jsonl` (3 rows, sha 446bfdbb).
  - `LCMOVES_MOVES_R.jsonl` (0 rows).
  - `LCMOVES_NEGATIVES.jsonl` (125 rows, sha 93201558): stage, reason, knobs, full member programs and r0/r1 numbers. This includes every untested candidate (stages r2_missing, r1_missing, no_effective_dither_r2).
- **Code** (`lcmoves/code`): lcm_core 48aaae38, lcm_plan 94ebbc93, lcm_analyze 2d9f2df2, lcm_run d11c856a, lcm_wave 429eac60, lcm_smoke ba875b40, lcm_families 4df417f5, population 6af5ee23.
  - Tests: `lcmoves/tests/` (offline_select, lcm_smoke2 4faefa85, lcm_amend2 25dc0bca, lcm_status, lcm_report_stats).
- **Hand-off to REGIME.** `lcm_families.py` is importable: `snapshot`, `feasible`, `propose`, `Executor`. The REGIME agent was messaged.
- **Results.** Local `lcmoves/results/final/` (LCM_SUMMARY.json with every verdict, the per-moment dither record, availability per tick, ledger, MOMENTS). The host keeps every RESULT.json at `/root/autodl-tmp/lcmoves_20261106/runs` (59 MB).
- **Records.** experiment_history.json and OPEN_QUESTIONS are not updated; that is left to the integrator.
