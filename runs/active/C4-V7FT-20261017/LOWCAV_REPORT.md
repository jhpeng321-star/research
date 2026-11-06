# LOWCAV_REPORT: group cooperation at low CAV share on wz3to2 (phase lowcav, 2026-11-06 15:32-15:55 +08)

**Plain summary (10 lines)**
1. **Yes, down to a 0.2 CAV share. Not shown at 0.1.** Replicated group cooperation means a move that beats both the base and the best single member in all 3 replicates and passes the later-harm and safety checks.
2. At 3,200 veh/h the admitted moments were 0/8 at share 0.1 (Wilson 95% 0–0.32), 1/8 at 0.2 (0.02–0.47) and 2/8 at 0.3 (0.07–0.59). At 2,800 × 0.2 it was 1/6 (0.03–0.56).
3. At 0.1, groups exist and look good in single runs: 4/8 moments had r0 synergy. But all 4 flipped sign under the first 2-s common dither (r1 S = +520 with LB95 −41, −1,359, +428 with LB95 −311, −20 veh-s), so none is robust.
4. **Opportunity is not the bottleneck at 0.2 and above, but it is at 0.1.**
   - Some usable group (pair or wall, any tier) existed at 69% / 97% / 100% of 10-s ticks at 0.1 / 0.2 / 0.3, and 95% at 2,800 × 0.2.
   - Standard geometry (3-lane wall ≤ 200 m or closing-lane pair ≤ 100 m) existed at only 16% / 55% / 74%, and 45% at 2,800 × 0.2.
   - At 0.1, most opportunities are 2-lane walls (53% of ticks), because lane-0 CAVs are scarce.
5. The 4 admitted moves are 2 hard 3-lane walls (both at 0.3) and 2 closing-lane pairs (both at 0.2). Mean synergy is +1.2k to +1.6k veh-s per 300 s, 4–7% of base J. Advantage over [t0, t0+420) was +490 to +7,379 veh-s.
6. All 4 had every member within 100 m (V2V) at t0 and were observable from 4 V2V slots. While held, members stayed within 100 m in 2/4 moves, within 200 m in 3/4 and within 300 m in 3/4. None was fully inside the RSU field at t0, but all members were inside it 1–53 s into the hold.
7. Every dither was non-inert: each changed some J by ≥ 50 veh-s, so all 4 admissions are "robust*" in the NEWMAP2 sense.
8. Combining with NEWMAP2's different worlds, share 0.2 gives 2/14 at 3,200 (0.04–0.40) and 2/12 at 2,800 (0.05–0.45). This is consistent with a rate of about 10–20% of decision moments at 0.2–0.3.
9. Safety: 357 runs, 0 failed, 0 collisions, 0 teleports. One screened (not admitted) hard wall at 0.3 had one emergency step (−6.13 m/s²).
10. Budget: 114,300 / 120,000 sim-s on cpuV (own ledger, ≤ 10 workers at nice 19). TRAIN worlds were 989350–989353 only, base was native CAVs, and gpuA was not used.

---

## 1. Question, authority, design
- Owner, 15:16: "can you also check if cooperation can be done for lower cav counts in this setting?" Phase lowcav, cap 120,000 sim-s. The registration (REGISTRATION.md "LOWCAV", 15:32, plus a 15:34 RSU-range note) was written before any physics.
- The pipeline is NEWMAP2 (N3) as executed, including its amendments 3–4. Code: `lowcav/code/`.
  - `lc_core` 287a53b3 is nm2_core plus a read-only 10-s CAV-position log in the natives.
  - `lc_plan` afa2740f; `lc_run` 77ebc014 (cap 120,000, ≤ 10 workers); `lc_analyze`.
  - Map: byte copy of wz3to2 (net b6277898).
- **Cells.** Primary: 3,200 veh/h × share {0.1, 0.2, 0.3} on worlds 989350–989353. Secondary: 2,800 × 0.2 on worlds 989350–989352. CRN across cells, nested CAV sets. Natives run 0–900 s. Moments are the earliest qualifying save time in each window [450, 550] and [800, 900]. That gives 30 window-moments, all valid.
- **Families (B only, no R arm).**
  - Closing-lane pair: pair_g (22 m/s, 120 s) and pair_t (22 m/s until the lane-0 CAV is 150 m before the drop).
  - Abreast wall: wall_g (22 m/s, 120 s, dmax 1.0) and wall_h (15 m/s, 90 s, dmax 1.5).
  - pair_h was dropped: it had 0 of 16 admitted in NEWMAP2.
- **Low-share member tiers** (geometry only; zone [X_B − 1,800, X_B − 400]):
  - pair: std |dx| ≤ 100 m, else long ≤ 200 m;
  - wall: std3 (one CAV per lane, spread ≤ 200 m), else long3 (≤ 400 m), else a two-lane wall {1,2} or {0,2} (≤ 200 m).
- **Funnel and admission (unchanged from NEWMAP2).**
  1. Screen r0 (H = 300).
  2. Solos for the best-T variant per family.
  3. The best-LB95(S) r0 candidate per moment goes to r1 (dither −1 m/s² for 2 s on the non-member CAV nearest the drop), then r2 (seeded random non-member within 1 km).
  4. Each of r0, r1 and r2 needs LB95(T) > 0, LB95(S) > 0 and S ≥ 1 veh-s (30-s block bootstrap, B = 2,000).
  5. Later harm: T > 0 over [t0, t0 + 420) on r0.
  6. Group arms: 0 collisions and 0 teleports.
  - Replicate stages were budget-ordered per whole candidate. No candidate was skipped.

## 2. Opportunity: does a usable group exist?
These are fractions of the 10-s ticks in [300, 900] s (4 worlds × 61 ticks; 3 × 61 at 2,800). Zone CAVs is the mean number of CAVs in the selection zone.

| cell | zone CAVs | std pair (≤ 100 m) | any pair (≤ 200 m) | std3 wall | 3-lane wall ≤ 400 m | only a 2-lane wall | any wall | **std group** | **any group** |
|---|---|---|---|---|---|---|---|---|---|
| 3200 × 0.1 | 5.8 | 0.15 | 0.17 | 0.09 | 0.15 | 0.53 | 0.68 | **0.16** | **0.69** |
| 3200 × 0.2 | 12.4 | 0.50 | 0.57 | 0.50 | 0.61 | 0.35 | 0.96 | **0.55** | **0.97** |
| 3200 × 0.3 | 19.5 | 0.72 | 0.75 | 0.67 | 0.76 | 0.24 | 1.00 | **0.74** | **1.00** |
| 2800 × 0.2 | 9.0 | 0.41 | 0.49 | 0.32 | 0.44 | 0.49 | 0.93 | **0.45** | **0.95** |

- Restricting to the two decision windows gives the same picture: 0.1 std 0.16 / any 0.80; 0.2 0.56 / 0.97; 0.3 0.66 / 1.00; 2,800 × 0.2 0.50 / 0.94.
- Member tiers of the screened moments:
  - At 0.1: 4/8 moments had a std pair and std3 wall; the other 4 had only a two-lane {1,2} wall.
  - At 0.2: 7/8 std3 walls (one with a long pair), 1 two-lane.
  - At 0.3: 7/8 std; 1 long3 wall without any pair.
  - At 2,800 × 0.2: 5/6 std, 1 two-lane.

## 3. Admission per cell

| cell | moments | B variants with LB95(T) > 0 (pairs / walls) | moments with an r0 synergy candidate | passed r1 | passed r2 + later harm | **admitted moments** | Wilson 95% | admitted move(s): mean S over 3 replicates (veh-s / 300 s) |
|---|---|---|---|---|---|---|---|---|
| 3200 × 0.1 | 8 | 5/8 / 11/16 | 4 | 0 | 0 | **0** | 0–0.32 | – |
| 3200 × 0.2 | 8 | 5/14 / 9/16 | 3 | 1 | 1 | **1** | 0.02–0.47 | pair_g +1,227 |
| 3200 × 0.3 | 8 | 8/14 / 14/16 | 5 | 3 | 2 | **2** | 0.07–0.59 | wall_h +1,165; wall_h +1,636 |
| 2800 × 0.2 | 6 | 6/10 / 6/12 | 2 | 1 | 1 | **1** | 0.03–0.56 | pair_t +1,307 |

- **Context from NEWMAP2** (worlds 989200–989202, same pipeline): 3,200 × 0.2 1/6, 2,800 × 0.2 1/6, 3,200 × 0.4 2/6, 3,200 × 0.6 3/6.
  - Pooled share 0.2: 3,200 gives 2/14 (0.04–0.40) and 2,800 gives 2/12 (0.05–0.45).
  - Cells with n = 6–8 are not statistically separable. The only clear contrast is 0.1, where 0 of 4 r0 synergy moments survived the first dither, against 5 of 10 at 0.2–0.3 and 2,800 × 0.2.
- **Group value without replication (r0 screen).**
  - Walls usually beat base at every share: mean r0 T is +983 / +798 / +2,059 veh-s at 0.1 / 0.2 / 0.3.
  - Wall synergy at r0 (where solos ran): mean +385 (n = 7, 5 > 0) at 0.1, −118 (n = 6) at 0.2, +1,106 (n = 8, 6 > 0) at 0.3.
- **0.1 failures in detail** (r0 → r1 S, LB95 in parentheses):

| moment | wall | r0 S | r1 S |
|---|---|---|---|
| 989350 t520 | std3 wall_g | +2,068 | +520 (LB95 −41) |
| 989351 t490 | two-lane wall_g | +1,140 | −1,359 |
| 989351 t810 | std3 wall_h | +1,704 | +428 (LB95 −311) |
| 989353 t450 | two-lane wall_g | +1,110 | −20 |

  In every case the dither changed J by 1,075–2,742 veh-s. A single CAV-share-0.1 group effect is the same size as the noise from one 2-s CAV pulse.

## 4. Admitted moves (12 rows in LOWCAV_MOVES_B.jsonl, 3 replicate seeds each)

| cell | world, t0 | move (lanes, tier) | S per replicate (LB95) | T per replicate | T over [t0, t0 + 420) | solo gains, r1 | approach speed at t0 |
|---|---|---|---|---|---|---|---|
| 3200 × 0.2 | 989352, 450 | pair_g (0, 1; std) | 587 / 829 / 2,265 (201 / 432 / 1,011) | 427 / 1,919 / 2,349 | 1,246 | −160, −1,386 (both harmful) | 8.3 m/s |
| 3200 × 0.3 | 989351, 810 | wall_h (0, 1, 2; std3) | 2,395 / 752 / 347 (1,489 / 55 / 3) | 3,846 / 1,466 / 6,534 | 7,379 | −391, −283, +1,451 | 5.1 m/s |
| 3200 × 0.3 | 989352, 450 | wall_h (0, 1, 2; std3) | 1,776 / 1,776 / 1,355 (351 / 351 / 558) | 2,967 / 2,547 / 2,587 | 6,525 | −11, −276, +1,191 | 12.3 m/s |
| 2800 × 0.2 | 989352, 800 | pair_t (0, 1; std) | 1,745 / 1,328 / 849 (994 / 654 / 260) | 1,705 / 2,234 / 1,052 | 490 | −245, −40 (both harmful) | 5.5 m/s |

- **Robustness.**
  - Every dither changed base, group or best-solo J by ≥ 50 veh-s (max |ΔJ| vs r1: 1,496 / 2,121; 4,325 / 4,765; 421 / 513; 751 / 456).
  - At 989352 t450 (0.3), dither 1 moved only the base (S unchanged).
  - 989351 t810 (0.3) passed r2 by a thin margin (LB95(S) = +3).
- **Pure group effects.** Both pairs (0.2) are better than base, while both of their solos are harmful.
- **Spacing and range.**

| move | max member distance at t0 | max distance while held | farthest member from the drop at t0 | all members in RSU field after |
|---|---|---|---|---|
| pair_g (0.2) | 40.9 m | 158 m | 463 m | 1 s |
| wall_h 989351 t810 (0.3) | 39.4 m | 39.4 m | 1,709 m | 53 s |
| wall_h 989352 t450 (0.3) | 50.4 m | 59.2 m | 1,261 m | 52 s |
| pair_t (2,800 × 0.2) | 37.9 m | 361 m | 1,370 m | 43 s |

  - V2V ≤ 100 / 200 / 300 m: 4 / 4 / 4 at t0, and 2 / 3 / 3 while held.
  - All 4 are 4-slot observable at t0. None is inside the RSU field at t0, because the selection zone starts 400 m before the drop.
  - The RSU field is lane 0 in [X_B − 454, X_B] and lanes 1/2 in [X_B − 875, X_B + 150] (descriptor c4/nm-abstract 65d5f619).
- **All selected groups, admitted or not** (pair_g / wall_g member sets):
  - pairs are within 100 m at t0 in 22/23 cases;
  - walls are within 100 m at t0 in 19/30 and within 200 m in 29/30;
  - while held, many members drift beyond 300 m: median 317–400 m for walls at 3,200, because jammed lanes separate.

## 5. Safety
- 357 runs (15 natives, 332 branches, 10 extensions): 0 failed, 0 collisions, 0 teleports.
- One emergency step (a < −4.5 m/s²) occurred: −6.13 m/s² in the screened hard wall 3,200 × 0.3 989351 t550 wall_h. It was rejected at screen (T = −3,867).
- Admitted arms:
  - minimum acceleration −3.19 to −3.53 m/s² (base −3.40 to −3.70);
  - hard-brake veh-s for group vs base: 4–5.5 vs 7–8.5 (0.2 pair); 8.5 vs 23–37 and 1–4 vs 10–15.5 (0.3 walls); 4–6.5 vs 1–5.5 (2,800 pair; slightly more braking).
- Screened group arms vs base, mean hard-brake veh-s: 9.8 vs 15.8 (0.1); 11.2 vs 10.8 (0.2); 14.8 vs 21.5 (0.3); 6.4 vs 8.3 (2,800 × 0.2).

## 6. Limitations
- n = 6–8 moments per cell from 3–4 worlds, so the Wilson intervals overlap. Claim only that "admitted at 0.2 and 0.3; 0/8 at 0.1". Do not claim a rate difference between 0.2 and 0.3.
- These are NEWMAP2 funnel shortcuts, so the admitted counts are lower bounds:
  - solos ran only for the best-T variant per family;
  - only one candidate per moment was replicated;
  - there was no move-dither arm.
- The base is native CAVs. A MAPPO base could absorb part of the gains.
- At 0.1 only one wall member set per moment was screened (the best tier). Longer-reach or 2-lane walls at 0.1 were tested at only 4 moments.
- The schema matches NEWMAP_MOVES_B (coopmwm-mined-move-v1) except that `newmap2_id` is replaced by `lowcav_id`, and `member_tier` and `rsu_range` are added.
  - The ingestion note still applies: `group_lane_bank.binding()` must admit 989350–989399.
- experiment_history.json and OPEN_QUESTIONS were not updated; that is left to the integrating agent, as for N3. 5,700 sim-s are unspent and unallocated.

## 7. Evidence
- **cpuV `/home/ubuntu/lowcav_20261106/`** (66 MB):
  - `code/` (+ PIDS.txt 357 own PIDs, RUNNER_PIDS.txt);
  - `maps/wz3to2`, `MOMENTS.json`;
  - `runs/D{d}_p{pp}/s{seed}/{native, m_t{t0}/{r0, r1, r2, r0ext}/{base, g__*, s__*}}/RESULT.json`;
  - `jobs/*.jsonl` and logs;
  - `ledger/CHARGES.jsonl` (357 rows, 114,300 sim-s);
  - `results/`.
- **Local:**
  - `runs/active/C4-V7FT-20261017/lowcav/{code, results}`: LC_SUMMARY.json, LC_TICKS.json, MOMENTS.json, CHARGES.jsonl, PIDS.
  - `LOWCAV_MOVES_B.jsonl` (12 rows, sha f2b802b3…) and `LOWCAV_NEGATIVES.jsonl` (102 rows, sha 076ad9da…).
- **Compliance:**
  - TRAIN 989350–989353 only;
  - no sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path; no truth or vType parameter file read;
  - HDVs never commanded;
  - own PIDs only, nothing killed, none left running;
  - no git reset, clean, stash drop or push.
