# PLANGEN_REPORT: verified group-cooperation plans on wz3to2 for the clean4 bank (phase plangen, 2026-11-06 15:24-15:50 +08)

**Plain summary**
1. **B: 6 admitted group moves at 6 distinct moments, giving 18 per-seed rows in PLANGEN_MOVES_B.jsonl.** 4 are abreast walls (wall_h 3, wall_g 1) and 2 are closing-lane pairs (both pair_t). By cell: 3200 × 0.6 has 4 of 28 moments; 3200 × 0.4 has 2 of 11 screened moments.
2. **R (random proposals, count-matched effort): 5 admitted moves at 5 moments, giving 15 rows in PLANGEN_MOVES_R.jsonl.** 4 are in 3200 × 0.6 and 1 in 3200 × 0.4.
3. With identical verification effort, random proposals are admitted almost as often as structured ones (5 vs 6 of 39 screened moments). They are weaker and much less local, though:
   - mean admitted S per 300 s: B +2.1k vs R +0.9k;
   - 4-slot V2V observable at t0: B 6 of 6 vs R 1 of 5;
   - members within 100 m at t0: B 6 of 6 vs R 1 of 5.
4. The admission rate is lower than N3's small sample suggested. At 3200 × 0.6 it is 4 of 28 moments (Wilson 95% 0.06–0.31), against N3's 3 of 6. Pooled with N3: 3200 × 0.6 is 7 of 34, and 3200 × 0.4 is 4 of 17.
5. Robustness: 4 of 6 B and 4 of 5 R admissions survived at least one dither that moved J by at least 50 veh-s. Two B rows have inert dithers (989205 pair_t and 989211 wall_h), so their robustness is untested.
6. Safety: 1,009 runs, with 0 failed, 0 collisions, 0 teleports and 0 steps below −4.5 m/s².
   - Screened group arms vs base, hard-brake veh-s mean: 10.6 vs 12.8.
   - Lowest group acceleration: −4.04 m/s².
   - Two admitted hard walls add braking: 989211 wall_h 16.5 vs 6.0, and 989204 wall_h 6.0 vs 5.0.
7. Budget: 308,220 of 308,400 sim-s (300,000 plus MAIN's 8,400 top-up, amendment 1), own cpuV ledger, at most 20 workers at nice 19. gpuA was not used: a 300-s branch costs about 0.5 CPU-s, so the budget, not compute, was the limit.

## 1. Authority and design
- **Authority.** Owner 15:14 (phase plangen ≤ 300,000 sim-s). MAIN restricted the worlds to 989203–989349. Registration "PLANGEN registration" at 15:25 in REGISTRATION.md, written before any physics; amendment 1 follows it.
- **Pipeline.** This is N3's newmap2 pipeline, unchanged:
  - `nm2_core.py` (ad6b4bd6) and `population.py` (6af5ee23) are verbatim copies;
  - the planner, runner and analysis keep the same decision rules (`pg_plan` e8de81f5, `pg_run` 057f6f1f, `pg_analyze` cc70b0fe).
  - Moments: windows [450, 550] and [800, 900] s, at the earliest saved time that has both a pair and a wall.
  - B variants: pair_g, pair_h, pair_t, wall_g, wall_h.
  - Funnel: screen r0 → solos → one candidate per moment and arm → r1 and r2 with common-CRN dithers → 120-s extensions.
  - Admission: r0, r1 and r2 each need LB95(T) > 0, LB95(S) > 0 and S ≥ 1; T over [t0, t0+420) must be > 0; group arms need 0 collisions and 0 teleports.
- **R fairness.** R gets the same effort per moment as B:
  - 5 random proposals (B screens 5 variants);
  - solo sets for the top-2 T proposals (B: at most 2, one per family);
  - one replicated candidate per moment;
  - the same dithers, later-harm check and safety tests.
- **Cells and worlds.** Demand 3200 veh/h (peak profile), base native CAVs, HDVs never commanded.
  - Share 0.6: worlds 989203–989216 (14).
  - Share 0.4: worlds 989203–989208 (6).
  - The shared world ids give CRN across the two cells.
- **Waves.**
  - w1: 12 moments, 90,780 sim-s.
  - w2: 22 moments, 169,440 sim-s.
  - w3: 6 moments, 5 screened; the screen stage held back a 20,000 sim-s reserve.
  - Amendment 1: r2 and extensions for 4 moments that were left without r2 in w3, 8,400 sim-s.
- **Budget miss in w3.** The w3 reserve was too small (its solos cost 2.6k per moment against about 2k estimated). One moment (pg_D3200_p40_s989208_t800) was never screened, and is not counted in any table or in the negatives. Skipped units are listed in BUDGET_SKIPS.jsonl.

## 2. Counts per cell (B / R admitted moments)
| cell | worlds | moments | B admitted | B Wilson95 | B robust* | R admitted | R robust* |
|---|---|---|---|---|---|---|---|
| 3200 × 0.6 | 14 | 28 | 4 (wall 3, pair 1) | 0.06–0.31 | 2 | 4 | 3 |
| 3200 × 0.4 | 6 | 11 screened (+1 not screened) | 2 (wall 1, pair 1) | 0.05–0.48 | 2 | 1 | 1 |
| total | 20 world-cells | 39 screened | 6 | – | 4 | 5 | 4 |

"Robust*" means admitted, and at least one dither replicate changed the J of the base, group or best solo by ≥ 50 veh-s.

## 3. Counts per family (both cells, screened variants)
| family | screened | LB95(T) > 0 | solos run | r0 synergy candidates | replicated | admitted | r0 S where solos ran: mean / median / n > 0 |
|---|---|---|---|---|---|---|---|
| abreast_wall (g/h) | 78 | 58 | 32 | 15 | 15 | 4 (wall_h 3, wall_g 1) | +387 / +102 / 19 of 32 |
| closing_pair (g/h/t) | 117 | 45 | 23 | 11 | 6 | 2 (pair_t 2) | +196 / 0 / 16 of 23 |
| R_random | 195 | 80 | 52 | 20 | 17 | 5 | +62 / 0 / 30 of 52 |

- **Replicate attrition** (38 replicated candidates):
  - 18 failed r1;
  - 9 failed r2;
  - 0 failed the later-harm check;
  - 11 were admitted.
- As in N3, most failures are sign flips of S under a 2-s, −1 m/s² pulse on a single non-member CAV.
- Walls convert best: 4 of 15 replicated walls were admitted, against 2 of 6 pairs and 5 of 17 R.

## 4. Admitted moves
| arm | cell | world, t0 | variant (lanes) | S per replicate (veh-s / 300 s) | T per replicate | T over [t0, t0+420) | solo gains, rep 1 | V2V distance at t0 / while held (m) | 4-slot observable | all members in RSU field at t0 |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.6 | 989205, 450 | pair_t (0, 1) | 349 / 349 / 350 (dithers inert) | 5,119 ×3 | 9,075 | 4,765, 4,769 | 15 / 16 | yes | no |
| B | 0.6 | 989211, 800 | wall_h (0, 1, 2) | 2,193 ×3 (dithers inert) | 3,033 ×3 | 4,366 | 840, −734, −367 | 42 / 73 | yes | no |
| B | 0.6 | 989213, 800 | wall_h | 1,141 / 1,141 / 4,234 | 5,662 / 4,448 / 8,451 | 10,545 | 104, 845, 4,522 | 21 / 608 | yes | no |
| B | 0.6 | 989216, 450 | wall_g | 3,901 / 4,026 / 4,382 | 7,509 / 7,071 / 2,206 | 12,005 | 3,608, 985, 368 | 13 / 44 | yes | no |
| B | 0.4 | 989204, 800 | wall_h | 3,326 / 1,329 / 3,326 | 5,207 / 3,144 / 5,207 | 12,530 | 462, −26, 1,880 | 60 / 370 | yes | no |
| B | 0.4 | 989206, 840 | pair_t (0, 1) | 807 / 1,816 / 820 | 2,706 / 1,785 / 2,660 | 4,273 | 1,899, 488 | 10 / 926 | yes | no |
| R | 0.6 | 989204, 450 | R2 (0, 1, 2) | 505 / 1,589 / 1,303 | 5,446 / 1,678 / 5,137 | 6,292 | −1,150, −945, 4,941 | 285 / 286 | no | no |
| R | 0.6 | 989205, 450 | R0 (2, 1) | 1,225 ×3 (dithers inert) | 7,380 | 14,848 | 6,154, 2,991 | 74 / 80 | yes | yes |
| R | 0.6 | 989209, 450 | R1 (2, 0) | 529 ×3 | 1,179 / 1,179 / 1,372 | 807 | 417, 649 | 290 / 290 | no | no |
| R | 0.6 | 989215, 450 | R1 (2, 0) | 888 / 888 / 734 | 10,292 / 10,292 / 9,487 | 19,545 | 9,404, 288 | 280 / 280 | no | no |
| R | 0.4 | 989207, 800 | R3 (0, 1, 0) | 1,055 / 681 / 321 | 1,151 / 1,239 / 3,413 | 256 | −147, 96, −388 | 179 / 762 | no | no |

- **Super-additive cases.** In 989211 wall_h, two of the three solos hurt, and the wall gives +3.0k. In 989207 R3, two solos hurt (−147, −388) and the group gains +1.2k.
- **Observability.**
  - B: 6 of 6 within 100 m at t0 and 4-slot observable; 3 of 6 stay within 100 m while held.
  - R: 1 of 5 within 100 m at t0 and 1 of 5 4-slot observable.
  - RSU range (registered wz3to2 descriptor: RSU field x ≥ X_B − 875 m on through lanes, x ≥ X_B − 454 m on the closing lane): only 1 of 11 admissions has all members inside at t0, because moves start 400–1,800 m upstream. All 11 have at least one member inside the field before release.
  - "V2V ≤ 100 m or RSU field at t0": B 6 of 6, R 1 of 5.
- **Rows.** Each row uses the NEWMAP_MOVES_B schema (all 47 keys) plus four new keys: `moment`, `rsu_range`, `v2v100_or_rsu_field_t0` and `phase`.
  - per-seed `synergy_veh_s`, `synergy_lower95`, `advantage_veh_s` and `adv_lower95`;
  - `replicate_summary`, `later_harm`, `dither`, `safety`, `move`, `plan_sha256`, `restart` (cpuV native state);
  - `profile`, `policy_profile` and `perturbation_profile`.
- **Ingestion note for N4.** `group_lane_bank.binding()` must admit wz3to2 TRAIN 989203–989349. Restart states are on cpuV.

## 5. Limitations
- **Base and rates.** The base is native CAVs only; N4 re-verifies on the final base. The per-cell sample is still small (n = 28 and 11), and the rates are lower than N3's.
- **Funnel shortcuts inherited from N3 amendments 3–4.** There is one replicated candidate per moment and arm, and no move-dither arm. The counts are therefore lower bounds.
- **Inert dithers.** The dithers are inert for 2 B rows and 1 R row (identical S across replicates). Treat those rows as less robust.
- **R vs B row counts.** R and B are effort-matched per moment, but the row counts differ (B 18, R 15). N4 should weight or subsample if exact matching is needed.
- **Records.** experiment_history.json and OPEN_QUESTIONS were not updated; that is left to the integrator. REGISTRATION.md holds the registration, amendment 1 and a close-out line.

## 6. Evidence
- **cpuV `/home/ubuntu/plangen_20261106/`** (118 MB):
  - `code/`;
  - `runs/D3200_p{60,40}/s{seed}/{native, m_t{t0}/{r0, r1, r2, r0ext}/{base, g__*, s__*}}/RESULT.json`;
  - `jobs/w{1..4}_*.jsonl` and logs;
  - `ledger/CHARGES.jsonl` (1,009 rows, 308,220 sim-s);
  - `results/`, `MOMENTS.json`, `WORLDS.json`, `BUDGET_SKIPS.jsonl`.
- **Local `runs/active/C4-V7FT-20261017/plangen/`:**
  - `code/`, the same files plus `pg_deliver.py` (append-only delivery);
  - `results/` (PG_SUMMARY.json with all 400 verdicts, CHARGES.jsonl, PIDS, MOMENTS, WORLDS, BUDGET_SKIPS).
- **Run directory:**
  - `PLANGEN_MOVES_B.jsonl` (18 rows, sha 369ccd4f…);
  - `PLANGEN_MOVES_R.jsonl` (15 rows, 7c43b502…);
  - `PLANGEN_NEGATIVES.jsonl` (379 rows, 79e981f4…; stage and reason for every screened, non-admitted variant).
- **Compliance:**
  - TRAIN worlds 989203–989216 only;
  - no sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path opened; no hetdata_truth or vType/driver truth file read;
  - HDVs never commanded; own PIDs only, nothing killed; no git reset, clean, stash drop or push;
  - no plangen process is left on cpuV.
