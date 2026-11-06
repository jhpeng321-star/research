# NEWMAP2_REPORT: deeper cooperation probe on wz3to2 (phase newmap2, task N3, 2026-11-06 14:40-15:20 +08)

**Plain summary (12 lines)**
1. **Recommended training cell: demand 3,200 veh/h × CAV share 0.6.** There, 3 of 6 decision moments (3 worlds × 2 windows) had a group move with admitted synergy, robust in all 3 replicates; the Wilson 95% interval is 0.19–0.81. All 3 were abreast walls, and all were observable from one member's 4 V2V slots at t0.
2. Runner-up: 3,200 × 0.4, with 2 of 6 (one closing-lane pair, one wall). Demand matters most. At 3,200 veh/h, 6 of 18 moments were admitted, versus 2 of 18 at 2,800 and 1 of 18 at 2,400. Share matters less: 4/18 at 0.6, 3/18 at 0.4, 2/18 at 0.2.
3. Base = native CAVs. N1's MAPPO-on-wz3to2 sanity result (N1_READY.json) has not appeared, so no MAPPO-base cell was run.
4. **B: 9 admitted variants at 9 distinct moments, in 6 of 9 cells, giving 27 example rows (3 replicate seeds each) in NEWMAP_MOVES_B.jsonl.** 7 are walls and 2 are closing-lane pairs. Synergy is +1.1k to +5.1k veh-s per 300 s (mean of the 3 replicates), which is 2–30% of base J.
5. Admission is strict. All three replicates (undithered plus two common-CRN dithers) need LB95(T)>0 and LB95(S)>0. The advantage must still be >0 over [t0, t0+420) (later-harm check), and every group arm must have 0 collisions and 0 teleports.
6. Robustness caveat: in 2 of the 9 B admissions (2,800 × 0.2 and 2,800 × 0.4), both dithers were inert (|ΔJ| < 50 veh-s). Their robustness is untested; 7 of 9 survived at least one dither that changed the outcome.
7. The funnel is steep and near-chaotic. Of 36 r0 synergy candidates, 32 were replicated, and 12 failed already at the first dither and 10 at the second. Replication is required: single-run admission overstates cooperation about 3×.
8. Walls beat pairs: walls 7 admitted of 102 screened, pairs 2 of 159. Pairs pass r0 almost as often, but they fail replication more (9 replicated → 2 admitted, versus walls 16 → 7). Gentle and hard work about equally well (wall_h 4, wall_g 3).
9. Pure group effects exist. In the 2,400 × 0.6 pair, both solos hurt (−660 / −327 veh-s) while the pair gains +2,056. In the 3,200 × 0.6 wall at 989201 t450, the solos give −3,583 / −2,420 / −38 and the wall gives +1,856.
10. **R (random proposals, identical verification): 1 admitted of 108, giving 3 rows in NEWMAP_MOVES_R.jsonl.** Mean r0 synergy of the random proposals is −558 veh-s (B walls +637, B pairs +486). Structured selection matters.
11. Safety: 965 runs, 0 failed, 0 collisions, 0 teleports. Mean hard-brake veh-s in the screened group arms was 7.9, versus 9.2 for base; the lowest group acceleration was −4.18 m/s² (base −3.84).
12. Budget: 298,980 / 300,000 sim-s on cpuV (own ledger, ≤24 workers after MAIN's cap, nice 19). TRAIN worlds were 989200–989202 only; gpuA was not used. Four amendments, all registered before the affected physics, cut the design to fit the budget (§2).

---

## 1. Question and authority
- Owner, 14:37: "do more probe of this new map and see if it can have more coop". Spec N3 is in `E:/agent-work/_sol/newmap_spec.md`.
- Phase newmap2, cap 300,000 sim-s. Registration block "NEWMAP2 (task N3)" in REGISTRATION.md, written at 14:50, before any newmap2 physics; amendments 1–4 follow it.
- Question: on wz3to2, which demand × CAV-share cell gives the most frequent and robust *group* cooperation, i.e. a multi-CAV move that beats both the base and the best single member? And can SUMO-verified per-seed examples for training arms B (and R) be produced from it?

## 2. Design (as executed)
- **Scenario.**
  - Map: a byte-identical copy of `assets/research/newmap_candidates/wz3to2` (net sha b6277898…). Lane 0 of `up` ends at X_B = 2,696 m.
  - Same scenario streams, HDV population and J as newmap_probe (code derived from `probe_core.py`).
  - 0.5 s physics, 1 s control, CAV speedMode 27. HDVs are never commanded.
- **Cells.**
  - Demand {2,400, 2,800, 3,200} veh/h on the peak profile; CAV share {0.2, 0.4, 0.6}.
  - Worlds 989200–989202, with CRN across cells: identical request streams, scaled by rate, and nested CAV sets.
  - Native discharge at the drop under breakdown was about 2,000–2,800 veh/h, so the three demands straddle capacity. 26 of 27 natives had approach-speed breakdowns (< 10 m/s), with fewer at a higher CAV share.
- **Moments** (geometry only, never outcome-based).
  - Natives run 0–900 s, with states every 10 s in [450, 550] and [800, 900].
  - Per window, the earliest saved time with both a closing-lane pair and an abreast wall; otherwise a pair; otherwise a wall. All 54 window-moments were valid: 53 had a pair, 51 had a wall.
  - Zone: x ∈ [X_B − 1,800, X_B − 400] on `up`.
  - Pair: a lane-0 CAV plus a lane-1 CAV with |dx| ≤ 100 m (minimum dx).
  - Wall: one CAV per lane, spread ≤ 200 m (minimum spread).
- **Moves.**
  - Every member holds target speed v_w with slowDown commands (a ≤ amax 1.0, ≥ −dmax) and lane-change mode 513.
  - A member is released at the drop (x ≥ X_B) or at the end of the hold.

  | variant | family | v_w (m/s) | hold | dmax |
  |---|---|---|---|---|
  | pair_g | closing_pair | 22 | 120 s | 1.0 |
  | pair_h | closing_pair | 15 | 90 s | 1.5 |
  | pair_t (timing) | closing_pair | 22 | until the closing-lane CAV is 150 m before the drop (≤ 150 s) | 1.0 |
  | wall_g | abreast_wall | 22 | 120 s | 1.0 |
  | wall_h | abreast_wall | 15 | 90 s | 1.5 |
  | R0, R1 | R_random | random {15, 18, 22} | random {90, 120, 180} s | random {1.0, 1.5} |

  - R0 and R1 each use 2–3 random zone CAVs within 300 m of each other, seeded by `nm2_R_v1:{sid}:{i}`.
- **Funnel.** All arms of a replicate are matched CRN branches from the same saved native state.
  1. Screen (r0, H = 300 s): base plus every group.
  2. Solos for the best-T variant per family (B pair, B wall, R) with LB95(T) > 0.
  3. r0 candidates (LB95(T) > 0, LB95(S) > 0, S ≥ 1 veh-s, safe) go to replicate r1. r1 adds a common dither: −1 m/s² for 2 s on the non-member CAV nearest the drop, in every arm. Under amendment 3, only the best-LB95(S) candidate per moment and arm is replicated.
  4. r1 survivors go to replicate r2, which uses the same dither on a seeded random non-member CAV within 1 km of the drop. They also get a 120-s extension of the base and group from the saved t0+300 states (later-harm check).
- **Statistics.**
  - S = min_i J(solo_i) − J(group), with the best solo re-chosen per replicate and no zero floor. T = J(base) − J(group).
  - J = road speed deficit + source waiting, in veh-s.
  - One-sided 95% lower bound from a 30-s circular-block bootstrap (B = 2,000, RNG 20261103) over [t0, t0+300). Same definitions as NEWMAP_RESEARCH.
- **Integrity.** A continuous 420-s base run equals the 300-s run plus the 120-s extension from its saved state within 0.05 veh-s; the first 300 bins are bitwise equal.
- **Amendments** (REGISTRATION.md; each registered before the affected physics):
  1. Dropped wall_l (18 m/s / 180 s). Replicate and later-harm horizon reduced from 450 to 420 s.
  2. MAIN cap: ≤ 24 cpuV workers, seeds within 989200–989399.
  3. One replicated candidate per moment and arm (4 other r0 candidates are reported as `r0_not_picked`). Replicates run the main window only.
  4. Later harm is tested on the undithered replicate via advantage T (WALLMINE's later-harm quantity). The planned move-dither arm (v_w − 1 m/s, 2 s start delay) and solo extensions were dropped for budget.

## 3. Results per cell
"Robust*" = admitted, and at least one dither replicate changed the base, group or best-solo J by ≥ 50 veh-s.

| cell (demand × share) | moments | B admitted moments | B robust* | Wilson95 | R admitted | admitted pair / wall | mean S of admitted (veh-s / 300 s) | B variants with LB95(T) > 0 at screen |
|---|---|---|---|---|---|---|---|---|
| 2400 × 0.2 | 6 | 0 | 0 | 0–0.39 | 1 | 0/0 | – | 12/23 |
| 2400 × 0.4 | 6 | 0 | 0 | 0–0.39 | 0 | 0/0 | – | 7/30 |
| 2400 × 0.6 | 6 | 1 | 1 | 0.03–0.56 | 0 | 1/0 | +2,219 | 10/30 |
| 2800 × 0.2 | 6 | 1 | 0 (dithers inert) | 0.03–0.56 | 0 | 0/1 | +2,225 | 18/28 |
| 2800 × 0.4 | 6 | 1 | 0 (dithers inert) | 0.03–0.56 | 0 | 0/1 | +2,324 | 11/30 |
| 2800 × 0.6 | 6 | 0 | 0 | 0–0.39 | 0 | 0/0 | – | 5/30 |
| 3200 × 0.2 | 6 | 1 | 1 | 0.03–0.56 | 0 | 0/1 | +5,052 | 15/30 |
| 3200 × 0.4 | 6 | 2 | 2 | 0.10–0.70 | 0 | 1/1 | +2,997, +1,124 | 21/30 |
| **3200 × 0.6** | 6 | **3** | **3** | 0.19–0.81 | 0 | 0/3 | +1,683, +1,816, +2,982 | 11/30 |

- **Marginals** (B admitted moments out of 18): by demand, 2400: 1, 2800: 2, 3200: 6. By share, 0.2: 2, 0.4: 3, 0.6: 4.
- **Timing:** 7 of the 9 admissions fall in the early window (450–550 s, breakdown onset); 2 fall in 800–900 s.
- **Approach speed at t0:** congested moments (< 10 m/s) 5 of 29 admitted; free-flow moments (≥ 10 m/s) 4 of 25.
- **Interpretation.** The table counts per cell are small (n = 6). The demand effect (3200 vs the rest: 6/18 vs 3/36) is the clearest signal. Choosing 0.6 over 0.4 at 3,200 rests on 3 vs 2 moments and is not statistically separable. Use 3,200 × 0.6, and 3,200 × 0.4 if MAPPO or N4 constraints favour 0.4.

## 4. Families, variants and the funnel

| family | screened | LB95(T) > 0 | solos run (best per moment) | r0 synergy candidates | replicated | admitted | mean r0 S where solos ran |
|---|---|---|---|---|---|---|---|
| closing_pair (g/h/t) | 159 | 55 | 30 | 12 | 9 | 2 (pair_t, pair_g) | +486 (median +185, 18/30 > 0) |
| abreast_wall (g/h) | 102 | 55 | 34 | 17 | 16 | 7 (wall_h 4, wall_g 3) | +637 (median +326, 23/34 > 0) |
| R_random | 108 | 47 | 33 | 7 | 7 | 1 | −558 (median −95, 10/33 > 0) |

- **Replicate attrition** (the 32 replicated candidates):
  - 12 failed the r1 main-window test.
  - 10 failed r2.
  - 0 failed the later-harm check.
  - 10 were admitted (9 B, 1 R).
  - Failures were sign flips of S under a 2-s, −1 m/s² pulse on one non-member CAV. Examples: 2400 × 0.4 pair_h S +1,568 → −211; 2800 × 0.6 wall_g S +2,915 → −897.
- **Pairs versus walls.** Pairs reach r0 synergy about as often as walls, but replicate worse. The closing-lane pair mechanism (NEWMAP_RESEARCH §5.3) exists, but here it is fragile. Walls covering all three lanes are the more reliable form.

## 5. Admitted moves (examples)

| arm | cell | world, t0 | variant (lanes) | S per replicate (LB95) | T per replicate | T over [t0, t0+420) | solo gains, r1 | V2V max distance at t0 / while held (m) | 4-slot observable |
|---|---|---|---|---|---|---|---|---|---|
| B | 2400 × 0.6 | 989200, 450 | pair_t (0, 1) | 2,383 / 2,383 / 1,890 (1,383 / 1,383 / 1,094) | 2,056 / 2,056 / 876 | 2,047 | −660, −327 (both harmful) | 4 / 4 | yes |
| B | 2800 × 0.2 | 989200, 450 | wall_h (0, 1, 2) | 2,224 / 2,224 / 2,226 (≥ 1,356) | 4,491 (all) | 10,500 | 1,750, 303, 2,267 | 86 / 86 | yes |
| B | 2800 × 0.4 | 989201, 450 | wall_h | 2,324 (all three; dithers inert) | 7,092 | 14,456 | 398, −1,203, 4,769 | 122 / 122 | no |
| B | 3200 × 0.2 | 989200, 530 | wall_h | 4,374 / 4,915 / 5,866 (≥ 2,642) | 6,629 / 7,169 / 7,942 | 14,383 | 780, 698, 2,255 | 79 / 79 | yes |
| B | 3200 × 0.4 | 989200, 460 | pair_g (0, 1) | 3,139 / 3,139 / 2,712 (≥ 1,461) | 3,969 / 3,969 / 3,403 | 6,908 | −19, 831 | 19 / 53 | yes |
| B | 3200 × 0.4 | 989202, 800 | wall_g | 1,166 / 1,040 / 1,167 (≥ 476) | 3,791 | 10,116 | −717, 0, 2,625 | 53 / 1,084 | yes |
| B | 3200 × 0.6 | 989200, 450 | wall_g | 2,199 / 653 / 2,197 (≥ 386) | 2,644 / 3,893 / 2,642 | 4,817 | −387, −753, 445 | 32 / 328 | yes |
| B | 3200 × 0.6 | 989201, 450 | wall_h | 1,894 / 1,659 / 1,894 (≥ 656) | 1,856 / 2,413 / 1,856 | 6,678 | −3,583, −2,420, −38 (all harmful) | 33 / 33 | yes |
| B | 3200 × 0.6 | 989201, 800 | wall_g | 3,569 / 2,618 / 2,760 (≥ 1,717) | 6,845 / 9,529 / 6,959 | 10,347 | 365, 0, 3,276 | 21 / 977 | yes |
| R | 2400 × 0.2 | 989200, 450 | R0 (2, 1, 2; 15–22 m/s random) | 1,934 / 1,934 / 1,161 (≥ 481) | 4,353 / 4,353 / 3,221 | 7,692 | 2,419, 948, 1,043 | 218 / 218 | no |

- **V2V range** of the B admissions:
  - at t0: 8 of 9 within 100 m, 9 of 9 within 200 m;
  - while all members were still held: 5 of 9 within 100 m, 6 of 9 within 200/300 m. In three congested walls one lane stays jammed while another moves, so held members drift 330–1,080 m apart.
  - The R admission spans 218 m and is not slot-observable.
- **Rows.** Each row carries:
  - `v2v_range` (100/200/300 m flags at t0 and while held), `observable` (4-slot rule, ego plus 3 nearest CAVs within 100 m), `observability`;
  - per-seed `synergy_veh_s` / `synergy_lower95` / `advantage_veh_s` / `adv_lower95`, `best_solo`, `solo_gain_veh_s`;
  - `replicate_summary` (S per replicate, mean, t95 CI), `later_harm`, `dither` (the seed's common dither), `safety`, `move` (exact plan parameters), `plan_sha256`, `restart` (native state path, t0, inserted count);
  - `profile` = issued acceleration per second for each member, or the measured value once released; `policy_profile` = the same member's measured native acceleration in that replicate's base arm; `perturbation_profile` = their difference. `duration_s` = the last member release second.
- **Format (coopmwm-mined-move-v1).** Every binding field in `MINED_BINDING_FIELDS` is present and type-checked locally. **Ingestion note for N4:** `group_lane_bank.binding()` refuses world seeds outside 986000–986071. That check must admit the wz3to2 TRAIN range before ingestion. Restarts come from the cpuV native states, not from a replay bank.

## 6. Safety
- 965 runs: 27 natives, 900 branches and 38 extensions. 0 failed, 0 collisions, 0 teleports. One vehicle-step with a < −4.5 m/s² across all runs; none in a screened or admitted group arm.
- Screened group arms versus base: hard-brake (a < −3) veh-s mean 7.85 vs 9.21; minimum acceleration −4.18 vs −3.84 m/s².
- Admitted arms: minimum acceleration −3.17 to −3.48 m/s². Hard-brake veh-s for group vs base ranged from 0.5 vs 19.5 down to 21.5 vs 21.5. Exceptions: 2800 × 0.2 wall_h 13.0 vs 4.5, and 3200 × 0.6 989201 t450 wall_h 13.5 vs 5.0. **The hard walls can add braking.**

## 7. Limitations
- The base is native CAVs. N1 had not delivered MAPPO-on-wz3to2 (no `N1_READY.json` on c4/nm-abstract at 15:15), and the budget is spent. MAPPO smoothing might absorb part of these gains.
- n = 6 moments per cell from 3 worlds; Wilson intervals overlap across most cells.
- Funnel shortcuts:
  - solos ran only for the best-T variant per family;
  - only one candidate per moment and arm was replicated;
  - the move-dither arm and solo later-window extensions were dropped (amendments 3–4).
  - The admitted counts are therefore lower bounds on what exists, but the tests applied to each admission are strict.
- Dithers were inert for 2 admissions (2800 × 0.2 and 2800 × 0.4). Treat those rows as less robust (`replicate_summary` shows identical S).
- **R vs B is not count-matched.** B screened 6 variants per moment, R only 2 random proposals. R has 3 rows, B 27. N4 should subsample or weight if the arms must be matched.
- Records: experiment_history.json and OPEN_QUESTIONS were not updated; that is left to the integrating agent. A close-out line was appended to REGISTRATION.md.

## 8. Evidence
- **cpuV `/home/ubuntu/newmap2_20261106/`** (119 MB):
  - `code/`: nm2_core / nm2_plan / nm2_run / nm2_analyze, PIDS.txt (965 own PIDs), RUNNER_PIDS.txt;
  - `maps/wz3to2`, `MOMENTS.json`;
  - `runs/D{d}_p{pp}/s{seed}/{native, m_t{t0}/{r0, r1, r2, r0ext, r0cont}/{base, g__*, s__*}}/RESULT.json`;
  - `jobs/*.jsonl` (the `*_superseded_*` files were planned and never run);
  - `ledger/CHARGES.jsonl` (965 rows, 298,980 sim-s);
  - `results/`.
- **Local:**
  - `runs/active/C4-V7FT-20261017/newmap2/{code, results}`: code sha nm2_core ad6b4bd6, nm2_plan 5ef8eb9d, nm2_run f3c0300c, nm2_analyze b0074755, nm2_tables 5a901b00, population 6af5ee23 (verbatim). results = NM2_SUMMARY.json, MOMENTS.json, CHARGES.jsonl, PIDS.
  - Run directory: `NEWMAP_MOVES_B.jsonl` (27 rows, sha 88adb68a…), `NEWMAP_MOVES_R.jsonl` (3 rows, d6b4b1af…), `NEWMAP_NEGATIVES.jsonl` (359 rows, bbb3dfbb…; stage and reason for every non-admitted screened variant).
- **Compliance:**
  - TRAIN worlds 989200–989202 only;
  - no sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path opened; no hetdata_truth or vType/driver/vehicle_params truth file read;
  - HDVs never commanded; original maps untouched;
  - own PIDs only, nothing killed; no git reset, clean, stash drop or push.
