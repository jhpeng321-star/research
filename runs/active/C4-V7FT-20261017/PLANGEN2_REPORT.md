# PLANGEN2_REPORT: more verified group-cooperation plans for clean4 on wz3to2 (phase plangen2, 2026-11-06 16:10-16:35 +08)

**Plain summary (10 lines)**
1. **B: 8 admitted group moves, 24 per-seed rows** (`PLANGEN2_MOVES_B.jsonl`), at the canonical cell 3,200 veh/h × CAV share 0.6 on fresh TRAIN worlds 989007-989018. PLANGEN had 6 moves on this map.
2. Family mix: walls 3 (wall_h 2, wall_g 1), closing-lane pairs 2 (pair_t, pair_g), lane_balance 1, early_merge 1, zipper_order 1. Every family that was asked for produced at least one admission.
3. Admission rule as in PLANGEN/N3: 3 replicates (undithered + 2 common-CRN dithers) each need LB95(T)>0, LB95(S)>0 and S≥1. Later harm (T over [t0, t0+420) > 0) and 0 collisions/teleports in group arms are also required.
4. 7 of 23 screened moments held ≥1 admitted B move (Wilson 95% 0.16-0.51), versus PLANGEN's 4 of 28 at this cell. Part of the gain comes from the new families, and part from replicating every r0 candidate, which added 1 B and 2 R admissions.
5. Size: mean S over the 8 B moves is +3.2k veh-s per 300 s (range +1.2k to +7.2k), and 7 of 8 are super-additive. The best is lane_balance s989007 t800, with S of +6.7k/+7.0k/+7.8k from solos ≤ +173.
6. **R (random proposals, identical verification): 6 admitted moves, 18 rows.** Mean S is +1.4k. None of the 6 is 4-slot observable, against 7 of 8 for B. Three of the six are random lane_balance proposals.
7. Robustness caveat: in 3 B moves (pair_t and early_merge at s989015 t800, pair_g at s989018 t450) both dithers were inert (|ΔJ| ≤ 10 veh-s), so their robustness is untested. The other 5 B moves survived at least one dither that changed J by ≥ 50.
8. V2V: 7 of 8 B moves span ≤ 100 m at t0 and all 8 span ≤ 200 m; 7 of 8 fit one member's 4 V2V slots. The early_merge move spans 151 m.
9. Safety: 983 jobs, 0 failed, 0 collisions, 0 teleports, 0 steps below −4.5 m/s². The lowest acceleration in a screened group arm was −4.37 m/s²; admitted arms ranged from −3.07 to −3.69.
10. Budget: 292,110 / 300,000 sim-s (gpuA 225,030, cpuV 67,080). Following MAIN's 16:2x instruction, cpuV was stopped after wave 2 and wave 3 ran on gpuA only, at ≤12 workers. The last 7,890 sim-s could not carry a full moment funnel and were left unspent.

---

## 1. Authority and question
- **Authority.** Owner 15:44: "try everything to boost the chances and have more agents work on it to guarantee speed". MAIN set phase plangen2 at ≤300,000 sim-s, TRAIN worlds 989000-989099, cell 3200 × 0.6, network wz3to2.net.xml sha b6277898.
- **Registration.** REGISTRATION.md "PLANGEN2 registration" at 16:10, written before any physics. Amendment 1 (16:2x) records the MAIN host reallocation and the cpuV → gpuA sub-cap transfer.
- **Question.** Produce more SUMO-verified per-seed group-cooperation plans (B) for clean4, plus a random-procedure arm (R) that receives identical verification effort.
- **Closest prior work.**
  - PLANGEN: B 6 / R 5, walls and closing pairs, worlds 989203-989216.
  - COOPKINDS: lane_balance 3, early_merge 1, zipper 1, worlds 989400-989402.

## 2. Design (as executed)
- **Pipeline: PLANGEN/N3 unchanged.**
  - Natives run 0-900 s. States are saved every 10 s in two windows, [450, 550] and [800, 900].
  - Per window, t0 is the earliest save time that has both a closing pair and an abreast wall (all 24 moments had both).
  - Screen r0: base plus every variant, H = 300.
  - Solos: for B, the best-T variant of each family; for R, the top-5 T proposals. All require LB95(T)>0, T≥1 and a safe group arm.
  - r0 candidate → r1 (−1 m/s² dither for 2 s on the non-member CAV nearest the drop, all arms) → r2 (seeded random non-member CAV within 1 km) → 120-s extensions of r0 base and group.
  - Statistics: 30-s circular-block paired bootstrap, B = 2,000, RNG 20261103. The best solo is re-chosen in each replicate, with no zero floor.
- **B variants at t0.**

  | variant | members and rule | parameters |
  |---|---|---|
  | pair_g, pair_t | lane-0 + lane-1 CAV, \|dx\| ≤ 100 m, hold v_w | pair_g: 22 m/s, 120 s. pair_t: 22 m/s until the lane-0 CAV is 150 m before the drop |
  | wall_g, wall_h | one CAV per lane, spread ≤ 200 m | wall_g: 22 m/s, 120 s, dmax 1.0. wall_h: 15 m/s, 90 s, dmax 1.5 |
  | lane_balance | 2-3 lane-1 CAVs, 600-1,800 m upstream, spread ≤ 300 m; `changeLane(2)` until 300 m before the drop | |
  | early_merge | 2-3 lane-0 mergers `changeLane(1)` from X_B − 1,200 m; 1-2 lane-1 room-makers | room-maker tau 2.0 s |
  | zipper_order | lane-0 CAV M (lcMode 512, merges at X_B − 100); lane-1 CAV Y tracks slot x_M − 25 m | |

  - pair_h was dropped: it had 0 admissions in both NEWMAP2 and PLANGEN.
  - The lane families use COOPKINDS rules and member sets.
- **R: 7 proposals per moment, mirroring B's 4 hold variants and 3 lane variants.**
  - R0-R3: random holds (PLANGEN `nm2_R_v1`).
  - R4-R6: one random lane_balance, early_merge and zipper_order each. Parameters come from the COOPKINDS R grid and the member set is a random valid set.
- **Registered difference from PLANGEN.** Every r0 candidate is replicated (N3's original design before its amendment 3), capped at 5 per moment and arm. B has ≤ 1 candidate per family by construction. 30 moment-arms had candidates: 18 with 1, 8 with 2 and 4 with 3.
- **Worlds.**
  - Used: 989007-989018 (989000-989006 skipped because NEWMAP-PROBE had already simulated them on wz3to2).
  - gpuA: 989007-989010, 989013-989015, 989017-989018. cpuV: 989011, 989012, 989016.
  - Moment pg2_D3200_p60_s989018_t800 was not screened (budget receipt in `BUDGET_SKIPS.jsonl`).
- **Base and commands.** The base is native CAVs (cav_typical, speedMode 27, LC2013). The core asserts that every commanded id is a CAV; HDVs are never commanded.

## 3. Funnel per family (23 screened moments)

| family | screened | LB95(T)>0 | solos run | r0 candidates (all replicated) | admitted | mean r0 S where solos ran (median, n>0) |
|---|---|---|---|---|---|---|
| closing_pair (g/t) | 46 | 14 | 11 | 5 | **2** | +786 (+0, 7/11) |
| abreast_wall (g/h) | 46 | 27 | 17 | 7 | **3** | −3 (−50, 7/17) |
| lane_balance | 23 | 10 | 10 | 7 | **1** | +1,043 (+983, 7/10) |
| early_merge | 20 | 6 | 6 | 3 | **1** | +349 (+175, 3/6) |
| zipper_order | 20 | 4 | 4 | 3 | **1** | +1,036 (+750, 3/4) |
| R_random | 155 | 64 | 61 | 21 | **6** | −218 (+0, 30/61) |

- **Replicate attrition** over the 46 replicated candidates:
  - 25 failed r1 and 7 failed r2, almost all through sign flips of S under a single 2-s pulse.
  - 0 failed the later-harm check.
  - 14 were admitted (B 8, R 6).

## 4. Admitted moves
S and T are in veh-s per [t0, t0 + 300), listed per replicate (r1 / r2 / r3).
- "Dither ΔJ" is the largest change of base, group or best-solo J caused by each dither. "inert" means both changes were ≤ 10 veh-s.
- "Super-add." means T(r1) exceeds the sum of the solo gains.

| arm | world, t0 | variant (lanes) | S per rep (LB95) | T per rep | T over 420 s | solo gains r1 | super-add. | span t0 / held (m) | 4-slot | dither ΔJ | hard-brake g/b, min a |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 989007, 800 | lane_balance (1,1,1) | 6,723 / 6,986 / 7,810 (≥4,795) | 6,897 / 7,288 / 8,453 | 11,677 | 135, 82, 173 | yes | 96 / 541 | yes | 757, 1,284 | 4.5/17.5, −3.11 |
| B | 989008, 450 | wall_g (0,1,2) | 2,344 / 3,955 / 1,990 (≥351) | 2,504 / 4,917 / 2,504 | 5,571 | −75, 80, 161 | yes | 36 / 61 | yes | 3,723, 353 | 1.5/1.0, −3.69 |
| B | 989008, 810 | wall_h (0,1,2) | 4,036 / 551 / 4,042 (≥160) | 7,607 / 3,491 / 7,615 | 16,657 | 32, −173, 3,571 | yes | 28 / 405 | yes | 8,191, 8 | 6.5/17.5, −3.15 |
| B | 989013, 800 | wall_h (0,1,2) | 749 / 1,849 / 1,137 (≥565) | 3,467 / 5,166 / 3,738 | 5,833 | −1,148, −10, 2,718 | yes | 8 / 612 | yes | 1,431, 271 | 16.5/13.5, −3.30 |
| B | 989015, 800 | pair_t (0,1) | 2,690 (all three; LB 944) | 5,418 | 8,707 | 2,729, −1,590 | yes | 4 / 52 | yes | inert | 2.0/3.0, −3.07 |
| B | 989015, 800 | early_merge (0,0,0,1,1) | 1,281 (all three; LB 335) | 5,823 | 10,387 | 4,542, 2,476, 4,443, 3,080, −1,281 | no | 151 / 292 | no | inert | 3.5/3.0, −3.07 |
| B | 989017, 450 | zipper_order (0,1) | 3,389 / 1,485 / 3,389 (≥840) | 3,385 / 1,540 / 3,386 | 5,554 | −72, −3 | yes | 40 / 173 | yes | 1,024, 2 | 10.0/23.0, −3.23 |
| B | 989018, 450 | pair_g (0,1) | 4,797 / 4,798 / 4,789 (≥2,947) | 572 / 573 / 554 | 625 | −5,686, −4,225 | yes | 4 / 24 | yes | inert (≤10) | 3.5/0.5, −3.17 |
| R | 989008, 810 | R0 hold (2,0) | 2,658 / 602 / 2,680 (≥306) | 5,642 / 4,972 / 5,633 | 13,895 | 2,984, −31 | yes | 226 / 551 | no | 4,745, 31 | 4.5/17.5, −3.15 |
| R | 989012, 800 | R4 lane_balance (1,1,1) | 2,462 / 5,141 / 2,186 (≥1,250) | 1,238 / 5,613 / 2,102 | 4,604 | −1,358, −1,504, −1,224 | yes | 125 / 586 | no | 5,269, 869 | 16.0/6.0, −3.54 |
| R | 989013, 450 | R4 lane_balance (1,1,1) | 273 / 216 / 160 (≥73) | 522 / 452 / 440 | 510 | 247, 249, 33 | no | 159 / 219 | no | 102, 112 | 4.0/2.5, −3.25 |
| R | 989015, 800 | R0 hold (2,2) | 1,103 (all three) | 6,072 | 8,917 | 750, 4,969 | yes | 297 / 301 | no | inert | 2.0/3.0, −3.07 |
| R | 989015, 800 | R4 lane_balance (1,1,1) | 361 (all three) | 5,179 | 8,716 | 4,647, −2,277, 4,818 | no | 289 / 529 | no | inert | 2.0/3.0, −3.07 |
| R | 989017, 450 | R1 hold (1,1,1) | 477 / 3,442 / 477 (≥288) | 851 / 3,796 / 851 | 1,823 | 374, 71, 263 | yes | 158 / 158 | no | 3,912, 3 | 19.5/23.0, −3.24 |

**Notes**
- **Moment s989015 t800.** The dither target produced no change in any arm, so all four admissions there (B pair_t, early_merge; R R0, R4) rest on one unperturbed trajectory. Weight them lower, as N3 did with its inert-dither rows.
- **pair_g s989018 t450.**
  - The group barely beats base: T = +0.57k.
  - Its large S comes from both solos being very harmful: −5.7k and −4.2k.
  - It is a genuine "group avoids the solo harm" case, but the advantage is small.
- **Timing.** By window, B admissions are 3 early and 5 late. By approach speed at t0, 6 were congested (< 10 m/s) and 2 were near free flow (16.5 and 20.8 m/s).

## 5. B versus R (identical per-moment effort)
- **Effort.** Both arms had the same caps: 7 proposals each, ≤ 5 solo sets, all r0 candidates replicated with ≤ 5 per arm, and the same dithers and tests. In practice R used more solo sets (61 vs 48), because more random proposals passed LB95(T).
- **Admissions.** B 8 moves at 7 moments; R 6 moves at 5 moments. Mean S: B +3.2k, R +1.4k.
- **Observability.** B moves are local: 7 of 8 have a ≤ 100 m span and are 4-slot observable. R moves are spread out: 0 of 6 are observable, with spans of 125-297 m.
- **R's best moves** come from the lane_balance kind (3 of 6). Random lane_balance is close to the B rule, differing only in member choice.

## 6. Safety

| host | jobs | breakdown | failed | collisions | teleports | emergency steps (a < −4.5) | min group a | min solo a | mean hard-brake veh-s (group / base) |
|---|---|---|---|---|---|---|---|---|---|
| gpuA | 748 | 9 natives, 710 branches, 29 extensions; plus 15 smoke | 0 | 0 | 0 | 0 | −4.37 m/s² | −4.12 m/s² | 13.0 / 12.0 |
| cpuV | 220 | | 0 | 0 | 0 | 0 | −3.85 m/s² | | 13.8 / 12.6 |

- On gpuA the smoke jobs were the 15 arms of moment s989007 t450 run for 30 s each, 450 sim-s in total.
- **Admitted group arms:**
  - Minimum acceleration was −3.07 to −3.69 m/s².
  - Hard-brake veh-s vs base is listed in §4. It was higher than base in wall_g s989008 t450 (1.5 vs 1.0), wall_h s989013 t800 (16.5 vs 13.5), pair_g s989018 t450 (3.5 vs 0.5) and early_merge (3.5 vs 3.0).

## 7. Budget, hosts, compliance
- **Ledgers** (phase "plangen2", precharged per job, over-cap jobs refused):

  | host | ledger | rows | sim-s charged | sub-cap |
  |---|---|---|---|---|
  | gpuA | `/root/autodl-tmp/plangen2_20261106/ledger/CHARGES.jsonl` | 763 | 225,030 | 232,920 after amendment 1 |
  | cpuV | `/home/ubuntu/plangen2_20261106/ledger/CHARGES.jsonl` | 220 | 67,080 | 80,000, then frozen |

  - Total: 292,110 / 300,000 sim-s. No job was refused.
- **Hosts and timing.**
  - Waves 1-2 ran 16:15-16:23: gpuA at 14 workers (≥ 6 of its 20-core quota left free; WM-B was idle there), cpuV at 6 workers, nice 19.
  - Wave 3 ran 16:25-16:26 on gpuA only, at 12 workers.
  - cpuV's wave 2 had already finished all physics (16:22:55) when MAIN's cpuV-off instruction arrived. cpuV WORKERS was set to 0 at 16:23, and nothing ran there afterwards except one nice-19 read-only safety tally and file copies.
- **gpuA interpreter.** `/root/mining20261106/cpuv_home/miniconda3` (python 3.10.16, SUMO 1.27.1, numpy 2.2.6, identical to cpuV), used read-only.
- **Processes.** Own PIDs only, listed in `code/PIDS.txt` and `DRIVER_PIDS.txt` per host. Nothing was killed or signalled. No own process remains on either host.
- **Compliance.**
  - Seeds used: 989007-989018 only. Never 989100+, 964000-964199, 962xxx or 977xxx.
  - No sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path was opened. No hetdata_truth or vType/driver truth file was read (HDV vTypes are drawn only into the SUMO route file).
  - No git reset, clean, stash drop or push.
- **Record correction.** The 14 placeholder negatives of the budget-unscreened moment pg2_D3200_p60_s989018_t800 (stage `screen_missing`) were removed from `PLANGEN2_NEGATIVES.jsonl` at 16:33. They were budget skips, not rejections.

## 8. Deliverables and evidence
- **Rows** (in `runs/active/C4-V7FT-20261017/`):
  - `PLANGEN2_MOVES_B.jsonl`: 24 rows = 8 moves × 3 replicate seeds.
  - `PLANGEN2_MOVES_R.jsonl`: 18 rows = 6 moves × 3.
  - `PLANGEN2_NEGATIVES.jsonl`: 296 rows, each with a stage (screen 201, solos 63, r1 25, r2 7) and a reason.
  - Schema: PLANGEN `coopmwm-mined-move-v1`, plus `r_kind`, `member_roles` and `commands` (lane-change / tau / lcMode commands as [el, id, kind, value]).
  - `restart` points to the host's native state. Restart files live on gpuA under `/root/autodl-tmp/plangen2_20261106/runs/...`; cpuV restarts are under `/home/ubuntu/plangen2_20261106/runs/...`.
  - **Ingestion note for N4:** `group_lane_bank.binding()` must admit 989000-989099.
- **Code.** `runs/active/C4-V7FT-20261017/plangen2/code/`: pg2_core fb00fe48, pg2_plan e2e69e87, pg2_run b6916768, pg2_analyze 115fedd1, pg2_deliver beab48a4, pg2_wave 5badd966, population 6af5ee23. The patch provenance is in `plangen2/provenance/`.
- **Local results.** `plangen2/results/`:
  - `fetch_w{1,2,3}_{host}/`: per-wave PG2_SUMMARY.json and row files.
  - `host_{gpuA,cpuV}/`: ledgers, PIDs, MOMENTS, WORLDS, logs, BUDGET_SKIPS.
- **Host evidence.** Every RESULT.json with per-second J. gpuA `/root/autodl-tmp/plangen2_20261106/` (67 MB); cpuV `/home/ubuntu/plangen2_20261106/` (24 MB).
- **Records.** REGISTRATION.md (PLANGEN2 block, amendment 1, close-out) and CLEAN1_STATUS.md PLANGEN2 lines. experiment_history.json and OPEN_QUESTIONS are not updated; that is left to the integrating agent.
