# COOPKINDS_REPORT: which kinds of multi-CAV cooperation give real group synergy on wz3to2 (task N8, phase coopkinds, 2026-11-06)

**Plain summary (12 lines)**
1. Works best: **lane balancing** (2-3 lane-1 CAVs move to lane 2 upstream): admitted 3/24 moments (3 of 5 replicated candidates), all at demand 3,200 veh/h; super-additive in 2 of 3.
2. Admitted once each: **coordinated early merge** 1/24, **zipper** (late closing-lane merge + lane-1 slot partner) 1/20, **queue-front discharge** (short headway) 1/23.
3. Not admitted: **platoon/short headway** 0/24 and **two-row rolling VSL** 0/24 (all replicated screen candidates, 2 of 5 and 2 of 3, flipped sign under a dither; the rest were unreplicated); **random R proposals** 0/24 (R file empty).
4. Overall 6 admitted of 139 rule-selected situations; 5 of 24 world-windows hold >=1; lower bound, since 10 screen candidates were left unreplicated by the budget.
5. Admitted = in 3 replicates (1 undithered + 2 common-CRN dithers) one-sided 95% LB of T (vs native) and S (vs best single CAV) > 0, S >= 1 veh-s, 420-s later harm holds, group collision-free.
6. Size: S = +0.9k..+5.7k veh-s per 300 s (1.5-14% of base delay); T = +1.7k..+10.5k veh-s; 4 of 6 super-additive (e.g. 3 CAVs with solo gains ~0 give +3,522 together).
7. No family wins on average: mean T over all moments and mean S over screened candidates have 95% CIs including 0; synergy is state-dependent, so a policy must learn when to act.
8. V2V at t0: 4/6 admitted groups span <=100 m, 5/6 <=200 m, 6/6 <=300 m; 4/6 fit one ego's 4 V2V slots. During the move members spread to 131-561 m (none stays <=100 m).
9. Safety: 584 jobs, 0 collisions, 0 teleports, 0 steps with a < -4.5 m/s^2; min group accel -3.95 (base -3.62); hard-brake veh-s group-base CIs include 0 except fewer for early merge and platoon (HDVs).
10. Robustness caveat: 2 of 27 dithers were inert and 4 more moved J < 10 veh-s; the discharge case's 3 dithers never propagated, so its robustness is untested; early merge and lane balancing s989401 t800 are best supported.
11. Builder note: under speedMode 27 slowDown cannot exceed the CAV's IDM speed, so short-headway kinds (platoon, discharge) need a headway action (this probe used setTau); lane kinds need only changeLane.
12. Budget 199,920/200,000 sim-s on cpuV (<=12 workers, nice 19; gpuA unused); seeds 989400-989402; base native CAVs (not MAPPO); N3's closing-lane pair / single abreast wall not duplicated.

---

## 1. Question, authority, compliance
- **Question:** which KINDS of multi-CAV cooperation give real group synergy on the wz3to2 lane-drop map?
  - Synergy (project definition): S = min_i J(solo_i) − J(group), per seed.
  - Admission: the one-sided 95% lower bound of S must be > 0, from a 30-s circular-block paired bootstrap (B = 2,000, RNG 20261103).
- **Authority:** owner 14:53, phase **coopkinds**, capped at 200,000 sim-s.
  - Registered in REGISTRATION.md at 15:06, before any physics. Amendment 1 (15:10) set the replicate priority; amendment 2 (15:17) added one extra discharge replicate.
  - Own precharge ledger: cpuV `/home/ubuntu/coopkinds20261106/ledger/CHARGES.jsonl` (584 rows, 199,920 sim-s, phase "coopkinds").
  - Own PIDs only: `code/PIDS.txt` lists 584 entries. Nothing was signalled or killed.
- **Seeds:** 989400–989402, from N8's own sub-range 989400–989599. No 964000–964199, 962xxx or 977xxx seed was used.
- **Prohibited paths:** no sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path was opened. No hetdata_truth or vType/driver truth file was read.
  - HDV vTypes are drawn fresh into the SUMO route file, using a verbatim population.py (sha 6af5ee23).
  - HDVs are never commanded. Each branch asserts that every commanded vehicle is a CAV.
- **Map:** a copy of `assets/research/newmap_candidates/wz3to2` (net b6277898, MAP_SPEC 386400b7). The original merge map and the candidate files are untouched. No git reset, clean, stash drop or push.

## 2. Design
- **Cells:** demand {2800, 3200} veh/h (peak profile) × CAV share {0.4, 0.6} × 3 worlds = 12 native prefixes (0–900 s).
  - States were saved every 10 s in two windows, [450, 550] and [800, 900].
  - All natives had 0 collisions and 0 teleports. The approach to the drop slowed to 2.5–3.5 m/s in every world.
- **Situations:** selected by geometry only, never by outcome. Per window, each family takes the earliest save time at which it has a valid member set (the best-geometry set, with fixed B parameters).
  - R = one seeded random proposal per window: random family, time, valid member set and parameters.
- **Arms:** every arm, base included, branches from the same saved native state (matched random numbers).
  - base = all CAVs native;
  - group;
  - every member solo.
- **Families and rules:** CAV-only commands. Members use laneChangeMode 513, and everything is restored on release.

| Family | Members | Rule |
|---|---|---|
| platoon | consecutive CAV string, 2–4 CAVs, lane 1 or 2 | short headway: `setTau` 0.6 s (CACC-like) until X_B + 600 |
| early_merge | 2–3 lane-0 CAVs + 1–2 nearby lane-1 room-makers | mergers `changeLane(1)` from X_B − 1,200 m; room-makers tau 2.0 s around the merge zone |
| zipper | lane-0 CAV M + lane-1 CAV Y | M: no early merge (lcMode 512), merges from X_B − 100; Y tracks the slot x_M − 25 m |
| lane_balance | 2–3 lane-1 CAVs (spread ≤ 300 m, 600–1,800 m upstream) | `changeLane(2)`, stay in lane 2 until the drop |
| discharge | 2–3 most-downstream queued CAVs (v < 8 m/s, at or after the drop) | tau 0.6 s until X_B + 800 |
| vsl_rows | two 2-CAV abreast rows 300–1,000 m apart | all hold 18 m/s for 150 s, released at the drop |

- **Technical finding for builders (N2/N9).** Under speedMode 27 the commanded speed is capped by the CAV's own IDM speed. So `slowDown` can only make a CAV *slower* than IDM.
  - Short-headway modes (platoon, discharge) cannot be realised by speed commands alone.
  - This probe used a per-CAV headway parameter (`vehicle.setTau`).
  - A runtime that wants these kinds needs an equivalent "headway mode" action, or a different speedMode.
- **Funnel:**
  1. Screen r0: base + group, H = 300.
  2. Solos wherever the 95% lower bound of T is > 0.
  3. r0 candidates (95% lower bounds of T and S both > 0, S ≥ 1, group safe) get r1: a common dither of −1 m/s² for 2 s on the non-member CAV nearest the drop, in all arms, H = 420.
  4. r1 survivors get r2: the same dither on a seeded random non-member CAV within 1 km of the drop.
  5. Later harm: T > 0 and S > 0 over [t0, t0 + 420) in r1 and in r2.
  - Amendment 1: tier 0 is the 2 best candidates per family plus the 2 best R. Tier 1 was then run by the lower bound of S, while the budget allowed.

## 3. Results per family
Counts per situation: one situation = one (world, window). T and S are in veh-s over [t0, t0 + 300). The CIs are t-based 95%.

| Family | Situations | LB95(T) > 0 | r0 synergy candidates | Replicated | **Admitted** | Mean T, all [95% CI] | Mean S given LB95(T) > 0 [95% CI] |
|---|---|---|---|---|---|---|---|
| lane_balance | 24 | 10 | 7 | 5 | **3** | −50 [−1,571, +1,471] | +574 [−1,112, +2,260] |
| early_merge | 24 | 10 | 5 | 3 | **1** | +236 [−635, +1,107] | +576 [−200, +1,353] |
| zipper | 20 | 6 | 1 | 1 | **1** | −1,979 [−4,376, +417] | +541 [−741, +1,824] |
| discharge | 23 | 8 | 2 | 2 | **1** | −1,033 [−2,624, +558] | −522 [−3,074, +2,029] |
| platoon | 24 | 8 | 5 | 2 | 0 | −720 [−1,798, +359] | +113 [−225, +451] |
| vsl_rows | 24 | 10 | 3 | 2 | 0 | −1,039 [−3,033, +956] | +162 [−1,465, +1,788] |
| R (random) | 24 | 8 | 4 | 2 | 0 | −756 [−2,098, +585] | −274 [−1,174, +627] |

**Admitted per cell:**

| Cell | Admitted |
|---|---|
| D3200_p40 | 2 (both lane balancing) |
| D3200_p60 | 2 (lane balancing, discharge) |
| D2800_p40 | 1 (early merge) |
| D2800_p60 | 1 (zipper) |

**Failed replicates:** of the 11 failures, 9 were sign flips of S under a dither and 2 failed only the later-harm check (lane_balance s989402 t800 in D3200_p40; R vsl_rows s989401 t540 in D3200_p60). Examples of sign flips:
- platoon s989400 t800 (D2800_p40): r0 S = +514, r1 S = −1,173;
- vsl_rows s989401 t450 (D3200_p60): r0/r1 S = +4,646 / +4,057, r2 S = −1,419;
- discharge s989400 t450 (D3200_p60): r1 S = +5,224, r2 S = −6,494.

Near-chaotic sensitivity, as in NEWMAP_RESEARCH Section 5.4.

### 3.1 Admitted situations (S and T per replicate r0 / r1 / r2)

| Family | World, t0 | Members (lanes) | T | S | Solo gains | Super-additive? | V2V spread at t0 / during hold | 4 slots | Dithers that changed J (r1, r2) | Arrivals in 300 s (base / group) |
|---|---|---|---|---|---|---|---|---|---|---|
| lane_balance | D3200_p40 s989401 t800 | 478, 479, 480 (1, 1, 1) | +3,522 / +1,742 / +3,933 | +3,429 / +1,939 / +3,840 | +93 / −49 / −36 | **yes** (sum +8) | 15 / 131 m | yes | yes, yes (group 753) | 194 / 210 |
| lane_balance | D3200_p40 s989402 t450 | 283, 286, 291 (1, 1, 1) | +10,480 / +6,713 / +10,480 | +3,911 / +4,115 / +3,871 | +6,569 / +2,465 / +4,976 | no (sum 14,011) | 185 / 286 m | no | yes, no (< 2) | 174 / 245 |
| lane_balance | D3200_p60 s989401 t800 | 593, 594, 597 (1, 1, 1) | +1,915 / +1,915 / +1,822 | +884 / +884 / +1,116 | +1,031 / −685 / −1,876 | **yes** (sum −1,530) | 15 / 185 m | yes | no (inert), yes (group 257) | 193 / 190 |
| early_merge | D2800_p40 s989402 t800 | 543, 552, 555 (0) + 551, 554 (1) | +4,663 / +2,351 / +5,553 | +2,106 / +722 / +1,709 | +2,557 / −652 / 0 / −432 / −70 | **yes** (sum 1,403) | 282 / 529 m | no | yes, yes | 171 / 201 |
| zipper | D2800_p60 s989400 t880 | M 563 (0), Y 565 (1) | +6,582 / +6,582 / +8,455 | +2,767 / +2,816 / +7,756 | +3,386 / +3,814 | no (sum 7,200) | 8 / 420 m | yes | no (≈ 9), yes | 233 / 273 |
| discharge | D3200_p60 s989401 t800 | 342, 345, 347 (1, 1, 1) | +5,751 / +5,751 / +5,747 | +5,749 / +5,749 / +5,745 | −2,627 / −1,834 / +1 | **yes** (sum −4,459) | 27 / 561 m | yes | no, no (≤ 9); amendment 2 r3: no (≈ 1) | 193 / 207 |

Mechanisms observed in the tracks:
- **Lane balancing.** The group shifts 2–3 lane-1 vehicles out of the merge-receiving lane together. One CAV moving alone does nothing (solos ≈ 0) or hurts. Throughput rose by up to +71 vehicles in 300 s.
- **Zipper.** Y cannot keep its slot in congested lane 1, so the pair spreads to 420 m. The gain is mostly M using the closing lane to its end, plus Y's own effect. It is a sub-additive pair.
- **Discharge.** Each tight-headway CAV alone is harmful (−2.6k, −1.8k). All three together discharge +14 more vehicles in 300 s.
  - Its admission rests on an unperturbed trajectory: all 3 dithers, including the amendment-2 r3 dither on a moving lane-2 CAV, changed J by < 10 veh-s.
- **Later harm.** It held in every admitted case: T and S over 420 s were positive in r1 and r2, at +1.1k to +14.8k.

### 3.2 V2V range (all B situations, members' maximum pairwise distance)

| Family | At t0: ≤ 100 / ≤ 200 / ≤ 300 m | During the move: ≤ 100 / ≤ 200 / ≤ 300 m | 4-slot observable at t0 |
|---|---|---|---|
| platoon | 14 / 20 / 24 of 24 | 11 / 17 / 21 | 16 |
| early_merge | 0 / 9 / 17 of 24 | 0 / 1 / 8 | 1 |
| zipper | 20 / 20 / 20 of 20 | 7 / 10 / 13 | 20 |
| lane_balance | 11 / 23 / 24 of 24 | 1 / 7 / 11 | 14 |
| discharge | 22 / 23 / 23 of 23 | 3 / 6 / 9 | 21 |
| vsl_rows | 0 / 0 / 0 of 24 (rows 300–1,000 m apart by design) | 0 / 0 / 0 | 0 |

### 3.3 Safety (all 584 jobs)
- 0 failed jobs.
- 0 collisions and 0 teleports in every arm: natives, bases, groups, solos and replicates.
- 0 steps with a < −4.5 m/s².
- Lowest acceleration in a group arm: −3.95 m/s² (early merge). Base: −3.62.
- Hard-brake (a < −3 m/s²) veh-s, group minus base, mean [95% CI]:

| Family | All vehicles | HDVs |
|---|---|---|
| early_merge | −3.1 [−6.1, −0.1] | −1.6 [−3.7, +0.5] |
| platoon | −1.1 [−3.3, +1.2] | −1.7 [−2.8, −0.6] |
| lane_balance | +1.8 [−3.3, +6.9] | +1.0 [−2.5, +4.5] |
| zipper | +0.3 [−4.3, +4.9] | +0.4 [−2.8, +3.6] |
| discharge | −0.5 [−3.5, +2.5] | −0.8 [−3.0, +1.4] |
| vsl_rows | −2.2 [−7.8, +3.4] | −1.1 [−4.6, +2.3] |
| R | +2.1 [−2.5, +6.8] | +1.5 [−1.4, +4.3] |

## 4. Interpretation and use
- **New kinds beyond N3's walls and pairs:** lane balancing is the strongest state-dependent group kind on wz3to2 at high demand, and it is super-additive.
  - It is a natural V2V group move: the members start ≤ 185 m apart, and in 2 of 3 cases ≤ 15 m apart.
  - It needs only `changeLane` commands to CAVs, and no headway action.
- Early merge, zipper and discharge each gave one admitted, high-value case. Discharge's robustness is untested.
- Platoon and two-row VSL gave no replicated synergy under these fixed rules.
- **Caveats:**
  - The base is native CAVs, not MAPPO u0.
  - 3 worlds per cell; fixed, untuned rule parameters.
  - Situation choice = first valid moment per window.
  - Near-chaotic dynamics: a single dither flips signs.
  - 10 r0 candidates were left unreplicated by the budget (platoon 3, early merge 2, lane balancing 2, vsl_rows 1, R 2).
  - The MAIN 15:02 decision means these rows count only after re-verification in the coopmwm runtime.
  - The headway families need a runtime headway action (see Section 2).

## 5. Deliverables and evidence
- **`COOPKINDS_MOVES_B.jsonl`:** 18 rows = 6 admitted situations × replicate seeds 1–3 (sha 317d1e69).
  - Format: nm2 / coopmwm-mined-move-v1 style.
  - `move` holds the full role and parameter spec. `restart` gives the native state, t0 and inserted count. Also included: per-replicate S, T and lower bounds, later harm, dither effect, `v2v_range`, observability and safety.
  - `profile` is the realised per-member acceleration. There is no policy profile, because base member tracks were not recorded.
- **`COOPKINDS_MOVES_R.jsonl`:** 0 rows. No R proposal was admitted.
- **`COOPKINDS_NEGATIVES.jsonl`:** 157 rows (B 133, R 24), each with a stage and status:

  | Status | Rows |
  |---|---|
  | screen_fail_LB95T | 103 |
  | r0_no_synergy | 33 |
  | r0_only_budget | 10 |
  | r1_fail | 8 |
  | r2_fail | 3 |

- **Code:** `runs/active/C4-V7FT-20261017/coopkinds/code/`: ck_core, ck_plan, ck_run, ck_tier1, ck_r3, ck_analyze, and population (the verbatim copy).
- **Local results:** `coopkinds/results/` (ANALYSIS.json, CHARGES.jsonl, PIDS.txt, TIER1_LOG.json, and copies of the row files).
- **cpuV evidence:** `/home/ubuntu/coopkinds20261106/`, 41 MB: runs/ (every RESULT.json with per-second J), jobs/, ledger/, out/.
- **Records:** REGISTRATION.md (COOPKINDS block plus amendments 1–2) and CLEAN1_STATUS.md COOPKINDS lines.
  - `protocols/experiment_history.json` and OPEN_QUESTIONS were not updated. That is left to the integrating agent.
