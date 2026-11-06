# UPPER_BOUND_REPORT: centralized full-information ceiling for group cooperation (2026-11-06)

## Plain summary
1. Question: if a centralized planner sees everything and searches walls of 2–6 mainline CAVs in SUMO, how much GROUP cooperation (synergy = J(best single member alone) − J(group)) is there to win?
2. Answer: essentially none. In 12 registered TRAIN situations (one per demand × CAV-share cell), no wall had material synergy on any seed.
3. Strict per-seed admission (one-sided LB95 > 0) passes in 1 of 12 situations (986006), and there synergy is only +0.03 and +1.7 veh-s, about 0.2% of that move's gain.
4. Large TOTAL gains do exist: 6 of 12 situations have a wall that beats the frozen MAPPO u0 base on the search seed (+206 to +975 veh-s over 180 s, LB95 > 0).
5. In all 6, one member doing the same move alone matches or beats the whole wall. In 986066 the best single CAV gives +1,024 against the wall's +457. Six other admitted groups scanned (A1) also had synergy ≤ 0.
6. Ceiling on the search seed (clairvoyant, walls, gain floored at 0): mean +264, median +110 veh-s per situation. Including the measured single-CAV moves: mean +321, median +110.
7. Fair estimate on held-out CRN seeds 2–4: the planner acts in 6/12 situations. Gain per situation: mean +201, median +42 veh-s. 16 of 18 held-out evaluations are positive and 13 of 18 are admitted against the base.
8. Later harm (A3, 660-s window, 3 of the 6 acting situations): 986036 keeps its gain (+662, LB +261); 986006 is uncertain (+775, LB −314); 986048 reverses to −2,649 (LB −4,185). The 180-s ceiling therefore overstates for some moves.
9. Observability: 0 of the 6 beneficial walls fit inside one member's 4 or 6 slots (members spread 118–389 m). Because one CAV carries the whole gain, no partner observation is needed to capture it.
10. Safety: 0 collisions and 0 teleports in all 256 graded branches. For the acting moves, TTC<3 s falls in 21 of 24 seed evaluations and hard brakes fall in 11 (rise in 6).
11. Implication for the paper: the decentralized 4-slot policies should be judged against a ceiling of about 0 for group synergy. Total-benefit ceilings (held-out mean about +200 veh-s per merge-pressure moment, median +42) are single-CAV in nature.
12. Budget: 148,995 of 150,000 sim-s (own ledger, phase `upper_bound`), cpuV only (≤ 12 cores, nice 19), gpuA unused.

## Setup (what was run)
- **Phase / ledger**: `upper_bound`, cpuV `/home/ubuntu/upperbound20261106/CHARGES.jsonl` (70 rows, all precharged, never replayed; hard cap 150,000 checked under a file lock).
- **Code**: branch `c4/upper-bound` (worktree `E:/agent-work/c4-upper-bound`), commits 7a6c99d8 (main study) and f4954bdb (amendments).
  - Files: `tools/mining/upperbound.py` (sha 913cef80), `upperbound_amend.py` (A1/A2, b9daa28e), `upperbound_amend3.py` (A3, f31ea5cd), plus the no-physics orchestration test `upperbound_fake_test.py`.
  - The code is deployed on top of the unchanged coop_probe source (c4/coop-probe 5e6eb9e4 as deployed in `/home/ubuntu/coopprobe20261106/source`, same G5 env overlay sha 33efc058 and environment).
- **Reused machinery, unchanged**:
  - `sv.capture`, `sv.build_branches`, `sv.run_branch` and `SubprocessBranchRunner` (prefix replay with bitwise fingerprint check, CRN restore);
  - `mine_verify._pol` (seed-1 policy reference);
  - `rb.solo_candidate` and `rb.replicate` (seed salts `upperbound#<sid>#r<k>`, evaluation ranges excluded);
  - `bank_statistics._stats` (the probe's per-seed one-sided 95% paired bootstrap, B = 2000).
- **What is new**: only the orchestration. Every branch of a situation shares one window [t0, t0+180] (D_max 60 s + tail 120 s, as the probe's screens), so each seed's policy branch is run once and reused.
- **Registration** (`upper_bound/REGISTRATION.json`), written before any physics:
  - **Worlds**: the repeat-0 TRAIN world of each of the 12 cells (986000 + 6·cell). Fallback and fill rules were registered but not needed: all 12 worlds qualified.
  - **Collection**: arm B, frozen u0 (`frozen_u0.pt` sha 5ea22964), noise seed 20261106 + world + 9000, stopped at 490 s.
  - **Moment (config/time rule, state only)**: t0 is the 5-s tick in [300, 420] that maximizes (min(#eligible, 6), #ramp vehicles in the merge zone s∈[−100, 75.44], Σ1/(1+v_ramp), #mainline vehicles in the zone). A tick qualifies only with ≥ 1 ramp vehicle in the zone and ≥ 2 eligible CAVs.
  - **Radius**: eligible = mainline-lane (lane 0/1) CAVs at s∈[−400, 75.44] m that are policy-controlled at t0, regardless of sensing slots. Three acting walls include ramp-origin CAVs already on lane 1 (ramp.22/26/27).
- **Search (seed 1), 12 walls per situation**, 144 in total:
  - **Coarse screen (8)**: the probe lead law (onset 0, hold 30 s, −1 m/s² offset on the policy reference) on member sets front-4, front-2, front-6 and upstream-4; then front-4 with depth 0.5, h20/d2.0, onset 10 and gradual release 20 s (linear taper); then fill laws.
  - **Refinement (4)**: 2 neighbours of each of the top-2 (depth step, member-count step, hold +15, release toggle, onset +5).
  - **Best move**: the lowest J among walls with no collision or teleport increase.
- **Evaluation**:
  - The planner ACTS only if its seed-1 best move is admitted against the base (adv LB95 > 0). Otherwise it chooses "no move" (gain 0).
  - Held-out seeds 2 and 3: policy + move.
  - Member solos on any seed where the move is admitted against the base. The probe and native rule: synergy is measured only where the group beats the base.
- **Amendments** (written before their own execution, after seeing seed 1–3 results; disclosed in `AMENDMENT_A1A2.json` and `AMENDMENT_A3.json`):
  - **A1**: synergy of other admitted walls with a distinct member set (5940 sim-s).
  - **A2**: held-out seed 4, policy + move, no solos (6600).
  - **A3**: 660-s later-harm window on seed 1 for the 3 largest seed-1 gains that fit under the cap (6120). A3 re-derived the 180-s increments from the extended tapes, and they equal the original J values exactly (determinism check passed for all 6 branches).
  - The registered Phase 6 (seeds 4/5 with solos) was skipped by its own budget rule.

## Per situation and per seed
Each seed cell shows gain vs base (LB95) / synergy (LB95), in veh-s on [t0, t0+180]; positive = better.

| world (cell: CAV share, vph) | t0 | elig. | chosen wall | J_base s1 | seed 1 (search) | seed 2 | seed 3 | seed 4 (A2, gain only) | best single CAV s1 | slots 4 / 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 986000 (c0: 0.05, 1900) | 340 | 2 | 2 CAV o0 h30 d1.5 | 3368 | +206 (+92) / 0.0 (0.0) | +246 (+129) / 0.0 | +250 (+130) / 0.0 | +217 (+107) | +206 | N / N |
| 986006 (c1: 0.05, 2400) | 375 | 2 | 2 CAV o0 h30 d1 | 13691 | +975 (+443) / **+0.03 (+0.007)** | +981 (+440) / **+1.7 (+0.10)** | +1196 (+548) / −0.0 | +1406 (+649) | +975 | N / N |
| 986012 (c2: 0.05, 2700) | 405 | 3 | 3 CAV o10 h30 d1 | 22003 | +267 (+35) / −0.03 (−0.2) | +873 (+393) / −20.6 (−68.9) | +332 (+47) / −0.1 | +73 (+25) | +267 | N / N |
| 986018 (c3: 0.1, 1900) | 420 | 3 | 3 CAV o0 h20 d2 | 5572 | +15 (−3) / n/m | no move | no move | no move | n/m | N / N |
| 986024 (c4: 0.1, 2400) | 330 | 5 | 5 CAV o0 h30 d0.5 | 13194 | +0.2 (−0.2) / n/m | no move | no move | no move | n/m | N / N |
| 986030 (c5: 0.1, 2700) | 400 | 3 | 3 CAV o0 h30 d0.5 | 16696 | 0.0 / n/m | no move | no move | no move | n/m | Y / Y |
| 986036 (c6: 0.2, 1900) | 315 | 6 | 4 CAV o0 h30 d1 | 3680 | +472 (+213) / −126 (−226) | +84 (−12) / n/m | +693 (+306) / +1.1 (−62) | −184 (−391) | **+598** | N / N |
| 986042 (c7: 0.2, 2400) | 320 | 6 | 6 CAV o0 h30 d1 | 11329 | −83 (−152) / n/m | no move | no move | no move | n/m | N / N |
| 986048 (c8: 0.2, 2700) | 390 | 4 | 4 CAV o0 h30 d0.5 | 25081 | +771 (+365) / 0.0 (0.0) | +486 (+200) / −262 (−458) | +164 (+49) / 0.0 | −397 (−659) | +771 | N / N |
| 986054 (c9: 0.4, 1900) | 410 | 8 | 5 CAV o10 h30 d1 | 8272 | 0.0 / n/m | no move | no move | no move | n/m | N / N |
| 986060 (c10: 0.4, 2400) | 345 | 5 | 2 CAV o0 h30 d0.67 | 11287 | −0.0 / n/m | no move | no move | no move | n/m | Y / Y |
| 986066 (c11: 0.4, 2700) | 395 | 7 | 5 CAV o0 h20 d2 | 24748 | +457 (+74) / −566 (−924) | +148 (−31) / n/m | +173 (−186) / n/m | +492 (+170) | **+1024** | N / N |

Notes on the table:
- **n/m**: not measured, because the group was not admitted against the base on that seed (no solos run).
- **no move**: the seed-1 best was not admitted, so the planner does nothing (gain 0, no branches).
- **Slot columns**: observable from some member's 4 or 6 slots, using the clean3 rule (self + K−1 nearest CAVs within 100 m) at t0. For the beneficial walls, the most members any one member covers is 1–3 at 4 slots and 1–4 at 6 slots.
- **A1** (other admitted groups, seed 1): synergy −114, −145, −385, −325 and 0.0 (986036; one 4-CAV group observable at 6 slots, +150 gain), and −566 (986066 4-CAV subset). No positive synergy.
- **Later harm** (A3, seed 1, [t0, t0+660]):
  - 986006: +775 (LB −314), not admitted.
  - 986048: −2,649 (LB −4,185). The 180-s gain is deferred harm, with TTC<3 s up by +91.5 veh-s.
  - 986036: +662 (LB +261), admitted.
  - 986066, 986012 and 986000 were not checked (cap).
- **Mechanism hint (state at t0)**:
  - In 5 of the 6 beneficial situations, the member that carries the gain is a single upstream CAV 216–397 m before the merge. Four of them are on lane 0, the lane not adjacent to the ramp, at 27–30 m/s; one is on lane 1.
  - In 986066 it is a lane-1 CAV in the merge queue (s = −9 m, 0.8 m/s).
  - The move eases it by 1–2 m/s² for 20–30 s, i.e. single-CAV upstream metering. The other wall members add nothing (synergy exactly 0.0) or hurt.

## Aggregate
| measure | value |
|---|---|
| situations / with a qualifying moment | 12 / 12 (one per cell, no fallback) |
| situations with admissible synergy (LB95 > 0 on some seed) | **1/12 strict** (986006: +0.03 and +1.7 veh-s); **0/12 material** (≥ 10 veh-s or ≥ 1% of gain) |
| synergy among all measured multi-CAV groups (seed 1: 6 best + 6 A1) | max +0.03 veh-s; 7 of 12 ≤ −100 veh-s |
| search-seed (clairvoyant) ceiling, walls, floored at 0 | mean **+263.6**, median **+110.5** veh-s / situation (180-s window) |
| same, including measured single-CAV moves | mean +321.3, median +110.5 |
| situations where the planner acts (seed-1 best admitted) | 6/12 (all three 5%-CAV cells, 2 of 3 at 20%, 1 of 3 at 40%; none at 10%) |
| held-out (seeds 2–4) planner gain per situation, 0 where no move | mean **+200.9**, median **+42.1** veh-s |
| held-out evaluations of acting moves | 18: mean +402, median +248; 16 positive, 13 admitted vs base |
| later-harm check (A3, 3 situations) | 1 admitted, 1 uncertain, 1 reversed (−2,649) |
| observability of the chosen wall, all 12 | 4 slots 2/12, 6 slots 2/12 (both in no-gain situations 986030, 986060) |
| observability of the 6 beneficial walls | 4 slots **0/6**, 6 slots **0/6** |
| safety, all 256 graded branches | 0 collisions, 0 teleports (absolute and vs base). Acting moves over 24 seed evaluations: hard-brake Δ −4 … +4.5 (11 down, 6 up); TTC<3 s Δ −92 … +47.5 veh-s (21 down, 3 up), 180-s window |

## Budget (sim-s, phase `upper_bound`)
| part | sim-s |
|---|---|
| collection (12 × 490) | 5,880 |
| screen (12 × 9 branches incl. policy) | 59,445 |
| refine (12 × 4) | 26,420 |
| seed-1 solos (6 situations) | 11,040 |
| held-out seeds 2, 3: policy + move | 13,200 |
| held-out solos | 14,350 |
| A1 extra solos | 5,940 |
| A2 seed 4 | 6,600 |
| A3 later-harm (660 s) | 6,120 |
| **total** | **148,995 / 150,000** (1,005 unspent) |

256 branches + 12 collections, all graded (no PrefixMismatch, no failure). Wall time ≈ 25 min on cpuV (11 branch workers + orchestrator, nice 19). gpuA was not used. All own PIDs (`PIDS.jsonl`) have exited.

## Caveats
- **The search is coarse, not exhaustive.** It covers 12 open-loop walls per situation (−offset on the seed-1 policy reference, then MAPPO resumes), so the ceilings are lower bounds on the true centralized optimum.
- **Synergy coverage is partial.** The best move was chosen by total benefit. Synergy is known for 12 groups on seed 1 and 7 group-seeds held out, not for every searched wall.
- **Seed 1 is in-sample.** Moves were selected on it, so it is clairvoyant. Seeds 2–4 share the identical t0 state and differ only in post-t0 policy noise; they are the fair transfer estimate.
  - As in the probe, on held-out seeds members follow seed-1 base commands plus offsets. Group and solos share this, so within-seed synergy stays paired.
- **The window is short.** J is team delay over 180 s. A3 shows that later harm can erase or reverse the gain (986048), and 3 of the 6 acting moves have no later-harm check.
- **Sampling is narrow.** There is one situation per cell (repeat-0 world) and t0 lies in [300, 420] only. This sample is not the probe's lead situation (R 986061 t455, synergy +337 on 1 of 6 seeds), so the two results do not contradict each other. Here, a registered config/time sample found no material synergy.
- **Observability is evaluated at t0 only**, using the clean3 slot rule.

## Evidence paths
- **cpuV `/home/ubuntu/upperbound20261106/`**:
  - registration and amendments: `REGISTRATION.json`, `AMENDMENT_A1A2.json`, `AMENDMENT_A3.json`;
  - ledger: `CHARGES.jsonl`;
  - logs: `RUN_LOG.jsonl`, `PIDS.jsonl`, `logs/`;
  - collections: `collect/` (snapshots + contexts);
  - branch tapes: `verify/`, `verify_ext/` (≈ 0.9 GB);
  - receipts and claims: `receipts/`, `claims/`;
  - results: `results/` (`branches.jsonl`, `plans.jsonl`, `perseed*.jsonl`, `amend_a{1,2,3}.jsonl`, `SUMMARY.json`, `EXTRA_STATS.json`).
- **Local copy** (no tapes or snapshots): `runs/active/C4-V7FT-20261017/upper_bound/`.
- **Status lines**: `CLEAN1_STATUS.md`, prefix `UPPERBOUND`.
