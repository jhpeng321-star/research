# WALLMINE_REPORT: verified-wall mining, 2026-11-06 13:16-13:45 (WALLMINE-20261106)

## Summary
1. **Budget.** 147,550 of 150,000 sim-s used (gpuA 89,075, cpuV 58,475). Each host had its own precharged ledger; nothing was replayed. 0 collisions and 0 teleports in every screened, solo and native window.
2. **New B examples: 1.** The 3-CAV wall `mainline.164/167/168` at B 986061 t485 (op2 B situation, cell 10) was admitted on CRN seed 2:
   - synergy +95.5 veh-s (LB95 +33.2);
   - advantage over the policy +448.3 (LB95 +145.5);
   - n_reps 6;
   - **slot-observable**: at t0, egos 167 and 168 have both other members in their 3 nearest-CAV slots (checked against the decision sample's V2V block).
3. **Two more admitted B rows are near-duplicates, not new evidence.** They come from the coop-probe lead situation, 986061 t455 seed 1. The walls are `173/191/176/192` and `173/176/192`:
   - added member `mainline.173` is inert;
   - synergy +336.5 (LB95 +73.4) versus the probe's +336.7;
   - not observable.

   They carry `new_situation=false` and `near_duplicate_of` in `WALLMINE_MOVES.jsonl`.
4. **R: 0 admitted.**
   - 10 R situations were screened (33 walls): 2 walls beat the policy on seed 1, and both had synergy ≈ 0 (+1.3 LB95 −4.2; −0.0).
   - Another 4 R situations were refused by the budget cap at 0 charge.
   - `WALLMINE_MOVES_R.jsonl` is empty.
   - Matching R wall rows that already exist: the coop probe's R-native lead wall, seeds 1/5/6 (`COOP_PROBE_MINED_ALL_ARMS.jsonl`, bank `/home/ubuntu/coopprobe20261106/minebank/bank/replay_bank/R`). MAIN withdrew these from the clean3 file at 07:4x. They remain available if a registered update wants R matched.
5. **Main finding.** Walls that beat the policy are common: 15 of 94 had seed-1 LB95 > 0. Walls with group synergy are rare: 3 of 15, and all three are in world 986061 with a jammed ramp queue.
   - Many walls gain +150 to +770 veh-s, but **one member does all the work** (synergy −0.0 to +2.8).
   - Even the cooperative wall is a per-seed bet. 986061 t485 lost to the policy on seeds 3-6 (−96 to −324). On seed 1 it passed the 120 s window but failed the 600 s later-harm window.

## Setup
- **Code.** Branch `c4/wallmine` (worktree `E:/agent-work/c4-wallmine`, uncommitted). It adds `tools/mining/wallmine.py`, final deployed sha `e67400e0`; earlier deployments are listed in REGISTRATION.
  - The script runs on the unchanged coop-probe source: `probe.py` b604d4d9, `probe_native.py` 420b3b2e, G5 overlay 33efc058.
  - Base: frozen MAPPO u0 (`frozen_u0.pt` sha 5ea22964).
  - SUMO was the only ranker. No WM was used.
- **Hosts.**

  | host | arm | root | cap | resources |
  |---|---|---|---|---|
  | gpuA | B | `/home/ubuntu/wallmine20261106_gpuA` (= `/root/mining20261106/cpuv_home/...`, the parity-qualified clone) | 90,000 | ≤20 cores |
  | cpuV | R | `/home/ubuntu/wallmine20261106_cpuV` | 60,000 | ≤7 workers, nice 19 |

  Caps were rebalanced from 105k/45k at amendment 1. The total of 150,000 never changed.
- **Situations.** Registered TRAIN worlds only (986049–986071, cells 8/10/11). No variant or evaluation seeds were used.
  - **B.** 27 op2_collect01 B situations (free) plus 9 new B episodes:
    - 986066–986071 in cell 11;
    - 986061–986063 in cell 10;
    - new episodes use noise offset 8000 and record per-second CAV tracks.
  - **R.** Coop-probe R wall/hp situations (free) plus 1 new R episode (986061).
  - **Order.** Queues were ranked by observable features: a slot-observable wall exists, ramp slow count, number of profiled mainline CAVs, then t0.
- **Candidates.** Mainline walls of 3–4 CAVs, shape o0 h30 d1 (the verified shape). Slot-observable member sets come first, then the probe-slate sets.
  - Slot-observable means every member is inside one member's 4 trigger slots: ego plus its 3 nearest CAVs within 100 m (`track2_backend.observe`, partner_slots 4).
  - Observability is checked on frames t0..t0+10, using the per-second track where one exists and t0 rows otherwise.
  - Exact repeats of coop-probe or op2 screens were excluded.
- **Procedure.**
  1. Screen on seed 1: one `sv.verify` with policy + K walls (K ≤ 5, then ≤ 3).
  2. Seed-1 solos for walls with advantage LB95 > 0. Synergy = best solo − group on the identical CRN and window.
  3. Walls with seed-1 synergy LB95 > 0 go through the unchanged native path: `probe_native.run` = `sv.verify` + `verify_perseed.confirm_all8`, with extra seeds 4, i.e. seeds 1–6.
- **Admission, per (wall, seed).** All four must hold:
  - native admitted: LB95 vs policy > 0, own-seed safety, later harm passed;
  - that seed's synergy LB95 > 0;
  - 0 collisions or teleports in the group window;
  - never pooled across seeds.
- **Spend by stage.** Native spend is labelled `wallmine` in the ledgers.

  | stage | gpuA | cpuV |
  |---|---|---|
  | collect | 10,800 | 1,200 |
  | screen | 49,775 | 29,140 |
  | synergy | 13,880 | 8,545 |
  | native | 14,620 | 19,590 |

## Per world seed (tried / observable / beat policy / synergy measured / synergy > 0 / native (wall, seed) / admitted)
| host, arm | world (cell) | situations | walls tried | slot-observable | beat policy s1 (LB>0) | synergy measured | synergy LB>0 s1 | native (wall,seed) | admitted |
|---|---|---|---|---|---|---|---|---|---|
| gpuA B | 986061 (10) | 5 | 12 | 7 | 1 | 1 | 1 | 6 | **1** (seed 2) |
| gpuA B | 986060 (10) | 2 | 8 | 6 | 0 | 0 | 0 | 0 | 0 |
| gpuA B | 986062 (10) | 2 | 6 | 5 | 0 | 0 | 0 | 0 | 0 |
| gpuA B | 986063 (10) | 3 | 11 | 6 | 0 | 0 | 0 | 0 | 0 |
| gpuA B | 986066 (11) | 3 | 9 | 9 | 3 | 3 | 0 | 0 | 0 |
| gpuA B | 986067 (11) | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| gpuA B | 986069 (11) | 2 | 5 | 4 | 3 | 3 | 0 | 0 | 0 |
| gpuA B | 986070 (11) | 2 | 6 | 3 | 3 | 3 | 0 | 0 | 0 |
| cpuV B (relabelled R situation, deviation) | 986061 (10) | 1 | 3 | 1 | 3 | 3 | 2 | 12 | 2 (near-duplicates, seed 1) |
| cpuV R | 986060 (10) | 2 | 7 | 2 | 1 | 1 | 0 | 0 | 0 |
| cpuV R | 986061 (10) | 4 | 12 | 9 | 0 | 0 | 0 | 0 | 0 |
| cpuV R | 986062 (10) | 1 | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| cpuV R | 986068 (11) | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| cpuV R | 986069 (11) | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| cpuV R | 986070 (11) | 1 | 3 | 1 | 1 | 1 | 0 | 0 | 0 |
| **total** | | **31** | **94** | **61** | **15** | **15** | **3** | **18** | **3 (1 new)** |

## Synergy values (seed 1; veh-s; LB95 one-sided; obs = slot-observable)
| arm | situation | n | obs | group−policy (LB95) | synergy (LB95) |
|---|---|---|---|---|---|
| B | op2 986061 t485 164/167/168 | 3 | yes | +119.3 (+37.5) | **+110.7 (+36.8)** |
| B | op2 986070 t540 (3 walls) | 3–4 | 1 of 3 | +185.5 (+68.6) | 0.0 / −0.0 / −0.1 |
| B | op2 986069 t420 (3 walls) | 3–4 | yes | +209.4 / +768.5 / +768.5 | −0.0 (one member, 182, does it) |
| B | wm 986066 t410 (2 walls) | 3–4 | yes | +157.7 / +327.7 | −372.2 / −202.1 |
| B | op2 986066 t410 | 3 | yes | +281.6 (+121.6) | +2.8 (−3.1) |
| B* | 986061 t455 173/191/176(/192) | 3–4 | 1 of 3 | +153.7 / +490.1 / +490.1 | +0.1 / **+336.5 (+73.4)** / **+336.5 (+73.4)** |
| R | cp 986060 t510 219/218/220 | 3 | yes | +0.1 | −0.0 |
| R | cp hp 986070 t570 277/279/281 | 3 | yes | +339.2 (+123.9) | +1.3 (−4.2) |

`*` is the relabelled R situation verified as B (see Deviations).

Native per-seed results:
- **986061 t485 B (164/167/168).** Group−policy by seed: +119.3 (later harm failed) / **+448.3 admitted, synergy +95.5 (LB95 +33.2)** / −323.6 / −133.0 / −96.4 / −162.5.
- **986061 t455 B\*, both walls.** Seed 1 admitted (+490.1, synergy +336.5). Seeds 2–6: −171.5 / −289.8 / +103.9 (LB95 −143.1) / −388.7 / −394.9. These CRN replicates differ from the probe's seeds 2–6, which use different verification-id salts.

## Outputs (this directory)
- **`WALLMINE_MOVES.jsonl`** (3 B rows, sha bb94bab7). Clean3 mined-row format (`coopmwm-mined-move-v1`) plus seed and situation tags, `synergy_veh_s`, `synergy_lower95`, `n_reps`, `observable`/`observability`, `new_situation`, `near_duplicate_of` and `bank_host`.
  - Checks run on cpuV with the clean3 source 3e5bc9f9:
    - `group_lane_bank.binding()` passes on all 3 rows;
    - `bank_breadth.mined_move()` maps each row to an executable wall move (o0 h30 d1, 3–4 actors).
  - `admit_mined` was not run: it needs the clean3 config. The feed operator should run it.
  - **The new row's bank is on gpuA** at `run_root` `/home/ubuntu/wallmine20261106_gpuA/minebank/bank` (234 MB). Before any registered feed, copy `replay_bank/B` to the same path on cpuV.
  - The two near-duplicate rows' bank is on cpuV.
- **`WALLMINE_MOVES_R.jsonl`:** 0 rows.
- **`WALLMINE_NEGATIVES.jsonl`** (106 rows, sha f56cb9be). Every losing (wall, seed), with stage, advantage, synergy, safety, observability and reason:
  - B: 50 screen, 10 synergy, 15 native;
  - R: 29 screen, 2 synergy.
- **Nothing was written to cpuV `/home/ubuntu/mined`.**

## Deviations
- **cpuV ran one B-labelled situation instead of R.** The cpuV queue de-duplicated by state id and picked the coop probe's `ctxB` copy of R 986061 t455, i.e. the R-collected snapshot already relabelled B. So cpuV ran one B-labelled screen, synergy and native confirmation: 25,035 sim-s: screen 2,420, synergy 3,025, native 19,590. This is the probe's own relabel procedure, but it was not the registered R arm. Disclosed in REGISTRATION amendment 1, after which queues were filtered by arm.
- **The registration header time (13:1x) was an estimate.** It is corrected in amendment 1.
- **Amendments 1–3** rebalanced caps, set K ≤ 3, extra seeds 2 then 1, added new cell-10 episodes and set the final reserve to 1,500. All were registered before use, and no rule or statistic changed.

## Evidence
- **Host roots:**
  - gpuA `/home/ubuntu/wallmine20261106_gpuA/` (1.3 GB): `CHARGES.jsonl`, `results/`, `screen/`, `synergy/`, `minebank/`, `collect/`, `logs/`, `PIDS.txt`.
  - cpuV `/home/ubuntu/wallmine20261106_cpuV/` (808 MB): same layout.
- **Local copies of the small files:** `runs/active/C4-V7FT-20261017/wallmine/{gpuA,cpuV}/` (ledgers, results, outputs, queues, logs).
- **All own processes have exited on both hosts.**

## Recommendation
Under the current vocabulary, more seed-1 screening has a low yield: 3 synergy hits in 94 walls, and only world 986061 produced any. Three paths look more promising:
- Target jammed-ramp situations in cell 10 (e.g. 986061) with many more CAV-track collections.
- Or let walls start before the queue forms: onset > 0, or upstream members.
- For R parity, either feed the existing R-native lead-wall rows at a registered update or accept B-only mined rows.
