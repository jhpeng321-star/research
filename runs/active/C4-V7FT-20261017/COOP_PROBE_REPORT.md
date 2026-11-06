# COOP_PROBE_REPORT: cooperation feasibility probe, 2026-11-06 (owner approval 05:10)

## Summary
1. Question: does real GROUP cooperation exist? The test is synergy = J(best single-CAV solo) − J(group), computed per CRN seed on the common window. Admission requires the one-sided 95% lower bound (LB95) > 0.
2. **Wall setting: YES.** The lead 4-CAV mainline wall (R 986061 t455) has seed-1 synergy +336.7 veh-s (LB95 +73.6). Native all8 admits it on 3 of 6 seeds. Its 3-CAV sub-wall is admitted on 1 of 6 seeds. On the 9 other TRAIN seeds/moments screened, no wall was admitted.
3. **High-pressure setting: YES, weakly.** Cell 11 (2700 vph × 0.4 CAV) has one wall situation, R 986067 t340 (3 CAV). It has seed-1 synergy +271.4 (LB95 +104.0) and is admitted natively on 1 of 6 seeds. The other 9 screened moments were not admitted.
4. **Less-cooperative HDV variant: NO.** Eleven screens were run: 3–4-CAV walls, G5 open_gap_merge and ramp_assist. Every synergy was 0.00 or negative. No admitted example.
5. Wherever cooperation exists, it is a per-seed bet, not a robust effect. The same wall loses heavily to policy on most other seeds:
   - lead wall: positive on 3/6 seeds;
   - 3-CAV sub-wall: positive on 1/6;
   - hp wall: positive on 1/6, with group-vs-policy as low as −505 veh-s on losing seeds.
6. Clean3 deliverable: **5 B rows** (arm B, frozen MAPPO u0 base) in `MINED_MOVES.jsonl` (sha 7a225dbd).
   - Locations: cpuV `/home/ubuntu/mined/` and this run directory.
   - Coverage: 2 situations, both 3–4-CAV walls.
7. The WM did not find the winners. It predicted negative synergy for both SUMO-positive walls: −21.6 for the lead and −8.4 for hp 986067 t340. Those walls were verified only because SUMO checks the WM's top pick in every context.
8. Budget: 240,078 of 250,000 sim-s, charged to the cpuV coop_probe ledger. No collisions or teleports in any graded group window.

## Setup (what was run)
- **Phase and ledger.** Phase `coop_probe`. Ledger cpuV `/home/ubuntu/coopprobe20261106/CHARGES.jsonl`: 107 rows, precharged, never replayed.
- **Code.** Branch `c4/coop-probe`, worktree `E:/agent-work/c4-coop-probe`, files:
  - `tools/mining/probe.py`
  - `probe_plan.py`
  - `probe_wmq.py`
  - `probe_wm.py`
  - `probe_native.py`
  - `probe_summary.py`
  - `probe_site.py`

  These run on top of the mining-op2 source, which includes the G5 env overlay sha 33efc058 and the G5 solo patch.
- **Hosts.**
  - SUMO ran on cpuV (SUMO 1.27.1, nice 19, up to ~40 concurrent branches).
  - The WM proposer ran on the gpuA GPU. It used FULL-v4 with the mining lane-capable `evaluate_lane_grid`:
    - K8/H120 for walls;
    - K4/H120 for hp/lowcoop, a qualified startup shape;
    - 11 probe workers, all stopped at the end.
  - gpuA output is 38 MB in `/root/coopprobe`.
- **Worlds.** Registered TRAIN only, 986000–986071. No 964000–964199, 962xxx or 977xxx seeds. No sealed or holdout paths were opened, and no truth or driver-parameter file was read.
- **Collection.** 17 plain frozen-u0 episodes, arm R, 1200 s each. Up to 3 decision snapshots per episode, selected by observable features only: the number of mainline CAVs in s∈[−400,100] and merge pressure. Each snapshot stores per-second policy references for every CAV in the band.
- **Procedure.**
  1. The WM ranks each context's slate by predicted synergy (paired per draw) and by saving.
  2. SUMO screens the WM top pick on seed 1: policy branch, group and every member solo in one verification (shared CRN and window, tail 120 s).
  3. Positives get replicate seeds through the unchanged native path: `sv.verify` plus `verify_perseed.confirm_all8` with extra_seeds. This covers seed 2 and seeds 3+ with registered salts, per-seed later-harm windows (tail 600 s), per-seed safety and per-seed batched solos.
- **Statistic.** Per seed, a 30-s circular-block paired bootstrap (B=2000) of best solo minus group, per second over the common window (`bank_statistics._stats`, same as mining `per_seed_v1`). Results are never pooled.
- **Owner-approved variant (lowcoop_v1, probe only).** Every textbook HDV vType gets `lcCooperative=0.0`, `lcCooperativeSpeed=0.0` and `lcSpeedGain=0.2`, set in the route-file vType attributes through a hook on `population.vtype_attrs`.
  - The CAV vType (`cav_typical`) is unchanged.
  - 799–879 HDV vTypes were patched per episode (count recorded in each `COLLECTION_RECEIPT.json`).
  - Every variant branch replays with the same variant; prefix fingerprints matched.
- **Limitation.** Plans are absolute profiles built on the seed-1 policy reference, as in mining. On replicate seeds the members therefore follow seed-1 base commands plus offsets. Group and solos share this, so within-seed synergy remains paired.

## Setting 1: wall lead and wall search (spend 138,805 sim-s)
Lead situation: R 986061 t455, ramp queue jammed. Wall = mainline.191/176/192/193 (lane 0/1 behind the ramp head), o0 h30 d1 (all members −1 m/s² for 30 s).

| move | seed set | per-seed group−policy (veh-s) | per-seed synergy (LB95) | admitted |
|---|---|---|---|---|
| 4-CAV wall, probe screen seeds | 1 / 2 / 3 / 4 / 5 / 6 | +490.3 / −171.8 / +389.4 / −253.7 / −164.0 / −204.9 | **+336.7 (+73.6)** / −274.2 / **+255.1 (+27.7)** / −260.5 / −164.0 / −204.9 | 2/6 |
| 4-CAV wall, extended tail 600 s | 1 / 3 | +11,800.6 / +7,602.9 | **+7,619.9 (+6,184.0)** / **+6,989.6 (+5,484.9)** | 2/2 |
| 4-CAV wall, native all8 (R and B identical) | 1..6 | +490.3 / −261.6 / −11.3 / +90.8 / +344.1 / +487.4 | **+336.7 (+73.6)** / – / – / – / **+344.1 (+42.4)** / **+487.4 (+100.8)** | 3/6 (B rows) |
| 3-CAV sub-wall 191/176/192, native B | 1..6 | +490.3 / −236.6 / −242.1 / −505.3 / −368.5 / −242.8 | **+336.8 (+73.6)** / – … | 1/6 (B row) |
| 4-CAV wall o0 h20 d0.6 | 1 / 2 | +51.9 / −3.9 | −0.002 / −3.9 | 0/2 |
| 3-CAV wall o0 h20 d0.6 | 1 | +51.9 | −0.0 | 0 |
| ramp_assist 176/192 + ramp.47 h20 d0.6 | 1 | +52.0 | +0.107 (+0.008) | technically yes; negligible |

A dash means the seed was not admitted against policy, so no solos were run.

Other moments: R 986061 t350/t475, plus new collections on 986060/062/063/064/065. One WM-top 3–4-CAV wall was screened per context, all on seed 1:

| seed | moment | n | group−policy | synergy (LB95) |
|---|---|---|---|---|
| 986060 | t510 | 4 | +0.1 | −0.05 (−0.16) |
| 986062 | t580 | 3 | +69.2 | +15.6 (−1.3) |
| 986063 | t300 / t330 / t520 | 4/3/3 | −0.0 / +330.0 (one member does it) / −0.0 | 0.0 / −0.0 / −0.03 |
| 986064 | t470 / t575 | 3/3 | 0.0 / 0.0 | 0.0 / 0.0 |
| 986065 | t320 / t375 | 3/3 | −199.8 / −31.7 | −199.8 / −51.9 |

Per family and seed (tried / beat-policy / synergy-measured / synergy-admitted; screen and confirmation rows):
- wall 986061: 10/5/10/3 screen, plus native 12 seed-examples with 4 admitted.
- wall 986060: 1/1/1/0
- wall 986062: 1/1/1/0
- wall 986063: 3/1/3/0
- wall 986064: 2/2/2/0
- wall 986065: 2/0/2/0
- ramp_assist 986061: 1/1/1/1 (negligible +0.1 veh-s)

The R 986061 t350 and t475 contexts were never screened: their WM ranks did not land before the budget and time ran out.

**Verdict, setting 1: real group cooperation found: YES.** It is found in one situation (4 admitted (wall, seed) examples across the 4-CAV and 3-CAV variants). It is seed-dependent, and the same wall is strongly negative on most other seeds.

## Setting 2: high-pressure worlds (spend 60,817 sim-s)
Worlds were chosen by registered config only: highest demand (2700 vph) and highest CAV share (0.4) = cell 11, seeds 986066–986071, 6 collection episodes. Screens covered 986066–986068; WM ranks for 986069–071 did not arrive in time.

| seed | moment | family, n | group−policy (LB95) | synergy (LB95) |
|---|---|---|---|---|
| 986066 | t340 | yield_lane (G5), 2 | −0.0 | −0.0 (−0.01) |
| 986066 | t340 / t400 / t430 | wall 3 / 3 / 4 | +0.1 / −1.6 / +98.0 (+22.7) | 0.0 / −1.65 / −0.0 |
| 986067 | **t340** | wall 3 (mainline.144/112/122) | **+369.3 (+123.4)** | **+271.4 (+104.0)** |
| 986067 | t375 / t395 | wall 3 / 3 | −45.3 / +375.2 (+142.2) | −193.2 / −44.7 (best solo +419.9) |
| 986068 | t315 / t455 / t475 | wall 3 / 3 / 3 | 0.0 / +87.8 (−65.4) / +80.8 (+7.7) | 0.0 / +15.2 (+4.06) / −0.0 |

Native B confirmations:
- **986067 t340, 6 seeds:**
  - seed 1: admitted and cooperative (+271.4, LB95 +104.0); this is the B row.
  - seeds 3/4: beat policy (LB95 +6.8/+32.7), but synergy +0.4 (−0.0) and +79.5 (−10.5), so not admitted.
  - seeds 2/5/6: not admitted (−495.8 on seed 5).
- **986068 t455, 2 seeds:**
  - seed 1: not admitted against policy (LB95 −65.4).
  - seed 2: beat policy (+24.9), synergy +0.1 (−0.03).
  - Result: 0 rows.

Per family and seed (tried / beat / measured / admitted):
- wall 986066: 3/2/3/0
- wall 986067: 3/2/3/1
- wall 986068: 3/2/3/1 (screen synergy LB > 0 but not admitted against policy natively)
- yield_lane 986066: 1/0/1/0

**Verdict, setting 2: real group cooperation found: YES, weakly.** There is one situation with 1 admitted seed out of 6. Many moments have big group gains with 0 synergy, meaning one member does all the work.

## Setting 3: less-cooperative HDVs (variant lowcoop_v1; spend 40,456 sim-s)
Same cell 11 worlds (986066–071), 6 variant episodes. Every screen ran policy (baseline), group and each solo in the variant.

| seed | moment | family, n | group−policy | synergy (LB95) |
|---|---|---|---|---|
| 986066 | t400 | wall 3 | 0.0 | 0.0 (−0.0) |
| 986066 | t400 | open_gap_merge (G5) ×2 pairs | 0.0 / 0.0 | 0.0 (0.0) / 0.0 (0.0) |
| 986066 | t440 / t485 | wall 3 / 3 | −0.0 / −0.0 | +0.01 (−0.21) / −0.0 |
| 986067 | t375 | wall 3 | +0.5 | −0.03 (−0.04) |
| 986068 | t310 | ramp_assist 2 | −0.1 | −0.11 (−0.22) |
| 986068 | t425 / t455 | wall 3 / 3 | −0.0 / −103.3 | −0.0 / −157.0 (−249.2) |
| 986069 | t325 / t360 | wall 4 / 3 | −3.5 / −17.3 | −3.5 (−11.1) / −17.3 (−27.1) |

Per family and seed (tried / beat / measured / admitted):
- wall 986066: 3/1/3/0
- wall 986067: 1/1/1/0
- wall 986068: 2/0/2/0
- wall 986069: 2/0/2/0
- open_gap_merge 986066: 2/0/2/0
- ramp_assist 986068: 1/0/1/0

G5 lane moves needed `C4NM_V2V=1` (the single-writer V2V lane runtime) in the branch runner, as mining had. The first G5 attempts failed for that reason and were re-run under new tags; the failed charges were kept.

The open_gap_merge synergy is exactly 0.0 in the variant too. The ramp lane change still adds nothing over the main-only effect: the mining mechanism (gate passes only at SUMO's native merge) persists.

**Verdict, setting 3: real group cooperation found: NO.** There are 0 admitted examples. Every measured synergy is ≤ +0.01 veh-s.

## Clean3 deliverable (MAIN 07:2x / 07:4x)
`MINED_MOVES.jsonl` holds 5 rows, all `"arm": "B"`, sha 7a225dbd. It is on cpuV at `/home/ubuntu/mined/` and in this directory. The previous empty mining file is kept as `.mining_op2_empty_20261106`.

The rows:
- 4-CAV lead wall, seeds 1/5/6;
- 3-CAV lead wall, seed 1;
- hp 986067 t340 3-CAV wall, seed 1.

Each row carries:
- the native binding (`run_root` `/home/ubuntu/coopprobe20261106/minebank/bank`, `verification_id`, `plan_sha256`);
- seed, `situation_id`, world, t0, members;
- `profile`, `policy_profile` and `perturbation_profile`;
- `synergy_veh_s` and `synergy_lower95` (keyed by seed), advantage, and `n_reps`.

How the B verification was done: the situations were collected under the R-labelled noise stream and **relabelled B for verification**. The label is the only change: the base is the same frozen MAPPO u0 (clean2 `A_checkpoint_000.pt`, sha 5ea22964, the same file as `frozen_u0.pt`), with the same world, prefix, t0 sample and CRN. Verification then ran again natively in a separate B bank. R rows were withdrawn (kept in `COOP_PROBE_MINED_ALL_ARMS.jsonl`); R must come from R's own procedure.

Not verified here: clean3's own `admit_mined` acceptance of these rows. The launch operator should run it.

## Evidence paths
- **cpuV `/home/ubuntu/coopprobe20261106/`:**
  - `CHARGES.jsonl`
  - `results/{wall,wall_ext,hp,lowcoop,native}.jsonl`
  - `results/SUMMARY.json`
  - `minebank/` (native banks and `MINED_EXAMPLES`)
  - `verify/` (branch receipts and tapes)
  - `collect/`
  - `jobs/`
  - `wmrank/` and `wmq/` (WM queries and ranks)
  - `PIDS.txt`
- **Local copy of small results:** `runs/active/C4-V7FT-20261017/coop_probe/`
- **Status lines:** `CLEAN1_STATUS.md`, lines prefixed `PROBE`.
