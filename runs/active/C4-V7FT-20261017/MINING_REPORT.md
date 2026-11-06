# MINING_REPORT (S-MINING operator 2, Claude) - 2026-11-06 05:15 +08

**Bottom line: 0 admitted G5 (move, seed) examples, so MINED_MOVES.jsonl is empty.** No verified joint move beat its best single CAV with a 95% lower bound above 0. The largest measured synergy was +0.005 veh-s, which is noise. The launch threshold (3 or more moves from 2 or more families) cannot be met with the accepted G5 vocabulary on this evidence.

## Budget and hosts
- Mining ledger: gpuA `/root/mining20261106/CHARGES.jsonl`, phase `mining`. Charged **472,341 of 500,000 sim-s**. This includes the 960 parity and 6,000 prior-operator collection sim-s. cpuV was not used for physics.
- Collection: 44 new TRAIN episodes (seeds 986046-986071, B and R), 52.8k sim-s, 82 merge-pressure contexts.
  - Selection used observables only: local ramp count, local mainline count, sum of 1/(1+v_ramp), and the pair gap.
  - A second collection of 16 episodes required the ramp gate to be feasible before the merge. It captured **0** such moments.
- Verification: native all8, run by `sumo_verify.verify` and `verify_perseed.confirm_all8`.
  - Each decision runs the group and the policy on both CRN seeds, an extended later-harm window, and every member solo.
  - Synergy is measured on the common window. Admission is per (move, seed) under per_seed_v1, never pooled.
- Code: branch `c4/s-mining`, files `tools/mining/{collect2,mine_verify,mine_extract,g5_solo_patch}.py`. Run root `/root/mining20261106/op2_verify01` (about 4.3 GB on gpuA, left in place).

## Fixes needed to execute G5 at all (important for clean3)
1. **The frozen SUMO runtime lacks the G5 env.** `/home/ubuntu/e3final/runtime` env's `CavSumoEnv.step()` rejects `lane_actors`, so every executed G5 lane request crashed its branch (wave v2, 15 decisions lost).
   - Mining fix: a mining-only overlay. It is the frozen env plus the accepted 9fd32b4e additive hunks (sha 33efc058), loaded after the frozen setup. Branch prefix-parity checks still pass.
   - **Clean3 will hit the same error unless its physics runtime carries the G5 env.**
2. **G5 solos could not run.** `replay_bank.solo_candidate` yields a one-member G5 plan, which `group_lane.from_plan` rejects.
   - Mining fix (`g5_solo_patch.py`): mark the solo plan and issue only that member's own role. Group plans are unchanged.
3. Per-actor lane telemetry now confirms ramp lane changes were `issued` and `completed`.

## Why synergy is zero (mechanism)
- **open_gap_merge.** The ramp attempt is a single tick at onset+1, and the accepted gate passes only when a safe gap already exists. That is exactly when SUMO's native merge happens.
  - In every executed case the group equals the main-only control (`gap_open`) to within 0.1 veh-s, and best solo = group.
  - The vocabulary cannot open a gap before the attempt, because main easing starts at onset.
- **yield_lane.** The ramp CAV has no role (zero offsets, no lane command), so synergy is structurally 0.
- **Up to 6 CAVs (owner 04:10).** Accepted G5 rows must have exactly 2 members, and the clean3 reader calls `group_lane.from_plan`. Multi-CAV moves can therefore only be reported, not delivered. `moves.MOVE_MAX_MEMBERS` is 4.

## Counts, tried / admitted-vs-policy (both CRN seeds counted), excluding the 15 runtime-crash decisions
| family | 986046 | 986047 | 986060 | 986061 | 986062 | 986063 | 986068 | cooperative |
|---|---|---|---|---|---|---|---|---|
| open_gap_merge (G5) | 2/1 | 6/0 | 6/0 | 28/6 | 6/0 | 8/0 | 14/5 | 1 (v1, +0.005, lane blocked, so not G5-valid) |
| yield_lane (G5) | 2/0 | 2/0 | 2/0 | 10/1 | 4/0 | 2/0 | 4/1 | 0 |
| gap_open (control) | 2/1 | 2/0 | 2/0 | 10/2 | 4/0 | 2/0 | 6/2 | 0 |
| ramp_assist (probe) | 4/0 | - | 4/0 | 12/3 | 4/0 | - | 4/0 | 0 |
| wall 3-4 CAV (probe) | - | - | 4/0 | 12/2 | 4/0 | - | - | 0 |
| merge-timed upstream wall (c) | 0 tried: never 2 or more CAVs upstream of the zone in a sensed packet with 20 s or more before the merge |

Decisions per wave:
- v1: 9 (initial slate, G5 attempts gate-blocked).
- v2: 15 (runtime crash).
- v3b: 1.
- v4: 4 (G5 geometry with gate-feasible timing).
- p1: 7 (multi-CAV probe).

Seeds 986064, 986065, 986067 and 986071 appear only in v2.

## Near-misses and potential
- Synergy was measured on 22 examples, all between -0.001 and +0.005 veh-s. There is no positive-synergy near-miss.
- Several moves beat the policy but with zero synergy, meaning one member does all the work:
  - B 986068 t450: open_gap_merge, +83.5 to +87.5 veh-s on seed 1 (lower bound up to +52). Fails on seed 2 at -225.
  - R 986061 t455: ramp_assist d1/h30, +390 (lower bound +181), synergy -0.0.
- Unmeasured: R 986061 t455, 4-CAV wall d1/h30, +490 on seed 1 (lower bound +136) and +175 for d0.6 on seed 2. Its solos were cut by the budget cap, so synergy is unknown. It is the only open multi-CAV lead.
- WM (GPU, wm_lane04): K8/H300, ranking only.
  - Old slate: about 0 synergy everywhere, except +5.9 on R 986062 t505 (saving -1.3). SUMO verified 0.0 there.
  - The G5-pair ranker (wm_lane04/g5pairs) was stopped for lack of budget.

## Recommendation
To get real joint moves, the vocabulary needs a pre-opened gap. For example: the main eases for k seconds, then the ramp attempts at onset+k (k > 1). Or allow a ramp-assist acceleration role. That needs a G5 vocabulary change and acceptance, which is out of mining scope. Fix the clean3 runtime env first in any case.

Note: this file replaced operator 1's 00:38 zero-count report locally. Their original is preserved on cpuV at /home/ubuntu/coopmwm20261106_s_mining/MINING_REPORT.md (7217 bytes).
