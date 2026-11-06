# HICAV_PROBE_REPORT: high-CAV-share wall cooperation probe, 2026-11-06 (owner 12:46, item 1)

## Summary
1. **Verdict: NO.** At 0.6 and 0.8 CAV share (2700 vph, current map), group cooperation is not more frequent than at 0.4 and not more selectable.
2. **Frequency:** a material admissible wall on seed 1 at 2 of 12 moments (0.6) and 1 of 10 (0.8). At 0.4 it was about 1 of 10.
3. **Robustness:** the admitted walls replicate on only 2/4, 1/4 and 1/4 seeds, the same per-seed bet seen at 0.4 (3/6 and 1/6).
4. **Synergy:** seed-1 values +732, +142 and +74 veh-s (LB95 +364, +107, +22); replicate seeds range from +377 to −159.
5. **Best case:** 995004 t355 at 0.6 is a pure 5-CAV group effect (no solo helps): 4-seed mean +277 veh-s vs policy, worst seed −0.4.
6. **Observability:** no admitted wall fits one member's 4 or 6 slots. Members span 181–253 m, beyond the 100 m sensing range.
7. **Slots shrink with share:** the 3rd-nearest CAV is a median 13.5 m away at 0.6 and 10.3 m at 0.8. Only 8 of 84 walls fit in 4 slots (all k ≤ 4).
8. **Low authority:** at 6 of 22 moments every wall was within 1 veh-s of policy; walls often lost (down to −394 veh-s).
9. **MAPPO sanity:** MAPPO u0 vs native team delay +9.0 / −0.1 % (0.6) and −0.1 / −4.8 % (0.8), so on par. 0 collisions; more TTC<3 s exposure, less hard braking.
10. **Safety:** 0 collisions and 0 teleports in every wall, solo, replicate and episode window.
11. **Budget:** 149,715 of 150,000 sim-s on the own cpuV ledger, ≤ 12 cores. 12,000 of it was a pre-physics guard refusal, kept in the ledger.

## Key findings (detail)
2. **Frequency.** A material admissible wall on seed 1 (beats policy, synergy LB95 > 0, synergy at least 1 veh-s) was found at 2 of 12 merge-pressure moments at 0.6 and 1 of 10 at 0.8. At 0.4 (coop_probe hp, narrower WM-top search) it was 1 of about 10.
3. **Per-seed robustness.** The admitted walls do not replicate:
   - 995004 t355 (5 CAVs) on 2 of 4 seeds;
   - 995000 t355 (5 CAVs) on 1 of 4;
   - 995103 t365 (4 CAVs) on 1 of 4.
   This is the same "per-seed bet" pattern as at 0.4 (3/6 and 1/6).
4. **Synergy values.** Seed-1 admitted values were +732, +142 and +74 veh-s (LB95 +364, +107, +22). Replicate seeds ranged from +377 to −159.
   - The best case is 995004 t355 at 0.6, a pure group effect: no single member helps. It has a 4-seed mean of +277 veh-s vs policy and never loses more than 0.4 veh-s.
5. **Observability.** None of the 3 admitted walls fits one member's 4 slots (ego + 3), and none fits 6 slots.
   - Their members span 181–253 m, more than the 100 m sensing range. Even ignoring range, they would need 8–14 slots.
   - Higher share makes this worse. The 3rd-nearest CAV is a median of 13.5 m away at 0.6 and 10.3 m at 0.8, so 4 slots cover only about ±10–14 m.
   - Over all 84 tried walls, 8 fit in 4 slots (all k ≤ 4) and 26 fit in 6.
6. **Many moments give walls no authority.**
   - 6 of 22 moments had every wall within 1 veh-s of policy.
   - At 8 of 22 moments, no wall beat policy and at least one wall cost ≥ 1 veh-s. Across all moments, walls cost up to −394 veh-s at 0.6 and −266 at 0.8.
7. **MAPPO sanity** (base = frozen MAPPO u0, 2 paired worlds per share): team delay vs native was +9.0 % / −0.1 % at 0.6 and −0.1 % / −4.8 % at 0.8. That is on par with native, so MAPPO is not broken outside its training range.
   - 0 collisions and 0 teleports in all 14 episodes.
   - Hard braking is lower than native; TTC<3 s exposure is higher (+6 % to +61 %).
8. **Safety.** 0 collisions and 0 teleports in every wall, solo and replicate window. Hard-brake deltas were −8 to +3.5; TTC deltas −75.5 to +26.5 veh-s.
9. **Budget.** 149,715 of 150,000 sim-s on the own cpuV ledger (64 rows), using at most 12 cpuV cores.
   - 12,000 of that was charged with no physics: the first collection attempt was refused pre-physics by a guard and is kept in the ledger.

## Setup
- **Authority.** Owner 12:46 "do it all", item (1), phase `hicav_probe` ≤ 150,000 sim-s. MAIN reallocation: ≤ 12 cpuV cores (gpuA was not used).
- **Registration.** `REGISTRATION.md` 13:03, plus amendments at 13:14 and the remaining-budget allocation. All were registered before the physics they cover.
- **Worlds.** A fresh registered range, **995000–995199**: 995000–995004 at share 0.6, 995100–995104 at share 0.8, both at 2700 vph. No 962xxx, 964000–964199 or 977xxx seeds were used.
  - Seed membership in the runtime SEALED/BLOCKED sets was checked at every setup; there were no hits.
  - Generator: the same `CavSumoEnv` textbook population `reset(seed, demand_vph, penetration)` as the training cells. Only the CAV fraction differs. Role `hicav` admits exactly these worlds.
- **Base.** Frozen MAPPO u0 (clean2 `A_checkpoint_000`, sha 5ea22964) on MAPPO 169, for all CAVs, as in clean3 and coop_probe. Settings: stochastic u0 residual, arm label B, noise_seed = 20261106 + seed + 9000.
- **Comparator.** The true-native plant (`tools.coopnative.plant`, CAVs released to SUMO) on 995000/995001/995100/995101.
- **Moments.** Selected from observables only (coop_probe selector): merge pressure at t0 ∈ [300, 500] with ≥ 3 mainline CAVs in s∈[−400, 100]. Score ranks 0–1 of each episode were screened; 2 rank-2 moments at 0.6 were also screened while budget remained.
- **Walls.** Nested k = 3, 4, 5, 6 mainline CAVs nearest behind the ramp head (coop_probe slate geometry), o0 h30 d1 on the seed-1 policy reference. SUMO-only search; no WM ranking.
- **Stages** (TAIL 120 s; `sumo_verify` branches with exact prefix replay):
  - A: seed 1, policy + all walls.
  - B: seed 1, policy + member solos of walls with adv LB95 > 0. Policy J reproduced bit-for-bit in all 7 B runs.
  - C: replicate seeds via `rb.replicate`, salt `hicavprobe#sid#r`.
- **Statistic.** per_seed_v1 (`bank_statistics._stats`, 30-s circular-block paired bootstrap, one-sided LB95). Never pooled.
- **Materiality rule** (registered 13:14, after noise-level "admissions" appeared): solos and replicates are run only for effects ≥ 1 veh-s. Formal (LB95 > 0) counts are reported alongside.

## MAPPO sanity (team delay = speed deficit + source wait, veh-s; t = 1200, i.e. 900 s controlled)
| share | world | MAPPO-u0 | native | MAPPO − native | TTC<3 s veh-s M/N | hard brake M/N | coll / tele |
|---|---|---|---|---|---|---|---|
| 0.6 | 995000 | 160,473 | 147,226 | +13,247 (+9.0 %) | 1008 / 625 | 13.5 / 24 | 0 / 0 |
| 0.6 | 995001 | 176,657 | 176,798 | −141 (−0.1 %) | 963 / 832 | 22 / 86 | 0 / 0 |
| 0.8 | 995100 | 184,404 | 184,538 | −134 (−0.1 %) | 1274 / 1005 | 6 / 80 | 0 / 0 |
| 0.8 | 995101 | 222,997 | 234,217 | −11,220 (−4.8 %) | 1134 / 1067 | 11.5 / 73.5 | 0 / 0 |

All 5 MAPPO episodes per share:
- **0.6:** mean team delay 163,420; up to 89 CAVs.
- **0.8:** mean team delay 188,676; up to 124 CAVs.
- **Both:** clamp/override fraction 0.90–0.91, the same as at 0.4 in the coop_probe collections (0.89–0.90). There were no slot or early-termination anomalies.

Caveat: only 2 pairs per share, so this is a sanity check, not an efficacy estimate.

## Wall search per share
| | share 0.6 | share 0.8 |
|---|---|---|
| moments screened / walls tried | 12 / 46 | 10 / 38 |
| walls beating policy (adv LB95 > 0), formal / material (≥ 1 veh-s) | 9 / 6 | 12 / 6 |
| moments with a material wall beating policy | 3 | 2 |
| moments where every wall is within 1 veh-s of policy | 3 | 3 |
| walls with synergy measured (solos run) | 7 | 9 |
| seed-1 admitted, formal (walls / moments) | 2 / 2 | 5 / 2 (3 are noise: 995104 t475, synergy +0.0004) |
| **seed-1 admitted, material (moments)** | **2 / 12** (995004 t355, 995000 t355) | **1 / 10** (995103 t365) |
| group gain but zero synergy (one member does all) | 995001 t425: +126 for every k, best solo +126 | 995101 t425: +111, best solo +114 (synergy −3) |

Admitted walls, per seed:
- group−policy is group adv vs policy; synergy = J(best solo) − J(group). Positive = cooperation helps.
- ✓ = admitted (adv LB95 > 0 and synergy LB95 > 0).

| wall | seed 1 | seed 2 | seed 3 | seed 4 | admitted |
|---|---|---|---|---|---|
| 0.6, 995004 t355, 5 CAVs: group−policy | +732.0 | −0.4 | −0.4 | +377.1 | |
| synergy (LB95) | **+732.0 (+364.1) ✓** | −0.4 (−0.8) | −0.4 (−0.7) | **+377.1 (+201.6) ✓** | 2/4 |
| 0.6, 995000 t355, 5 CAVs: group−policy | +171.7 | +44.7 | +174.2 | −4.5 | |
| synergy (LB95) | **+142.2 (+107.2) ✓** | −27.1 (−65.2) | +58.3 (−21.7) | −16.3 (−98.9) | 1/4 |
| 0.8, 995103 t365, 4 CAVs: group−policy | +425.2 | −159.3 | −2.6 | +47.8 | |
| synergy (LB95) | **+74.0 (+21.9) ✓** | −159.0 (−303.4) | −2.8 (−5.8) | −27.9 (−46.3) | 1/4 |
| 0.8, 995104 t475, 4 CAVs (noise-level) | +0.05 / syn +0.0004 | −0.04 | −0.08 | – | formal 1/3, immaterial |

Notes on these walls:
- 995103's 5-CAV wall: seed 1 only, +421.8, synergy +70.6 (LB95 +21.3).
- 995103's 3-CAV wall: +72.9, which is below each of its solos (+188 to +216), so synergy is −143.
- 995004 t355 solos are all exactly 0 on every seed. Its effect exists only as a group.

**Comparison with 0.4** (COOP_PROBE_REPORT, cell 11 = 2700 × 0.4):
- 1 admitted moment of about 10 screened (986067 t340, 1/6 seeds), with one WM-top wall per moment.
- The lead wall 986061 t455 admitted on 3/6 seeds.

Here the search was broader (all nested 3–6 CAV walls in SUMO), yet the hit rate is similar (2/12 and 1/10) and per-seed replication is similar (3/8 and 1/4).

## Observability (zero sim, t0 CAV rows; slots = ego + nearest CAVs by |ds| ≤ 100 m)
| | share 0.6 | share 0.8 |
|---|---|---|
| walls fitting 4 slots / 6 slots | 5 / 18 of 46 | 3 / 8 of 38 |
| by k (fits4 / fits6) | k3 4/10, k4 1/5, k5 0/3, k6 0/0 (of 12, 12, 12, 10) | k3 3/5, k4 0/3, k5 0/0, k6 0/0 (of 10, 10, 10, 8) |
| median CAVs within 100 m of a wall member | 15 | 22 |
| median distance to the 3rd / 5th nearest CAV | 13.5 / 28.8 m | 10.3 / 19.3 m |
| material admitted walls fitting 4 / 6 slots | 0 / 0 (spans 181, 253 m) | 0 / 0 (span 189 m) |

For the admitted walls, the best-placed member sees only 3 of 4 (or 2 of 3) other members within 100 m. Ignoring range, they would need 14, 8 and 9 slots.

The cooperative walls are long platoons spread across lanes 0 and 1. Per-CAV V2V slots cannot assemble them; at higher share, slot coverage shrinks further.

## Safety counters
- **Wall windows (A, 84 walls):** collisions and teleports deltas were 0 everywhere. Hard-brake deltas were −8 to +1.5 (0.6) and −1 to +1 (0.8). TTC<3 s deltas were −75.5 to +26.5 and −33 to +15.5 veh-s.
- **Admitted walls:** TTC changes of −75.5, +5.5 and −33.
- **Replicates (C):** 0 collisions and 0 teleports; hard brake up to +3.5 (995103, seed 2).
- **Episodes:** 0 collisions and 0 teleports in all 14 (10 MAPPO, 4 native).

## Budget and deviations
Ledger: cpuV `/home/ubuntu/hicavprobe20261106/CHARGES.jsonl`, 149,715 / 150,000 sim-s, 64 rows, precharged and never replayed. By component:

| component | sim-s | jobs | note |
|---|---|---|---|
| native | 4,800 | 4 | |
| collect (attempt 2) | 12,000 | 10 | |
| collect (attempt 1) | 12,000 | 10 | refused pre-physics (0 physics) |
| screen A | 58,800 | 22 | |
| solos B | 24,135 | 7 | |
| replicates C | 37,980 | 11 | 7,500 of these on the noise-level 995104 wall, started before the materiality rule |

1. **Attempt-1 fault.** `tools.coopmwm.accel_protocol.guard` keeps its own `ORIGINAL_WORLDS`, so the `hicav` role was refused before any SUMO step. The fix routes only the `hicav` role through it. A zero-charge guard-chain test passed, and collections were re-run as `_a2`.
2. **Budget skips.** The driver's budget guard skipped 2 rank-2 screens (995002 t425, 995004 t375). It also skipped the 995000 t355 solos, which were then run manually under the registered remaining-budget allocation.
3. **Replicate seeds.** Seeds 2–3 were used for every admitted wall, and seed 4 for the three material ones. Seeds 5–6 were not run (budget).

## Evidence
- **cpuV `/home/ubuntu/hicavprobe20261106/`** (2.0 GB):
  - `CHARGES.jsonl`
  - `WORLDS.json`
  - `PLAN*.json`
  - `DRIVER_STATE.json`
  - `results/{A,B,C,B_skipped}.jsonl`
  - `SUMMARY.json`
  - `collect/` (snapshots and contexts)
  - `native/`
  - `verify/` (branch tapes and receipts)
  - `logs/`
  - `PIDS.txt` / `PIDS.jsonl`. All own processes have exited.
- **Local small copy:** `runs/active/C4-V7FT-20261017/hicav_probe/`. It holds the results, summary, ledger, and the collection and native receipts.
- **Code:** branch `c4/hicav-probe`, commit a2e38b74 (not pushed), worktree `E:/agent-work/c4-hicav-probe`:
  - `tools/mining/hicav.py`
  - `tools/mining/hicav_site.py` (deployed as `sitecustomize.py`)
  - `tools/mining/hicav_summary.py`
  - The code runs on the coop_probe cpuV source.
- **Records:**
  - `REGISTRATION.md`: HICAV-PROBE entries.
  - `CLEAN1_STATUS.md`: PROBE-HICAV lines.
  - The master ledger (`protocols/experiment_history.json`) has not been updated by this operator. The integrating agent owns that close-out.
- **Not opened or read:** no sealed or holdout paths, and no truth or driver-parameter files (`hetdata_truth.csv` files in scenario dirs were never opened). HDVs were never commanded, and the preserved network was not modified.
