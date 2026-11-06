# CLEAN4_GATE_FAIL - pre-launch potential gate failed (no launch)

Rule: MAIN 2026-11-06 pre-launch potential gate: in the chosen training cell (1) robust admissible group synergy (per-seed CI95 low>0, later-harm and dither checks pass) at >=15% of screened bottleneck moments; (2) >=8 verified B examples; (3) >=50% of those examples reachable within the configured V2V/RSU range. Fail -> no launch, CLEAN4_GATE_FAIL.md, stop.

Chosen training cell: {'cell_key': 'D3200_p60', 'demand': 3200, 'penetration': 0.6}

| check | value | required | pass |
|---|---|---|---|
| robust admissible group-synergy share of screened moments | 0.000 (0/8) | >= 0.15 | False |
| verified B examples | 1 | >= 8 | False |
| reachable within V2V/RSU range (100.0 m) | 0.000 (0/1) | >= 0.50 | False |

Sources: cpuV:/home/ubuntu/clean4_prelaunch_verify_93e304a9/POTENTIAL_INPUTS.json (sha256 062cb8645f5bde3bb63fb43520e9296a8411a0db5c365adb7a73934f346ab706); cpuV:/home/ubuntu/clean4_prelaunch_verify_93e304a9/CLEAN4_MOVES_B.jsonl (sha256 a4698f613f0fe0413166208a6c1cc1784329a1de4ece67034fcbc65748e0ea18); cpuV:/home/ubuntu/clean4_prelaunch_verify_93e304a9/MOVE_LIST.json (sha256 fe3836396a6482092a459124d02bd806e3a3d5866d967dea7158dd973a6f1a3f); cpuV:/home/ubuntu/clean4_prelaunch_verify_93e304a9/CHARGES.jsonl (sha256 b7eca87e0f25f40dc3a434dfd43a5aff76b133e1b61dd562c2d51c13a6aa2465)

Receipt sha256: b1a81fb12dadba0e34f2fc87176260375bf6dcfc5b78c442038baa9f2bfb1870

## Plan A outcome (phase clean4_prelaunch_verify, 75,802 of 400,000 sim-s; 0 collisions/teleports in admitted windows)
- Frozen run list (MOVE_LIST 1d629e43): 11 B wall/pair moves from PLANGEN (6) and PLANGEN2 (5), re-verified on base MAPPO u0 with exact N2 LaneDropMove plans (lane_drop_macro on every verified plan).
- Outcomes: 9 screen_rejected (no group with seed-1 advantage LB95>0 AND synergy LB95>0 AND >=1 veh-s); 1 member_absent_or_past_drop (B008 989013 t800: a frozen member is not upstream of the drop at t0 on the MAPPO base); 1 confirmed_admitted on seed 1 only (B002 989211 t800 wall_h, 3 CAVs: synergy +62.5 veh-s LB95 +0.34, advantage +1,983 LB95 +1,150), not admitted on seeds 2-3 (not robust) and its members span 241 m at t0 (> 100 m V2V/RSU reach).
- Cell D3200_p60: screened 8 moments, robust 0; D3200_p40: screened 2, robust 0. R: 0 rows by design (option a).
- Interpretation (descriptive): on the MAPPO u0 base the native-base probe admissions (PLANGEN/PLANGEN2) do not reproduce; group synergy is rarer than the probes suggested. Per the registered rule: NO LAUNCH; no added budget, no criteria change.
- Evidence: runs/active/C4-V7FT-20261017/clean4_verify/ (POTENTIAL_INPUTS, POTENTIAL_GATE, CLEAN4_MOVES_B, NEGATIVES, NEGATIVE_FRAMES, OBS_TABLES, MOVE_LIST, CHARGES, move_status).
