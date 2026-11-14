# AMENDMENT 37 - exploratory zero-SUMO description: cell-searched plans vs the D2800 bank fd385a2b

Description only (for the paper). Inputs: the existing WM searches STRATEGY_SEARCH_D3200_p20.json (5674d956) and
STRATEGY_SEARCH_D2400_p20.json (9fa5fa53) (WM-E ft_r1, no new search), the A37 candidate files CANDIDATES_<cell>.json
(top 9 per cell by WM score; comparator rows and plans the executor refuses skipped) and the five plan templates of bank
fd385a2b (GUIDANCE_R6_WMONLY_PRIORITY.json). Nothing here uses any SUMO outcome: no a35 panel result, no verification
result (none exists yet); WM scores are the search's own imagined scores. SUMO lanes: 0 = closing lane, 1 = middle,
2 = leftmost. "gate x" = the plan starts only when the observed closing-lane field speed is <= x m/s; "lanes" trigger =
members chosen per lane/rank inside the zone instead of the one-CAV-per-lane wall.

## Main differences

- **D3200_p20 (heavier demand) -> slower, same shape.** Seven of nine candidates are the same two-CAV lane-1 + lane-2
  hold as the D2800 bank, but every wall role holds at **18 m/s** (bank: 18-22, mean 19.5); the top two plans
  (WM #1 mut_0d1deed5, #2 mut_82ad29f5) are the same plan except that #2 starts only in congestion (gate 5 m/s).
  Onsets are staggered more often (42% of roles start 20 s late vs 20% in the bank) and the lane-2 member brakes
  harder more often in the top plans (dmax 1.5). Two candidates are new structures absent from the bank: a 4-CAV
  "lanes" plan in the far zone that includes the closing lane (mut_b1a94539; WM-feasible on 0 of 30 states, so its WM
  score is 0) and a 2-CAV near-zone lanes plan (rand_154d387c, a random immigrant kept by the search).
- **D2400_p20 (lighter demand) -> faster and smaller.** Six of nine candidates are **single-CAV lane-2 holds**
  (bank: none; all five bank plans use two CAVs), at 19-22 m/s (mean 20.6 vs 19.5), with shorter cooldowns
  (mean 118 s vs 140 s) and shorter holds (two 90 s plans). The two-CAV plans at this cell hold both lanes at
  **21 m/s** (bank: 18-22 mixed per lane). One plan is congestion-gated (mut_1857ceb0, gate 15 m/s).
- **Unchanged across all three sets:** hold-only plans (no lane changes), wall trigger in the full/near/far zones,
  one session slot, roles released at hold end or at the drop; no plan in either cell is one of the five bank plans
  (strategy keys differ), and the hand-written duo12 family appears as duo12_v18 (D3200, 18 m/s) and duo12_v21
  (D2400, 21 m/s) vs DUO_HOLD duo12_v20 (20 m/s) - i.e. the search moved the hand-written hold's speed down at the
  heavy cell and up at the light cell.

### D2800_p20 bank fd385a2b (5 WM plans; LEARNED_TRIGGER / BANK_ALWAYS)
| plan | rank / row class | members | trigger/zone | SUMO lanes | v_w (m/s) | hold (s) | onset (s) | dmax | cooldown (s) | gate / start_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| mut_006c290e03ad3d3c | situational | 2 | wall/full | 1/2 | 18/22 | 120/180 | 0/20 | 1.0/1.0 | 200 | None / None |
| mut_0d50e4e18041530d | situational | 2 | wall/near | 1/2 | 20/19 | 120/120 | 0/0 | 1.0/1.0 | 120 | None / None |
| mut_6c2bf382711cbc37 | situational | 2 | wall/full | 1/2 | 20/18 | 120/90 | 0/0 | 1.5/1.5 | 120 | None / None |
| mut_cfdbafe5f3316f1a | situational | 2 | wall/full | 1/2 | 19/21 | 120/120 | 0/0 | 1.0/1.5 | 120 | None / None |
| mut_b66c4207002b5a15 | unconditional_fallback | 2 | wall/near | 1/2 | 20/18 | 120/90 | 20/0 | 1.0/1.5 | 140 | None / None |

### D3200_p20 candidates (top 9 by WM score; verification order)
| plan | rank / row class | members | trigger/zone | SUMO lanes | v_w (m/s) | hold (s) | onset (s) | dmax | cooldown (s) | gate / start_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| mut_0d1deed54cf85a43 | WM #1 score 97.74 | 2 | wall/full | 1/2 | 18/18 | 120/120 | 20/0 | 1.0/1.5 | 140 | None / None |
| mut_82ad29f594da0373 | WM #2 score 70.9 | 2 | wall/full | 1/2 | 18/18 | 120/120 | 20/0 | 1.0/1.5 | 140 | 5.0 / None |
| duo12_v18 | WM #3 score 32.88 | 2 | wall/full | 1/2 | 18/18 | 120/120 | 0/0 | 1.0/1.0 | 120 | None / None |
| mut_b14f5c979f796e93 | WM #4 score 27.07 | 1 | wall/full | 2 | 18 | 90 | 0 | 1.0 | 90 | 5.0 / None |
| mut_ad6205c9d248ab0f | WM #5 score 10.41 | 2 | wall/full | 1/2 | 18/18 | 120/120 | 0/0 | 1.5/1.0 | 120 | None / None |
| mut_ed21f451c77f2878 | WM #7 score 0.9 | 2 | wall/full | 1/2 | 18/18 | 180/90 | 20/0 | 1.0/1.0 | 200 | None / None |
| mut_b1a94539f69c5715 | WM #8 score 0.0 | 4 | lanes/far | 0/1/2/0 | 19/19/19/18 | 120/120/120/90 | 0/20/20/20 | 1.5/1.5/1.0/1.0 | 140 | None / None |
| mut_a3534d24f4102ed1 | WM #10 score -3.14 | 2 | wall/full | 1/2 | 22/18 | 120/180 | 20/0 | 1.0/1.0 | 180 | None / None |
| rand_154d387cac4e120a | WM #11 score -6.51 | 2 | lanes/near | 1/2 | 22/18 | 120/90 | 0/20 | 1.0/1.0 | 120 | None / None |

### D2400_p20 candidates (top 9 by WM score; verification order)
| plan | rank / row class | members | trigger/zone | SUMO lanes | v_w (m/s) | hold (s) | onset (s) | dmax | cooldown (s) | gate / start_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| mut_a857168d4aae7b79 | WM #1 score 62.93 | 2 | wall/full | 1/2 | 21/21 | 120/120 | 0/0 | 1.5/1.5 | 120 | None / None |
| duo12_v21 | WM #2 score 59.22 | 2 | wall/full | 1/2 | 21/21 | 120/120 | 0/0 | 1.0/1.0 | 120 | None / None |
| solo2_v20_h120 | WM #3 score 50.87 | 1 | wall/full | 2 | 20 | 120 | 0 | 1.0 | 120 | None / None |
| mut_62e9d6ccfa482266 | WM #4 score 41.39 | 1 | wall/full | 2 | 19 | 120 | 0 | 1.5 | 120 | None / None |
| mut_d964f49fab144787 | WM #5 score 40.31 | 1 | wall/far | 2 | 20 | 120 | 20 | 1.0 | 140 | None / None |
| mut_fd73f0eab0e4fc39 | WM #6 score 30.7 | 2 | wall/full | 1/2 | 21/21 | 120/120 | 20/0 | 1.5/1.5 | 140 | None / None |
| mut_1857ceb0537425f5 | WM #7 score 29.17 | 1 | wall/full | 2 | 20 | 90 | 0 | 1.0 | 90 | 15.0 / None |
| solo2_v20_h90 | WM #8 score 27.07 | 1 | wall/full | 2 | 20 | 90 | 0 | 1.0 | 90 | None / None |
| mut_e61e09262d304800 | WM #9 score 25.09 | 1 | wall/far | 2 | 22 | 120 | 0 | 1.0 | 120 | None / None |

### Summary statistics (role-level; zero SUMO)
| set | plans | mean members | share with lane-1 role | role v_w mean [min-max] | hold mean | share onset 20 s | share dmax 1.5 | cooldown mean | gated / lanes-trigger |
|---|---|---|---|---|---|---|---|---|---|
| D2800 bank | 5 | 2.00 | 1.00 | 19.5 [18-22] | 120 | 0.20 | 0.40 | 140 | 0 / 0 |
| D3200_p20 | 9 | 2.11 | 0.89 | 18.6 [18-22] | 120 | 0.42 | 0.26 | 139 | 2 / 2 |
| D2400_p20 | 9 | 1.33 | 0.33 | 20.6 [19-22] | 115 | 0.17 | 0.42 | 118 | 1 / 0 |
