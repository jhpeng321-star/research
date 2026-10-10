# SLOT-POOL CENSUS V3 (FEAT3 pack_v3, WM-N1 shards_v3, K_max=192)

Read-only census over every featurized run: how many vehicle-ticks the featurizer still dropped because more than K_max=192 concurrent vehicles were present (192-slot pool), from the shard ``meta.json`` drop counters (``n_dropped_slot_steps`` = featurizer ``n_dropped_concurrent``). Counts only; no outcomes.

- shard verify: ok=True over 2463 shards (DONE.json output-sha verification; all shards self-describe format=v3 k_max=192)
- runs: 2463 (runs with drops: 63)
- dropped vehicle-ticks (total): 134,708
- rostered vehicle-ticks (total): 377,385,090
- dropped / rostered vehicle-ticks (measured residual @K=192): 0.035700%
- ticks total: 3,190,623; ticks with a full 192-slot pool: 10,301 (0.322900% of ticks, upper-bound indicator set of drop-affected ticks)
- A5 census residual estimate (pooled, 300-run sample): 0.139900% (57,157 / 40,855,639); delta measured - estimate: -0.104200%

## By batch

| batch | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| chunk1 | 436 | 14 | 19,171 | 67,207,894 | 0.028500% | 2,543 |
| chunk1_fix27 | 84 | 0 | 0 | 12,009,224 | 0.000000% | 0 |
| chunk2 | 448 | 11 | 6,048 | 70,014,148 | 0.008600% | 1,136 |
| chunk3 | 448 | 20 | 10,542 | 68,893,526 | 0.015300% | 2,179 |
| chunk4 | 648 | 1 | 1,007 | 100,246,457 | 0.001000% | 147 |
| truearms_chunk2 | 112 | 3 | 1,093 | 16,614,708 | 0.006600% | 238 |
| truearms_chunk3 | 112 | 4 | 1,271 | 16,358,363 | 0.007800% | 371 |
| walls_pilot_chunk4 | 175 | 10 | 95,576 | 26,040,770 | 0.367000% | 3,687 |

## By loading (mainline demand, mNNNN)

| loading | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| m2000 | 189 | 0 | 0 | 17,880,740 | 0.000000% | 0 |
| m2400 | 259 | 1 | 761 | 35,769,198 | 0.002100% | 166 |
| m2600 | 758 | 9 | 21,811 | 117,661,475 | 0.018500% | 1,090 |
| m2700 | 1257 | 53 | 112,136 | 206,073,677 | 0.054400% | 9,045 |

## By CAV penetration (pNN)

| penetration | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| p02 | 434 | 14 | 5,481 | 65,279,316 | 0.008400% | 1,111 |
| p05 | 433 | 5 | 5,861 | 66,174,975 | 0.008900% | 978 |
| p10 | 490 | 25 | 100,811 | 78,052,534 | 0.129200% | 5,267 |
| p20 | 503 | 15 | 16,414 | 78,196,327 | 0.021000% | 2,261 |
| p40 | 603 | 4 | 6,141 | 89,681,938 | 0.006800% | 684 |

## By loading x penetration

| loading/penetration | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share |
|---|---|---|---|---|---|
| m2000/p02 | 71 | 0 | 0 | 6,665,322 | 0.000000% |
| m2000/p05 | 28 | 0 | 0 | 2,594,758 | 0.000000% |
| m2000/p10 | 10 | 0 | 0 | 801,150 | 0.000000% |
| m2000/p20 | 20 | 0 | 0 | 2,000,121 | 0.000000% |
| m2000/p40 | 60 | 0 | 0 | 5,819,389 | 0.000000% |
| m2400/p02 | 39 | 0 | 0 | 5,119,403 | 0.000000% |
| m2400/p05 | 54 | 0 | 0 | 7,523,506 | 0.000000% |
| m2400/p10 | 18 | 0 | 0 | 2,833,591 | 0.000000% |
| m2400/p20 | 69 | 0 | 0 | 9,143,070 | 0.000000% |
| m2400/p40 | 79 | 1 | 761 | 11,149,628 | 0.006800% |
| m2600/p02 | 81 | 0 | 0 | 12,761,053 | 0.000000% |
| m2600/p05 | 148 | 0 | 0 | 22,206,006 | 0.000000% |
| m2600/p10 | 165 | 5 | 21,176 | 25,478,422 | 0.083100% |
| m2600/p20 | 166 | 3 | 634 | 26,370,728 | 0.002400% |
| m2600/p40 | 198 | 1 | 1 | 30,845,266 | 0.000000% |
| m2700/p02 | 243 | 14 | 5,481 | 40,733,538 | 0.013500% |
| m2700/p05 | 203 | 5 | 5,861 | 33,850,705 | 0.017300% |
| m2700/p10 | 297 | 20 | 79,635 | 48,939,371 | 0.162700% |
| m2700/p20 | 248 | 12 | 15,780 | 40,682,408 | 0.038800% |
| m2700/p40 | 266 | 2 | 5,379 | 41,867,655 | 0.012800% |

Per-run counts (counts only) are in `SLOT_POOL_CENSUS_V3.json` under `runs`.
