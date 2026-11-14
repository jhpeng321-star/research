# SLOT-POOL CENSUS (FEAT1 pack_v2c, WM-N1 shards_v2)

Read-only census over every featurized run: how many vehicle-ticks the featurizer dropped because more than K_max=128 concurrent vehicles were present (frozen 128-slot pool), from the shard ``meta.json`` drop counters (``n_dropped_slot_steps`` = featurizer ``n_dropped_concurrent``). Counts only; no outcomes.

- runs: 2463 (runs with drops: 1894)
- dropped vehicle-ticks (total): 21,368,334
- rostered vehicle-ticks (total): 356,151,464
- dropped / rostered vehicle-ticks: 5.999800%
- ticks total: 3,190,623; ticks with a full 128-slot pool: 1,200,454 (37.624400% of ticks, upper-bound indicator set of drop-affected ticks)

## By batch

| batch | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| chunk1 | 436 | 330 | 3,757,896 | 63,469,169 | 5.920800% | 211,259 |
| chunk1_fix27 | 84 | 63 | 647,183 | 11,362,041 | 5.696000% | 42,205 |
| chunk2 | 448 | 362 | 4,291,093 | 65,729,103 | 6.528500% | 246,514 |
| chunk3 | 448 | 328 | 4,372,044 | 64,532,024 | 6.775000% | 224,284 |
| chunk4 | 648 | 484 | 4,422,670 | 95,824,794 | 4.615400% | 274,100 |
| truearms_chunk2 | 112 | 92 | 1,088,304 | 15,527,497 | 7.008900% | 60,671 |
| truearms_chunk3 | 112 | 81 | 1,143,590 | 15,216,044 | 7.515700% | 54,949 |
| walls_pilot_chunk4 | 175 | 154 | 1,645,554 | 24,490,792 | 6.719100% | 86,472 |

## By loading (mainline demand, mNNNN)

| loading | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| m2000 | 189 | 0 | 0 | 17,880,740 | 0.000000% | 1,051 |
| m2400 | 259 | 114 | 663,267 | 35,106,692 | 1.889300% | 55,023 |
| m2600 | 758 | 624 | 5,874,668 | 111,808,618 | 5.254200% | 360,125 |
| m2700 | 1257 | 1156 | 14,830,399 | 191,355,414 | 7.750200% | 784,255 |

## By CAV penetration (pNN)

| penetration | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share | pool-full ticks |
|---|---|---|---|---|---|---|
| p02 | 434 | 316 | 4,060,557 | 61,224,240 | 6.632300% | 213,331 |
| p05 | 433 | 308 | 3,564,926 | 62,615,910 | 5.693300% | 200,829 |
| p10 | 490 | 432 | 5,171,578 | 72,981,767 | 7.086100% | 267,553 |
| p20 | 503 | 394 | 4,575,853 | 73,636,888 | 6.214100% | 247,431 |
| p40 | 603 | 444 | 3,995,420 | 85,692,659 | 4.662500% | 271,310 |

## By loading x penetration

| loading/penetration | runs | runs w/ drops | dropped veh-ticks | rostered veh-ticks | drop share |
|---|---|---|---|---|---|
| m2000/p02 | 71 | 0 | 0 | 6,665,322 | 0.000000% |
| m2000/p05 | 28 | 0 | 0 | 2,594,758 | 0.000000% |
| m2000/p10 | 10 | 0 | 0 | 801,150 | 0.000000% |
| m2000/p20 | 20 | 0 | 0 | 2,000,121 | 0.000000% |
| m2000/p40 | 60 | 0 | 0 | 5,819,389 | 0.000000% |
| m2400/p02 | 39 | 20 | 122,421 | 4,996,982 | 2.449900% |
| m2400/p05 | 54 | 25 | 112,473 | 7,411,033 | 1.517600% |
| m2400/p10 | 18 | 11 | 155,411 | 2,678,180 | 5.802900% |
| m2400/p20 | 69 | 16 | 67,763 | 9,075,307 | 0.746700% |
| m2400/p40 | 79 | 42 | 205,199 | 10,945,190 | 1.874800% |
| m2600/p02 | 81 | 65 | 725,909 | 12,035,144 | 6.031600% |
| m2600/p05 | 148 | 101 | 807,476 | 21,398,530 | 3.773500% |
| m2600/p10 | 165 | 142 | 1,244,013 | 24,255,585 | 5.128800% |
| m2600/p20 | 166 | 148 | 1,631,593 | 24,739,769 | 6.595000% |
| m2600/p40 | 198 | 168 | 1,465,677 | 29,379,590 | 4.988800% |
| m2700/p02 | 243 | 231 | 3,212,227 | 37,526,792 | 8.559800% |
| m2700/p05 | 203 | 182 | 2,644,977 | 31,211,589 | 8.474300% |
| m2700/p10 | 297 | 279 | 3,772,154 | 45,246,852 | 8.336800% |
| m2700/p20 | 248 | 230 | 2,876,497 | 37,821,691 | 7.605400% |
| m2700/p40 | 266 | 234 | 2,324,544 | 39,548,490 | 5.877700% |

## Top 20 runs by dropped vehicle-ticks

| run key | n_ticks | n_vehicles | dropped veh-ticks | rostered veh-ticks | pool-full ticks |
|---|---|---|---|---|---|
| walls_pilot_chunk4/w8659m2700p10t420__w-mid-early | 1,201 | 456 | 73,914 | 141,035 | 802 |
| walls_pilot_chunk4/w8662m2600p10t300__w-mid-early | 1,201 | 483 | 64,989 | 144,820 | 780 |
| chunk1/w8457m2700p20t540__switching2 | 1,201 | 493 | 63,956 | 153,600 | 1,200 |
| walls_pilot_chunk4/w8663m2700p10t480__w-mid-late | 1,201 | 500 | 62,709 | 151,493 | 841 |
| chunk1/w8457m2700p20t540__native | 1,201 | 512 | 58,511 | 153,600 | 1,200 |
| chunk3/w8585m2700p10t480__switching1 | 1,201 | 496 | 56,500 | 153,600 | 1,200 |
| chunk2/w8512m2700p10t540__pi | 1,201 | 488 | 55,765 | 153,600 | 1,200 |
| chunk3/w8585m2700p10t480__switching2 | 1,201 | 502 | 53,115 | 153,600 | 1,200 |
| chunk3/w8585m2700p10t480__pi | 1,201 | 501 | 52,148 | 153,600 | 1,200 |
| chunk3/w8585m2700p10t480__fixed2 | 1,201 | 500 | 51,596 | 153,600 | 1,200 |
| chunk3/w8585m2700p10t480__native | 1,201 | 508 | 51,304 | 153,600 | 1,200 |
| chunk2/w8512m2700p10t540__switching3 | 1,201 | 504 | 51,078 | 153,600 | 1,200 |
| truearms_chunk3/w8585m2700p10t480__native | 1,201 | 515 | 50,495 | 153,600 | 1,200 |
| chunk1/w8473m2700p40t540__base | 2,281 | 774 | 49,782 | 257,570 | 1,393 |
| chunk3/w8585m2700p10t480__fixed1 | 1,201 | 511 | 49,540 | 153,600 | 1,200 |
| chunk1/w8433m2700p05t480__pi | 1,201 | 504 | 48,988 | 153,600 | 1,200 |
| truearms_chunk3/w8585m2700p10t480__pi | 1,201 | 507 | 48,625 | 153,600 | 1,200 |
| chunk2/w8523m2600p20t540__switching2 | 1,201 | 501 | 48,519 | 153,600 | 1,200 |
| walls_pilot_chunk4/w8659m2700p10t420__w-lo-early | 1,201 | 439 | 47,968 | 137,890 | 741 |
| walls_pilot_chunk4/w8667m2700p10t420__w-mid-late | 1,201 | 497 | 47,191 | 143,324 | 629 |

Per-run counts (counts only) are in `SLOT_POOL_CENSUS.json` under `runs`.
