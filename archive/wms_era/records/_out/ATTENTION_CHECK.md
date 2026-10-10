# Attention check: does the C1 WM-S transformer look at the car ahead?

Generated 2026-10-09T18:08:34Z. CPU only; no training, no SUMO, no model-code change (forward hooks on `residual_net.blocks[b].attn`).

## Answer in five lines

1. Not a driver-like forward gaze: only 6.4% of the attention goes to the car directly ahead (9.6% of what is left after the car's own slot), less than a uniform read of its 13 candidates would give (9.2%); 59.9% goes to cars in the neighbouring lane(s) (nearest one ahead and nearest one behind ~20% each) and 18.3% to itself.

2. Cars behind matter as much as cars ahead: 42% of the weight is on cars behind vs 40% ahead; 38% is on cars >50 m away (15% beyond 100 m); distance does taper it: 47% of the non-self weight is >50 m away vs 70% of the candidates.

3. Situations change it little: leader weight 5.2% in free flow -> 8.7% when following (the one sensible shift) -> 5.3% in a jam; main-road cars about to change lane do not look more at the other lane, ramp cars about to merge do (main-lane weight 78% vs 62% for ramp-zone cars not merging, n=102 query-steps); in dense traffic it spreads over more cars (effective 3.4 -> 7.0) but less than in proportion to the candidates (8 -> 24).

4. Heads: lead-car heads L1H0, L1H3, L2H0 (L1H3 up to 34% on the leader when following); who-is-behind head L1H2; neighbouring-lane readers L0H1, L0H2, L1H1, L2H1; mostly-self heads L0H0, L0H3, L2H3; long-range readers L1H1, L1H2, L2H1.

5. Verdict: a MOBIL-like neighbourhood reader (self + nearest lead/lag in the adjacent lanes + followers), not a car-following reader; plausible for a residual on top of an IDM/MOBIL physics base, but not "mostly the car ahead". Attention is where information is read from, not causal importance.


## 0. Setup

* Checkpoint: `WM-S-C1FAST-20261006/box/full_struct/wm_s.pt` (sha256 `87ed02e0707e5e4e...`, staged copy verified byte-identical), structured WM-S, flags from its TRAIN_RECEIPT (residual rollout, no lane head, no driver history, entries OFF, receipt physics).
* Code: worktree `E:/rd-wt/ws-attn`, branch `wms/attn-check`, base commit `21219a1798f8`. Network: `HdvResidualNet`, 3 `_KinBlock`, 4 heads, d_model 128; keys = self + vehicles within 150 m in the same or an adjacent lane (lane index difference <= 1) + learned pair bias on (ds, dlane, dv, |ds|, same-lane, self).
* Rollout: `WmsCheckpointModel.predict_states`, i.e. exactly the `main2_score.py` / `wall_c1_score.py` path (CAVs teacher-forced from the recorded arm, entries off), 60 s, seed 0. Reproduction check against the stored harness scores for the first two states: w8660m2700p10t300__w-lo-early identical (max |diff| 0.0, 45 values at 10/30/60 s); w8663m2700p10t480__w-lo-early identical (max |diff| 0.0, 45 values at 10/30/60 s).
* States: 40 rollouts = first state per world (sorted by world, t0_s, key), first 20 worlds, from `seen_controls.json` (20 states), `wall_same.json` (20 states). 40 (set, world) pairs; penetration values present: [10, 20, 40]. For `wall_same` the teacher arm is the record's own arm (`wall`); for `seen_controls` it is `arm0`.
* Sampling: attention captured at every 5th 0.5 s step (every 2.5 s; owner suggested every 10th) = 960 sampled (state, step) pairs, 61129 HDV query-steps in total. Queries = every present non-CAV vehicle; keys include CAVs and HDVs.
* Sanity: attention mass outside the candidate set = 0.0e+00; max |row sum - 1| = 3.0e-07; 105 isolated queries (no neighbour at all) are excluded from the "excluding self" columns; every candidate receives non-zero mass.
* Roles (relative to the query at that step; `ds = s_key - s_query`, ds > 0 = ahead): self; **leader** = nearest same-lane vehicle ahead; ahead_2nd/3rd/4th+ = further same-lane ahead; **follower** = nearest same-lane behind; behind_2nd+ = other same-lane behind; for each adjacent lane the nearest vehicle ahead / behind is `adj_*_near`, all others `adj_*_far`. "Behind (any lane)" = all non-self keys with ds <= 0. Percentages are the mean over query-steps of the attention mass (each query's attention row sums to 100% including self).
* Situations of the query (speed v, bumper gap to its same-lane leader, 5 m vehicle length): **free** v > 20 m/s and gap > 50 m (no leader within 150 m counts as free); **jam** v <= 5 m/s; **following** = everything else; **merge** = on-ramp lane 2 or s in [-50, 125.44] m (merge zone [0, 75.44] m +- 50 m); **lane change** = the vehicle's lane differs from its current lane at any of the next 4 steps (2 s). The speed classes partition all queries; merge / lane change overlay them, so the `*_plain` rows (outside merge, no lane change) give the clean car-following picture.

## 1. Where the mass goes (all HDV queries, pooled)

Average number of candidates a query can attend to: **13.5** including itself (**12.5** other vehicles; per role: leader 0.94, ahead_2nd 0.68, ahead_3rd 0.36, ahead_4th+ 1.52, follower 0.92, behind_2nd+ 2.47, adj_ahead_near 0.99, adj_ahead_far 1.83, adj_behind_near 0.96, adj_behind_far 1.79). CAVs are 20.9% of the other candidates.

| role | L0 | L1 | L2 | mean of 3 layers | uniform-over-candidates baseline | lift (mean / uniform) |
|---|---|---|---|---|---|---|
| self | 19.9% | 9.3% | 25.7% | **18.3%** | 10.6% | 1.73x |
| leader | 4.3% | 11.8% | 3.2% | **6.4%** | 9.2% | 0.70x |
| ahead_2nd | 1.8% | 2.9% | 1.2% | **2.0%** | 5.6% | 0.35x |
| ahead_3rd | 0.8% | 1.3% | 0.6% | **0.9%** | 2.0% | 0.45x |
| ahead_4th+ | 1.7% | 3.3% | 1.1% | **2.0%** | 6.4% | 0.32x |
| follower | 4.7% | 7.4% | 1.2% | **4.4%** | 9.0% | 0.49x |
| behind_2nd+ | 5.0% | 12.1% | 0.9% | **6.0%** | 13.5% | 0.45x |
| adj_ahead_near | 20.9% | 20.4% | 17.6% | **19.6%** | 9.4% | 2.08x |
| adj_ahead_far | 11.4% | 10.7% | 4.3% | **8.8%** | 12.7% | 0.69x |
| adj_behind_near | 19.0% | 11.3% | 29.7% | **20.0%** | 9.2% | 2.17x |
| adj_behind_far | 10.6% | 9.6% | 14.4% | **11.5%** | 12.4% | 0.93x |
| *behind, any lane (ds <= 0)* | 39.3% | 40.3% | 46.2% | **41.9%** | | |
| *ahead, any lane (ds > 0)* | 40.8% | 50.4% | 28.0% | **39.7%** | | |
| *> 50 m away* | 37.5% | 51.0% | 25.7% | **38.1%** | | |
| *> 100 m away* | 12.6% | 23.2% | 9.8% | **15.2%** | | |

Grouped (mean of layers and heads): self 18.3%; **same-lane ahead 11.3%** (leader alone 6.4%); same-lane behind 10.4%; **adjacent lanes 59.9%** (ahead 28.4%, behind 31.5%). Uniform baseline: ahead-same 23.2%, behind-same 22.4%, adjacent 43.8%, leader 9.2%. Excluding self (renormalised per query), the leader gets **9.6%** of the remaining mass. Mean |ds| of the attended non-self vehicles is **55 m**; effective number of attended vehicles (exp entropy, incl. self) is **4.7** of 13.5 candidates.

95% bootstrap CIs over the 40 states (layer/head mean): self [17.6%, 19.1%], leader [6.3%, 6.6%], leader_ex_self [9.2%, 10.0%], ahead_same_lane [10.7%, 11.9%], behind_same_lane [10.1%, 10.7%], adjacent_lane [59.2%, 60.7%], behind_any [41.4%, 42.5%], far50 [37.4%, 38.6%], far100 [15.0%, 15.5%]; mean |ds| ex-self [54, 55] m; effective N [4.5, 5.0].

Attention mass by signed distance (non-self keys; mean over layers and heads), vs the share of candidates in that bin:

| bin | ahead 0-20 | ahead 20-50 | ahead 50-100 | ahead 100-150 | behind 0-20 | behind 20-50 | behind 50-100 | behind 100-150 |
|---|---|---|---|---|---|---|---|---|
| mass (incl. self in denominator) | 11.4% | 12.0% | 11.2% | 5.2% | 10.4% | 9.8% | 11.7% | 10.1% |
| candidate share (of non-self) | 5.5% | 9.7% | 18.8% | 16.7% | 5.2% | 9.3% | 18.2% | 16.5% |

## 2. Per block and head (all HDV queries)

Mass in percent; `adj ahead/behind` = both adjacent lanes; `behind any` = ds <= 0, any lane; `far>50` = non-self keys more than 50 m away; `effN` = exp(entropy) of the row; `leader ex-self` = leader share after removing self.

| block / head | self | leader | ahead 2nd-3rd | ahead 4th+ | follower | behind 2nd+ | adj ahead | adj behind | behind any | far>50 | effN | leader ex-self |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0H0 | 28.7% | 9.4% | 5.9% | 4.8% | 9.9% | 13.0% | 13.8% | 14.6% | 37.5% | 41.7% | 8.4 | 14.7% |
| L0H1 | 16.9% | 2.0% | 1.2% | 0.5% | 2.0% | 1.9% | 39.8% | 35.8% | 39.7% | 40.2% | 5.9 | 3.4% |
| L0H2 | 2.9% | 4.3% | 2.4% | 1.2% | 6.3% | 4.8% | 38.0% | 40.0% | 51.1% | 41.2% | 6.8 | 4.7% |
| L0H3 | 31.2% | 1.3% | 0.8% | 0.2% | 0.6% | 0.5% | 37.5% | 27.8% | 28.8% | 26.8% | 3.4 | 5.0% |
| **L0 mean** | 19.9% | 4.3% | 2.6% | 1.7% | 4.7% | 5.0% | 32.3% | 29.6% | 39.3% | 37.5% | 6.1 | 7.0% |
| L1H0 | 20.2% | 15.0% | 1.9% | 0.4% | 1.2% | 0.2% | 42.3% | 18.9% | 20.3% | 26.7% | 2.8 | 25.9% |
| L1H1 | 0.2% | 4.0% | 4.5% | 7.0% | 1.7% | 4.8% | 39.2% | 38.6% | 45.1% | 75.2% | 7.2 | 4.0% |
| L1H2 | 15.9% | 1.4% | 1.2% | 1.9% | 23.0% | 42.6% | 1.7% | 12.3% | 77.9% | 73.1% | 5.2 | 3.9% |
| L1H3 | 0.8% | 26.8% | 9.2% | 4.0% | 3.5% | 0.9% | 41.2% | 13.5% | 17.9% | 29.0% | 4.4 | 27.0% |
| **L1 mean** | 9.3% | 11.8% | 4.2% | 3.3% | 7.4% | 12.1% | 31.1% | 20.8% | 40.3% | 51.0% | 4.9 | 15.2% |
| L2H0 | 27.2% | 12.3% | 6.9% | 4.2% | 3.1% | 2.0% | 32.9% | 11.3% | 16.3% | 19.4% | 5.5 | 18.4% |
| L2H1 | 5.0% | 0.4% | 0.2% | 0.1% | 1.2% | 1.6% | 15.6% | 75.9% | 78.7% | 66.3% | 3.2 | 1.0% |
| L2H2 | 24.9% | 0.2% | 0.1% | 0.0% | 0.3% | 0.1% | 22.6% | 51.7% | 52.1% | 13.5% | 2.4 | 2.0% |
| L2H3 | 45.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 16.5% | 37.7% | 37.7% | 3.6% | 1.8 | 5.1% |
| **L2 mean** | 25.7% | 3.2% | 1.8% | 1.1% | 1.2% | 0.9% | 21.9% | 44.2% | 46.2% | 25.7% | 3.2 | 6.6% |
| **all mean** | 18.3% | 6.4% | 2.9% | 2.0% | 4.4% | 6.0% | 28.4% | 31.5% | 41.9% | 38.1% | 4.7 | 9.6% |

## 3. By situation of the query (mean over layers and heads)

| situation | n query-steps | cand. (incl. self) | self | leader | ahead 2nd-3rd | ahead 4th+ | follower | behind 2nd+ | adj ahead | adj behind | behind any | far>50 | effN | leader ex-self | uniform leader |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 61129 | 13.5 | 18.3% | 6.4% | 2.9% | 2.0% | 4.4% | 6.0% | 28.4% | 31.5% | 41.9% | 38.1% | 4.7 | 9.6% | 9.2% |
| free | 26861 | 8.4 | 25.0% | 5.2% | 0.9% | 0.1% | 5.2% | 3.0% | 29.9% | 30.7% | 38.9% | 38.7% | 3.5 | 9.2% | 11.8% |
| follow | 22306 | 14.0 | 16.4% | 8.5% | 3.5% | 1.5% | 3.9% | 6.5% | 28.9% | 30.8% | 41.3% | 37.3% | 5.0 | 11.9% | 8.6% |
| jam | 11962 | 23.9 | 7.0% | 5.3% | 6.2% | 7.4% | 3.4% | 11.9% | 24.2% | 34.7% | 49.9% | 37.9% | 7.1 | 6.2% | 4.7% |
| merge | 9310 | 22.1 | 9.7% | 6.2% | 4.6% | 2.6% | 3.3% | 11.5% | 27.9% | 34.3% | 49.0% | 39.8% | 6.7 | 7.8% | 5.0% |
| lc_any | 3818 | 15.9 | 18.6% | 7.0% | 4.2% | 2.3% | 4.3% | 7.4% | 27.5% | 28.7% | 40.4% | 36.9% | 5.7 | 10.3% | 7.2% |
| lc_main | 3716 | 16.0 | 18.8% | 7.1% | 4.3% | 2.3% | 4.3% | 7.5% | 27.4% | 28.2% | 40.0% | 36.9% | 5.6 | 10.5% | 7.4% |
| lc_ramp | 102 | 15.8 | 12.0% | 2.2% | 0.4% | 0.0% | 4.7% | 3.2% | 29.3% | 48.2% | 56.1% | 37.6% | 6.5 | 2.8% | 2.0% |
| no_lc | 57311 | 13.3 | 18.3% | 6.4% | 2.8% | 2.0% | 4.4% | 5.9% | 28.5% | 31.7% | 42.0% | 38.1% | 4.7 | 9.5% | 9.4% |
| free_plain | 25183 | 8.1 | 24.9% | 5.2% | 0.8% | 0.1% | 5.3% | 3.0% | 29.8% | 30.8% | 39.2% | 38.5% | 3.4 | 9.1% | 12.0% |
| follow_plain | 16704 | 12.5 | 17.4% | 8.7% | 3.2% | 1.4% | 4.1% | 5.7% | 28.8% | 30.8% | 40.6% | 36.8% | 4.7 | 12.4% | 9.4% |
| jam_plain | 6859 | 23.5 | 7.3% | 5.3% | 6.5% | 9.8% | 3.3% | 10.2% | 23.8% | 33.8% | 47.3% | 38.0% | 7.0 | 6.3% | 5.0% |
| plain | 48746 | 11.8 | 19.9% | 6.4% | 2.4% | 1.9% | 4.6% | 5.0% | 28.6% | 31.2% | 40.8% | 37.8% | 4.3 | 9.8% | 10.1% |
| merge_no_lc | 8565 | 22.0 | 9.4% | 6.2% | 4.7% | 2.7% | 3.2% | 11.5% | 27.9% | 34.3% | 49.1% | 40.0% | 6.6 | 7.9% | 5.0% |
| lc_main_nomerge | 3073 | 14.3 | 20.1% | 7.3% | 4.4% | 2.4% | 4.4% | 6.6% | 27.3% | 27.5% | 38.5% | 36.5% | 5.3 | 11.1% | 7.9% |

Definitions: `all` = every HDV query; `free` = free flow: speed > 20 m/s and bumper gap to leader > 50 m (or no leader); `follow` = following: not free, speed > 5 m/s; `jam` = slow / jam: speed <= 5 m/s; `merge` = in or near merge zone: on-ramp lane 2, or s in [-50, 125.44] m; `lc_any` = query changes lane within the next 2 s (4 steps); `lc_main` = lane change within 2 s, query starts in lane 0/1; `lc_ramp` = lane change within 2 s, query starts on ramp lane 2 (the merge); `no_lc` = no lane change within the next 2 s; `free_plain` = free flow, outside merge zone, no lane change in 2 s; `follow_plain` = following, outside merge zone, no lane change in 2 s; `jam_plain` = slow/jam, outside merge zone, no lane change in 2 s; `plain` = free + following + jam, all outside the merge zone and without a lane change in 2 s; `merge_no_lc` = in/near merge zone, no lane change in 2 s; `lc_main_nomerge` = lane change within 2 s, mainline lanes, outside merge zone.

Bootstrap 95% CIs over states (layer/head mean) for the key situations:

| situation | self | leader | leader ex-self | ahead same lane | behind same lane | adjacent lanes | behind any | far>50 | effN |
|---|---|---|---|---|---|---|---|---|---|
| all | [17.6%, 19.1%] | [6.3%, 6.6%] | [9.2%, 10.0%] | [10.7%, 11.9%] | [10.1%, 10.7%] | [59.2%, 60.7%] | [41.4%, 42.5%] | [37.4%, 38.6%] | [4.5, 5.0] |
| free_plain | [24.2%, 25.8%] | [5.0%, 5.5%] | [8.5%, 9.7%] | [5.8%, 6.4%] | [8.2%, 8.5%] | [59.5%, 61.6%] | [38.5%, 39.7%] | [38.0%, 39.0%] | [3.3, 3.5] |
| follow_plain | [16.5%, 18.2%] | [8.4%, 8.9%] | [11.6%, 13.3%] | [12.9%, 13.7%] | [9.5%, 10.2%] | [58.6%, 60.5%] | [39.8%, 41.3%] | [36.0%, 37.5%] | [4.6, 4.9] |
| jam_plain | [6.2%, 8.6%] | [5.0%, 5.8%] | [5.7%, 7.1%] | [20.5%, 22.5%] | [13.0%, 13.9%] | [56.4%, 58.9%] | [45.8%, 48.5%] | [35.7%, 39.9%] | [6.6, 7.3] |
| merge_no_lc | [8.6%, 10.5%] | [5.9%, 6.6%] | [7.4%, 8.4%] | [13.0%, 14.1%] | [14.0%, 15.5%] | [61.2%, 63.0%] | [47.8%, 50.4%] | [39.1%, 40.8%] | [6.2, 7.1] |
| lc_main_nomerge | [18.6%, 21.7%] | [6.9%, 7.6%] | [10.1%, 12.1%] | [13.5%, 14.6%] | [10.3%, 11.6%] | [53.7%, 55.8%] | [36.6%, 40.1%] | [34.9%, 38.0%] | [4.9, 5.8] |
| lc_ramp | [10.5%, 13.7%] | [1.3%, 3.2%] | [1.7%, 4.0%] | [1.5%, 3.7%] | [6.1%, 9.6%] | [75.9%, 79.4%] | [52.0%, 60.0%] | [35.2%, 40.1%] | [5.6, 7.6] |

## 4. Which heads specialise (head x situation)

**Leader mass (incl. self in denominator)**

| head | free_plain (n=25183) | follow_plain (n=16704) | jam_plain (n=6859) | merge_no_lc (n=8565) | lc_main_nomerge (n=3073) | lc_ramp (n=102) | all |
|---|---|---|---|---|---|---|---|
| L0H0 | 9.5% | 12.0% | 7.3% | 5.9% | 9.3% | 0.9% | 9.4% |
| L0H1 | 2.3% | 2.5% | 0.8% | 1.0% | 2.8% | 0.5% | 2.0% |
| L0H2 | 3.1% | 5.6% | 2.4% | 6.8% | 4.6% | 0.3% | 4.3% |
| L0H3 | 1.4% | 1.6% | 0.5% | 1.4% | 1.6% | 0.0% | 1.3% |
| L1H0 | 11.5% | 25.6% | 4.7% | 11.0% | 19.4% | 13.5% | 15.0% |
| L1H1 | 5.0% | 3.0% | 3.9% | 3.5% | 3.9% | 2.1% | 4.0% |
| L1H2 | 1.4% | 1.5% | 1.9% | 0.6% | 2.0% | 0.1% | 1.4% |
| L1H3 | 21.5% | 34.0% | 25.8% | 29.8% | 26.2% | 3.9% | 26.8% |
| L2H0 | 6.8% | 17.7% | 15.8% | 13.9% | 15.5% | 4.5% | 12.3% |
| L2H1 | 0.4% | 0.6% | 0.5% | 0.3% | 0.8% | 0.0% | 0.4% |
| L2H2 | 0.0% | 0.3% | 0.5% | 0.2% | 0.9% | 0.1% | 0.2% |
| L2H3 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

**Mass on cars behind (any lane, ds <= 0)**

| head | free_plain (n=25183) | follow_plain (n=16704) | jam_plain (n=6859) | merge_no_lc (n=8565) | lc_main_nomerge (n=3073) | lc_ramp (n=102) | all |
|---|---|---|---|---|---|---|---|
| L0H0 | 29.6% | 34.0% | 43.9% | 60.2% | 37.2% | 81.8% | 37.5% |
| L0H1 | 41.7% | 41.9% | 33.9% | 34.7% | 39.0% | 22.2% | 39.7% |
| L0H2 | 55.8% | 51.1% | 48.4% | 40.1% | 52.8% | 45.2% | 51.1% |
| L0H3 | 21.2% | 24.1% | 36.2% | 53.8% | 26.3% | 61.9% | 28.8% |
| L1H0 | 16.4% | 17.8% | 44.5% | 18.1% | 18.6% | 27.3% | 20.3% |
| L1H1 | 48.2% | 46.9% | 27.6% | 45.8% | 46.3% | 70.1% | 45.1% |
| L1H2 | 76.8% | 78.8% | 66.1% | 90.5% | 70.4% | 88.4% | 77.9% |
| L1H3 | 19.3% | 14.2% | 17.9% | 20.0% | 18.1% | 47.6% | 17.9% |
| L2H0 | 14.6% | 13.7% | 18.2% | 24.4% | 14.8% | 39.8% | 16.3% |
| L2H1 | 77.1% | 78.6% | 85.3% | 80.2% | 71.7% | 91.4% | 78.7% |
| L2H2 | 44.2% | 50.6% | 74.8% | 63.9% | 42.5% | 49.6% | 52.1% |
| L2H3 | 25.2% | 35.5% | 70.4% | 57.6% | 23.9% | 48.1% | 37.7% |

**Mass on adjacent-lane vehicles (ahead + behind)**

| head | free_plain (n=25183) | follow_plain (n=16704) | jam_plain (n=6859) | merge_no_lc (n=8565) | lc_main_nomerge (n=3073) | lc_ramp (n=102) | all |
|---|---|---|---|---|---|---|---|
| L0H0 | 31.9% | 28.7% | 12.4% | 30.2% | 28.3% | 66.7% | 28.4% |
| L0H1 | 68.9% | 74.3% | 89.7% | 89.3% | 66.5% | 95.8% | 75.5% |
| L0H2 | 82.1% | 77.8% | 80.7% | 65.9% | 75.2% | 98.7% | 78.1% |
| L0H3 | 54.9% | 63.0% | 89.3% | 81.3% | 62.5% | 94.8% | 65.3% |
| L1H0 | 55.1% | 56.3% | 88.0% | 70.4% | 53.8% | 32.9% | 61.2% |
| L1H1 | 89.4% | 82.4% | 39.4% | 63.2% | 82.3% | 87.4% | 77.7% |
| L1H2 | 14.2% | 14.7% | 3.4% | 20.2% | 13.3% | 80.7% | 14.0% |
| L1H3 | 74.2% | 51.6% | 14.1% | 36.5% | 51.6% | 91.0% | 54.7% |
| L2H0 | 55.8% | 45.7% | 8.6% | 38.6% | 36.5% | 28.3% | 44.2% |
| L2H1 | 91.5% | 90.9% | 95.3% | 90.7% | 87.8% | 98.1% | 91.5% |
| L2H2 | 68.2% | 74.3% | 90.1% | 84.6% | 61.1% | 79.1% | 74.3% |
| L2H3 | 41.3% | 54.9% | 80.6% | 75.2% | 38.4% | 76.9% | 54.2% |

**Self mass**

| head | free_plain (n=25183) | follow_plain (n=16704) | jam_plain (n=6859) | merge_no_lc (n=8565) | lc_main_nomerge (n=3073) | lc_ramp (n=102) | all |
|---|---|---|---|---|---|---|---|
| L0H0 | 42.2% | 27.4% | 9.5% | 10.9% | 22.4% | 5.1% | 28.7% |
| L0H1 | 23.8% | 16.5% | 5.7% | 5.8% | 19.6% | 3.2% | 16.9% |
| L0H2 | 4.2% | 2.6% | 0.8% | 1.6% | 2.7% | 0.3% | 2.9% |
| L0H3 | 42.6% | 33.2% | 7.6% | 14.1% | 31.8% | 5.0% | 31.2% |
| L1H0 | 32.7% | 14.2% | 0.8% | 13.5% | 13.6% | 32.5% | 20.2% |
| L1H1 | 0.4% | 0.1% | 0.3% | 0.1% | 0.1% | 0.4% | 0.2% |
| L1H2 | 19.6% | 15.1% | 15.7% | 6.2% | 19.7% | 7.9% | 15.9% |
| L1H3 | 1.4% | 0.4% | 0.9% | 0.1% | 0.6% | 0.1% | 0.8% |
| L2H0 | 35.9% | 24.2% | 16.4% | 17.1% | 26.8% | 45.6% | 27.2% |
| L2H1 | 5.9% | 4.7% | 2.3% | 4.6% | 6.8% | 0.7% | 5.0% |
| L2H2 | 31.7% | 25.0% | 8.2% | 14.5% | 35.8% | 20.5% | 24.9% |
| L2H3 | 58.7% | 45.1% | 19.4% | 24.8% | 61.6% | 23.1% | 45.8% |

**Effective number of attended vehicles (exp entropy)**

| head | free_plain (n=25183) | follow_plain (n=16704) | jam_plain (n=6859) | merge_no_lc (n=8565) | lc_main_nomerge (n=3073) | lc_ramp (n=102) | all |
|---|---|---|---|---|---|---|---|
| L0H0 | 5.8 | 8.6 | 13.0 | 12.2 | 8.0 | 9.0 | 8.4 |
| L0H1 | 5.2 | 6.4 | 4.7 | 7.1 | 6.8 | 5.9 | 5.9 |
| L0H2 | 4.6 | 6.8 | 7.8 | 11.6 | 8.0 | 7.5 | 6.8 |
| L0H3 | 3.0 | 3.7 | 3.5 | 3.8 | 3.9 | 8.3 | 3.4 |
| L1H0 | 2.1 | 2.8 | 4.1 | 3.3 | 3.7 | 2.7 | 2.8 |
| L1H1 | 4.2 | 6.6 | 14.1 | 11.0 | 7.4 | 12.4 | 7.2 |
| L1H2 | 3.1 | 4.6 | 10.3 | 8.6 | 4.9 | 7.9 | 5.2 |
| L1H3 | 2.6 | 4.2 | 8.4 | 6.2 | 5.5 | 4.7 | 4.4 |
| L2H0 | 3.5 | 5.5 | 10.7 | 6.7 | 6.7 | 4.2 | 5.5 |
| L2H1 | 2.9 | 3.5 | 2.6 | 3.8 | 3.8 | 8.8 | 3.2 |
| L2H2 | 1.9 | 2.4 | 2.6 | 3.3 | 3.0 | 4.5 | 2.4 |
| L2H3 | 1.6 | 1.8 | 2.0 | 2.2 | 2.1 | 2.6 | 1.8 |

## 5. CAV versus HDV as attended vehicle

Of the attention mass on other vehicles (non-self), **21.7%** goes to CAV keys, versus a CAV share of 20.9% among the non-self candidates (about 21.9% of slots are CAVs). The net has no CAV flag, so any difference reflects kinematics / position only.

| role | CAV share of that role's mass |
|---|---|
| leader | 21.1% |
| ahead_2nd | 21.4% |
| ahead_3rd | 22.1% |
| ahead_4th+ | 20.9% |
| follower | 21.1% |
| behind_2nd+ | 18.6% |
| adj_ahead_near | 22.7% |
| adj_ahead_far | 23.1% |
| adj_behind_near | 22.2% |
| adj_behind_far | 20.3% |

## 6. Consistency between the two state sets

| set | n query-steps | self | leader | leader ex-self | ahead same lane | behind same lane | adjacent lanes | behind any | far>50 | effN |
|---|---|---|---|---|---|---|---|---|---|---|
| seen_controls | 31486 | 19.3% | 6.6% | 10.4% | 11.9% | 10.7% | 58.0% | 41.4% | 37.7% | 4.8 |
| wall_same | 29643 | 17.3% | 6.2% | 8.8% | 10.7% | 10.1% | 62.0% | 42.5% | 38.5% | 4.7 |

## 7. Plain reading: the numbers behind the five lines

Quoted from the tables above (head-label thresholds: leader_head >= 10%, rear_same_lane_head >= 40%, adjacent_reader_head >= 75%, self_head >= 28%, far_reader_head >= 65%).

1. It does not behave like a forward-gazing driver: the same-lane leader gets 6.4% of the mass (9.6% once self is removed, lift 0.70x vs a uniform read of the 13.5 candidates), all same-lane cars ahead 11.3% (uniform 23.2%); 59.9% goes to adjacent-lane cars, concentrated on the nearest one ahead (19.6%, 2.1x its candidate share) and behind (20.0%, 2.2x) - the MOBIL new-leader / new-follower pair - plus 18.3% on itself.

2. Cars behind: 41.9% of all mass (51.3% of the non-self mass) sits on vehicles behind the query, as much as ahead (39.7%); far cars: 38.1% is >50 m away (46.6% of non-self mass vs 70.2% of candidates) and 15.2% >100 m (18.6% vs 33.2%); mean attended distance 55 m, effective 4.7 vehicles of 13.5 candidates. So it is distance-tapered but symmetric front/back, not forward-focused.

3. Situation effects are modest: leader mass rises from 5.2% (free flow) to 8.7% when following (9.1% -> 12.4% ex-self) and drops to 5.3% in a jam, where mass moves to farther same-lane cars; effective N grows 3.4 (free) -> 7.0 (jam) / 6.6 (merge zone) but less than in proportion to the candidate count (8 -> 24); mainline cars about to change lane do not shift toward the adjacent lane (54.8% vs 59.8% plain), only ramp cars about to merge do (77.5% adjacent vs 62.2% in the merge zone without a lane change; only 102 query-steps).

4. Heads specialise: leader/forward heads L1H0 15.0%, L1H3 26.8%, L2H0 12.3% (L1H3 reaches 34.0% when following); rear-looking same-lane head L1H2 65.6% (follower + others behind); adjacent-lane readers L0H1 75.5%, L0H2 78.1%, L1H1 77.7%, L2H1 91.5%; self/identity heads L0H0 28.7%, L0H3 31.2%, L2H3 45.8%; long-range (>50 m) readers L1H1 75.2%, L1H2 73.1%, L2H1 66.3%. Layer 1 is the longitudinal layer (leader 11.8%, same-lane behind 19.5%), layer 2 is local and low-entropy (effN 3.2, adjacent-behind 44.2%).

5. Verdict: it reads a MOBIL-like neighbourhood (self, nearest adjacent-lane lead/lag, followers, a few leader heads that sharpen when following) rather than a human's forward gaze. At the layer-mean level the split is nearly situation-independent (adjacent lanes stay 54-63% in every situation) although single heads re-allocate (e.g. L1H1 adjacent-lane mass 89.4% in free flow vs 39.4% in a jam). Leader-following is already handled by the physics base, so the residual net is not required to re-read the leader. Attention weights are not causal importance: they show where information is read from, not what drives the correction.

Caveat: attention weights show where information is read from, not how much it changes the output; the residual net is added on top of the physics base (IDM + MOBIL), which already encodes car-following on the leader, so the net's job is a correction and it need not re-read the leader. Weights are also averaged over three stacked layers whose later layers attend to already-mixed representations.
