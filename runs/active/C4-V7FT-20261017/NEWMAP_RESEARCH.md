# NEWMAP_RESEARCH: a map where group cooperation can happen (phase newmap_probe, 2026-11-06)

**Plain summary (15 lines)**
1. Most promising map: **wz3to2**, a 3-lane freeway whose right lane closes at x = 2,700 m (work-zone lane drop). The weave and 2-to-1 closure maps showed no group synergy.
2. On wz3to2, 3-CAV walls (one CAV per lane, abreast) gave **8 admitted cooperative cases out of 34** (17 situations x 2 wall rules; LB95 of benefit and of synergy both > 0, no collisions), from 4 of 7 worlds.
3. Admitted synergy was +613 to +5,623 veh-s per 360 s window (2-26% of the window delay). The current merge map gave about +250 to +490 on a single situation.
4. Mechanism check: in both tested cases one 2-CAV pair (always including the closing-lane CAV) reproduces the whole group effect. Example: the lane-0 and lane-1 CAVs each ADD +3,522 / +3,191 veh-s of delay alone, but together they SAVE 2,963.
5. This is a state-dependent opportunity, not an average win. Mean synergy per decision is -247 veh-s [95% CI -1,432, +938] for the gentle wall W2 (22 m/s, 120 s) and -1,421 for W1 (15 m/s, 90 s).
6. The W2 mean group benefit is +1,898 [+77, +3,718], but the best single CAV usually captures it. A learned policy or selector must choose when to form the wall.
7. No synergy elsewhere: 2-to-1 walls 0/9 (mean synergy -1,444, CI below 0), weave walls 0/7, gap-maker + merger pairs 0/12 on all maps. SUMO native merging already does the gap work.
8. Caveat: the dynamics are near-chaotic. A 1-s, -0.5 m/s^2 pulse moved delay by more than 300 veh-s in 3 of 19 cases. All 8 admitted cases survived one common-dither replicate, but the dither had no effect in 4 of them, so admission counts are optimistic.
9. Budget: 148,520 of 150,000 sim-s on cpuV (12 workers or fewer, nice 19); seeds 989000-989006 from the new range 989000-989199. 359 jobs, 0 failed, 0 collisions, 0 teleports. The base was native CAVs, not MAPPO.
10. Migration cost, part 1: the stack is hard-wired to the merge geometry (LANE_TABLE; the 0-75.44 m merge-zone coordinate in 21 src/wms files; ramp = WM lane 2; RSU zone; scenario builder). Map abstraction and model-frame changes take about 3-4 days.
11. Migration cost, part 2: a new WM corpus plus retrain and fidelity qualification take about 2-4 days with GPU nights; this is the critical path and the highest risk.
12. Migration cost, part 3: retraining the speed-only MAPPO comparator costs about 7.0M sim-s (about 10 h on cpuQ). The coop pipeline (V2V layout reusable, checkpoints not) plus evaluation takes about 3 days. Total: about 9-15 working days.
13. Recommendation: do NOT migrate before Nov 15. Keep the paper on the current map, and add this SUMO-only lane-drop result (super-additive pairs, with its limits) as a short section on where multi-CAV cooperation is physically necessary.
14. After Nov 15: move to wz3to2, first with a SUMO-oracle learner or selector with a WALL / closing-lane partner role (about 3-5 days), then port the WM.
15. Evidence: cpuV /home/ubuntu/newmap20261106; local newmap_probe/results; maps in assets/research/newmap_candidates/{wz3to2,wz2to1,weave}.

---

## 1. Question, approval and compliance

- **Approval.** The owner asked to "create a new map where [group cooperation] can happen and research in that idea". This was approved as phase **newmap_probe**, capped at **150,000 sim-s**.
  - Mid-task, MAIN limited cpuV to **12 cores**, because clean3 was paused and the remaining cores were shared.
  - All later batches used at most 12 workers. The 9- and 10-worker calibration and prefix batches also stayed within that limit.
- **Definition (project).** The synergy definition is the project's own:
  - `S = min_i J(solo_i) − J(group)` and `T = J(base) − J(group)`.
  - J = team delay in veh-s over the common window [t0, t0+H): on-road speed deficit `dt·max(0, 1 − v/31.29)` plus source waiting (`c4.branching` definition).
  - Solos have no zero floor.
  - Per situation, a 30-s circular-block paired bootstrap of the per-second difference is used: B = 2000, RNG 20261103, one-sided 95% lower bound (`bank_statistics.paired_bootstrap`, per_seed_v1).
  - **Admitted** = LB95(T) > 0 AND LB95(S) > 0 AND the group arm is collision- and teleport-free.
  - Across situations: mean, t-based 95% CI, and a world-cluster bootstrap 95% CI.
- **Seeds.** A newly registered range, **989000–989199**, of which 989000–989006 were used. It was checked absent from REGISTRATION.md, `protocols/experiment_history.json` and CLEAN1_STATUS.md.
  - All random streams are map-namespaced (`newmap_requests_v1:{map}:{seed}`; HDV draws use `sample_driver("{map}:{seed}", id)`).
  - No 964000–964199, 962xxx or 977xxx seeds were used.
- **Hard rules.** No sealed, holdout, heldout, cal_split, chunk5, j4b or inc1 path was opened. No hetdata_truth or driver/vehicle truth file was read.
  - The HDV vType *definitions* come from a verbatim copy of `traffic_shepherd/c4/population.py` (SHA256 6af5ee23…).
  - Per-vehicle parameters were freshly drawn for the new seeds; no truth table was produced or read.
  - HDVs are never commanded.
  - Only new files were created; no existing asset was changed.
  - Only own PIDs were used (`PIDS.txt`, 359 entries); nothing was killed.
  - No git reset, clean, stash drop or push.

## 2. Literature: where multi-CAV cooperation is needed

*Verified in this session* means the title, venue and key claim were checked by web search or fetch during this task. *From memory* means a well-known reference that was not re-verified here.

| Setting | Why one CAV cannot capture the gain | Mechanism needed | Reported penetration and demand |
|---|---|---|---|
| **Lane drop / work-zone bottleneck (N→N−1)** | A single slowed CAV only meters its own lane; traffic in the other lanes passes it, and the lane changes add turbulence. A moving bottleneck's capacity depends on how many lanes it blocks (Daganzo & Laval 2005, TR-B 39, *from memory*). The capacity drop comes from forced merges at the taper (Chen & Ahn 2018, TR-B 108, *verified*). | Coordinated metering: a moving bottleneck or VSL formed by vehicles across lanes, then platoon release. Han, Chen & Ahn 2017, TR-B 98 (connected-vehicle VSL that avoids capacity drop at a fixed bottleneck, *verified*). Čičić, Xiong, Jin & Johansson 2022, IEEE T-ITS 23(7) (platoons as controlled moving bottlenecks; the **number of lanes occupied is a control input**; median delay −75.6% in simulation; arXiv 1907.13049, *verified*). Piacentini, Goatin & Ferrara 2018 (moving bottleneck of coordinated vehicles, *verified title*). | Vinitsky et al. 2018 ITSC (Bay-Bridge 4→2→1 bottleneck): **10% AVs give +200 veh/h (+25%)** at high inflow, comparable to ramp metering (*verified*). Vinitsky, Lichtle, Parvate & Bayen 2023, ACM TCPS (decentralized MARL): **+20% at 5% up to +33% at 40%** penetration (arXiv 2011.00120, *verified*). Capacity drop sets in at inflows of about 2,200 veh/h in that benchmark (as reported by a citing paper, arXiv 2208.00268). |
| **2→1 closure, zipper merge** | One CAV can open one gap. Platoon-level gap redistribution and the ordering of a stream of mergers need several vehicles. | Zipper ordering, gap redistribution and lane-change grouping: Chen, Srivastava & Ahn 2021, TR-C 123:102950 (*verified*). Cooperative late merge (CLMS), ASCE J. Transp. Eng. A, doi 10.1061/JTEPBS.0000632 (*verified title*). | Mostly mixed traffic with growing penetration. CLMS-type studies report the largest gains above about 50% (*fragmentary, unverified numbers*). |
| **Short acceleration-lane on-ramp at high demand** | The mainline CAV's gap creation is a single-vehicle action. Multi-CAV gains need several ramp vehicles or slots arranged at once. | Sequencing and slot assignment: Rios-Torres & Malikopoulos 2017 survey, IEEE T-ITS 18(5) (*from memory*). Xu et al. grouping 2018 (arXiv 1804.01250, per the clean2 report). Kumaravel, Malikopoulos & Ayyagari, platoon merging (ACC 2021). | **The current project map is already this case** (75.44 m acceleration lane). Coop probe and mining: native SUMO merging does the job; synergy is about 0. |
| **Weaving section** (on-ramp closely followed by an off-ramp) | Entering and exiting streams must cross. One CAV can yield, but swapping positions needs complementary roles in both streams. | Distributing lane-change positions and coordinated crossing. Tilg, Yang & Menendez 2018, TR-C 96:3–21: optimizing desired lane-change positions gives **up to +15% capacity** (single-lane mainline plus auxiliary lane, *verified*). "Optimizing operations at freeway weaves with CAVs", TR-C 2021 (*verified title*). Bi-level weave control, arXiv 2403.16225 (*verified title*). | Mostly high CAV shares. |
| **Multiple closely spaced on-ramps** | A gap created for ramp 1 can be consumed before ramp 2, so coordination across merges is needed. | Coordinated metering across ramps. HERO coordinated ramp metering: Papamichail et al. 2010 (*from memory*). | A web search found **no CAV-specific multi-ramp study**. This is a literature gap. |
| **Signalised corridor platooning** | The saturation-flow gain needs consecutive connected vehicles: a short headway is a property of a pair or string. | CACC platoon discharge. Lioris, Pedarsani, Tascikaraoglu & Varaiya 2017, TR-C 77:292–305: platoons can **double or triple** saturation flow (*verified*). | Assumes about full CAV/CACC. It needs short-headway CAV types, which conflicts with the project's single cav_typical vType rule, and it is urban, so it is out of scope. |
| **Contrast: single-vehicle wave damping** | Here a single vehicle *suffices*, which explains the merge-map finding. | Jam-absorption driving: Nishi et al. 2013, TR-B 50 (*from memory*). A ring-road AV: Stern et al. 2018, TR-C 89 (*from memory*). Flow: Wu et al., T-RO 2021 (*per the clean2 report*). | A single AV dampens waves. |

**Takeaway.** The literature points to **lane coverage**: a moving bottleneck or metering across all lanes, in front of a capacity-drop bottleneck. This is the one setting where the *group* is physically necessary. Gap creation and merge assistance are largely single-vehicle actions. That is consistent with the project's merge-map evidence, where only rare multi-CAV mainline walls were synergistic.

## 3. Candidate maps (new files only)

Location: `E:/research-dense/assets/research/newmap_candidates/<name>/`.
- Each folder holds `.nod/.edg/.con.xml`, `<name>.net.xml` (netconvert 1.27.1), `<name>.routes.xml`, `MAP_SPEC.json`, `DEMAND_CHOSEN.json`, and an illustrative `<name>.example.{rou.xml,sumocfg}` for sumo-gui viewing.
- Builder: `runs/active/C4-V7FT-20261017/newmap_probe/build_maps.py`.

| Map | Geometry | Probe demand (chosen by native calibration) |
|---|---|---|
| **wz3to2** | `up` 3 lanes × 2,696 m. Lane 0 has no connection at x = 2,696 (forced lane change). `wz` 2 lanes × 1,296 m. | Peak 2,800 veh/h (0.8× for 0–300 s, 1.0× for 300–1500 s, 0.6× for 1500–1800 s). There are intermittent breakdowns at the drop: approach speed 4–13 m/s from about 500 s, with partial recovery. At 3,400 and 4,000 veh/h the breakdown is persistent. |
| **wz2to1** | `up` 2 lanes × 2,696 m, then 1 lane × 1,296 m. | Peak 1,200 veh/h. SUMO's lane-end merging gives a severe capacity drop: discharge is about 770 veh/h at 1,400 demand, and the approach crawls at 1–2 m/s. |
| **weave** | `up` 2 lanes × 2,361 m. The on-ramp (315 m, 22.22 m/s) adds auxiliary lane 0 on `weave` (3 lanes, 471 m). Lane 0 diverges to the off-ramp. `down` 2 lanes × 961 m. 20% of mainline traffic exits. | Main 2,200 veh/h plus ramp 900 veh/h. The default insertion (departPos `base`) starved two-lane insertion: 35k–234k veh-s of source waiting. Config B (main departLane `free`, departPos `last`) fixed it. Weave breakdown is then gridlock-like in 3 of 4 worlds (approach speed 0.2–6 m/s). |

**Common to all maps (copied from the project):**
- 0.5 s physics, 1 s control, time-to-teleport −1, collision.action warn, departSpeed max.
- Poisson requests per stream; CAV identity drawn per vehicle at **penetration 0.3**.
- HDVs: one textbook IDM/LC2013 vType per vehicle (population medians: T 1.2 s, a 1.2, b 2.0, s0 2 m).
- CAVs: `cav_typical`, with speedMode 27 set on insertion.
- CAV commands: `slowDown(v, clip(v + a·1 s, 0, allowed), 1 s)`; release = `setSpeed(−1)`. Acting members use laneChangeMode 513 (LC_MODE_POLICY), restored on release.
- Mainline speed limit 31.29 m/s (the J normalisation).

## 4. Probe protocol

- **Prefixes.** For each world (map, seed), one native run from 0 to 1,120 s. SUMO states (`--save-state.rng`, precision 17) and CAV snapshots are saved every 10 s inside three windows: 400–520, 700–820 and 1000–1120 s.
  - Branch-from-state repeatability was verified: two identical base branches gave a bitwise-equal J series.
  - The native continuation does **not** equal a loaded-state base exactly (an observed 4 veh-s per 60 s mismatch). So **every arm, base included, is a branch from the same saved state**.
- **Situation selection.** Selection uses geometry only (V2V-observable) and is never outcome-based. In each window, take the earliest saved time with a valid group.
  - **Wall:** one CAV per upstream lane, 400–1,800 m before the bottleneck, longitudinal spread ≤ 60 m; minimum spread wins.
  - **Gap pair:**
    - lane-drop maps: a merger CAV in the closing lane 30–500 m before the drop, plus a gap-maker CAV in the adjacent through lane 0–40 m behind it;
    - weave: an entering CAV on aux lane 0, plus a gap-maker in weave lane 1 or `up` lane 0, 0–40 m behind it.
- **Arms per situation** (horizon H = 360 s for walls, 300 s for gaps): base (all native), group, and every member solo.
- **Rules:**
  - **Wall W1:** target speed 15 m/s for 90 s, braking ≤ 1.5 and accelerating ≤ 1.0 m/s², members keep their lane.
  - **Wall W2:** 22 m/s for 120 s, braking ≤ 1.0 m/s². Same situations and the same base arm as W1.
  - **Gap pair:** the gap-maker brakes at 1.5 m/s² for 3 s, then holds speed for 5 s. The merger holds speed and requests `changeLane` into the target lane for up to 10 s.
- **Robustness and mechanism checks** on wz3to2 and wz2to1 walls:
  - **Pulse noise floor:** one wall member at −0.5 m/s² for 1 s, 19 situations.
  - **Dither replicate:** each of the 8 admitted situations was re-run in all arms with one common extra pulse on the non-member CAV nearest the drop.
  - **Partial walls:** all 2-of-3 pairs in 2 admitted situations.

## 5. Results

### 5.1 Per map and family

One situation = one (world, decision time). S and T are in veh-s per window. The cluster CI resamples worlds.

| Map / family | n (worlds) | Admitted | S>0 / LB95(S)>0 | Mean S [t95] {cluster95} | Mean T [t95] | Mean best-solo gain |
|---|---|---|---|---|---|---|
| **wz3to2 wall W2** | 17 (7) | **5** | 7 / 5 | **−247 [−1,432, +938] {−1,195, +495}** | **+1,898 [+77, +3,718]** | +2,145 |
| wz3to2 wall W1 | 17 (7) | 3 | 6 / 4 | −1,421 [−3,198, +356] {−2,434, −295} | +1,128 [−1,567, +3,823] | +2,549 |
| wz3to2 gap pair | 4 (2) | 0 | 0 / 0 | −1,348 [−3,226, +530] | −346 | +1,002 |
| wz2to1 wall W1 | 9 (4) | 0 | 2 / 1 | −1,444 [−2,655, −233] | −516 [−2,838, +1,805] | +928 |
| wz2to1 gap pair | 2 (1) | 0 | 0 / 0 | −2,629 | +1,042 | +3,671 |
| weave wall W1 | 7 (3) | 0 | 2 / 0 | −114 [−369, +140], median 0 | +969 [−745, +2,683] | +1,084 |
| weave gap pair | 6 (2) | 0 | 3 / 0 | −2,474 [−5,907, +959] | −3,041 [−6,905, +823] | −567 |

**State split on wz3to2, by approach speed at t0 (an RSU-detector-like observable):**
- **Congested states** (< 10 m/s, n = 12):
  - W1: T = +3,081 [+937, +5,225], S = −586;
  - W2: T = +2,402 [+322, +4,481], S = −464.

  Walls help in congestion, but one member often captures most of it.
- **Free-flow states** (n = 5):
  - W1 is harmful: T = −3,559;
  - W2 is roughly neutral: T = +688, S = +272.
- **W2 vs W1 on the same 17 situations:** ΔT = +770 and ΔS = +1,174 per situation. The gentler wall is better.

### 5.2 Admitted cooperative situations on wz3to2

| World | t0 (s) | Variant | Best-solo gain | Group T | S (LB95) | S as % of J_base | Dither replicate S |
|---|---|---|---|---|---|---|---|
| 989001 | 440 | W1 | −352 (all three solos harmful: −3,522 / −3,191 / −352) | +3,181 | **+3,533 (+2,432)** | 26% | same (inert dither) |
| 989006 | 1090 | W1 | +5,207 | +6,708 | +1,500 (+906) | 2.2% | +1,288 |
| 989006 | 700 | W1 | +6,987 | +8,221 | +1,235 (+871) | 2.4% | same (inert) |
| 989000 | 1000 | W2 | +3,887 | +5,035 | +1,149 (+399) | 4.4% | same (inert) |
| 989001 | 440 | W2 | +1,601 | +2,213 | +613 (+248) | 4.5% | same (inert) |
| 989004 | 1090 | W2 | +3,702 | +5,519 | +1,817 (+773) | 3.3% | +1,297 |
| 989006 | 1090 | W2 | +2,628 (solos +596 / 0 / +2,628) | +8,251 | **+5,623 (+3,577)** | 8.3% | +6,254 |
| 989006 | 450 | W2 | +3,496 | +4,829 | +1,333 (+845) | 4.1% | +2,050 |

### 5.3 Mechanism: partial walls

The partial walls were measured in `PARTIAL_WALLS.json`. The table lists J per arm in veh-s; in brackets is the benefit versus base (+ = less delay).

| Situation | base | group | lane0+1 | lane0+2 | lane1+2 | solo lane0 | solo lane1 | solo lane2 |
|---|---|---|---|---|---|---|---|---|
| 989001 t440 W1 | 13,638 | 10,457 (+3,181) | **10,675 (+2,963)** | 14,671 (−1,033) | 17,066 (−3,428) | 17,160 (−3,522) | 16,829 (−3,191) | 13,989 (−352) |
| 989006 t1090 W2 | 68,081 | 59,830 (+8,251) | 67,485 (+596) | **59,830 (+8,251)** | 65,454 (+2,628) | 67,485 (+596) | 68,081 (0, inert) | 65,454 (+2,628) |

- **Super-additive pairs.** In each case one pair that includes the **closing-lane CAV (lane 0)** delivers the whole group effect, and the pair's effect exceeds the sum of its solos.
- **989001 t440, group arm:**
  - All three members ran at 15.0 m/s abreast for 90 s.
  - At t0+120 they crossed the drop at 22–23 m/s without stopping.
- **989001 t440, solo arms:**
  - The lone slowed CAV was bypassed.
  - The lane-0 and lane-1 solo members ended stuck in the drop queue at 0.4 m/s at t0+120.
- **What the pair needs.** The closing-lane CAV and a partner in another lane, abreast, act together as a moving bottleneck. This lets the drop queue drain and then releases a platoon that merges without stopping. That is the moving-bottleneck mechanism from the literature (Section 2), on a lane drop.

### 5.4 Noise floor, replicates and safety

- **Pulse noise floor** (19 situations): 16 had |ΔJ| < 1 veh-s. The other three: −357, +1,482 and +3,256 veh-s.
  - The congested lane-drop dynamics bifurcate: a tiny perturbation sometimes triggers or averts a breakdown.
  - The per-run block bootstrap does not include this bifurcation noise.
  - Best-of-n solo selection biases S **downward** under such noise.
  - Admission based on one run is optimistic. Rigorous inference needs several dithered replicates per situation.
- **Dither replicates:** 8 of 8 stayed admitted. In 4 of them the dither changed outcomes and S kept its sign (+1,288 to +6,254). In the other 4 the dither was inert, producing bitwise-identical arms.
- **Safety:** 0 collisions and 0 teleports in all 359 runs.
  - Hard-brake (a < −3 m/s²) veh-s, group vs base on wz3to2 walls: W1 13.7 vs 17.4; W2 15.1 vs 17.4.
  - Minimum acceleration in a group arm: −3.61 m/s² for W1, but **−7.96 m/s² for W2** (one emergency-braking event, collision-free).

### 5.5 Comparison with the current merge map

**Previous evidence on the current map:**
- Coop probe: about 1 situation (R 986061 t455, 4-CAV wall) admitted on 3 of 6 seeds, with synergy about +250 to +490 veh-s on a 120-s tail. None was admitted in the other 9 screened TRAIN moments, and none in the less-cooperative HDV variant.
- Mining: 0 admitted G5 moves.
- Clean2 B: mean synergy −0.034.

**On wz3to2:**
- 8 admitted situation-variants, from 4 of 7 worlds, in 34 situation-variants.
- Typical admitted synergy is +0.6k to +5.6k veh-s over 360 s.
- Super-additive pairs where each member alone is harmful.

**The comparison is not like-for-like:**
- the base here is native, not frozen MAPPO u0;
- the windows differ;
- there is one penetration (0.3) and one demand per map.

Even so, wz3to2 produces group-necessary situations much more often and with larger gains.

## 6. What migrating the stack would take

Code facts inspected:
- **Geometry:** `traffic_shepherd/c4/geometry.py` hard-codes LANE_TABLE, the edge ids 1191554636 / 1487606179 / 1487606178 / 11553048#1, HERE_OFFSETS and MERGE_S1_M = 75.44.
  - **21 files** in `src/wms` reference the merge-zone constant.
  - **43** `src/wms` + `traffic_shepherd` files mention the ramp.
- **WM physics:** `src/wms/physics/config.py` encodes ramp-merge rules (merge zone [0, 75.44], merge_end_wait, mainline_merge_yield).
- **Policy observations:** `src/wms/control/obs.py` builds own features such as `s_to_merge_end_m` and a merge-zone flag.
- **Scenario builder:** `traffic_shepherd/scenarios.py` fixes the network, ramp 400 veh/h, mainline 1,900–2,800 veh/h and the seed guards.
- **MAPPO (aecd7d70 / speed-only iteration 169) and clean3:**
  - per-CAV own features (13 dims);
  - relayed reports (13 dims, up to 16);
  - an RSU group encoder over sensor cells with merge geometry;
  - clean3's flat residual model (29 per-CAV features plus V2V).
- **V2V** (`V2V_OBS_READY.json`): K = **4 partners × 6 relative features** (gap, lane offset, v, a, last command, presence; nearest first, 100 m). It is **map-agnostic**: a 3-CAV wall fits in 4 slots.

| Component | Needed for wz3to2 | Reuse | Effort |
|---|---|---|---|
| Scenario/env layer (`scenarios`, `c4.env`, `sumo_io` boundary edges, sensing lane map, seed guards, adapter config) | Map abstraction: per-map edge table, demand family, boundary edges | CommandOwnership, speedMode-27 conventions, population, branching (saveState), recorder schema | 1–1.5 d |
| Model frame (LANE_TABLE, s-coordinate, merge-zone constants in 21 files, lane rules, RSU cells, obs features) | The 3-lane frame fits if WM lane 2 is redefined as the *closing lane*. But the mandatory merge zone becomes about 2.7 km instead of 75 m, and merge-end rules, entrance-queue logic and RSU zone must be re-parameterised and re-tested. | Lane count (3), slot pool K = 192, obs layout | 1.5–3 d incl. tests |
| WM data | New corpus: native + undirected random (genrand) + policy rollouts, about 0.5–1.5M sim-s. Throughput is about 124k sim-s per worker-hour, so roughly 1 h of compute; QA is the real cost. | genrand / collect code | 0.5–1 d |
| WM retrain or fine-tune + qualification | The lane head (lane_v5) and merge calibration were learned for the ramp. The C1, distributional-fidelity and objective-ranking gates must be re-passed. Historically the WM needed many overnight GPU rounds (R3 to 13E). | Architecture and training code; weights only as initialisation | 2–4 d (GPU nights), **highest risk** |
| Speed-only MAPPO comparator | Retrain on the new map: about 169 it × 46 ep × 900 s ≈ **7.0M sim-s** (SELMAPPO design record); selmappo2 took about 9.6 h on cpuQ. | Trainer unchanged once the env adapter exists | ~1 d incl. evaluation |
| Clean3-style coop pipeline (frozen base + residual/V2V, move library, mined bank, G-levers, synergy statistics) | Wall roles must cover 3 lanes, and the ramp roles (gap/ramp_assist) are replaced by closing-lane roles. Bank and teaching data regenerated; checkpoints are not transferable. | V2V layout, statistics, verify_perseed / bank code, this probe harness | 2–3 d |
| Evaluation / registration / safety gates | New seed blocks, parity canaries, panels | Existing tooling | 1–2 d |
| **Total** | | | **≈ 9–15 working days**. Even with parallel agents, at least 6–8 calendar days, with the WM on the critical path. |

## 7. Recommendation given the Nov 15 deadline (9 days)

1. **Do not migrate the main stack before Nov 15.** The WM fidelity work alone has taken multiple weeks on the current map, and a fixed-rule wall on wz3to2 is not positive on average. A learned policy would have to discover the state-dependence, and that cannot be done and verified in 9 days.
2. **Use this result in the paper as a short, honest SUMO-only section.** Framing: "On a merge with a short acceleration lane, single-CAV smoothing captures almost all gains. At a lane-drop bottleneck, CAV coalitions that cover the closing lane plus another lane create super-additive gains that no single CAV achieves."
   - Report the partial-wall table, the 8/34 admission rate, the mean-synergy CIs and the noise-floor caveat.
   - Cost is zero new sim-s if reused as is. About 1,480 sim-s remain in this phase.
   - A properly powered version would need more budget: about 3 dithered replicates per situation for about 20 situations is roughly 100k sim-s.
3. **After the deadline, migrate to wz3to2 as the cooperation testbed**, in two steps:
   - first a SUMO-oracle learner or selector with a WALL action and a closing-lane partner role, on the existing SUMOCF / SELMAPPO tooling: env layer only, about 3–5 days;
   - then port the WM, treating the closing lane as WM lane 2.
4. **Drop the weave and 2→1 candidates** unless their SUMO modelling is fixed. The weave gridlocks under LC2013. The 2→1 lane end gives an unrealistically low discharge, and walls there are harmful on average.

## 8. Limitations

- **Base controller:** native CAVs, not frozen MAPPO u0. A smoothing policy may capture part of the gains.
- **Scope:** one penetration (0.3) and one demand per map. Fixed rule parameters (two wall variants), not tuned.
- **Sample size:** n = 17 situations per wall variant on wz3to2, from 7 worlds.
- **Inference:** per-situation inference ignores bifurcation noise (see 5.4).
- **Not tested here:** 4-CAV walls, other penetrations, an RSU-only state trigger, or learned roles.
- **Calibration:** the demand choice used native outcomes on seed 989000. That world's later wall situations were selected by geometry only, but its native trajectory informed the demand level.
- **Records:** `protocols/experiment_history.json` and OPEN_QUESTIONS were **not** updated. That is left to the integrating agent; a REGISTRATION.md block was appended.

## 9. Evidence and reproducibility

- **cpuV `/home/ubuntu/newmap20261106/`** (88 MB):
  - `code/` (probe_core, make_jobs, make_dither, make_pairs, run_jobs, analyze, build_maps, population copy);
  - `maps/`;
  - `runs/<map>/<demand>/s<seed>/{native,wallW1_t*,wallW2_t*,*_d1,gap_t*}/<arm>/RESULT.json` (per-second J series, safety counters, member trajectories);
  - `jobs/*.jsonl`;
  - `ledger/CHARGES.jsonl` (365 rows, precharged, **148,520 sim-s**);
  - `code/PIDS.txt`;
  - `SUMMARY_stage{1,2,3}.json`;
  - `PARTIAL_WALLS.json`.
- **Local copy:** `E:/research-dense/runs/active/C4-V7FT-20261017/newmap_probe/`, containing the code and `results/{SUMMARY_stage3.json, FINAL_TABLE.json, PARTIAL_WALLS.json, CHARGES.jsonl, PIDS.txt}`.
- **Maps:** `E:/research-dense/assets/research/newmap_candidates/{wz3to2,wz2to1,weave}/`.
- **Spend by category (sim-s):**

  | Category | sim-s |
  |---|---|
  | native prefixes incl. calibration | 30,240 |
  | wall W1 | 53,640 |
  | wall W2 | 24,480 |
  | gap pairs | 14,400 |
  | dither replicates | 14,400 |
  | pulse noise floor | 6,840 |
  | partial walls | 2,160 |
  | smoke, debug, insertion test, example check | 2,360 |

- **Status lines:** CLEAN1_STATUS.md, lines prefixed `NEWMAP`.

Sources (web-verified this session; Han & Ahn 2018 TR-B 107:146–166, Han, Chen & Ahn 2017 TR-B 98:113–134 and Chen & Ahn 2018 TR-B 108:1–20 were confirmed as citations in search results without a captured stable URL): [Čičić et al. arXiv 1907.13049](https://arxiv.org/abs/1907.13049), [Vinitsky et al. 2018 ITSC](https://ieeexplore.ieee.org/abstract/document/8569615), [Vinitsky et al. arXiv 2011.00120](https://arxiv.org/abs/2011.00120), [Yan et al. arXiv 2208.00268](https://arxiv.org/pdf/2208.00268), [Chen, Srivastava & Ahn 2021 TR-C](https://www.sciencedirect.com/science/article/abs/pii/S0968090X20308470), [CLMS, ASCE](https://ascelibrary.org/doi/abs/10.1061/JTEPBS.0000632), [Tilg et al. 2018 (TRID)](https://trid.trb.org/view/1549768), [Weave CAV optimisation TR-C 2021](https://www.sciencedirect.com/science/article/abs/pii/S0968090X21000978), [Bi-level weave control arXiv 2403.16225](https://arxiv.org/html/2403.16225v1), [Lioris et al. arXiv 1511.00775](https://arxiv.org/pdf/1511.00775), [Piacentini et al. 2018](https://www.researchgate.net/publication/326515865_Traffic_control_via_moving_bottleneck_of_coordinated_vehicles), [CAV on-ramp merging review arXiv 2202.09457](https://arxiv.org/pdf/2202.09457).
