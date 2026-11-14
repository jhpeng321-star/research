Plain summary — S-GROUPDIAG, 2026-11-05, read-only:
- B helps traffic, but its +726.400 veh-s is almost entirely recoverable by one ramp CAV per useful state; the best-solo mean is +726.434.
- World 986048 supplies 90.48% of the net benefit: group +3,286.289, ramp.22 alone +3,286.289; their difference is only +0.000037 veh-s.
- World 986042 is exactly single-actor-equivalent; in 986046 the best ramp solo is slightly better than the group.
- Only ONE joint library move is selected/executed in the seven registered B states: an upstream three-car wall, with only +0.0794 veh-s extra over best solo.
- The other six states offer no joint macro at any recorded group decision; their geometry, departures and delayed onsets poorly match the move library.
- Training does execute joint moves: u6 has 96 accepted starts, 93 with >=2 physically affected actors; 78 of those 93 occur in six practice episodes.
- F2 works: zero rejected non-none proposals; nevertheless 79.05% of u6 family checks are unavailable, and 61.66% of its recorded none choices are forced.
- B's eight admitted native-plan examples have positive matched-window synergy; the teaching bank is NOT entirely single-actor-dominant.
- Native profiles are compressed into six wall labels and one none label; the projected labels themselves are not certified to retain native synergy.
- Older bank gain fields can compare unequal horizons: 986048's apparent +958.68 synergy becomes -287.48 on the common recorded window.
- The speed-deficit team reward and post-move safety penalty give no explicit credit for beating the best solo or for the second actor's marginal contribution.
- Best one-day route: certify executable joint labels, create earlier role-aware pair opportunities, then add sparse counterfactual credit; a full QMIX/QPLEX rewrite is lower priority.
- Zero SUMO, zero charges, no remote writes or running-process inspection/control; no prohibited paths or hidden population/driver truth opened.

# Clean2 group-cooperation diagnosis and literature research

## 0. Scope and interpretation

**Owner questions:** why no group benefit, where B's total benefit comes from, whether synergistic examples exist, how the second member is credited, and what is feasible within roughly one day and the remaining approximately 2.5M sim-s.

This is an analysis of existing evidence, not an intervention. cpuV was accessed through a WSL OpenSSH multiplexed connection with a dedicated control socket, 12-second connection bound, keepalive bounds, and a 110-second bound per remote Python read/arithmetic command. Remote Python used `-B` and disabled bytecode writing. No simulation/training/accounting entry point was invoked. No running process was inspected, adopted, signalled, stopped, or restarted. Remote path readers rejected both lexical and resolved paths containing the owner's excluded names before opening them. Static implementation code and observable receipt fields were read; hidden truth datasets were not.

Run root, abbreviated **ROOT** below:

`/home/ubuntu/coopmwm20261105_clean2`

Source root, abbreviated **SRC**:

`/home/ubuntu/coopmwm20261105_clean2_src_285663a1/pytools/tools/coopmwm`

Evidence cutoff: the complete u6 replicated diagnostic; B/R update receipts u1–u9; all **16 B/u6 training episode tapes**; seven B/u6 all-group continuation tapes; selected matched solo/reference tapes; current update-bound bank summaries and practice bindings. The last metadata capture was **2026-11-05 14:25:50 UTC / 22:25:50 +08**. These receipts do not establish current process health or a later u24 outcome.

The registered “test states” here are the seven prospectively selected **TRAIN diagnostic states**, not a separate inaccessible evaluation partition. Selection was by resolvable multi-CAV context counts and seed, not by Clean1 outcomes. Their receipts can therefore diagnose mechanisms, but should not become an outcome-selected new training set.

### Definitions that prevent misleading conclusions

Let lower `J` mean less source-inclusive network delay, in veh-s:

```
D_group = J_base - J_group
D_i     = J_base - J_solo_i
S       = D_group - max_i D_i = min_i J_solo_i - J_group
```

`S` is the registered **group-over-best-single synergy**. Solos have no zero floor. It is not group-minus-sum-of-solos, a Shapley value, or total benefit. A useful single CAV can have large `D_i` and exactly zero `S`.

Each solo substitutes the current policy on one original group member; group/reference/solo share captured state, prefix and CRN, with background actuators on the common base. Interventions begin at `t0+1`. The bundle includes the continuous learned correction and subsequent trigger decisions, not just a fixed named move. Relevant implementations: `policy_synergy.py:25–50,95–119,137–145`; `track2_monitor.py:36–47,462–493,518–585,601–630`; `launch_production.py:278–319`.

The published estimate is an **equal-world mean**, with equal-state averaging inside each world. The three 986040 states each have weight 1/15; each other state has weight 1/5. Thus +726 is a mean benefit under this weighting, not a raw sum across the seven states.

## 1. Where B's +726.400 comes from

### 1.1 Exact state/member decomposition

All gains below are `base minus branch`, higher better. The full underlying J values are in Appendix A.

1. **r0/s0, world 986040, window 300–600, group [ramp.22, mainline.100].**
   - Group **−0.391263442**.
   - Solo ramp.22 **−0.378614645**; solo mainline.100 **−0.000002011**.
   - Best solo is mainline.100; synergy **−0.391261431**.
   - Two independent ease_off starts, no joint option. This state contributes **−0.026084229** to the published mean.
2. **r0/s1, world 986040, window 300–720, same two members.**
   - Group, both solos, and base are **exactly equal**: gain 0, synergy 0.
   - mainline.100 accepts ease_off with onset 24 s but is observed for only 20 post-t0 ticks; ramp.22 selects none. There are **zero nonzero library-offset member-ticks**.
   - Nonzero continuous residuals are present. Equality of aggregate delay does not imply an unloaded policy.
3. **r0/s2, world 986040, window 310–460, group [mainline.127, mainline.126, mainline.129].**
   - Group **+0.250637984**.
   - Solo mainline.127 **−1.004376036**; mainline.126 **−0.986063728**; mainline.129 **+0.171252827**.
   - Synergy **+0.079385156**.
   - The only joint library start in the panel: mainline.126 chooses a three-member wall at t311, onset10, hold28, depth0.433615 m/s²; later, the three members start separate ease_off moves.
   - Entire state contributes **+0.016709199** to the mean; its extra-over-best-solo portion contributes **+0.005292344**. This is evidence of a small interaction in the policy bundle, not a wall-only causal ablation.
4. **r1/s0, world 986048, window 300–720, group [ramp.20, ramp.22].**
   - Group **+3,286.288989699**.
   - Solo ramp.20 **−0.000230383**; ramp.22 **+3,286.288952870**.
   - Synergy **+0.000036829**, only about **0.0000011%** of the group gain.
   - No joint candidate is offered to either group member. ramp.22 starts ease_off at t301 (onset2, hold21, depth0.758131) and t361 (onset10, hold55, depth0.923961); ramp.20 starts its own ease_off.
   - Contribution **+657.257797940**, **90.48% of the net +726.400**.
5. **r2/s0, world 986042, window 600–1020, group [mainline.267, ramp.38].**
   - Group **+138.600386204**.
   - Solo mainline.267 **0**; ramp.38 **+138.600386204**.
   - Synergy **exactly 0**. No joint option; ramp.38 starts ease_off at t601 (onset27, hold32, depth0.680967), then none at t661.
   - Contribution **+27.720077241**.
6. **r3/s0, world 986046, window 300–660, group [mainline.108, ramp.18, mainline.109].**
   - Group **+207.155871661**.
   - Solo mainline.108 **−0.000037548**; ramp.18 **+207.222004936**; mainline.109 **−0.012159669**.
   - Synergy **−0.066133275**: the other members slightly worsen the best ramp solo.
   - All three select individual easing; ramp.18 uses onset2, hold71, depth0.434545. No joint candidate is offered.
   - Contribution **+41.431174332**.
7. **r4/s0, world 986068, window 600–1020, four-member group.**
   - Group **−0.000314583**.
   - Solo mainline.264 **0**; ramp.51 **0**; mainline.267 **−0.000195884**; mainline.270 **−0.000314583**.
   - Synergy **−0.000314583**.
   - Four accepted ease_off starts, but all onsets exceed the relevant member's remaining presence: **zero nonzero library offsets** on the original group.
   - Contribution **−0.000062917**.

The exact accounting identity is:

```
mean best-solo gain       +726.433685523
mean extra-over-best     -  0.034073958
                         --------------
mean group gain          +726.399611565 veh-s
```

**Answer to “is it all one member?”:** not one universal vehicle ID across worlds, but essentially one **state-specific ramp member** in every materially beneficial state. The three large-positive ramp solos alone contribute +726.422269 under the registered weighting, slightly more than the net group mean. One small upstream state adds genuine positive interaction; it is overwhelmed by small negative group interactions elsewhere.

B u6 total-benefit CI is **[+27.701201, +2,013.195193]**; synergy mean **−0.034073958**, CI **[−0.075601910, −0.000048185]**. The latter upper endpoint rounds to 0.00 but is actually slightly negative. These are descriptive u6 diagnostics with five world replicates, not the u24 verdict. R's total benefit is −59.383381 and synergy −59.392844; its world986046 group loses −296.928231 versus base, while its best solo loses only −0.000028234. B–R is therefore not evidence of positive B synergy.

### 1.2 What traffic quantity improves, and what cannot be attributed

Recomputing `reference.window - all.window` shows **+726.199612 veh-s of the mean is on-road delay**, and **+0.200000 is outside-source waiting**. World986048's entire +3,286.289 and world986042's +138.600 are on-road; world986046 supplies +206.155872 on-road plus +1.0 source-wait veh-s before the 1/5 weighting.

The data do **not** support saying “the second CAV opened a gap and made the ramp car merge earlier.” In all/group, winning solo, and base, the main beneficial ramp members first enter lane1 at the same recorded times: ramp.22/986048 **t367**, ramp.38/986042 **t651**, ramp.18/986046 **t326**. Their sampled own trajectories are identical or very close; ramp.38's displayed 10-second trajectory samples are exactly equal. The observed gain is in the wider traffic continuation, not a demonstrated earlier ramp-entry event. Resolving the exact downstream vehicle/lane-change cascade would require a fuller observable trajectory join; that mechanism is not inferred from aggregate J alone.

Likewise, **“single-CAV ease_off-associated policy gain” is defensible; “all +726 was caused by the ease_off offset alone” is not.** The matched solos isolate a member's current policy bundle. The learned residual also acts before onset and outside active moves. No move-disabled/residual-fixed causal branch exists here, and none was generated for this report.

## 2. How much joint execution actually occurs?

### 2.1 Count actors, not a family name or membership list

In this API:

- `ease_off`: one actuator.
- `gap_open`: two or more context members, but **only members[0] actuates**. It is not a >=2-executing-actor move.
- `wall` and `ramp_assist`: all listed members actuate together, subject to presence, release and the final command/safety path.
- `none` on a follower can mean an earlier same-tick joint acquisition already owns it. It is not necessarily an independently chosen refusal to cooperate.

For actual execution I count a unique accepted commitment by its move and start time, and require >=2 actors simultaneously present with nonzero offsets. The stronger physical count additionally requires each of >=2 issued-command changes to exceed the production receipt's **1e−6 m/s²** threshold. Simultaneous independent ease_off moves are reported separately, not relabelled as one joint plan. Sources: `moves.py:30–36,129–146,156–210`; `feasible_proposals.py:77–127`; `clean2_receipts.py:23–48`.

### 2.2 Registered B/u6 group states

Across the **original intervention groups only**, after t0:

- **23 selections:** ease_off18, none4, wall1, gap_open0, ramp_assist0.
- **19 non-none starts accepted; zero rejected**.
- **One joint start**, in **1/7 states**, or 1/23 selections and 1/19 accepted starts.
- That wall has >=2 nonzero actors for **28 seconds**, and >=2 physically changed commands for **27 seconds**.
- Total original-group exposure is 656 CAV-ticks across 363 state-seconds with a group member present.
- Independent or joint moves produce >=2 nonzero offsets simultaneously for **84** state-seconds; >=2 physically affected actors for **63**. Most such simultaneity is independent easing, not a joint commitment.
- **Only 1/23 decision rows offers a joint macro, and B chooses it.** The other six states never offer wall/ramp_assist to the original group along the observed decision path.

This is a much stronger explanation than “B's joint head refuses available cooperation.” On this small panel, the joint macro is usually absent from the slate.

### 2.3 Geometry and timing explain much of the absence

The generator is specialized: wall requires an upstream, multi-lane group before s=−100 m, within the registered span, with at least 20 s before first arrival at the zone. Gap/ramp families require relevant ramp/mainline vehicles in the registered zone **[−100,75.44] m**. They cannot reproduce arbitrary native bank groups. Sources: `moves.py:235–320`; `variety.py:64–72`.

Actual first post-t0 observable group geometry:

- r0/s0 and s1: mainline.100 is already s≈+0.96 m/lane0, and ramp.22 is already merged into lane1 at s≈+26.31 m. Neither is an upstream wall member; there is no lane2 merger in this pair.
- r0/s2: the three members are approximately **1.5–1.6 km upstream**, spanning lanes0/1. This is the panel's wall-compatible state.
- r1: both members are lane2 ramp cars, one stopped at the ramp end. They do not provide the two-lane upstream wall geometry or the ramp/mainline role pair.
- r2: mainline.267 is already lane0/s≈+141.42 m; ramp.38 is stopped at s≈+75.44 m. The original pair is not the configured ramp/mainline-in-zone pair.
- r3: the two mainline members are lane0/s≈+92.13 and +187.95 m, while ramp.18 is stopped at the ramp end. Again the useful partners have passed the narrow zone.
- r4: all four are already downstream, s≈+264.72…+494.62 m; ramp.51 is already in lane1. They remain present only 3–11 ticks, versus chosen onsets9–25 s.

There is also a **precision-boundary exclusion to audit**: stopped ramp rows are stored as s=**75.44000244140625**, while `_zone_pools` applies `s <= 75.44` literally. Thus those observed ramp-end rows lie just outside the configured upper bound. This is a concrete representation discrepancy, not evidence that fixing it alone creates an eligible partner or positive synergy. A precision-consistent geometry contract is preferable to arbitrary zone padding.

Absence of a joint macro does not mathematically prohibit all policy synergy: two independently selected longitudinal actions could interact. It does establish that the intended joint-macro channel is unused on six of these seven observed paths.

### 2.4 Training: joint moves exist, but opportunities and practice concentration dominate

All B/u6 episode tapes were recalculated, removing replayed practice-prefix seconds through each practice t0 exactly as training does. Counts match the update receipt: **2,097 decisions, 125,529 CAV outcomes, 48,424 nonzero library-offset CAV-ticks, 38,387 physically changed issued commands**.

- u6 selected ease_off1,099, none892, wall87, gap_open10, ramp_assist9.
- True joint starts = wall87 + ramp_assist9 = **96/2,097 = 4.58% of decisions**, or **7.97% of 1,205 accepted starts**.
- **94/96** ever have >=2 simultaneous nonzero offsets; **93/96** ever have >=2 physically affected actors. This is 93/1,205 = **7.72% of accepted starts**.
- Joint commitment exposure is **3,637 commitment-seconds** with >=2 offsets, **2,956 commitment-seconds** with >=2 physically affected actors. These are summed per commitment and can overlap in clock time.
- **14/16 episodes** contain at least one physically actuated joint commitment.
- Only **15** of those 93 commitments occur in the ten normal episodes; **78 (83.9%)** occur in the six practice episodes. Two practice episodes are the same 986052 example/seed repeated; the 986068 practice episode alone supplies 33. Later vehicles in a restarted episode are not the original registered intervention group.

Longer update-level history, using accepted-selection receipts (physical >=2 execution audited separately at u6):

- B u1…u9 joint starts: **103, 110, 92, 121, 136, 96, 160, 145, 152**.
- Corresponding decision totals: **2,668, 2,502, 2,405, 2,482, 2,794, 2,097, 2,771, 2,593, 2,544**.
- Combined B: **1,115/22,856 = 4.88%** of decisions; 1,115/13,684 = **8.15%** of non-none starts.
- R, same procedure: **772/18,080 = 4.27%** of decisions; zero rejected non-none starts as well. Different counts reflect own traffic/practice histories, not different algorithm settings.

The owner's approximately 50% ease_off /44% none /5% wall is the **late training selection mix** (B/u9: 49.57%,43.99%,5.27%, with gap0.47% and ramp0.71%). It is not the original-group u6 diagnostic histogram above.

### 2.5 F2 is successful, but zero rejection is not abundant joint feasibility

Two different rates must be distinguished:

1. **Post-selection execution refusal:** B/u6 **0/1,205 = 0%**. Across u1–u9 B **0/13,684**, R **0/11,039**. The old Clean1 overlapping-acquisition refusal explanation should not be carried forward as an unfixed Clean2 bug.
2. **Pre-selection unavailable-family checks:** B/u6 **6,631/8,388 = 79.05%**; across B/u1–u9 **72,052/91,424 = 78.81%**. This includes ownership and geometry exclusions and is not an unsafe-action rate.

u6's 892 none rows include **550 forced none-only masks (61.66%)**. Among the remaining 1,547 rows with actual non-none options, 342 choose none: **22.11%**. Thus 42.54% none overall substantially overstates deliberate abstention.

Joint macros are offered in **170/2,097 decision rows** (11.0% of the 1,547 non-forced rows); B selects them in **96/170 = 56.47%**. The generator already balances slots **among feasible families**, but ease_off is the only feasible family in many states: **11,629/12,376 = 93.96% of non-none candidate slots** are easing. Simply asking again for “family balancing” would duplicate F2 without addressing geometry, ownership priority, arrival timing or state coverage.

## 3. Is synergy structurally reachable? Native plans versus executable teaching

### 3.1 Do not use legacy unequal-window synergy as an oracle

The first-seed bank receipts often grade all candidates through a window set by the longest candidate, while member-only solos end after the chosen plan's duration plus tail. Directly subtracting those full-window *improvements* can compare different horizons. `bank_statistics.paired_window` deliberately uses their recorded common interval (`bank_statistics.py:255–266,453–478`).

I re-read the original group/reference tapes and every available member-only tape for the four registered source plans with explicit solo links, using cumulative fingerprint metrics at the **same t0 and common end**:

- **986048/t300:** registered group window ends720; all available source-plan solos end510. Over **300–510**, native group J=24,545.904130, reference J=24,258.422326; solo Js are 24,258.422684/ramp.20 and 24,545.903970/ramp.22. **Native gain −287.481804; synergy −287.481446.** The legacy full-gain arithmetic advertised +958.678083 synergy. That is not matched-horizon positive synergy.
- **986042/t600:** registered end1020, common solo end840. Over **600–840**, native gain **+166.959530**, best ramp.38 solo exactly equals group: **synergy 0**, versus the legacy +1,866.371737 field.
- **986046/t300:** registered end660, common solo end480. Over **300–480**, native gain **+9.806637**, best ramp.18 solo gain +9.802550: **synergy +0.004087731**, versus legacy +424.270215. This tiny positive does not make the safety-failed source plan an admitted lesson.
- **986068/t600:** registered end1020, common solo end960. Over **600–960**, native gain **+2,776.035714**, best solo/ramp.51 gain +2,668.307933: **synergy +107.727781**. This is substantive matched-window native synergy, and this source example is in the effective teaching bank.

For the three 986040 source plans, **complete graded native solo receipts were not available in the inspected bank locations**. A planned job listed in a confirmation manifest is not a measured outcome. Their native-plan synergy remains unestablished here, not zero-imputed.

Second-seed supplements further limit a broad “oracle synergy everywhere” story: 986048 and 986042 report **0** extra-over-solo, 986046 **−0.000173780**, 986068 **−0.507024361**. These are distinct verification seeds, not the five diagnostic world replicates.

**Reachability verdict:** some native profiles demonstrate positive joint interaction in related source-policy continuations, especially 986068; it is not true that every registered source plan is synergistic. Existing solo tapes do not cover the full original diagnostic horizon for these four plans, and their background policy differs from the common-u0 diagnostic. Native candidate execution can also start at t0, whereas the policy diagnostic preserves the captured t0 action and intervenes from t0+1. Therefore no complete, same-background/full-window native-oracle ceiling for all seven registered tests is established by the existing receipts. Generating that ceiling would require new branches, which were not run.

### 3.2 The actual admitted B teaching examples are not all single-actor-dominant

Use **`B_UPDATE_006.json → learning.bank_summary.entries → seed_synergy_total_veh_s`**, not stale `BANK_QUALITY.json` or its legacy `cooperation_gain_veh_s` field. The effective bank contains **8 cooperative seed examples from 7 plans** (7 seed1, 1 seed2), all with positive matched-window bootstrap lower bounds:

- 986052/t300, seed1: synergy **+0.000496**, lower95 **+0.000058**; projects to none.
- 986066/t300, seed1: **+935.741842**, lower95 **+282.946243**.
- Same 986066 plan, seed2: **+371.983911**, lower95 **+203.857743**.
- 986068/t600, seed1: **+107.727781**, lower95 **+18.575654**.
- 986003/t390, seed1: **+894.878501**, lower95 **+84.676987**.
- 986026/t300, seed1: **+40.101264**, lower95 **+9.181080**.
- 986036/t300, seed1: **+249.907436**, lower95 **+106.891289**.
- 986047/t360, seed1: **+311.501767**, lower95 **+113.979331**.

These within-example block-bootstrap bounds certify the bank's own admission rule; they are not across-world generalization intervals. Only the 986066 plan is cooperative on both verification seeds; its seed2 later-harm is explicitly not measured under the registered per_seed rule. The evidence supports **native synergistic examples exist**, not “all native lessons generalize.”

The current mean matched native synergy across the eight examples is **+363.980375 veh-s**. R has three admitted examples, with matched native synergies approximately **+0.0000324, +0.0002674, +456.893784**. B/R use the same learning procedure and family matching, not identical teacher quality.

### 3.3 The bottleneck is transfer from native profiles to the executable vocabulary

I re-evaluated the captured source's pure `from_plan` projection functions locally, with the registered parameter windows (onset0–30 s, hold20–90 s, depth0.3–1.0 m/s², <=4 actors). No traffic model or simulator was imported/run.

The seven B source plans project to **six wall-shaped labels and one none**:

- 986066: six native members/150 s → **three actors**, onset9/hold90/depth0.550669. The projection drops the native best-solo member ramp.24.
- 986068: four native members/240 s → **ramp.51 + mainline.270**, onset2/hold90/depth1.0.
- 986003: two native members/300 s → two-member wall, onset3/hold22/depth1.0.
- 986036: six native members/30 s → four-member wall, onset2/hold28/depth1.0.
- 986026: three native members/30 s → three-member wall, onset1/hold29/depth1.0.
- 986047: six native members/30 s → four-member wall, onset6/hold24/depth1.0.
- 986052: native profile's best least-squares representable move is **none**.

The shared move has one negative box profile across its actors. It cannot directly express the native member-specific timings, positive/negative segments, long holds, or all six members. In particular, ramp_assist currently gives both actors the **same easing window**, rather than complementary yield/advance roles. Gap_open includes a partner but only one actuator. These are real representational restrictions.

**Native plan verified != projected macro verified.** Least-squares closeness of offsets does not preserve nonlinear traffic outcomes, safety or synergy. The corresponding projected wall can also be absent online because native plan membership is not generated by online wall geometry. Bank trigger teaching inserts the exact projected label into a candidate slot after a visibility check, rather than requiring the online proposal generator to produce it. Sources: `moves.py:376–496`; `bank_breadth.py:228–268`; `launch_production.py:74–108`.

Clean2 improved breadth, but supervision is still narrow: **four usable B trigger contexts (three wall, one none)**, versus three for R (two wall, one none). CE family mass is **50% none /50% wall**, independently of the native AWR advantage; B CE declines 2.1251→1.2120→0.7321 at u1/u6/u9. Continuous native-profile imitation and discrete projected-label imitation are different pathways; learning the latter does not demonstrate successful reproduction of the former.

Practice has eight available B seed contexts and three R contexts. B/u6 uses six practice episodes from five unique contexts: **four wall-labelled and two none-labelled starts**. Matched family capacities are bounded by the smaller arm's pool and the repeat cap. Raising only the requested practice fraction cannot create missing verified contexts or a gap/ramp teacher family. Sources: `bank_breadth.py:272–380`; `SMOKE_FIX_RECEIPTS_PASS.json`; update-bound practice receipts.

## 4. Does the second member get extra credit?

### 4.1 Exact active reward and optimization path

For CAV i on a physical one-second control tick:

```
speed_i = -max(0, 1 - observed_v_i / 31.29)
r_i     = speed_i + mean(speed_j for authorized observable neighbours j)
          - count(post-actuation observable closing pairs with TTC < 1.5 s)
```

Neighbour weight and safety pair-event weight are both fixed at1.0. Empty-neighbour mean is zero. Own sensing and authorized timestamped V2V packets determine the domain. The safety term starts strictly after the first nonzero library offset of a currently active move containing that CAV and lasts through expiry. It is **temporal attribution, not a causal counterfactual**. Sources: `track2_reward.py:1–29,90–150,187–325`; `track2_safety_reward.py:14–23,46–71,75–115`.

The production trainer uses **per-CAV finite-episode discounted returns**, gamma=`exp(-1/600)`, scaled by the inherited reward unit10,000, stopping at episode end/terminal tick/CAV departure. Continuous residual PPO uses a zero baseline; trigger PPO uses return minus its recorded **per-CAV value**. This is not a COMA joint-action critic. Dense teacher, WM group credit and the separate coop_reward shaper are OFF. Sources: `track2_backend.py:736–792,794–900`; `launch_production.py:36–40`.

At B/u6, the reported equal-acting-CAV-mean sums are own **−3,741.082784**, neighbour addition **−2,964.178109**, safety penalty **−101.941667**, total **−6,807.202561**. There are 1,296 per-CAV observable safety pair-event counts, not necessarily 1,296 unique network events. The episode-level mean aggregation explains why the penalty sum is not −1,296.

### 4.2 Direct answer

**No explicit extra credit is assigned for the second member being necessary, or for joint execution beating individual execution.** Both members can benefit indirectly if their actions improve observable speeds or prevent future close calls. That is ordinary outcome credit, not second-member credit.

For a fixed observed speed/safety outcome, changing a move label from one actor to two creates no positive reward term. A second member that briefly slows to help another can immediately worsen its own speed reward and neighbours' mean. Later network benefits may occur outside its neighbourhood or after its departure, so they may never enter that member's return.

Joint choice itself is centralized only at a small macro level: one ego acquires a group move. Its trigger log-probability is trained using **that ego's** local return. Other acquired members can have none-only masks; those rows have no categorical action-choice gradient for choosing the joint macro. They still receive continuous-action learning on their own returns. There is no `J_without_second − J_group`, no max-solo subtraction, no shared realized macro advantage assigned to the proposer, and no fitted pair-interaction value. Sources: `feasible_proposals.py:96–127`; `trigger_policy.py:202–266,289–332`; `track2_backend.py:742–761,891–899`.

F4 can actually increase the **scope of negative attribution** when additional members become move-exposed: the same unsafe observable pair may be seen from multiple CAVs' reward domains, though it is deduplicated within each CAV's calculation. This is not proof joint moves are penalized on net; it explains why F4 is not a cooperation bonus. Removing it is not an evidence-led way to create useful synergy.

### 4.3 Why simply increasing neighbour reward is insufficient

The optimized quantity is local, normalized speed deficit plus temporal safety cost. The registered endpoint is extra source-inclusive network-delay benefit beyond the **best** same-state member intervention. These can disagree even with perfect optimization. More neighbour weight neither compares joint against solo nor repairs teacher/action mismatch, sparse joint opportunities, departed-agent credit, or coarse 60-second trigger cadence.

## 5. Literature: what is relevant, and what it does not guarantee

Web research used primary arXiv records, selected full HTML papers, and Crossref metadata. The recommendations below are adaptations; none of these papers demonstrates this project's exact group-over-best-solo criterion merely by calling a method cooperative.

### 5.1 Difference rewards and counterfactual credit — closest objective fit

**Difference rewards:** replace an undifferentiated team score with an agent's marginal effect relative to a default action: `D_i = G(a) − G(a_-i,c_i)`. For delay costs, `D_i = J_without_i − J_group`. This tells a second helper whether it improved the coalition, even when its own speed fell. The default should be common-base control, not vehicle removal, so traffic demand stays matched.

Castellini et al., **Difference Rewards Policy Gradients** (AAMAS2021 extended abstract; Neural Computing and Applications2022), [arXiv:2012.11258](https://arxiv.org/abs/2012.11258), develops Dr.Reinforce and a learned reward-network alternative. The paper distinguishes reward-function differencing from learning a full Q critic and discusses counterfactual estimation/generalization difficulties. For Clean2, delayed queue consequences are a reason to use measured rollout differences, not assume a one-step reward model captures the full effect.

Foerster et al., **Counterfactual Multi-Agent Policy Gradients** (AAAI2018), [arXiv:1705.08926](https://arxiv.org/abs/1705.08926), uses a centralized critic and an agent-action-marginalized baseline while holding other agents' actions fixed:

`A_i = Q(s,a) − sum_a'i pi_i(a'i|history_i) Q(s,(a_-i,a'i))`.

**Project application:** start with sparse measured member-reversion labels and a small macro-level critic; full COMA requires consistent joint-action records, enough counterfactual support, and changes to the current per-CAV trigger objective. It does not automatically force complementary actions if those are absent from the vocabulary.

### 5.2 QMIX/QPLEX — scalable joint value, but not a drop-in reward patch

Rashid et al., **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning** (ICML2018), [arXiv:1803.11485](https://arxiv.org/abs/1803.11485), mixes local utilities into a centralized joint Q under monotonicity, allowing decentralized greedy action selection. It addresses team-value learning and scalability. Monotonicity restricts representation of some action interactions/tradeoffs; “one yields while another exploits the gap” should be tested as a joint payoff problem rather than assumed to fit.

Wang et al., **QPLEX: Duplex Dueling Multi-Agent Q-Learning** (ICLR2021), [arXiv:2008.01062](https://arxiv.org/abs/2008.01062), provides a complete IGM function class through duplex dueling decomposition. This expands value representation within the individual-global-max framework; it does not remove all decentralized observability or data-support constraints.

**Project application:** either could be applied to a discrete coordination layer while retaining MAPPO below it. Both require an off-policy value-training/replay/target-network path and new checkpoint state. Neither is a credible one-day first repair when executable joint labels and opportunities have not yet been certified.

### 5.3 Coordination graphs / joint-action heads — best small architectural option

Böhmer, Kurin and Whiteson, **Deep Coordination Graphs** (ICML2020), [arXiv:1910.00091](https://arxiv.org/abs/1910.00091), factors joint value into individual utilities and **pairwise action payoffs**, maximized by message passing. Its relative-overgeneralization experiments explicitly show why individually unattractive actions can be valuable together. This closely matches a temporary yielding sacrifice that pays off only if a partner synchronizes.

**Project application:** for two-member local merge groups, directly enumerate a small shared menu of **role-specific joint templates** or use a pairwise payoff head. With <=4 members, a sparse compatibility graph or bounded macro slate is more realistic than a complete vehicle-level algorithm rewrite. Existing wall/ramp macros already coordinate execution; the missing features are complementary roles, feasible opportunity generation and coalition-aligned value/credit.

### 5.4 Communication and role assignment — V2V is already a prerequisite, not the missing solution by itself

Foerster et al., **Learning to Communicate with Deep Multi-Agent Reinforcement Learning** (NeurIPS2016), [arXiv:1605.06676](https://arxiv.org/abs/1605.06676), introduces RIAL/DIAL for learning communication useful to shared tasks. The lesson is task-conditioned communication, not simply increasing message dimensions.

Wang et al., **RODE: Learning Roles to Decompose Multi-Agent Tasks** (ICLR2021), [arXiv:2010.01523](https://arxiv.org/abs/2010.01523), learns restricted role action spaces and a slower role selector from action effects. Clean2 can use simpler geometry-assigned roles first: **mainline gap-maker, ramp inserter, upstream protector**, each with a target merge time and different allowed profile. Full learned role discovery is unnecessary for a one-day experiment.

**Project application:** use the existing V2V link for a common macro ID, participant/role assignment, target gap/merge time, commitment expiry and acceptance/acknowledgement. A five-second predictive intent and a 60-second macro commitment are different semantics. Test the usefulness of explicit commitment intent before adding more communication capacity.

### 5.5 Traffic-specific merging, gap creation and curriculum

- Chen et al., **Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic** (2021 preprint, revised2022), [arXiv:2105.05701](https://arxiv.org/abs/2105.05701). The full paper combines parameter sharing, local rewards, invalid-action masking, priority-based safety supervision and an easy-to-hard traffic-density curriculum. It explicitly describes through-lane vehicles accelerating/decelerating to create space and ramp vehicles adjusting speed and cutting in. **Transfer:** retain masking/safety, but make curricula coordination-required and verify solos; local rewards alone do not establish synergy.
- Xu, Feng, Zhang and Li, **A Grouping Based Cooperative Driving Strategy for CAVs Merging Problems** (2018), [arXiv:1804.01250](https://arxiv.org/abs/1804.01250). Groups preserve passing orders to reduce sequencing search. **Transfer:** assign useful order/roles and a shared merge slot rather than form an arbitrary nearby set and brake everyone identically.
- Kumaravel, Malikopoulos and Ayyagari, **Decentralized Cooperative Merging of Platoons of Connected and Automated Vehicles at Highway On-Ramps** (ACC2021), [arXiv:2002.04826](https://arxiv.org/abs/2002.04826), [DOI](https://doi.org/10.23919/ACC50511.2021.9483390). Joint scheduling of platoon entry times plus optimal longitudinal control supplies a concrete mechanism for coordination. The evaluated assumptions include preformed stable CAV platoons and ideal communication; its performance numbers should not be transferred to this mixed-traffic setting.
- Li, Zhou, Ye, Jiang and Wang, **Sequencing-enabled Hierarchical Cooperative CAV On-ramp Merging Control with Enhanced Stability and Feasibility** (2023, revised2024), [arXiv:2311.14924](https://arxiv.org/abs/2311.14924). Upper-level sequencing and lower-level distributed MPC/virtual car following coordinate timings and spacing. **Transfer:** a role/time-slot macro above the frozen low-level controller is structurally closer to successful merging coordination than a common negative box.
- Scarinci, Heydecker and Hegyi, **Analysis of Traffic Performance of a Merging Assistant Strategy Using Cooperative Vehicles**, IEEE T-ITS16(4),2094–2103 (2015), [DOI](https://doi.org/10.1109/TITS.2015.2394772). Traffic-specific merging-assistant/gap-creation precedent; bibliographic identity verified via Crossref and the platoon paper's references. It is a mechanism reference, not evidence of this project's best-solo endpoint.
- Ibork, Won and Das, **Z-Merge: Multi-Agent Reinforcement Learning for On-Ramp Merging with Zone-Specific V2X Traffic Information** (2025), [arXiv:2511.14910](https://arxiv.org/abs/2511.14910). Coordinates lane-changing and gap adaptation with a hybrid action space and zone-specific V2X information in SUMO/MOSAIC. **Transfer:** the relevant difference is executable action/role/zone design; Clean2 already has V2V, and adding learned lateral control would be a new-run action-space change.
- Wu et al., **Flow: A Modular Learning Framework for Mixed Autonomy Traffic**, IEEE T-RO2021, [arXiv:1710.05465](https://arxiv.org/abs/1710.05465). System-level improvement at low AV adoption can arise from a small local controller. **Interpretation here:** a large network gain from one CAV is plausible and useful; it is not by itself multi-CAV synergy.

### 5.6 Synergy-shaped reward: a concrete project adaptation

For the same measured state, background, horizon and CRN, define:

```
T = J_base - J_group
S = min_i J_solo_i - J_group
M_i = J_group_with_i_reverted_to_base - J_group
objective = normalized(T) + lambda * normalized(S) - unchanged safety costs
```

`M_i` is second-helper marginal credit; `S` is the registered group-over-best-solo target. For two actors, the member-reversion branches are the two solos, so the same branch set can support both. For >2 actors, a solo and a leave-one-out coalition are different interventions; do not substitute one for the other.

Start with a sparse, prospectively fixed subset of **ordinary training opportunities**, measured in SUMO by a later authorized worker, or use existing verified **training-bank** branches as supervised credit labels. Do not train on these diagnostic outcomes. Preserve the negative side of synergy so harmful joint actions can be suppressed, and retain total benefit so the policy cannot win merely by making solo counterfactuals worse.

Important implementation distinctions:

- A COMA action-marginalized baseline is a variance-reduction device with a particular conditioning structure. A best-solo shaping term generally **changes the objective**; it is not automatically an unbiased baseline or potential-based shaping.
- If using a learned critic/WM for dense counterfactuals, validate signs and ordering against measured group/solo returns on separate ordinary TRAIN contexts. Existing `group_credit=OFF` is not evidence such a method has already failed; switching it on without that calibration would be unsupported.
- Using only a subset of solos underestimates the best-solo gain and therefore **overestimates synergy**. With cost notation, `min_subset J_solo >= min_all J_solo`. Partial counterfactuals are useful heuristics/credit estimates, not certificates of positive full synergy. Confirm promising labels against **all** member solos.
- Counterfactuals must retain complete recorded horizons and include full-prefix replay cost. Cropping the horizon until a favorable interaction appears would answer a different question.

## 6. Ranked concrete levers for THIS project

Expected effects below are directional mechanisms, not numerical promises. All changes apply to **B and R with identical code, hyperparameters, sampling rules, verification effort and switch boundary**; each keeps its own verified teacher corpus. A remains the registered reference unless a successor design explicitly changes it.

### Rank 1 — Certify the executable teacher, not merely its native ancestor

**Change:** add a label-fidelity/admission step to the existing F3 bank path. For each candidate lesson, bind the actual executable macro, its local visibility/roles, issued-command effect, common-window group/base/all-solo outcomes, safety and synergy. Prefer 2-actor native profiles that the online action set can actually represent; retain negatives and near-zero controls in the research dataset. Do not automatically turn every rich native plan into a wall CE target.

**Why first:** B already has native positive-synergy examples; the unproven bridge is `native profile → projected label → online proposal → physical effect → retained synergy`. Current CE can improve while this bridge fails.

**Expected synergy effect:** high information value; potentially high learning benefit if positive executable labels survive. A zero-survivor result would diagnose the current vocabulary before spending most remaining training budget.

**Engineering/cost:** approximately 2–4 h to bind/audit existing labels, **0 sim-s** offline. A proposed bounded validation batch of **8 contexts/arm ×2 arms ×2 verification seeds ×(group+base+2…4 solos) ×1,200 s** costs **153,600–230,400 sim-s** at full episode length. This is a conservative prospective estimate, not a quote/dispatch; existing reusable exact branches may reduce it.

**Risk:** too few survivors, overfitting to a small bank, or B/R family support mismatch. Verify matched horizons rather than recycle legacy gain fields.

**Clean2 compatibility:** the existing `bank_breadth.inputs`/PLAN_FIX handoff can consume additional verified examples at a **prospectively registered update boundary**, provided the existing action semantics and common B/R family procedure are retained. Changed projection semantics require new source/config bindings and a labelled successor phase. This report applies neither.

### Rank 2 — Create feasible opportunities before singleton acquisition consumes them

**Change:** use an observable opportunity graph and role-aware ordering to propose a few disjoint 2-CAV coordination options **before** the current lexicographic singleton acquisition. Preserve ownership masks and exact conditional slates for PPO. Use approach-to-merge/remaining-time information to decide when a pair is meaningful; audit precision-consistent zone endpoints. Keep none and single-CAV actions available. Consider an earlier event-driven decision opportunity rather than blindly shortening the global trigger interval.

**Why:** joint offered in only170/2,097 training rows and1/23 original-group diagnostic rows; B chooses a joint option in56.5% of offered training rows and the one diagnostic opportunity. F2 already fixes rejected starts; its next limitation is **opportunity coverage**, not refusal after selection. Stable ID ordering lets early singleton choices own partners before later agents can propose a coalition.

**Expected synergy effect:** high on joint exposure, uncertain on positive synergy until Rank1's measured label bridge exists. Avoid rewarding joint activity merely for being joint.

**Engineering/cost:** approximately 4–8 h for bounded pair arbitration plus deterministic ownership/slate checks; **0 incremental sim-s** for existing scheduled collection, plus a prospective **14,400–20,000 sim-s** qualification envelope. Any extra trigger frequency affects computational wall time, not simulated horizon; verify it does not create excessive reselection/churn.

**Risk:** more bad pairs, changed CRN/candidate distribution, short-opportunity moves, and inconsistent PPO likelihoods if slates are regenerated. Repairing coordinate precision alone cannot recover partners already downstream or on the wrong lane.

**Clean2 compatibility:** tensor shapes can remain checkpoint-compatible, so a registered successor continuation is technically feasible. It changes behavior and the registered fixed source/distribution, so it is **not an unchanged Clean2 result**. Do not silently switch under `no_outcome_conditioned_changes=true`.

### Rank 3 — Replace common braking with a small complementary-role macro head

**Change:** for two-CAV opportunities, choose a shared merge slot and a small joint template: mainline yields while ramp tracks the slot; mainline clears while ramp waits; or upstream protector plus gap-maker. Allow per-role onset/hold/depth, including maintain/release and bounded positive corrections where already permitted by the command chain. Share a macro ID and role commitment through existing V2V. Retain the frozen low-level MAPPO and final safety writer.

**Why:** current wall/ramp_assist applies the same negative box to all actors; gap_open has only one actuator. Some native synergistic plans require different member profiles, six actors, or longer durations. Increasing CE cannot recover omitted action dimensions.

**Expected synergy effect:** highest structural upside if measured 2-actor complementary examples exist; no guarantee in states already downstream. This is the smallest architectural change that directly addresses complementary actions.

**Engineering/cost:** approximately 6–12 h for a tightly scoped two-actor head/template library, plus Rank1-style shared verification **153,600–230,400 sim-s** and a qualification envelope. The same validation batch can cover Ranks1–3; do not add their costs three times.

**Risk:** invalid gap slots, role ambiguity, missing partner acceptance, safety-clipped roles, and newly enabled harmful actions. A raw bonus for two acting members would reward useless braking.

**Clean2 compatibility:** new action descriptors/roles/profile parameters generally require **a new run or explicitly versioned successor architecture**, with new checkpoint/optimizer state and B/R symmetry. It cannot honestly be presented as the original unmodified u24 test.

### Rank 4 — Sparse measured difference/synergy credit at the macro boundary

**Change:** train the initiating macro decision on common measured coalition outcomes; give second-member residual/role decisions marginal `M_i` credit, and add a bounded, explicitly registered `S` term while retaining total benefit and F4. Use same-state ordinary-training counterfactuals, not correlations between accepted-move and none rewards.

**Expected synergy effect:** directly aligned with the registered endpoint; likely more informative than multiplying the neighbour weight. Requires actions and data from Ranks1–3 that can genuinely produce interaction.

**Engineering/cost:** approximately 4–8 h for sparse two-actor labels and macro credit plumbing. For **8 opportunities/arm ×2 arms ×2 seeds ×4 branches ×1,200 s**, full group/base/two-solo measurement is **153,600 sim-s**. Reusing Rank1's verified training batch avoids duplicate measurement. A full dense COMA critic with robust off-policy counterfactual coverage is a larger undertaking.

**Risk:** high variance, max-solo selection bias, wrong background/horizon, partial-solo optimism, reward scaling, and stale critic labels. Existing native-bank branches are valid for their historical interventions, not automatically counterfactual outcomes for a newly changed policy.

**Clean2 compatibility:** a versioned reward/credit continuation is technically possible at a registered boundary if all bindings/monitor histories are updated; scientifically it is a **new recipe phase**. New centralized critic tensors or joint policy heads favor a new run. Preserve the original u24 criterion rather than replace it with total benefit.

### Rank 5 — Coordination-required curriculum and broader matched practice

**Change:** select fresh ordinary TRAIN contexts by observable geometry/arrival conflict, then certify that useful joint execution beats every solo. Start with controllable two-actor opportunities; increase HDV uncertainty/traffic density only after the mechanism works. Rotate distinct contexts before repeats, match family capacities for B/R, and reserve separate ordinary TRAIN audit contexts. Use proposed verified imports, not targeted additions of the seven diagnostic states.

**Expected synergy effect:** medium-to-high transfer benefit and more informative use of each training second. Current u6's 83.9% practice concentration and two duplicated none-labelled episodes make “more practice” alone an inadequate remedy.

**Engineering/cost:** approximately 2–4 h for selector/curriculum plumbing once verified labels exist. Replacing existing scheduled starts is **0 incremental collection sim-s**, but acquiring/certifying new examples costs the Rank1 batch or additional registered search budget. Match **usable** context counts/support, not only requested fractions.

**Risk:** artificial training distribution, loss of ordinary-state performance, and selecting apparent synergy from noise. Pure density curriculum may still be solvable by one CAV and fail the cooperation objective.

**Clean2 compatibility:** compatible verified bank imports can use the existing F3 boundary mechanism. A changed state-selection/curriculum rule is an amended successor phase, with its own description of exposure. Changing the evaluation panel would require a new design and would not rescue the registered result.

### Rank 6 — Add a useful baseline and delayed team credit; defer full value decomposition

**Change:** fit a small local/coalition value baseline for the continuous path and use explicit macro outcome windows so useful consequences after a participant exits can be credited. Log proposer/member advantages by actual actuation and role. This is distinct from increasing the KL cap or imitation weight.

**Expected synergy effect:** variance reduction and better delayed credit, but indirect without executable joint opportunities. Larger layers, more entropy, or more generic neighbour reward are lower-priority levers.

**Engineering/cost:** approximately 3–6 h for a small critic/baseline prototype, **0 incremental physics** if trained on existing ordinary training returns, plus qualification. New return attribution changes require careful terminal semantics. A QMIX/QPLEX/DCG learner replacement is **>1 day risk**, new replay/target networks and additional sample demand; deprioritize it under this budget.

**Risk:** baseline error, credit leakage, uncalibrated long-horizon predictions, changed optimizer state, or faster learning of the wrong signal.

**Clean2 compatibility:** new critic/return semantics require a labelled successor or new run; a diagnostic-only advantage breakdown is compatible immediately and costs no physics.

## 7. One-day decision and budget discipline

**Recommended decision sequence:**

1. **First 2–4 h, zero physics:** bind matched-window native labels, project them, test online generability/visibility and precision boundaries, and inspect actual issued-command exposure. This report supplies much of that diagnosis already. Do not fit current policies to the diagnostic outcomes.
2. **Next 4–8 h of engineering, if a follow-up is chosen:** implement only the smallest role/opportunity/credit package supported by ordinary training examples. Prefer a 2-actor macro and sparse measured credit over a broad MARL rewrite. Keep B/R completely symmetric.
3. **A later authorized validation batch:** allocate at most approximately **0.15–0.23M sim-s** to full group/base/all-solo qualification on new ordinary training opportunities. Train only if executable, safe, positive-synergy examples survive; a zero-survivor result points toward action/state redesign, not a higher imitation coefficient.
4. **Use remaining scheduled learning and evaluation:** keep total benefit, best-solo benefit, synergy, offered/accepted/physically-joint rates, and safety separate. No success declaration from joint frequency or a better B–R contrast.

**The owner's approximately 2.5M remaining is not all discretionary.** The captured Clean2 budget allocates **2,795,800 of3,184,160 sim-s**, with **388,360 unallocated at registration**. Planned training alone is1,843,200, fixed interim/final345,600 each, replicated diagnostics143,760, plus lessons/gates/smoke. The separate plan-fix reservation is also protected. These are reservation/plan facts, not a fresh live-ledger balance.

A rough full A/B/R update with16×1,200 episodes each and six matched practice references for each B/R arm is **72,000 sim-s**. An eight-update successor at that mixture would therefore cost approximately576,000 before its additional tests/counterfactuals. It cannot be casually added alongside every remaining Clean2 obligation just because the global unused balance is approximately2.5M. A short successor can fit only after explicit allocation/reallocation; no reallocation or run change was made here.

The current configuration declares **`no_outcome_conditioned_changes=true`**. “Can switch at a registered boundary” above means technical feasibility under an explicit prospective owner amendment and new bindings, not existing permission or preservation of the original unchanged-run claim. Compatible verified-bank updates are the smallest existing integration point; architecture/reward changes should be reported as a successor phase or new run. The fixed **u24 positive synergy CI lower>0 plus safety** remains the relevant Clean2 criterion.

## 8. Bottom line

**Why no group benefit?** The metric subtracts a best solo that already captures essentially all useful improvement. Most diagnostic groups have passed the library's useful cooperative geometry or do not match its roles; only one joint macro is even offered. Training has genuine joint actuation, but it is sparse on normal episodes, concentrated in practice, imitated through a lossy profile projection, and optimized without second-helper counterfactual credit.

**Where is B's benefit?** Mostly wider on-road traffic delay following interventions on ramp.22/986048, ramp.38/986042 and ramp.18/986046. They select individual easing while the continuous residual also acts. One ramp CAV per state reproduces the gain; the receipts do not establish a wall/gap-opening joint mechanism or earlier merging.

**How to enhance cooperation?** Make a useful, feasible, executable **complementary two-actor action** available in a state that requires it; certify its synergy on matched full counterfactuals; then teach and credit that exact action. The literature supplies methods for each link. More V2V, more wall labels, or more team-reward weight alone does not repair the missing links.

## Appendix A. Complete B/u6 branch-J audit

All J values are veh-s. Solo index follows the stated group order. These values reconstruct every gain in §1, including the harmful/zero states.

- **r0/s0**, group [ramp.22,mainline.100]: base3153.742231791035; group3154.133495233199; solo0=3154.1208464359133; solo1=3153.7422338021724.
- **r0/s1**, same group: base=group=solo0=solo1=5459.583808123821.
- **r0/s2**, group [mainline.127,mainline.126,mainline.129]: base1883.6057702311819; group1883.3551322476176; solo0=1884.6101462672416; solo1=1884.5918339589855; solo2=1883.4345174038217.
- **r1/s0**, group [ramp.20,ramp.22]: base62054.09438227486; group58767.80539257626; solo0=62054.09461265799; solo1=58767.805429405336.
- **r2/s0**, group [mainline.267,ramp.38]: base54036.489354673104; group53897.88896846931; solo0=54036.489354673104; solo1=53897.88896846931.
- **r3/s0**, group [mainline.108,ramp.18,mainline.109]: base34375.81279737048; group34168.656925709736; solo0=34375.81283491818; solo1=34168.59079243465; solo2=34375.824957039746.
- **r4/s0**, group [mainline.264,ramp.51,mainline.267,mainline.270]: base83494.67997176785; group83494.68028635118; solo0=83494.67997176785; solo1=83494.67997176785; solo2=83494.68016765214; solo3=83494.68028635107.

## Appendix B. Evidence map and reproducibility

Primary remote evidence:

- `ROOT/track2_diagnostics/replicated_u006_live.json`, receipt SHA field **b2f5c2bfee8ccf468c52908c4cf3a0e401338b0b5049cf9ecd2ffbfababb5d83**. Contains all64 B/R branch receipts and exact result paths, state/world weighting, both metric reports and bootstrap seeds.
- `ROOT/ISOLATED_REGISTRATION.json`, registration SHA field bound by diagnostics **c0a6c035aeb815d76eedfecb8527632e68ceef48aa297bc4eb5718dc63512fc7**; seven `ROOT/registered_states/rRR_sSS/RECEIPT.json` records. Snapshot payloads were not needed for this analysis.
- B branch tapes follow `ROOT/branches/p3_replicated_u006_B_rRRR_sSSS_bNN/p3_replicated_u006_B_rRRR_sSSS_bNN/<all|soloN|reference>/RESULT.json[.gz]`. `RESULT.json` can be a lossless-gzip envelope; it is not a zero-step result.
- `ROOT/B_UPDATE_001.json`…`009.json`, corresponding R receipts; B/u6 and B/u9 were also captured with full learning/fix receipts. Training's canonical config SHA field is **f6489716c060f0a647148bcf247bdd6051635b712396a0eda1eb3d75b9af563d**. The raw `CONFIG.json` byte-text SHA captured22:25+08 is **827d24dd9708adc7cba649b24b48fa74e65a7f9255766f40a67c49038c091bb6**; these are different hash definitions, not interchangeable.
- `ROOT/jobs/t2c_warm_u006_B_<world>/RESULT.json[.gz]` and `t2c_warm_u006_pNN_B_<world>/RESULT.json[.gz]`, exactly10 normal+6 practice B episodes. Prefix removal uses update-bound practice t0. Recalculated decisions, CAV outcomes, nonzero offsets and physical-change counts match the production receipt exactly.
- `ROOT/replay_bank/{B,R}/PRACTICE_PLAN_*_001.json`…`009.json` retain exact example/plan/verification/snapshot bindings. These supplied native plans and per-seed matched synergy, without recomputing admission from unguarded project imports.
- Registered source-plan solos: `ROOT/replay_bank/B/confirm_solo/{verifications,branches}`; second-seed links/receipts under `confirm_solo_replication`; base confirmations plus `confirmations_safety0426` where present. Original source group/reference tapes were read only through the permitted `_branches` paths in the registered receipts, including the source-bank LS label trees.
- `ROOT/SMOKE_FIX_RECEIPTS_PASS.json` proves F1 full-strength physical actuation, F2 no refusal, F3 actual context/family/pool counts, F4 safety reward. `ROOT/CLEAN2_BUDGET.json` supplies prospective allocations.

Source references use SRC and original captured line numbering; no deployment code was edited. Pure local projection evaluation used only `_plan_offsets`, `_project_window`, `from_plan`, and `none_move` extracted from captured `moves.py`, with the registered constants cross-checked against `variety.py:64–72`.

Local analysis helpers and machine-readable captures are under **`E:/Temp/opencode/sgroup_*`**, notably `sgroup_diagmetrics.jsonl`, `sgroup_trainmetrics.jsonl`, `sgroup_bankcompact.jsonl`, `sgroup_nativepaired.jsonl`, `sgroup_mechanism.jsonl`, and `sgroup_geometry.jsonl`. They are scratch provenance; the report's conclusions are backed by the remote immutable receipts identified above. Temporary helpers were only created locally. No borrowed results or literature abstracts were treated as new simulation evidence.

## Resume verification — 2026-11-06

The fresh session recovered this complete report and its local `sgroup_*` captures before doing further work. At initial inspection the report was untracked under the ignored `runs/` tree; during resumed checks, commit `46885cc8` appeared preserving its original 485 lines. That commit is retained, and this additive verification note records the independent local checks rather than repeating the analysis. No SSH connection, remote command, process inspection, job dispatch, new web research, simulation or charge was needed in this resumed session. The evidence cutoff in §0 remains unchanged.

`python -B E:/Temp/opencode/sgroup_verify.py` passed against the saved captures: all B/R branch gain and synergy calculations match their replicated receipt; all 16 B/u6 training tapes reconcile to the update's 2,097 decisions, 48,424 nonzero-offset member-ticks and 38,387 physically changed commands; 96 accepted joint commitments include 93 physically joint executions, of which 15 are in normal episodes; all eight admitted native teaching examples have positive matched-window synergy and positive lower bounds. The 14-line opening summary and six ranked levers also pass the artifact checks. This verifies the recovered evidence, not later remote progress or the u24 result.
