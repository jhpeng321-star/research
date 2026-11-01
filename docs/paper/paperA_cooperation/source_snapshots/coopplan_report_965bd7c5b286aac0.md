# S-COOPPLAN — exact-model planner improves delay and executes substantial cooperative actions
Complete 2026-11-01, branch `c4/coopplan`: H120 original12 + extension24 =36 fresh development worlds; separate MAIN02:21 H300 confirmation18 + wall6 =24 additional worlds. All60 pairs complete.

## Main finding
The SUMO-oracle cooperative planner reduces both paired J600 and J900 on the pooled36-world panel; the extension alone also improves both scores.
It executes52 group actions with complete counterfactuals;24 have positive superadditive C_G, including a joint action that helps when BOTH member-alone actions harm.
Pooled mean C_G is positive but its clustered CI includes0: **robust full-team delay benefit and concrete cooperation cases; no robust positive population-mean superadditivity claim**.
This is the perfect-model benchmark for the pluggable WM-PHYS predictor, whose efficacy/calibration is not measured here.
**H300 amendment:** walls now execute twice, including one exact positive-superadditive wall (C_G+181.571). Both new panels improve J600; J900 remains uncertain, and H300 mean C_G is not positive. Separate estimates below retain these limitations.

## Frozen experiment and panel completeness
Original DEV983100–983111: one world per .05/.10/.20/.40 CAV share ×1900/2400/2700 demand cell; extension DEV983200–983223: two additional worlds per cell.
All36 paired worlds complete, never fitted/tuned on; exact guards and three-host registration/range/safe-name exposure audits passed before physics.
SUMO1.27.1 deterministic full-rerun oracle K1 vs pure frozen speed-only MAPPO169; identical arrivals/noise/native lane logic/safety.
H120, cadence60, decisions t300..960, then MAPPO-only after1020 through1200; warmup300 excluded from J600/J900.
Candidates: MAPPO O0, coordinated slow O1, cross-lane wall O2, yield gap O3, longer gap O4; inherited geometry/magnitudes and fixed setting rotation.
Choose minimum predicted team delay only when strict paired P(beats MAPPO)>=.8 AND lower mean; exact K1 means strict improvement.
COOPFREE had no complete learned TRAIN checkpoint at either panel's predispatch freeze; no untrained substitute or late checkpoint change.
Extension reservations protected all full pairs and both solo obligations before prediction; **zero quota-pruned decisions**, so the nondefault selection rule remained identical across cohorts.
Skipping provably O0-only lookaheads changes no executed action; cadence60/H120 retained throughout.

## Paired full-team delay — saving = MAPPO minus planner, veh-s/world
J includes on-road speed deficit plus source waiting.95% percentile CIs use10000 whole-world resamples, RNG20261101.
- **Original12:** J600 **+396.280 [0.894,1128.773]**; J900 **+1195.148 [-568.178,3724.114]**.
- **Extension24:** J600 **+774.471 [199.815,1480.502]**; J900 **+1778.158 [714.789,3076.597]**.
- **Pooled36:** J600 **+648.407 [209.699,1173.616]**; J900 **+1583.821 [588.246,2765.414]**.
World-level contrasts remain heterogeneous; positive averages do not mean every world improves. No decision/episode pseudoreplication.

## Complete cooperation evidence
C_G=(J0−JG)−sum_i(J0−Ji), exact H120; member-alone twins retain the joint targets and replay the entire executed prefix.
All52 executed two-member group actions have BOTH solos (104total):19original actions +33extension actions; no sign/outcome exclusions.
Original13 unsampled actions received26new twins (22560sim-s); all original O0/joint episodes and six already-paid twin pairs were reused.
- **Original12:** event-weighted mean C_G **+46.112 [3.665,109.339]**;10/19positive,6/12worlds active.
- **Extension24:** **+7.757 [-86.168,112.091]**;14/33positive,10/24worlds active.
- **Pooled36:** **+21.771 [-42.074,90.467] veh-s/action**;24/52positive,16/36worlds active.
C_G intervals resample whole worlds with ALL their actions, including no-action worlds in the sampling frame; the event mean uses the resampled action count.
Original12 has1/10000 zero-action draw with undefined event mean, reported/excluded from percentiles; extension/pooled have none.
The original first-action-only six-sample estimate was−38.379 [−94.759,−1.022],2/6positive; this different estimand and its immutable artifacts are retained, not replaced by a selected positive subset.
Descriptive largest positive case: **983223/t960/O4**, joint saving D_G **+568.333707**, solo savings **−302.399975/−18.034632**, hence **C_G+888.768314 veh-s**. Cooperation changes both individually harmful actions into a beneficial joint action.
Largest negative case is also retained:983210/t780/O1, D_G+560.957650, solo savings+560.957144/+623.253167, C_G−623.252661; beating MAPPO is not sufficient for superadditivity.
Across432 decisions: MAPPO380; O1slow17; O3yield14; O4longer-gap21; O2wall0. The extension contributes O1=12/O3=10/O4=11.
Unchanged COOPBEHAV on all52 same-prefix O0-reference comparisons: **7supported yields,112supported gap events,2supported cross-lane-metering events,345supported wave-damping events**.
Whole-episode observational candidates (69yield/4594gap/448cross-lane/9110wave) are not causal counts; supported metering does not establish an executed O2wall result.

## Safety, exactness and CPU verification
Pooled36 scored900s episodes per arm, MAPPO/planner: **collisions0/0, teleports0/0**; TTC<3 exposure21662.0/20280.5veh-s; hard braking765.0/733.0veh-s.
Extension24 alone: collisions0/0, teleports0/0; TTC14657.5/13657.5; braking545.0/511.0. These are descriptive totals, not universal safety guarantees.
Original12 safety remains unchanged: TTC7004.5/6623.0; braking220.0/222.0. Pooled safety clamped1446/3557 requested member-command records.
All **603full-prefix checks pass** (247original including2mechanics +356extension/backfill); original first production baseline also matched the1200s mechanics reference bitwise.
Extension frozen host suite27/27 on EACH host/0skips plus real S-RB-DECIDE parity, SUMO process creation blocked; local CPU/analysis35/35PASS.
Final QA verifies source/design/panel identity, all36pairs/432decisions, all52credits, paired best-action choices, CSV/summary recomputation, zero outstanding reservations, and exact accounting.
Three final figures were rendered and visually checked. Observed branch timer median11.918s/max24.474s excludes native300s warmup/reset; this remains synchronous offline planning, not a real-time claim.

## H120 exact accounting and operational closure
**H120 actual=charged565140sim-s <=800000;234860remaining before H300 dispatch**, no failure/incomplete attempt/physics retry/refund/top-up.
Original220980 retained; extension344160 =57600paired episodes +22560original missing twins +53520new group twins +210480lookaheads.
Extension hosts: cpuQ78480/193000cap; gpuA101520/193000; gpuR164160/193020. All three COMPLETE, zero active attempts/outstanding reservations.
Original704760upper and MAIN01:28 remaining579020allocation preserved; atomic admission stayed below the original800000global cap.
cpuQ briefly waited below2GiB free disk; headroom returned and existing workers resumed automatically, without source/admission/PID changes.
nice19/CUDAhidden/single-thread/shared-resource yielding; no training, prohibited truth/assignment/path reads, new venv, deletion, push or peer PID actions.

## Predictor interface, reproducibility and deliverables
`Predictor.predict(observable_state,candidate_actions,horizon,K)` returns paired distributions of team delay, breakdown, member delay/effects and safety, with draw IDs shared across actions.
The SAME `Planner` accepts exact `SumoOraclePredictor` or callback-bound `WMPhysPredictor`; WM uses one complete hidden-driver/arrival draw per k across all actions, including identity-bound tensor payloads.
WM callbacks must bind RB-INT's immutable runtime and public motion/history/request-counter state; absence fails explicitly. No WM qualification or calibrated-confidence claim is implied by the toy/ABI parity checks.
Original runtime/design:a0934600/53b0edf4ea8e6fc72165691727e923505ed6a805af9c0294efd63f2ab3389a5f; extension source:c2646eb1/designSHA17356a3ee726ffea189da8d1f0e10cb376e4eae1293cfdaf1014c68a8a826918.
Designs: [COOPPLAN_DESIGN.md](COOPPLAN_DESIGN.md) (63lines at original freeze, now65with pointers), [extension design](COOPPLAN_EXTENSION_DESIGN.md), [H300 design](COOPPLAN_H300_DESIGN.md) (each<=70lines); execution/API: `HANDOFF_COOPPLAN.md`, `tools/coopplan/`.
Final artifacts: `coopplan/extension/final/` — RECEIPTS/SUMMARY/ALL_GROUP_CREDITS/BEHAVIOURS/QA JSON plus original12/extension24/pooled36 CSVs; all result bytes are Git-preserved and SHA-bound.
Original first-sample receipts/figures remain under `coopplan/final/`; completed original all-group milestone remains under `coopplan/extension/original_all_groups/`.
Final plots: [cohort paired CIs](coopplan/extension/final/paired_cohorts.png), [all world contrasts](coopplan/extension/final/paired_worlds.png), [all-group cooperation](coopplan/extension/final/all_group_cooperation.png).
Read-only reproducibility: `python -B -m tools.coopplan.extension_report --verify`; example extrema: `--examples`. Never relaunch or recollect an existing immutable attempt/result directory.

## MAIN02:21 H300 continuation — COMPLETE, separate confirmation and wall panels
Prospective design/source03256981; all three frozen hosts30CPU/0skips plus actual WMDECIDE H300 parity PASS before one-shot dispatch.
Fresh never-fit confirmation18DEV983300–983317: TWO worlds per .4/.2/.1 x2700/2400/1900 cell, high-share first; separate wall6DEV983318–983323: .2/.4 x2400/2700, extra repeats in .4cells.
ALL new predictions and ALL-executed-group twins useH300; cadence60/same inherited grid/safety/noise/strict0.8 gate. Completed H120 panels above are preserved.
Prospective234720continuation=57600paired +177120adaptive lookaheads/complete solos;9780/world,78240per host, global upper799860/800000,140unallocated.
Fixed per-world quotas protect every full pair and both solos before prediction; cyclic pruning/MAPPO fallback expected. Availability of O2 is logged before budget admission at every decision.
Corrected HOLD COOPFREE learned checkpoint absent at shared freeze (legacy weights excluded); no outcome-dependent late switch.
New immutable `_v3` supervisors cpuQ693091/gpuA348726/gpuR768580 all COMPLETE, no STOP/incomplete attempts/outstanding reservations; no replay/relaunch/refund.
H120-versus-H300 summaries compare different fresh panels and different budget availability, not a paired causal horizon effect. Full-rerun predictions remain synchronous offline.

### Paired H300 delay and ALL-group cooperation credit
Same score units and10000 whole-world bootstrap RNG20261101; all executed groups have both member-alone twins, no sign exclusions.
- **Confirmation18:** J600 **+767.979 [157.470,1606.905]**; J900 **+821.068 [-944.174,2576.533] veh-s/world**.
  Ten group actions (O1=2/O2=1/O3=3/O4=4);5/10 positive C_G in9/18active worlds.
  Event-weighted mean C_G **-100.774 [-1071.707,785.406] veh-s/action**.
- **Wall6:** J600 **+1481.955 [472.316,2510.373]**; J900 **+1437.522 [-489.175,3975.582]**.
  Four group actions (O1=3/O2=1);0/4 positive C_G in4/6active worlds.
  Mean C_G **-681.287 [-1295.768,-144.359]**;12/10000 zero-action resamples undefined/excluded and disclosed.
Both panels support J600 benefit, not a robust J900 benefit or positive mean superadditivity. The wall-panel credit interval is negative but based on only four actions/six worlds.
Largest H300 positive case:983310/t480/O4, D_G+360.193374, solos-2303.698878/+144.292849, **C_G+2519.599403**.
Largest negative retained:983309/t660/O3, D_G+2651.254557, solos+2807.265931/+3018.391618, **C_G-3174.402992**.

### Why O2 did not emerge atH120, and what H300 actually shows
Under the original nearest-two upstream/adjacent-lane eligibility rule, H120 O2 was available/evaluated37/432decisions, helpful versus MAPPO in3, but outscored by another option in all3; chosen0. Mean evaluated D_O2-950.761, range[-2269.631,+243.858]. This was not an unavailable/disabled option.
- **H300 confirmation18:** O2 available22/216decisions, evaluated9, budget-skipped13, helpful2, chosen1 (one helpful wall outscored). Mean evaluated D_O2-1170.145.
- **H300 wall6:** O2 available17/72, evaluated6, budget-skipped11, helpful1, chosen1. Mean evaluated D_O2-925.232.
Eligibility is logged BEFORE pruning at every registered decision; the24 H300 budget-skipped wall opportunities are untested, not rejected by the predictor. Overall quota pruning affected31confirmation and32wall-panel decisions, including partial candidate pruning.
**Executed confirmation wall983303/t360:** members mainline.171/mainline.167, O2/L3 magnitude3,60s; exactH300 **D_G+13.851684**, solos**-205.778376/+38.059435**, **C_G+181.570625**.
**Executed wall-test wall983321/t480:** members mainline.208/mainline.209, magnitude6,60s; **D_G+2442.101011**, solos**+2510.676566/+360.283172**, **C_G-428.858726**. A helpful joint wall need not be superadditive; its strongest solo helps more.
Thus the H300 planner demonstrably executes a superadditive wall and a large-benefit wall; these separate-world results do not prove that a longer horizon alone caused either difference.

### H300 safety, behaviours, accounting and verification
MAPPO/planner confirmation18: collisions0/0,teleports0/0,TTC10951.5/10334.0veh-s,braking413.5/457.0veh-s (**braking increases**). Wall6: collisions0/0,teleports0/0,TTC4316.0/3894.5,braking113.0/103.5. Safety clamped331/975 total requested member commands.
Unchanged same-prefix causal COOPBEHAV over all14groups: confirmation2supported yield/18gap/39wave events; wall panel1supported gap/1cross-lane-metering/11wave. Observational-only events remain separately labelled in the full receipts.
All14executed groups have28exact solos; all138new full-prefix gates PASS,741total including H120. Every new branch usedt+300 full-rerun cost; all24pairs and source/design/frozen-proposal bindings verified.
**GLOBAL actual=charged725280sim-s <=800000;74720unused.** New160140=57600paired +21000group twins +81540lookaheads; prior565140 unchanged. Host new actual/charged cpuQ53820,gpuA56520,gpuR49800 (cap78240each). Fixed nontransferable world quotas and absent geometrically available options leave budget unused; no outcome-dependent quota reallocation or extra dispatch is armed.
All three hosts30CPU/0skips+WMDECIDE parity; local40CPU/analysis tests PASS. Independent analysis QA checks all24pairs/288decisions, per-world9780quotas, exact H300 predictions and both solos, best-action gate, CSV/summary/credit recomputation, zero pending reservations and exact accounting.
Measured new branch timer median11.092s/max21.226s excludes warmup/reset, retaining the60s cadence as an offline cost-controlled experiment. All three new figures rendered and visually checked.
Artifacts `coopplan/h300/final/`: immutable RECEIPTS/SUMMARY/ALL_GROUP_CREDITS/BEHAVIOURS/QA JSON, confirmation18/wall6 CSVs; SUMMARYsha256`1a5ec2b0dbf310e49247e8593fb99caf46e88aeaf778dfdba21540908526a54d`.
Figures: [horizon/cohort paired CIs](coopplan/h300/final/horizons.png), [wall opportunity audit](coopplan/h300/final/wall_opportunities.png), [all H300 group credits](coopplan/h300/final/h300_cooperation.png).
Reproduce read-only: `python -B -m tools.coopplan.h300_report verify`; exact extrema/walls: `python -B -m tools.coopplan.h300_report examples`. Retain completed H120 estimates above; WM transfer/calibrated confidence remains unqualified.
