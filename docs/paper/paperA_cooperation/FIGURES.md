# Figure/table manifest

**Use rule:** generated historical panels retain their original controller, population and intervention labels. They do not fill the new audit or learner slots. All empirical values below come from named source exports, with frozen input hashes and exact row/column or JSON-pointer provenance in the CSV behind each plot. No new bootstrap, model prediction, interpolation estimate, CI arithmetic or missing-result imputation is used.

## Reproduction

From `E:/agent-work/c4-papera`:

```powershell
# Portable: rebuild from committed source snapshots only.
python -B docs/paper/paperA_cooperation/build_figures.py
# Check exported values, identities, interaction algebra and image integrity.
python -B docs/paper/paperA_cooperation/build_figures.py --check
```

Initial capture used the same command with `--freeze-inputs`. That option reads only the explicit allowlist in the script and refuses to replace a differing snapshot. It is not needed for a portable rebuild. PNGs are **300 dpi**; PDFs embed TrueType fonts. <!-- Source: E:/agent-work/c4-papera/docs/paper/paperA_cooperation/build_figures.py, save/rcParams. -->

`inputs/source_manifest.json` records original and snapshot hashes. Original CSV bytes are preserved. `inputs/WMREQUAL3_projection.json` retains only the needed summary fields and contract from the existing analysis JSON; its original-file hash remains in the manifest. `figure_validation.json` records source-value/output checks and output hashes. Figure IDs and editorial dimensions are not empirical quantities.

## F1 — exact-replay cooperation protocol

- **Purpose:** define the intervention and measurement before displaying any positive/negative outcomes.
- **Status: GENERATED — symbolic design schematic.** [PNG](figures/F1_exact_replay_protocol.png), [PDF](figures/F1_exact_replay_protocol.pdf), [CSV behind diagram](figures/F1_exact_replay_protocol.csv).
- **Source:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`. Native-release audit semantics; no invented numerical example.
- **Generating command:** portable builder above, which produces F1 alongside all available quantitative panels.
- **Caption claim:** “Physical/RNG state, requests, group and controller context are matched across native-release, GROUP and singleton branches. Saved fleet+source delay defines C_G. The learner's mean-background exploration experiment is a separate intervention. WM-S is a candidate surrogate whose branch-effect fidelity requires validation.”
- **Placement:** III; compact full-width or column-width after final label-size check.

## T1 — controller and intervention contracts

- **Purpose:** prevent native/CTRL0, lane-head/speed-only MAPPO and wall-roster conflation.
- **Status: DESIGN DATA READY; final manuscript table PENDING.** `controller_contracts.csv` records source paths, version/actuator contracts and unresolved receipts.
- **Source:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/agent-work/c4-papera/baselines/circles/provenance.json`.
- **Generating command:** no inference/reanalysis. The CSV was transcribed from the cited designs with `apply_patch`; typeset its selected fields in IEEEtran. [[PENDING: final table-rendering command with manuscript source / owner]].
- **Columns:** controller/version; allowed observations; action and lane ownership; disabled-member action; group/context semantics; parity/coverage status. Keep outcome magnitudes in F2, not T1.
- **Remaining:** [[PENDING: final arm hashes, precise live observation binding, refusal/abort counts and source-report freeze / S-COOPMEASURE, S-SUMOCF]].

## F2 — primary speed-only existing-controller audit

- **Purpose:** answer whether each existing controller has positive interaction, separately from individual and group gains.
- **Status: PENDING.** [[PENDING: full audit CSV with MAPPO-speed, PI, CIRCLES, CTRL0 and wall / S-COOPMEASURE]]. Canary outcomes are not panel inference.
- **Source expected:** `E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md` and its eventual analysis CSV. Currently this report documents verified mechanics and WAIT_SUMOCF_SNR_TABLE only.
- **Layout:** forest plot of C_G with world CIs, plus aligned D_G/ΣD_i, eligible groups and positive fraction; stratified/whole-panel estimates explicit. Include native-repeat and wall coverage/abort indicators.
- **Generating command:** [[PENDING: producer's final analysis/export command and package renderer after real CSV schema lands / S-COOPMEASURE, S-PAPERA]]. Do not run a guessed producer command or synthesize placeholder bars.
- **Caption insert:** [[PENDING: selected group/window, clock origin, fixed population, source CI recipe and all controller outcomes / S-COOPMEASURE]].

## F2H — historical interaction and its uncertainty

- **Purpose:** show why fleet improvement cannot fill an interaction claim, and preserve the prior inconclusive finding.
- **Status: GENERATED — historical diagnostic/context.** [PNG](figures/F2H_historical_cooperation.png), [PDF](figures/F2H_historical_cooperation.pdf), [CSV behind plot](figures/F2H_historical_cooperation.csv).
- **Generating command:** portable builder above.
- **Data source:** `E:/research-dense/docs/paper_results/COOPMEAS2.csv`, `cohort=all`. Left: primary-clock D_G, sum_D_i, C_G, D_all for historical ours/MAPPO. Right: directly exported paired `ours_minus_mappo` C_G at every exported clock. Original file frozen in `inputs/COOPMEAS2.csv`.
- **Caption:** “Historical lane-head MAPPO and cycle-1, **28 worlds**, native-release branches. At **300 s**, paired cycle-1−MAPPO C_G **+327 [−395,+1035] veh-s**; interaction advantage is not established. Fleet savings are distinct from fixed-group intervention. Whiskers preserve source **95%** stratified world intervals, **5000 draws**, seed **20261028**. The **60/120 s** clocks are secondary; all eligible tiny/null groups remain included. The adverse **60 s** cycle-1 C_G **−2.982 [−6.131,−0.196] veh-s** remains in the complete frozen source CSV.” <!-- Sources: E:/research-dense/docs/paper_results/COOPMEAS2.md:3-20,39-70; E:/research-dense/docs/paper_results/COOPMEAS2.csv:2. -->
- **Placement:** V-A context; shrink to a primary-only inset if space is tight, keeping full PNG/CSV in artifact. It never replaces F2.

## F3 — new learner interaction and traffic efficacy

- **Purpose:** separate “C_G rises” from “beats MAPPO,” with development selection visibly labeled.
- **Status: PENDING.** [[PENDING: per-update training credit, fixed diagnostic-pair dev credit, full paired efficacy and later-clock exports / S-SUMOCF, S-COOPEVAL]].
- **Sources expected:** amended S-SUMOCF receipts; S-COOPEVAL complete report and CSVs. Final plots must distinguish selected training groups, the smaller fixed dev diagnostic group, and native-release audit groups/backgrounds.
- **Layout:** interaction with fixed-comparator CIs and absolute solo/group terms; aligned learned−MAPPO/CTRL0 cost and safety, plus all-candidate learning history with budgets. Do not use adaptive training mean as a common-panel improvement curve.
- **Generating command:** [[PENDING: schema-bound producer invocation and package renderer after actual receipts / S-COOPEVAL, S-PAPERA]]. The existing `tools/coopeval/cli.py` is a producer entry point, not a completed export receipt.
- **Contract gate:** amended design includes later-clock scoring; validate loader/clock origin and the amended number of checks before plotting. [[PENDING: harness reconciliation / S-COOPEVAL]].

## F3H — historical imagined-policy failure versus MAPPO

- **Purpose:** retain completed negative comparator evidence behind the oracle-learning motivation.
- **Status: GENERATED — historical/context artifact.** [PNG](figures/F3H_historical_policy_failure.png), [PDF](figures/F3H_historical_policy_failure.pdf), [CSV behind plot](figures/F3H_historical_policy_failure.csv).
- **Generating command:** portable builder above.
- **Data source:** `E:/research-dense/docs/paper_results/E3FINAL_A.csv`, `share=All,demand_vph=All,metric=team_delay_vs`, three directly exported pairwise contrasts. Frozen full CSV: `inputs/E3FINAL_A.csv`.
- **Caption:** “Historical frozen cycle-1 and lane-head MAPPO, **228 paired worlds**; equal-cell means and source pointwise **95% CIs**. The registered primary at **600 s** is cycle-1−MAPPO **+683.5 [+256.9,+1122.4] veh-s**, worse for cycle-1; native-relative benefit does not imply MAPPO superiority or cooperation. Earlier **120/300 s** contrasts and native comparisons are exploratory. Costs include source waiting and all requested vehicles. Horizontal offsets separate points only; the clocks are categorical and no interpolation is implied.” <!-- Sources: E:/research-dense/docs/paper_results/E3FINAL_A.md:3-14,18-35; E:/research-dense/docs/paper_results/E3FINAL_A.csv. -->
- **Placement:** artifact/negative-result inset in V-B if needed. Not a result for the new learner or candidate A/B/B-L.

## F4 — surrogate reliability by horizon, with contracts separated

- **Purpose:** retain the old bounded direction result and quantify corrected native-lane failure without presenting a single common-model curve.
- **Status: GENERATED.** [PNG](figures/F4_surrogate_reliability.png), [PDF](figures/F4_surrogate_reliability.pdf), [CSV behind plot](figures/F4_surrogate_reliability.csv).
- **Generating command:** portable builder above.
- **Sources:** (a) `E:/research-dense/docs/paper_results/E2_direction_horizon.csv` fields `accuracy,lo,hi,threshold`; (b,c) `E:/agent-work/c4-wmrequal3/tools/wmrequal3/evidence/ANALYSIS.json`, `/matched/<horizon>/J_median` and `/matched/<horizon>/decomposition`. Full explicit pointers and original-file SHA are exported per plotted row.
- **Caption:** “(a) Historical E2 direction panel, **54 worlds**, unbound checkpoint alias: **90 s 0.8895 [0.8424,0.9341]**, source PASS, versus **120 s 0.7526 [0.6824,0.8187]**, FAIL, historical threshold **0.8684**. (b) Corrected WM2 native-lane recorded-intervention contract, **36 worlds / 72 matched arm contrasts**, MAPPO excluded: sign accuracy **0.571/0.444** at **60/90 s**; overall NOT QUALIFIED. Only nonzero recorded effects enter sign denominators; exact `sign_n` is in the plot CSV. (c) At **300 s**, total-effect MAE **1376.6 veh-s** and upstream/merge-road MAE **1497.6 veh-s**; component MAEs need not add because signed errors cancel. Source sign CIs are shown; no missing MAE CIs are invented. Lines in the magnitude panel only connect observed summary horizons.” <!-- Sources: E:/research-dense/docs/paper_results/E2_direction_horizon.csv:5-6; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:3-18,31-43; E:/agent-work/c4-wmrequal3/tools/wmrequal3/evidence/ANALYSIS.json. -->
- **Important:** this scores arm-minus-native effects, not full B/G/ONLY C_G surrogate accuracy or deployment latency. Native-lane replay and synthetic entrant-feedback qualifications remain in main text.
- **Placement:** V-C; combine into a compact full-width figure after checking rendered label sizes.

## F5 — observable cooperative behavior examples

- **Purpose:** connect measured interaction with an interpretable gap/yield/cross-lane-hold process.
- **Status: PENDING.** [[PENDING: paired member-release trace examples, exposure and censoring / S-COOPBEHAV]].
- **Source expected:** S-COOPBEHAV exports from declared SUMOCF/audit traces; plan in `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Generating command:** [[PENDING: producer's final exact trace-to-figure command / S-COOPBEHAV]].
- **Caption requirements:** label outcome-selected exploratory exemplars; fixed group/clock; controller commands versus realized acceleration; observed nearby ramp vehicle; no hidden labels. Whole-controller WMDATA arms cannot stand in for member-release branches.
- **Placement:** one compact exemplar inset only if a valid paired trace is delivered; otherwise artifact pending, never a fabricated time-space plot.

## T2 — physical outcomes and actual budget

- **Purpose:** expose safety/fairness/throughput and cost of oracle measurements/training.
- **Status: PENDING for new controller panel/learner.** [[PENDING: full paired physical vector and actual/charged lineage including warm start, prefixes, canaries and failures / S-COOPEVAL, S-SUMOCF, S-COOPMEASURE]].
- **Source expected:** complete producer reports plus durable accounting summaries. Allocation is not consumption; screen-only no-step costs are not new SUMO panel measurements.
- **Generating command:** [[PENDING: schema-bound export and final IEEEtran table source / S-COOPEVAL, owner]].
- **Main-text minimum:** collisions/teleports/dropped steps, TTC/braking and worst-group/service effects alongside efficiency; incident zeros are not safety equivalence. Full components/cells remain in the artifact.
