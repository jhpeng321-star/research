# Figure/table manifest

**Use rule:** generated historical panels retain their original controller, population and intervention labels. They do not fill the new audit or learner slots. All empirical values below come from named source exports, with frozen input hashes and exact row/column or JSON-pointer provenance in the CSV behind each plot. No new bootstrap, model prediction, interpolation estimate, CI arithmetic or missing-result imputation is used.

**WM-primary ordering:** F1W (proposed WM+real-value method), F3V (offline decisive gate), F3 (WM-policy exact verification versus MAPPO and oracle reference) are the method/result spine. F2 is the exact audit; F4 diagnoses why the value-tail route is needed. Existing generated historical assets remain intact. [[PENDING: WMVALUE results and registered WM-policy receipts / S-WMVALUE, MAIN WM policy producer]]. <!-- Source: owner MAIN ADDENDUM; E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md. -->

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

## F1W — primary WM imagination plus real-trained terminal value

- **Purpose:** separate real-label value fitting, short WM counterfactual group imagination and exact policy verification in one method diagram.
- **Status: PENDING — method direction/design only.** [[PENDING: policy registration and final WM/value/actor/continuation contract / MAIN WM policy producer, S-WMVALUE]].
- **Sources:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; owner MAIN ADDENDUM. Do not reuse the oracle learner's sampled-time endpoint as the fixed-end offline value label.
- **Diagram plan:** recorded real outcomes → observable V; matched B/G/ONLY short WM branches → imagined costs plus branch-specific terminal V → D/C_G → actor update; frozen policy → artifact-valid exact SUMO audit/evaluation. Show oracle-trained backup as a parallel reference, not the primary training signal.
- **Generating command:** [[PENDING: package schematic renderer after the policy contract is frozen / S-PAPERA]]. No empirical bars or performance labels until receipts arrive.
- **Caption claim:** “Proposed WM-primary cooperative training uses a real-trained value tail rather than extending the WM beyond its validated horizon. Offline continuation prediction and exact policy interaction/traffic outcomes are separate gates.”
- **Placement:** IV; combine with a reduced F1 replay inset if needed.

## F1 — exact-replay verification protocol

- **Purpose:** define the intervention and measurement before displaying any positive/negative outcomes.
- **Status: GENERATED — symbolic design schematic.** [PNG](figures/F1_exact_replay_protocol.png), [PDF](figures/F1_exact_replay_protocol.pdf), [CSV behind diagram](figures/F1_exact_replay_protocol.csv).
- **Source:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/research-dense/docs/paper_results/COOPMEAS2.md`. Native-release audit semantics; no invented numerical example.
- **Generating command:** portable builder above, which produces F1 alongside all available quantitative panels.
- **Caption claim:** “Physical/RNG state, requests, group and controller context must be matched across native-release, GROUP and singleton branches; delay saving defines C_G. This schematic states requirements, not a certificate of saved-state reliability. Later oracle-learner restore checks failed, motivating full-prefix reruns. Training mean-background credit is a separate intervention; WM+value branch fidelity also needs validation.” <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md. -->
- **Placement:** III; compact full-width or column-width after final label-size check.

## T1 — controller and intervention contracts

- **Purpose:** prevent native/CTRL0, lane-head/speed-only MAPPO and wall-roster conflation.
- **Status: DESIGN DATA READY; final manuscript table PENDING.** `controller_contracts.csv` records source paths, version/actuator contracts and unresolved receipts.
- **Source:** `E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md`; `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md`; `E:/agent-work/c4-papera/baselines/circles/provenance.json`.
- **Generating command:** no inference/reanalysis. The CSV was transcribed from the cited designs with `apply_patch`; typeset its selected fields in IEEEtran. [[PENDING: final table-rendering command with manuscript source / owner]].
- **Columns:** controller/version; allowed observations; action and lane ownership; disabled-member action; group/context semantics; parity/coverage status. Keep outcome magnitudes in F2, not T1.
- **Remaining:** [[PENDING: WM-trained arm and corrected oracle arm hashes, live observation binding, refusal/abort counts and source freeze / MAIN WM policy producer, S-COOPMEASURE, S-SUMOCF]]. Current primary audit producer scope is MAPPO/wall plus delivered oracle policy; PI/CIRCLES/CTRL0 are optional/history.

## F2 — primary speed-only existing-controller audit

- **Purpose:** answer whether each existing controller has positive interaction, separately from individual and group gains.
- **Status: PENDING — audit release/migration unresolved.** [[PENDING: corrected full MAPPO-speed/wall audit and paired trained arms / S-COOPMEASURE, S-SUMOCF, MAIN WM policy producer]]. The learner's corrected full-rerun mechanics now PASS, but the audit has not yet delivered its own release/versioned migration. PI/CIRCLES/CTRL0 are optional/history. Canary outcomes and the old-design first world are not corrected-panel inference.
- **Source expected:** `E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md` and eventual analysis CSV. Report requires explicit artifact PASS, corrected SNR/design binding and versioned full-panel admission. `E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md` now reports corrected mechanics PASS and selection, without useful interaction signal. The original saved-state probe remains invalid.
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

## F3V — decisive offline WM+value test (primary prerequisite)

- **Purpose:** test whether a real-trained tail improves paired long-outcome prediction beyond full WM and no-WM alternatives.
- **Status: BLOCKED — delivered WMVALUE INCOMPLETE before fitting.** All method/horizon metrics and CIs are NA. Record the source/label telemetry blocker; no performance plot can be generated. [[PENDING: permitted data recovery/amendment and decisive predictions/export / S-WMVALUE, MAIN]]. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md. -->
- **Delivered source:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md` (INCOMPLETE; no numerical performance export); design in `WMVALUE_DESIGN.md` in the same directory. A future admissible test needs its own hash-bound CSV/receipt.
- **Layout:** full WM, WM60+V, WM90+V, command-aware no-WM V and FIT-only arm/time means; aligned sign agreement, MAE and ranking, plus paired sign-improvement/MAE-reduction CIs and each gate result. Both short horizons retained. These horizon labels are from the frozen design. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:40-71. -->
- **Generating command:** [[PENDING: producer's actual schema-bound analysis/export command and package renderer / S-WMVALUE, S-PAPERA]]. This task will not train V or run model rollouts.
- **Caption requirements:** reserved TRAIN worlds, world-cluster uncertainty, same pair/time rows, real fleet+source cost endpoint, sensor/queue admissibility, recorded arm-specific continuation rather than a same-state policy intervention. Report every baseline; offline PASS does not establish C_G or learned gain.
- **Placement:** V-B before policy outcomes.

## F3 — WM-primary policy interaction and traffic efficacy; oracle backup/reference

- **Purpose:** separate “C_G rises” from “beats MAPPO,” with development selection visibly labeled.
- **Status: PENDING.** [[PENDING: registered WM-policy checkpoint/history, exact paired C_G and efficacy; corrected oracle backup / MAIN WM policy producer, S-SUMOCF, S-COOPMEASURE, S-COOPEVAL]]. No completed learner outcome for either route.
- **Sources expected:** WMVALUE verdict, separate WM policy receipts, corrected S-SUMOCF receipts and S-COOPEVAL/audit exports. Distinguish WM-imagined group credit, real oracle sampled-window credit, deployed-mean dev pair diagnostics, and native-release audit C_G.
- **Layout:** WM-trained versus MAPPO (primary); oracle-trained versus MAPPO (backup/reference) on matching worlds/contracts; absolute D_G/ΣD_i with C_G and paired uncertainty; traffic cost/safety and complete candidate history against real/charged budget. Add a gain-retention ratio only with an established positive, resolved oracle denominator. A finite trained oracle is not a theoretical upper bound. Do not use adaptive training means as a matched-panel improvement curve.
- **Generating command:** [[PENDING: schema-bound producer invocation and package renderer after actual receipts / S-COOPEVAL, S-PAPERA]]. The existing `tools/coopeval/cli.py` is a producer entry point, not a completed export receipt.
- **Contract gate:** distinguish WMVALUE fixed-end remaining costs from oracle sampled-start credit and control-start dev horizons. The earlier evaluation loader/budget still needs reconciliation, and the WM arm has no evaluation binding yet. [[PENDING: WM/oracle harness and clock reconciliation / S-COOPEVAL, MAIN]].

## F3H — historical imagined-policy failure versus MAPPO

- **Purpose:** retain completed negative comparator evidence motivating the real-trained value-tail route and the exact oracle reference.
- **Status: GENERATED — historical/context artifact.** [PNG](figures/F3H_historical_policy_failure.png), [PDF](figures/F3H_historical_policy_failure.pdf), [CSV behind plot](figures/F3H_historical_policy_failure.csv).
- **Generating command:** portable builder above.
- **Data source:** `E:/research-dense/docs/paper_results/E3FINAL_A.csv`, `share=All,demand_vph=All,metric=team_delay_vs`, three directly exported pairwise contrasts. Frozen full CSV: `inputs/E3FINAL_A.csv`.
- **Caption:** “Historical frozen cycle-1 and lane-head MAPPO, **228 paired worlds**; equal-cell means and source pointwise **95% CIs**. The registered primary at **600 s** is cycle-1−MAPPO **+683.5 [+256.9,+1122.4] veh-s**, worse for cycle-1; native-relative benefit does not imply MAPPO superiority or cooperation. Earlier **120/300 s** contrasts and native comparisons are exploratory. Costs include source waiting and all requested vehicles. Horizontal offsets separate points only; the clocks are categorical and no interpolation is implied.” <!-- Sources: E:/research-dense/docs/paper_results/E3FINAL_A.md:3-14,18-35; E:/research-dense/docs/paper_results/E3FINAL_A.csv. -->
- **Placement:** artifact/negative-result inset in V-B if needed. Not a result for the new learner or candidate A/B/B-L.

## F4 — surrogate reliability by horizon, with contracts separated

- **Purpose:** motivate the real-trained value tail with the old bounded direction result and corrected native-lane failure, without presenting a single common-model curve or treating a proposed short horizon as already verified.
- **Status: GENERATED.** [PNG](figures/F4_surrogate_reliability.png), [PDF](figures/F4_surrogate_reliability.pdf), [CSV behind plot](figures/F4_surrogate_reliability.csv).
- **Generating command:** portable builder above.
- **Sources:** (a) `E:/research-dense/docs/paper_results/E2_direction_horizon.csv` fields `accuracy,lo,hi,threshold`; (b,c) `E:/agent-work/c4-wmrequal3/tools/wmrequal3/evidence/ANALYSIS.json`, `/matched/<horizon>/J_median` and `/matched/<horizon>/decomposition`. Full explicit pointers and original-file SHA are exported per plotted row.
- **Caption:** “(a) Historical E2 direction panel, **54 worlds**, unbound checkpoint alias: **90 s 0.8895 [0.8424,0.9341]**, source PASS, versus **120 s 0.7526 [0.6824,0.8187]**, FAIL, historical threshold **0.8684**. (b) Corrected WM2 native-lane recorded-intervention contract, **36 worlds / 72 matched arm contrasts**, MAPPO excluded: sign accuracy **0.571/0.444** at **60/90 s**; overall NOT QUALIFIED. Only nonzero recorded effects enter sign denominators; exact `sign_n` is in the plot CSV. (c) At **300 s**, total-effect MAE **1376.6 veh-s** and upstream/merge-road MAE **1497.6 veh-s**; component MAEs need not add because signed errors cancel. Source sign CIs are shown; no missing MAE CIs are invented. Lines in the magnitude panel only connect observed summary horizons.” <!-- Sources: E:/research-dense/docs/paper_results/E2_direction_horizon.csv:5-6; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:3-18,31-43; E:/agent-work/c4-wmrequal3/tools/wmrequal3/evidence/ANALYSIS.json. -->
- **Important:** this scores arm-minus-native effects, not full B/G/ONLY C_G surrogate accuracy or deployment latency. Native-lane replay and synthetic entrant-feedback qualifications remain in main text.
- **Placement:** V-C; combine into a compact full-width figure after checking rendered label sizes.

## F5 — observable cooperative behavior examples

- **Purpose:** connect measured interaction with an interpretable gap/yield/cross-lane-hold process.
- **Status: DESCRIPTIVE COUNTS DELIVERED; valid paired positive-cooperation exemplar PENDING.** [[PENDING: artifact-valid paired release/mean-reference trace examples, coverage and censoring / S-COOPBEHAV, corrected trace producer]]. Original probe positive-credit discovery is not eligible because its branch costs are invalidated.
- **Source expected:** S-COOPBEHAV exports from declared SUMOCF/audit traces; plan in `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Generating command:** [[PENDING: producer's final exact trace-to-figure command / S-COOPBEHAV]].
- **Caption requirements:** label outcome-selected exploratory exemplars; fixed group/clock; issued commands versus realized acceleration; observed nearby ramp vehicle; no hidden labels. Whole-controller WMDATA/WMDATA2 rates have reported exposure and are descriptive, without member-level causal isolation. Original probe clusters cannot establish beneficial C_G after the artifact verdict. <!-- Sources: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:113-198; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:41-53. -->
- **Placement:** one compact exemplar inset only if a valid paired trace is delivered; otherwise artifact pending, never a fabricated time-space plot.

## T2 — physical outcomes and actual budget

- **Purpose:** expose safety/fairness/throughput and cost of WM/value acquisition/training, oracle reference and exact verification.
- **Status: PENDING for new policies; baseline-only receipt delivered.** [[PENDING: full paired WM/oracle physical vector and actual/charged lineage including real value labels, warm start, prefixes, canaries and failed branches / S-COOPEVAL, S-WMVALUE, MAIN WM policy producer, S-SUMOCF, S-COOPMEASURE]]. Completed DEV baseline does not fill learned-arm rows. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md, Completed fixed DEV baseline. -->
- **Source expected:** complete producer reports plus durable accounting summaries. Allocation is not consumption; screen-only no-step costs are not new SUMO panel measurements.
- **Generating command:** [[PENDING: schema-bound export and final IEEEtran table source / S-COOPEVAL, owner]].
- **Main-text minimum:** collisions/teleports/dropped steps, TTC/braking and worst-group/service effects alongside efficiency; incident zeros are not safety equivalence. Full components/cells remain in the artifact.
