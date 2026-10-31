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
# Night additions use pinned final report summaries, never overwrite older inputs.
python -B docs/paper/paperA_cooperation/build_night_figures.py
python -B docs/paper/paperA_cooperation/build_night_figures.py --check
# Add only the late expanded-data screen, preserving inherited artwork bytes.
python -B docs/paper/paperA_cooperation/build_night_figures.py --only F4D
```

Initial capture used the same command with `--freeze-inputs`. That option reads only the explicit allowlist in the script and refuses to replace a differing snapshot. It is not needed for a portable rebuild. PNGs are **300 dpi**; PDFs embed TrueType fonts. <!-- Source: E:/agent-work/c4-papera/docs/paper/paperA_cooperation/build_figures.py, save/rcParams. -->

`inputs/source_manifest.json` records original and snapshot hashes. Original CSV bytes are preserved. `inputs/WMREQUAL3_projection.json` retains only the needed summary fields and contract from the existing analysis JSON; its original-file hash remains in the manifest. `figure_validation.json` records source-value/output checks and output hashes. Figure IDs and editorial dimensions are not empirical quantities.

## F1W — primary WM imagination plus real-trained terminal value

- **Purpose:** separate real-label value fitting, short WM counterfactual group imagination and exact policy verification in one method diagram.
- **Status: PENDING — method direction/design only.** [[PENDING: policy registration and final WM/value/actor/continuation contract / MAIN WM policy producer, S-WMVALUE]].
- **Sources:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; owner MAIN ADDENDUM. Do not reuse the oracle learner's sampled-time endpoint as the fixed-end offline value label.
- **Diagram plan:** recorded real outcomes → observable V; matched B/G/ONLY short WM branches → imagined costs plus branch-specific terminal V → D/C_G → actor update; frozen policy → artifact-valid exact SUMO audit/evaluation. Show oracle-trained backup as a parallel reference, not the primary training signal.
- **Generating command:** [[PENDING: package schematic renderer after the policy contract is frozen / S-PAPERA]]. No empirical bars or performance labels until receipts arrive.
- **Caption claim:** “Proposed WM-primary cooperative training uses short imagination and a real-trained value tail instead of relying on an unqualified long rollout. The short route itself still needs qualification. Offline continuation prediction and exact policy interaction/traffic outcomes are separate gates.”
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
- **Remaining:** corrected backup checkpoint hashes are delivered in SUMOCF_RUN.md, but its audit migration is not. [[PENDING: WM-trained arm, audit observation binding and refusal/abort counts / MAIN WM policy producer, S-COOPMEASURE]]. Current producer scope is MAPPO/wall plus reserved oracle policy; PI/CIRCLES/CTRL0 are optional/history. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:81-88. -->

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
- **Status: RUNNING — causal reconstruction, real fit and adapter gates complete; short-first order active.** All predictive metrics/CIs remain NA pending the complete once-only test; no performance bars are eligible. [[PENDING: complete short-first predictions/export and frozen verdict; long A only if needed / S-WMVALUE existing short dispatcher/finalizer]]. <!-- Sources: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md; E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:148-159. -->
- **Delivered source:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md` (RUNNING); amended `WMVALUE_DESIGN.md` in the same directory. Frozen values and reserved label checks are execution receipts, not numerical prediction results. Use the eventual hash-bound complete CSV/receipt.
- **Layout:** full WM, WM60+V, WM90+V, command-aware no-WM V and FIT-only arm/time means; aligned sign agreement, MAE and ranking, plus paired sign-improvement/MAE-reduction CIs and each gate result. Both short horizons retained. These horizon labels are from the frozen design. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:40-71. -->
- **Generating command:** [[PENDING: producer's actual schema-bound analysis/export command and package renderer / S-WMVALUE, S-PAPERA]]. This task will not train V or run model rollouts.
- **Caption requirements:** reserved TRAIN worlds, world-cluster uncertainty, same pair/time rows, real fleet+source cost endpoint, sensor/queue admissibility, recorded arm-specific continuation rather than a same-state policy intervention. Show every completed frozen baseline comparison; a decisive short-first FAIL may leave A NA, but no PASS bypasses A. Offline PASS does not establish C_G or learned gain. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:148-159. -->
- **Placement:** V-B before policy outcomes.

## F3 — WM-primary policy interaction and traffic efficacy; oracle backup/reference

- **Purpose:** separate “C_G rises” from “beats MAPPO,” with development selection visibly labeled.
- **Status: PENDING for WM-policy and full native-release audit; backup completed unsuccessfully (F3B).** [[PENDING: registered WM-policy checkpoint/history, exact paired C_G and efficacy / MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]]. Backup DEV outcomes cannot fill WM-policy rows. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-46. -->
- **Sources expected:** WMVALUE verdict, separate WM policy receipts, corrected S-SUMOCF receipts and S-COOPEVAL/audit exports. Distinguish WM-imagined group credit, real oracle sampled-window credit, deployed-mean dev pair diagnostics, and native-release audit C_G.
- **Layout:** WM-trained versus MAPPO (primary); backup versus MAPPO on matching contracts; group/solo/interaction and paired uncertainty; cost/safety and full candidate history. Gain-retention ratio is **NOT ESTIMABLE** at this cut because no beneficial backup denominator is established. Do not draw a ratio panel or use adaptive training means as a deployed improvement curve. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-40. -->
- **Generating command:** [[PENDING: schema-bound producer invocation and package renderer after actual receipts / S-COOPEVAL, S-PAPERA]]. The existing `tools/coopeval/cli.py` is a producer entry point, not a completed export receipt.
- **Contract gate:** distinguish WMVALUE fixed-end remaining costs from oracle sampled-start credit and control-start dev horizons. The earlier evaluation loader/budget still needs reconciliation, and the WM arm has no evaluation binding yet. [[PENDING: WM/oracle harness and clock reconciliation / S-COOPEVAL, MAIN]].

## F3B — completed SUMO backup: no primary-horizon win

- **Purpose:** preserve every declared evaluated checkpoint and the horizon-ranking reversal, rather than leave the stopped backup as pending.
- **Status: GENERATED, source-checked and visually inspected.** [PNG](figures/F3B_stopped_sumo_backup.png), [PDF](figures/F3B_stopped_sumo_backup.pdf), [CSV behind plot](figures/F3B_stopped_sumo_backup.csv). Renderer pins the immutable final report snapshot; full source `E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md`, final DEV table and cooperation section. `night_figure_validation.json` binds source/output hashes. Published summary precision is retained, not presented as an independent per-world reanalysis.
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_night_figures.py`; validate with the same command plus `--check`. This renderer transcribes published summary cells into a source-located CSV; it computes no new CIs or scientific predictions.
- **Layout:** paired learned-minus-MAPPO J600 and J900, checkpoint categories, zero reference and copied source CIs; a separate small C_G panel with its much smaller scale and eligible/missing counts. No interpolation between checks; no clipping away adverse costs or pooling credit definitions.
- **Caption claim:** “Fixed **24-world** development panel, **u3/u7/u12 J600 +508.3/+857.9/+548.5 veh-s**, versus **J900 −633.4/−465.0/−584.5**. Source **95% world-bootstrap CIs**, **2000 draws/seed 0**: middle J600 adverse, all other displayed cost intervals include zero. Sparse deployed-mean pair C_G **−0.002589/+0.001270/0 veh-s**, **2/2/3** eligible worlds and **1/1/0** missing groups, does not show useful cooperation. This diagnostic uses MAPPO-mean background/learned continuation, not native-release audit. Owner stopped training; later checkpoints are preserved but untested on DEV.” <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-46. -->
- **Placement:** V-B compact negative-result panel; full backing data in artifact. It is not an oracle upper-bound success.

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
- **Future qualification boundary:** this is a preserved paired-effect diagnostic, not distributional qualification. New owner direction prioritizes breakdown/traffic-distribution calibration and expected controller effects for future WM fixes; no distributional result or replacement curve is delivered yet. [[PENDING: distributional/conditional-ensemble figure and frozen gate / S-WMBREAKDOWN, S-ENSPROBE, MAIN]]. Existing figures and the frozen WMVALUE gate keep their original meaning. <!-- Source: E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER22:18 and OWNER22:22. -->
- **Placement:** V-C; combine into a compact full-width figure after checking rendered label sizes.

## F4N — night falsifiers: calibration and queue/service model

- **Purpose:** make the failed long-tail remedies visible without combining different panels into one horizon qualification.
- **Status: GENERATED, source-checked and visually inspected.** [PNG](figures/F4N_failed_tail_remedies.png), [PDF](figures/F4N_failed_tail_remedies.pdf), [CSV behind plot](figures/F4N_failed_tail_remedies.csv). Immutable report snapshots are pinned in `build_night_figures.py`; original sources `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md` and `E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md`. Exact plotted source cells/hashes and image checks are in `night_figure_validation.json`.
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_night_figures.py`; validate with `--check`. Source-table cell export only, no new statistics.
- **Layout:** separate categorical MAE panels: raw WM2/arm-mean/selected calibrator in grouped CV; queue/state-only/arm-mean on the complete reserved-world panel. No invented MAE whiskers. Include failed qualification labels, not a success badge for the best CV point.
- **Caption claim:** “Grouped-CV raw WM2 versus no-WM arm-mean MAE at **120/300/600 s** is **263/1719/6013** versus **185/999/3298 veh-s**. Selected calibrators fail the frozen SPEEDHEAD gate at every horizon: NOT USABLE. Separate queue prototype on **eight reserved TRAIN worlds/48 contrasts per horizon** has MAE **276/1362/5296** versus state-only **239/932/3642 veh-s**: KILL. These point estimates have different panels and are not a common model curve or C_G-fidelity test.” <!-- Sources: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:14-50; E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md:12-32. -->
- **Scope:** raw WM is not worse on every transfer panel; queue comparison to raw WM has fewer complete worlds and is omitted here rather than mixed. Tiny-pulse floor remains a scoped discussion result, not a blanket irreducible-noise curve.

## F4D — expanded-data merge head: amended early-stop screen

- **Purpose:** replace the stale pending-head slot with the completed screen, keeping the unscored later checkpoints explicit.
- **Status: GENERATED — completed step10 screen only.** [PNG](figures/F4D_expanded_data_screen.png), [PDF](figures/F4D_expanded_data_screen.pdf), [CSV behind plot](figures/F4D_expanded_data_screen.csv). The renderer pins `source_snapshots/wmfix_v2data_3e0bd4cf680cefa2.md`; original source `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md:3-42`. Exact values, source line/column, original SHA and immutable snapshot accompany each plotted row.
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_night_figures.py --only F4D`; validate with `python -B docs/paper/paperA_cooperation/build_night_figures.py --check`. The selective command preserves inherited artwork; source-summary transcription only, no new experiment or statistics.
- **Layout:** separate short/long MAE panels, entry-path step0 versus trained step10. Both axes start at zero and their different scales are disclosed. No unscored bars or invented CIs.
- **Caption claim:** “Full **52-world TRAIN panel/104 contrasts per horizon**: paired-J MAE **252.1303→250.6552 veh-s at 120 s**, **1718.2620→1885.3962 at 300 s**. Published changes **−0.585%/+9.727%**. The amended both-horizon screen stops after step10; **steps 20/30 unscored**, not a full-gate PASS/FAIL. All **30** optimizer updates retain finite/frozen receipts; optimizer completion does not supply later-panel accuracy.” <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_PILOT_V2DATA.md:3-42. -->
- **Placement:** V-C compact repair inset or artifact if space is tight. This TRAIN-panel screen does not qualify the WM+value route or provide reserved-world generalization.

## F5 — observable cooperative behavior examples

- **Purpose:** connect measured interaction with an interpretable gap/yield/cross-lane-hold process.
- **Status: CORRECTED PAIRED REPORTS DELIVERED; beneficial deployed exemplar NOT ESTABLISHED.** Corrected u3 DEV and u15 sampled TRAIN reports now exist. [[PENDING: independently replicated beneficial deployed WM-policy example / MAIN WM policy producer, S-COOPBEHAV]]. Original probe positive-credit discovery remains invalidated. <!-- Source: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:165-236. -->
- **Source expected:** S-COOPBEHAV exports from declared SUMOCF/audit traces; plan in `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Generating command:** [[PENDING: producer's final exact trace-to-figure command / S-COOPBEHAV]].
- **Caption requirements:** outcome-selected TRAIN exemplars are exploratory, not deployed-policy discoveries. Keep fixed group/clock, issued-versus-realized actions, exposure/censoring and native/mean-reference semantics. u3 has no paired-supported mechanism; u15 contains supported events but even its largest positive-C_G record has negative D_G. Original-probe clusters stay withdrawn. Whole-controller rates do not isolate members. <!-- Sources: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:119-236; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:41-53. -->
- **Placement:** one compact exemplar inset only if a valid paired trace is delivered; otherwise artifact pending, never a fabricated time-space plot.

## T2 — physical outcomes and actual budget

- **Purpose:** expose safety/fairness/throughput and cost of WM/value acquisition/training, oracle reference and exact verification.
- **Status: BACKUP SAFETY/BUDGET DELIVERED; WM-policy and full fairness/service vector PENDING.** Backup TRAIN **3,194,716 actual / 3,205,151 charged sim-s**, DEV **184,049 actual=charged**; no DEV collisions/teleports, paired TTC intervals include zero. Keep invalidated work, prefixes and final drain in totals. [[PENDING: WM-policy physical outcomes, full service/fairness vector and total warm-start/value/data lineage / S-COOPEVAL, S-WMVALUE, MAIN WM policy producer, owner integrator]]. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:48-88. -->
- **Source expected:** complete producer reports plus durable accounting summaries. Allocation is not consumption; screen-only no-step costs are not new SUMO panel measurements.
- **Generating command:** [[PENDING: schema-bound export and final IEEEtran table source / S-COOPEVAL, owner]].
- **Main-text minimum:** collisions/teleports/dropped steps, TTC/braking and worst-group/service effects alongside efficiency; incident zeros are not safety equivalence. Full components/cells remain in the artifact.

## Paper B figure routing

- Existing winner-by-condition PNG/PDF and CSVs: `E:/agent-work/c4-c1switch/docs/paper_results/c1switch/recorded_winner_by_condition.{png,pdf}`, `recorded_world_winners.csv`, `recorded_cell_winshares.csv`; command `python -m tools.c1switch.winners` in its worktree. Completed recorded/fresh headroom is descriptive, not switching efficacy; keep this figure in Paper B. <!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:15-19,31-42. -->
- Recovered C1GAP wall CIs and genuine-history PI results belong to the scoped Paper B prediction panels; they do not fill F3V or F3. C1CLOSED has no complete-panel figure/verdict yet: [[PENDING: complete feedback CSV/PNG / S-C1CLOSED existing finalizer]]. No partial performance plot is imported. <!-- Sources: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md; E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md; E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md. -->
- Latest shared-ledger reread: corrected switching canary, both parity gates and independent BB reuse PASS; bounded main dispatch is RUNNING. There is no complete main performance panel, so no switching efficacy figure is imported. New lever/distributional-breakdown follow-ups are prospective and have no result figure eligible for this cut. <!-- Source: E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, v7 MAIN DISPATCHED/OBSERVED RUNNING; OWNER22:05/S-LEVERPROBE22:07 and OWNER22:10/S-WMBREAKDOWN. -->
