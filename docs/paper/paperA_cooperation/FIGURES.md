# Figure/table manifest

**Use rule:** generated historical panels retain their original controller, population and intervention labels. They do not fill the new audit or learner slots. All empirical values below come from named source exports, with frozen input hashes and exact row/column or JSON-pointer provenance in the CSV behind each plot. No new bootstrap, model prediction, interpolation estimate, CI arithmetic or missing-result imputation is used.

**Current drafting spine:** F1 replay, F3G WM-guided/SUMO-graded method, F2 corrected audit, F3 pending A/B/R efficacy and paper-measure interaction, F6 exact-planner reference, F7 bounded teacher/ranking diagnostic. Existing F3V/F3T/F4P preserve failed value, inconclusive exact-student and final predictor diagnostics as artifact panels. WMTEACH quality is delivered, scale paused; current COOPMWM B/R held for volume qualification. No pending learned panel is filled by an older plot. <!-- Sources: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM.md:3-31; E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:34-64,103-113; E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md. -->

**Closing amendment supersedes set-point method slots:** all A/B set-point runs are engineering history; all arms now held for acceleration-plan restart. Owner requests a **96-world** final panel; its new metadata/guards/exposure/quote are pending. F3G must depict the frozen successor, F3 must await its proposal-free outcomes. F7 is unchanged cached-option diagnostic, not qualification of the new CEM/profile search. <!-- Sources: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM.md:3-14; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN19:30/19:34/19:38. -->

## F3G — primary method: WM-guided exploration, SUMO-graded

- **Purpose:** show that the WM selects candidate exploration while SUMO alone supplies reward; keep the diagnostic replay path separate.
- **Data source:** `E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md`, source-pinned in `source_refresh.json`; `COOPMWM.md` current guidance-volume hold. Existing Overleaf v2 has a schematic placeholder, not a finished empirical figure.
- **Layout:** identical MAPPO acceleration residuals → A PPO / B committed CEM group-profile proposer / R matched random-profile proposer → unchanged safety → SUMO reward. Show short screen/long rescore/Thompson candidate choice only after implementation freeze. Annotate final logged route correction/imitation/trust and proposal-free actor. Dashed E2 replay box must say member-revert diagnostic, not singleton C_G.
- **Status: PENDING final schematic.** [[PENDING: numeric throughput/K/quota freeze and manuscript schematic / S-COOPMWM, owner]]. Existing six-page v2 skeleton is preserved; no performance graphic fabricated.
- **Generating command:** [[PENDING: final manuscript schematic renderer after engineering release / owner]]. Sources/diagram are symbolic, with no invented example values.

## F7 — available teacher/ranking evidence; not a live win

- **Purpose:** motivate mean-delay guidance without claiming calibrated model probabilities or learned efficacy.
- **Data sources:** `E:/agent-work/c4-bdrank/docs/paper_results/bdrank/option_concordance.csv`, byte-preserved in `inputs/F7_option_concordance_source.csv`; hash in `inputs/F7_source_manifest.json`. Report `E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24-46`; quality report `E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:103-113,197-204`. Backing CSV includes source line/hash and all denominator/tie/CI-status columns.
- **Layout/caption:** separate available-option pair concordance from recorded-controller probability ties. v4 **65.5/75.9%** versus v5 **63.8/51.7%** at H120/H300, **58 option pairs** each; separate controller H600 **112/120** probability ties. Small roster-only/source-prior/cold-memory diagnostic, not student fitting or C_G fidelity. No invented world CIs for singleton cells. <!-- Source: E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:24-32. -->
- **Status: GENERATED, source-checked and visually inspected.** [PNG](figures/F7_option_delay_ranking.png), [PDF](figures/F7_option_delay_ranking.pdf), [backing CSV](figures/F7_option_delay_ranking.csv). Each bar has **58 untied informative pairs / 11 contributing worlds** from the **12-state** diagnostic; exact ties excluded, unavailable pairs not imputed. No invented CIs. The separate controller-tie count is caption context, not a plotted option result. <!-- Sources: E:/agent-work/c4-bdrank/docs/paper_results/bdrank/option_concordance.csv, pooled/mean_delay/mean_delay; E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md:30-32. -->
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_evening_figures.py`; check with `--check`. Initial capture used `--freeze-inputs`. `evening_figure_validation.json` checks exact cells, concordance count identity, absent CIs, image integrity and hashes. No new inference or statistical interval computation.

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
# Completed morning audit/value/planner, from new immutable pins only.
python -B docs/paper/paperA_cooperation/build_morning_figures.py
python -B docs/paper/paperA_cooperation/build_morning_figures.py --check
# Final FULL-v4 diagnostic, pinned final report only; preserves every older asset.
python -B docs/paper/paperA_cooperation/build_noon_figures.py
python -B docs/paper/paperA_cooperation/build_noon_figures.py --check
# Completed exact-teacher student: frozen summary/credits/worlds/QA, not WM outcomes.
python -B docs/paper/paperA_cooperation/build_student_figures.py
python -B docs/paper/paperA_cooperation/build_student_figures.py --check
# Evening option ranking from a byte-preserved producer CSV.
python -B docs/paper/paperA_cooperation/build_evening_figures.py
python -B docs/paper/paperA_cooperation/build_evening_figures.py --check
```

Initial capture used the same command with `--freeze-inputs`. That option reads only the explicit allowlist in the script and refuses to replace a differing snapshot. It is not needed for a portable rebuild. PNGs are **300 dpi**; PDFs embed TrueType fonts. <!-- Source: E:/agent-work/c4-papera/docs/paper/paperA_cooperation/build_figures.py, save/rcParams. -->

`inputs/source_manifest.json` records original and snapshot hashes. Original CSV bytes are preserved. `inputs/WMREQUAL3_projection.json` retains only the needed summary fields and contract from the existing analysis JSON; its original-file hash remains in the manifest. `figure_validation.json` records source-value/output checks and output hashes. Figure IDs and editorial dimensions are not empirical quantities.

## F1W — original WM imagination plus real-trained terminal value

- **Purpose:** separate real-label value fitting, short WM counterfactual group imagination and exact policy verification in one method diagram.
- **Status: PROPOSED historical method diagram; frozen value route FAILED.** Not the current teacher/student implementation. [[PENDING: owner manuscript choice of predecessor inset / owner]]. No success diagram may label this route validated. <!-- Sources: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md; E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md. -->
- **Sources:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md`; owner MAIN ADDENDUM. Do not reuse the oracle learner's sampled-time endpoint as the fixed-end offline value label.
- **Diagram plan:** recorded real outcomes → observable V; matched B/G/ONLY short WM branches → imagined costs plus branch-specific terminal V → D/C_G → actor update; frozen policy → artifact-valid exact SUMO audit/evaluation. Show oracle-trained backup as a parallel reference, not the primary training signal.
- **Generating command:** [[PENDING: package schematic renderer after the policy contract is frozen / S-PAPERA]]. No empirical bars or performance labels until receipts arrive.
- **Caption claim:** “The proposed WM-primary route computes short imagined costs plus real-trained terminal value. Its frozen continuation test failed; no validated actor training follows. Exact policy interaction and traffic outcomes remain separate requirements.” <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md. -->
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
- **Delivered/remaining:** corrected audit v2 binds final-u15, speed-only MAPPO and wall with complete coverage/safety. Native release is not planner O0 or free-form HOLD. [[PENDING: WM-trained arm and final manuscript typesetting / MAIN WM policy producer, owner]]. PI/CIRCLES/CTRL0 remain optional/history. <!-- Sources: E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md:178-234; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->

## F2 — primary speed-only existing-controller audit

- **Purpose:** answer whether each existing controller has positive interaction, separately from individual and group gains.
- **Status: GENERATED — corrected complete v2, conditional eligible-world inference.** [PNG](figures/F2_corrected_controller_audit.png), [PDF](figures/F2_corrected_controller_audit.pdf), [backing CSV](figures/F2_corrected_controller_audit.csv).
- **Source:** `E:/agent-work/c4-coopmeasure/tools/coopaudit/evidence/v2_analysis/COOPAUDIT.csv`, `cohort=eligible21,outcome=600`, frozen as `source_snapshots/audit_v2_csv_0abe14ec6f66b030.csv`; source report `E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md`. Saved-state outcomes excluded.
- **Layout:** forest C_G with copied CIs, adjacent D_G/ΣD_i point means. Backing CSV also carries positive fractions/CIs. All-allocation missing intervals remain missing, never zero bars.
- **Generating command:** morning renderer above; check validates exact source cells, C_G algebra, identities and image integrity. `morning_figure_validation.json` is the receipt.
- **Caption:** “Fixed pair at t300, **60 s** native-release intervention, fleet+source **J600** after selection. **21/28** eligible, **seven** missing, **three** wall-feasible pairs. MAPPO C_G **+67.679 [−1179.061,+1154.090]**, SUMOCF-u15 **+67.757 [−1178.903,+1154.094] veh-s**; wall numerically null. Conditional eligible-cell-weighted **95% stratified world CIs, 5000 draws/seed 20261031**. Full-allocation CIs undefined. Canary positive detectability is separate, not pooled. Paired trained-arm rise is unresolved.” <!-- Source: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md:9-44. -->

## F2H — historical interaction and its uncertainty

- **Purpose:** show why fleet improvement cannot fill an interaction claim, and preserve the prior inconclusive finding.
- **Status: GENERATED — historical diagnostic/context.** [PNG](figures/F2H_historical_cooperation.png), [PDF](figures/F2H_historical_cooperation.pdf), [CSV behind plot](figures/F2H_historical_cooperation.csv).
- **Generating command:** portable builder above.
- **Data source:** `E:/research-dense/docs/paper_results/COOPMEAS2.csv`, `cohort=all`. Left: primary-clock D_G, sum_D_i, C_G, D_all for historical ours/MAPPO. Right: directly exported paired `ours_minus_mappo` C_G at every exported clock. Original file frozen in `inputs/COOPMEAS2.csv`.
- **Caption:** “Historical lane-head MAPPO and cycle-1, **28 worlds**, native-release branches. At **300 s**, paired cycle-1−MAPPO C_G **+327 [−395,+1035] veh-s**; interaction advantage is not established. Fleet savings are distinct from fixed-group intervention. Whiskers preserve source **95%** stratified world intervals, **5000 draws**, seed **20261028**. The **60/120 s** clocks are secondary; all eligible tiny/null groups remain included. The adverse **60 s** cycle-1 C_G **−2.982 [−6.131,−0.196] veh-s** remains in the complete frozen source CSV.” <!-- Sources: E:/research-dense/docs/paper_results/COOPMEAS2.md:3-20,39-70; E:/research-dense/docs/paper_results/COOPMEAS2.csv:2. -->
- **Placement:** V-A context; shrink to a primary-only inset if space is tight, keeping full PNG/CSV in artifact. It never replaces F2.

## F3V — completed offline WM+value predecessor

- **Purpose:** test whether a real-trained tail improves paired long-outcome prediction beyond full WM and no-WM alternatives.
- **Status: GENERATED — frozen necessary-condition FAIL.** [PNG](figures/F3V_failed_wm_value_gate.png), [PDF](figures/F3V_failed_wm_value_gate.pdf), [backing CSV](figures/F3V_failed_wm_value_gate.csv). Full A NA; no partial A estimate is plotted. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md. -->
- **Delivered source:** `E:/agent-work/c4-wmvalue/tools/wmvalue/evidence/short_test.json`, pinned `source_snapshots/wmvalue_short_test_7dcebed8a07b9132.json`; report `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md`. Exact metrics and all paired contrasts are exported with JSON pointers.
- **Layout:** full WM, WM60+V, WM90+V, command-aware no-WM V and FIT-only arm/time means; aligned sign agreement, MAE and ranking, plus paired sign-improvement/MAE-reduction CIs and each gate result. Both short horizons retained. These horizon labels are from the frozen design. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:40-71. -->
- **Generating command:** morning renderer above; no value fit/model rollout. Plot shows sign and MAE CIs; backing CSV includes Spearman and every B/C−D/E paired sign/MAE contrast. Complete frozen test has no surviving short method.
- **Caption requirements:** reserved TRAIN worlds, world-cluster uncertainty, same pair/time rows, real fleet+source cost endpoint, sensor/queue admissibility, recorded arm-specific continuation rather than a same-state policy intervention. Show every completed frozen baseline comparison; a decisive short-first FAIL may leave A NA, but no PASS bypasses A. Offline PASS does not establish C_G or learned gain. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md:148-159. -->
- **Placement:** V-B before policy outcomes.

## F3 — WM-primary policy interaction and traffic efficacy; oracle backup/reference

- **Purpose:** separate “C_G rises” from “beats MAPPO,” with development selection visibly labeled.
- **Status: PENDING for current COOPMWM A/B/R; audit v2 delivered in F2; backup unsuccessful in F3B.** [[PENDING: fixed proposal-free checkpoints, paired B−MAPPO/B−A/B−R efficacy and common singleton-alone C_G audit / S-COOPMWM, S-COOPMEASURE, S-COOPEVAL]]. Neither oracle planner F6 nor backup DEV fills current rows. Old B low-volume runs are superseded, not a completed guided-learning curve. <!-- Sources: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:38-70; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM.md:3-31. -->
- **Sources expected:** WMVALUE verdict, separate WM policy receipts, corrected S-SUMOCF receipts and S-COOPEVAL/audit exports. Distinguish WM-imagined group credit, real oracle sampled-window credit, deployed-mean dev pair diagnostics, and native-release audit C_G.
- **Layout:** WM-trained versus MAPPO (primary); backup versus MAPPO on matching contracts; group/solo/interaction and paired uncertainty; cost/safety and full candidate history. Gain-retention ratio is **NOT ESTIMABLE** at this cut because no beneficial backup denominator is established. Do not draw a ratio panel or use adaptive training means as a deployed improvement curve. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-40. -->
- **Generating command:** [[PENDING: schema-bound producer invocation and package renderer after actual receipts / S-COOPEVAL, S-PAPERA]]. The existing `tools/coopeval/cli.py` is a producer entry point, not a completed export receipt.
- **Contract gate:** W-GUIDE has a prospective public-metadata final-panel binding, not completed outcomes. Member-revert `C_G` and the paper's singleton-alone C_G must never share a forest-plot label: for a matched pair they have opposite signs (M-CREDIT). Keep window credit and episode/control-start J clocks separate. [[PENDING: branch/continuation mapping and source-bound common-audit export / S-COOPMWM, S-COOPMEASURE]]. <!-- Source: E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM_DESIGN.md:24-25,60-64. -->

## F3T — successor teacher → observable student: quality, efficacy and interaction

- **Purpose:** directly answer whether a WM-trained student generalizes useful group moves and beats MAPPO, with exact-teacher student as a separately labeled reference.
- **Status: GENERATED for completed exact-teacher student; PENDING for paused WM student.** [PNG](figures/F3T_exact_teacher_student.png), [PDF](figures/F3T_exact_teacher_student.pdf), [summary/event CSV](figures/F3T_exact_teacher_student.csv), [paired world CSV](figures/F3T_exact_teacher_student_worlds.csv). Source-pinned checks include producer QA, world subtraction/means, all-event interaction algebra and image integrity. WM teacher-quality is COMPLETE (F7); scale/student/live remain pending under explicit owner pause. Generated states must not be counted as labels. WM student's current live design supplies efficacy only, not new exact solos. [[PENDING: complete corpus/frozen student/live exports / S-WMTEACH]]. <!-- Sources: E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md; E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:26-64,103-113. -->
- **Data sources:** `E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md`, `DISTILL_LIVE_DESIGN.md`; `E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md`; shared authority `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`, MAIN11:47/11:48 and producer registrations. Future exact CSV locations must come from delivered producer manifests.
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_student_figures.py`; verify with `--check`. `student_figure_validation.json` binds immutable `distill_live/{SUMMARY,CREDITS,WORLD_METRICS,QA}.json` inputs. No producer training, live run or new CI computation.
- **Layout:** teacher quality (matched available-option sign/rank/regret, coverage and observation contract); honest outer-world CV and move/helpful-state agreement; complete paired student-minus-MAPPO J600/J900/safety on common fresh DEV; exact D_G/solos/C_G and pruned/executed counts separately. No positive gain ratio before a compatible resolved denominator.
- **Caption constraints:** EXAM fits only for S-DISTILL, imagined TRAIN/FIT targets only for WMTEACH; planner never-fit DEV remains post-freeze diagnostics. Distinguish final selection-CV from honest nested CV, exact-teacher **D600** labels from live **H120** credit, and WM **H300** student plan. Live audit quotas can force O0; no exact result selects actions. Both freeze/exposure histories required before common-panel comparison. <!-- Sources: E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md:12-34; E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL_LIVE_DESIGN.md:14-35; E:/agent-work/c4-wmteach/docs/paper_results/WMTEACH.md:19-43. -->
- **Delivered caption:** “Exact-teacher student, **24 fresh DEV worlds**, MAPPO−student **J600 −96.482 [−575.232,+288.473] / J900 −54.565 [−294.222,+188.886] veh-s/world**. Complete H120 credit for **six** executed groups, mean **C_G −35.890 [−59.837,+6.218] veh-s/action**, **zero** quota prunes. No group has D_G>0 and C_G>0. Source **95% whole-world bootstrap CIs / 10000 draws**, event mean includes inactive worlds and discloses **35** zero-event draws. Exact-teacher learning did not demonstrate MAPPO superiority; this is not WM training.” <!-- Sources: E:/agent-work/c4-distill-1101/docs/paper_results/DISTILL.md:31-44; E:/agent-work/c4-distill-1101/docs/paper_results/distill_live/SUMMARY.json; E:/agent-work/c4-distill-1101/docs/paper_results/distill_live/CREDITS.json. -->

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
- **Distributional boundary:** WMBREAKDOWN and ENSPROBE source-backed figures remain separate contracts. Final FULL-v4 report/publication now delivered; switching rule FAIL and full-world CIs NA, earlier intervals retracted. It cannot rehabilitate saved-state credits or fill policy outcomes. <!-- Sources: E:/agent-work/c4-wmbreakdown/docs/paper_results/WMBREAKDOWN.md; E:/agent-work/c4-ensprobe/docs/paper_results/ENSPROBE.md; E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:40-59. -->
- **Successor caption constraint:** retain adverse confidence diagnostics beside improved final CRPS and aggregate controller-effect signs. Keep reserved/development splits, available cases and source identities separate; A/B/C versus FULL are not settings-only ablations. Final report pin `source_snapshots/wmphys_morning_44ad6b2a1dd2cc09.md`; existing F4 inputs/artwork keep original pins. <!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:10-11,17-59. -->
- **Placement:** V-C; combine into a compact full-width figure after checking rendered label sizes.

## F4P — final FULL-v4: prediction improvement is not action qualification

- **Purpose:** show final WM2/FULL-v4 delay-distribution point scores, breakdown/HDV qualification contrasts and overconfident selected calls together, with no invented CIs.
- **Status: GENERATED, source-checked and visually inspected.** [PNG](figures/F4P_final_wmphys_diagnostics.png), [PDF](figures/F4P_final_wmphys_diagnostics.pdf), [backing CSV](figures/F4P_final_wmphys_diagnostics.csv). `noon_figure_validation.json` binds the final-report hash, plotted/exported cells and output hashes. Scientific verdict is complete.
- **Sources:** `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md`, final pin `source_snapshots/wmphys_morning_44ad6b2a1dd2cc09.md`; `E:/agent-work/c4-rb-int/docs/rbint/noon/20261101T113555+0800/PUBLICATION.json`. The publication explicitly reuses the verified complete frozen FULL-v4 snapshot after a redundant transfer timeout; no repeated inference is required.
- **Generating command:** `python -B docs/paper/paperA_cooperation/build_noon_figures.py`; verify with `--check`. Package-only summary transcription; no producer forecast, scoring or training invocation. The CSV also retains paired CRPS changes, HDV position change and range coverage; the plot uses the named subset without inventing whiskers.
- **Caption requirements:** complete model panels with primary reserved-world point comparison; full-world CIs NA and earlier numeric CIs retracted; all conditions required and breakdown-closeness FAIL; correct aggregate effect signs do not validate individual action probabilities or C_G; source absolute scores and paired differences rounded independently. SUMOCORE's different panel and startup-collision FAIL stay separate. <!-- Sources: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:17-59; E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md:22-27. -->

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
- **Status: corrected learner reports delivered without beneficial deployed example; exact-planner examples now delivered separately.** Planner selected gap-hold and H300 wall cases have exact joint/solo evidence; they are not learned or WM-selected. [[PENDING: independently replicated beneficial WM-policy example / MAIN WM policy producer, S-COOPBEHAV]]. <!-- Sources: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:165-236; E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:39-43,100-102. -->
- **Source expected:** S-COOPBEHAV exports from declared SUMOCF/audit traces; plan in `E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md`.
- **Generating command:** [[PENDING: producer's final exact trace-to-figure command / S-COOPBEHAV]].
- **Caption requirements:** outcome-selected TRAIN exemplars are exploratory, not deployed-policy discoveries. Keep fixed group/clock, issued-versus-realized actions, exposure/censoring and native/mean-reference semantics. u3 has no paired-supported mechanism; u15 contains supported events but even its largest positive-C_G record has negative D_G. Original-probe clusters stay withdrawn. Whole-controller rates do not isolate members. <!-- Sources: E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md:119-236; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md:41-53. -->
- **Placement:** one compact exemplar inset only if a valid paired trace is delivered; otherwise artifact pending, never a fabricated time-space plot.

## F6 — exact-model planner reference: benefit versus interaction

- **Purpose:** show what a correct predictor unlocks while keeping team savings, local synergy, horizon and cohort distinct.
- **Status: GENERATED.** [PNG](figures/F6_exact_model_planner.png), [PDF](figures/F6_exact_model_planner.pdf), [summary CSV](figures/F6_exact_model_planner.csv), [original byte-preserved pooled-world CSV](figures/F6_pooled36_source_worlds.csv).
- **Sources:** `E:/agent-work/c4-coopplan/docs/paper_results/coopplan/extension/final/SUMMARY.json`, `pooled36.csv`; separate `E:/agent-work/c4-coopplan/docs/paper_results/coopplan/h300/final/SUMMARY.json`. Immutable hashes/pointers in backing CSV and `morning_figure_validation.json`; source report `E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md`.
- **Generating command:** morning renderer above. Copied producer intervals; CSV identities, paired-cost subtraction and equal-world pooled means checked independently. No new bootstrap or episode is run.
- **Caption:** “H120 original/extension/pooled cohorts are separate views of the same complete benchmark, not independent replications. Pooled **36-world J600 +648.407 [209.699,1173.616] / J900 +1583.821 [588.246,2765.414] veh-s/world**; **52-action C_G +21.771 [−42.074,90.467] veh-s/action**, whole-world **95%** CIs. Separate H300 confirmation/wall cohorts improve J600 but not robust J900 or mean synergy. Every executed group has both solos; inactive worlds remain in world resampling, undefined zero-action draws disclosed. H300 and H120 differ in worlds/budget pruning: not a paired horizon effect. Exact offline planning is not learned WM cooperation.” <!-- Source: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:22-40,73-113. -->
- **Placement:** V-D main positive reference; compact H120 row may be used in manuscript, complete H300 panel retained in artifact. Producer's detailed existing wall-opportunity and all-action figures/CSVs remain at the explicit paths in COOPPLAN.md, with read-only generation commands there.

## T2 — physical outcomes and actual budget

- **Purpose:** expose safety/fairness/throughput and cost of WM/value acquisition/training, oracle reference and exact verification.
- **Status: BACKUP SAFETY/BUDGET DELIVERED; WM-policy and full fairness/service vector PENDING.** Backup TRAIN **3,194,716 actual / 3,205,151 charged sim-s**, DEV **184,049 actual=charged**; no DEV collisions/teleports, paired TTC intervals include zero. Keep invalidated work, prefixes and final drain in totals. [[PENDING: WM-policy physical outcomes, full service/fairness vector and total warm-start/value/data lineage / S-COOPEVAL, S-WMVALUE, MAIN WM policy producer, owner integrator]]. <!-- Source: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:48-88. -->
- **Source expected:** complete producer reports plus durable accounting summaries. Allocation is not consumption; screen-only no-step costs are not new SUMO panel measurements.
- **Closing accounting update:** COOPFREE original closed total remains historical; fresh continuation final accounting pending. WMPLAN held with **6480 charged/3480 actual sim-s**, no endpoint. COOPMWM reports **811640 charged sim-s** before acceleration restart; some jobs draining, so not a closed-actual total. Old **5,005,200** allocation is lineage only; owner redesign/enlarged final panel requires a new exact quote within **6000000** cap, retaining all prior charges. [[PENDING: final closed accounting and successor allocation / S-COOPFREE, S-COOPMWM]]. <!-- Sources: E:/agent-work/c4-coopfree/docs/paper_results/COOPFREE.md:313-429; E:/agent-work/c4-wmplan/docs/paper_results/WMPLAN.md:3-24; E:/agent-work/c4-coopnative/docs/paper_results/COOPMWM.md:3-14; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN19:38. -->
- **Generating command:** [[PENDING: schema-bound export and final IEEEtran table source / S-COOPEVAL, owner]].
- **Main-text minimum:** collisions/teleports/dropped steps, TTC/braking and worst-group/service effects alongside efficiency; incident zeros are not safety equivalence. Full components/cells remain in the artifact.
- **New delivered reference costs/safety:** exact planner **725280 actual=charged sim-s**; H120 braking **765→733 veh-s**, H300 confirmation **413.5→457.0**, zero collisions/teleports. COOPBANDIT **1567326 actual / 1569840 charged sim-s**; COOPFREE terminal **1952433 closed-actual=charged**, zero unclosed, adverse braking/discovery-analysis STOP. S-DISTILL now **72960 actual=charged**, not its **199200** upper reservation; MAPPO/student hard braking **452.5/438.5**, TTC exposure **13516.5/13580.5 veh-s**, zero collisions/teleports. These totals lack safety-superiority CIs. S-WMTEACH **100000** approved live cap is allocation, not consumption. <!-- Sources: E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md:45-48,105-108; E:/agent-work/c4-coopbandit/docs/paper_results/COOPBANDIT.md:58-59; E:/agent-work/c4-coopfree/tools/coopfree/publications/hold/HOLD_STOP.json; E:/agent-work/c4-distill-1101/docs/paper_results/distill_live/SUMMARY.json; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN11:48. -->

## Paper B figure routing

- Existing winner-by-condition PNG/PDF and CSVs: `E:/agent-work/c4-c1switch/docs/paper_results/c1switch/recorded_winner_by_condition.{png,pdf}`, `recorded_world_winners.csv`, `recorded_cell_winshares.csv`; command `python -m tools.c1switch.winners` in its worktree. Completed recorded/fresh headroom is descriptive, not switching efficacy; keep this figure in Paper B. <!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:15-19,31-42. -->
- Recovered wall/conditional-history panels remain Paper B. C1CLOSED **complete** CSV/PNG exist (`C1CLOSED_error_horizon.png` and source CSVs beside report), with inconclusive current endpoint superiority and common-plant scope; do not label them partial or import as WM-policy efficacy. <!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:2-48. -->
- Reference switching remains partial; switching-v2 incomplete with final-only fleet evaluation in progress after SSH-only recovery. No partial efficacy figure imported. Final FULL-v4 switching rule FAIL; point improvements do not override breakdown or missing uncertainty. <!-- Sources: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md; E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, MAIN11:37 CORRECTION and S-C1SWITCH2 NOON SUPERVISION. -->
