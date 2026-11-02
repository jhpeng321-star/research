# Paper B — figure/table plan and delivered artwork

**Current refresh:** existing numerical figure sources are unchanged; delivered PNGs,
vector twins and backing CSVs remain valid at their declared evidence cuts. Current
switching coverage and successor/ranking/teacher diagnostics are specified below
without relabeling historical panels or creating missing paired estimates. The
owner-approved Overleaf delivery is separate; its empty figure boxes are preserved.

## Reproduction and provenance

From `E:/agent-work/c4-paperb`, render entirely from the package's frozen data:

```powershell
python -B "docs/paper/paperB_c1_worldmodel/build_figures.py"
```

To deliberately refresh the explicitly allowlisted existing E1/E2 CSVs and sibling reports before rendering:

```powershell
python -B "docs/paper/paperB_c1_worldmodel/build_figures.py" --refresh-inputs
```

The refresh command reads only the paths enumerated in `build_figures.py:INPUTS`; no globbing, model loading, fitting, resampling or simulation. `data/SOURCE_MANIFEST.json` stores exact source paths, byte counts and SHA256s. The complete allowed source CSVs and report snapshots are preserved alongside derived plot CSVs. Report extraction transcribes existing summary cells; it does not create raw observations or missing CIs. Rendering requires Python and Matplotlib (versions are recorded in `VERIFICATION.json`).

Every delivered figure has a PNG and vector PDF twin. Use the vector version at final typesetting and preserve aspect ratios. Numerical figures include units, explicit metric names, and source metadata; the schematic is qualitative. Keep the source comments below in the owner's Markdown drafting source. All plotting dimensions/dpi/font settings in code are editorial rendering settings, not experimental measurements.

## Main manuscript slots

### Architecture — delivered family schematic

- **Purpose:** make the executable-controller feedback loop distinct from learned human dynamics and source bookkeeping.
- **Files:** `figures/fig_architecture.{png,pdf}`; backing edge list `data/architecture_edges.csv`.
- **Source:** `E:/research-dense/docs/WM_STATE_PLAN.md`, model/addition; `E:/research-dense/docs/WM_CHANGE_LOG.md`, settings and lane history; local `src/wms/model/spec.py`.
- **Generating command:** common render command above, `architecture()`.
- **Status:** GENERATED; family-level. Current common-plant result binding is now delivered in C1CLOSED.md/analysis receipt; historical and later family labels remain essential. This schematic does not depict the separate evolving WM-PHYS candidate.
- **Caption claim:** “WM-S separates executable CAV control, evaluated on predicted observations, from learned human dynamics and road/source accounting. Settings, residual gate and lane/merge machinery are versioned: historical residual-only outputs did not wire learned settings into the physics; later hybrid models do.”
- **Placement:** method; full-width schematic only if its small text remains readable. Do not add oracle future lanes or hidden types to the online input arrows.

### Feedback comparison / controller-selection evidence — separate delivered protocols

- **Purpose:** connect prediction to offline controller-choice regret and constant-controller references.
- **Files:** `figures/fig_regret.{png,pdf}`, `data/regret.csv`, `data/regret_paired.csv`; complete source export `data/fair.source.csv`.
- **Selectors:** `decisions/B_history_strict/{wm13e,bbtraj}/60/team`; oracle regret plus WM-row constant-selector regrets.
- **Source:** `E:/research-dense/docs/paper_results/E1_C1_fair.csv` and `.md`, strict-history team rows.
- **Generating command:** common render command, `extract()` and regret block in `render()`.
- **Status:** GENERATED exploratory selection panel including the source's paired contrast, independently reconstructed in C1_GAPS. Exact historical team-cost/exposure binding remains `[[PENDING: original implementation/lineage / S-C1GAP + MAIN]]`. Completed autonomous feedback has a separate new panel below, not a relabeling of this one.
- **Caption claim:** “Exploratory strict-history offline team-cost selection at **60 s**, **32 starts / 21 worlds**. WM-13E regret **13.64 [5.382,26.2] veh-s**, BB-traj **22.05 [10.05,37.86]**; constant selectors use the same exported metric. Upper bars are source marginal **95%** CIs. Lower panel separately reproduces the source's paired **WM-minus-BB regret −8.412 [−19.23,−0.3071] veh-s**; it is not inferred from the marginal bars.” <!-- Sources: E:/research-dense/docs/paper_results/E1_C1_fair.md:159-199,2100; CI level E:/research-dense/docs/paper/IEEE_IV_2027_DRAFT.md:107,133. -->
- **Placement:** prediction/decision results. Do not relabel this as a policy-training result or deployed receding-horizon selector evaluation.

### Horizon boundary — delivered frozen direction curve

- **Purpose:** show the source's tested sign envelope, including failure.
- **Files:** `figures/fig_direction.{png,pdf}`, `data/direction.csv`; complete source `data/direction.source.csv`.
- **Source:** `E:/research-dense/docs/paper_results/E2_direction_horizon.csv` and `.md`.
- **Generating command:** common render command, direction block in `render()`.
- **Status:** GENERATED; source verdicts retained; `[[PENDING: exact checkpoint, panel and decision-rule binding / S-C1GAP]]`.
- **Caption claim:** “On the frozen registered direction panel, tested **10/30/60/90 s** clocks PASS and **120/180 s** FAIL, across **54 worlds**, threshold **0.8684**. Whiskers reproduce source world-cluster CIs. Verdict labels are exported source decisions, not a new CI-threshold rule. This panel is distinct from corrected WM2 J qualification and overall fidelity.” <!-- Source: E:/research-dense/docs/paper_results/E2_direction_horizon.md:3-10. -->
- **Placement:** reliability subsection. Lines connect tested clocks for readability, not proof at all intervening clocks. Do not extend the curve to unobserved horizons.

### Regional error and entry repair — delivered separate panels for a shared diagnostic slot

- **Purpose:** expose regional concentration and error cancellation, while preserving different clocks/estimands.
- **Files:** `figures/fig_regional.{png,pdf}`, `data/regional.csv`; `figures/fig_entry.{png,pdf}`, `data/entry.csv`.
- **Sources:** `E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md`, Additive errors; `E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md`, Result and attribution.
- **Generating command:** common render command, report extraction and regional/entry blocks in `render()`.
- **Status:** GENERATED from source report cells. No uncertainty bars are invented; these diagnostic report cells lack component CIs. Keep individual captions/clock labels if composing side by side in LaTeX.
- **Regional caption claim:** “Corrected matched recorded-intervention J error at **300 s**: upstream/merge road contributes **96.0%** of squared-error covariance, source **−7.1%**, downstream **+11.1%**. Share is `100 E[e_component e_J]/E[e_J²]`, not a causal fraction or MAE fraction. Common total-J order-statistic weights preserve additive error accounting; one-tick source clock. Originals replay recorded longitudinal commands; entrants use feedback.” <!-- Source: E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md:7-8,31-40. -->
- **Entry caption claim:** “Under the separate left-boundary-clock comparison at **300 s**, zero-head entry-path repair reduces source MAE **584.0→82.5 veh-s**, with upstream **1456.9→1393.0**, while total J MAE worsens **1449.1→1536.2 veh-s**. Better source accounting removes partial cancellation with road error; it does not qualify total prediction.” <!-- Source: E:/agent-work/c4-wmfixmerge/docs/paper_results/WMFIX_RETARGET.md:25-48. -->
- **Placement:** reliability/diagnosis. Signed bars, not a pie chart. Do not label the upstream/merge region a measured merge-queue causal mechanism.

## Supporting artwork — historical teacher-forced response

- **Purpose:** illustrate the historical unseen-wall trajectory evidence without conflating it with feedback control.
- **Files:** `figures/fig_wall_teacher_forced.{png,pdf}`, `data/wall_teacher_forced.csv`; recovered-report snapshot `data/wall.source.md`.
- **Source:** `E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md`, `wall_same` HDV error table and caveats; recovery provenance in adjacent `recovery.json` (reviewed by this task, not copied as raw state data).
- **Generating command:** common render command, wall extraction and wall block in `render()`.
- **Status:** GENERATED from existing report cells, rather than a preexisting wall CSV. This plot's new CSV is a traceable tabular transcription of those cells. No marginal error bars are supplied because the original table's IQR is dispersion, not uncertainty. The newly delivered C1_GAPS independently supplies marginal CIs at the short headline clock; these are recorded in the ledger and frozen audit snapshot, not retroactively treated as IQR-derived bars. Paired world-bootstrap differences are not inferred from plotted median gaps.
- **Caption claim:** “Historical unseen-wall HDV response prediction, entries disabled and recorded CAV motion supplied to structured and BB-traj. BB-id learns CAV motion and is not teacher matched. At **60 s**, the reported median errors are **8.71 / 25.41 / 40.07 m**, respectively; **67 states / 29 worlds**. The longer-clock count differs and cohorts must not be pooled.” <!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md:7-28,290-296. -->
- **Placement:** supporting material/package, or a compact historical table alongside the completed common-plant feedback comparison. It never fills the full-feedback result slot.

## Planned tables — specifications for the owner's LaTeX

### Protocol / exposure binding table

- **Purpose:** prevent checkpoint, feedback, and population mixing across evidence panels.
- **Columns:** checkpoint hash; corpus and controller exposure; evaluation controller/reference; observations/history; CAV plant; original/entrant lane ownership; entry/exit/source clock; metric/population; uncertainty unit; claim status.
- **Rows:** historical wall; strict-history team selection; frozen direction export; corrected WM2 recorded-command qualification; corrected-clock zero-head entry path; completed C1CLOSED common-plant feedback; partial reference switching; pending wall-inclusive main switching.
- **Data sources:** `CLAIMS.md` source pointers and source manifest; insert only receipt-supported cells.
- **Generating command:** owner typesets receipt-supported protocol cells; final complete table awaits `[[PENDING: remaining initialization/exposure/direction bindings / S-C1GAP + MAIN]]`. Completed C1CLOSED cells can be filled from its report/CSV/analysis receipt now.
- **Status:** SPECIFIED / PENDING final bindings. No synthetic CSV of guessed bindings.

### Final feedback comparison table

- **Purpose:** show the completed scoped structured-versus-black-box comparison under executable feedback, including inconclusive superiority and decision-trust limits.
- **Columns:** controller; horizon; eligible checkpoint; common worlds/starts; endpoint error; paired WM–BB difference/CI; effect sign/magnitude; oracle regret; execution caveat.
- **Source:** `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md`, `C1CLOSED_bootstrap.csv`, `C1CLOSED_ANALYSIS.json`; frozen package source/selected CSVs described below.
- **Generating command:** overnight renderer below selects existing estimates and CIs only. Producer analysis command in its worktree is `python -B tools/c1closed/analyze.py`; the package does not rerun it.
- **Status:** COMPLETE scoped comparison; final table can use `data/closed_loop.csv`. Principal-clock endpoint superiority is INCONCLUSIVE, earliest contrasts favor BB, and wall feedback stays unavailable. Keep cohort/plant caveats and current-model direction/regret alongside relative delay gains.

### Calibration / negative-result row — complete negative verdict

- **Purpose:** show whether completed calibration changes decision trust beyond raw WM and arm-mean, while retaining failed repair outcomes.
- **Columns:** selected calibrator; frozen selection scope; grouped CV; independent transfer; matched reference/arms; sign/MAE/rank; final gate verdict.
- **Sources:** `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md`, grouped CV/SPEEDHEAD tables and final verdict; earlier repair failure reports retain their separate scope.
- **Generating command:** bounded `build_evening_figures.py` report-to-CSV transcription/render, described below; no calibrator, fitting or inference.
- **Status:** final calibration NOT USABLE, no longer PENDING. Manuscript can use a compact sentence plus split-labeled MAE panel; keep the multimetric gate in the caption, not an implication that MAE alone qualifies credit.

## Evening additions — existing-evidence rendering plan

Render command from the Paper B worktree:

```powershell
python -B "docs/paper/paperB_c1_worldmodel/build_evening_figures.py" --refresh-inputs
```

Subsequent reproduction omits `--refresh-inputs` and uses frozen package snapshots. Exact allowlist/hash receipt: `data/EVENING_SOURCE_MANIFEST.json`. These additions use source tables and existing CSVs only; no new bootstrap, model execution, fitting or simulator run. **Status: GENERATED and visually reviewed**; backing rows, source hashes, tie accounting and late missingness are checked by `validate_package.py`.

### PI diagnosis — separated starts and paired endpoint contrasts

- **Purpose:** show the adverse original-start short-clock comparison alongside the genuine-history extension without falsely drawing one horizon curve.
- **Delivered files:** `figures/fig_pi_diagnosis.{png,pdf}`, `data/pi_diagnosis.csv`, frozen `data/pidiag.source.md`.
- **Source:** `E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md`, original/shifted teacher-forced WM2 rows and within-shifted-start history contrast.
- **Generating command:** common evening renderer; report row/contrast extraction.
- **Caption claim:** “Common-ID WM2−BB-traj endpoint error: original snapshot starts **+17.85 [3.24,31.27] m at 60 s**, **−47.82 [−95.20,−27.75] at 120 s**; separate full-history shifted starts **−45.05 [−52.73,−25.25] / −144.74 [−250.11,−36.22] m**. Within shifted PI starts, full-history−snapshot **−13.47 [−26.92,−1.49] m at 60 s**. All are descriptive world-bootstrap intervals; teacher-forced CAVs and conditional presence, entries OFF. Different start clocks are not a trajectory curve or an isolated history intervention.” <!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:7-14,25-37,45. -->
- **Placement:** compact prediction-results supporting panel; wall results and reset-controller feedback are not pooled. A favorable endpoint contrast does not establish regret or full-feedback superiority.

### Calibration — grouped CV versus independent transfer

- **Purpose:** make the no-WM comparison and transfer failure visible.
- **Delivered files:** `figures/fig_calibration.{png,pdf}`, `data/calibration.csv`, frozen `data/calibration.source.md`.
- **Source:** `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md`, both complete split tables.
- **Generating command:** common evening renderer; exact MAE/sign/correlation/order transcription, plotting MAE only.
- **Caption claim:** “Paired-effect MAE on grouped CV at **120/300/600 s** is raw WM2 **263/1719/6013 veh-s**, no-WM arm-mean **185/999/3298**, selected calibration **174/865/3152**. On SPEEDHEAD transfer selected calibration is **212/1783/4754**, raw **176/1399/4701**, arm-mean **198/1568/4073**. Frozen multimetric gate fails transfer at every clock: NOT USABLE for long-horizon credit. CV selection is not independent confirmation; changed-reference WMREQUAL3 transfer is not plotted here.” <!-- Source: E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md:9-50,58-60. -->
- **Placement:** trust-limit supporting panel or compact table. No invented CI bars, and no universal raw-versus-arm-mean ordering beyond the stated split.

### Switching opportunity — recorded scenarios and fresh same-state choices

- **Purpose:** separate scenario diversity from measured same-state headroom and from still-missing predictor efficacy.
- **Delivered files:** `figures/fig_switch_headroom.{png,pdf}`, `data/switch_headroom.csv`; original full backing exports `data/recorded_winners.source.csv`, `data/step0b.source.csv`. The agent's existing richer figure is `E:/agent-work/c4-c1switch/docs/paper_results/c1switch/recorded_winner_by_condition.png` (vector twin beside it).
- **Sources:** `E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md` and the two named CSVs in its `c1switch/` directory. Original condition-figure generating command is `python -m tools.c1switch.winners` in that worktree (analysis only); the bounded evening renderer creates the compact package panel from its existing exports.
- **Caption claim:** “Recorded **600 s** unique wins across **90 worlds**: native **22**, CTRL0 **17**, MAPPO **19**, PI **29**, CIRCLES **3**. Fresh full-rerun **16-state** headroom: at **60 s** native/CTRL0/MAPPO unique wins **2/9/3**, plus **two ties**; at **90 s**, native/CTRL0/MAPPO/PI/CIRCLES **2/7/1/3/1**, plus **two ties**. Median best-to-second margins **1.832371 / 6.853775 veh-s**, including ties. Separate panels count worlds versus states. These are oracle opportunities, not demonstrated WM/BB switching gains.” <!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:5-13,31-39. -->
- **Status:** opportunity reports/CSVs complete. Reference main stopped incomplete; its authorized partial analysis is summarized in C1-SWITCH and retained as `data/switch_partial.source.md`, without an efficacy headline figure. Main wall-inclusive WM-versus-RL evidence is `[[PENDING: complete paired picker results / S-C1SWITCH2 + selector agents]]`. Opportunity artwork remains valid; neither winner panel implies predictor benefit.
- **Placement:** optional supporting plot; not an additional main claim if the final selector campaign stays incomplete.

### Perturbation reference — preserve the exported missingness

- **Purpose:** document that a tiny native perturbation supplies no measured short-window chaos explanation while retaining aggregate sensitivity and late survivor loss.
- **Source/backing CSV:** `E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR_error_horizon.csv`; frozen copy `data/floor.source.csv`; report `data/floor.source.md`.
- **Generating command:** evening renderer snapshots the existing CSV only; agent source export is already complete. No paper-agent floor experiment or subtraction from WM error.
- **Caption/text claim:** “Two **+0.01 m/s** one-tick HDV pulses per native world, **six worlds**: original-HDV world-median endpoint errors **zero through 120 s**; no joint survivors at **300 s**, so endpoint cells unavailable, not zero. Aggregate delay sensitivity is nonzero. This is a limited pulse response, not an RNG floor.” <!-- Source: E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md:7-22. -->
- **Status:** source COMPLETE; supporting CSV, no combined WM/floor curve because population/arm/estimands differ. RNG-only floor remains unavailable.

## Overnight completed-result artwork

```powershell
python -B "docs/paper/paperB_c1_worldmodel/build_overnight_figures.py" --refresh-inputs
```

Omit `--refresh-inputs` to reproduce from local frozen bytes. Exact safe allowlist and
hashes: `build_overnight_figures.py:INPUTS`, `data/OVERNIGHT_SOURCE_MANIFEST.json`.
Selection/transcription only: no experiment, model execution, fit or resampling.
**Status: GENERATED and visually inspected.** Historical/evening artwork is preserved.

### Completed autonomous common-plant contrast

- **Purpose:** separate paired endpoint-error evidence from relative aggregate-delay gains and controller-judge trust.
- **Files:** `figures/fig_closed_loop.{png,pdf}`, backing `data/closed_loop.csv`; exact producer bootstrap CSV `data/closed.source.csv`, report/runtime snapshots beside it.
- **Source:** `E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED_bootstrap.csv`; select `wm2-minus-bbtraj/common`, PI/MAPPO, `position` and `J_ae`. Every source clock, including missing/sparse endpoint cells, stays in the backing CSV.
- **Generating command:** overnight renderer, `extract()`/`render()`; no CI reconstruction from marginal summaries.
- **Caption claim:** “Completed autonomous common-plant comparison, **40 worlds**, genuine **60 s** history. Paired WM2−BB-traj endpoint error at **60 s**: PI **−5.17 [−18.02,+7.09] m**, MAPPO **−3.45 [−13.42,+4.58] m**; neither excludes zero. Earliest endpoints favor BB. At **120 s**, endpoint cohort falls to **33 worlds**; sparse **180 s** and unavailable **300 s** endpoints stay in the CSV. Aggregate J-error contrasts retain **40 worlds**, with relative WM gains at longer clocks; these do not establish accurate effects/ranking or low absolute error. Shading reproduces producer **95% whole-world-bootstrap intervals** at tested clocks, not simultaneous functional bands.” <!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:4-12,14-30,35-48; C1CLOSED_bootstrap.csv,wm2-minus-bbtraj/common/pi,mappo/position,J_ae. -->
- **Scope/placement:** current autonomous results, separate from teacher-forced wall and E1 selector. Approximate common CAV plant, survivor-conditioned endpoints, differing baseline corpus, provisional exclusion lineage and failed exact cross-host parity remain visible.

### Distributional long-tail diagnosis

- **Purpose:** test the claim that only realized-path error is poor while traffic distributions remain useful.
- **Files:** `figures/fig_distribution.{png,pdf}`, backing `data/distribution.csv`, exact `data/distribution.source.csv` and report snapshot.
- **Source/selectors:** `E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown/distribution_summary.csv`; `reserved8`, `team`, `pooled`, targets `absolute/effect`. Absolute comparator is `cell_arm_climatology_crps`; effect comparator is `climatology_crps`. These are explicitly different baseline columns.
- **Generating command:** overnight renderer; all selected source columns/CIs retained in backing CSV. No fabricated marginal bars or inferred significance from displayed columns.
- **Caption claim:** “Separately reported **eight-world** evaluation slice, previously used by calibration. At **120/300/600 s**, absolute team-delay CRPS WM **1522/7129/25523**, cell×arm climatology **763/3584/8267 veh-s**. Arm−CTRL0 effect CRPS WM **390/2420/6906**, paired-effect climatology **171/968/3286**. At **600 s**, absolute K8 range coverage **12.5%**, real below all futures **87.5%**. Baselines use development outcomes only; effect dispersion can be excessive even when absolute futures miss real delay. Cost-only imagined futures do not supply new-panel event-rate/timing tests.” <!-- Source: E:/agent-work/c4-wmbreakdown/docs/paper_results/WMBREAKDOWN.md:6-10,23-30; distribution_summary.csv,reserved8/team/pooled/absolute,effect. -->
- **Placement:** optional reliability panel or compact source-bound sentence under page pressure; proposed future qualification gate is not a measured pass.

### Short imagination plus value — complete negative gate

- **Purpose:** show the failure of a real-value tail to turn short-range WM predictions into long-run controller credit.
- **Files:** `figures/fig_value_tail.{png,pdf}`, `data/value_tail.csv`, `data/value.source.md`.
- **Source:** `E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md`, B/C/D/E method table; gate uses paired contrast table, not overlap of these marginal CIs.
- **Generating command:** overnight renderer report transcription. No missing A row is manufactured.
- **Caption claim:** “Frozen test on **eight reserved FIT worlds / 400 pairs**: WM60+V sign **47.25%**, MAE **8877.72 veh-s**; WM90+V **43.25% / 10117.34**; V+commands **58.75% / 4460.00**; arm/time mean **72.25% / 3038.93**. Source **95% world-bootstrap CIs** shown. Both short methods fail necessary sign-and-MAE improvement against each no-WM comparator; the full conjunction FAILS regardless of unmeasured long-WM A. Remaining fleet+source cost to the fixed absolute endpoint is not a short-window cost.” <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md:4-28,33-38. -->
- **Placement:** supporting negative-result artwork, or compact text; it is not evidence for a validated policy plug-in.

### Main wall-inclusive WM-versus-RL switching

- **Purpose:** the requested controller-choice comparison with the primary fleet-PPO baseline and secondary per-CAV MAPPO selector.
- **Planned columns/panels:** full-team episode cost/paired differences; decision regret; each model's training/data/compute budget; library/cadence/plant/handovers; safety; complete coverage; optional independently qualified model version.
- **Sources:** `E:/agent-work/c4-c1switch2/docs/paper_results/C1SWITCH2_DESIGN.md`, `HANDOFF.md`, eventual complete identity-bound outputs and final selector receipts.
- **Generating command/status:** `[[PENDING: complete paired coverage/exports and producer analysis/render command after authorized continuation / S-C1SWITCH2 + MAIN]]`. Both final checkpoint/source/config/budget receipts are delivered as `SELMAPPO_FINAL.json` and `SELMAPPO2_FINAL.json` in the source worktree. Fleet evaluation is complete, while WM/per-CAV coverage and BB physics are incomplete/unavailable. The latest switching reconciliation records exited supervisors/observer, an active GPU hold and an unstarted-scope CPU continuation request awaiting authority. No partial main efficacy figure. Sources: `E:/agent-work/c4-c1switch2/HANDOFF.md`, Current execution state; selector reports in `SOURCE_REFRESH.json`.

### Physics rebuild — final diagnostic ledger; no adoption claim

- **Purpose:** preserve the independent adoption test without turning favorable available-case cells into a qualified replacement model.
- **Sources available now:** final `E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md`; `docs/rbint/noon/20261101T113555+0800/PUBLICATION.json` in that worktree binds the frozen FULL-v4 comparison, endpoint and distribution/credibility exports under `docs/rbint/noon/20261101T093000+0800/FULL-v4/`. Exact paths/hashes are in `SOURCE_REFRESH.json`. The final publication explicitly reuses complete same-source forecasts; do not recollect them.
- **Final status:** delay-distribution/endpoint point estimates and controller-effect signs improve, but breakdown-rate distance worsens: final point-rule FAIL, keep WM2. Earlier numerical CIs were retracted; full-world uncertainty remains unavailable under the sparse-stratum plan. Same-source action probabilities are not calibrated. Earlier-source A/B/C and later FULL cannot form a settings-only ablation figure. Final report/exports/adoption decision are resolved, not PENDING.
- **Editorial use:** concise results text from WM-PHYS in `CLAIMS.md`, or an owner-typeset diagnostic row with separate metrics/units, explicit “point estimates; full-world uncertainty unavailable” and the failed gate beside improvements. No winner/adoption graphic. The existing artwork remains source-valid; unchanged inputs need no rerender. Successor interim support and remaining gate are specified next.
- **Companion figures:** exact-planner artwork remains owned by Paper A and `E:/agent-work/c4-coopplan/docs/paper_results/COOPPLAN.md`. Exact-teacher student live exports and WM-teacher quality reports are now complete; use `DISTILL.md` and `WMTEACH.md` at the explicit paths in `SOURCE_REFRESH.json`. WM scale/student/live transfer is paused and remains `[[PENDING: completed WM-teacher scale-labelled corpus/frozen student/live transfer / S-WMTEACH + MAIN]]`. No companion artwork is needed for the core Paper B prediction claim.

### Cached ranking versus useful selection — optional diagnostic row

- **Purpose:** test whether good ordering or lower predicted breakdown probability alone supplies useful controller selection.
- **Data source:** `E:/agent-work/c4-bdrank/docs/paper_results/BDRANK.md`, Q2/Q3; existing CSV/evidence index `E:/agent-work/c4-bdrank/docs/paper_results/bdrank/README.md`. Exact report hash is in `SOURCE_REFRESH.json`.
- **Generating command:** no new renderer; owner typesets the already reported C1-RANK cells with separate development/reserved and controller/available-option labels. Producer reproduction is documented in `E:/agent-work/c4-bdrank/tools/bdrank/README.md`; no producer analysis is restarted here.
- **Status:** completed cached analysis, optional manuscript row; no new image generated. Development ranking favors mean delay, probability saturation hides choices, and development-selected rules lose on the separate reserved panel. Keep tie-adjusted and untied metrics distinct; singleton-stratum full-panel CIs remain unavailable. Not a live picker result or causal breakdown-cost estimate.

### Successor physics — adverse interim, incomplete final gate

- **Purpose:** keep later engineering readiness and matched-cohort uncertainty distinct from a passed adoption condition.
- **Data source:** `E:/agent-work/c4-rb2-int/docs/paper_results/WMPHYS_V5.md`, available-replicated-cell and separate reserved-subset interims; report hash in `SOURCE_REFRESH.json`.
- **Generating command:** owner-typeset optional diagnostic row from WM-SUCCESSOR; no final-result renderer or new image. Do not pool cohorts or fabricate unobserved short source clocks.
- **Caption claim if used:** “On **10 matched complete K8 worlds / five replicated cells**, v5−v4 delay CRPS is **+166.9 [64.6,269.3] / +1046.1 [356.5,1735.7] / +3618.5 [2051.0,5185.9] veh-s at 120/300/600 s**. Conditional within-family **95%** intervals; available-data interim, not final qualification or sequential inference. The separate reserved subset has singleton-stratum CI limitations.” <!-- Source: E:/agent-work/c4-rb2-int/docs/paper_results/WMPHYS_V5.md:69-83,121-134. -->
- **Status:** INTERIM source complete; full comparison exited incomplete/held. `[[PENDING: complete final successor comparison/adoption and calibrated confidence after authorized continuation / S-RB2-INT + MAIN]]`. Short team-direction accuracy is partially identified; road-only and resolved-pair accuracy must not fill a missing full-team accuracy column.

## Visual QA

The task inspected every generated PNG. The schematic's initial arrows/text overlapped and were rerouted before delivery. Direction/entry titles were shortened/wrapped for column width. No numerical panel shares an axis with an incompatible metric or clock. `validate_package.py` verifies data selection, source hashes, figure files and provenance; this is package QA, not scientific requalification.
