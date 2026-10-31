# Paper B — figure/table plan and delivered artwork

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
- **Status:** GENERATED; family-level, not a final-checkpoint architecture receipt. `[[PENDING: final checkpoint-specific component labels / S-C1CLOSED]]`.
- **Caption claim:** “WM-S separates executable CAV control, evaluated on predicted observations, from learned human dynamics and road/source accounting. Settings, residual gate and lane/merge machinery are versioned: historical residual-only outputs did not wire learned settings into the physics; later hybrid models do.”
- **Placement:** method; full-width schematic only if its small text remains readable. Do not add oracle future lanes or hidden types to the online input arrows.

### Feedback comparison / controller-selection evidence — selection panel delivered, feedback comparison pending

- **Purpose:** connect prediction to offline controller-choice regret and constant-controller references.
- **Files:** `figures/fig_regret.{png,pdf}`, `data/regret.csv`, `data/regret_paired.csv`; complete source export `data/fair.source.csv`.
- **Selectors:** `decisions/B_history_strict/{wm13e,bbtraj}/60/team`; oracle regret plus WM-row constant-selector regrets.
- **Source:** `E:/research-dense/docs/paper_results/E1_C1_fair.csv` and `.md`, strict-history team rows.
- **Generating command:** common render command, `extract()` and regret block in `render()`.
- **Status:** GENERATED exploratory selection panel including the current export's paired numerical contrast, now independently reconstructed in C1_GAPS with world resampling. `[[PENDING: full-feedback unseen-controller WM/BB comparison and exact team population/exposure binding / S-C1CLOSED + S-C1GAP]]`. No blank or invented feedback panel is emitted.
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
- **Placement:** supporting material/package, or a compact historical table if the pending feedback comparison is unavailable. It never fills the full-feedback result slot.

## Planned tables — specifications for the owner's LaTeX

### Protocol / exposure binding table

- **Purpose:** prevent checkpoint, feedback, and population mixing across evidence panels.
- **Columns:** checkpoint hash; corpus and controller exposure; evaluation controller/reference; observations/history; CAV plant; original/entrant lane ownership; entry/exit/source clock; metric/population; uncertainty unit; claim status.
- **Rows:** historical wall; strict-history team selection; frozen direction export; corrected WM2 recorded-command qualification; corrected-clock zero-head entry path; completed feedback comparison when available.
- **Data sources:** `CLAIMS.md` source pointers and source manifest; insert only receipt-supported cells.
- **Generating command:** no final table generator until `[[PENDING: C1/exposure/direction bindings / S-C1GAP + S-C1CLOSED]]`; owner may typeset the scoped source facts now with literal PENDING cells.
- **Status:** SPECIFIED / PENDING final bindings. No synthetic CSV of guessed bindings.

### Final feedback comparison table

- **Purpose:** the empirical headline: structured versus comparable black-box errors/effects under executable excluded-controller feedback.
- **Columns:** controller; horizon; eligible checkpoint; common worlds/starts; endpoint error; paired WM–BB difference/CI; effect sign/magnitude; oracle regret; execution caveat.
- **Source:** `[[PENDING: C1CLOSED.md and corresponding result CSV / S-C1CLOSED]]`.
- **Generating command:** `[[PENDING: agent's existing CSV export command; rendering only after receipt arrives / S-C1CLOSED]]`.
- **Status:** PENDING. No historical teacher-forced number is promoted to this table.

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
- **Status:** opportunity reports/CSVs complete; main efficacy `[[PENDING: corrected-adapter complete regret/episode results / S-C1SWITCH]]`. Current design/registration supersedes report-header main-status language. The anonymity-key retention STOP and later authorized corrected continuation are validation history, not negative or positive selector efficacy.
- **Placement:** optional supporting plot; not an additional main claim if the final selector campaign stays incomplete.

### Perturbation reference — preserve the exported missingness

- **Purpose:** document that a tiny native perturbation supplies no measured short-window chaos explanation while retaining aggregate sensitivity and late survivor loss.
- **Source/backing CSV:** `E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR_error_horizon.csv`; frozen copy `data/floor.source.csv`; report `data/floor.source.md`.
- **Generating command:** evening renderer snapshots the existing CSV only; agent source export is already complete. No paper-agent floor experiment or subtraction from WM error.
- **Caption/text claim:** “Two **+0.01 m/s** one-tick HDV pulses per native world, **six worlds**: original-HDV world-median endpoint errors **zero through 120 s**; no joint survivors at **300 s**, so endpoint cells unavailable, not zero. Aggregate delay sensitivity is nonzero. This is a limited pulse response, not an RNG floor.” <!-- Source: E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md:7-22. -->
- **Status:** source COMPLETE; supporting CSV, no combined WM/floor curve because population/arm/estimands differ. RNG-only floor remains unavailable.

## Visual QA

The task inspected every generated PNG. The schematic's initial arrows/text overlapped and were rerouted before delivery. Direction/entry titles were shortened/wrapped for column width. No numerical panel shares an axis with an incompatible metric or clock. `validate_package.py` verifies data selection, source hashes, figure files and provenance; this is package QA, not scientific requalification.
