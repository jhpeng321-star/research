# Paper B — paragraph-by-paragraph drafting outline

## Title and governing claim

**Working title:** “WM-S: Physics-Informed Prediction under Unseen Traffic Controllers and the Limits of Its Verdicts.”

**Intended headline:** zero-shot closed-loop unseen-controller prediction versus black-box prediction, with explicit evidence-bounded controller-selection use. The completed full-feedback superiority sentence remains `[[PENDING: C1CLOSED.md / S-C1CLOSED]]`. The owner can draft the model, historical response evidence, decision evidence, and failure envelope now.

**Contribution bullets to preserve:** executable-controller/environment factorization; controller-exclusion evaluation separating recorded-action and feedback protocols; control-oriented reliability diagnosis linking regret, horizons, entrance accounting and regional error. Novelty is the combination and its measurements, not invention of physics-informed WMs, residual dynamics, or simulator admissibility.

## Format and page budget

Use IEEE conference two-column `IEEEtran`, numbered sections/equations/citations, readable figure text and IEEE bibliography style. Owner target is **six pages including references**, deadline **2026-11-15**. <!-- Source: owner S-PAPERB brief. -->

Editorial allocations, not measured pagination: title/abstract/introduction **0.85**; related work **0.45**; problem/protocol **0.55**; method **1.00**; experimental design **0.55**; results **1.55**; discussion/conclusion **0.40**; references **0.65**, total **6.00 pages**. Figure/table space is included within these allocations. <!-- Source: this package's editorial plan, derived to satisfy owner six-page constraint; not empirical data. -->

At typesetting, shorten prior-attempt history before dropping protocol caveats or negative results. Reserve architecture, feedback comparison/selection, and reliability panels; place expansive qualification history in repository artifacts.

## Abstract — exact sentence claims

- **A-P1:** New CAV controllers require forecasts of surrounding human responses under changed feedback actions, motivating controller/environment factorization. No novelty priority claim.
- **A-P2:** WM-S combines a versioned IDM/MOBIL-style core, history-conditioned settings, gated residual and entry/merge handling with executable controller code. Cite C1-FACTOR, WM-SETTINGS, WM-LANES.
- **A-P3:** `[[PENDING: exact eligible closed-loop WM-versus-BB result / S-C1CLOSED]]`; until supplied, say historical teacher-forced response and exploratory feedback-selection evidence support distinct limited capabilities.
- **A-P4:** Report the bounded tested direction envelope and the distinction between source-error repair and unresolved road error; use HORIZON-SIGN and ERR-ENTRY with their source comments.
- **A-P5:** These measurements motivate a declared controller/horizon/population envelope for choosing controllers; no universal certification claim.

## I. Introduction

- **I-P1 — exact claim:** Trajectory plausibility alone cannot answer which unseen feedback controller produces lower traffic cost; controllers react to their own predicted consequences. Establish the prediction/evaluation problem, not policy-learning efficacy.
- **I-P2 — exact claim:** Separating known controller execution from unknown human response provides an architectural route to controller transfer without fitting a controller-label map. State that transfer must be measured under controller-data exclusion. Evidence C1-FACTOR/C1-EXPOSURE.
- **I-P3 — exact claim:** Controller assessment can fail despite good short-clock state error because lane timing, source waiting and upstream traffic alter aggregate intervention effects. Evidence HORIZON-UNIVERSAL/ERR-REGION/ERR-ENTRY.
- **I-P4 — exact claim:** This paper contributes the factorization, a protocol-separated transfer study, and a control-oriented reliability account. Qualify the full-feedback comparison with C1-CLOSED; do not advertise control-policy superiority or learned-driver type recovery.

## II. Related work

- **II-P1 — exact claim:** IDM and MOBIL supply analytical behavior priors; traffic residual modeling and physics-informed offline multi-agent WMs already exist. Cite `treiber_idm`, `kesting_mobil`, `sheng_residual`, `liu_physics_wm`; explain WM-S's explicit microscopic transition versus Liu's physics-supervised state inference and learned transition/reward ensemble.
- **II-P2 — exact claim:** Learned driving WMs and interactive simulators address action-conditioned scene evolution; trajectory forecasting is related but a different evaluation contract. Cite `hu_gaia1`, `gulino_waymax`, `salzmann_trajectronpp`, `nayakanti_wayformer`. The in-house BB baselines are not these published systems.
- **II-P3 — exact claim:** Admissibility literature already argues that verdicts require action-conditioned validation. WM-S contributes traffic-specific paired effects, regret, population accounting and failure localization, not a new universal certification standard. Cite `oefinger_admissibility`, `rahmani_survey`; refer to `RELATED_WORK.md` for exact comparison bounds.

## III. Problem and evaluation contracts

- **III-P1 — exact claim:** Given permitted history and common initial road/source state, predict the traffic consequences of controller code executed on predicted observations. Define `u=pi(o(x_hat))`, `x_hat'=F_theta(x_hat,u,xi)`. No future observations re-anchor the horizon. Teacher forcing of CAVs is distinct from feedback controller execution; historical uses of “open loop” may refer to no re-anchoring, not absence of controller feedback.
- **III-P2 — exact claim:** Define cost `J`, intervention difference `Delta J=J(pi)-J(ref)` (negative favors the controller), offline choice and oracle regret. All-request source+road J, exported E1 `team`, and initial-vehicle proxies retain distinct names until bound. Evidence C1-MAPPO and ERR-REGION.
- **III-P3 — exact claim:** Zero-shot means before exposure to the evaluated controller's outcomes. Specify checkpoint/corpus/exposure, originals/entrants, lane ownership, observation channels, reference arm, and WORLD sampling unit. Same-world other-arm training is permitted only when explicitly labeled; no unseen-world wording. `[[PENDING: hash-linked protocol table / S-C1GAP + S-C1CLOSED]]`.

## IV. WM-S method

- **IV-P1 — exact claim:** Driver history/neighbors condition learned physical settings inside later hybrid IDM transitions, with defaults for limited observations; historical residual-only configurations are distinguished. Include the compact equation `a_hat=a_IDM(x_hat; eta_hat)+g(k) r_theta`. Evidence WM-SETTINGS.
- **IV-P2 — exact claim:** A step-aware gated correction models physics mismatch; gate and bounded-lane mechanisms are checkpoint-dependent. The current evidence does not isolate universal component gains. Evidence change log, WM-LANES and WM-REPAIR.
- **IV-P3 — exact claim:** Lane/merge event timing and open-road request/admission/exit bookkeeping connect individual predictions to road/source outcomes. Account for originals and entrants with explicit command/lane ownership. Evidence C1-FACTOR, ERR-ENTRY, corrected qualification report.
- **IV-P4 — exact claim:** Training is multi-step trajectory supervision of human dynamics; controller mechanisms remain executable code. Give final history, loss, curriculum and gate details only from the final eligible receipt. `[[PENDING: final model/training binding and ablations / S-C1CLOSED]]`.
- **Visual:** architecture schematic, with historical/later variants distinguished; figure specification in `FIGURES.md`.

## V. Experiments

- **V-P1 — exact claim:** Simulator highway-merge study uses existing evidence panels with different checkpoints and observation/execution contracts. Specify network/config identities from eligible receipts; describe synthetic simulator behavior without reading assignment files. `[[PENDING: compact experimental binding / S-C1GAP + S-C1CLOSED]]`.
- **V-P2 — exact claim:** Compare structured, in-house BB-traj, BB-id where supported, and simple physics/keep-speed references under declared training/input contracts. BB-id's learned CAV path is not equivalent to the teacher-matched BB-traj path; absent feedback capability is unavailable, not a loss.
- **V-P3 — exact claim:** Preserve source uncertainty and panel membership. Regret comparisons require matched controller/cost support; source repair is a paired diagnostic; regional share is covariance accounting. Mark exploratory outcomes, exclusions, missing histories and metric denominators. Do not create new pooled intervals from exported marginal intervals.

## VI. Results

### A. Unseen-controller prediction and short-clock choice

- **VI-A-P1 — exact claim:** The recovered historical unseen-wall report supports structured HDV-response error superiority over BB-traj with paired world-bootstrap uncertainty, under recorded CAV motion; BB-id is separately qualified. Include C1-WALL point values and paired contrasts from `CLAIMS.md`, with source comments. `wall_pilot` is sensitivity, not an independent pooled replication.
- **VI-A-P2 — exact claim:** `[[PENDING: full-feedback unseen PI/wall or other excluded-controller comparison and paired result / S-C1CLOSED]]`. If unavailable, explicitly state the full-feedback extension remains unresolved; the historical sentence cannot fill this paragraph.
- **VI-A-P3 — exact claim:** Strict-history E1 offers exploratory offline controller-choice evidence with measured oracle regret and constant-selector references. Report WM and BB marginal regret intervals without asserting a paired model win; include the difference versus always-MAPPO. Evidence C1-MAPPO.
- **Visual:** final feedback comparison is pending; available selection panel is `fig_regret.png`. Historical wall evidence has its own labeled supplementary plot; never label it feedback.

### B. Where verdict trust ends

- **VI-B-P1 — exact claim:** The frozen sign export's tested short clocks pass, then tested longer clocks fail; checkpoint binding is unresolved and overall fidelity qualification is a separate test. Evidence HORIZON-SIGN and HORIZON-UNIVERSAL; show `fig_direction.png`.
- **VI-B-P2 — exact claim:** Corrected timeline/native-lane accounting retains large paired-J error and negative long-clock correlation. Upstream/merge-region covariance dominates the error account but is not a queue-specific causal decomposition. Evidence ERR-REGION; show signed covariance bars, never a pie chart.
- **VI-B-P3 — exact claim:** Entry-path repair reduces source-wait MAE while total MAE worsens because source/road errors cancel. Keep left-boundary and one-tick clocks separate. Evidence ERR-ENTRY; show entry component contrast adjacent to, but separately labeled from, the covariance panel.
- **VI-B-P4 — exact claim:** Failed merge-response and smoothing attempts do not establish a repaired WM; final calibration is `[[PENDING: completed verdict / S-WMCALIB]]`. Keep a compact negative-result statement and the baseline/repair comparison; prior-attempt inventory belongs in the package rather than a long manuscript detour.

## VII. Discussion, threats, conclusion

- **VII-P1 — exact claim:** Existing results support bounded decision evidence, while failing effects/magnitudes and entry cancellation prevent unrestricted use as a controller judge. Recommend controller-, plant-, metric-, population- and horizon-specific validation before selection; recommendation is not measured deployment performance.
- **VII-P2 — exact claim:** Simulator/prior similarity, exposure, unequal baseline contracts, panel selection, missing uncertainty and sensing/plant mismatches limit external validity. Answer the most material objections from `THREATS.md` without implying real-road validation.
- **VII-P3 — exact claim:** Conclude with the factorization and the measured limits. Fill the zero-shot feedback superiority clause only if C1-CLOSED becomes SUPPORTED. Keep failed qualification findings even if a later calibration succeeds.

## References and drafting close-out

- Import `refs.bib` with IEEE bibliography style; preserve verified metadata and literal citation keys.
- `[[PENDING: owner prose, author/affiliation metadata, exact conference template and compiled pagination / owner]]`.
- Numeric findings must retain adjacent HTML source comments in the Markdown drafting source, even if comments are removed by final typesetting.
