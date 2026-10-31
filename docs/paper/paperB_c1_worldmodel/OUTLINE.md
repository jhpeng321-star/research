# Paper B — paragraph-by-paragraph drafting outline

## Title and governing claim

**Working title:** “WM-S: Physics-Informed Prediction under Unseen Traffic Controllers and the Limits of Its Verdicts.”

**Supported drafting headline:** physics-informed response prediction for unseen controllers, anchored by the recovered historical teacher-forced wall comparison and its paired uncertainty. Add the genuine-history PI diagnosis as a conditional extension, and the exploratory selection result as decision evidence. **Intended full-feedback extension:** `[[PENDING: complete eligible C1CLOSED result / S-C1CLOSED]]`. Do not put “closed-loop superiority” in the title/abstract as an established finding at this cut.

**Contribution bullets to preserve:** executable-controller/environment factorization; controller-exclusion evaluation separating recorded-action and feedback protocols; control-oriented reliability diagnosis linking regret, horizons, entrance accounting and regional error. Novelty is the combination and its measurements, not invention of physics-informed WMs, residual dynamics, or simulator admissibility.

## Format and page budget

Use IEEE conference two-column `IEEEtran`, numbered sections/equations/citations, readable figure text and IEEE bibliography style. Owner target is **six pages including references**, deadline **2026-11-15**. <!-- Source: owner S-PAPERB brief. -->

Editorial allocations, not measured pagination: title/abstract/introduction **0.85**; related work **0.45**; problem/protocol **0.55**; method **1.00**; experimental design **0.55**; results **1.55**; discussion/conclusion **0.40**; references **0.65**, total **6.00 pages**. Figure/table space is included within these allocations. <!-- Source: this package's editorial plan, derived to satisfy owner six-page constraint; not empirical data. -->

At typesetting, shorten prior-attempt history before dropping protocol caveats or negative results. Reserve architecture, feedback comparison/selection, and reliability panels; place expansive qualification history in repository artifacts.

## Abstract — exact sentence claims

- **A-P1:** New CAV controllers require forecasts of surrounding human responses under changed feedback actions, motivating controller/environment factorization. No novelty priority claim.
- **A-P2:** WM-S combines a versioned IDM/MOBIL-style core, history-conditioned settings, gated residual and entry/merge handling with executable controller code. Cite C1-FACTOR, WM-SETTINGS, WM-LANES.
- **A-P3:** Recovered historical wall response beats the frozen in-house black boxes under its teacher contract. Genuine-history WM2 PI advantage is a separately shifted conditional test, not a fair full-feedback win; use C1-WALL/C1-PI. `[[PENDING: exact eligible autonomous WM-versus-BB result / S-C1CLOSED]]`.
- **A-P4:** Report the bounded tested direction envelope and the distinction between source-error repair and unresolved road error; use HORIZON-SIGN and ERR-ENTRY with their source comments.
- **A-P5:** Completed calibration fails its transfer gate and state-dependent switching opportunity is not yet selector efficacy; these results motivate a declared controller/horizon/population envelope, without universal certification.

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
- **III-P3 — exact claim:** Zero-shot means before exposure to the evaluated controller's outcomes. Specify checkpoint/corpus/exposure, originals/entrants, lane ownership, observation channels, reference arm, and WORLD sampling unit. Historical old-C1/BB comparisons share seen-world training; newer diagnosis changes world coverage too. Name the completed corpus audit and incomplete initialization chain separately. `[[PENDING: remaining lineage and complete feedback protocol table / S-C1GAP + S-C1CLOSED]]`.

## IV. WM-S method

- **IV-P1 — exact claim:** Driver history/neighbors condition learned physical settings inside later hybrid IDM transitions, with defaults for limited observations; historical residual-only configurations are distinguished. Include the compact equation `a_hat=a_IDM(x_hat; eta_hat)+g(k) r_theta`. Evidence WM-SETTINGS.
- **IV-P2 — exact claim:** A step-aware gated correction models physics mismatch; gate and bounded-lane mechanisms are checkpoint-dependent. The current evidence does not isolate universal component gains. Evidence change log, WM-LANES and WM-REPAIR.
- **IV-P3 — exact claim:** Lane/merge event timing and open-road request/admission/exit bookkeeping connect individual predictions to road/source outcomes. Account for originals and entrants with explicit command/lane ownership. Evidence C1-FACTOR, ERR-ENTRY, corrected qualification report.
- **IV-P4 — exact claim:** Training is multi-step trajectory supervision of human dynamics; controller mechanisms remain executable code. Give final history, loss, curriculum and gate details only from the final eligible receipt. `[[PENDING: final model/training binding and ablations / S-C1CLOSED]]`.
- **Visual:** architecture schematic, with historical/later variants distinguished; figure specification in `FIGURES.md`.

## V. Experiments

- **V-P1 — exact claim:** Simulator highway-merge study uses existing evidence panels with different checkpoints and observation/execution contracts. Specify network/config identities from eligible receipts; describe synthetic simulator behavior without reading assignment files. `[[PENDING: compact experimental binding / S-C1GAP + S-C1CLOSED]]`.
- **V-P2 — exact claim:** Compare structured, in-house BB-traj, BB-id where supported, and simple physics/keep-speed references under declared training/input contracts. BB-id's learned CAV path is not equivalent to the teacher-matched BB-traj path; absent feedback capability is unavailable, not a loss.
- **V-P3 — exact claim:** Preserve source uncertainty and panel membership. Regret comparisons require matched controller/cost support; source repair is a paired diagnostic; regional share is covariance accounting. Original versus shifted history starts and teacher-forced versus reset-controller feedback retain separate rows. Mark survivor loss, exploratory component selection and unadjusted intervals; no synthesized paired intervals.
- **V-P4 — exact claim:** Tiny-pulse sensitivity, recorded scenario winners and full-rerun same-state winners answer different questions. Replay equality validates execution; pulse response tests a limited chaos explanation; headroom tests opportunity. None tests whether a WM selector captures that opportunity. C1-FLOOR/C1-SWITCH; main selector evaluation stays `[[PENDING: complete corrected-adapter campaign / S-C1SWITCH]]`.

## VI. Results

### A. Unseen-controller prediction and short-clock choice

- **VI-A-P1 — exact claim:** The recovered historical unseen-wall report supports structured HDV-response error superiority over BB-traj with paired world-bootstrap uncertainty, under recorded CAV motion; BB-id is separately qualified. Include C1-WALL point values and paired contrasts from `CLAIMS.md`, with source comments. `wall_pilot` is sensitivity, not an independent pooled replication.
- **VI-A-P2 — exact claim:** The PI diagnosis finds genuine regression at original starts, a major training-coverage confound, and a favorable WM2 full-history teacher-forced contrast on separately shifted starts. Use exact C1-PI/C1-COVERAGE contrasts; preserve adverse original and reset-feedback results. Do not combine the original-start later-clock advantage with shifted-start history gains into one curve.
- **VI-A-P2b — exact claim:** `[[PENDING: complete autonomous unseen PI/wall or other excluded-controller comparison / S-C1CLOSED]]`. The registered common-plant campaign is running; partial tasks, framework canaries and reset-controller stress tests cannot fill this paragraph.
- **VI-A-P3 — exact claim:** Strict-history E1 offers exploratory offline controller-choice evidence with measured oracle regret and constant-selector references. Report marginal regrets plus the separately exported paired WM-minus-BB advantage and difference versus always-MAPPO; numerical pairing is now available, while eligibility/plant/cost bindings remain pending. Evidence C1-MAPPO; do not infer the paired interval from marginal bars.
- **VI-A-P4 — exact claim:** Recorded library diversity and fresh same-state headroom justify investigating switching, with small best-to-second margins; they do not show that WM-guided switching works. The stochastic-key anonymity repair caused a mechanical retention stop; the authorized corrected continuation remains `[[PENDING: full efficacy results / S-C1SWITCH]]`. Include only a compact opportunity sentence if page pressure requires. Evidence C1-SWITCH.
- **Visual:** final feedback comparison is pending; retain `fig_regret.png`. Evening diagnosis and headroom artwork are specified in `FIGURES.md`; historical wall artwork remains teacher-forced.

### B. Where verdict trust ends

- **VI-B-P1 — exact claim:** The frozen sign export's tested short clocks pass, then tested longer clocks fail; checkpoint binding is unresolved and overall fidelity qualification is a separate test. Evidence HORIZON-SIGN and HORIZON-UNIVERSAL; show `fig_direction.png`.
- **VI-B-P2 — exact claim:** Corrected timeline/native-lane accounting retains large paired-J error and negative long-clock correlation. Upstream/merge-region covariance dominates the error account but is not a queue-specific causal decomposition. Evidence ERR-REGION; show signed covariance bars, never a pie chart.
- **VI-B-P3 — exact claim:** Entry-path repair reduces source-wait MAE while total MAE worsens because source/road errors cancel. Keep left-boundary and one-tick clocks separate. Evidence ERR-ENTRY; show entry component contrast adjacent to, but separately labeled from, the covariance panel.
- **VI-B-P4 — exact claim:** Completed grouped-CV calibration and independent transfer do not qualify long-run policy credit. Raw WM2 loses to arm-mean in grouped CV; transfer cell orderings differ and the selected calibrator fails its frozen gate. This is stronger than the afternoon missing-result statement. Evidence WM-CALIB; retain split/reference/selection labels.
- **VI-B-P5 — exact claim:** The tiny native-pulse experiment supplies no measured short-window chaos explanation: endpoint medians remain near zero, with finite aggregate sensitivity and later survivor loss. It is not an irreducible RNG bound. Evidence C1-FLOOR.
- **VI-B-P6 — exact claim:** Failed merge-response/smoothing and the killed queue-service prototype leave the long tail unresolved. The data-only pilot's candidate gate and short-WM-plus-value comparison remain `[[PENDING: complete respective gates / S-WMFIXMERGE + S-WMVALUE]]`. Keep one compact negative-result sentence; full attempt history and companion backup belong in the package.

## VII. Discussion, threats, conclusion

- **VII-P1 — exact claim:** Existing results support bounded decision evidence, while failing effects/magnitudes and entry cancellation prevent unrestricted use as a controller judge. Recommend controller-, plant-, metric-, population- and horizon-specific validation before selection; recommendation is not measured deployment performance.
- **VII-P2 — exact claim:** Simulator/prior similarity, coverage-confounded checkpoint comparisons, unequal baseline contracts, shifted-history starts, controller-memory resets, survivor loss and sensing/plant mismatches limit external validity. Software restore artifacts and detector-ID stochastic keys require end-to-end execution audits; repaired mechanics alone are not scientific wins. Use `THREATS.md`.
- **VII-P3 — exact claim:** Conclude with the factorization and the measured limits. Fill the zero-shot feedback superiority clause only if C1-CLOSED becomes SUPPORTED. Keep failed qualification findings even if a later calibration succeeds.

## References and drafting close-out

- Import `refs.bib` with IEEE bibliography style; preserve verified metadata and literal citation keys.
- `[[PENDING: owner prose, author/affiliation metadata, exact conference template and compiled pagination / owner]]`.
- Numeric findings must retain adjacent HTML source comments in the Markdown drafting source, even if comments are removed by final typesetting.
