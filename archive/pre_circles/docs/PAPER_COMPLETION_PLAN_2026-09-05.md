# A bounded path from the current evidence to the paper

> **Superseded as the primary route by the owner's clarification:** standalone
> trained damping is required. Read [the restored training-to-paper route](TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md).
> This document's decision to make learned damping optional does not satisfy
> that objective. Its evidence audit remains useful; its 1208-rollout budget
> is a wall-only alternative and is not the full trained-damping project budget.

Date: 2026-09-05. Status: **reviewed evidence and proposed replacement work plan; not an executed experiment contract.**

The owner stopped Stage 2 and requested an evaluation and a direct paper plan. No simulation, training, reserved-seed inspection, or controller modification was performed for this plan. At the final process check, no matching research-dense Python/SUMO process was found. The stopped Stage 2 directory contains 32 completed seed files. Current checked-out HEAD is `0f7dd47`; the earlier handoff's HEAD and launch status are stale.

This document explicitly proposes narrowing the September 5 handoff. It does not silently change its frozen treatment or authorize restarting generation. The owner's online-information requirement is controlling: **the RSU receives information acquired by CAVs and makes the decision; unrestricted SUMO traffic state is not an online input.**

## 1. Decision: build one selective-control paper

Working title: **Selective CAV Metering from Cooperative Vehicle Reports in an IDM Highway Merge**.

Question: Can an RSU use reports from participating CAVs to identify when a sustained upstream intervention is worth its network-wide traffic cost?

Three contributions to establish:

1. Measure the intervention's benefit/cost boundary and discharge mechanism under the tested mixed-traffic conditions.
2. Learn an implementable report-to-action rule, including its information limitations, and compare it with a fairly fitted simple rule.
3. Evaluate the frozen complete policy on untouched scenarios, accounting for ordinary IDM, always-on intervention, conventional control, unfinished traffic, and comfort costs.

The mandatory paper does **not** need residual RL, a successful decentralized damping method, a learned collapse-onset detector, a safety classifier, or a learned release rule. The existing damping controller is an experimental base arm, not an established beneficial contribution. A gain over that arm alone is insufficient: ordinary IDM must be compared on matching demand realizations.

Use "sustained CAV metering" to describe the present actuator accurately. Its control band is fixed in space, while CAV membership changes as vehicles pass through. It does not construct a persistent, synchronized, all-lane moving formation.

## 2. What existing results establish

### Retain as development evidence, with version labels

- The sustained treatment has a loading-dependent tradeoff. In Stage 1b-prep-5, mean W-minus-D delay was +2.57 s/vehicle at 2200, +1.93 at 2400, -1.50 at 2600, and -5.99 at 2700 veh/h, with 30 seeds per cell. High-loading health-collapse incidence remained 30/30 in both arms. This supports investigating discharge improvement, not claiming high-loading collapse prevention. These are results for the implemented composite, whose command-restoration discrepancy is described below.
- Stage 1c completed 120 seeds and 600 treatment/control branches, with 12 exact D repeats. Unlike Stage 2, its runner correctly uses a 2400-step horizon. The saved terminal metadata does not establish strict natural clearance; approximately 1915–1917-step endings are a consistency observation, not proof that all requested trips were served.
- Stage 1c-2 indicates that hold duration matters: at fixed engagement near t=250, the observed best tested holds were 300 s at 2400, 650 s at 2600, and 450 s at 2700. The 2400 benefit remains uncertain. These cell optima are development observations, not a learned policy and not a universal duration map.
- A fair loading threshold already captured most of the measured privileged-information value. The multifeature advantage was +0.160 s/vehicle with CI90 [-0.014,+0.328]. That is no clear demonstration that extra model complexity helps.
- The frozen analytical damping controller is not an established improvement. At 2200, D-minus-native mean delay was +2.08 s/vehicle, median +1.891, with health-collapse counts 24 to 29. Do not generalize this to all IDM-compatible damping methods.
- The old R1/R1b releases never fired by rule in the reported 210 cases. This rejects those rules in their tested setting, not the possibility of every state-based release controller.

### Do not carry these forward as established claims

- **"Onset is solved."** The reported AUC 0.944 uses a sampling/label scheme that includes already-onset cases as negatives. A proper future-onset study needs causal sampling among cases still at risk. It is unnecessary for the proposed first paper.
- **"CAV observations cannot support learning."** Earlier near-chance studies concerned different local damping or compact-wall actions and targets. Conversely, current full-state loading predictability does not demonstrate CAV-report sufficiency. That question remains open.
- **"The overlay preserves damping everywhere else."** During HOLD, touched CAVs leaving the band receive a damping command followed by `setSpeed(-1)` repeatedly. A mock reproduced the ordering. The outcome effect is unmeasured; corrected labels must be generated under a new version.
- **"Stage 2 contains full-profile training outcomes."** Its runner imports 1080 steps despite the 2400-second contract and demand lasting through 1800. The 32 completed files contain 128 branches ending at 1080. Passing 96 prefix checks does not repair this mismatch. These rows cannot become full-profile labels by renaming them.
- **"The existing health is a calibrated traffic-risk probability."** It is a heuristic, computed from full merge-edge state, with legacy scaling and thresholds. Its density proxy divides by lane count twice; call frequency also affects its state. Retain its versioned historical result as a diagnostic, not the proposed policy input or principal outcome.

For paper statistics, recompute confidence intervals for the stated statistic. Some historical prose pairs a mean with an interval for the median; the old fitter's quantity named R-squared is squared correlation, not predictive R-squared.

## 3. Proposed operational structure

**Collect CAV reports -> one scheduled decision -> availability check -> choose no intervention or a hold -> execute locally -> taper -> return to the base controller.**

### When

Use one prespecified decision at t=250 s after scenario start, with the preceding 60 s of reports. This is supported as a useful development decision epoch by the existing fixed-time experiments. The RSU owns the clock; no simulator health trigger supplies the opportunity. The clock schedules the decision but is not a regression feature.

This is an intentional scope reduction: the paper evaluates **selection at a specified opportunity**, not arbitrary-time collapse forecasting or a continuously adaptive deployment. Explain that limitation in the title/abstract claims and methods. An event-triggered or repeatedly reconsidered policy would require a different matched treatment dataset and is follow-up work.

### Can we issue the intervention?

Define a simple executable availability predicate, not a learned safety guarantee. Proposed nominal contract: at least one fresh report, no older than 2 s, from an actuator-ready CAV approaching the merge within the declared 800 m upstream reporting region. If it fails, choose the base controller. A command is executed only by a reporting, eligible CAV actually in the band, subject to the existing local vehicle safety constraints.

This is a proposed new predicate and must be versioned and tested. It does not require simultaneous occupancy of all lanes or guarantee useful delivered dose. Empty-band intervals can occur after the regime starts. Log command start, band membership, participating lane coverage, and delivered CAV-seconds separately from scheduled hold time. Do not retrospectively remove low-dose cases from policy evaluation.

### Whether and how long

For report history x, estimate expected differences from the corrected base arm:

`delta_H(x) = E[J(D + intervention of duration H) - J(D) | x]`

for H in {300,450,650} s. Include the no-intervention action with value zero. Choose the smallest predicted value and engage only when it is below a margin `-m`; start with m=0 and permit a small prespecified development-only margin grid if needed. The complete choice, including m, is frozen before validation.

This predicts expected intervention value; it is not a prediction of the true individual counterfactual and not a probability of safety. A benefit probability is optional diagnostics and cannot replace the magnitude of gains/losses.

The simple comparator uses the same admissible reports, opportunity, availability rule, hold choices, and tuning access. Its scalar feature, threshold, and a small duration map are selected inside development folds. A fixed 0.15 health threshold and the historical exact inflow/count thresholds are not imported into the new policy.

### Execute and release

Keep the existing physical settings: mainline band 200–400 m before the merge, target 10 m/s, original tau plus 3 s, command deceleration clip 0.3 m/s², and a 5 s taper. Correct command restoration so that a vehicle leaving the band immediately receives the intended base command and restored tau. Freeze regime-duration and taper semantics explicitly. No new strength, geometry, recurrent engagement, or release-rule search is part of this plan.

The command clip does not bound the braking of following HDVs. Report their measured exposure separately.

## 4. The smallest useful CAV observation contract

Use a declared kinematic sensor abstraction rather than implementing a complete perception system. The primary proposed sensing assumption is CAV self state plus locally measured nearest same-lane front/rear gap and relative speed within 100 m. Restrict detections to the nearest visible vehicle in each direction; do not query through intervening vehicles. The exact capabilities are assumptions of this simulation study, not verified hardware performance.

Each CAV sends a timestamped 1 Hz report with its own location/lane, speed, acceleration, command mode, and valid local measurements. Start with a one-control-step message delay. Retain source and receipt times, validity, range censoring, and missingness. Do not silently turn an undetected object into an empty lane.

The RSU summarizes **CAV experiences**, rather than pretending to reconstruct an exact object census. Aggregate distinct reporting CAVs and their measurements within fixed upstream, merge-approach, and downstream map regions over a 60 s history. A compact frozen feature list should include regional reporting count, mean speed, slow fraction, speed trend, front/rear gap statistics, relative closing speed, command activity, and report age/missingness. Cap the list at roughly 30 scalar inputs; publish its exact schema before simulation.

Because the units are CAV reports, two CAVs observing the same HDV are two correlated experiences, not two distinct vehicles in a density count. Do not label such sums total flow or density. This avoids making perfect cross-CAV object association an undeclared input or a new research prerequisite.

Exclude exact all-vehicle counts, true penetration corrections, hidden demand, private HDV parameters/intention, future routes, seed identity, future onset, outcome fields, and exact simulator shadows from online features and missing-value fill. Static map geometry and CAV self-identification are admissible.

Save the raw causal reports and receipt histories once. The same development rollouts can then support, without more SUMO runs:

- Self-state-only versus self-state plus local sensing.
- Short versus 60 s history.
- Omission of coverage/age features.
- Range reduction from 100 to 50 m, and imposed report loss/delay, for observation sensitivity.
- A full-state reference, labeled privileged and never deployed.

Offline perturbations assess selection sensitivity at the shared predecision prefix. They do not prove robustness of postdecision communication and command execution; the prospective stress block addresses that separately.

## 5. Fix the labels and integrity once

Before production generation, implement and test these together:

1. Use the declared 2400 s horizon. Early termination is allowed only after demand has ended and requested-but-uninserted, active, and pending traffic are all empty. Persist explicit termination reason and counts.
2. Check effective final commands on in-band, exiting, reentering, and tapering CAVs. Calling damping first is insufficient. Assert parameter restoration and no lingering wall authority.
3. Preserve exact D prefixes for D-plus-wall branches and exact D repeats. Pair native/VSL comparators by exogenous scenario, routes, and CAV assignment; their microscopic prefixes need not equal D.
4. Check the observation boundary: changing hidden state without changing CAV reports must not change the RSU action. Test timestamp cutoffs, absent regions, no eligible reporters, and stale commands.
5. Use a common population denominator: all vehicles requested by the demand schedule, not only completed merge exits. Include source waiting, in-network time, unfinished trips, ramp/mainline costs, and completion counts.

Proposed primary J is excess time in the system, including source waiting, accrued through the common 2400 s horizon, divided by the common requested population. Subtract the same route-specific free-flow reference in every arm. Operationally, integrate the number of requested vehicles not yet at their destination, then subtract that common reference. This yields comparable paired differences even if an arm leaves unfinished traffic. Call it a finite-horizon measure when clearance is incomplete; do not claim lifetime travel delay in that case. Report conventional completed-trip time loss additionally for continuity with past results.

Use full simulator outcomes for offline supervision and evaluation. That is compatible with CAV-only inference. No paired counterfactual simulation runs online.

## 6. One bounded experiment program

These are planning caps, not a promise that a fixed sample size can demonstrate an arbitrarily small benefit. A branch means one complete simulation rollout. Deduplicate identical selected policies and reuse unchanged D branches. Interrupted invalid branches are not scientific evidence.

### A. Engineering reconciliation: at most 20 new branches plus 8 repeats

Use four already exposed development scenarios, chosen by predefined loading strata, with N (native IDM), D, and all three corrected holds: 4 x 5 = 20. Include eight exact repeats across the program. Unit/contract checks precede SUMO. These runs verify treatment, sensing, timestamps, outcomes, and throughput; four seeds do not establish efficacy or feature sufficiency.

If a check fails, fix the implementation and repeat only affected engineering checks. Do not create another scientific treatment gate or tune physical settings from these results.

### B. A reusable development library: 80 seeds, 400 branches

Proposed cohort: registered development seeds 8001–8080, with their deterministic hidden demand assignments retained, at p20. Their earlier partial exposure is acceptable because this entire cohort is development; exclude it permanently from prospective claims. Verify assignment manifests before reuse. All updated artifacts receive a new treatment/observer version and directory.

Run N, D, H300, H450, H650 once per seed at the fixed opportunity. Retain all reports and per-vehicle source/outcome accounting. Every later threshold, small-model, duration, or feature ablation operates on this shared library without rerunning traffic.

Fit only an interpretable scalar rule, a regularized linear action-value model, and at most one shallow boosted-tree challenger with a tiny fixed tuning grid. All preprocessing, feature/threshold choices, margins, model selection, and duration choices occur inside seed-grouped nested folds. Do not split snapshots from one seed across train and test. No deep sequence model or RL.

The hindsight best action is an estimate of available headroom under the sampled potential outcomes, not a realizable policy. The central metric is cross-fitted policy cost with paired seed intervals, not AUC. Also show performance within loading bands and report coverage so that recognizing different regimes is not advertised as predicting seed-specific variation.

### C. Conventional comparator tuning: at most 60 branches

On 20 preselected development seeds, tune one road-level VSL comparator over 10/15/20 m/s, active from the same decision epoch for 450 s, then restore the original speed limit. Implement a schedule around the existing VSL primitive; it currently acts from its first step. It acts on the upstream edge and affects all vehicles, so disclose its greater actuator reach and different spatial extent. It uses no privileged dynamic observation. It is a modest conventional comparator, not an assertion that every VSL or ramp-metering method is beaten.

Do not expand this into a second controller research project. Select its speed using development costs before any validation outcomes are opened.

### D. Prospective validation: 40 untouched seeds, at most 240 branches

After freezing both candidate gates and analysis, compare six arms: N, D, the development-selected best constant hold, the fitted simple report rule, the selected multifeature report policy, and tuned VSL. Use the same scenario realization for each arm. Register a fresh block after an exclusion audit; do not assume old reserved cohorts match the new experiment or open them for this plan.

This block verifies live report-to-action execution and estimates practical policy gains. Do not tune thresholds or refit to its outcomes. If the multifeature policy does not convincingly beat the rule, select the rule for the paper; a fitted threshold is still a data-derived policy. Keep both outcomes visible.

If both report policies lack useful value, finish with the scoped treatment/observability conclusion rather than initiating another feature or wall search. Any architecture revision after this point is a separately budgeted new study, not an automatic continuation.

### E. Locked final evaluation: 60 seeds, at most 360 branches

After the final bundle is hashed and the existing final-run authorization conditions are met, use reserved seeds 5001–5060 once under the newly frozen scenario contract, provided the exposure audit confirms they remain untouched. Keep 4601–4660 sealed. Evaluate the same six comparators, deduplicating identical policies.

Do not retune after reading the final block. This is the confirmation dataset for the headline result, not another development gate. Validation and final results are reported separately.

### F. One stress test: 30 fresh seeds, at most 120 branches

At p20, keep the physical scenario family but degrade report/command communication according to a predeclared setting: 20% independent packet loss and three-control-step delivery delay, with explicit stale-message and local-command expiry semantics. Compare N, D, and the two frozen report gates. This tests the required communication assumption in closed loop. Do not add automatic p30, new-network, new-driver, and event-timing campaigns.

Higher penetration, a second corridor, altered driver models, and arbitrary decision times remain outside the primary claim. They can strengthen a later version but should not all be prerequisites to finishing this paper.

**Maximum planned total: 28 + 400 + 60 + 240 + 360 + 120 = 1208 branches.** This is the budget for the entire corrected study, not another preliminary gate before a larger generation. It replaces the old 360-seed, four-arm Stage 2 expansion (up to 1440 branches before repeats and subsequent evaluation).

Measured runtime context: across 120 existing Stage 1c seed files, median recorded elapsed time is 1913 s per five-arm seed and the 90th percentile is 2516 s. Dividing by five gives rough equivalents of 6.4 and 8.4 minutes per branch; this includes the original runner/hardware conditions and is not a benchmark of the corrected code. At unchanged throughput with eight workers, 1208 branches corresponds arithmetically to roughly 16–21 hours of simulation wall time, plus overhead. Engineering, fitting, writing, interference from other work, and failures are additional. Re-estimate from phase A before scheduling; do not promise an overnight completion.

## 7. Exact decisions that end the study

Proposed new analysis convention: paired seed-bootstrap 95% intervals for mean differences, clearly distinguished from historical CI90 results. Freeze multiplicity handling for primary comparisons before validation; report simultaneous intervals or a prespecified hierarchical test order rather than choosing the favorable comparison afterward.

- **Practical efficacy:** seek improvement over N and the best constant intervention, not D alone. A suggested planning target is at least 1 s/requested vehicle in point-estimated mean saving with a paired interval excluding zero. This 1 s margin is a proposed scientific relevance target, not a measured threshold. Use development paired variance to forecast achievable interval width before validation. If the budget cannot resolve that margin, narrow the claimed effect rather than silently adding seeds.
- **Need for learning complexity:** claim extra value from the multifeature policy only if it improves over the equally tuned report-based rule prospectively. Otherwise ship the rule and report the comparison honestly. Do not claim individualized counterfactual prediction from a positive aggregate policy result.
- **Duration claim:** the main action space may include duration selection. An assertion that adaptation itself improves on a learned fixed-duration gate requires that specific held-out comparator. It is not established by hindsight best-hold curves. The core budget does not promise a separate adaptive-duration superiority claim.
- **Cost and safety reporting:** report collisions, teleports, TTC exposure, braking magnitude/duration by location, source queues, ramp/mainline delay, and unfinished traffic with uncertainty. Investigate new collisions or teleports before calling a policy acceptable; do not reinterpret additional deceleration-cap counts as a proof of safety or suppress them as irrelevant. Zero observed collisions is not a general safety guarantee.
- **If the desired benefit does not appear:** complete a truthful mechanism/limitation manuscript or reconsider submission scope. Do not generate another large cohort merely because a desired positive story did not emerge. No plan can guarantee IEEE IV acceptance or positive outcomes.

## 8. Build the manuscript while the fixed study runs

Write now: introduction around selective upstream intervention; corridor/demand/IDM scope; exact actuator; CAV report assumptions; causal paired supervision; evaluation population and cost accounting. Reserve positive claims in the abstract until final evaluation.

Five figures with distinct jobs:

1. Corridor and information/control diagram: where each CAV measures, what reaches the RSU, where commands act, and one-shot decision timing.
2. Corrected intervention effect across loading and duration, with uncertainty and native-IDM comparison. Historical results can motivate this, but are not pooled across treatment versions.
3. Report availability and policy decisions: observation ablation, threshold/learned comparison, coverage, and engagement rate.
4. Final paired policy comparison: mean network cost, completion/source backlog, and mainline/ramp allocation for all six arms.
5. Mechanism and costs: space-time speed/discharge and braking for prespecified representative cases, plus aggregate communication-stress results. Pick examples by a declared median-case rule, not by visual success.

Appendix/reproducibility: exact report schema; model/threshold grids; duration and command semantics; seed exclusions; code/scenario/config/model hashes; terminal accounting; repeat checks; complete per-seed outcomes. Legacy health formula and failed release/observation studies belong in diagnostics or supporting discussion, not five separate headline contributions.

Suggested work order is engineering and methods writing first, development library and offline fitting second, frozen validation and figure generation third, final confirmation/stress and manuscript completion last. No new simulation is needed to draft methods, repair the analysis definitions, or construct figure scripts with clearly marked development inputs.

## 9. Immediate next work package

Implement a new versioned runner/observer and the deterministic tests in section 5; write the executable contract, exact feature schema, outcome formula, seed-role manifest, model grid, and comparison order. Prepare the four-scenario engineering batch and the draft methods. Review those concrete files before proceeding to the experiment phases under the user's execution instructions. Do not resume `run_stage2_generation` unchanged, train RL, or launch the old next-agent prompt.

## Evidence locations

- [Active handoff as originally written](HANDOFF_2026-09-05.md) and [old execution prompt](STAGE2_NEXT_AGENT_PROMPT.md).
- [Stage 1b development results](SUSTAINED_WALL_V1_STAGE1B_PREP5_RESULTS.md), [Stage 1c release results](SUSTAINED_WALL_V1_STAGE1C_RELEASE_RESULTS.md), and [Stage 1c-2 results](STAGE1C2_RESULTS.md).
- [CAV-only audit](../output/cav-reported-rsu-observation-assessment-2026-09-05.md), [escalation audit](../output/escalation-structure-and-open-questions-2026-09-05.md), and [health audit](../output/health-feature-explained-2026-09-05.md).
- `experiments/idm/run_stage2_generation.py`, `stage2_generation_contract.py`, `run_stage1c_release.py`, `rsu_feature_capture.py`, `fit_stage1c2_partC.py`; `policies/damping_plus_wall.py`; `policies/inflow_control.py`; `env/core.py`; `env/features.py`.
- Runtime/inventory reads: `runs/probe/stage2_generation_p20_s8001_8240/seeds/` and the four `runs/probe/stage1c_release_m*_p20_s*/seeds/` directories. Reserved validation/final artifacts were not opened.
