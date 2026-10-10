# IDM v5.4 shadow-observation v2 development protocol

**Status:** frozen before replay  
**Scientific role:** one bounded, development-only observation repair  
**Treatment status:** unchanged frozen v5.4 P30 compact moving wall  
**Control status:** unchanged frozen decentralized analytical IDM damping  
**Contract:** `idm_v5_4_shadow_observation_v2/1`

## 1. Question and stop boundary

This phase asks whether information available at the declared decision can
separate three questions that the v1 observation contract did not solve:

1. can the proposed wall receive at least 80% of its requested dose;
2. will a dose-qualified wall remain admissibly safe; and
3. among safe, dose-qualified walls, will it provide frozen joint safety or
   discharge benefit over identical decentralized IDM damping?

The existing development evidence remains unchanged: 120 seeds, 480 declared
windows, 199 Gate-F admissions, 179 dose-qualified cases in 101 seeds, 106 safe
and 73 unsafe cases, and 53 jointly beneficial versus 126 nonbeneficial cases.
The current selected shallow benefit AUC is about 0.642; safety is weaker. Its
ten exploratory commands span nine seeds and achieve 80% safety and 50% joint
benefit, so the original confirmatory policy gate failed. Health and health
change remain excluded as primary variables.

This is the only authorized v2 feature-repair phase. If its frozen improvement
and policy gates fail, feature searching stops. No threshold, label, treatment,
control arm, or safety definition may be changed in response to the result.

## 2. Frozen population, sources, and partitions

- Development seeds: exactly 4401--4520.
- Decisions: controller steps 620, 660, 680, and 700.
- Scenario: IDM, mainline demand 2700 veh/h, ramp demand 400 veh/h, CAV
  penetration 30%, one-second controller interval and 0.5-second SUMO step.
- Existing labels and paired effects:
  `runs/eval/idm_wall_v5_4_predictor_development_p30_s4401_4520`.
- Treatment-blind Gate-F source:
  `runs/probe/idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520`.
- Baseline diagnosis:
  `runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/report.json`.

The contract pins the SHA-256 hashes of the label manifest, label summary,
feasibility analysis, and v1 diagnosis. Labels are joined by exact `case_id`
only after observation capture completes. The extractor never receives a label
row.

Seeds 4601--4660 belong to the already-frozen v1 prospective validation and
must not be read. Seeds 5001--5060 remain sealed final-paper seeds. If v2 passes,
a new untouched partition will be proposed in a review-only protocol and will
not be opened without approval.

## 3. Frozen arms and exact 10-second shadow

The planning state is the common predecision state. The damping command is
computed first. No wall override has yet been applied.

- Arm A: frozen analytical IDM damping only.
- Arm B: the identical damping controller plus the frozen v5.4 moving-wall
  overlay, including its real assembly, adaptive target, rate limit, front and
  rear safety projection, compactness, same-lane reserve/handoff, tau ramp, and
  emergency zero-authority handoff rules.

The two logical clones are implemented by deterministic replay from time zero,
not TraCI `saveState`/`loadState`. SUMO does not guarantee that save/load
captures every car-following, lane-changing, flow, and RNG internal. Each twin
therefore uses the same scenario, seed, routes, CAV selection and action prefix,
and must produce the exact same decision-state digest. A digest mismatch
invalidates the shadow and the seed artifact. Native IDM and LC2013 lane changes
remain enabled and identical to the real experiment. The branches diverge only
after the wall command.

Each shadow lasts exactly ten controller seconds after the decision. Reports
are emitted at 1, 2, 5, and 10 seconds. This is a prospective planning feature,
not the stored realized treatment branch, and it does not replace or relabel the
existing 20-second outcome experiment.

All four decision-specific twins retain the same 710-controller-step episode
horizon as the shared control trace, even though each twin stops recording ten
seconds after its decision. This prevents the episode-progress observation from
changing merely because a decision occurs at step 620 rather than 700.

The shadow records:

- dose/authority: assembly state and time, speed convergence time, member span,
  minimum occupied-lane coverage, command saturation and projection survival,
  available authority, same-lane reserves, handoffs, and projected ten-second
  override fraction;
- safety: minimum front/rear gaps and TTC, per-lane TTC/gap quantiles, follower
  compression, required follower deceleration, affected followers, braking and
  low-TTC exposure, and boundary/release margin;
- traffic response: arm-specific and paired upstream/ramp flow, downstream
  merge discharge, speed, density, queue proxy, braking, and wall-induced
  inflow reduction.

Missing vehicles are represented by value plus availability mask. A vehicle
that exits, enters, or changes lane remains part of the exact SUMO trajectory;
lane-change/entry/exit events are explicit. No carry-forward imputation crosses
an entry or exit. An undefined TTC is represented as a 120-second right-censored
value plus a censoring mask. Quantiles require at least two observations;
otherwise their availability mask is false.

## 4. Leakage-safe decision record

At each of all 480 decisions, before any treatment command, capture a deep,
canonical copy of:

- road, lane, lane index, lane position, distance to merge, speed,
  acceleration, length, type and route index for influence-region vehicles;
- leader/follower linkage, net gaps and relative speed;
- available IDM/vehicle parameters: tau, minGap, accel, decel, emergency and
  apparent decel, speed factor, allowed/max speed, action-step length, speed
  mode and lane-change mode;
- active/controlled CAV membership;
- the just-issued frozen damping command and projection diagnostics for every
  controlled CAV;
- Gate-F proposed wall members and ranked same-lane reserve CAVs, distance to
  the release boundary, formation/authority/safety preflight;
- causal upstream, ramp and downstream detector histories.

Internal vehicle IDs are permitted only for trajectory linkage, deterministic
sorting, member matching, and replay auditing. They are stripped before model
construction. Seed is metadata/group identity only and never a learned input.

The capture phase marker is
`after_identical_damping_before_wall_override`. Extra detector calls that mutate
the real inflow tracker are prohibited.

The following are prohibited model inputs: actual dose/duration, completion,
release or abort reason, realized outcomes, safety/oracle labels,
counterfactual effects, postdecision samples outside the newly generated
ten-second shadow, seed identity, and absolute vehicle identity. Mutation tests
must prove that changing every stored dose, outcome, oracle and postdecision
field cannot change the predecision feature bytes.

## 5. Causal history and trajectory representation

Capture the last 60 seconds at a fixed one-second interval so a conditional
60-second sensitivity does not require another SUMO replay. The primary screen
uses only the last 20 seconds. Every sample is at or before the decision.

Vehicles are linked internally through the window, then exported into anonymous
region/lane/relative-order slots with a missingness mask. Raw IDs and seed never
enter a tensor. The record preserves position, lane, speed, acceleration,
leader/follower gap, relative speed, type class and lane-change/entry/exit
events, with ramp, mainline premerge and downstream populations separated.

Three temporal representations are compared:

- A: the exact current engineered temporal-summary baseline on the same folds;
- B: compact physical trajectory features; and
- C: one small regularized temporal-convolution encoder using the anonymous,
  masked 20-second tensor.

The encoder has no attention, at most two causal convolution layers, at most 16
channels per layer, global masked pooling and one regularized logistic head.
Architecture and regularization selection occur inside inner grouped folds. A
60-second tensor is evaluated only if the 20-second physical/sequence
representation fails the observation gate; it is a declared sensitivity, not
an open feature search. If C does not outperform B, B is retained.

For honest comparison, representation A is reported twice: the exact frozen v1
feature set (including health terms only because the historical baseline used
them) and the same engineered baseline with health/health-change removed. No v2
physical, trajectory, shadow or combined candidate may use health or health
change. The 0.03 improvement gate is measured against the exact frozen v1
benchmark, not only against the health-excluded variant.

Physical trajectory features are grouped as follows:

1. Wave dynamics: shockwave direction/speed, amplitude and its growth/decay,
   front position, section-to-section braking propagation, compression
   persistence, and recovery-versus-growth state.
2. Follower chains: compression and acceleration propagation, minimum
   chain gap/TTC, consecutive braking length, string amplification and
   leader-to-follower braking lag.
3. Platoon/headway: size, mean/variance/CV/quantiles, variance evolution,
   successive-speed correlation, platoon length/density change and
   desired-speed heterogeneity proxies.
4. Merge interaction: ramp arrival relative to mainline gaps, accepted/rejected
   gap evolution where observable, ramp/mainline relative speed, lane-change
   and forced-yield proxies, merge-induced disturbance and cumulative mainline
   compression.
5. Lane temporal state: first destabilizing lane, inter-lane speed/flow/
   occupancy imbalance, CAV coverage change and movement into/out of feasible
   wall positions.

Historical trajectory answers “how did this state develop?” The exact shadow
answers “what is this proposed action expected to do?” Their held-out
combination is tested for complementary value.

## 6. Compact physical feature blocks

The model screen does not indiscriminately expose every scalar.

- Sectional transitional instability combines short temporal response with
  upstream/downstream spatial speed and occupancy imbalance. Its dynamic
  instability score and all components remain separately ablatable.
- Merge pressure/breakdown state includes mainline/ramp arrivals, downstream
  discharge, net and cumulative 20/60-second excess demand, ramp state, merge
  speed/spacing, headway distribution, platoon heterogeneity and downstream
  storage. It is probabilistic; no fixed critical density is assumed.
- Proposed-wall response contains only the ten-second twin differences in dose,
  authority, assembly, follower compression, TTC/deceleration, coverage,
  reserves and inflow/discharge response.

The primary v2 candidate is the smallest block combination that survives
held-out leave-one-block-out ablation. The complete union cannot win merely by
having more features.

This design follows merge-breakdown variables in [Han and Ahn
(2018)](https://doi.org/10.1016/j.trb.2017.11.007), section-based temporal and
spatial instability in [Tak and Yeo
(2026)](https://doi.org/10.1016/j.trc.2025.105483), spatial/temporal/ramp
measurements in [Zhao and Dong-O'Brien
(2023)](https://doi.org/10.3390/a16060298), and trajectory-derived conflict
indicators in [Islam and Abdel-Aty
(2023)](https://doi.org/10.1016/j.amar.2023.100275). Their published accuracy
claims are not transferred to this simulation or evaluation.

## 7. Targets and secondary effects

- D: duration-dose success (realized duration fraction at least 0.80), on all
  Gate-F admissions.
- S: unchanged frozen safety admissibility, on dose-qualified admissions.
- B: unchanged joint benefit, on safe dose-qualified admissions. Delay remains
  a noninferiority guardrail; queue area is not a selection criterion.

Secondary small paired-effect regressors separately predict hard-brake onset,
hard-brake exposure, low-TTC exposure, merge discharge and delay. They test
observability and cannot redefine B.

## 8. Grouped model screen

All decisions from a seed remain in one fold. Use the same five outer
stratified seed-grouped folds as the v1 diagnosis (`random_state=20260831`) and
four inner grouped folds. Recompute the current baseline in those exact folds.
Compare regularized logistic regression, additive-spline logistic regression,
and shallow gradient boosting. The sequence model is separately bounded as in
section 5. The secondary paired-effect model is small and regularized.

Report ROC-AUC, average precision, Brier score, log loss, recall, selected
safety, selected joint benefit, selected dose, commands, independent command
seeds, enrichment over feasible-always-wall, clustered bootstrap intervals,
leave-one-seed-out sensitivity, shuffled-label controls, held-out permutation
importance, leave-one-block-out ablations, and cross-seed nearest-neighbor
opposite-label rates. Report training versus held-out sequence performance and
20/60-second sensitivity. Preserve every selected harmful, unsafe and
failed-dose case.

Threshold selection and feature/model selection occur only inside grouped
development folds. No row-level split and no after-outcome threshold adjustment
is allowed.

## 9. Frozen go/no-go gates

An observation family passes only when all applicable conditions hold:

- safety or conditional-benefit ROC-AUC improves by at least 0.03 over the
  correctly matched current baseline;
- gain is not driven by one seed and remains directionally positive under seed
  removal;
- shuffled labels remain near chance and Brier/log loss do not materially
  deteriorate;
- at least 10 commands across at least eight independent seeds;
- selected safety at least 90%, joint-benefit precision at least 50%, and
  selected dose success at least 80%;
- no selected excess collision or teleport and no single beneficial outlier;
- coverage does not collapse solely to obtain precision.

An 80% safety / 45% benefit policy is exploratory enrichment only. It is not a
confirmatory pass.

If no block passes, stop feature search, classify whether the remaining failure
is dose, safety, breakdown-risk, or treatment-effect observability, and report
near-identical opposing-label cases. Preserve deterministic Gate F and assess a
high-safety/lower-benefit-confidence-bound abstention policy. If that also lacks
support, the paper fallback is the verified damping-preserving escalation
architecture, deterministic feasibility supervisor, wall mechanics/operating
boundary, and an honestly exploratory or negative learned gate.

## 10. Execution, integrity, and runtime

Capture writes one atomic seed artifact only after all four decision records
and every admitted shadow pass validation. Resume skips only an artifact with
the exact schema, contract hash, seed, case IDs and decision grid; malformed or
mismatched artifacts fail closed. The immutable manifest binds source hashes,
scenario, frozen arm manifests and this contract. `progress.json` is written at
start, every ten newly completed seeds, and completion. Workers are restricted
to 1--4.

All new machine artifacts remain under
`runs/eval/idm_wall_v5_4_shadow_observation_v2_development/`: resumable seed
captures under `capture/` and fitted-screen artifacts under `analysis/`.

Expected four-worker runtime is 1.5--2.5 hours: about 20--40 minutes for one
causal prefix/history capture per seed plus exact ten-second treatment twins for
199 admissions and analysis. The first four seeds are timed before scale-up. A
missing Python or SUMO process during an expected run is diagnosed immediately.

Before the long replay, focused tests must pass for contract hashes, sealed
partitions, causal cutoff, forbidden inputs, outcome-mutation invariance,
deterministic repeat, prefix identity, damping-only twin identity, out-of-band
damping identity, target-only wall overrides, shadow report times, lane/vehicle
entry-exit handling, atomic resume and four-worker enforcement. After results,
run the full `python -m pytest -q` suite.
