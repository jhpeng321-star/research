# CIRCLES RL candidate: IDM merge transfer and penetration check

Status: twelve trajectories completed and verified. Technical compatibility
passes. The constrained adapter is unsuitable as an accepted 5--30% damping
baseline. Direct output is a promising p20 candidate with measured trade-offs;
its performance across penetrations is not yet tested. No accepted checkpoint.

## Completed results

The same weights ran successfully at every tested penetration. Both native
references exactly reproduced saved severe-braking and effort intervals;
system-accounting differences were below 1e-8 vehicle-seconds per interval.
All twelve array hashes/totals passed verification and 1000 saved actor
decisions per controller trajectory reproduced exactly in singleton inference.
Every requested vehicle arrived; all runs recorded zero collision involvements,
teleports and invalid model outputs. This is finite-case evidence, not a safety
guarantee.

For each demand below, values are changes versus the paired native reference:
severe braking %, effort %, system seconds/request, and short-TTC vehicle-seconds.

- **2000 veh/h, constrained:** p5: -8.96%, -1.23%, -0.187 s, -21.5;
  p10: -3.57%, +3.85%, +1.295 s, +60;
  p20: +9.96%, +21.27%, +2.660 s, +136;
  p30: +3.98%, +30.52%, +5.621 s, +116.
- **2200 veh/h, constrained:** p5: -3.21%, +0.44%, +2.845 s, +1.5;
  p10: +12.92%, +17.55%, +8.469 s, +466.5;
  p20: +24.05%, +23.46%, +7.105 s, +492;
  p30: +23.32%, +31.64%, +11.430 s, +477.5.
- **2000 veh/h, direct p20:** -18.76%, -5.29%, +2.084 s, +53.
- **2200 veh/h, direct p20:** -4.63%, +0.94%, +1.474 s, +55.5.

Direct p20 therefore trades reduced severe braking for roughly 1.3--2.0%
more system time and 4.8--11.0% more short-TTC exposure. It improves both
damping costs only in the lower-demand case. Nonzero costs do not automatically
disqualify a controller in metastable traffic; tolerances and primary outcomes
must be specified prospectively, rather than chosen to approve these cases.

The constrained adapter modifies 11.5--14.5% of issued commands and uses native
mode at 33.6--48.7% of CAV decision times. Direct p20 modifies 2.0--4.5% and
uses native mode at 30.4--39.8%. Mode frequencies are measured along each arm's
own resulting trajectory. The paired experiment identifies the effect of the
whole projection change; it does not separate acceleration, jerk and front/rear
constraints or isolate which individual clipped actions caused the difference.

![Paired development outcomes](../output/published_idm_fit_2026-09-06.png)

## Recommended decision and next bounded work

Do not freeze the constrained adapter or continue custom PPO to rescue it.
Keep this external model as a candidate, with its weights frozen. The evidence
supports investigating a compatible direct-acceleration execution contract;
it does not support copying the model into the old tight limits and assuming
the published result transfers.

Before further full rollouts, resolve the documented command cadence and
leader-loss behavior, record the exact allowed acceleration/safety interface,
and finish the source-to-checkpoint attribution. A small mechanics check should
establish requested versus executed actions and causal return to IDM. Existing
handback mechanics evidence can be reused where the command path is identical.
Do not infer successful release merely from getSpeedWithoutTraCI equality.

Then register one transfer decision panel for the chosen fixed adapter at
5%, 10%, 20%, 30%, paired against IDM across multiple development seeds and
both demands, with declared safety/service tolerances and no wrapper search.
Only a passing fixed configuration advances to untouched acceptance seeds.
If this transfer fails, record the candidate as unsuitable; no automatic
successor training or expanded pilot campaign is justified by this report.

This is a concrete path to an adopted learned baseline while preserving
escalation as the paper's contribution. Wall/predictor generation remains
paused. The completed check launches no follow-on simulations or training.

## Question and experimental scope

Can the frozen external controller run with our IDM traffic, CAV-acquirable
observations, and 5--30% CAV penetration, and does the transferred controller
improve our damping outcomes? The code path is separate from stopped custom
learning campaigns. No training, acceptance-seed use, or wall generation occurs.

The two exposed cases are seed 911001 at 2000 mainline veh/h and seed 911003
at 2200 mainline veh/h. These are demand settings, not measured densities.
Each has one native reference, four constrained controller runs at nominal
5%, 10%, 20%, 30%, and a direct-output diagnostic at 20%: twelve full 2400 s
trajectories. Requests, IDM driver types, routes and random seed are paired.
CAV assignment is nested with penetration and matches the existing assignment
at 20%. The route manifest's original CAV population is metadata for 20%; the
individual result's designated population and realized penetration describe
the actual controller assignment in that run.

There is only one exposed seed per demand, so differences are case outcomes,
not population estimates, confidence intervals, or proof of generalization.
The check does not cover 15%, 25%, overload demand, sensor noise or delayed
messages. All seeds remain excluded from prospective acceptance.

## Controller and IDM compatibility

The candidate is the first RL-labelled export selected before physical
outcomes from [CIRCLES algos-stack](https://github.com/CIRCLES-consortium/algos-stack/tree/3b3ee3b91afdae93075c8b90334d65748cd9ff74/controllers),
`09_04_rl_finaldecn3p0_maxacc0p25/super_resolution.onnx`.
The [provenance and graph audit](PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md)
records its hash, licence, exact rule wrapper and attribution limits.
It is an externally released RL candidate; an exact bridge to a particular
paper's evaluated weights is not established.

The actor receives only ego speed, leader speed and bumper gap in SI units.
Leader speed is reconstructed from ego speed and measured relative speed.
Normalization is embedded in the graph. There is no global health, hidden
driver parameter, true penetration or demand input. Singleton inference avoids
the export's cross-vehicle broadcasting defect. Invalid output returns to IDM.

All vehicles retain their IDM type. HDVs receive no controller commands.
CAVs receive an absolute acceleration converted to a one-second speed target;
this replaces their longitudinal request while active and is not an addition
to an approximate IDM acceleration. Missing leader coverage or an infeasible
command returns the CAV to native IDM. SUMO speedMode31 remains enabled.
Thus this is mixed RL/IDM operation, not a claim that the active CAV still
executes unmodified IDM.

The constrained arm uses the existing command projection: acceleration
[-1,+0.5] m/s2, a 0.5 m/s2 change from current acceleration per one-second
command, front/rear bounds, and allowed speed. The direct diagnostic uses
[-3,+1.5], allowed speed and SUMO safety, without the additional workspace
jerk/front/rear projection. It is outside the tighter damping command contract.
Neither command projection establishes bounds on every executed acceleration
or jerk, because native safety and traffic interactions can intervene.

Both arms use 1 s commands, 0.5 s SUMO physics and the existing same-lane
100 m sensor with visibility ending at lane boundaries. These are transfer
assumptions. They do not reproduce the original ROS deployment. The export's
large-gap blend is centered at 120 m, beyond this sensor's maximum leader gap;
that discrepancy is a concrete adapter limitation, not evidence that it alone
causes any observed traffic effect.

## Measures and verification

Severe braking is vehicle-seconds with acceleration below -3 m/s2. Effort is
the sum of squared acceleration integrated over all vehicles and time.
System time includes source waiting and is divided by all requested vehicles.
We also measure source waiting, TTC below 3 s, unfinished requests, collision
involvements and teleports. Full-system data are evaluation only, never actor
inputs. Lower is preferable for each reported cost.

Results and exact arrays are in
`runs/probe/published_idm_fit_check_2026-09-06/results/`; the fixed design is
`fit_manifest.json`, paired outcomes are `summary.json`, and independent array,
reference and singleton replay checks are `verification.json` in that directory.
The implementation is `experiments/idm/check_published_idm_fit.py`; postprocessing
is `experiments/idm/analyze_published_idm_fit.py`.

The initial preflight caught JSON serialization before simulation. The first
parallel launch then encountered the existing Windows append-before-lock bug.
Two completed trajectories were retained; ten failed startup allocations were
preserved and retried once after an isolated lock correction. No controller,
scenario or metric changed. `preflight_amendment.json` and
`startup_amendment.json` preserve both revisions. The planned completed scope
is 28,800 simulated seconds; the failed startup allocations are separately
charged a conservative upper bound of 24,000 seconds because their exact
tick counts were not saved. This is not additional scientific sample size.

## How it would enter escalation

An accepted frozen damping adapter would act locally on every CAV. The RSU
would fuse timestamped CAV state and detected-neighbor reports, first assess
whether a safe wall can be formed, then predict its incremental benefit over
continuing that same damping controller. Only selected wall members switch
command ownership; release must return to the identical frozen damping
adapter. No wall and active wall commands may simultaneously own a vehicle.

The damping actor need not know penetration. At low penetration a CAV can
still damp individually, while a wall may lack the required CAV positions and
lane coverage. The latter should be measured from feasible formations rather
than enforced by an arbitrary penetration cutoff. Health can be a defined
summary of CAV reports, but should not replace raw observations, coverage and
uncertainty in the escalation feature contract.

This architecture is compatible in principle; the current diagnostic does
not implement or validate the moving-wall overlay with this new controller.
