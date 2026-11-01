# S-BDRANK cached-only analysis contract

Owner 2026-11-01 16:32; analysis frozen before scoring by this task. No simulator,
model runtime, training, inference, dispatcher, or process-control imports.
Inputs are explicit registered observable reductions / completed prediction caches.
All opened paths pass the prohibited-path guard; metadata source-name strings are
not dereferenced. Peer worktrees and jobs are read-only.

## Populations and detector

- Q1: WMDATA development969400–969431, separate evaluation969432–969439,
  separate descriptive WMDATA2 replication969550–969599 (including generic_random).
- Q2/Q3: source22cc32f6 FULL-v4, all20worlds ×5controllers ×K8, with
  development969420–969431 and reserved969432–969439 strictly separate.
- Registered LRDIAG BIC downward-step+trend function SHA
  `6de4df039e108508cf71f94abfe2e81136d5a9a502bb4803980411c0fa5b3f10`;
  upstream primary, merge/ramp diagnostics, 10s vehicle-time bins, 60s prehistory.
  Independently selected dominant steps at H120/300/600 are not nested events.
  Pre-existing steps are a separate status. No behavioral speed thresholds.
- Depth is minus the registered fitted downward step, only when event=True.
  The registered detector has NO duration / queue-extent / recovery output.
  Stop those items: time to horizon is NOT duration; regional means are NOT queue
  length. No new behavioral definitions are invented.

## Q1 estimands

J is cumulative team_delay_vs = road_delay_vs + source_wait_vs (veh-s).
Export episode-level event, exact onset and depth joined to all3 J clocks; report
status means and within-cell associations, not a causal breakdown cost.
Event/no-event differences are additionally standardized over controller×cell
groups where both statuses occur; unsupported groups stay missing.
Onset/depth conditional correlations and slopes are descriptive, not binned by
arbitrary thresholds. Also export exact-onset conditional delay means.
Within each world compare ALL controller pairs, excluding pairs with a pre-existing
step. Regress J_a−J_b through zero on occurrence E_a−E_b, timing
(E_a*t_a−E_b*t_b)/H, and depth E_a*d_a−E_b*d_b. No-event timing/depth contributions
are zero; unselected candidate-step drops are never treated as events.
Report nested R² and order-independent Shapley allocations of in-sample explained
paired variation over all3! block orders, residual, coefficients, rank, and
leave-one-world-out R². This is association, not causal mediation; negative CV R²
is allowed. Pairs are never treated as independent worlds. Cell fits are sparse.

## Q2 metrics and ties

For each controller/horizon/region: P(new post-start event), conditional detected
onset/depth, expected detected-event depth (E*d, zero without a new event), mean J,
pre-existing probability, and restricted onset-free time (RMST): onset when new
event, H when censored/no event, zero when pre-existing. This is the integral of
the detector-onset survival over [0,H], not an imputed exact onset. Conditional
event-only timing/depth comparisons are separate, with support counts.
Pair concordance is reported BOTH against the matching factual signal and against
factual J ordering (the action-relevant target). Exact equality defines ties;
only floating-point delay/depth arithmetic uses 1e-8 absolute numerical tolerance.
Export concordant/discordant, predicted-only/real-only/both ties; concordance uses
only pairs untied on both sides. Tie-adjusted score gives half credit to predicted
ties on factually distinct pairs, excluding factual ties. Never count saturation
as successful ranking. Option delay ranking uses available O0–O4, not unavailable
options; paired effects suffice to rank because common O0 cost cancels.

## Q3 frozen rules and uncertainty

Hypothetical menu is all5 recorded controllers, not an executable online policy.
Minimize (a) event probability, (b) negative RMST, (c) expected detected-event depth,
(d) mean J, (e) lexicographic (event probability, mean J). Primary upstream.
All exact ties prefer MAPPO, then native, CTRL0, PI, CIRCLES; no factual labels
break prediction ties. Report full tie sets and alternative tie-choice saving
ranges. Regret is chosen J minus minimum factual J over the same available menu;
saving is factual MAPPO J minus chosen J. No confidence gate is added.
Select largest development mean saving separately per horizon, then persist the
selection BEFORE calculating any reserved decision result. Fixed-rule reserved
comparisons are confirmation only; option EXAM diagnostics never select a rule.
World bootstrap percentile95% intervals:10000 draws, seed20261101, fixed
demand×share strata, ALL scored pairs/actions preserved within each sampled world.
Every contributing stratum must have >=2worlds or CI=NA (including pooled results
containing a singleton); empty informative bootstrap draws are disclosed, not
silently discarded. Q1 regression intervals are not claimed; descriptive fit/CV
and underlying pairs are supplied. All numerical output is retrospective.

## Missing fields / reproduction

WMTEACH runtime saves breakdown=None and scalar costs, not speed histories.
Stop option event/onset/depth rules; no inference is launched to fill them.
Inference estimates use recorded wall seconds only; GPU minutes mean occupied
single-worker time, not a throughput promise. Missing exact detector labels also
require archived exact trajectories, or unauthorized new SUMO if none exist.
Round2 controller v5 is included only after its entire registered comparison is
complete; partially collected worlds are not promoted to a panel. Snapshot its
completion receipts read-only and report pending without touching jobs.

Run `python -B tools/bdrank/analyze.py`, then
`python -B tools/bdrank/verify.py`. Only task-owned output files are written.
