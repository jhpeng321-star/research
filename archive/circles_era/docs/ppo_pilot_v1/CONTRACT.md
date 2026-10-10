# PPO pilot frozen contract (ppo_pilot_v1)

Machine-authoritative source: `traffic_shepherd/ppo_contract.py`
(`contract_sha256()` pins the registration to the frozen constants).
This page is the human-readable statement of the frozen action, observation
and reward specifications. Nothing here may be tuned against development
outcomes.

## 1. Objective and scope

Can a small PPO policy learn useful wall activation and HOLD-continuation
decisions from CAV-acquired, RSU-aggregated information? The learned action
is experimental benefit selection, NOT a validated production benefit gate.
The production admission contract is unchanged; training runs through an
explicitly scoped diagnostic policy (`wall_ppo/1`).

## 2. Action contract (frozen)

- One centralized discrete actor (Discrete(2)) with a value critic; the
  analytical feasibility gate is not a learned network.
- Inactive phase: WAIT(0) or ACTIVATE(1). Assembly: existing deterministic
  lifecycle (no learned commands). HOLD: RELEASE(0) or CONTINUE(1). Exit:
  existing deterministic release/handback.
- Decision cadence: every 5 s (control grid 1 s). Earliest activation 120 s;
  latest 1800 s; at most ONE activation per episode.
- Wall-speed profile frozen: the lifecycle_v2 entry profile (target =
  mean reported member speed − 2 m/s, capped 24 m/s) — the profile with the
  most verified completed maneuvers in both exposed seeds and the base of
  all registered continuation evidence. No speed search.
- HOLD: base 20 s (verified); CONTINUE adds +10 s; maximum total HOLD 40 s.
  CONTINUE is committed only after the hold_extend/2-style admission
  evaluates the PROPOSED remaining maneuver (extended hold + exit);
  an infeasible continuation is declined and recorded, leaving the current
  schedule unchanged. This is the explicit, verified generalization of
  hold_extend/1's single extension — never silent reuse. RELEASE schedules
  the taper at the current tick.
- Episode end: forced taper at 2350 s if still holding; the 2400 s horizon
  always completes (no truncation).
- Protective exits: unchanged lifecycle machinery (assembly deadline, lost
  compact hold, clearance, response incompatibility, no feasible owned
  exit, exit deadline).
- ACTIVATE masking: available only when a feasible proposal for the frozen
  per-scenario cohort exists in current report evidence (cheap check);
  masked actions are never sampled, PPO log-probabilities use the same
  mask, and the full admission is re-checked at execution with requested /
  available / executed / rejection recorded per decision.

## 3. Observation contract (frozen)

Only timestamped CAV self-state and locally sensed-neighbor reports
available to the RSU at the decision tick, plus the pilot's own phase
bookkeeping. 29 features (see `ppo_contract.FEATURE_NAMES` and the table in
`ppo_features.build_features`): cohort speeds/accel/spread/span/gaps;
all-reporter and regional (approach ≤ 400 m, downstream) statistics;
braking-reporter count; report counts/ages/coverage; phase one-hots;
elapsed and remaining permitted HOLD; episode clock; activation budget;
5 s trends; and the activation-mask value itself. Temporal information via
a fixed stack of the last 4 decision ticks (20 s span, zero-padded),
input dimension 116. Missing reports → statistic excluded + coverage
feature; no global density, penetration, hidden type, futures, matched
reference outcomes, shadow rollouts or hindsight labels. Normalization:
Welford mean/variance fitted on TRAINING decision steps only (round-start
state used for each round's update), frozen for evaluation. The
exploratory post-release fusion signals are NOT used and remain
unvalidated for wall entry.

## 4. Reward contract (frozen)

Per control tick, from the same interval counters the outcome evaluator
sums: `r_t = −(Δservice/250000 + Δwait/40000 + Δbraking/2000 +
Δeffort/200000 + Δttc/2000) − 25·collisions − 5·teleports − 5e-5·[wall
owned]`. Scale constants are round numbers from the two registered damping
episodes (a typical full-episode component ≈ 1 scaled unit). Intervention
bound: wall ownership ≤ 100 control steps → maximum episode contribution
0.005 scaled units, strictly below the smallest registered-material benefit
(service ≥ 1 s/request ≈ 0.006 at n=1415), so accumulation cannot dominate
physical benefits. Raw components, normalized components, intervention
cost and return are logged separately. Reward ordering is pinned by
fixtures: unnecessary intervention loses to damping; helpful intervention
can win despite its cost; harmful intervention scores worst; delayed
penalties are retained (a collision at 2000 s lands fully in the return);
a better return never changes the registered acceptance classification.
γ = 1.0, GAE λ = 0.95, full-episode rollouts, no credit reset at release:
consequences hundreds of seconds later reach earlier decisions with full
undiscounted weight (λ decays advantage traces, not the return).

## 5. Scenarios, splits, budgets, stopping

Training scenarios (8): 4107/4108 m2700 (exposed) + 4109 m2700, 4110 m2000,
4111 m2700, 4112 m2000, 4113 m2700, 4114 m2000. Development (4, disjoint):
4115 m2700, 4116 m2000, 4117 m2700, 4118 m2000. All twelve lie in the
declared exposed partition 4107–4132; none is in any sealed block; the
partition's archived-era (different map/actuator) assignments are recorded
here honestly. One optimizer seed 20260907; training episode i uses
scenario `TRAINING_SCENARIOS[i % 8]` and sampling seed 20260907 + i.
Ceilings (maximum allocations, not targets): mechanics 2×120 s; references
12×2400 s; training 32×2400 s; development 12×2400 s; total 134,640 s —
enforced through the real launch path by the persistent campaign ledger
(`runs/active/budget/ppo_pilot_campaign.json`). Stopping rules: a failed
mechanics allocation stops the campaign until reported; non-finite loss /
observation / reward; any contract violation; any budget rejection. A
policy that legitimately prefers damping is a result, not a bug. PPO
configuration: 2×64 tanh MLP, lr 3e-4, clip 0.2, 4 epochs, minibatch 256,
entropy 0.01, value 0.5, grad-norm 0.5, 8 updates × 4 episodes. Evaluated
checkpoint: the final completed update (best-dev selection forbidden).

## 6. Development evaluation (frozen before training)

One frozen policy (deterministic argmax deployment), one fixed-wall
comparator (lifecycle_v2 at 600 s) and one simple predeclared report-based
rule (ACTIVATE at the first decision tick in [600,1200] where the frozen
cohort reports fresh with mean speed < 18 m/s; always RELEASE at the first
HOLD decision) on each of the four development scenarios; damping supplied
by the matched references. Grading: per-scenario paired differences
(treatment − reference) on all physical metrics, original guardrails,
complete requested service, collisions, teleports, activation times,
delivered HOLD durations, release/abort causes, refusal rates and time
spent intervening.
