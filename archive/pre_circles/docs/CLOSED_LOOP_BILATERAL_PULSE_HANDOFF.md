# Closed-Loop Bilateral Pulse Handoff

**Workspace:** `D:\research-dense`  
**Date:** 2026-08-30  
**Status:** user-authorized treatment redesign; implementation partially present;
post-implementation tests and SUMO mechanism pilot not yet verified  
**Working paper title:** *Counterfactual Escalation from Decentralized CAV
Damping to Coordinated Moving-Wall Control*

## 1. Authority and immediate objective

The user explicitly authorized replacing the fixed open-loop P30 activation
with a new closed-loop bilateral pulse that continuously checks immediate-
follower compression and traffic response, adapts or terminates early, and
returns smoothly to native IDM.

This authorization permits implementation and a small development-only
mechanism pilot. It does **not** authorize a long PPO/MAPPO campaign,
predictor-scale data generation, paper-scale evaluation, or claims that the
damping controller, moving wall, or escalation predictor works.

The immediate objective is:

> Finish and verify a separately versioned, low-authority, one-CAV,
> closed-loop bilateral damping pulse; then compare it against exact native-IDM
> abstention and the matched fixed P30 treatment on already inspected seeds.

The scientific question is whether feedback makes the treatment more stable
and less harmful than fixed P30. Observation learning and RL come only after
that mechanism gate passes.

## 2. Required reading and workspace rules

Read completely before editing:

1. `AGENTS.md`
2. this handoff
3. `docs/IDM_P30_SIX_PENETRATION_VALUE_DIAGNOSIS.md`
4. `docs/IDM_REAR_AWARE_FEEDBACK_TREATMENT_PROPOSAL.md`
5. `docs/IDM_BINARY_OPTION_REWARD_RESULTS.md`
6. `docs/IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md`
7. `docs/IDM_DECENTRALIZED_CONTROL_CONTRACT.md`

Preserve the dirty worktree and every existing user change. Work only in
`D:\research-dense`; do not resume medium-density optimization in
`D:\research_v2`. Shared SUMO assets remain in
`D:\research\highway_merge_shepherd`.

Before launching any long run, inspect active Python/SUMO processes and avoid
competing with another experiment. Put transient experiment artifacts under
`runs/probe/` or `runs/eval/`, checkpoints under `runs/rllib/`, and shareable
files under `output/`.

## 3. Paper logic that must be preserved

The paper is not claiming that decentralized damping solves every metastable
realization. Its architecture is deliberately hierarchical:

```text
IDM mixed traffic
        |
        v
Low-authority decentralized CAV damping
  - safe local intervention when one CAV has useful authority
  - permitted to abstain or fail to resolve some metastable realizations
        |
        v
Counterfactual escalation decision
  - Is a coordinated wall feasible?
  - Conditional on feasibility, is escalation expected to help?
        |
        v
Temporary coordinated moving wall
  - high-authority inflow control
  - release back to the identical frozen damping arm
```

The user does not require the damping controller to win every seed. The final
acceptance concept should be predeclared statistical benefit with bounded harm:
positive paired mean on untouched seeds, more helpful than harmful outcomes
(preferably a majority helpful when the deadband permits), a high helpful-or-
neutral fraction, no added hard safety failures, acceptable service, and a
bounded lower tail. Metastability explains outcome diversity; it does not
permit hiding severe or systematic harm.

Do not assume a numerical critical density in advance. Empirically identify
where decentralized control has useful authority and where residual risk
supports escalation to the wall.

## 4. Evidence completed before this handoff

### 4.1 Fixed P30 has authority but unstable value

The six-penetration development grid completed all 18 cells and contains 178
paired third-risk-onset decisions across demands 1900, 2000 and 2100 veh/h,
CAV penetrations 5%, 10%, 15%, 20%, 25% and 30%, and seeds 4121--4132.

The fixed P30 action produced real command and outcome changes, but its benefit
did not generalize reliably across penetration and seed. The existing
instantaneous 134-field state achieved AUROC 0.407 in the six-penetration
diagnosis, and the best compact 13-field option context achieved AUROC 0.566.

### 4.2 Extra causal temporal observations did not solve the problem

A p15 temporal replay created 30 complete causal histories with 31 one-second
frames each. It contained 12 helpful, 10 harmful and 8 neutral P30 effects.
Grouped leave-one-seed-out performance remained inadequate:

- current 134 fields: AUROC 0.142, balanced accuracy 0.233;
- temporal levels: AUROC 0.475, balanced accuracy 0.500;
- temporal trends: AUROC 0.550, balanced accuracy 0.575;
- all 52 temporal fields: AUROC 0.517, balanced accuracy 0.442;
- current plus temporal: AUROC 0.283, balanced accuracy 0.308.

The frozen authorization thresholds were AUROC 0.70 and balanced accuracy
0.65. Temporal observations were not promoted into the actor.

### 4.3 Conditional expected value also failed to generalize

Leakage-safe ridge screens selected hyperparameters and activation thresholds
on seeds 4121--4128, then opened held development seeds 4129--4132 once. The
nontrivial option-context policy activated 16/58 held rows and selected 3
helpful, 5 harmful and 8 neutral outcomes, with mean policy reward -0.0018.
No feature family passed.

Even the forbidden exact demand/penetration cell upper bound failed to
generalize reliably. This is why the current blocker is treatment stability,
not simply a missing feature or a neural-network capacity problem.

### 4.4 Limited fallback remains available

A fixed 5--10% penetration rule had held-development coverage 25.9%, selected
7 helpful, 3 harmful and 5 neutral activations, and achieved conditional mean
reward +0.0434. It is an honest narrow-domain comparator or fallback, not a
learned cross-penetration controller.

### 4.5 Reward status

The fixed 60-step paired binary option reward is valid. Its audit retained 90
decisions and passed replay identity, exact effect reproduction, sign,
ordering, boundedness, component balance, and hard-constraint gates. The
ordinary variable-lifetime multi-agent reward remains rejected. Any future RL
must use a paired option/contextual-bandit or equivalent fixed-horizon causal
abstraction.

At the last recorded repository checkpoint, the complete suite reported 700
passing tests. That result predates the latest bilateral runtime integration
edits and must not be presented as verification of the current partial slice.

## 5. Authorized treatment definition

### 5.1 Concept

Fixed P30 says: select one CAV, impose a fixed low terminal-speed ceiling for
up to 20 steps, and observe the result afterward.

The bilateral pulse instead says: select one physically suitable upstream CAV,
ramp away from native IDM for a short probe, measure the front disturbance and
the immediate follower response, hold only while causal response is improving,
and release early when benefit is absent or rear/safety/service conditions
deteriorate.

"Closed loop" means the treatment remeasures its effect during execution.
"Bilateral" means the front objective is limited by the state of the immediate
follower behind the controlled CAV.

### 5.2 Invariants

- Native IDM is the exact abstention arm and hard fallback.
- Only the declared selected CAV receives the pulse.
- Other CAVs retain the identical native-IDM/decentralized behavior.
- The existing comfort, jerk, TTC, secure-gap, SUMO safe-speed and collision
  projections remain authoritative and run after the feedback request.
- Front damping demand may never override rear protection.
- A conflict between damping and rear safety causes release toward native IDM,
  not stronger braking.
- Release is smooth and returns to exact native IDM; never release to an
  uncontrolled SUMO command.
- Treatment decisions and diagnostics use only causal information available at
  or before the current control step.
- Maximum active duration remains 20 control steps.

### 5.3 Candidate selection

For each upstream CAV, compute at least:

- usable runway to the detected disturbance;
- estimated shock-interception time from relative CAV/shock speed;
- front and rear secure-gap headroom;
- front and rear TTC/closing state;
- time remaining before merge, lane exit, or loss of authority.

Reject unsafe rear states, insufficient probe time and nonpositive
interception. Rank feasible candidates using a frozen score favoring timely
interception, runway and rear headroom. Distance to the merge alone must not
choose the absorber.

### 5.4 Front feedback and rear protection

The front target is FollowerStopper/Jam-Absorption-Driving-shaped, not an
unconditional P30 terminal target. Close leader gaps may request stronger
slowing; intermediate gaps interpolate toward leader/downstream speed; open
gaps return toward native IDM. The current partial contract uses P50 as a
reference endpoint, but P50 is not a mandatory achieved speed.

Compute a rear-protection acceleration floor from follower gap, follower
closing speed, follower braking capability and secure-gap margin. The requested
acceleration may be no more negative than this floor. Apply the existing hard
safety projection last.

### 5.5 Lifecycle

1. **Probe:** jerk-limited ramp from native IDM for 3--5 control steps.
2. **Measure:** track disturbance/wave response, severe braking, projection
   survival, follower gap/closing/acceleration, TTC and service indicators.
3. **Conditional hold:** continue only if response improves and rear,
   safety and service signals remain admissible.
4. **Early release:** trigger on absent response, worsening wave energy, rear
   compression, follower severe braking, TTC/service deterioration, lane
   change, disturbance resolution or lost authority.
5. **Maximum duration:** cap probe plus hold at 20 steps.
6. **Fast smooth out:** promptly converge to exact native IDM through the
   existing projected-release contract.

### 5.6 Diagnostics required per activation

Record:

- selected vehicle, lane, runway and candidate-ranking components;
- requested and realized probe/hold/release steps;
- command delta relative to native IDM at every active/release step;
- front target, rear acceleration floor and final projected command;
- projection survival/conflict reasons;
- follower gap, closing speed, acceleration, TTC and severe braking;
- causal disturbance presence, spread/energy/braking response;
- service metrics on the identical paired horizon;
- phase transitions and exact early-release reason;
- post-release response through the full outcome horizon.

## 6. Current code state as of 2026-08-30 02:19 local time

No Python, SUMO or Ray process was active at the timestamp above. The previous
agent had begun implementation after the treatment decision, but did not leave
a passing test report or SUMO pilot artifact.

Files already added or edited for this treatment include:

- `idm/rear_aware_feedback.py`
  - pure versioned contract `idm_rear_aware_feedback_pulse/1`;
  - five-step strength ramp `(0.2, 0.4, 0.6, 0.8, 1.0)`;
  - maximum 20 active steps;
  - `FeedbackResponse`, `treatment_strength()` and
    `feedback_release_reason()`;
  - causal release reasons for rear floor, projection conflict, follower hard
    braking, rear compression, disturbance worsening/resolution, insufficient
    authority and absent probe improvement.
- `idm/decentralized_damping.py`
  - action-set identifier
    `activation_rear_aware_feedback_pulse_v1`;
  - abstain plus one rear-aware feedback mode;
  - P50 endpoint reference.
- `env/idm_residual.py`
  - partial front/rear feedback command construction;
  - probe/hold response state;
  - release evaluation and projected return toward native IDM;
  - diagnostics hooks.
- `tools/run_idm_decentralized_authority_verification.py`
  - treatment identity/fingerprint support;
  - fixed 20-step validation;
  - probe-dose minimum;
  - matched P30 decision-source support;
  - development-only blocker preventing this pilot from authorizing training.
- `tests/test_idm_rear_aware_feedback.py`
  - pure ramp, hold and release-reason tests.
- `tests/test_idm_residual.py`
  - probe blending and smooth early-release tests.
- `tests/test_idm_decentralized_authority_verification.py`
  - distinct matched-treatment and protocol tests.

Treat this as **partial, unverified implementation**. Audit the diff rather
than recreating it. In particular, verify that the runtime measurements really
are causal, correspond to the declared immediate follower and disturbance,
and persist correctly across probe, hold and release. Do not assume that the
presence of unit tests means SUMO behavior is correct.

The proposal file still says the treatment is not approved. Supersede that
status with a separately frozen implementation/pilot protocol recording the
user authorization; preserve the proposal as provenance rather than silently
rewriting history.

## 7. Immediate implementation and verification plan

### Step 1: audit and freeze before further behavior changes

Read the current partial implementation end to end and produce a short report
before editing:

- exact front feedback equation and its variables;
- exact rear-floor equation and physical assumptions;
- candidate admission/ranking rule;
- probe/hold/release state transitions;
- exact mapping from control steps to simulated seconds;
- every release reason and its threshold;
- files requiring completion;
- exact matched pilot cohort and source root;
- expected runtime.

The user has authorized the treatment family, but thresholds and the exact
pilot cohort must still be frozen before looking at its efficacy. If changing
the P50 reference, candidate ranking, probe length, maximum duration, selected
CAV count, follower definition, decision landmark, penetration, or primary
outcome, explain and freeze the choice before the pilot.

### Step 2: finish the smallest coherent runtime slice

- Complete assemble/probe/hold/release state persistence.
- Ensure baseline feedback state is latched at activation and not contaminated
  by post-treatment information.
- Ensure follower identity changes are handled explicitly; do not silently
  compare different followers as if they were the same physical response.
- Ensure lane change, departure, target loss and disturbance loss produce
  explicit release reasons.
- Keep the hard safety projection final and authoritative.
- Verify out-of-target vehicles receive no feedback-pulse command.
- Verify projected release converges to the exact native-IDM command.
- Preserve abstain replay and predecision identity.
- Finish dose, response, follower and release diagnostics.

### Step 3: tests before SUMO pilot

Run focused tests first:

```powershell
python -m pytest -q `
  tests/test_idm_rear_aware_feedback.py `
  tests/test_idm_residual.py `
  tests/test_idm_decentralized_authority_verification.py `
  tests/test_idm_projected_release.py
```

Then run:

```powershell
python -m pytest -q
```

Record the exact counts, failures and environment. Do not launch the SUMO
pilot with a failing relevant test.

### Step 4: freeze the matched six-seed mechanism pilot

The proposal names already inspected development seeds:

```text
4123, 4127, 4129, 4130, 4131, 4132
```

It specifies 2000 veh/h and matched native-IDM, fixed-P30 and feedback-pulse
comparisons. The latest diagnosis concerns third-risk-onset decisions. The
proposal did not itself freeze a CAV penetration/source root; therefore the
next agent must explicitly verify and freeze that choice before execution.

The most directly supported candidate is the existing p15, 2000 veh/h,
third-risk-onset source:

```text
runs/eval/idm_paired_option_penetration_grid_l3_v1/
  idm_paired_option_grid_p15_m2000_l3_v1
```

This is a recommendation, not permission to silently change the scientific
cohort. Confirm that it contains the intended matched decisions and adequate
prior helpful/harmful diversity. If not, stop and document a better source
before running.

Subject to that freeze, the expected runner command is:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense

python tools/run_idm_decentralized_authority_verification.py `
  --output runs/probe/idm_rear_aware_feedback_p15_m2000_l3_six_seed_v1 `
  --seeds 4123 4127 4129 4130 4131 4132 `
  --demand 2000 `
  --workers 4 `
  --action-set activation_rear_aware_feedback_pulse_v1 `
  --option-steps 20 `
  --safety-horizon-steps 60 `
  --cav-penetration 0.15 `
  --decision-index 3 `
  --maximum-landmarks-per-cell 3 `
  --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
  --candidate-ranking runway `
  --candidate-count 1 `
  --landmark risk_onset `
  --decision-source runs/eval/idm_paired_option_penetration_grid_l3_v1/idm_paired_option_grid_p15_m2000_l3_v1 `
  --response-trace-contract idm_phase_response/1 `
  --diagnostic-spatial-history
```

The runner's feedback action set produces abstain, abstain replay and feedback
branches at the frozen P30 decision/target. The fixed-P30 comparison is in the
decision-source artifact rather than a fourth branch in the same output. Add a
small audited analyzer if necessary to join only exact common case/decision/
target identities; never compare unmatched rows.

### Step 5: frozen pilot integrity gates

Before efficacy interpretation require:

1. exact abstain-versus-abstain replay identity;
2. exact predecision branch identity;
3. zero control commands in both abstention branches;
4. feedback commands only on the declared target CAV;
5. no hard safety-projection bypass;
6. no added collisions or teleports;
7. complete causal outcome horizons;
8. at least five of six **feasible** activations receive the full declared
   probe dose; report seeds with no valid decision separately;
9. phase traces and command deltas exceed replay/numerical noise;
10. release returns to exact native IDM without an uncontrolled interval;
11. follower identity and any change/handoff are explicitly logged;
12. fixed-P30 and feedback rows share the exact decision, target and
    predecision state before effect comparison.

### Step 6: frozen development efficacy gates

Retain the pulse for broader development only if, among exact common qualified
cases:

- it has fewer harmful activations than fixed P30;
- it produces meaningful benefit beyond the existing 0.02 replay deadband in
  at least two independent seeds;
- it retains at least two nonbeneficial seeds, demonstrating outcome diversity
  rather than a degenerate label;
- it adds no low-TTC or immediate-follower severe-braking failure;
- it does not conceal arrival, discharge, delay or other service loss;
- its gains are not created solely by aborting every treatment after a trivial
  dose;
- release/post-release effects are included, not truncated at the end of the
  active command.

This is a mechanism-development gate, not final paper verification. It need
not win all six seeds. Report every unfavorable result.

## 8. What happens after the pilot

### If the mechanism gate passes

1. Freeze the pulse implementation and hashes.
2. Run a limited broader development screen across prespecified operating
   cells using only already opened development seeds.
3. Confirm positive average effect, more helpful than harmful outcomes,
   bounded lower-tail harm, adequate dose and safety/service compliance.
4. Re-audit the treatment-independent decentralized observation for the new
   pulse decision. Do not assume the failed P30 benefit features are optimal.
5. Reuse or version the passing paired fixed-horizon option reward only after
   exact effect/constraint reproduction for this pulse.
6. Run a short paired contextual-bandit/semi-Markov PPO smoke, not ordinary
   per-step MAPPO with the rejected reward.
7. Prospectively validate and freeze the damping checkpoint on untouched
   seeds.
8. Revalidate the damping-preserving moving-wall overlay and treatment dose.
9. Only then generate escalation data and train the two-stage wall-feasibility
   and conditional-benefit predictor.

### If dose/mechanics fail

Diagnose candidate feasibility, rear-floor conflict, projection survival,
follower identity, lost disturbance authority and release implementation.
Repair mechanics before adding seeds or changing learning algorithms.

### If mechanics pass but value remains unstable

Do not add a larger neural network or tune reward weights to hide the failure.
Use the honest 5--10% fixed-P30 operating-domain fallback or a conservative
abstention controller, and make coordinated moving-wall escalation the main
high-authority contribution.

## 9. Claims and actions still prohibited

Until prospective evidence exists, do not claim:

- the bilateral pulse works or generalizes;
- a trained IDM damping checkpoint exists;
- probabilities are calibrated;
- collapse prevention has been demonstrated;
- the moving wall has passed its corrected dose/headroom pilot;
- the escalation predictor works;
- results generalize to real traffic or all IDM parameterizations.

Do not:

- train another predictor;
- generate a paper-scale dataset;
- run a long PPO/MAPPO campaign;
- reuse Krauss outcomes as primary IDM evidence;
- tune on seeds 4121--4132 and later call them prospective;
- discard harmful or constraint-failed cases;
- change the treatment definition after viewing pilot outcomes without
  versioning and disclosing the change.

## 10. Seed provenance

Seeds 4107--4132 and all cells already inspected with them are development
only. In particular, seeds 4121--4128 formed the recent training/development
split and seeds 4129--4132 were opened once as held development. None may be
used for final prospective claims.

Maintain a machine-readable list of every additional seed opened. Reserve a
fresh, untouched seed set for controller acceptance, another for wall/predictor
development as required by the protocol, and a final prospective set for paper
evaluation.

## 11. Expected time from this handoff

Assuming the partial implementation is coherent:

- audit/freeze: 1--2 hours;
- finish implementation and diagnostics: 2--4 hours;
- focused and complete tests: 1--2 hours;
- six-seed mechanism pilot: 1--3 hours;
- analysis and documented decision: 1--2 hours.

Best case is approximately 5--7 hours; a realistic first go/no-go result is
8--12 hours. One treatment correction can extend this by 1--3 working days.
A prospectively verified damping checkpoint is more realistically 2--4 working
days after the mechanism begins working. Wall revalidation and the escalation
predictor follow afterward.

## 12. Concise prompt for the receiving agent

Use the following message after opening `D:\research-dense`:

> Read `AGENTS.md` and
> `docs/CLOSED_LOOP_BILATERAL_PULSE_HANDOFF.md` completely, then follow the
> handoff's required reading. Preserve the dirty worktree and all user changes;
> do not resume medium optimization. The user has explicitly authorized the
> new closed-loop bilateral damping pulse. Audit the partial implementation
> already present, report the exact equations/state machine/files/cohort before
> further behavioral changes, finish the smallest causal and safety-preserving
> implementation, run focused tests and `python -m pytest -q`, then freeze and
> run only the matched six-seed development mechanism pilot. Do not run long
> RL, train the escalation predictor, generate paper-scale data, or claim the
> controller works until the documented gates pass. Continue from existing
> files rather than rebuilding them, and report every harmful or failed case.

