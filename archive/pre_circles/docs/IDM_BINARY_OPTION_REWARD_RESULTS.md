# IDM Binary Activation Option-Reward Results

**Date:** 2026-08-29  
**Status:** reward contract passed; paired option-wrapper implementation
authorized; ordinary multi-agent reward and RL training remain unauthorized

## Decision

The fixed 60-second paired reward in `idm/option_reward.py` is valid for the
binary native-IDM-abstain versus P30-activate learning problem. It passes every
predeclared identity, reproduction, sign, ordering, neutral, correlation,
component-balance, hard-failure and boundedness gate.

The existing per-step multi-agent reward remains rejected. The passing reward
requires an identical counterfactual abstention branch during training or
stored paired option samples. It cannot be silently plugged into the ordinary
single-trajectory RL environment.

## Reward

For one predecision state, both abstention and P30 activation run for exactly
60 controller steps. The physical effect is the equal-weight reference-scaled
reduction in corridor severe braking and acceleration energy:

```text
u_B = (B_abstain - B_activate) / (60 * B_ref)
u_E = (E_abstain - E_activate) / (60 * E_ref)
u   = 0.5*u_B + 0.5*u_E

R(abstain)  = 0
R(activate) = tanh(u)    if every constraint passes
R(activate) = -2         otherwise
```

Delay, arrival, discharge, low-TTC change, dose, completion and safety are
constraints. They do not add positive dense reward. Queue area is absent. A
good service result cannot conceal a bad physical effect, and a favorable
physical term cannot compensate for a failed constraint.

There is no activation penalty in version 1. This avoids reversing small
causal benefits. A conservative deployment threshold can later prefer
abstention for predicted neutral states without changing training labels.

The frozen protocol is `IDM_BINARY_OPTION_REWARD_PROTOCOL.md`.

## Audit cohort

The audit used the nine existing ranked-neighbor P30 development roots at
1900, 2000 and 2100 veh/h, recurrences 1--3, and seeds 4121--4132. It retained
all 90 decisions:

- 79 constraint-qualified treatments;
- 11 constraint-failed treatments;
- among qualified treatments, 29 helpful, 34 neutral and 16 harmful;
- 45 qualified nonneutral effects;
- 41 effects with absolute magnitude at least 0.06.

No new SUMO run or prospective seed was used.

## Results

Every gate passed:

- all nine source protocols, horizons, reference hashes and decision identities
  matched;
- all 90 summary cases were retained;
- independently computed constraints exactly matched the stored constraints;
- independently computed severe, energy and composite effects reproduced the
  stored values with maximum error 0;
- maximum abstain-replay reward was exactly 0;
- every qualified nonneutral reward sign agreed;
- every substantial effect was ordered correctly;
- maximum qualified neutral absolute reward was 0.01008, below the 0.02
  deadband;
- Pearson reward/effect correlation was 0.99944;
- Spearman reward/effect correlation was 1.0;
- pooled absolute component mass was 70.39% severe braking and 29.61%
  acceleration energy, below the 80% dominance cap;
- all 11 failed treatments received exactly -2 and ranked below every
  admissible reward;
- all rewards were finite and within the frozen range;
- lifetime, live-agent count, queue, previous online returns, seed and decision
  identifiers were verified to have no effect on reward.

The 11 constraint failures comprised five insufficient-dose cases, four
discharge-budget failures and two arrival-budget failures. There were no
collision, teleport, TTC, horizon or unexpected-command failures in this
cohort. One positive physical effect was correctly overridden because its dose
was insufficient; this verifies that a favorable composite cannot bypass the
contract.

The authoritative artifact is
`runs/eval/idm_binary_option_reward_audit_v1`, containing `summary.json`,
`records.jsonl` and `summary.md`.

## Authorization boundary

This result establishes that a good causal reward function now exists for the
binary option problem. It authorizes the next implementation step:

1. build a resumable paired one-step option-training wrapper;
2. feed it a frozen treatment-independent decentralized observation;
3. return this reward after both 60-step branches finish;
4. verify that branch order, selected-CAV departure and worker concurrency do
   not change reward;
5. run a 2,000-step, then 10,000-step learning smoke.

It does not yet authorize training because the paired wrapper is not wired and
the actor observation remains a separate validity question. In particular:

```text
valid paired reward                   = true
paired option-wrapper implementation  = authorized
ordinary multi-agent reward           = false
RL training                            = false
```

## Verification

Focused reward tests pass, and the complete repository suite passes:

```text
700 passed
```

The dirty worktree and all existing user changes were preserved.
