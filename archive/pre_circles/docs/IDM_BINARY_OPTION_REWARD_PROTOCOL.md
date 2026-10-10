# IDM Binary Activation Option-Reward Protocol

**Status:** frozen before implementing or executing the reward audit  
**Date:** 2026-08-29  
**Scope:** development-only binary abstain-versus-P30 reward validation; no RL
training authorization

## Purpose

The next decentralized actor is allowed to be imperfect. Its objective is to
make P30 activation more likely to help than harm on average while the existing
shield remains responsible for command safety. The reward must therefore encode
the causal 60-second physical effect of one activation decision, not the
variable lifetime of the selected CAV or the number of other CAV agents alive.

The current per-step multi-agent return is rejected for this purpose. No
reweighting of that return can make its treatment and control horizons equal.

## Training abstraction

One learning sample is one frozen predecision state and two matched simulator
branches:

1. preserve the treatment-independent decentralized observation;
2. run native-IDM abstention for 60 controller steps;
3. run the selected P30 option from the identical state for 60 steps;
4. issue at most 20 shielded P30 commands;
5. calculate one paired option reward;
6. terminate that learning sample.

This is a one-step contextual-bandit/semi-Markov PPO abstraction. The future
actor remains decentralized. The counterfactual branch is training-only and is
not a deployment input.

## Frozen physical utility

Let `C` denote the identical abstention branch, `A` the activation branch,
`H=60`, `B` the corridor severe-braking count sum, and `E` the corridor
acceleration-energy sum. The reference rates come from
`runs/eval/idm_decentralized_lower_reward_refs_v1.json`, SHA-256
`737b9ce4bb1945f7681b807941c86509ae3143128bd0729d2cb2aee5237c5b58`.

```text
u_B = (B_C - B_A) / (H * B_ref)
u_E = (E_C - E_A) / (H * E_ref)
u   = 0.5*u_B + 0.5*u_E
```

The binary rewards are:

```text
R(abstain)  = 0
R(activate) = tanh(u)       when every paired constraint passes
R(activate) = -2            otherwise
```

There is no activation cost in version 1. Neutral-action preference can be
implemented later as a deployment threshold, without changing the causal
training target. Reward is bounded in `[-2, 1)`. No selected-CAV lifetime,
agent count, configured demand identifier, delay, queue, flow, arrival count,
discharge count, recurrence, seed, or simulation time enters the dense utility.

## Paired constraints

The reward override uses the existing deployment-relevant paired contract:

- activation completes exactly 60 outcome steps;
- at least 16 of 20 treatment commands are delivered;
- no unexpected command;
- no added collision or teleport relative to abstention;
- low-TTC exposure increases by at most two steps;
- mean corridor delay-rate increase is at most 1 s/vehicle;
- arrival loss is at most one vehicle;
- merge-discharge loss is at most one vehicle;
- no treatment safety abort when that diagnostic is present.

These values are constraints only. A favorable braking/energy term cannot
compensate for a failure, and a service improvement cannot make reward more
positive. Queue area is not used.

Because several constraints are counterfactual deltas, this reward is valid
only for a paired option wrapper or stored paired samples. It is not authorized
for the ordinary single-trajectory multi-agent environment.

## Frozen audit cohort

Use the nine existing ranked-neighbor P30 development roots at mainline demand
1900, 2000, and 2100 veh/h, decision recurrences 1--3, seeds 4121--4132:

```text
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m1900_l1_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m1900_l2_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m1900_l3_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2000_l1_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2000_l2_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2000_l3_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2100_l1_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2100_l2_v1
runs/eval/idm_decentralized_p30_ranked_neighbor_validation_m2100_l3_v1
```

These are already opened development seeds. No prospective seed is consumed.
The audit must retain every case and separately report constraint-failed arms;
it may not discard an unfavorable qualified outcome.

## Pass/fail gates

All gates are mandatory:

1. all input action sets, horizons, reference hashes and case identities match;
2. module-generated paired constraints exactly reproduce stored constraints;
3. module-generated physical components and composite reproduce stored effects
   within `1e-12`;
4. abstain-versus-abstain replay reward is exact within `1e-12`;
5. every qualified helpful activation has positive reward and every qualified
   harmful activation has negative reward;
6. every qualified nonneutral sign agrees, with fraction exactly 1.0;
7. every qualified effect separated from zero by at least 0.06 is ordered
   correctly;
8. maximum qualified neutral absolute reward is no greater than the frozen
   0.02 deadband;
9. Pearson reward/effect correlation is at least 0.95 and Spearman correlation
   is at least 0.99 among qualified actions;
10. neither severe braking nor energy contributes more than 80% of pooled
    absolute paired component mass;
11. every constraint-failed action receives exactly -2 and ranks below every
    qualified reward;
12. rewards are finite and in the frozen range;
13. the reward implementation exposes no selected-CAV lifetime, agent-count,
    queue, delay, flow, seed, recurrence, or postdecision-observation argument;
14. focused tests and the complete repository suite pass.

A pass validates the reward contract and authorizes implementation of the
paired option wrapper. It does not validate observation sufficiency, authorize
a long training run, or claim that a learned controller works.
