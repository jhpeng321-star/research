# IDM Decentralized Reward-Credit Diagnosis

Status: the current online reward is not authorized for long MAPPO training.
This diagnosis uses development-only paired outcomes and does not promote a
controller.

## Frozen outcome versus current credit

The scientific action label is a 60-second paired corridor outcome after one
decentralized activation decision. The current multi-agent environment instead
returns a cooperative one-step corridor reward to every CAV currently present
in the action dictionary. The selected CAV's trajectory terminates when that
vehicle leaves SUMO. Treatment can change that departure time, so treatment and
control accumulate different reward horizons for the very action being
compared.

This is a structural credit-horizon mismatch. It cannot be repaired by changing
the weights of the existing severe-braking, acceleration-energy, TTC, delay or
efficiency terms.

## Fresh evidence

The audit covers 79 dose- and safety/service-qualified fresh p30 cases from
seeds 4121--4132 at 1900, 2000 and 2100 veh/h:

- selected-CAV return/effect correlation: 0.733;
- selected-CAV nonneutral sign agreement: 84.4%;
- safety-only component reweighting: correlation 0.733 and the same 84.4%
  agreement;
- full-horizon sum over all active agents: correlation 0.884 and 88.9% sign
  agreement.

The team aggregate improves because it continues after the selected vehicle
leaves, but its magnitude depends on how many other CAV agents are alive. It is
not a clean return for the original decentralized decision and still fails the
frozen requirement that every nonneutral sign agree. Neutral cases can also
accumulate large return differences, confirming that return magnitude is not
calibrated to the paired deadband.

The artifact is
`runs/eval/idm_decentralized_p30_fresh_lower_reward_alignment_diagnosis_v1/summary.json`.

## Correct training abstraction

The damping gate is a temporally extended option, not a new freely selected
action every second. After a treatment is selected and validated, the clean
learning abstraction is an option-level decentralized decision:

1. freeze one causal candidate observation at the declared risk landmark;
2. choose abstain or the frozen damping option once;
3. latch execution for 20 seconds under the deterministic safety shield;
4. continue the environment to the fixed 60-second outcome horizon even if the
   selected physical CAV exits;
5. return one centered, component-audited option reward and terminate that
   decision sample.

This may be implemented as a contextual-bandit/one-step PPO training wrapper or
an equivalent semi-Markov option environment. The actor remains decentralized:
it receives only the frozen causal local/RSU observation. A centralized critic
may use training-only global state, but deployment cannot require it.

Keeping a departed vehicle as a zero-observation "ghost" in the ordinary
multi-agent episode is not preferred. It complicates agent identity, permits
overlapping decision horizons, and still allows returns beyond the declared
option unless synthetic termination and re-entry are handled exactly.

## Reward contract before implementation

The option reward must be centered using frozen reference scales and must log
separate components for severe braking, acceleration energy, TTC and bounded
service. Collision, teleport, projection bypass and service-budget violations
remain hard penalties/gates rather than quantities a positive safety term may
hide. The chosen formula must pass on paired development branches:

- every qualified nonneutral reward sign agrees with the paired effect label;
- reward/effect correlation is at least 0.80;
- neutral reward magnitude is bounded relative to the deadband;
- no component contributes more than 80% of absolute reward mass;
- abstain replay is exact;
- no reward uses future information in the decision observation.

Implementing this wrapper before the p30/p50 treatment decision would encode a
rejected action into the training interface. Therefore the wrapper is the next
reward step only after a damping treatment passes dose, safety and development
efficacy gates. Until then, MAPPO and any long training run remain unauthorized.
