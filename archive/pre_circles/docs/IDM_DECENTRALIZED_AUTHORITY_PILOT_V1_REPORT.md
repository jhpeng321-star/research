# Decentralized IDM Authority Pilot V1

Status: paired software/authority evidence passed; **PPO training not
authorized** because structured mode authority failed the frozen projection
gate.

## Cohort and artifacts

- demand: 2200 veh/h mainline, 400 veh/h ramp;
- development seeds: 4107--4112;
- one first-available two-candidate decision per seed;
- ten paired branches per case, 60 total;
- 20-step option, 40-step safety and 60-step service horizons;
- artifact: `runs/eval/idm_decentralized_authority_pilot_v1/summary.json`.

Seeds are development evidence only. No validation or final seeds were opened.

## Passed evidence

- a two-candidate decision existed in all six seeds;
- exact discovery/branch and all-branch predecision identity;
- exact all-abstain replay;
- all 48 treatment branches realized 20/20 command steps;
- all horizons completed;
- zero collisions and teleports;
- all treatments passed the frozen TTC, delay, arrival and discharge bounds;
- 11 helpful, 29 neutral and 8 harmful action-state pairs;
- 39.58% of treatments exceeded the 0.02 replay deadband;
- helpful outcomes occurred in seeds 4109, 4110 and 4112;
- nonbeneficial outcomes occurred in five seeds;
- candidate-placement contrasts exceeded the deadband in 37.5% of matched
  comparisons; candidate slots 1 and 2 won four and five comparisons;
- every dose-qualified branch changed influence-region acceleration energy.

This verifies that decentralized activation and absorber placement have real,
safe, state-dependent authority. It also verifies that an always-act policy
would be wrong: candidate 1 was harmful for all four modes in seed 4107, while
candidate 1 was helpful and candidate 2 harmful in seed 4109.

## Failed evidence

The initial analyzer incorrectly measured projection survival against the raw
analytical acceleration. A source audit found cases where every mode requested
a different acceleration but the comfort/jerk shield projected all of them to
the same 0.5 m/s2 command. Projection survival was corrected to compare each
mode directly with the paired nominal-mode trajectory for the same candidate.

Under the corrected paired metric:

- median structured-mode projection survival is 0.0, below the frozen 0.50
  requirement;
- 19 of 36 nonnominal candidate/mode branches have exactly zero executed
  command difference from nominal;
- only seven of 36 exceed 0.50 survival.

The favorable outcome diversity therefore establishes activation/placement
authority, but it does not establish a learnable four-mode strength choice.
Training the current five-action policy would waste samples on modes that are
often physically identical after projection and could produce misleading mode
credit.

## Decision

Do not start PPO with the present five-mode action space. The next design must
be frozen before new outcomes are inspected. Defensible options are:

1. add action-specific projected-authority masks and admit a mode only when its
   paired one-step command differs materially from nominal;
2. redesign mild/reference values so they alter the command without weakening
   comfort, jerk, TTC or secure-gap limits;
3. reduce the learned choice to activation/candidate selection plus one or two
   empirically distinct modes.

The hard safety shield must not be relaxed to manufacture mode diversity.
