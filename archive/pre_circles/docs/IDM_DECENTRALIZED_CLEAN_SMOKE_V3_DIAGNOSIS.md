# IDM decentralized clean-smoke v3 diagnosis

Status: the software/training smoke is clean, but no trained checkpoint is
accepted.  Longer PPO training is not authorized yet.

## Clean checkpoint evaluation

The paired development evaluation used seeds 4107, 4109, and 4110 and compared
checkpoint 0, 2k, 5k, and 10k with native IDM (always abstain), forced eligible
activation, and the frozen analytical selected-absorber controller.

- All arms had zero collisions and teleports.
- Checkpoints 2k and 5k reproduced always-activate behavior.
- Checkpoint 10k activated selectively (29 of 233 eligible decisions).
- Checkpoint 10k reduced mean hard-braking events from 11.33 under the
  analytical controller to 10.67, but low-TTC events increased from 2.00 to
  2.33 and the smoke gate failed.
- Deployment thresholds 0.50, 0.54, 0.58, and 0.62 all failed.  The learned
  confidence score did not reliably separate helpful from harmful actions.

Artifacts:

- `runs/eval/idm_decentralized_activation_clean_smoke_v3_eval`
- `runs/eval/idm_decentralized_activation_clean_smoke_v3_threshold_screen`

## Missing observation-sufficiency gate

The earlier activation gate established causal action diversity and reward
alignment, but checked only observation dimension and nonduplication.  It did
not run the frozen leave-one-seed-out helpful/harmful discrimination test for
the p30 speed-ceiling action.

The new action-specific pool contains 51 paired cases across 14 development
seeds and both 2200 and 2700 veh/h cells.  On dose-qualified, safety/service
admissible nonneutral cases, the 134 actor-visible physical features achieve:

- leave-one-seed-out AUROC: 0.778 (gate 0.70);
- balanced accuracy: 0.679 (gate 0.65);
- helpful outcomes in nine seeds and harmful outcomes in six seeds.

This is positive evidence that a decentralized predictor is plausible.  The
gate is not complete because only 22 qualified nonneutral cases remain, below
the frozen minimum of 24.

Artifact: `runs/eval/idm_decentralized_activation_sufficiency_pooled_v1`.

## Treatment-feasibility blocker

Only 39 of 51 actions realized at least 16 of the requested 20 command steps
(76.5%, below the frozen 80% threshold).  Safety/service admissibility excluding
short dose is 96.1%.

The short-dose causes are:

- seven lane changes by the selected CAV;
- five rear-secure-gap aborts.

At the second 2700 risk onset, only 6 of 13 actions were dose-qualified.  Every
outer-lane candidate in that cohort terminated early, while every inner-lane
candidate completed.  Several failures occurred when the p30 reference reached
the 8 m/s floor and requested a large terminal-speed drop.  The pre-action SUMO
lane-change desire/capability flags did not separate all failures from
completions, so a simple intent mask is not supported.

## Required decision before the next build

Do not extend PPO from the current checkpoint.  Freeze and screen one revised
treatment definition first.  The recommended option is a **bounded-drop p30
speed ceiling**: retain p30 as the traffic-responsive target but cap how far
below the selected CAV's decision-time speed the 20-second ceiling may be.  This
does not raise braking authority, bypass jerk/TTC/secure-gap protection, or lock
lanes.  It directly addresses the observed 8 m/s floor cases.

Alternatives are less attractive:

- lane locking changes lateral traffic dynamics and should not be introduced
  merely to manufacture dose;
- same-lane handoff changes one-CAV damping into a coordinated multi-CAV
  treatment and blurs the later moving-wall contribution;
- p70 was already rejected as too weak;
- a lane-only feasibility rule is overfit to this small development cohort.

After freezing the bounded-drop values, run a small paired development screen.
Only if dose, safety/service, action diversity, reward alignment, and
observation sufficiency all pass should another clean 10k PPO smoke begin.
