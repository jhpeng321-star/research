# IDM P30 Six-Penetration Expected-Value Diagnosis

**Date:** 2026-08-30  
**Status:** development diagnosis complete; fixed P30 actor training rejected

## Decision

Do not run a longer PPO campaign on the current fixed, open-loop P30 option.
The treatment has real physical authority and mature-wave outcome diversity,
but neither instantaneous local state, causal option context, 30-second spatial
history, nor a conservative expected-value formulation identifies a policy
that generalizes to held development seeds.

This is not evidence that decentralized damping is impossible. It rejects the
current combination of third-risk-onset timing, one selected CAV, a fixed
20-second P30 ceiling, and the present actor information as a paper-ready
learned controller.

## Frozen data and integrity

The mature-wave dataset contains 178 paired decisions over mainline demands
1900, 2000 and 2100 veh/h and penetrations 5%, 10%, 15%, 20%, 25% and 30%.
Seeds 4121--4128 are the training/development split and 4129--4132 are the
one-time held development split. All remain excluded from prospective claims.

The p15 temporal replay added 30 complete causal histories, each containing 31
one-second frames ending at the declared decision. It retained exact abstain
replay, discovery and predecision identity. The three cells contained 12
helpful, 10 harmful and 8 neutral P30 effects. Every branch had zero collisions
and teleports; dose was 90%, 100% and 100%, while the last two cells had only
80% service/safety admissibility under the frozen composite constraints.

## Observation result

The existing snapshot families failed before this diagnosis: current 134-field
AUROC was 0.407 and the best 13-field option context reached 0.566 on the full
six-penetration dataset.

The new causal temporal contract contains 22 latest corridor levels, their
30-second fitted trends expressed as ten-second change, and 8 slow-front
detection/motion fields. On p15 third-onset decisions, grouped leave-one-seed-
out results were:

- current 134 fields: AUROC 0.142, balanced accuracy 0.233;
- temporal levels: AUROC 0.475, balanced accuracy 0.500;
- temporal trends: AUROC 0.550, balanced accuracy 0.575;
- all 52 temporal fields: AUROC 0.517, balanced accuracy 0.442;
- current plus temporal: AUROC 0.283, balanced accuracy 0.308.

The frozen authorization thresholds were AUROC 0.70 and balanced accuracy
0.65. Temporal state is therefore not promoted into the actor.

Artifact:
`runs/eval/idm_temporal_option_context_screen_p15_l3_v1/summary.json`.

## Conditional expected-value result

Because metastability makes individual outcomes intrinsically variable, a
second leakage-safe screen asked the easier and more relevant question:
whether activation has positive conditional expected reward in an observable
operating state. Ridge value models used paired activation reward, equal cell
weighting and causal actor inputs only. Configured demand, seed, realized dose,
future outcome and reward were forbidden inputs. Regularization and the
activation threshold were chosen only from seed-grouped out-of-fold predictions
on seeds 4121--4128 before opening seeds 4129--4132.

No feature family passed. The nontrivial option-context policy activated 16 of
58 held rows (27.6%) but selected 3 helpful, 5 harmful and 8 neutral outcomes,
with mean policy reward -0.0018. The compact zone/controller/option family
selected 3 helpful and 6 harmful outcomes and mean reward -0.0721. The only
family with no held harmful activation covered just 4 of 58 rows and failed the
20% coverage gate.

Even a deliberately nondeployable upper bound using the forbidden exact
demand/penetration cell failed to replicate. Cells selected from training had
held coverage 25.9%, 7 helpful versus 6 harmful activations and mean policy
reward -0.0325. Therefore the immediate blocker is treatment stability, not
just a missing demand estimator.

Artifact: `runs/eval/idm_expected_option_value_screen_l3_six_pen_v1/summary.json`.

## Honest limited fallback

A fixed rule that activates P30 only at 5--10% penetration performs reasonably
on the held development rows: 25.9% coverage, +0.0434 conditional reward,
7 helpful, 3 harmful and 5 neutral activations. It works in only two of six
penetration cells, uses a trivial operating-domain rule rather than learned
traffic response, and cannot satisfy the intended cross-penetration claim. It
is retained only as a comparator or a scope-reduced fallback.

## Required scientific decision

The recommended next treatment is a separately versioned closed-loop bilateral
pulse, not another P30 strength, percentile, duration, reward weight or larger
network. It should:

1. ramp slowly from native IDM for a short probe;
2. compute explicit immediate-follower/rear protection;
3. hold only while causal wave/braking response improves;
4. release early on rear compression, TTC/service deterioration, absent
   response, lane change or lost authority; and
5. return smoothly to exact native IDM.

This materially changes the decentralized treatment definition and requires
explicit approval under the master protocol before implementation. The
alternative is to accept the limited 5--10% P30 operating domain, document the
scope honestly, and move the high-authority scientific contribution to the
coordinated moving-wall escalation stage.

## Software added by this diagnosis

- `idm/temporal_option_context.py`;
- `tools/screen_idm_temporal_option_context.py`;
- `tools/screen_idm_expected_option_value.py`;
- focused unit tests for both screens.

No predictor-scale data, prospective seed, moving-wall experiment or long PPO
run was opened.
