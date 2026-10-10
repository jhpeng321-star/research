# Registered hold-extension check: raise-26 HOLD 20 -> 30 seconds

Status: registered bounded physical block implementing the selected Option B
proposal ([NEXT_BLOCK_PROPOSAL_V2](../output/joint_design_v1/NEXT_BLOCK_PROPOSAL_V2.md))
with the [proposal corrections](../output/hold_extend_v1/PROPOSAL_CORRECTIONS.md).
This document freezes the hypothesis, treatment, budget and stopping rules
**before** any new simulation. It authorizes no training, extra seeds,
threshold search, further duration variants or successor campaign.

## Question (two separate comparisons)

- **A (continuation effect):** from the verified raise-26 history, does
  extending HOLD from 20 s to 30 s improve the same-prefix outcome?
  Comparator: the existing raise-26 H20 arms (target_adjust_v3).
- **B (complete escalation):** does the H30 escalation qualify against
  matched damping under the unchanged original benefit criteria, with the
  strict-joint diagnostic reported separately?

## Treatment (exact)

1. Original entry, cohort and assembly (unchanged; byte-equivalent before
   the HOLD+5 adjust branch).
2. Verified target change 24 -> 26 m/s at measured HOLD+5 (the delivered
   hold_target_adjust/2 interface).
3. At measured HOLD+20 — before the original taper command is issued — one
   registered +10 s extension request. Feasibility is recomputed from
   acquired reports; feasible -> apply exactly once (HOLD continues at
   26 m/s, nominal HOLD 30 s); infeasible -> decline and run the original
   H20 schedule (recorded physical outcome).
4. Unchanged coast exit and identical-CIRCLES handback. No second extension,
   speed change, replacement cohort or re-engagement.

Source-derived expectations (verified from the raise-26 registrations;
runtime still uses measured readiness and rejects unexpected history):
4107 HOLD start 612 s, extension decision 632 s; 4108 HOLD start 618 s,
extension decision 638 s.

**Pre-block offline check (passed):** replaying each source arm's acquired
reports at the extension tick through the unchanged remaining-maneuver plan
under the extension profile: 4107 feasible (min traffic margin 33.4 m),
4108 feasible (243.5 m). Thin 4107 margins mean a per-tick protective
decline remains a possible physical outcome.

## References (reused, not rerun)

- Primary same-prefix: `s4107/s4108_m2700_p20_t600_hold_adjust_up_wall`
  (target_adjust_v3, raise-26 H20).
- Whole-policy: matched damping arms (observer_replay_v1).
- keep/lower/release: context only. Reruns only if an integrity mismatch
  invalidates reuse — in that case stop and report within budget.

## Mechanics prerequisite (120 s, charged)

Isolated fixture reproducing: entry/assembly, raise to 26 at HOLD+5,
extension at HOLD+20, total 30 s compact HOLD at 26 m/s, unchanged coast
exit, identical-CIRCLES handback, command contract, zero collisions and
teleports, full 120 s horizon. An implementation or verification failure
stops corridor execution; the verifier is never edited after a failed
fixture to declare the gate passed.

## Corridor allocation (2 x 2400 s, charged)

`s4107_m2700_p20_t600_hold26_ext10_wall`,
`s4108_m2700_p20_t600_hold26_ext10_wall` under a new registration with exact
source/model/adapter/scenario/profile identities, seed roles, reference
paths, prefix-verification requirements (intervals/traffic/policy/telemetry
and predecision snapshot fields identical to the raise-26 sources strictly
before 632/638 s; divergence at the branch is intentional), original
criteria, 2400 s accounting and zero retries. Parallel execution after
source freeze. Total new charged allocation: **4920 s**.

## Invariants preserved

1 s control / 0.5 s physics; frozen sensing/report contract; CommandOwnership
sole longitudinal writer; −0.3 m/s² voluntary wall braking; 0.15 m/s³
command-change bound; HDVs receive no commands; unchanged protective exits;
identical damping handback; issued commands distinguished from measured
responses; acceptance data sealed.
