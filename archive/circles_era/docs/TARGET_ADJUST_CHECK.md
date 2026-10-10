# Registered continuation target-adjustment check

Status: registered bounded physical block under the 2026-09-07 owner task
"can CAV-acquired information identify which moving-wall action improves
outcomes over damping". This document freezes the question, actions and
budget before any new simulation. It authorizes no sweep, retry, extra seed,
duration/braking factorial, map change, training or acceptance use.

## Question (decision B — continuation)

From the same executed coordination, does the maintained target value change
full-horizon outcomes when entry, cohort, assembly, the original 20 s HOLD and
the finite exit law are all held fixed? This isolates the continuation action
that the earlier contrasts could not: the entry-target comparison (24 vs
25.68/27.43) changed assembly and handback timing together with maintained
speed (see CORRECTIONS.md C3), and the duration comparison changed HOLD length
at fixed target.

## Why ±2 m/s, frozen before outcomes

- ±2 m/s is the formation's own design magnitude (target_speed_reduction_mps).
- Reachability: ±2 m/s within the remaining 15 s of HOLD is feasible under the
  −0.3 m/s² absolute bound (≥6.7 s) and the resolver's 0.15 m/s³ slew clipping;
  the down direction raises distance margins, the up direction consumes ≈32 m
  of the ≥129.7 m (4107) and ≥264.1 m (4108) minimum reported traffic margins
  at the branch ticks (age-0 acquired, Part 3).
- Both adjusted targets stay within the allowed-speed bound and below/around
  ambient member speeds; symmetry avoids a one-sided search.
- Selection used only acquired information, the registered contract and the
  verified source prefixes. No outcome of any adjusted arm was seen.

## Arms

Branch at measured HOLD start + 5 s (4107: 617 s; 4108: 623 s), verified
against the immutable source formation files (612/618 s). Adjustment replaces
the proposal's maintained target (24 → 22 or 26 m/s); the original scheduled
coordination end (HOLD start + 20 s) and the unchanged coast exit and handback
follow. If the adjusted commitment is infeasible, the lifecycle's own early
exit is the recorded outcome; no replacement value is searched.

- `s4107_m2700_p20_t600_hold_adjust_down_wall` (22 m/s) / `..._up_wall` (26)
- `s4108_m2700_p20_t600_hold_adjust_down_wall` / `..._up_wall`

Reused references: the unchanged-target H20 continuations
(`s4107_..._lifecycle_v2_wall`, `s4108_..._cohort_v1_wall`) and the verified
observer-replay damping arms. Exact pre-branch identity required in intervals,
traffic, policy, telemetry (< 617/623 s) and the predecision snapshot fields.

## Mechanics prerequisite (120 s, charged)

One fixture: complete maneuver at target 24, branch 5 s into HOLD, adjust to
22 m/s. Must show: resolver-slew-compliant transition, adjusted target reached
before the scheduled taper, full 20 s HOLD preserved, coast exit and identical
handback, zero collisions/teleports. Failure stops the block; the allocation
stays charged.

## Budget

Exactly 120 + 4×2400 = 9720 simulated seconds of the 9840-second ceiling
(recreated prefixes are inside each episode's 2400 s). Two episode slots stay
unallocated. Independent arms may run in parallel; active code is frozen
while episodes run. Zero automatic retries.

## Stopping rules

Stop on any command, ownership, sensing, prefix, receipt or contract defect;
preserve the charged failure. An early exit caused by adjusted-target
infeasibility is a recorded outcome, not a defect to retry. No replication,
predictor fitting or prospective/acceptance evaluation follows automatically.
With two exposed seed groups, all comparisons remain exploratory; the strict
joint diagnostic never replaces the original labels.
