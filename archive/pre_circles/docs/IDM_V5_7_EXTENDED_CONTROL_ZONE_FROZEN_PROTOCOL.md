# IDM v5.7 extended-control-zone frozen protocol

Status: **frozen before wall outcomes.**

> Supersession note: this design remains an unexecuted artifact. Its added
> segment is not in the declared 200--1,900 m wall band. The separately
> versioned v5.7.1 protocol uses the original corridor to isolate active-band
> distance and duration.

This is a four-seed development mechanism experiment, not RL, predictor training, validation, or a general wall claim. Only seeds 4401, 4402, 4404 and 4405 may be opened.

## Derived scenario and control arm

`scenarios/idm_v5_7_extended_zone/` contains an isolated derivative of the long IDM network. It adds one 900 m, two-lane, 31.29 m/s straight edge upstream of the original source node. All original merge/ramp/downstream edges, demand flows, vehicle types, IDM settings and 30% seeded CAV placement are retained. The damping-only arm is the frozen analytical decentralized IDM controller.

## Treatment definition

Profile: `idm_v5_7_extended_zone_low_authority_wall` / `idm_compact_moving_wall/5.7-extended-zone-low-authority`.

- A treatment-blind control scout selects the first state with complete occupied-lane, front/rear-safety-admissible CAV membership in the frozen 200--1,900 m distance-to-merge band. It does not read wall outcomes.
- Both paired branches replay from time zero. Until that exact scout step, both execute identical damping. The treatment then receives a 58 s explicit wall request; the control remains damping.
- Members are latched and same-lane handoff remains limited to one safe replacement per lane. Out-of-band CAV actions always come from the same damping policy.
- Assembly is 4--10 low-authority control steps, never exceeding 0.3 m/s2 wall-induced deceleration. The shared adaptive wall target and relative-speed servo remain safety projected at every step.
- HOLD is a requested 40 s after assembly completes. Full occupied-lane coverage and span <=50 m are mandatory in ASSEMBLY and HOLD.
- TAPER is an 8 s monotone linear blend to the already-computed damping action. No CAV is released to uncontrolled SUMO. Taper span is reported separately as intended dissolution, without hiding safety or efficacy failure.
- The terminal horizon is 178 s after the treatment-blind decision (58 s requested lifecycle + 120 s post-release observation), identical within each seed pair. Bottleneck arrival/onset/discharge and completed service are reported so upstream travel delay is not credited as improvement.

## Frozen four-case gate

Pass requires replay and predecision identity; exact out-of-band damping; >=3/4 safe complete walls; >=80% realized requested duration in feasible cases; >=80% lane coverage; <=50 m HOLD span; no new collision/teleport/unsafe TTC/excess hard braking/service loss; >=2 independent material congestion improvements of at least 10 s beyond replay noise; >=1 feasible neutral/nonbeneficial case; acceptable delayed release safety; and exact damping handoff.

If damping-only geometry removes the merge transition/headroom, no wall branch is permitted. If safe dose passes without material benefit, v5.7 is a no-go for 0.3 m/s2 authority and no additional seeds may be opened.
