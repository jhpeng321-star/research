# IDM v5.7.1 original-corridor expanded-control-zone protocol

Status: **frozen before baseline or wall execution.**

The prior v5.7 extended-network design is preserved unexecuted. Its 900 m extension was outside the 200--1,900 m eligible wall region, so it would have altered upstream storage and arrival timing without testing a longer active wall-control zone. It is not referenced by this protocol or runner.

## Fixed experiment

This correction uses the verified original long IDM corridor and exactly four development seeds: 4401, 4402, 4404 and 4405. Mainline demand is 2700 vph, ramp demand 400 vph, CAV penetration 30%, decision time exactly t=880 s, and the control arm remains frozen decentralized IDM damping. Merge geometry, network, vehicle/IDM/lane-change parameters and route assets are unchanged.

At t=880 only, treatment-blind selection checks the 200--1,900 m band for full occupied-lane, trajectory-coherent, front/rear-safe membership and life-cycle runway. Infeasible seeds are reported; no timing scan, replacement decision, density change or safety relaxation is allowed.

The wall is all-low-authority: 0.3 m/s2 maximum wall-induced deceleration in assembly, HOLD and taper. Assembly is 4--10 seconds, HOLD is 40 seconds, taper is an 8-second monotone blend to the already-computed damping command, and the maximum lifecycle is 58 seconds. Span <=50 m and full lane coverage are mandatory in ASSEMBLY and HOLD. Members are latched, may have at most one safe same-lane handoff, and all out-of-band CAVs retain exact damping.

Each pair runs through 178 seconds after t=880, including 120 seconds after the requested wall lifecycle. The runner reports congestion intervals in controller seconds, safety/service indicators, member/treatment dose, handoffs and exact release-to-damping evidence.

## Gates

Gate A runs two damping-only replays per seed. It requires exact replay, complete horizons, the original corridor only, t=880, and a congestion opportunity in at least three seeds. Gate B consumes the Gate-A decision rows only and requires >=3 feasible formations. Gate C then runs paired branches only for feasible approved seeds.

The final four-case result requires all machine-readable integrity and safety checks, >=3 safe complete walls, >=80% dose and coverage, HOLD span <=50 m, service noninferiority, two independent >=10-second improvements exceeding replay noise, and at least one feasible nonbeneficial case. Any safety, service, compactness, dose, or benefit failure is a no-go; it does not open more seeds.
