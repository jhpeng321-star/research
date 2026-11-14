# IDM v5.5.1 adaptive hold/release protocol

Status: **frozen before v5.5.1 SUMO outcomes**. This is a development-only
mechanism repair, not predictor evidence or a prospective validation.

## Question and frozen treatment

The v5.4 causal diagnosis ruled out replay mismatch and initial assembly as the
dominant source of treatment-only hard braking. Its supported failure mechanism
is accumulated follower compression during HOLD and the subsequent
taper/release discharge. Version 5.5.1 changes only that lifecycle law.

Both arms execute the identical frozen analytical IDM P30 decentralized damping
controller at every step. The treatment adds one compact, full-occupied-lane
moving wall at absolute SUMO time 880 s, with the existing one-member-per-lane
selection, same-lane handoff, activation projection, 2700/400 demand, 30% CAV
penetration, and 1080 s common endpoint. It never uses an uncontrolled SUMO
handoff. No density, timing, geometry, wall count, endpoint, seed partition,
or primary congestion definition changes.

The maximum decision window remains 20 s (880--900); it is a maximum policy
window, not a required 20 s full-authority hold. The state machine is
ASSEMBLING -> HOLDING -> TAPERING -> exact damping. A hard emergency abort is
separate from a safety-triggered planned taper.

## Frozen v5.5.1 law and rationale

Every HOLD step reads the first three same-lane followers of each wall member,
up to the existing 100 m influence-zone bound. No new RSU/CAV observation is
introduced. The direct follower is identity-latched on first observation. A
replacement or disappearance of a previously latched follower causes a safe
release; it is never silently treated as the same vehicle trajectory.

The soft compression predicate applies only to the immediate, identity-latched
follower. It is the disjunction of:

- follower acceleration <= -3 m/s2;
- a rear-TTC projection clip/conflict; or
- a risk-gated compression trend: rear gap <= 20 m (the v5.2 activation
  floor) **or** rear TTC <= 6 s (the existing rear-aware feedback guard),
  and, for that same follower since HOLD entry, rear-gap loss > 3.5 m or
  closing-speed increase > 1 m/s.

A previously latched direct follower that changes or becomes unavailable starts
the normal safe taper immediately. The second and third chain followers remain
on the existing hard-safety layer only; they do not create a broad new soft
trigger merely by entering the chain.

The risk-gated predicate uses the one-sample existing rear-aware response rule.
On that sample the controller cannot tighten either speed or tau authority and
latches an irreversible linear five-second release, using the existing v5.3/v5.4 taper slope and the
existing 1 m/s2 wall command envelope. Release targets are a monotone blend to
the per-step frozen damping target; tau is monotonically removed. The wall can
never return to a stronger command after this latch.

Hard aborts remain unchanged: collision, teleport, controlled or follower
hard braking <= -4.5 m/s2, or an influence-zone TTC < 1.5 s. A hard abort
immediately returns exact damping and is not a completed adaptive lifecycle.

The threshold values are inherited safety quantities, not optimized against
4401, 4402, 4404, or 4405. They are an engineering guard, not a universal
literature-validated follower-compression law.

## Completion and dose

Policy completion is separate from physical dose. An adaptive soft-release
that reaches exact damping is policy-complete, but reports its actual assembly,
hold, taper, controlled-CAV-seconds, speed/tau/acceleration authority, and
coverage. It must never be labelled a 20 s hold unless it actually held for
20 s. Hard abort, incomplete release, loss of required active-lane coverage,
or no meaningful applied authority are reported separately.

Each active step records the follower-chain trajectory, minimum rear gap/TTC,
maximum closing speed, braking follower count, member changes, phase
compactness, projection survival, control deltas, and base-action handoff.
Out-of-band CAV actions remain the exact damping actions.

## Tests and four-case gate

The focused controller tests and the full repository suite must pass before
SUMO. The only v5.5.1 outcome gate is paired control/treatment on previously
inspected development seeds 4401, 4402, 4404, and 4405. The gate passes only
if all four treatments are safety- and service-admissible; 4402 and 4405 retain
physical materiality beyond replay noise; 4401 and 4404 create no material
congestion, safety, or service harm; and the v5.5.1 telemetry shows less
follower compression or release-wave severity than v5.4. Every case, early
release, low-authority treatment, and harmful result remains reported.

Failure stops expansion and predictor work. A pass permits only the separately
frozen fresh 4407--4430 feasibility/headroom screen.

## Fingerprint

The machine-readable treatment fingerprint is
`experiments/idm/wall_v5_5_1_contract.py:profile_manifest()`. It differs from
v5.4 through the explicit follower-aware adaptive-release law and its frozen
thresholds.
