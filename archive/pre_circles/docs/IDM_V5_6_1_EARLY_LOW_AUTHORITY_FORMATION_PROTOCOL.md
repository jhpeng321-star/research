# IDM v5.6.1 early low-authority formation protocol

Status: **treatment-blind feasibility stop; treatment implementation not authorized**

Date: 2026-09-01

## Scientific definition evaluated

v5.6.1 proposed a separately causal early decision for bounded 0.3 m/s2
position/speed alignment, followed by compact full-lane metering at t=880.
The frozen causal sequence would have been:

`DAMPING → EARLY_ASSEMBLING → COMPACT_READY → HOLDING at 880 → TAPERING → DAMPING`.

The control arm remains frozen analytical IDM disturbance-P30 damping at 2700
vph mainline, 400 vph ramp, and 30% CAV penetration.  The physical endpoint
would remain sustained premerge congestion through t=1080/natural clearance.
Treatment-specific commands before 880 would have made the earlier assembly
instant—not 880—the causal decision; no claim of identity through 880 was
permitted.

## Frozen treatment-blind scan

The predeclared absolute-clock grid was 860, 865, 870, and 875 s.  The active
metering point was fixed at 880 s.  Only four already inspected development
seeds were read: 4401, 4402, 4404, and 4405.  Every scan branch was
damping-only and ended at the t=880 state; no wall outcome, congestion label,
hard-brake label, delay, or known benefit was read to select a start.

The scanner required one continuous CAV member per occupied lane, 0.3 m/s2
kinematic projection, member identity continuity to t=880, final membership
inside the original 200--1000 m holding band, full coverage, front/rear gaps
at least 20 m, front TTC at least 3 s, rear TTC at least 4 s, final span at
most 50 m, and monotone convergence.

The assembly envelope was 70 m, derived before scanning as the 50 m final
compactness limit plus at most five seconds of the existing 4 m/s member
speed-spread envelope.  It is not a post-hoc relaxation of the 50 m active
wall definition.  A temporary wider span would have been permitted only in
EARLY_ASSEMBLING; HOLD would still require at most 50 m.

The selection rule was the latest grid time feasible in **all four** seeds.
No time passed, so `selected_assembly_start_s = null`.  This prevents a
v5.6.1 treatment contract, lifecycle implementation, or wall outcome run.

## Required implementation if a future independent scan passes

The current binary pulse is not suitable: it conflates assembly, HOLD, and
release.  A future separately approved treatment would need two clocks,
explicit EARLY_ASSEMBLING and COMPACT_READY states, non-metering early
speed-only alignment with no tau tightening, final readiness checks at 880,
phase-specific dose accounting, and exact return to damping.  None of those
scientific treatment changes are implemented by this protocol.
