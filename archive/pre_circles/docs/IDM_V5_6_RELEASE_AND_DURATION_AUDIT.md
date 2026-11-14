# IDM v5.6 Candidate C/D release and duration audit

Status: **both stopped before outcome execution: focused safety no-headroom**

Candidate C was specified as discharge-aware simultaneous release, rather than
lane staggering: lane staggering breaks the full-lane compact-wall definition
and would add a geometry/lane-change treatment.  Its fixed non-outcome rule is
two consecutive one-second samples after a five-second minimum HOLD satisfying
20-second downstream discharge >= combined mainline/ramp merge entries and
nonincreasing downstream storage, followed by the existing five-second taper.

Matched telemetry shows that rule cannot begin early in 4401 or 4402 (4402
has only one equality sample near the planned release).  Those cases therefore
remain on their known unsafe fixed-release trajectories.  Candidate C cannot
satisfy the common all-four safety gate and is stopped without redundant SUMO
outcomes.

Candidate D was specified as corridor-wave-growth termination: after a
five-second minimum HOLD, release after two consecutive samples whose trailing
five-second premerge mean speed is below its 875--879 baseline and whose speed
standard deviation is above that baseline.  It retains v5.4 authority and
taper.  It can shorten 4404, preserves 4405's full dose, but does not alter
4401 or 4402 before their unsafe divergence (both conditionally trigger only
around t=895).  It also cannot pass the common all-four gate under deterministic
replay, so it is stopped before new outcomes.
