# IDM v5.6.2 phase-split-authority protocol

Status: **frozen executable development-only treatment definition**

v5.6.2 keeps the frozen t=880 decision, compact two-lane geometry, 20-second
window, five-second taper, IDM damping control and t=1080 endpoint. It differs
only in the separately versioned authority lifecycle below. There is no altered
member rule, support CAV, partial-lane wall, reward, predictor, endpoint, or
fresh-seed opening.

The prior adaptive path used `compact_servo_a_max_dec_mps2` for projection,
assembly, HOLD and taper; `compact_assembly_a_max_dec_mps2` was ignored.  The
new explicit `compact_initial_assembly_servo_a_max_dec_mps2` is therefore the
only high-authority field and is logged separately.  Projection and runtime
now share that one-step authority cap, and a candidate that cannot become
compact after the projected command is vetoed before treatment.  Authority is
monotone from 1.0 to 0.3 and never reverses.

## Frozen lifecycle and transition

`DAMPING → ASSEMBLING_HIGH_AUTHORITY → SETTLING_TO_LOW_AUTHORITY →
HOLDING_LOW_AUTHORITY → TAPERING_LOW_AUTHORITY → DAMPING`

1. **ASSEMBLING** issues at most one command with a 1.0 m/s² deceleration cap.
   It must reach the existing full-lane, 50 m compactness condition. Failure
   returns immediately to exact damping.
2. **SETTLING** is a separate two-command maximum phase inside the existing
   20-second window. It reads each member's actual SUMO acceleration after the
   assembly command. The next per-member target acceleration is no more
   restrictive than `min(-0.3, a_previous + 0.5 * dt)` when the prior response
   is below the low-authority envelope; a member already inside the envelope
   is not tightened. Thus for 1 Hz and a realised -1.0 m/s² assembly response,
   the nominal values are -0.5 then -0.3 m/s².
3. The 0.5 m/s³ wall-specific jerk bound is newly frozen from the analytical
   IDM controller's configured jerk scale; it was not an earlier wall bound.
   It is logged separately from assembly and HOLD dose.
4. During SETTLING, speed targets never become more restrictive and the wall
   restores the pre-settling tau fraction after each actuator call, preventing
   any tau/headway tightening. Loss of compactness, full coverage, member
   identity, front/rear safety, or realised acceleration telemetry is an
   invalid lifecycle and a hard handoff to exact damping.
5. Only after every member's realised acceleration is inside the -0.3 m/s²
   envelope can **HOLDING** begin. HOLD and the five-second monotone taper
   never use more than 0.3 m/s² authority. Hard safety always overrides
   SETTLING; it never waits for the jerk schedule.

The four-case gate uses 4401, 4402, 4404, and 4405 only. It requires exact
replay/prefix identity, one high-authority step at most, compact/full-lane
readiness, settling jerk at most 0.5 m/s³, at most two settling commands,
0.3 m/s² HOLD+taper, no lifecycle abort/collision/teleport, all-case
safety/service admission, retained 4402/4405 benefit, no 4401/4404 harm,
nontrivial dose, exact damping handoff and full delayed-safety observation.
Phase-specific diagnostics record actual/target acceleration, jerk, tau freeze,
controlled CAV-seconds, coverage, compactness, identity changes, duration and
abort reason. Failure opens no fresh data and authorizes no predictor/RL work.
