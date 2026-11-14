# IDM v5.6 Candidate A low-authority audit

Status: **entry gate passed; four-case screen required**

The generic `WallHoldPolicy` default deceleration cap is 0.3 m/s2, but the
compact adaptive-servo overlay temporarily assigns
`compact_servo_a_max_dec_mps2=1.0` before `act_with_targets()` during
assembly/HOLD and taper.  The predecision adaptive projection also uses the
servo cap.  Thus `wall_params.a_max_dec=0.3` did not constrain the v5.4 or
v5.5.1 compact wall.

From all 142 active v5.5.1 member-command samples, 48 were decelerating.  All
48 exceeded 0.3 m/s2; 40/142 active samples (28.2%, and 83.3% of deceleration
commands) were at least 0.95 m/s2.  By seed, mean deceleration was 0.933,
0.995, 0.934, and 0.940 m/s2 for 4401, 4402, 4404, and 4405.  A 0.3 cap is
therefore a material action change, not a cosmetic parameter change.

A source-compatible conservative reconstruction predicts all four formations
can reach the existing 2.0 m/s speed-spread tolerance under 0.3 m/s2; 4402
needs approximately two seconds rather than one.  This is mechanical
plausibility, not outcome evidence.  Candidate A must set both the runtime
servo cap and projection cap to 0.3 m/s2; changing only one would violate the
feasibility contract.
