# S-CAPREL — owner-approved CPU build, 2026-10-30 14:00
Shared-function SHA: `37bc2a57176ccb391c4c7755a284e32e6d834632` (S-DISTILL2 import ready).
Base: `9613f956`; branch `c4/caprel`; SUMO parity re-check pending registration.

API: `wms.control.residual.executed_from_residual(state, r, law)`.
State: WM frame, or `{v_mps, base_mps2, cap_mps2}`; scalar or tensor.
Law: `"base"` (default/OFF), `"ctrl0"`, or mapping with residual_reference/lo/hi/dt/accel_max.
CTRL0=max(-v/dt,min(b,c)); ON=max(-v/dt,min(clip(CTRL0+r,min(lo,CTRL0),max(min(hi,1.2),CTRL0)),c)).
No behavioural thresholds or driver settings are introduced.
