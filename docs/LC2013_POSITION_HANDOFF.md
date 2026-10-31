# CF / INT position-semantics evidence from S-RB-LANE

**Observed SUMO tapes use semi-implicit Euler displacement, `s_next=s+v_next*dt`.**
This is relevant to free-running WM-PHYS, not a change to the lane diagnostic's
recorded-position teacher forcing.

- Read-only public SUMO1.27.1 `MSVehicle.cpp:4954–4967` (`updateState`): when
  `gSemiImplicitEulerUpdate`, `deltaPos=SPEED2DIST(vNext)`; otherwise ballistic
  `getDeltaPos(...)`. No downloaded source was executed.
- Actual FIT969400–969431,160episodes,240..360s:2,346,379 changed-positive-speed
  HDV mainline intervals with stable identity/lane/normal edge. No LC-origin
  jumps, junction crossings or stopped/negative-speed ballistic exceptions.
- Euler displacement MAE **0.000034906m**, RMSE **0.000055810m** (float32
  observation quantization). Trapezoid displacement MAE **0.039019m**, RMSE
  **0.065423m**. Acceleration-change-weighted RMSE0.00007391 vs0.166399m.
- Audit: `tools/rblane/receipts/POSITION_INPUT_AUDIT.json`; generator
  `python -m tools.rblane.diagnose_positions`. Observable input arrays only;
  no SUMO, new WM inference, truth/assignment files or parameter fitting.

At inspection, `c4-rb-int/src/wms/model/wms.py:1854` still has the trapezoid
`s+0.5*(v+next_v)*dt`; `c4-rb-cf/src/wms/physics/idm_sumo.py:41–44` incorrectly
calls ballistic displacement the SUMO default. A velocity-kernel parity gate
does not establish plant/position parity.

**CF / INT integration action:** use the source Euler displacement for the
explicit SUMO physics mode, and align executable CAV/source-entry plant updates
to the same mode. Preserve legacy/checkpoint behavior behind default-OFF
selection. Test positive-speed transitions and actual speed-command execution;
the ballistic stop-time branch is not the default recorded plant.

LANE has not edited another workstream's files or silently changed the legacy
position integrator. This evidence alone does not quantify a long-horizon
delay/breakdown gain; the integrated distributional gate remains required.
