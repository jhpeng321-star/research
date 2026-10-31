# SUMO merge/insertion source receipt

Target tag: **v1_27_1**. Prior session downloaded text for reading only into the
new empty directory `E:/Temp/opencode/rb-merge-sumo-1271-20261031`.
No downloaded code was executed. Public source root:
https://github.com/eclipse-sumo/sumo/tree/v1_27_1/src/microsim

- `cfmodels/MSCFModel_IDM.cpp`: 44 (stepping .25 s), 138–160
  (recursive insertionFollowSpeed, convergence ACCEL2SPEED(1)), 179–199
  (stopSpeed), 218–220 (secure gap), 224–268 (iterated IDM).
- `lcmodels/MSLCM_LC2013.cpp`: 48–76 (physics constants), 108–122
  (population LC defaults), 234–343 (speed advice composition), 470–659
  (leader negotiation), 660–900 (follower negotiation), 1233–1245
  (strategic lookahead), 1294 (usable distance).
- `MSLane.cpp`: 606–660 (departSpeed), 852–1469 (insertion checks),
  especially 1435–1469 (subtract minGap before insertionFollowSpeed).
- Shared helper artifacts consumed as they appeared:
  `E:/agent-work/rb-shared/mergespec/{spec.md,test_cases.md}` and its
  source excerpt `MSBaseVehicle.cpp:1415-1416` (base front = length + POSITION_EPS).
  Corrected urgency (remaining time divided by 2, floored at one step),
  follower gap units, fall-behind branch, last/desired speed refusal and single
  chosen depart lane. The helper's `POSITION_EPS=.01` statement was subsequently
  corrected by INT's source reconciliation `f2d9a3ff`: SUMO v1_27_1
  `src/config.h.cmake:236/239` has NUMERICAL_EPS=.001 and POSITION_EPS=.1.
  SPEED_EPS=.01 is a distinct constant. The earlier helper-based result is retained.
  Unit vector T3 (IDM insertion at net gap 2m) independently compared.
- Later leader-advice audit: `MSCFModel.cpp:87-108` gives discrete Euler brakeGap
  with headway; `MSLCM_LC2013.cpp:547-557` requires time as well as space to
  overtake; `:637-655` adds follow-speed advice even when the neighbor is not
  currently blocking. Lookahead/urgency reuse the lane stream's persistent state.

## Geometry and right of way

Repository network `scenarios/idm_v5_7_extended_zone/osm.v5_7.extended.net.xml`
lines 61–81: ramp connection to acceleration lane has state M. Mainline connects
to separate lanes. The acceleration lane has no outgoing link at 75.44 m.
There is **no conflicting minor-priority ramp junction** in this network.
The relevant right-of-way obligation is the lateral changer's two-sided gap
acceptance. Do not invent a junction yield deceleration on the mainline.

Speed-limit audit: the same network's line22 sets the ramp APPROACH edge to
22.22m/s; line45 sets the acceleration lane (s>=0) to31.29m/s. Thus lane2 does
NOT have one constant limit along its whole route. `merge_network.py` encodes
this observable geometry. `driver_encoder.py:23-24,66-67` defines physical v0
relative to the reference road limit, so existing inferred speedFactor is
v0/referenceLimit. SUMO IDM desired speed and departSpeed=max use the local
lane limit times that factor. Never clip every driver to the unscaled posted
limit; equally, never accelerate along the22.22 approach toward31.29 by default.
These are source/network corrections, not estimates selected against FIT errors.

## Native-CAV background contract

SUMO native motion is its CF/LC pipeline when no external speed command is
issued. The inherited WM native-CAV head is a frozen approximation, not that
pipeline. CPU reproduction from the FIT969400 observed ramp-head trace shows a
packing-dependent false standstill: `control/native_cav.py:65` returns unmasked
dv from an argmin fallback index0 when has_leader=false, and head_features:80
still passes dv to the head. An unrelated slot0 mainline speed can therefore
change an unled ramp CAV's acceleration. No head was trained or modified here.
The new opt-in sumo_native callback consumes the model's existing inferred/
population-physics execution ceiling using the same project_native bounds,
and delegates lanes/merges. This is a native-arm source-mechanics diagnostic;
old frozen evaluation/controller bindings must remain distinguishable.

## Scope and explicit parity gaps

This implements a source-backed highway subset, not full SUMO state parity.
No roundabouts, pedestrians, signals, opposite driving, continuous lateral
motion, counter-lane blocker reservation, route divergence, or internal SUMO
waiting flags. Strategic lookahead reuses persistent LC2013 memory when supplied;
standalone callers use current speed as a cold-start fallback. Follower advice uses the Euler branch;
WM longitudinal integration remains its configured trapezoid/stop-time rule.
Lane stream supplies mainline LC2013 decisions; absent that, MOBIL physics is
the default mainline proposal. These limitations must accompany efficacy claims.
