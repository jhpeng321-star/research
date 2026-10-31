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

## Geometry and right of way

Repository network `scenarios/idm_v5_7_extended_zone/osm.v5_7.extended.net.xml`
lines 61–81: ramp connection to acceleration lane has state M. Mainline connects
to separate lanes. The acceleration lane has no outgoing link at 75.44 m.
There is **no conflicting minor-priority ramp junction** in this network.
The relevant right-of-way obligation is the lateral changer's two-sided gap
acceptance. Do not invent a junction yield deceleration on the mainline.

## Scope and explicit parity gaps

This implements a source-backed highway subset, not full SUMO state parity.
No roundabouts, pedestrians, signals, opposite driving, continuous lateral
motion, counter-lane blocker reservation, route divergence, or internal SUMO
waiting flags. Strategic lookahead currently uses instantaneous speed rather
than SUMO's persistent decaying maximum. Follower advice uses the Euler branch;
WM longitudinal integration remains its configured trapezoid/stop-time rule.
Lane stream supplies mainline LC2013 decisions; absent that, MOBIL physics is
the default mainline proposal. These limitations must accompany efficacy claims.
