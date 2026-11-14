# Gentle report-aware wall — one bounded development version

Frozen design before new outcomes: `gentle_report_wall/1`. Voluntary absolute
wall commands are in [-0.3, +1.5] m/s². CIRCLES and its adapter remain unchanged.
One command ramp is 0.15 m/s³ on the existing one-second control clock: two
updates from zero to full gentle braking, twelve from -0.3 to +1.5. This is a
design choice for gradual control, not a fitted optimum or measured jerk promise.

Every assembly, HOLD and wall-owned release request passes the same final
resolver. Its feasible interval intersects absolute wall bounds, previous wall
command +/-0.15, current acquired acceleration +/-0.15 and the speed target
bounds actually used by CommandOwnership. At entry, acquired measured
acceleration anchors the transition. Native IDM never gets an invented numeric
reference. Empty intersections or failed projected front/rear clearance abort
the entire cohort and resume the identical damping/native path. Protective
SUMO responses and damping handback may brake harder; these remain separate.
The clearance check includes both the actual next command and subsequent
ramped closure to the detected leader's reported speed. It refuses if the
available bumper gap cannot accommodate that conditional braking distance;
one-second TTC alone does not establish sufficient gentle stopping distance.

The cohort, t600 opportunity, mean-speed-minus-2 capped at 24 m/s target,
50-second assembly deadline, three readiness samples, 20-second compact HOLD,
10-second maximum release and map boundary are unchanged. The historical
FormationProfile/servo is preserved. A new wall-only subclass binds the gentle
authority and ramp into the actual proposal. Existing feasibility estimates now
receive 0.3 deceleration; additional ramp-integrated speed adjustment time and
distance are recorded. Estimates do not certify future geometry or extend time.

At release, reuse fresh downstream CAV self reports and the existing length,
gap and receipt-age reserve. Integrate maximal gentle braking under the fixed
ramp to the reported downstream speed, including its movement. If acquired
distance cannot accommodate that reduction, return to damping explicitly. If
compatible, continue a bounded report-aware wall transition. With numeric
local feedback, ramp toward it, or immediately hand back stronger protective
braking. Without a slowing reported constraint, ramp to explicit wall coasting
then release; zero wall acceleration is never labelled native. Missing evidence,
unsafe geometry and the original deadline remain terminal conditions.

Only timestamped acquired CAV packets and actual local damping references enter
runtime decisions. Global trajectories are offline diagnostics only. Preserve
requested servo acceleration, resolved wall request, issued acceleration and
measured response separately. Compare followers as well as members; report
peak deceleration, 0.5/1-second acceleration differences, exposures below -0.3,
-1, -2 and -3 m/s² and integrated squared acceleration (not fuel consumption).

Physical budget: first two deterministic controlled fixtures, each 120 seconds
including five-second warmup, zero retries; stop on implementation failure.
After both pass, exactly seeds 4107 and 4108 at M2700/p20/t600 for 2400 seconds
each (4800 total), reusing verified damping with source/scenario identity and
exact predecision state/report/command prefixes. Preserve original and
report-dissolution comparators. No sweeps, replacement seeds, automatic retry,
successor variant, or predictor training. Each allocation is consumed before
launch; failed trajectories remain evidence.

Unchanged benefit requirements: service gain >=1 second/request OR severe
braking gain >=0.02 vehicle-seconds/request; maximum cost increases service 0.5,
source waiting 0.1, severe braking 0.005, effort 0.5, TTC<3 exposure 0.02, each
per request. Complete all requested service in both arms and zero treatment
collisions/teleports. Preserve predecision event attribution and failed assembly.
These two exposed cases cannot establish generalization or justify training.
