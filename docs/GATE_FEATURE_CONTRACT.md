# What the RSU can observe, predict and authorize

The contribution is a decision to temporarily coordinate CAVs into one moving
wall, followed by return to the same fixed damping controller. A wall can be
considered in free, oscillating or congested traffic. No scalar global health
threshold is used to decide that a wall is needed.

## Two distinct decisions

1. **Can these particular CAVs deliver this particular maneuver?** Current
   lane membership, report age, local clearance and execution availability
   provide immediate refusal rules. Initial span, speed mismatch, target speed
   and remaining runway describe the formation challenge. Predicted delivery
   must subsequently be checked against actual assembly, continuous HOLD and
   handback; passing a geometric rule alone is not a delivery label.
2. **Would attempting that maneuver improve the outcome compared with continued
   damping?** Only matched offline experiments provide this label. Their
   physical differences include assembly failures and delayed consequences.
   The online gate receives the original CAV evidence, never the counterfactual
   simulator outcome or a privileged future rollout.

Both approvals must refer to the same current proposal, acquired evidence and
feature version. The scheduler admits one active wall, enforces a declared
action budget and cooldown, and obtains fresh approvals at later opportunities.
Without qualified artifacts it continues damping and records abstention.

## Acquired inputs

The implemented feature packet contains each proposed member's reported speed
and acceleration, speed limit, map lane and distance to the release boundary.
It records nearest front/rear bumper gaps, relative speeds and the individual
sensor's coverage, plus explicit detection-presence flags. Closing TTC is capped
at 30 seconds for the feature vector and is accompanied by detection presence;
an undetected neighbor is not represented as a proven long-TTC vehicle.

Observation age and receipt age are separate. Recent reported speeds provide
a mean and standard deviation; recent measured accelerations provide an RMS.
Issued local acceleration has its own presence flag, and native, damping and
wall ownership remain distinguishable. The initial candidate span, speed spread
and conditional assembly/runway estimates are included with availability flags.

These quantities use SI units: metres, seconds, metres/second and
metres/second². Counts describe reporting CAVs, not total traffic. Current local
damping output or explicit native fallback also establishes execution
availability; an old report does not prove that a member can act now.

The sensor model currently supplies ideal nearest-neighbor observations within
100 metres in the same lane, with 0.5-second physics and 1-second control intervals.
The current paired experiments use zero communication delay. Receipt handling
is implemented; performance under delayed, noisy or missing reports is not
therefore automatically validated.

## Boundaries and missingness

Required lanes come from the map, not the lanes that happen to be reported.
An unreported lane has unknown occupancy. Per-reporter coverage is not summed
into a claimed coverage union. Overlapping detections are not silently
deduplicated into a complete HDV census.

True penetration, demand designation, global density/health, hidden driver
parameters, episode/seed identity and future traffic are excluded from the
numeric feature vector. Seed, case, decision time and source/proposal hashes
are stored separately to audit labels and splits. They must not enter training
as model inputs. The number of reporting CAVs is legitimately observed even
though its relationship to the unobserved penetration varies.

## Physical labels for the registered natural panel

All costs use the full exogenous requested population as denominator. A useful
attempt must reduce total service by at least 1 second/request **or** severe
braking exposure by at least 0.02 vehicle-seconds/request. The maximum permitted
increases are 0.5 seconds/request of total service, 0.1 of source waiting,
0.005 vehicle-seconds/request of severe braking, 0.5 m²/s³/request of squared
acceleration effort, and 0.02 vehicle-seconds/request of TTC below 3 s exposure.
Treatment collisions, teleports or incomplete service preclude a beneficial label.
These are declared research margins, not universal safety standards.

Acceleration effort is the integral of measured acceleration squared, not fuel
energy. Braking measures time with measured acceleration below −3 m/s².
All attempted assembly, HOLD, taper and post-release consequences remain in
the 2,400-second outcome comparison. A refused proposal is retained as a decision
outcome but must not become an observed negative example of an unattempted wall.

## What remains to be learned

There is no trained benefit gate or calibrated feasibility model yet. The
implementation supports a qualified analytical feasibility rule, transparent
conjunction rules and small logistic models. Artifact integrity checks do not
prove statistical validity. Physical headroom, observation/label audits and
entire-seed training/calibration/prospective splits must pass first.

The current eight natural wall opportunities share only two exposed seeds.
They can diagnose mechanics and reveal possible effects; they cannot establish
calibration, generalization or a reliable low false-escalation rate. Existing
sealed seed blocks remain sealed. Evidence that CAV features are informative
must come from held-out prediction and physical decision value, rather than
from naming a feature "health" or reporting training accuracy.

The completed panel has five delivered walls and no attempt satisfying every
registered benefit/safety condition. Dataset v2 retains eight rows with 65
numeric CAV features and excludes one prior-safety-event row from future binary
benefit fitting: the whole-horizon zero-event condition had already failed
before its action. The registered rejection and all physical outcomes remain
unchanged. Prior/postdecision event accounting is offline attribution metadata,
never an online feature. Seven target-eligible rows with zero positives do not
establish trainability, calibration or observation sufficiency. No model is fitted.

