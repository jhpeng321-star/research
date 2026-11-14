# IDM wall v5.4 emergency-handoff support protocol

Status: **owner-authorized and frozen before v5.4 outcomes are opened.**

Date: 2026-08-31

## Scientific change

v5.3 failed its prospective integrity gate because a hard safety abort could
interrupt an in-progress five-second taper without recording the resulting
zero-authority handoff. v5.4 changes only that transition contract. When a
frozen hard safety abort fires during TAPERING, the controller applies no wall
command on that step, returns the identical damping action, restores wall tau,
and records taper blend zero plus the abort reason. It does not continue unsafe
wall authority merely to finish a nominal taper.

All geometry, member selection, reserves, handoffs, compactness enforcement,
wall targets, outcome budgets, analytical damping and feature definitions are
unchanged from v5.3. v5.3 artifacts and its no-go are not reclassified.

## Fresh cohort and fixed operating point

- IDM demand: 2700 veh/h mainline, 400 veh/h ramp.
- CAV penetration: 25%, frozen from the treatment-blind v5.3 scan.
- Fresh support seeds: **4333--4352**.
- Decision steps: 620, 660, 680 and 700.
- Common terminal controller step: 960; paired outcome horizon: 260 steps.
- Seeds 4401--4520 remain unopened predictor-development candidates.
- Seeds 4601--4660 remain unopened validation candidates.
- Seeds 5001--5060 remain sealed final-paper seeds.

## Stage R0: fixed-p25 treatment-blind availability

Run damping only. Reconstruct v5.4 feasibility at every frozen decision without
applying the wall or writing treatment outcomes. Proceed only if at least 14 of
20 seeds, 20 windows and two decision times are admitted. No penetration search
or outcome-dependent cohort selection is allowed.

## Stage R1: paired support

For all 80 seed/windows, store a refusal for deterministic physical vetoes and
open paired damping/v5.4 branches only for admitted starts. Repeat the first two
admitted starts exactly. The support pilot passes only if all of the following
are true:

- exact damping and treatment repeats;
- exact identity through the decision and identical first-request base actions;
- wall authority only for in-band members during treatment;
- identical out-of-band and post-release damping;
- nominal taper reaches zero, or a hard safety abort records the explicit zero
  damping handoff on the abort step;
- no partial-lane HOLD and no silently held compactness failure;
- zero excess collisions or teleports and effects above replay noise;
- at least 80% compact-composite dose success;
- dose-qualified, refusal/dose-failure, safe, unsafe, joint-positive and
  joint-negative support each satisfy the same independent-seed minima frozen
  for v5.3 (six dose-qualified seeds; three seeds for every remaining class).

Raw attempted-wall safety is reported but is not a support pass criterion. A
later trained selective gate must reach at least 90% safety among its chosen
activations on untouched validation seeds.

For the p30 amendment, Gate B uses the separately versioned safety-first label
`idm_escalation_gate_b_safety_service/1`: safety or discharge improvement can
define benefit, delay is a noninferiority guardrail only, and queue area never
selects treatment. This label was frozen before any v5.4 outcome was opened.

If any integrity or support gate fails, stop before predictor-scale generation.
If all pass, the wall/profile hash and label contract are frozen and development
generation on seeds 4401--4520 is authorized; model fitting remains separately
gated by development class counts.
