# IDM wall v5.3 mechanism-repair protocol

Status: **owner-approved frozen next treatment version (2026-08-30), scoped to
treatment-integrity repairs. A taper-only safety-efficacy claim is rejected.**

Date: 2026-08-30

## Post-draft diagnostic decision

The existing v5.2 artifacts were subsequently analyzed with release-aligned
hard-brake onsets. All five unsafe cases show delayed downstream divergence:
only one positive excess onset occurs in the first 20 seconds after release,
while 248 occur more than 60 seconds later. Safe and unsafe terminal wall
mechanics also overlap on every inspected metric. See
`IDM_WALL_V5_2_RELEASE_SAFETY_DIAGNOSIS.md`.

Consequently, the same-lane coverage, compactness, and fixed taper mechanics
below are adopted only as treatment-integrity repairs. The five-second taper is
not a supported explanation or repair for the delayed downstream safety
failures. Conditional safety-and-benefit selection, under the separately
approved two-stage estimand, addresses that heterogeneity.

## Purpose

Version 5.2 established that a compact, full-occupied-lane moving wall can
produce material and heterogeneous effects relative to frozen analytical IDM
damping. It did not establish a safe treatment: only 5 of 10 dose-qualified
activations passed the predeclared full-horizon safety constraints, versus the
required 90%.

This protocol defines the smallest next treatment-integrity iteration. It does
not claim to repair the delayed downstream safety failures, authorize predictor training, add
seeds to v5.2, change outcome labels, or weaken the safety budget.

## Frozen scientific baseline

The following remain unchanged:

- control arm: `idm_analytical_disturbance_p30_v1`, fingerprint
  `75649f14...fe2c7`, with zero learned residual;
- treatment arm outside the wall: the exact same analytical damping actions;
- treatment geometry: compact moving wall covering every occupied mainline
  lane at the declared decision;
- nominal treatment horizon: 20 seconds;
- wall speed/deceleration and safety-projection bounds from v5.2;
- treatment-blind feasibility inputs and paired branch identity contract;
- full-horizon collision, teleport, low-TTC, hard-braking, delay, discharge,
  and collapse measurements;
- the v5.2 safety and service thresholds, including the hard-brake budget;
- development seeds 4301--4312; these remain excluded from prospective claims.

No Krauss checkpoint, learned damping checkpoint, treatment outcome, or future
trajectory information may enter either arm before the decision.

## Evidence motivating exactly one repair

Two dose failures were mechanical:

1. 4301/660 lost a wall member after a lane change and continued for 17 seconds
   without full occupied-lane coverage.
2. 4312/660 retained both members but expanded from 21.0 m to 118.3 m, exceeding
   the frozen 50 m compactness cap for 45% of its samples.

Five dose-qualified cases failed safety only through excess global hard-brake
vehicle-steps. The excess appeared predominantly after release, not as direct
braking by active wall members or their immediate followers. The severe cases
4302/620 and 4308/680 also lost service, while 4307/680 and 4311/700 gained
service but crossed the safety budget.

These observations justify repairing spatial integrity and release dynamics.
They do not justify loosening the safety metric or fitting a predictor around
an unstable treatment.

## Frozen v5.3 treatment state machine

The state machine is deterministic and auditable:

`INACTIVE -> ASSEMBLE -> HOLD -> TAPER -> RELEASED`

Any hard safety violation transitions immediately to `RELEASED`. A coverage or
compactness failure may attempt only the bounded repair specified below; if it
cannot be repaired within the frozen allowance, it transitions to `TAPER` and
is recorded as a dose failure.

### 1. Assemble

- Latch one primary wall member for every occupied lane using only the
  treatment-blind candidate state at the declared decision.
- Latch a ranked same-lane reserve list at the same decision. Later treatment
  outcomes may not alter this order.
- Require predicted projection-survived authority for every primary member.
- Refuse activation if full occupied-lane coverage or the compactness contract
  cannot be satisfied at assembly.

### 2. Hold with spatial-integrity enforcement

- Apply the wall override only to the current wall members.
- All nonmembers retain byte-identical analytical damping actions.
- If a member leaves its assigned lane or disappears, promote only the next
  pre-latched same-lane reserve that currently passes the frozen instantaneous
  safety projection.
- Permit at most one handoff per lane per activation and record its identity,
  step, reason, and control delta.
- If no safe reserve exists, begin `TAPER`; never continue a partial-lane wall.
- If longitudinal span exceeds 50 m for two consecutive simulation samples,
  or no admissible command can reduce it toward the cap, begin `TAPER` and
  record `compactness_lost`.

### 3. Taper and release

The v5.3 integrity change is a predeclared smooth return to base
damping rather than a step removal of the coordinated target.

- Reserve the final 5 seconds of the existing 20-second horizon for a monotone
  taper of each member's wall-control delta to zero.
- Blend against the counterfactually identical analytical damping command at
  every step; the analytical command itself is never suspended.
- Apply the existing front/rear safety projection after blending.
- An early spatial-integrity failure starts the same taper immediately. It is
  still a dose failure unless the composite dose contract is met.
- At zero blend, clear wall membership and continue analytical damping without
  a SUMO-uncontrolled gap.
- Log downstream response through the full common horizon. The taper may not
  end the evaluation window or conceal post-release harm.

The exact taper function must be frozen before implementation. The default
proposal is a linear blend from 1 to 0 over 5 seconds. Testing multiple taper
shapes on outcomes and selecting the winner is not permitted under this
protocol.

## Required diagnostics

Every admitted case must report:

- requested and realized duration;
- the legacy duration-plus-lane dose field with its scope explicitly named;
- the compact composite dose verdict and all component thresholds;
- controlled CAV-seconds and controlled lane-seconds;
- occupied-lane coverage at every active sample;
- longitudinal span and fraction within the 50 m cap;
- primary/reserve identities and all handoffs;
- assembly, hold, taper, and release step counts;
- planned and realized release steps and explicit completion/release reason;
- base, raw wall, blended, projected, and applied command deltas;
- projection clips and front/rear conflicts;
- full-horizon hard-brake events by phase, vehicle, lane, and edge;
- low-TTC, collision, teleport, delay, discharge, and collapse deltas;
- exact replay/noise evidence at the precision used for treatment effects.

## Software and causal-integrity gates

Before any SUMO efficacy pilot:

1. damping-versus-damping replay is exact;
2. treatment-versus-treatment replay is exact;
3. branches are identical through the declared decision;
4. the first treatment request sees identical base actions and diagnostics;
5. only active wall members receive the overlay;
6. out-of-band CAVs retain identical analytical damping;
7. handoffs use only the pre-latched same-lane reserve order;
8. a failed handoff cannot leave a partial wall active;
9. compactness loss causes the declared taper/release path;
10. taper reaches exactly zero and returns to identical damping;
11. tau/speed-control state is restored for every released or replaced member;
12. all tests pass, followed by `python -m pytest -q`.

## Development and prospective sequence

### Stage 1: bounded development replay

Use only the already opened 4301--4312 cohort. First replay the seven diagnosed
cases 4301/660, 4302/620, 4307/680, 4308/680, 4309/620, 4311/700, and 4312/660.
This stage asks whether the repair engages for the intended reason; it is not a
new efficacy claim. Report every case, including regressions.

The version may advance only if:

- both spatial-integrity failures now release or safely hand off as specified;
- no case continues a partial-lane wall;
- no case silently exceeds the compactness contract;
- release returns exactly to analytical damping;
- safety does not worsen relative to its matched v5.2 branch;
- at least two of the three previously safe, beneficial cases remain safe and
  beneficial beyond the replay/noise deadband.

Only one v5.3 implementation is allowed. Failure returns a mechanism no-go; it
does not authorize threshold tuning on these cases.

### Stage 2: treatment-blind feasibility scan, then untouched confirmation

If Stage 1 passes, first run a treatment-blind p10--p30 feasibility scan for
v5.3, following the v5.2 scan discipline. Freeze the lowest penetration that
passes the predeclared availability rule before reading any v5.3 treatment
outcome. Only then, under separate Phase B approval, run the frozen schedule on
prospectively assigned seeds 4313 onward. Preserve seeds 5001--5060 for later
paper evaluation.

The confirmation passes only if:

- replay and all causal-integrity gates pass;
- at least 80% of admitted activations receive the compact composite dose;
- at least 80% occupied-lane coverage and at least 80% within-span geometry are
  satisfied by every dose-qualified activation;
- at least 90% of dose-qualified completed treatments are safety-admissible;
- there are beneficial completed treatments across at least three seeds;
- there are nonbeneficial completed treatments across at least three seeds;
- effects exceed exact replay/numerical noise;
- service gains do not conceal global hard-braking or downstream service loss;
- collapse-prevention diversity is required only if collapse prevention is
  restored as the primary claim.

If too few treatment-blind activations are admitted for these seed-diversity
gates, the pilot is inconclusive and may be expanded once to 20 total
predeclared seeds without changing the treatment or thresholds.

## Predictor architecture after, and only after, treatment pass

The eventual escalation decision is explicitly two-stage:

1. **Feasibility gate** `F(x)`: determines whether the frozen wall can be
   delivered with complete occupied-lane coverage, compact geometry, adequate
   projected authority, and admissible instantaneous safety. It uses only
   decision-time state and no treatment outcomes. The initial implementation
   should remain deterministic and rule-based so that physical deliverability
   is not confused with treatment benefit.
2. **Conditional benefit gate** `B(x)`: among cases with `F(x)=1` and adequate
   realized dose, estimates the conditional advantage of wall-plus-damping
   over damping alone under full-horizon safety and service constraints.

Deployment is `activate = F(x) AND B(x)`. The benefit label must retain harmful
wall cases; it must not discard them as infeasible after observing outcomes.
Training, calibration, and prospective predictor evaluation remain
unauthorized until the Stage 2 treatment gate passes.

## Owner decision and scope boundary

The owner approved the complete v5.3 integrity set on 2026-08-30: bounded
same-lane handoff, compactness enforcement, and the fixed in-horizon linear
taper. The taper-only efficacy pilot remains rejected, and no safety-repair
claim follows from the taper. The approved joint safety-and-benefit gate is the
scientific response to the delayed heterogeneous safety outcomes. Predictor-
scale generation remains outside Phase A.
