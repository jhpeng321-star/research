# IDM v5.7.2 upstream-convergent assembly wall protocol

Status: **frozen before feasibility capture or any wall outcome branch.**

Contract: `idm_compact_moving_wall/5.7.2-upstream-convergent-assembly`.
This is a new treatment, not a reinterpretation of v5.7.1. The control arm is
the identical frozen decentralized IDM damping policy. The treatment arm adds
temporary coordination only to the selected, in-band CAV cohort.

## Scientific treatment definition

At the fixed t=880 decision, CAVs may be spatially distributed upstream. They
are **candidate future wall members** while assembling, not a compact moving
wall. Assembly is excluded from compact-wall dose. The compact moving-wall
treatment begins only after the candidates form a safe, full-lane compact
cohort and sustain readiness for three one-second controller samples.

The three-sample requirement is frozen to prevent a single noisy compactness
reading from starting HOLD. It is a controller-time persistence check, not a
threshold selected from the four observed spans.

## Frozen scenario and timing

- Original long IDM corridor only; no 900 m extension.
- Mainline/ramp demand: 2700/400 vph; CAV penetration: 30%.
- Development seeds: 4401, 4402, 4404, 4405 only.
- Decision: exact absolute SUMO time t=880 s; eligible band 200--1,900 m.
- Maximum assembly: 10 s; HOLD: 40 s; taper: 8 s; treatment window: 58 s.
- Outcome: 120 s after release, or verified natural clearance under the
  v5.7.1 terminal rule.

## ASSEMBLY: bounded convergence, not wall dose

Members are latched with full required occupied-lane membership. Every
projected and realized step must pass front/rear gap and TTC checks, retain
member identity and lane coverage, and remain within 0.3 m/s2 maximum
wall-induced deceleration. No high-authority step exists. The projection must
show physically credible convergence, sufficient runway for assembly plus
HOLD/taper, terminal span <=50 m, terminal speed spread <=0.5 m/s, and three
consecutive ready samples within the 10-second budget.

Transient span is recorded but has no arbitrary cap: the cohort is not yet a
wall. Any finite bound is supplied only by the per-step safety and runway
constraints, never tuned from these seeds. Loss of identity, coverage, or a
safety constraint causes a safe abort to exact damping.

## HOLD and TAPER

HOLD begins only after compact readiness: full required-lane coverage, span
<=50 m, speed spread <=0.5 m/s, front/rear safety, and runway for the
remaining 40 s HOLD plus 8 s taper. HOLD continuously enforces span <=50 m,
shared target speed, position correction, 0.3 m/s2 authority, and at most one
safe same-lane handoff. Out-of-band CAVs retain identical damping.

TAPER lasts eight seconds, monotonically blends back to the already-computed
damping action, and intentionally reports its span separately. Delayed
post-release safety is evaluated; no CAV is released to uncontrolled SUMO.

## Gates

The v5.7.2 decision-scoped feasibility capture reuses no congestion or wall
outcome information. It evaluates the costly formation preview exactly at the
frozen decision and writes restartable treatment-blind checkpoints. At least
three of four seeds must safely project full-lane, terminally compact,
speed-matched, runway-admissible assembly. Otherwise no wall outcome runs.

Only then may paired Gate C run for approved cases. It requires identity,
at least three safe compact HOLD entries, >=80% realized HOLD+taper dose and
coverage, HOLD compactness and low authority, safety and service
noninferiority, two independent >=10-second material general-congestion
improvements beyond replay noise, one feasible neutral/nonbeneficial case,
and no single-seed benefit explanation. Sustained-collapse prevention is
reported separately and is never inferred from general congestion reduction.
