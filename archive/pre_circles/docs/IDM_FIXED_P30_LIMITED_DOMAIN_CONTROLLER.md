# Frozen Rule-Based Fixed-P30 Limited-Domain Damping Arm

**Status:** frozen as a development-only decentralized damping arm after the
limited-domain screen found no causal observation model that beats the fixed
rule (`docs/IDM_LIMITED_DOMAIN_P30_SCREEN_RESULTS.md`). It failed prospective
acceptance on seeds 4201--4220; no claim of a working or trained controller.
**Date:** 2026-08-30

## Definition (frozen)

A deterministic decentralized abstain-versus-activate rule:

```text
if CAV penetration not in {0.05, 0.10}:   abstain (native IDM)
if no qualified decision at the frozen landmark:  abstain
otherwise:  activate the frozen P30 terminal-speed treatment,
            candidate 1 by runway ranking, at the third risk-onset
            decision in the cell (decision_index 3, landmark risk_onset,
            max 3 landmarks per cell), 20 one-second option steps,
            legacy smooth jerk-limited return to exact native IDM, hard safety
            projection authoritative last.
```

- Treatment identity: `activation_speed_ceiling_v1` (P30, the treatment that
  produced the frozen paired grid), reference percentile p30, terminal speed
  floored at 8 m/s, speedMode 31, no percentile change.
- Deployment interpretation: penetration is a traffic-management
  configuration variable known at deployment and used as the admission rule
  outside the actor; the rule does not learn from outcomes.
- Domain cells: CAV penetration 0.05 and 0.10 at mainline demands 1900, 2000,
  2100 veh/h. No critical density is assumed; unsupported states abstain and
  remain candidates for later moving-wall escalation.

## Evidence as of this freeze (development-only, seeds 4107–4132)

- Six-penetration grid, 5–10% cells, 42 paired decisions (train 27, held 15).
- Held development (seeds 4129–4132, opened once): 7 helpful, 3 harmful, 5
  neutral, mean paired reward +0.04343, harmful fraction 20%, no added
  collision/teleport and service deltas within the frozen budget in the
  qualified activation set.
- Development pooled (42 decisions): 20 helpful, 11 harmful, 11 neutral,
  2 constraint-failed; all retained and reported.
- The screen found no causal observable model with policy value above this
  rule under the frozen gates (best held mean reward +0.0429, below the rule;
  per-demand positivity failed at 2000 veh/h). Learned activation was
  therefore not supported.

## Not claimed

No claim that damping works in general, that the rule is optimal, that any
probabilistic quantity is calibrated, that collapse is prevented, that the
moving wall or escalation predictor is validated, or that these development
results generalize. The controller is not a trained model and is deliberately
not used as a learned-activation demonstration.

## Status and required next step

1. FROZEN as development evidence and a conservative candidate policy.
2. Prospective acceptance on seeds 4201--4220 is complete and failed; see
   `IDM_P30_RULE_ARM_PROSPECTIVE_ACCEPTANCE_RESULTS.md`.
3. It is not promoted as the paper's validated damping controller. Any wall
   experiment that composes with this policy must state explicitly that the
   policy abstains outside 5--10% penetration.
