# IDM v5.6.2 phase-split-authority no-go diagnosis

Status: **final for this frozen treatment; do not add seeds or retune**

The v5.6.2 four-case gate fails even though its phase authority and immediate
safety/service constraints pass. This is not a jerk-transition safety failure.

1. **Insufficient sustained low-authority effect.** Reducing HOLD from the
   v5.4 high-authority servo to 0.3 m/s² removes the historical material
   benefit: 4402 falls from 91 s prevention to a nonmaterial 5 s reduction,
   and 4405 falls from 24 s shortening to a nonmaterial 6 s reduction.
2. **Compactness is not retained after assembly.** 4404 and 4405 remain
   full-lane but diverge after low-authority control/taper, reaching 153.92 m
   and 90.93 m. This invalidates the claimed 50 m compact-wall dose.
3. **The new SETTLING rule was correctly inactive, not disproven.** Actual
   post-assembly accelerations were already inside the low-authority envelope,
   so no jerk-relaxation command was required. The rule neither caused span
   loss nor recovered the lost sustained authority.

No fresh seed, predictor, RL, geometry, duration, authority, or release tuning
is permitted under this version. The evidence supports an authority-boundary
finding: a low sustained deceleration cap can avoid the previous immediate
safety pattern but cannot simultaneously retain compactness and both reference
benefits in this compact-wall formulation.

Any next route requires a new treatment definition, not a v5.6.2 adjustment.
Possible separately approved routes are revised geometry with treatment-blind
feasibility, discharge-aware/staggered release, support-CAV treatment, or an
authority-boundary paper.
