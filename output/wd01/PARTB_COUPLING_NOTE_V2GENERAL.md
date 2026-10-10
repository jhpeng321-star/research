# Part B coupling with Part A v2 general (docs only)

2026-09-29, world model session.

## What Part B must cover

Part A now predicts only the recon-provided vehicles over the commit window,
with no insertions. So Q(s, a) = D_A + V must still equal the total
system delay. Part B covers:

1. **Arrivals during the commit window (t, t + Δ].** Add a term D_arr(Δ). The
   simplest option is to include it in V's target: V is evaluated at s' but
   also accounts for vehicles that entered during (t, t + Δ]. The alternative
   is a separate small model on the observed entrance counts.
   - **Proposal:** Part B trains on the label "total delay over (t, t + Δ + τ]
     minus D_A,tracked(t, t + Δ)".
   - The deployable features (observed entrance counts, demand rate) are
     exactly the right inputs for this.
2. **The continuation after t + Δ** over all vehicles, as before
   (τ = T − Δ).
3. **Vehicles outside the recon zone.** Where the zone is smaller than the
   delay-accounting corridor, their delay is part of V.

## Consistency

The labels must use the same corridor and vehicle set as Part A's D_A
definition (the tracked set at t). The builder can derive per-vehicle labels
from existing logs (vehicle-level delay is already computed in the part4 delay
library), so no new simulation is needed.

## Status

The v0.3.1 deployable model predicts TOTAL delay over (t, t + τ]. To couple
with Part A v2 general, the next Part B version retargets to the "total minus
tracked-D_A" label defined above. It runs offline on existing logs, and the
acceptance rule is the same (it must beat a ridge given equal information).
