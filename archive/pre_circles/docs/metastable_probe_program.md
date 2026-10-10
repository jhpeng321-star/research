# Metastable-State Probe Program — formalized answers to every open suggestion

**Date**: 2026-08-14 · **Status**: DESIGNED — infra half-built (two longer maps DONE,
placement/zone/release settings specified) · **Launch order**: Phase 1 → 2 → 3 → 4

Every suggestion the user has made, formalized into a probe with a **pre-committed
PASS rule**. Nothing here is exploratory; each probe either confirms or kills one
specific claim. All probes reuse `_probe_metastable_feasibility.py` (schema-2 rows,
pen-scoped no-op cache, sustained-collapse metric h<0.15 for ≥60 steps after step 100,
density_at_onset) unless stated otherwise.

---

## New infrastructure (partly built)

| Item | Status | Needed by |
|---|---|---|
| **3644 m map** (`D:\research\highway_merge_shepherd_long2`) | ✅ built + smoke-verified | L1, O1 |
| **5244 m map** (`D:\research\highway_merge_shepherd_long3`) | ✅ built + smoke-verified | L1, O1, U1 |
| Parameterized `tools/build_longer_map.py` (`--src/--dst/--extend`, copies `_light`+`_ss`) | ✅ done | future maps |
| **CAV placement mode** (scatter vs contiguous block) in `CAVSelector` | to build (~20 lines) | F1 |
| **Hold-zone location** (hold applies only to CAVs in a distance-from-merge window) | to build (policy wrapper) | U1 |
| **hold_release_T** variant (hold for T steps, then no-op) | to build (policy wrapper) | H1 |
| **Queue-spill recorder** (upstream-most queued vehicle vs corridor start) | to build (probe recorder) | Q1 |

---

## Phase 1 — the decisive tests (1 night)

### P1 — Penetration sweep at 2700-long (THE leak test)
**Suggestion formalized**: the 20% throttle is leaky (HDVs between CAVs flow through),
so the storage-queue cannot form. If the leak is the binding constraint, more CAVs
must fix it.

- **Design**: p ∈ {0.2, 0.3, 0.4, 0.5} × {no_op, hold_stable_main} × 2700-long × 50 seeds.
- **Metric**: P(collapse), dUpper, dTTC, density_at_onset per p.
- **PASS**: controlled P(collapse) strictly ↓ in p, AND at some p ≤ 0.5:
  P(collapse) ≤ 0.5 while no-op ≥ 0.85 (a real onset shift at 2700).
- **Cost**: 4 p × 2 variants × 50 seeds ≈ 2–3 h with `CELL_PARALLEL=8`.

### L1 — Length sweep (the storage-queue / longer-lane claim)
**Suggestion formalized**: "with a long-enough lane, CAVs hold back cars without
shockwaves, storing the queue upstream so the merge stays below critical density."

- **Design**: L ∈ {1145, 1844, 2644, 3644, 5244} × {no_op, hold_stable_main} ×
  2700 (and 2900 if time) × 50 seeds at the promising cells.
- **Metric**: P(collapse) vs L; dUpper; also tests the L_min(ρ) law's extrapolation
  (law predicts 2700 → ~2094 m, so 3644/5244 should show a WIDE margin).
- **PASS**: P(collapse) monotone ↓ in L at fixed (ρ, p), and at some L ≤ 5244
  controlled P ≤ 0.4 with no-op ≥ 0.85.
- **Cost**: 8 h cells at 8 seeds + ~4 confirm cells at 50 seeds ≈ 2–3 h.

### O1 — Onset map at the winning (p, L) — the paper's central figure
**Suggestion formalized**: "push the highway to a higher density before an active
collapse."

- **Design**: after P1/L1 pick the winning (p, L); demand sweep
  {2400..2900 step 100} × {no_op, hold} × 50 seeds.
- **Metric**: P(collapse) vs demand; 50%-crossing; **Δonset (veh/h) with bootstrap CI**.
- **PASS**: Δonset ≥ +100 veh/h, CI excluding 0.
- **Cost**: ~2 h. Reuses the run_s8-style orchestrator.

---

## Phase 2 — mechanism probes (1 night)

### F1 — CAV contiguity (formation) probe
**Suggestion formalized**: the leak is HDVs *between* CAVs; a contiguous block of
CAVs (a moving gate) should plug it at equal penetration.

- **Design**: at (2700-long, p=0.3): CAVs scattered (current) vs CAVs as one
  contiguous block of consecutive departures. Same p, same seeds, same count.
- **Metric**: P(collapse), dUpper per placement mode.
- **PASS**: block placement P(collapse) < scattered P(collapse) by ≥ 0.2.
  If true → the design lever is *CAV platooning*, not just penetration.
- **Infra**: `CAVSelector(placement="block")` — mark consecutive departures until
  the p-quota is filled, per seed.

### U1 — Upstream hold-zone sweep
**Suggestion formalized**: "if I start intervention really upstream then my merge
density is a lot healthier."

- **Design**: on the 5244 m map at (2700, p=0.3): hold applied only to CAVs within
  distance-from-merge windows {0–500, 500–1500, 1500–2500, 2500–4000} m.
- **Metric**: ρ_merge, Δv at merge, P(collapse) per zone.
- **PASS**: some zone > 1500 m beats the near-merge zone on ρ_merge AND P(collapse).
  (The honest null — near-merge is best — is also a publishable result.)

### H1 — Hold-release hysteresis test (true metastable vs mere delay)
**Suggestion formalized**: the KWG claim is that the CAV pushes the system ONTO the
upper branch; if the state persists after control stops, it is metastable-state
control; if it collapses on release, it was just a delayed collapse.

- **Design**: hold for T ∈ {120, 300, 600} s then release to no-op for the rest of
  the episode, at (2400-long, 2600-long, p=0.3) + the Phase-1 winning cell.
- **Metric**: h trajectory after release; fraction of post-release steps with h ≥ 0.6;
  sustained collapse after release.
- **PASS**: at the controllable cells, ≥ 60% of post-release steps stay on the upper
  branch (h ≥ 0.6), vs no-op's collapse. The strongest scientific claim in the paper.

### Q1 — Queue-fit / spillback recorder
**Suggestion formalized**: storage-boundary arithmetic — the stored queue either fits
the corridor or spills past its start into gridlock.

- **Design**: add a recorder to the probe driver: upstream-most queued vehicle
  (v < 2 m/s) position vs corridor-entry position; spill = queue reaches entry.
- **Metric**: max queue extent, spill fraction per (ρ, L).
- **PASS**: spill fraction = 0 at all controllable cells; > 0 at the beyond-cells.

---

## Phase 3 — analysis passes (no new compute, on Phase 1–2 rows)

### E1 — Economics: queue vs collapse ("is the hold worth it?")
**Suggestion formalized**: a collapse creates a shockwave → TTC + stop-and-go delay;
the hold costs queue delay — there are densities where the hold no longer pays.

- **Design**: on O1/L1 rows, per variant: total delay (loss) and TTC under hold vs
  under collapse (no-op). Find the (ρ, L) crossover where delay(hold) >
  delay(collapse) = the **worth-it boundary**.
- **PASS**: at the controllable cells delay(hold) < delay(collapse); the worth-it
  boundary exists and is reported as a second curve in the phase diagram.

### V1 — Δv inversion table (the injection-Δv theory)
**Suggestion formalized**: critical velocity is set by injection Δv and density;
at 2700 the Δv is inverted (ramp arrives faster than the congested mainline).

- **Design**: from existing + Phase-1 rows: Δv at merge per entry lane across
  {2000..2900}; find the inversion density; correlate Δv reduction under hold with
  dUpper (dv_history is already recorded per variant).
- **PASS**: inversion density measured; Spearman(Δv_reduction, dUpper) > 0.3.

### D1 — density_at_onset comparison
**Suggestion formalized**: "how many cars was the highway carrying when it
collapsed" — control should let it carry MORE before giving way.

- **Design**: paired comparison of density_at_onset (hold vs no-op) per cell.
- **PASS**: Wilcoxon on the paired deltas at the controllable cells (p < 0.05).

### G1 — Gate validation + L_min(ρ,p) law refit
**Suggestion formalized**: when should the CAV take control? The phase diagram as a
decision rule.

- **Design**: (a) run the feasibility-gated controller (margin obs + soft mask)
  across the phase-map cells; verify activation matches the measured boundary;
  (b) refit the margin law with penetration as a dimension now that P1 supplies
  p ∈ {0.3, 0.4, 0.5} cells — the old fit was degenerate because the leak pinned
  everything at ρ*=2400; with the leak fixed the law may become identifiable.
- **PASS**: (a) gate matches the measured controllability field on ≥ 80% of cells;
  (b) leave-one-out ≥ 60% (up from 32%).

---

## Phase 4 — the trained policy (2 nights incl. training)

### E2 — Multi-scene v4 training + learned-boundary evaluation
**Suggestion formalized**: the ultimate claim — a *learned* policy reproduces the
measured boundary (the feasibility gate as learned behavior).

- **Design**: train v4 (margin obs + feasibility mask + health reward) on the
  multi-cell curriculum: coast-zone cell (2000-short), controllable cells
  (2400-short, 2700-long, and the Phase-1 winning cell). Evaluate on holdout seeds.
- **Metric**: does the learned policy activate in exactly the controllable region
  (action-mode distribution per cell), and does its P(collapse) match the rule
  controller?
- **PASS**: learned policy P(collapse) ≤ rule controller at the controllable cells,
  and no activation in the coast zone.

---

## Settings changes allowed (user-authorized)

- Longer maps (3644/5244 m) — **built**
- Episode length: H1 needs longer episodes (hold 600 s + 300 s observe)
- Hold strength (meter_dial), hold lane (ramp/mainline/both), trigger source
- CAV placement (scatter vs block) — F1
- CAV penetration up to 0.5 — P1

## How this maps to the user's open questions

| User's question | Probe | Verdict meaning |
|---|---|---|
| Is the leak the binding constraint? | P1 | p-sweep shows onset shift appears |
| Does longer lane rescue 2700? | L1 | P(collapse) ↓ with L |
| Higher density before collapse? | O1 | Δonset ≥ +100 veh/h |
| Does CAV *formation* matter? | F1 | block placement beats scatter at equal p |
| Is upstream intervention better? | U1 | hold-zone sweep |
| True metastable or just delay? | H1 | upper branch persists after release |
| Queue fits or spills? | Q1 | spill fraction = 0 in controllable cells |
| Is the hold worth it? | E1 | delay(hold) < delay(collapse) crossover |
| When should CAVs act? | G1 + E2 | gate matches boundary; learned policy reproduces it |
