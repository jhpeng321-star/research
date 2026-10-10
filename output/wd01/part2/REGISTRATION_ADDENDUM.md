# WD-01 Part 2 — REGISTRATION ADDENDUM (part-1 T2 addendum, pre-scoring)

- Scope: ADDENDUM to part-1 T2. Nothing in part 1 is rewritten; its
  registered verdicts stand exactly as recorded. This document freezes the
  corrected RAW overlap metric and the contract-level aggregate safety
  diagnostics BEFORE any of their numbers are computed.
- Registered (this file hashed): 2026-09-26, UTC in
  results/registration_addendum_record.json, written before addendum_run.py
  first executed on real data.
- Zero simulation; data = part-1 T2 regenerated-prefix rollouts re-run
  deterministically (same surviving checkpoint
  sha256 ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5,
  same 21 holdout runs x start ticks {120..480 step 60} x H in
  {30, 60, 90, 120} x {RAW, CONSTRAINED}).

## 1. Defect in part-1 RAW overlap (measured, not re-registered)

part-1 `raw_passthrough()` sorted vehicles by their CURRENT d and then
tested `d[i] <= d[leader]` along that same sorted order — false by
construction, so RAW overlap counts were identically 0 for all horizons.
Part-1's recorded C2 verdict for RAW ("zero overlaps") is therefore
UNSUPPORTED and is superseded by the corrected numbers below; part-1 files
are not modified.

## 2. Corrected RAW pass-through definition (frozen)

For one 1 s step t -> t+1, within a lane group: a violation is a
follower/leader pair ADJACENT in the d-order at t (leader = next smaller d,
the immediate in-group predecessor in the sorted order) whose order is
inverted at t+1: `d_follower(t+1) <= d_leader(t+1)`. The follower is
flagged; one event per (pair, step). This is gap-agnostic — adjacency in
the order, exactly the same barrier semantics the CONSTRAINED integrator
uses (front-to-back, `d'_self <= d'_leader`). Rate = flagged follower-steps
/ (vehicle-steps in the window) * 1000. The unit test
tests/test_raw_overlap_fix.py proves the old part-1 function returns 0 on
an inverted pair while the corrected one flags it.

Corrected RAW C2 verdict per H: zero vehicle-steps above 1.2 x lane-group
train max (part-1 cap) AND corrected overlap rate <= 1 per 1000
vehicle-steps. CONSTRAINED C2 is unchanged (integrator-flagged overlaps).

## 3. Contract-level aggregate safety metrics (frozen; DIAGNOSTICS ONLY)

Per eval (run x start tick x H x variant), predicted vs truth over the same
frozen t0 vehicle set, 1 s step (the model's native step; 0.5 s TTC is a
Part-A deliverable, not measurable from this model):

- `ttc_lt3_exposure_s`: vehicle-steps with closing TTC < 3 s (in-group
  leader from the snapshot's own d-order, closing > 0, gap/closing < 3);
  truth side uses the full truth snapshot's leaders (part-1 truth-event
  convention), predicted side uses rollout-state leaders.
- `ttc_lt1p5_exposure_s`: same at 1.5 s.
- `hard_brake_s`: vehicle-transitions with dv < -3 m/s over 1 s inside the
  window.
- `min_ttc_s`: minimum closing TTC in the window (null when no vehicle is
  closing anywhere in the window).
- `any_ttc_lt1p5`, `any_ttc_lt3`, `any_hard_brake`: booleans.

Reported per H x variant, pooled over the 147 evals:

- window-level detection recall = P(pred flag | truth flag) and precision =
  P(truth flag | pred flag) for the three booleans;
- Pearson and Spearman of pred vs truth for ttc_lt3_exposure_s,
  hard_brake_s, min_ttc_s (min-TTC correlation over evals where both are
  finite);
- mean bias (pred - truth) per continuous metric;
- fraction of evals where predicted exposure >= truth exposure.

These are DIAGNOSTICS with no pass/fail: the planner's safety gate (contract
section 6) consumes window aggregates of a plan against native, not
co-timed per-vehicle seconds, so window-level fidelity — not per-second
matching — is what the contract needs from these numbers.

## 4. Determinism

The surviving checkpoint, rollouts, integrators and event rules are the
part-1 ones (imported, not copied-and-edited); re-running is deterministic
on the same CPU build. Any deviation from part-1's stored aggregate
counts that is not explained by the corrected overlap metric is reported
as a DEVIATION.
