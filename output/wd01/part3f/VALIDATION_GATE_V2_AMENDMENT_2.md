# Gate v2 — Amendment 2: TTC<3 coverage moves from (c) to G-2

Recorded 2026-09-28 by the world model session, on the owner's decision (B)
(tmp/zcode_review/OWNER_DECISIONS_20260928.md, item 1, relayed by main discussion).
Written BEFORE any chunk-2 native Part A number exists.

## Disclosure

This amendment is made AFTER the TTC-informativeness finding: in every round
(p3i–p3q, VAL look #8) the TTC<3 exposure forecast was no better than a TRAIN
climatology by CRPS. Its coverage passed at VAL look #8 (0.8077) only because of
the frozen widening factor. Coverage of an uninformative forecast is not
evidence of a useful safety model, so the item moves to the check that can
test it: G-2, the paired relative-safety validation against replicated truth.

## Change

Gate v2 criterion (c) now reads:

(c) The member 10–90 % band covers the truth in ≥ 80 % of windows for the
    window **physics delay** (contract delay, dt − ds/limit), AND the ensemble
    CRPS of the physics delay is below the MAE of the deterministic IDM-normal
    (L0) on the same windows.

- TTC<3 exposure coverage and CRPS vs climatology are still computed and
  reported every time, labelled "diagnostic (G-2)". They are not gate items.
- (a) collisions, (b) lane changes, (d) speed are unchanged. Amendment 1
  (primary H = 30 s; H = 60 reported) is unchanged.
- The physics delay is the delay pinned in (c). This matches VAL look #8,
  whose binding (c) leg was the physics delay. The exposure-proxy delay is
  reported as a diagnostic.

## Consequences

- Absolute and paired TTC<3 / hard-brake outputs of Part A remain UNVALIDATED
  (G-2) regardless of the gate result. `relative_safety_validated` stays False
  in predict() meta until G-2 passes on replicated truth.
- VAL look #8 is NOT re-scored under this amendment. It was a FAIL on the
  physics-delay leg, which is unchanged.
