# predict() contract v2.1 note: Part A v2 general (docs only; draft for the planner)

**SUPERSEDED by PREDICT_CONTRACT_v2_1.md (frozen 2026-09-29).**

2026-09-29, world model session. This amends PREDICT_CONTRACT_v2.md for the owner's
"Part A v2 general" decision (PARTA_V2_GENERAL_DESIGN.md). It is a draft. The
planner (co-owner of the contract) must agree before code changes.

## Changes

1. **The Part A world is the recon sample.**
   - `predict_many` members M are replaced by the K recon samples. Each member
     is one recon sample rolled deterministically.
   - `K_arrivals` and the CRN arrival axes are REMOVED from Part A: Part A
     inserts no vehicles. Arrival handling moves entirely to Part B.
   - Clarification 8.14 (entrance queues within arrivals) no longer applies to
     Part A.
2. **D_A (delay over the commit window Δ = 30 s)** is defined over the TRACKED
   vehicles only (those present in the recon sample at t). The window delay of
   vehicles that arrive during (t, t + Δ] belongs to the Part B remainder.
3. **New output fields, per cell:**
   - `n_tracked`: vehicles tracked at t;
   - `n_exited_tracked`: tracked vehicles that left the modelled road before
     t + Δ;
   - `n_free_road_assumed`: tracked vehicles whose leader lay outside the recon
     zone at t, so free road was assumed. This is an optimistic-bias flag, and
     the planner should surface it;
   - `untracked_arrivals_expected`: optional, filled from Part B's inputs
     (observed entrance counts × Δ). It is informational only.
4. **Safety fields** (future_* and paired Δ-safety) are computed over tracked
   vehicles only. G-1/G-2 gating is unchanged, and the veto rules in §5 are
   unchanged.
5. **Metadata:**
   - `part_a_model = "v2_general"`;
   - `param_set_id` (the sha of the parameter file);
   - `recon_schema_version`;
   - `members_are_recon_samples = true`.

## Unchanged

- The commit window and continuation semantics (Δ = 30 s; Part B V^π(τ = T − Δ)).
- `cell_valid` / fail-closed validity, the native reference and
  `safety_veto_v2`'s parameters.
