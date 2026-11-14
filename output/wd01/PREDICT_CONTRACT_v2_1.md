# predict() contract v2.1 (FROZEN): Part A v2 general

Frozen 2026-09-29 by the world model session with the planner's review (four
clarifications incorporated). This amends PREDICT_CONTRACT_v2.md. It
supersedes the draft PREDICT_CONTRACT_v2_1_NOTE.md. Design:
PARTA_V2_GENERAL_DESIGN.md.

## 1. Axes (no transposition)

- **K** is the planner-supplied recon samples (StateSamples), with the SAME
  order and the SAME meaning as in v2. Every output array is [M, K], with the
  K axis in exactly the order the planner passed.
- **M** is the predictor's internal ensemble size:
  - **M = 1** for the deterministic Part A v2 general;
  - a Part B ensemble may keep M > 1.
- The kit validator MUST accept M = 1 explicitly.
- Abstain counting over K, and the plan-vs-native pairing over all M × K
  cells, are unchanged.

## 2. Arrivals input

- `StateSample.arrivals` stays in the contract as an OPTIONAL input.
- Part A v2 general IGNORES it. It must not refuse a sample that carries it,
  and it inserts no vehicles.
- Other predictors keep using it. In particular, the planner's perfect-sim
  oracle drives post-t0 insertions from it.
- Arrival handling is therefore predictor-specific. It is not removed from the
  interface.
- Clarification 8.14 (entrance queues within arrivals) applies only to
  predictors that use arrivals.

## 3. Delay scope and the invariant

**Invariant:** `delay_part_a_s + delay_part_b_s` = the total delay over the
SAME accounting corridor and the SAME vehicle set, including arrivals. The
planner decides on the total.

New meta field: **`part_a_scope` ∈ {"tracked", "all_vehicles"}.**

- **Part A v2 general declares "tracked".**
  - D_A is the delay over the commit window Δ = 30 s of the vehicles present
    in the recon sample at t.
  - Part B's target is the total minus the tracked D_A (see
    PARTB_COUPLING_NOTE_V2GENERAL.md), so the invariant holds.
- **The planner's perfect_sim declares "all_vehicles".**
- `part_a` values are compared across predictors only when `part_a_scope`
  is equal. Totals are always comparable.

## 4. New fields (optional for all predictors)

**Per cell:**

- `n_tracked`: vehicles tracked at t;
- `n_exited_tracked`: tracked vehicles that left the modelled road before
  t + Δ;
- `n_free_road_assumed`: tracked vehicles whose leader lay outside the recon
  zone at t (an optimistic-bias flag);
- `untracked_arrivals_expected`: informational, from Part B's observed-count
  inputs.

**Meta:**

- `part_a_model` (e.g. "v2_general");
- `param_set_id` (the sha256 of the parameter file);
- `recon_schema_version`;
- `part_a_scope`.

**Usage rules:**

- The planner LOGS these fields per decision (decisions.jsonl) and in the
  receipt.
- **No veto and no abstain on `n_free_road_assumed`.** A threshold would be an
  owner decision.
- Every field is OPTIONAL. Predictors that do not emit them (e.g.
  perfect_sim, which makes no free-road assumption) remain valid.

## 5. Safety

- The future_* fields and the paired Δ-safety fields are computed over tracked
  vehicles only when `part_a_scope` = "tracked".
- G-1/G-2 gating, the safety_veto_v2 parameters and the §5 consumer rules are
  unchanged.

## 6. Unchanged from v2

- Δ = 30 s commit and continuation (Part B V^π at τ = T − Δ).
- `cell_valid` / fail-closed validity, the native reference, and the required
  flags of v1.1 / v2.
