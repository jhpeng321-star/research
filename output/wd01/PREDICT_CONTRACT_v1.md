# predict() contract v1 — `wd01_predict/1`

Owner: world model session (WD-01). Co-drafted with the planner session
(PL-01), which accepted every clause on 2026-09-25. This document is
design text only. The typed Python stub, the validators and the conformance
tests are WD-01 code deliverables and go through ZCode.

Frozen 2026-09-25. Any change needs a new contract_version and agreement
from both sides.

## 1. Purpose

A single prediction interface with two implementations:

| implementation | owner | M (members) | K (samples) |
| --- | --- | --- | --- |
| perfect simulator (SUMO saveState/loadState branching) | planner (WM-41 checks exactness) | 1 | arrival/demand draws; common random numbers across branches |
| learned (Part A rollout + Part B delay estimator) | world model (WD-series) | ensemble size | reconstructor hidden-vehicle samples × arrival draws |

Continuous planner decisions run on the learned implementation. The
perfect simulator is for targeted experiments and for the evaluation "planner
with the learned model vs planner with the perfect simulator".

## 2. API

```
predict_many(state_samples, plans, controller_state, *,
             horizon_s=600.0, trajectory_s=None) -> list[Prediction]
predict(state_samples, plan, controller_state, **kw)
    == predict_many(state_samples, [plan], controller_state, **kw)[0]
```

- The output list has the same order as `plans`.
- The native reference (§5) is computed ONCE per call, on the same samples,
  and shared by every Prediction in the call.
- `trajectory_s=None` means the model's own `part_a_horizon_s`.
  Implementations may reject `trajectory_s > part_a_horizon_s`.
- Every output array is a numpy array (CPU). The compute device (CPU/CUDA)
  is internal to the implementation. Callers never pass or see a device.
- Every call is deterministic given its inputs, the model weights (by sha256)
  and the arrival seeds.

## 3. Inputs

### 3.1 `state_samples`: sequence of K `StateSample`

Each sample contains:
- `hybrid`: a reconstructor hybrid state, `wm30_hybrid_state/1` or its
  successor. It holds the observed-vehicle layer, the hidden-road field
  layer, and ONE sampled hidden-vehicle set.
- `arrivals`: an arrival draw for boundary inflow over `[t0, t0 + horizon_s]`
  at the mainline entrance and the ramp. It is either an explicit list
  `(t_s, entrance, speed_mps, is_cav, vehicle_type)` or a
  `{demand_model_id, seed}` pair that deterministically expands to one. The
  plan and native use the identical draw (common random numbers).
- `meta`: `{sample_id, time_s (= t0), merge_observed: bool, coverage: float}`.
  The planner uses `merge_observed` and `coverage` only to decide ABSTAIN.
  Models ignore them.

All K samples in one call share the same `time_s`.

### 3.2 `controller_state`

```
{cav_id: {action: "native"|"damping"|"pi-external-u"|"wall",
          wall_id: str|None,
          wall_phase: "assemble"|"hold"|"release"|None,
          time_in_phase_s: float,
          wall_speed_mps: float|None,
          locked: bool}}
```

This is a separate argument until the reconstructor adds a controller-state
block to the hybrid schema (requested 2026-09-25). After that, it is taken
from the state and the argument is kept only for validation. Ideal sensing
means these values are known exactly; they are not estimated.

### 3.3 `PlanCandidate`

```
{plan_version: 1,
 per_cav: {cav_id: "native"|"damping"|"pi-external-u"},
 walls: tuple[WallSpec]}

WallSpec = {wall_id: str,
            members: tuple[cav_id],  # ordered by lane index, one per lane of
                                     # the cross-section at the wall location
                                     # (a side-by-side pair on the 2-lane mainline)
            speed_mps: float,
            phase: "assemble"|"hold"|"release",
            release_at_s: float|None,   # exactly one of release_at_s / hold_s
            hold_s: float|None}         # hold duration after assembly completes
```

Validation rules. Violations raise and never return a silent guess.
- `per_cav` is the FULL map of every CAV that is not a wall member. The
  planner always sends it complete.
- Wall members never appear in `per_cav`.
- A CAV with `locked=True` may appear only as a member of its existing wall,
  under the same `wall_id`.
- Exactly one of `release_at_s` / `hold_s` is set. `phase="release"` means
  release now. Times are relative to t0.
- Several walls are allowed. Wall membership sets are disjoint.
- v1 plans are constant from t0 (no per-CAV schedule). The planner re-decides
  every tick. A `schedule` field is reserved for a later version; it would
  let campaign branches that switch every 20–120 s share the format.

## 4. Output: `Prediction`

Axes: every `[M, K]` array shares indices with native and with every other
Prediction in the call. This gives paired comparison.

### 4.1 Delay (units: vehicle-seconds)

| field | shape | meaning |
| --- | --- | --- |
| `delay_vs_native_s` | [M,K] | `plan_delay_s − native_delay_s` (negative = saving) |
| `plan_delay_s`, `native_delay_s` | [M,K] | total delay under the plan / native |
| `delay_part_a_s`, `delay_part_b_s` | [M,K] | diagnostic split of `plan_delay_s`: inside the Part A rollout / Part B estimate of the remainder |

Total delay = Σ over vehicles present at t0 or arriving in
`[t0, t0 + horizon_s]` of (time on section − free-flow time on the traversed
part of the section) + time spent waiting to enter at the mainline entrance
and at the ramp. Every term is truncated at `t0 + horizon_s`. Totals only;
there is no per-vehicle breakdown.

### 4.2 Safety (Part A window)

Every safety field covers `[t0, t0 + meta.safety_window_s]`, where
`safety_window_s = part_a_horizon_s`.

| field | shape | definition |
| --- | --- | --- |
| `collision` | bool[M,K] | any overlap / collision in the window |
| `min_ttc_s` | [M,K] | minimum closing TTC over all follower–leader pairs (`traffic_shepherd.metrics.closing_ttc`) |
| `ttc_lt3_exposure_s`, `native_ttc_lt3_exposure_s` | [M,K] | vehicle-seconds with closing TTC < 3 s |
| `hard_brake_s`, `native_hard_brake_s` | [M,K] | vehicle-seconds with acceleration < −3.0 m/s² (`metrics.braking_effort`) |
| `native_collision`, `native_min_ttc_s` | [M,K] | the same, for native |

The perfect simulator may add optional `full_horizon_*` versions. Gating
always uses the window fields.

### 4.3 Trajectories (Part A window, 1 s steps)

- `trajectories_tracked: {vehicle_id: float[M,K,T,4]}` covers CAVs and
  observed vehicles, whose ids are stable across samples. The columns are
  `(t_s, s_m, lane_id, speed_mps)`. lane_id is 0 = main_left, 1 = main_right,
  2 = accel (the WM-30 physical lanes). `s` is measured along the physical
  lane, with `d = OFFSET[lane] − s`. A vehicle that has left the section is
  NaN from the exit step on. T = `trajectory_s + 1` (includes t0).
- `trajectories_hidden`: OPTIONAL. A per-k list of ragged arrays
  `[n_k, T, 4]` plus per-k ids. Sampled ids are not stable across samples.
- `native_trajectories_tracked`: the same, for native. The planner's
  measured-interaction check compares tracked trajectories across plans.

### 4.4 Walls

`walls: {wall_id: {assembled: bool[M,K], assembled_at_s: float[M,K] (NaN if
not), safety_abort: bool[M,K], abort_reason: int8[M,K] (0 none, 1 safety,
2 assembly_timeout)}}`

### 4.5 Optional native_unlocked diagnostics

`native_unlocked_{delay_s, ttc_lt3_exposure_s, hard_brake_s, min_ttc_s,
collision}` [M,K]: every wall is dissolved at t0, including walls still in
assembly. These are for logging the cost of the assembly lock and are never
used for decisions. `meta.native_unlocked_available` records whether they
are present.

### 4.6 `meta`

`{contract_version: "wd01_predict/1", model_id, weights_sha256 (list, one per
member; "simulator:<sumo version>+<network sha>" for the perfect simulator),
sample_ids [K], member_ids [M], part_a_horizon_s, safety_window_s, horizon_s,
native_kept_walls: list[wall_id], native_unlocked_available: bool,
device: str (informational), wall_clock_s: float}`

## 5. Native reference

In native, every CAV that is not a member of a kept wall runs native IDM.
- Walls in phase ASSEMBLE continue: they are locked, and only a safety abort
  ends them. Their wall_ids are listed in `meta.native_kept_walls`.
- Walls in phase HOLD are dissolved at t0 and their members hand back to
  native. This lets "the current hold is worse than native" be detected.
- Walls in phase RELEASE finish their handback and count as dissolved.

## 6. How the planner reads the output (PL-01 declarations, recorded here)

- Saving agreement (≥ 80%) and the uncertainty margin are computed over ALL
  M×K cells of the paired `delay_vs_native_s`. K-only and M-only spreads are
  logged as diagnostics. The learned model's spread must therefore be
  calibrated over M×K (WD evaluation item 4).
- Absolute safety, required in ALL M×K cells: `collision == False` and
  `min_ttc_s ≥ 1.5`.
- Relative safety: `ttc_lt3_exposure_s` and `hard_brake_s` show no increase
  vs native, both on average and in ≥ 90% of cells, within a small tolerance
  (the planner sets the tolerance).
- Gating uses the `safety_window_s` window, including for the perfect
  simulator.
- The perfect simulator has M = 1 and K = arrival draws, so the 80% rule is
  not degenerate. WM-41 checks exactness.

## 7. Performance note

Continuous decisions use the learned model, so throughput matters. WD
evaluation item 5 reports predictions per second: plans × M × K per
wall-clock second, on CPU and, once available, on CUDA. `predict_many` must
batch all plans, members and samples in one pass.

## 8. v1 clarifications (agreed with PL-01, 2026-09-26; no field changes)

These record the stricter readings implemented in `output/wd01/predict/`
(WD-01 p2B). They are binding for v1.

1. A native-kept ASSEMBLE wall is encoded as `phase="assemble"`,
   `hold_s=+inf`. `+inf` is allowed only in `hold_s`.
2. A wall dissolved in native (HOLD or RELEASE at t0) stays in the native
   plan as `phase="release"`, `release_at_s=0.0`, keeping its members,
   because §3.3 forbids locked CAVs in `per_cav`. `meta.native_kept_walls`
   carries the keep/dissolve distinction.
3. Wall members are lane-ordered by the planner. `native_reference`
   orders members by `cav_id`. See v1.1 for the optional `lane_id`.
4. Time reference. `release_at_s` is an OFFSET FROM t0 (0 = release
   now). `hold_s` counts from assembly completion. Exactly one of the
   two is set in every phase. Adapters to the WM-40 dispatcher (absolute
   episode time) convert on their side.
5. `speed_mps` of a wall must be finite and > 0.
6. Trajectory NaN rows form a trailing suffix per (m, k): whole rows,
   no re-entry, and step 0 is always real.
7. `min_ttc_s` may be NaN ONLY in a VALID cell (see §9 `cell_valid`) and ONLY
   when that cell has no closing follower–leader pair
   (`has_closing_pair == False`). In that case it passes the TTC floor.
   NaN never signals a failed, invalid or early-stopped prediction. Those
   are flagged by `cell_valid == False`, and the planner fails closed on them
   (treats them as unsafe). Every other `[M,K]` float in a valid cell must be
   finite, and a non-finite value is a hard failure.
8. The §4.1 identities (`delay_vs_native = plan − native`,
   `part_a + part_b = plan`) are checked with `rtol=1e-6, atol=1e-9`.
9. Arrival entrance labels are `"mainline"` / `"ramp"`. `coverage` is in
   [0, 1].
10. `trajectory_s > part_a_horizon_s` may be rejected.
11. `meta.wall_clock_s` is excluded from determinism checks.
12. `validate_prediction` takes `n_plans` explicitly.
13. Action vocabulary: the contract uses `native / damping / pi-external-u`
    for `per_cav`, and walls through `WallSpec`. Mapping to the dispatcher's
    `wall-v2` is an adapter concern.

14. Entrance queues in arrivals (agreed with PL-01, 2026-09-26):
    (a) arrival rows may have `t_s < t0`. These are vehicles that were due
    but not yet inserted at t0, and their entry waiting counts from t0.
    (b) Each row may carry an optional `arrival_id`, which is identical
    between plan and native in every implementation, including the perfect
    simulator.
    (c) A vehicle still queued at `t0 + horizon_s` counts its wait truncated
    at the horizon.
    The queue at t0 is part of the state. The perfect simulator must
    preserve it across saveState/loadState, which WM-41 checks.

15. `meta.merge_observed` (agreed with PL-01, 2026-09-26) is True iff the
    whole merge region is within sensor coverage at t0 for that sample,
    REGARDLESS OF OCCUPANCY. An empty merge zone that is covered counts as
    observed. The merge region is the reconstructor's RSU region:
    - the on-ramp/accel lane;
    - the merge edge (all lanes);
    - the RSU mainline window upstream and downstream of the merge.
    The reconstructor sets it. Under ideal sensing with the RSU it is always
    True, except when the RSU reports for t0 are missing or older than the
    reconstructor's declared staleness bound. It must never be derived from
    vehicle presence.
    Addendum (WM-45 landed): hybrid_state/2 carries `merge_observed`
    (bool, or None when no RSU inputs are given), `merge_observed_reason`,
    `rsu_staleness_bound_s` (0.5) and `rsu_report_age_s` per sub-span
    (ramp, merge, main, down). The StateSample builder maps None to False
    (fail closed, counts toward ABSTAIN), because `meta.merge_observed` is a
    bool in this contract.

## 9. v1.1 (contract_version `wd01_predict/1.1`, backward compatible)

- `controller_state[cav_id]` gains an OPTIONAL `lane_id: int|None`
  (0 main_left, 1 main_right, 2 accel). When present for every member of a
  wall, validators check that `members` are ordered by lane index.
  `native_reference` then orders members by `lane_id`.
- NEW REQUIRED OUTPUTS in 1.1 (the owner's hard safety rule; main discussion,
  2026-09-26):
  - `cell_valid: bool[M,K]` (and `native_cell_valid: bool[M,K]`), REQUIRED
    in 1.1 outputs. False means the cell's outputs are unusable: model
    failure, an early-stopped branch, a simulator error, or a
    teleport-corrupted branch. Invalid cells may contain NaN in any field.
    A 1.0 output (no `cell_valid`) is accepted only through the v1 validator
    path. Consumers of 1.1 outputs must reject a missing `cell_valid`; they
    never default it to True.
  - `has_closing_pair: bool[M,K]` (and `native_has_closing_pair`): whether any
    follower–leader pair was closing in the safety window.
  - Validator rules:
    - in a valid cell, `min_ttc_s` is NaN if and only if
      `has_closing_pair == False`;
    - every other float in a valid cell is finite;
    - `collision` / `native_collision` are defined (not NaN-carrying) in
      valid cells.
  - The conformance kit includes a predictor that emits NaN `min_ttc_s` with
    `has_closing_pair == True`, or signals a failure only through NaN. The
    kit must reject both.
- FAIL CLOSED (consumer rule, PL-01 + main discussion): any cell with
  `cell_valid == False` (plan or native), a NaN in any delay field, or
  `collision == True` FAILS safety, for all M×K cells. The planner also
  treats a wall `safety_abort` in a plan cell as failing that plan (planner
  policy). The planner evaluates these vetoes before it reads `min_ttc_s`.
- Implementations accept both `wd01_predict/1` and `wd01_predict/1.1`
  inputs and report the version they implement in `meta.contract_version`.
