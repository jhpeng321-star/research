# predict() contract v2 — `wd01_predict/2`

Owner: world model session (WD-01). Agreed with the planner (PL-01). Direction
approved by the owner on 2026-09-26 (DESIGN_PARTB_V2.md §2–§2.2). This is a
DELTA on `PREDICT_CONTRACT_v1.md` (v1 + §8 clarifications + §9 v1.1).
Everything not changed here stays as in v1.1. v1.1 stays frozen.
Implementations and validators are dual-version: `wd01_predict/1`, `/1.1`
and `/2`.

## 1. Semantics

- A `PlanCandidate` is the action for the **commit window** `commit_s` (Δ)
  starting at t0.
- After Δ, the plan branch and the native branch both continue under
  `continuation`:
  - `"planner"`: the planner's policy π, identified by
    `meta.continuation_policy_id`;
  - `"native"`: all CAVs native after Δ. This is the old held-reference mode,
    kept as a cheap perfect-simulator mode and for V0.
- Δ is prediction semantics only; it is not a dispatch dwell. The planner may
  re-decide every tick.
- The native reference (v1 §5) applies to the commit window.

## 2. API

```
predict_many(state_samples, plans, controller_state, *,
             horizon_s=600.0, commit_s=30.0, continuation="planner",
             trajectory_s=None) -> list[Prediction]
```

`trajectory_s=None` means `commit_s`. An implementation may reject
`trajectory_s > part_a_horizon_s`.

## 3. Output changes

Delay (veh·s, [M,K]):
- `delay_part_a_s`: total delay (v1 §4.1 definition) accrued in
  [t0, t0+Δ]. Computed at the implementation's internal step, not from the
  1 s trajectory output.
- `delay_part_b_s`: expected remaining delay over [t0+Δ, t0+horizon_s]
  under the continuation. For the learned model this is V^π(s', horizon_s − Δ).
- `plan_delay_s` = part_a + part_b. `native_delay_s` is computed the same way
  for native. `delay_vs_native_s` = plan − native.
- Pairing: for a given cell (m, k), the same value-model member m and the same
  sample k are used for plan and native.

Safety within the commit window (absolute gate): the v1 window fields
(`collision`, `min_ttc_s`, `ttc_lt3_exposure_s`, `hard_brake_s`, native_*)
with `meta.safety_window_s = commit_s`.

Future safety, over (t0+Δ, t0+horizon_s] under the continuation. All are
[M,K], and all have `native_` versions:

| field | meaning |
| --- | --- |
| `future_ttc_lt3_exposure_s` | expected veh·s with TTC < 3 s |
| `future_hard_brake_s` | expected veh·s with a < −3 m/s² |
| `p_future_collision` | probability of any collision |
| `future_min_ttc_q05_s` | 5th percentile of the minimum closing TTC |

Validity (REQUIRED in v2, fail closed; never defaulted to True):
- `cell_valid` and `native_cell_valid` (v1.1) cover window fields and
  part_a.
- `future_cell_valid` and `native_future_cell_valid` cover part_b and all
  future_* fields.
- `future_has_closing_pair` and `native_future_has_closing_pair` bool[M,K]
  are REQUIRED in v2. They say whether any follower–leader pair is closing
  beyond Δ.
  - Learned model: False only when the predicted probability of a closing pair
    is < 0.05. This is conservative: the floor is applied whenever a pair is
    plausible.
  - Perfect simulator: realised pair existence.
- NaN never signals failure (v1 §8.7). In a valid future cell every future_*
  float is finite, except `future_min_ttc_q05_s` (and native_), which may be
  NaN ONLY where `future_has_closing_pair` is False.

Future walls: `walls[wall_id]` gains [M,K] `p_assembly_completes`,
`expected_assembly_time_s` (NaN only where `p_assembly_completes == 0`) and
`p_assembly_abort`. There are no native_ versions. The in-window v1 wall
fields are unchanged.

## 4. Meta additions

- `contract_version: "wd01_predict/2"`;
- `commit_s`, `continuation`;
- `continuation_policy_id`: str, or None when continuation == "native";
- `value_model_id` and `value_model_sha256`: list per member;
- `future_min_ttc_validated`: bool;
- `future_min_ttc_validation_id`: str or None. It points to the ledger/report
  record that validated the `future_min_ttc_q05_s` head. It must be non-None
  when the flag is True.

## 5. Consumer rules (planner; recorded here as binding)

Evaluated over ALL M×K cells:
1. Fail closed: any cell with `cell_valid`, `native_cell_valid`,
   `future_cell_valid` or `native_future_cell_valid` False, a NaN delay, or a
   window collision fails safety (v1.1 rule, extended).
2. Absolute window gate: no `collision` and `min_ttc_s ≥ 1.5` within Δ.
3. Future collision (owner, 2026-09-26): veto if `p_future_collision > 0.01`
   in ANY cell, AND veto if `p_future_collision > native_p_future_collision
   + tol` in the paired cell (tol configurable, default 0).
4. Relative safety: window difference + future difference for TTC<3 exposure
   and hard braking, tested with the planner's mean and ≥ 90%-of-cells rules.
5. **Gating item G-1:** the 1.5 s min-TTC floor applies only within Δ unless
   `meta.future_min_ttc_validated` is True, in which case the floor also
   applies to `future_min_ttc_q05_s` in cells where `future_has_closing_pair`
   is True. The planner REFUSES live or closed-loop
   mode outside controlled experiments unless `future_min_ttc_validated` is
   True.

## 6. Perfect simulator

- For `continuation="planner"` it runs π_k closed-loop inside each branch after
  Δ. π_k uses the learned models, at a coarse cadence (PL-04), on a registered
  subset.
- `continuation="native"` is the default mode for most perfect-sim checks.
- It reports `future_*` fields as realised values from the branch: the
  probability is 0 or 1 per cell, and q05 is the realised minimum.

## 7. Owner decisions (2026-09-27, via main discussion)

- A NATIVE window collision does NOT veto a plan cell. It is logged.
  Reference: `safety_veto_v2(native_collision_vetoes=False)`; planner:
  `PlannerConfig.native_collision_vetoes_plan=False`.
- Rule 4 additionally requires the commit WINDOW alone to be non-increasing
  for TTC<3 exposure and hard braking (same mean and ≥ 90%-of-cells tests on
  the window-only delta), on top of the summed window + future rule.
  Reference: `window_must_not_increase=True`.

## 8. Gating item G-2 (2026-09-27, main discussion)

- New meta fields: `relative_safety_validated: bool` and
  `relative_safety_validation_id: str|None` (non-None iff True).
- They cover the learned model's PAIRED (plan − native) TTC<3 exposure and
  hard-braking deltas, both in the window and in the future.
- Reason: the absolute TTC<3 exposure forecast of the native Part A emulator
  does NOT beat climatology (WD-01 p3n: CRPS 4.98 vs 3.01 at H=30). The
  paired deltas are unvalidated until a check against perfect-simulator truth
  passes: sign agreement and rank correlation, reported separately for pairs
  whose true delta is clearly nonzero.
- Consumer rule: while False, relative-safety vetoes computed from a learned
  prediction are UNVALIDATED. The planner refuses live or closed-loop mode
  (same as G-1), and controlled experiments must use the perfect simulator
  for safety checks, or log the gap explicitly.
- The perfect simulator reports True with its own validation id.

