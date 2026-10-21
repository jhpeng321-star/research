# F2: recorded pre-window survival conditioning

Implemented on `c4/v10r9w2`, without changing runtime geometry or the audit tool.

## Enable / data contract

The existing `model.merge_decision = "first_passage"` opt-in makes CLI assembly request
raw prefixes from original pack windows, ordinary C4 sources, and paired C4 sources.
Other modes retain their previous dataset kwargs, record keys and scoring path.
No new behavioural hyperparameter is introduced.

For programmatic dataset construction, pass `merge_pivot_history=True` to
`TrajectoryDataset.from_real_packs`, `.from_arrays`, `.from_pack`,
`build_c4_datasets`, or `PairedBranchSource`. Array sources need explicit
`history_kin`, `history_mask`, `history_target_mask`, and `history_is_cav`.
They may additionally supply `history_action_timeline` (a batched timeline mapping).
Neither short feature history nor long encoder features are treated as a raw scene.

The per-record `merge_pivot_prefix` is a mapping, or `None` when unavailable:

- `states`: float `[H+1,K,7]`, native recorded full-scene kinematics.
- `present`, `is_cav`: bool `[H+1,K]`, historical observations for every slot,
  including neighbours that have left before the window.
- `continue`: bool `[H,K]`, same identity in that slot across each transition.
- Optional `action_timeline`: unbatched runtime timeline fields for decisions before t0.

The final state is the recorded t0 anchor. Only the **H decisions strictly before t0**
are replayed. T0 motion is not evaluated for the prior. Collation preserves prefixes
as a ragged list, one mapping/`None` per batch row; it does not pad fake history ticks.
Entrant padding receives no prior. The supplied prefix must use the window's slot order.

Pack prefixes stop at the same run's start, honour shifted window t0, and read only
`veh_kin`, `veh_mask`, `target_mask`, current ownership and available current commands.
C4 prefixes use the equivalent recorded arrays and full historical command timeline.
No next-state acceleration, future trajectory teacher, driver parameters/truth tables,
or eventual event time supplies a decision input.

## Replay / score semantics

`build_pre_window_history(model, inputs, cfg)` returns a ragged list of frozen
`geometry`, eligible `history_mask`, `[1,K]` `certified`, and scene `n_steps`.
It invokes the existing `build_recorded_geometry` causal path: frozen forward,
one shared longitudinal integration, current CAV execution, and runtime decision
geometry/opportunity allocation. Encoder/controller/recurrent state starts causally
at the available run boundary. Slot reuse clears historical state and re-encodes the
new identity from observations available then. Window-t0 encoder features are never
fed backwards into this replay.

An identity is certified only with observed pre-risk entry and continuous history
through t0, without a prior absorbing event/risk exit or missing scene observation
during its risk exposure. A birth directly into risk remains unknown-age. Earlier
occupants of a reused slot cannot contribute to the current identity's maximum.
An observed entry first appearing at t0 can have an empty pre-window risk exposure.

`pre_window_maximum(model, histories, like)` evaluates the current learned boundary
on detached prefix geometry and returns `[B,K]`:

- `NaN`: uncertified; preserve unknown-age masking.
- `-inf`: complete entry history, but no eligible opportunity before t0.
- Finite / `+inf`: the exact eligible running maximum before t0.

Recorded and on-policy conditional scores use this maximum as the survival-to-entry
prior. Recorded TBPTT recomputes its boundary graph at every chunk, preserving the
pre-window denominator/max gradients without reusing a freed graph. No pre-window
NLL is added to the window's loss. No floors, tail clipping, age priors or boundary
thresholds are added. A non-record-setting event stays infinite NLL; impossible
survival after a `+inf` prior is reported as impossible, not repaired.

The existing `targets.prepared["merge_pivot_initial_maximum"]` manual override remains
supported: non-NaN entries override the recovered prior; NaNs defer to recovery.

## Audit API

After `MergePivotTerm.prepare(targets)`, inspect
`prep = targets.prepared["merge_pivot"]`:

- `known_before`, `known`: interval masks before/after automatic recovery.
- `pre_window_initial_maximum`: detached snapshot at prepare-time parameters.
- `pre_window_history`: frozen histories, for recomputation at another boundary setting.
- `unknown_age_sequences_before`, `unknown_age_sequences`,
  `pre_window_recovered_sequences`: `[batch, interval, slot]` lists.

`merge_pivot_metrics(prep, dt_s)` exposes the lists with a `merge_pivot_` prefix, plus:

- `merge_pivot_unknown_age_before`
- `merge_pivot_unknown_age_after`
- `merge_pivot_pre_window_recovered`
- `merge_pivot_pre_window_replayed_intervals` (scene ticks, not vehicle ticks)

These recovery counts are independent of scoring/support filters and describe
automatic history recovery, before any caller-supplied manual override. Existing
`tf_unknown_age`, `on_unknown_age`, event/mass/support counts continue to describe
the actual scorer. Thus audits can reconcile recovered age with physical support
and positive event mass separately.

## Focused verification and limitations

`tests/test_merge_pivot_pre_window.py` covers conditional analytic NLL, independent
chunk backward/gradient parity, on-policy prior use, unknown-history cases, slot
reuse, entry at t0, exact infinite limits/zero event mass, ragged/padded slots,
pre-t0-only state/command reads, long-history causality, C4 loader opt-in, and
default-off behaviour. Existing first-passage training/runtime/math tests are also
run with the requested CPU Python and workspace-local pytest roots.

Final focused run: **161 passed, zero failed/skipped**, using
`C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` and
`--basetemp=E:/agent-work/c4-v10r9w2/.pytest_f2_tmp`. This includes all four
`tests/test_merge_pivot{,_pre_window,_training,_runtime}.py` files and four
selected synthetic C4/data-loader regression tests. One existing read-only NumPy
warning comes from `src/wms/data/pack_history.py:45`. Ruff on the implementation
files/new tests and `git diff --check` pass.

No real-data audit or fitting is performed here. Window-bound switching providers,
hygiene-directed pack plans, missing historical CAV commands, unmatched/filled t0
scenes, and truncated entry histories remain unknown-age. C4 branches without
recorded pre-branch entry history cannot acquire that history from another arm.
Historical command replay uses the registered 0.5 s physics / 1 s controller clock.
Prefixes replay from the available run start: memory and replay cost scale with
that prefix length; no approximate history cap or cross-window replay cache is used.
Unrecorded simulator/controller state before that start is not reconstructed.
Existing on-policy TBPTT carry remains detached; full analytic boundary-gradient
parity is asserted for the recorded frozen path.
