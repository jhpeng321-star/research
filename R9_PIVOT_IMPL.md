# Round 9 A-only P1–P3 implementation

## Disposition

Implemented as **`model.merge_decision = "first_passage"`**, default off.
This document supersedes the hazard/temperature contract in `R9_INTERFACE.md`
**for this mode only**, following `ROUND9_ADVICE_PIVOT.md` items 1–19.
Implementation and synthetic checks are delivered; **qualification is PARTIAL**.
LCSTATE does not establish SUMO's internal longitudinal/LC geometry ordering.
No simulator-substep parity, 2,915-event support audit, real TRAIN smoke, fitted
override, grouped selection, or launch qualification is claimed.

Commits on `c4/v10r9w2`:
- `a60d1414`: shared exact first-passage primitives and 73 tests.
- `d679c0e6`: causal rollout, exact training integration, compatibility and tests.
- The subsequent config/documentation commit is identified in the final execution report.

## Files

- `src/wms/model/merge_pivot.py`: seven positive parameters, scene geometry,
  motion primitive, persistent state, exact analytic masses, ST estimator,
  latent-independent geometric opportunity allocation and final-lane conflict resolver.
- `src/wms/model/wms.py`: opt-in construction, checkpoint parameter migration,
  single causal LC substep, hard commits, latent lifecycle/carry, diagnostics,
  ST lane weights carried into the next residual-network input.
- `src/wms/model/spec.py`, `checkpoint.py`: new mode/width and old-checkpoint loading.
- `src/wms/train/losses/merge_pivot.py`: frozen causal recorded replay, analytic
  recorded/on-policy sequence scores, persistent posterior and support diagnostics.
- `src/wms/train/losses/compose.py`, `src/wms/train/unroll.py`: exclusive new-mode
  composition, recorded replay preparation, counters, explicit nonfinite-loss reporting.
- `src/wms/config/train.py`, `legacy_flat.py`, `src/wms/train/settings.py`:
  nested/flat config plumbing and B-off validation. Default new flat fields are omitted.
- `configs/c4_wm_v10round9.toml`, `tools/launch_round9.py`: draft recipe and local
  round-8 stage300 initializer.
- `tests/test_merge_pivot.py`, `test_merge_pivot_training.py`,
  `test_merge_pivot_runtime.py`: 126 new CPU contract checks.
- `tests/test_merge_core.py`: replace a developer-local checkpoint/skip with a
  workspace synthetic old-format checkpoint through the real loader.
- `tests/test_merge_core_v2_config.py`: correct stale placeholder-normalizer assertions
  to the already-established round-8 values 395.2 and 2.15; no round-8 config changes.

## P1: implemented WM transition contract

An event at output row `t+1` labels the lane transition during `[t,t+dt]`,
with `dt=0.5 s` in this recipe. The first post-merge row authorizes the event;
its position, acceleration, neighbors and cached LC bits are **not** decision inputs.

1. Snapshot identities, lanes, presence and current kinematics at `t`.
2. Evaluate the longitudinal network/physics branch once from current inputs/history.
3. Call shared `advance_longitudinal` once: `v'=max(v+a*dt,0)` and
   `s'=s+(v+v')*dt/2`, or the existing stop-time distance when configured.
4. Execute available current CAV commands/controller once. In the new mode a recorded
   future CAV teacher is not used. Without commands/controller, retain the frozen
   native longitudinal prediction and count this fallback in recorded diagnostics.
5. Build LC geometry at `(s',v')` with **pre-change lane membership**, including CAVs
   and every present target-lane vehicle. Recorded replay calls this same geometry
   entry point after the same motion/current-command execution.
6. Allocate exact-interval merge opportunities before willingness: current target
   occupancy and immutable CAV insertions authorized by current actions veto overlaps;
   overlapping ramp candidates use stable slot priority. Reserved target intervals
   block competing learned mainline insertions independently of any latent draw.
   Recorded replay uses the same allocator. Neighbor features still use pre-change lanes.
7. Evaluate the learned boundary/crossing once, then resolve remaining final occupancy
   conflicts without padding. Commit lanes once; an invariant requires every pivot
   crossing to commit. Remaining mainline conflicts use stationary/fixed occupancy,
   stable slot priority, and cancellation-cascade rechecks.
8. Apply road exits. Padded projection is refused in this mode. The legacy lane-end
   clamp/forced merge window is bypassed; no timer, gap relaxation or hazard floor
   supplies acceptance or hides noncompletion.

Positions use front bumpers: occupied interval `[s-length,s]`. Follower clearance
subtracts **ego** length; leader clearance subtracts **leader** length. All-target
strict interval intersection vetoes eligibility, including side-by-side vehicles
and non-nearest overlapping vehicles. Touching is zero intersection, but a present
side with zero clearance has zero learned passage support. There is no neighbor
distance cutoff or added clearance. Missing sides are unconstrained; a nonfinite
present observation vetoes the scene rather than pretending the neighbor is absent.

This ordering is enforced/tested as the **WM contract**, not asserted to reproduce
unlogged SUMO internals. LCSTATE §§1.4 and 10 leave that qualification unresolved.

## P2: two-sided learned boundary

All seven new parameters use plain softplus, without an added parameter floor:

`d_f = s_f + T_f*v_f + relu(v_f²-v²)/(2*b_f)`

`d_l = s_l + T_l*v + relu(v²-v_l²)/(2*b_l)`

`c_t = min(log(g_f/d_f), log(g_l/d_l))` on eligible positive finite clearances.
Absent sides contribute `+inf`; vetoes and constrained zero gaps give `-inf`.
There is no anticipation, residual/context correction, individual fitted location,
behavioral clipping or hand-set acceptance cutoff in this boundary.

Loading round 8 duplicates its **loaded effective** `s0,T,b` to both sides.
The old secure-gap effective scalars include that implementation's historical
`1e-6` parameter offset; migration copies their values, without adding a new floor.
Initialize `tau = old_sigma*sqrt(3)/pi`, matching Gaussian/logistic log-scale
variance. This is an initialization, not a behavioral temperature selection.
The old anticipation, residual, temperature, alpha-location and yield tensors are
retained unchanged for checkpoint transfer, frozen, and never executed in this mode.
Native pivot checkpoints load their seven tensors verbatim instead of reinitializing.

## P3: persistent first passage and exact supplied-path scoring

At first risk entry draw one `z ~ Logistic(0,1)`; `a=exp(tau*z)` has median one.
The hard crossing is `E_t and c_t + tau*z >= 0`. An eligible road with neither
neighbor side gives certain immediate passage. RNG endpoints are redrawn, not
tail-clipped. There is no per-tick acceptance coin.

`z`, running maximum, presence, identity, started and terminal flags are explicit
rollout-state tensors and survive nonzero chunk offsets. Identity/presence changes
reset them; neighbor switches do not redraw a driver. Leaving risk or committing
an event is absorbing for the same identity. Optional tensor `vehicle_id`/`identity`
is lifecycle metadata only, never a boundary feature. Without identity metadata,
slot identity is defined by contiguous presence, as in the existing data interface.

For a supplied eligible path:

`C_t=max(C_(t-1),c_t)`, `F_t=sigmoid(C_t/tau)`

`p_event=F_t-F_(t-1)`, `S_t=1-F_t`.

Finite increasing logits `a=C_prev/tau`, `b=C_cur/tau` use the stable exact form
`log(p)=logsigmoid(b)+logsigmoid(-a)+log(1-exp(-(b-a)))`.
Infinite limits are explicit. Non-record-setting events have `log(p)=-inf` and
infinite NLL; zero-mass/physical-failure counts are retained and the runner reports
them before rejecting a nonfinite optimization loss. **No epsilon likelihood floor**.
Exact max subgradients are used; no soft maximum or hazard/quadrature substitute.

Chunk contributions condition on prior survival: wait increment
`log(S_new)-log(S_entry)`, or event increment `log(p_event)-log(S_entry)`.
Recorded frozen-geometry prefix boundaries are replayed to reconstruct the prior
maximum graph, preserving the analytic boundary/tau gradients across TBPTT without
reusing a freed graph. Unknown-age sequences are masked/enumerated unless an
explicit `merge_pivot_initial_maximum` supplies a causal survival posterior.
No future event time or driver truth enters the prior. Events terminate scoring;
horizons are censored and junction/vanished exits are counted separately. On-policy
comparison-support failure is absorbing across chunks and cannot later regain support.

Conflict opportunities are allocated **before** latent-dependent willingness;
otherwise a scalar formula would incorrectly ignore joint-latent selection when
post-crossing arbitration canceled a crossing. Runtime emits `merge_pivot_cross`
and `merge_pivot_committed` and enforces their equality for ramp decisions.
Tests verify that conflicting mainline draws and ramp willingness do not change
the supplied-path CDF/event mass. The declared scheduler can defer a mainline
insertion even when its reserved ramp driver waits. This is an explicit conflict
ordering choice, not a clearance parameter or a claim about SUMO's internal priority;
its physical/event support and service effects require the real-data audit.
These are conditional recorded-path probabilities, not a joint free-running scene CDF.

Straight-through rollout backward is explicitly a surrogate:
hard forward plus the gradient of `sigmoid((c+tau*z)/merge_st_width)`.
It does not score likelihoods or change forward timing. Exact vetoes/infinities and
reservation-losing opportunities have zero adjoint. Hard-valued lane weights expose this adjoint
to monitoring losses and the next residual-network lane inputs; discrete neighbor
lookup remains hard. `merge_st_width` is a saved config value, never annealed.

The shrinkage term is `loss.merge_core.tau_reg_weight*tau²`, apportioned across
chunks. Its draft value is zero. Lambda (including zero) and the draft backward
width 1.0 require grouped inner validation/sensitivity checks before freezing a
qualified manifest. Neither is a selected result in this delivery.

## Draft recipe and initialization

The draft extends `c4_wm_v10round2.toml` **directly**, keeps round-8 data/curriculum
and normalizers `Z_TF=395.2`, `Z_ON=2.15`, and uses normalized `0.6:0.4` recorded/on-policy
weights. B supervision and the legacy core yield are off; P4 boundary context is
absent. Existing round-8 longitudinal/courtesy tensors are retained and frozen.
The optimizer has exactly seven new scalar tensors (verified); no legacy tensor
changes during four synthetic optimizer updates. Optimizer state is fresh.

With the required checkpoint placed inside this worktree, use the requested Python:

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/launch_round9.py `
  --init-from runs/active/c4_wm_v10_round8/wm_s_stage300.pt
```

The launcher checks `wm_s/1`, `merge_decision=secure_gap`, and `stage_s=300`, then
supplies `model.init_from` to the existing CLI. Its default is preflight; `--mode run`
uses that CLI's training execution. The actual checkpoint/data are absent here.

## Verification and remaining blockers

All commands used `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe` on CPU.
Pytest temporary roots were inside the worktree.

- `-m pytest tests/test_merge_core.py tests/test_merge_core_v2_config.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_context.py tests/test_merge_pivot.py tests/test_merge_pivot_training.py tests/test_merge_pivot_runtime.py -q --basetemp=E:/agent-work/c4-v10r9w2/.pytest_merge_tmp`
  → **190 passed, 0 failed, 0 skipped**. This explicitly expands `tests/test_merge_core*.py`.
- `-m pytest tests/unit -k 'merge or v8wm or train or config or compose' -q --tb=short --basetemp=E:/agent-work/c4-v10r9w2/.pytest_unit_tmp`
  → **615 passed, 2 failed, 1 skipped, 1313 deselected**.
  Workspace-local absent-data paths were supplied for `WMS_JUNCTIONFIX_DATA_ROOT`
  and `WMS_DATAHYG_DATA_ROOT` to avoid consulting external data caches.
- The two broader failures are unchanged tests:
  `tests/unit/cli/test_launch_script.py::test_run_forwards_mode_config_and_overrides`
  constructs a subprocess environment with `PY=None` because Windows has no `echo`
  executable; and `tests/unit/test_junctionfix.py::test_v4_config_differs_from_v3_only_by_the_switch_and_out_dir`
  expects a stale two-key config delta. Neither failing test nor its config/script
  was changed by this implementation.
- The skipped check is `tests/unit/train/test_replay_flags_merge_zone.py`:
  real TRAIN source `WMS_DATA_MERGE_CALIB`/`WMS_DATA_V5_NATIVE` is unavailable.
- Ruff on the new model/loss/tests/launcher and `git diff --check` pass.

S4 evidence covers exact supplied-path CDF/event/score parity within `1e-6`, shared-z
hard decisions, old-format checkpoint migration and native save/reload, no duplicate
motion/decision, persistent chunks/posterior, extreme gradients, zero masses,
mass conservation/exits, event termination, support absorption, lifecycle/neighbor
changes, all-target geometry and latent-independent simultaneous conflict allocation. Tests use
synthetic workspace checkpoints, not a missing deployed round-8 binary.

Not done: real TRAIN fitting/validation/audit, original 2,915-event physical/timing
support reconciliation, SUMO source/substep/priority parity,
S5 real 120/300 s optimizer/closed-loop smokes, paired safety
metrics, backward-width selection, learned override export, or A-only launch sign-off.
Required actions are to supply workspace-local round-8 stage300 and TRAIN/alignment
artifacts, resolve the two baseline test issues, and qualify the declared LC geometry
and conflict ordering against those artifacts before claiming full acceptance.
