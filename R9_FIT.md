# Round 9 stage-2A offline fitter

## Status

Implements `ROUND9_ADVICE_STAGE2A.md` items 22–26. The default entrypoint is
`tools/merge_fit_v2.py`, which dispatches to `tools/merge_fit_stage2a.py`.
P4/B remain off. Shared runtime geometry/causal update code was merged from
`c4/v10r9w2` in `501af88a`.

**Evidence verdict: CHANGES; execution is partial against the full brief.**
Exact-head parity and F1/F2 contracts pass. The eight-world smoke completes,
but its shared causal replay vetoes 422/608 events, scale quadrature is not
converged, authoritative run physics is missing, and cold full preparation
exceeds one hour. These failures are reported, not repaired with floors or
supported-only fitting. The real one-in-five TRAIN audit remains unopened.

## Comparator identity: two distinct laws in the actual round-8 code

`data/r9_train/wm_s_stage300.pt` is loaded with `weights_only=True`, strict
loading of all 149 tensors, CPU/evaluation mode, and no checkpoint mutation.
The complete driver encoder, learned alpha-location head/bias, residual,
temperature and persistent normal latent are retained.

An important incompatibility in the requested parity criterion is now tested:
the actual `wms.train.losses.merge_core_seqtf.seq_tf_chunk` **does not use** the
learned location or residual, whereas deployment's `_apply_merge_core` does.
They cannot be called the same mass function for this checkpoint. The fitter
therefore exports both, without changing the frozen runtime to force agreement:

- **M0:** untouched full learned deployed head, integrated over one persistent
  normal draw, evaluated on the common corrected causal geometry and F2
  observation protocol. Parameters, encodings and longitudinal dynamics are
  frozen; observation bookkeeping changes are explicit.
- **M0-original-runtime:** the same full frozen head on original runtime
  pre-motion geometry/envelope. Separates the original-runtime/bookkeeping
  effect from the candidate's parameter changes.
- **M0-seqTF:** the actual frozen `seq_tf_chunk` law, with zero location and no
  residual. Its masses/NLL are tested against actual `seq_tf_chunk`, including
  a real TRAIN merge window. Full-cohort scoring does not copy legacy
  unsupported-event omission.
- **M0-refit:** complete decision-head refit, including location and residual,
  on the same corrected geometry/F2 protocol as M0. The encoder/dynamics and
  yield head stay frozen on the cached fixed path.
- **M0-approx:** the former scalar-only M0, explicitly an approximation.
- **Soft-hazard:** corrected full-head soft-hazard fallback treatment. Its
  matched refit can be the same architecture/fit as M0-refit. Item 26 requires
  improvement against untouched exact M0; refit comparisons are descriptive,
  with no self-superiority gate. Absolute/relative timing and tail gates remain.

Full-head quadrature is separately tested against `_apply_merge_core` on
identical inputs, with a nonzero learned alpha head and a deliberately nonzero
residual clone. Decision masses/CDF use the advice's `1e-6` absolute parity
tolerance. This establishes identical-input parity, not a reconstruction of
unrecorded simulator inputs or joint commit-conflict likelihood.

## F1/F2 and diagnostic laws

**F1** learns six positive gap parameters plus `tau=softplus(rho)` and
`w=softplus(eta)`, with no offset in latent location:

```text
d_f = s_f + T_f*v_f + relu(v_f^2-v^2)/(2*b_f)
d_l = s_l + T_l*v   + relu(v^2-v_l^2)/(2*b_l)
c   = min(log(g_f/d_f), log(g_l/d_l))
h_t(z) = E_t * sigmoid((c_t + tau*z)/w), z ~ Logistic(0,1) once per identity
```

Absent sides contribute `+infinity`; present zero gaps contribute `-infinity`.
Structural eligibility, every target-lane interval intersection, and shared
latent-independent reservations remain hard vetoes. Infinite atoms and exact
zero masses are preserved. Independent conditional step noise does not redraw
the persistent driver latent. Conditional log survival is accumulated before
positive-weight quadrature/log-sum-exp; marginal hazards are never multiplied.

`P1`, `P1+P2` and strict analytic `P1-P3` remain attribution diagnostics. P1-P3
uses the exact eligible running maximum and retains non-record zero masses.
The F1 finite-margin `w -> 0` limit, normalization, vetoes, deterministic atoms,
extreme likelihoods, nonzero gradients and save/reload are covered by tests.

**F2** recovers available arrays from actual risk entry through the first
scored decision. `score_start` counts prefix decisions; history is conditioning
data, not scored exposure. Subtract `log S_(b-1)` from subsequent full-path
log masses/survival. This preserves the posterior across windows and makes
chunk scores telescope without double-counting prehistory. The same rule is
applied to every comparator/candidate.

Already-waiting cars with complete recorded prefixes are not unknown-age
exclusions. Truly missing history and zero conditioning denominators remain
unsupported, with infinite NLL and explicit flags; there is no population-prior
reset. Competing exits remain counted separately. Without an identified exit
model they have unsupported full-outcome NLL, not fabricated merges or
administrative censor scores.

## Determinism, observation scope and cohort

- Only workspace-local, nonrestricted TRAIN paths are opened. Resolved symlinks
  and traversal are checked; paths containing `sealed`, `holdout`, `heldout`,
  `cal_split`, `chunk5`, `j4b`, or `inc1`, and seeds 964000–964199, are refused.
  Guarded observation arrays use `allow_pickle=False`.
- CPU only, deterministic PyTorch, seeded bootstrap, one tensor thread per
  episode worker. Independent CPU episode preparation can use `--workers 6`;
  ordered aggregation is deterministic. F1/strict laws use float64; frozen WM
  heads use checkpoint float32. At 64 GH nodes, genuine positive float64 tail
  weights avoid float32 underflow without floors or dropping nodes.
- Shared `merge_pivot.advance_longitudinal` advances each causal input exactly
  once, retaining pre-change lanes. Future recorded positions never supply
  decision geometry. CAVs remain neighbors. Current replay uses the explicit
  WM-physics CAV fallback; it does not pretend native/PI controls were replayed.
- The frozen replay protocol is encode-once/no observational EMA, k=0, causal
  histories/wait/lane accumulators and persistent birth encoding. The reference
  helper accepts exact alternative runtime input/state overrides; different
  observation schedules are not silently equated.
- The copied checkpoint lacks authoritative `PhysicsConfig` metadata or a run
  sidecar. Without `--physics`, the required `--allow-unverified-physics`
  switch labels replay **DIAGNOSTIC**. Repository defaults are not evidence of
  simulator substep, exact lengths/projection or deployed controller parity.
- One retained cohort is used for every model. Reconciliation enumerates every
  observed risk identity, retained event/non-event and exclusion reason;
  duplicate risk identities are rejected. Paired comparisons check actual
  identities/order, not only equal counts. Event-ID cross-tabs retain physical
  veto, history, conditioning failure and each model's zero-mass flags.
- The LCSTATE 2,915-event cohort is a separate unresolved mapping, not an
  interchangeable denominator for these converted observations.

## Validation and untouched audit

Partition all discovered TRAIN worlds **before** smoke subsampling. Sorted
world indices `4::5` are reserved; all arms/sources/windows of a world stay
together. `--max-worlds` selects a prefix of the remaining fit worlds, so even
an eight-world smoke cannot inspect the reserved audit.

Fit worlds use grouped round-robin inner folds. Select the Cartesian penalty
grid, including `(0,0)`, for the exact F1 sum objective
`sum NLL + lambda_tau*tau^2 + lambda_w*w^2`. No scale cap or annealing endpoint
is imposed. All refits use the declared iteration/LR budget. A failed fit or
inner validation cannot be selected; invariant support failures abort the whole
fit and retain attempted parameters. No car is dropped to make fitting pass.

The audit is opened only with `--audit-once`, after primary fits/selection and
authoritative physics are ready. An exclusive
`results/r9_stage2a_audit_use.json` ledger is created **before** audit observation
reads. A failure still consumes that attempt; a second invocation is refused.
Do not delete that ledger to retune. Synthetic tests use their own isolated
ledger. Full-model weights, fit choices, checkpoint/input/implementation hashes,
cohort partition, physics and observation protocol are recorded.

Reports include paired whole-world bootstrap NLL/early-mass uncertainty,
event/censor/exit strata, NLL tails/worst world, timing/noncompletion bounds,
source/queue/demand tails, history coverage, and learned/attempted scales.
Quadrature refinement reports masses, NLL and gradients on fit worlds only;
nonconvergence is not silently certified. No post-event continuation is used.

## Exact commands

Run from the worktree root. All artifacts and test scratch directories stay
inside it. The interpreter is the supplied CPU environment.

### Verified eight-world smoke (audit unopened)

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/merge_fit_v2.py --source v5_native=data/r9_train/v5_native --source genrand_plain=data/r9_train/genrand_plain --models Soft-hazard,F1,P1,P1+P2,P1-P3 --allow-unverified-physics --max-worlds 8 --workers 6 --inner-folds 2 --iters 4 --n-quad 16 --shrinkage 0 0.01 --bootstrap 100 --seed 73 --out results/r9_stage2a_smoke8.json --md results/r9_stage2a_smoke8.md
```

### Full fit-world diagnostic command using the materialized inputs

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/merge_fit_v2.py --source v5_native=data/r9_train/v5_native --source genrand_plain=data/r9_train/genrand_plain --round8-params data/r9_train/wm_s_stage300.pt --models Soft-hazard,F1,P1,P1+P2,P1-P3 --allow-unverified-physics --workers 6 --inner-folds 4 --iters 100 --n-quad 16 --shrinkage 0 0.01 1 --bootstrap 2000 --seed 73 --out results/r9_stage2a_full.json --md results/r9_stage2a_full.md
```

This is the full-size fit/inner-validation command, **not an executed full fit
or an authorized audit/launch**. It uses all non-audit worlds across both
sources and leaves audit observations unopened. Before the coordinator's one
final audit, supply the authoritative workspace-local physics file, resolve
support/exit/quadrature failures and freeze the candidate/grid. The exact final
command is:

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/merge_fit_v2.py --source v5_native=data/r9_train/v5_native --source genrand_plain=data/r9_train/genrand_plain --round8-params data/r9_train/wm_s_stage300.pt --models Soft-hazard,F1,P1,P1+P2,P1-P3 --physics data/r9_train/round8_physics.json --workers 6 --inner-folds 4 --iters 100 --n-quad 16 --shrinkage 0 0.01 1 --bootstrap 2000 --seed 73 --audit-once --out results/r9_stage2a_full_audit.json --md results/r9_stage2a_full_audit.md
```

`data/r9_train/round8_physics.json` is a required owner-supplied input, currently
absent; do not manufacture it from defaults. Failed primary fits or missing
authoritative physics refuse audit consumption even if requested.

Replay caches under `results/r9_stage2a_cache` use safe tensor loading,
checkpoint/physics/observation/preparation-code hashes and atomic writes.
Changing only scoring/penalties does not require repeating frozen dynamics.
The old scalar CLI requires `--legacy-approx`; its exported M0 labels are
renamed M0-approx. It is not the default qualification path.

## Measured results and remaining blockers

- Eight globally selected **fit** worlds, eight native episodes (the sorted
  prefix contains no genrand worlds): **923 observed risk identities = 633
  retained + 290 exclusions**, 608 retained events, 25 non-events, complete
  entry histories for all retained cars, **35 prefix-conditioned survivors**.
- Cold compacted six-worker preparation **379.145 s**, fitting/diagnostics
  **92.604 s**, total **471.785 s**. A scoring/manifest verification using the
  same valid caches took **112.104 s** (18.149 s preparation/load, 93.771 s
  diagnostics). This repeat did not open the audit.
- Common corrected physical support: **186/608 events**. Exact M0, fallback,
  F1, P1 and P1+P2 each have **422 exact zero-mass events**. Strict P1-P3 has
  **451**. Original-runtime M0 has four; actual seqTF has zero. These are
  different geometry/law diagnostics, not grounds to delete events or soften
  vetoes. Refits correctly stop with `exact_support_failure`.
- Refinement 32→64 nodes changes fallback masses by as much as **0.05581** and
  F1 by **0.0003261** on this attempted initialization. This is not converged
  numerical evidence. Unsupported-car NLL/gradient convergence is explicitly
  unavailable.
- Cold full-preparation extrapolation from this dense native prefix is
  **9,099 s (~2.53 h)** before full fitting/diagnostics. Sparse attention,
  absent-slot compaction, batching, CPU workers and reuse improve cost, but
  **the cold <1 h laptop criterion is not met**. Cached smoke timings are not
  a claim that the uncached full corpus meets that budget.
- Exact missing actions/physics, simulator geometry/substep evidence,
  informative-exit handling, original LCSTATE mapping, F3 support repairs,
  numerical convergence and runtime F1/F2 parameter/forward-law parity remain
  prerequisites for qualification. F1/F2 cannot legalize a physical veto.

## Verification and commits

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest tests/test_merge_fit_stage2a.py tests/test_r9_fit_exact.py tests/test_r9_exact.py tests/test_merge_fit_v2.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core.py tests/test_merge_core_v2_onpolicy.py tests/test_merge_pivot.py tests/test_merge_pivot_runtime.py tests/test_merge_pivot_training.py --basetemp tmp/r9_stage2a_regression -q --tb=short
```

The final selected fitter/runtime regression suite passed **329 tests, zero
failures and zero skips**. The stage-2A file includes **18 tests**, including
actual single-use synthetic audit, fit-only
global-audit isolation, two-worker deterministic cache reuse, F1/F2,
identity/order reconciliation, width save/reload and fallback gate amendment.
Ruff passed for all six new helper/test modules. No full repository/SUMO launch
qualification is claimed. Generated reports/weights/caches remain ignored in
`results/`; the source data and unrelated untracked documents are preserved.

Implementation commits on `c4/v10pfit`:
- `501af88a`: merge shared first-passage runtime branch.
- `1d9d00a1`: exact frozen complete heads and separate actual seqTF parity.
- `e84d1833`: F1/F2, matched corrected comparators, audit/cohort protocol and
  bounded/parallel cached replay.
- Documentation/evidence handoff commit: `git log --oneline -1` after this file.
