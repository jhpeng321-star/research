# Round 9 A-only offline fitter

## Status and scope

`tools/merge_fit_v2.py` replaces the superseded DeepSeek context-model draft at
`72ae890d`. It implements the nested offline comparisons and exact sequence
statistics from `ROUND9_ADVICE_PIVOT.md` S3a/S3b. P4 and B are off.

**Current evidence verdict: CHANGES.** This is an offline recorded-path tool, not
launch authorization. In particular, this worktree contains neither a supplied
round-8 stage300 dynamics checkpoint nor verified simulator decision-substep
geometry. The default constant-current-acceleration proposal is explicitly
**DIAGNOSTIC**. It cannot qualify the required frozen longitudinal branch. A
stepwise adapter is supported for integration with the runtime agent's shared
causal update; the external ordering/geometry evidence remains necessary.

The two materialized `_sample` worlds are corpus-TRAIN, converter-`val` worlds.
They are suitable for a development smoke, not the fixed audit or the original
LCSTATE 2,915-event qualification cohort. Missing manifest episodes, original
LCSTATE denominator, its 16 no-left-want rows and two teleports, and actual
sample exclusions are recorded explicitly. No LC flags are used as model inputs.

## Commands executed (PowerShell, worktree root)

All paths used for data, artifacts, test scratch space and adapters must resolve
inside this worktree. The executable below is the owner-specified CPU interpreter.

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest tests/test_merge_fit_v2.py tests/test_merge_core_v2_seqtf.py tests/test_merge_core.py tests/test_merge_core_v2_onpolicy.py --basetemp tmp/r9_pytest -q
```

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/merge_fit_v2.py --source v5_native=_sample/v5_native --models M0,M0-refit,P1,P1+P2,P1-P3 --audit-frac 0.2 --inner-folds 2 --bootstrap 100 --round8-params default --out results/r9_fit_sample.json --md results/r9_fit_sample.md --seed 73 --max-worlds 2 --iters 4 --shrinkage 0 0.01
```

The smoke completes with 167 HDV sequences (163 events, four non-events), 162/163
physically supported events under the diagnostic proposal, and exact zero-mass
event counts M0=0, M0-refit=0, P1=1, P1+P2=1, P1-P3=40. Failed candidates retain
their attempted initialization and an explicit fit failure, not fitted-success
labels. M0-refit takes four updates. The two-world audit is empty by design;
sample fit diagnostics are labeled `fit_diagnostics_not_audit`. No audit
confidence interval is fabricated. Generated artifacts stay in ignored `results/`.

For the coordinator, after placing full TRAIN exports, the exact checkpoint or
scalar config, and the shared causal adapter **inside this worktree**, use:

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/merge_fit_v2.py --source v5_native=data/r9_train/v5_native --source genrand_plain=data/r9_train/genrand_plain --models M0,M0-refit,P1,P1+P2,P1-P3 --audit-frac 0.2 --inner-folds 4 --bootstrap 2000 --round8-params data/r9_train/wm_s_stage300.pt --causal-adapter tools/r9_decision_adapter.py --out results/r9_fit_full.json --md results/r9_fit_full.md --seed 73 --iters 200 --shrinkage 0 0.001 0.01 0.1 1
```

This is a handoff command, not a claim that these full-data/checkpoint/adapter
files are present or that this command has been run. Freeze the source cohort,
adapter and hyperparameter grid before the single final audit. Do not tune from
audit results. `--max-worlds` selects a sorted prefix and lists every omitted
world; use it for development, not as an audit-cohort repair.

## Input and baseline identity

- `--source NAME=DIR` is repeatable. `DIR` contains a corpus-`TRAIN`
  `C4_EPISODES_MANIFEST.json` and converted `episodes/<id>/*.npy` files.
  Read-only inputs are `veh_kin`, `veh_mask`, `vid`, `is_cav`, `time_s`, and,
  for an adapter, issued control channels. Arrays use `allow_pickle=False`.
  No pickle, truth table, SUMO run, or post-event hypothetical path is used.
- IDs only establish identity segments, action validity and world grouping;
  they are not behavioral inputs. Slots are split on absence or ID replacement.
  CAVs remain full-scene neighbors but are listed as supervision exclusions.
- `--round8-params default` reads and hashes
  `src/wms/model/merge_core_init.json` strictly, without copying rounded constants
  from the brief. **This file currently contains s0=2.1609596641, T=0.8617146481,
  b=1.9684840575, tau_a=1.0793451041, temperature=1.2781399549,
  sigma=0.1275461806, not the stage300 values quoted in the brief.** For actual
  stage300 evidence, supply the exact checkpoint/config.
- JSON configs use `s0_m`, `T_s`, `b_mps2`, `tau_a_s`, `temperature_m`,
  `alpha_log_sigma`, either at top level or under `fitted`. Checkpoints are loaded
  with `weights_only=True`, CPU mapping; standard `model`, `state_dict` and
  `model_state_dict` wrappers are supported. Their positive raw parameterization
  and sigma transform match `MergeCoreNet`.
- The untouched M0 is the **context-free round-8 seq-TF comparator**. Its scalar
  parameters are frozen exactly; its per-step anticipation, 150m core feature
  search, missing-side semantics, normal persistent latent and sequence
  quadrature remain unchanged. A checkpoint with learned nonzero runtime
  residual/location is refused rather than silently reconstructed without those
  terms. Such a checkpoint needs a complete frozen comparator integration to
  represent its deployed decisions; a scalar export alone cannot establish that
  stronger claim.

## Model and alignment contracts

- `M0`: untouched scalar context-free round-8 soft hazard.
- `M0-refit`: the same context-free soft-hazard architecture, refitted on fit
  worlds only. This is never labeled the untouched baseline.
- `P1`: one causal proposal, pre-change memberships, all-target interval veto,
  legacy soft-hazard secure-gap boundary and persistent normal quadrature.
- `P1+P2`: P1 geometry and physical veto plus separate positive
  `s_f,T_f,b_f,s_l,T_l,b_l`, logistic persistent assertiveness, learned soft-hazard
  temperature. No anticipation or legacy margin residual is added to P2.
- `P1-P3`: those six distances plus learned positive `tau`. One
  `z~Logistic(0,1)` per driver defines `a=exp(tau*z)`, median `a=1`. Conditional
  decisions are deterministic first passage. No per-step random coin is added.

All labels use the **first post-merge/authorizing output row**. The input is the
preceding recorded snapshot. That snapshot is advanced exactly once by the
proposal, retaining its lane membership. Recorded next-row position/velocity
never supplies decision geometry. Events terminate at the observed transition;
censors/exits terminate at observed support. Structural exits cannot regain
eligibility. Temporary physical overlap is a veto, not a permanent identity exit.

The physical mask uses every present target-lane vehicle, including CAVs and
side-by-side vehicles, without a search radius or learned padding. For front
coordinates, each vehicle occupies `[s-length,s]`; positive interval intersection
is vetoed and zero intersection is permitted. Leader gap subtracts leader length,
follower gap subtracts ego length. Converted lengths are nominal, so exact real
length/projection parity still needs external evidence. A physical-mask repair
is reported independently from model architecture. M0 is not retrofitted with it.

For P2:

```
d_f = s_f + T_f*v_f + relu(v_f^2-v^2)/(2*b_f)
d_l = s_l + T_l*v   + relu(v^2-v_l^2)/(2*b_l)
```

For P3, absent sides contribute `+inf`, a veto contributes `-inf`, and positive
finite gaps give `c=min(log(g_f/d_f),log(g_l/d_l))`. A present zero gap has
`c=-inf`. `C_t=max_{u<=t}c_u` and `F_t=sigmoid(C_t/tau)`. The event likelihood is
exactly `F_t-F_(t-1)` and censor/competing-exit survival is `1-F_t`.

The implementation evaluates finite record increments with:

```
log p_t = logsigmoid(new) + logsigmoid(-old) + log(-expm1(old-new))
```

where `new=C_t/tau`, `old=C_(t-1)/tau`. Non-records have exact log mass `-inf`.
Free-road `+inf` consumes the remaining survival as an immediate atom. Both
extreme tails have stable log masses. There is no epsilon likelihood floor, soft
maximum, gate rescue or behavioral clearance threshold. M0's existing internal
numerical clamps are retained solely for exact legacy parity.

## Fitting, splits, uncertainty and reports

- Worlds are sorted, audit indices are `4::5`, and every arm/window of a world
  stays together. An insufficient sample never borrows fit worlds for audit.
- Inner folds are deterministic round-robin world groups within the other
  four-fifths. The P2/P3 shrinkage grid includes zero and selects `lambda*tau^2`
  by inner mean validation NLL. Empty folds or failed fits cannot be selected.
  Other refits use the same iteration/LR budget. The soft P2 comparator integrates
  one persistent Logistic draw by Gauss-Legendre population-quantile quadrature;
  P3 integrates that same population law analytically.
- CPU float64, single-thread PyTorch, deterministic algorithms and seeded NumPy
  bootstrap are used. Full per-car gradients are accumulated over bounded
  128-car batches; geometry temporaries use 32 time rows. Batching does not reset
  driver quadrature or change the whole-sequence objective.
- Before optimizing any P1 candidate, every fit event must pass the physical
  mask. Nonfinite exact event/censor likelihoods abort that fit. Parameters are
  not fitted on a supported-only subset. The trace records finite/nonzero raw
  gradients on successful fits; zero-mass failures remain visible.
- Audit output includes every per-car loss, event/censor/exit strata, mean,
  nearest-rank p50/p90/p99, maximum, worst-world mean, physical-support failures,
  positive-margin share, exact zero-mass count, and fitted/attempted parameters.
- The paired CIs versus **both** M0s resample entire worlds, retaining all cars
  in each sampled world. The audit estimand is mean per-car delta (resampled
  sums/resampled counts); early mass uses the observed-event cohort. Standalone
  `world_bootstrap` defaults to equal-world means, with explicit weighting in
  the result. A single world is marked degenerate and cannot certify a gate.
- `e_i` is model median since zone entry minus observed event time. Unreached
  medians are lower-bounded, with infinite upper bounds. Unknown-age events stay
  in full-cohort summaries as uninformative bounds and are enumerated; known-age
  likelihood starts at zone entry, using available earlier observation history.
  Unknown-age likelihoods explicitly use the observed-start population prior;
  no unavailable pre-entry posterior is fabricated.
- Early mass is `F_i(t_i-)`, not event hazard. Population CDFs are mixtures of
  recorded-path CDFs, with post-support bounds. p25/p50/p90 errors are emitted
  only when identifiable; these are not free-running distributions.
- Noncompletion is reported at 120/300s since entry, by source, initial queue
  front/behind and demand. Complete empirical outcomes get world-bootstrap 95%
  bands. Post-event model survival stays bounded; observed competing exits
  consume remaining non-merge mass separately. Model bounds, absolute-error
  bounds, counts and certification against both M0s are explicit. No post-merge
  road continuation or invented exit-to-merge event is used.
- S3b gates use conservative full-cohort timing and noncompletion bounds.
  Insufficient or infinite uncertainty cannot pass. JSON is RFC-compatible;
  exact infinite losses/bounds are strings `+Infinity`/`-Infinity`, never floors.
  Detailed exclusions and per-car evidence live in JSON; Markdown is a summary.

## Shared causal adapter interface

The optional workspace-local Python file exports:

```python
def reset_episode(): ...  # optional; required for a stateful adapter

def decision_state(*, kin, present, is_cav, dt_s, actions):
    # kin [K,7], present/is_cav [K]; current snapshot only.
    # actions: None or current issued controller, issued_acceleration_mps2,
    # issued_target_speed_mps, valid; each [K], identity-bound at issuance.
    # Return numpy-compatible [K,7] geometry at the verified LC substep.
    # Preserve slot order/identities and pre-change lane one-hots.
    ...
```

Calls are strictly chronological, once per input tick. No future snapshot,
observed event, LC flags, world ID, vehicle ID or truth parameters are passed.
Load the frozen dynamics and numerical update contract in this adapter, with
workspace-guarded paths. If driver history needs identity reset masks beyond
absence detection, extend the shared integration contract before qualification.
An adapter hash is recorded, but supplying an adapter does not assert SUMO
substep parity. This fitter deliberately cannot self-sign that evidence.

## Verification and commits

The documented four-file CPU test command passes **107 tests, zero failures and
zero skips**. The documented real-data smoke exits successfully and writes
`results/r9_fit_sample.json` and `results/r9_fit_sample.md` with verdict CHANGES.

Synthetic tests cover exact running maxima, zero record increments, mass
conservation with censoring/competing exits, extreme logs and gradients,
infinite atoms/vetoes, persistent quadrature parity with `seq_tf_chunk`, alignment
through the real episode loader, one causal advance, same-slot ID replacement,
geometric overlap, all four refits, grouped bootstrap reproducibility, strict
JSON, workspace/seed refusals, and a ten-world end-to-end audit CLI.

Implementation foundation: `6242477e` (43 initial contract tests).
Audit, timing, tail calibration and causal diagnostics: `59ee4603`.
The final integration/documentation commit follows these on `c4/v10pfit`;
obtain its exact identity with `git log --oneline -4`.

Remaining evidence required: full materialized TRAIN cohort, exact stage300
baseline identity, shared frozen longitudinal adapter, verified simulator update
order and bumper conventions, original-cohort reconciliation, and runtime score/
decision parity. The sample's one physical-support failure and 40 exact P3
zero-mass events require diagnosis, not likelihood flooring or event deletion.
