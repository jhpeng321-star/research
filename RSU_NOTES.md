# C4 deployment world-model check (Option B)

## API and integration

`wms.control.rsu_check.RsuChecker` accepts:

```python
checker = RsuChecker(
    world_models=[wm_member_0, wm_member_1],
    candidates={"policy": learned_factory, "native": native_factory,
                "damping": damping_factory, "pi": pi_factory},
    reward_fn=team_reward,
    weights=RewardWeights(),
    horizon_s=20, decision_period_s=5, n_driver_samples=4, margin=2.0,
    seed=0, log_fn=decision_records.append,
)
choice, imagined_returns = checker.decide(state, history, history_mask=history_mask)
rsu_policy_fn = checker.rsu_policy_fn
accel = rsu_policy_fn(state, history, now_s=simulation_time_s)  # [1,K]
```

Use the batch-first WM-frame state in `wms.control.contract`: a single current
state `[1,K]`, optional short `history` / `history_mask`, and optional separate
`driver_history` / `driver_history_mask` for long-history encoders. Omitted masks
are inferred as all observed when a history is explicitly supplied. Long-history
WMs still require the correctly sized separate driver-history buffer.

`candidates` maps names to **zero-argument factories returning fresh
CavController callbacks** (`state -> [B,K,4]`, next position/lane/speed/acceleration).
Factories isolate mutable lifecycle state; callbacks must handle the imagination
sample batch and single-state execution. The callback's `physics_step_index` is
relative to its rollout or execution-plan start. Execution timestamps are strictly
increasing; without `now_s`, calls are assumed to be one control second apart.
Call `checker.reset()` at each episode boundary. Reset retains the audit log and
continues the local driver-sampling RNG stream.

`default_candidates(model, policy=..., group_encoder=..., safety=..., bridge=...,
damping=..., pi=...)` supplies the default family:

- `policy`: learned actor forced to `deterministic=True` through the injected
  `bridge(policy, group_encoder, safety)` (`make_cav_controller`).
- `policy_group_0`: zero group signal.
- `policy_group_1`: encoder output plus a fixed, reproducible feature perturbation.
- Further `policy_group_k`: injected `alternative_group_encoders`, e.g. encoders
  broadcasting signals from other zones. No true zone/driver information is inferred.
- `native`: WM-S `hold_teacher` (hold speed, not a new IDM implementation).
- `damping` and `pi`: injected factories adapting the frozen classic controllers.

Set `perturbation=0` for one group alternative, i.e. exactly five candidates. The
default perturbation is 0.1. Supply deployed policy/group/safety modules in eval
mode. The checker has no dependency on concurrently implemented control modules
or the SUMO environment. Classic adapters can wrap `TimelineControllers`, but
must respect the runner's one-second cadence and explicit physics-step clock;
its internal half-second call counter is not itself a runner-clock adapter.

The execution callback keeps the selected controller for the decision period,
recomputing closed-loop acceleration on each real state. It retains execution
lifecycle state when a later decision selects the same candidate; it constructs
a fresh controller when the choice changes. Imagination always uses separate
fresh instances. Acceleration is zero on absent/non-CAV slots. Bounds and safety
filtering are the injected controller/bridge's responsibility.

## Batched paired decision

For each ensemble member, all candidate x driver-sample rows are rolled out in
**one** WM call under `torch.no_grad()`, with `cav_teacher=None`. Candidate-major,
sample-minor ordering gives `[C*S,K]`, not C*S independent model calls. Ensemble
members are evaluated serially because each is a distinct model. WM eval mode is
used for encoding and rollout, and original per-submodule training flags are
restored even on failure.

The encoder runs once on the real recent history per member. Physical-unit driver
parameter perturbations use its uncertainty scale, capped to +/-50% of encoded
nonnegative values; embedding perturbations default to Gaussian std 0.05. Both
scales can be configured (`driver_noise_scale`, `embedding_noise_std`). Standard
normal draws come from a checker-local CPU generator and are shared across every
candidate and corresponding ensemble encoding (common random numbers). Sampled
parameters and embeddings stay fixed throughout each imagination. Models must
expose parameters or an effective driver embedding; a legacy residual-only WM
without either is rejected rather than pretending to sample driver uncertainty.

The contract team reward is summed over every imagined physics step in float64;
it is not averaged, discounted, or bootstrapped. The margin is therefore in
**summed team-return units**, and should be chosen for the reward weights/horizon.
For each alternative, paired differences subtract the policy return from the
same ensemble member and sample. Let M be ensemble_members x driver_samples:

```text
paired_mean = mean(alternative_return - policy_return)
paired_std  = sample_std(alternative_return - policy_return)
paired_lcb  = paired_mean - confidence_z * paired_std / sqrt(M)
eligible    = paired_mean > margin AND paired_lcb > 0
```

Default `confidence_z=1.96`; the comparisons are strict. The eligible alternative
with greatest imagined mean return wins. Ties between alternatives use insertion
order. Otherwise keep `policy`, including identical candidates, invalid/nonfinite
paired evidence, or M=1. Invalid numbers are logged as JSON null, not NaN/Infinity.

`decide` returns `(name, evidence_by_name)`. Each evidence record contains `mean`,
`std`, `paired_mean`, `paired_std`, `paired_lcb`, `eligible`, plus `[ensemble,sample]`
raw `returns` and `paired_diffs`. `decision_log` and optional `log_fn(record)` supply
the figure-ready decision number/time, candidate names, imagined evidence, choice,
margin, horizon, confidence multiplier, oracle label, and measured timing.

## CPU timing evidence

Reproducer (PowerShell; Python environment specified in the brief):

```powershell
$env:TEMP='E:/agent-work/tmp'
$env:TMP='E:/agent-work/tmp'
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe' -u tests/unit/test_c4_rsu.py
```

Measured 2026-10-12 on CPU, torch `2.12.1+cpu`, one torch thread, one randomly
initialized hybrid WM-S ensemble member, synthetic 192-slot traffic with every
fifth vehicle a CAV, 5 candidates x 4 driver samples (batch 20), 20 seconds / 40
physics steps. The candidate callbacks are cheap deterministic fakes; the WM
rollout and encoder are real. This measures model/checker execution, not the
cost of learned attention or frozen classic-controller implementations.

After one warmup, three full-decision measurements were **133165.19 ms,
117740.50 ms, 196380.23 ms**; respective rollout-only times were **133115.49 ms,
117695.37 ms, 196341.36 ms**. Median full decision: **133165.19 ms**; median
rollout: **133115.49 ms**. The three-measurement process emitted its complete JSON
report but exceeded the 600-second shell timeout during completion. These are
observed timings, not a successful timed-process exit. A shorter reproducer is
`tests/unit/test_c4_rsu.py --repeats 1 --no-warmup`.

The shorter, no-warmup run completed successfully: **192057.98 ms** full decision
and **191909.43 ms** rollout, with the same CPU/batch/slot/step configuration.
Its slower cold-run timing reinforces the real-time latency limitation.

Timing uses `perf_counter` and reports `total_ms`, `rollout_ms`, batch size, slots,
ensemble size, steps/member and device. An optional injected `synchronize` callable
brackets rollout timing for an asynchronous device; the implementation does not
import a GPU runtime or query/synchronize CUDA itself. All measurements here used
CPU. Hardware contention and additional ensemble members increase latency.

## Verification and outstanding acceptance gates

- Focused CPU tests: **26 passed**. Coverage includes strict paired margin/LCB,
  CRN pairing of parameters and embeddings, ensemble evidence, batch/history
  shapes, stability/decision cadence, episode reset, execution isolation,
  nonfinite rejection, deterministic group alternatives, restoration on failure,
  configuration validation, and a real hybrid WM-S smoke check.
- Full `tests/unit` collection under an in-memory constraints guard:
  **473 passed, 188 skipped, 1 failed** (87.60 s on the final run). The guard prevents protected-path reads,
  scratch writes, deletion, external data access, and subprocess execution;
  scratch-fixture and training-workstream tests are skipped before execution.
  No guard files or pytest caches were written. The failure is the existing
  `data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`
  registry/receipt regeneration mismatch. Its repair is outside this task's
  three-file edit allowlist. This does **not** establish an unrestricted full-suite
  pass. Pytest also reports the existing cache_dir warning when its cache plugin
  is disabled.
- `python -m ruff check --no-cache src/wms/control/rsu_check.py
  tests/unit/test_c4_rsu.py`: passed.

## Limitations

- **Oracle check:** imagination requires the full current WM state. Until the
  sparse-observation reconstructor is integrated, this is privileged oracle-state
  evaluation, not deployable sparse-CAV state inference.
- **Latency:** this CPU measurement greatly exceeds the default five-second
  decision period. The callback is synchronous and does not implement an
  asynchronous real-time scheduler. Deployment needs rollout optimization or an
  appropriate measured compute budget before meeting that cadence.
- Encoder uncertainty, the +/-50% relative cap, and embedding noise are sampling
  heuristics, not calibrated driver posteriors. The relative cap does not enforce
  each encoder parameter's absolute upper/lower calibration range. Ensemble and
  CRN samples can be correlated: the normal-multiplier LCB is a selection heuristic,
  not a calibrated safety guarantee or a multiple-comparison-corrected test.
- Reward TTC costs are not independently optimized by this selector; safety must
  be enforced by the injected safety/controller layer. No terminal-value estimate,
  future external arrivals, reconstructor uncertainty, or SUMO-vs-WM validation is
  added. Native here means the existing hold-speed teacher.
- No checkpoints/training data, SUMO runs, GPU work, or training were used for the
  implementation's tests and benchmark. The timing WM is untrained and cannot
  demonstrate controller-ranking quality or traffic benefit.
