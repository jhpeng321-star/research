# C4 Dreamer trainer

## API and timing

`wms.control.dreamer.DreamerTrainer(world_models, policy, group_encoder, critic,
start_states, *, observation_builder, reward_fn, safety, controller_factory=None,
config=None, reward_weights=None, device="cpu", output_dir=None)` takes its
control dependencies by injection. No parallel control implementation or simulator
is required to import or unit-test the trainer.

- `world_models` is a nonempty list of WM-S-compatible `nn.Module` members. They
  are put in evaluation mode and frozen with `requires_grad_(False)`. Their
  operations remain differentiable with respect to actions and imagined states.
- The policy returns reparameterised `(accel [B,K], log_prob [B,K])` from an
  observation and RSU signal; the group encoder accepts `(obs, state)`.
- `reward_fn(state, next_state, accel, RewardWeights)` returns the contract's
  reward/component tensors `[B]` per physics transition.
- An optional `controller_factory(policy, group_encoder, safety, record)` supplies
  the shared bridge callback. The trainer passes `record=None`, wraps the policy
  to enforce control cadence, and collects its own training statistics. Without
  the factory, an internal ballistic callback is used. Safety is evaluated at
  every physics step; policy decisions are held between control boundaries.
- `start_states.sample(batch_size, device=...)` returns `StartBatch(state,
  rollout_kwargs, metadata)`. State tensors are `[B,K]`. Driver and motion
  histories are the only observations of the recorded trajectory passed to WM-S;
  CAV/lane teachers, action timelines, and future recorded masks are refused.
- `imagine(batch, deterministic=False)` returns rewards/costs/returns/entropy
  `[E,H,B]`, actions `[E,H,B,K]`, disagreement `[H,B]`, component tensors, and
  the `H+1` control-boundary states for each member.
- `update()` samples starts, takes one actor and critic step, updates the target
  critic and dual, and returns a JSON-compatible metrics dictionary.
- `validate(sampler, batches=1)` evaluates deterministic imagined lambda-returns
  without updating weights, the dual, or the training RNG/sampler.
- `fit(max_updates, validation_states=..., validation_batches=...)` runs to a
  total update count; `save_checkpoint` / `load_checkpoint` support full resume.
  `load_checkpoint(..., policy_only=True)` loads just actor and group weights.

**Timing discrepancy in the brief:** the shared contract specifies a 1 s control
period and two 0.5 s physics steps per decision. Consequently the default `H=60`
means **60 seconds / 120 physics steps**, not 30 seconds. Set `horizon=30` for a
30-second rollout. The contract itself was not edited. A single whole-horizon WM
rollout preserves driver encoding, rolling histories, lane ages, and the WM's
existing `grad_decay_per_step`; there is no extra truncation of actor credit.

## Objectives and optimisation

Per-control-step rewards and TTC costs are the **sum** over their physics
transitions. Ensemble disagreement is the population standard deviation of the
unpenalised per-control-step team reward. It is zero for a singleton ensemble.
Matched policy-noise draws are replayed across members to avoid attributing
independent exploration noise to WM uncertainty.

Reward for TD(lambda) is `reward - uncertainty_weight * disagreement`.
`lambda_returns(rewards [H,B], values [H+1,B], gamma, lambda_)` includes the
terminal value bootstrap. The target critic's weights are frozen, but its state
derivatives remain active: analytic actor credit includes the terminal value and
propagates through the WM. Critic regression uses detached imagined state inputs
and detached lambda-return targets.

The actor objective is
`-mean(lambda_return)/return_scale - entropy_weight*mean(entropy)
+ lagrange_multiplier*mean(cost_ttc)`.
Entropy and action statistics include only present CAV slots. The multiplier uses
projected dual ascent toward the mean **control-step** TTC budget. Return scale
is `max(1, EMA(p95)-EMA(p05))`, a DreamerV3-style percentile range; critic MSE is
scaled by the same range. Actor/group and critic have separate Adam optimisers
and gradient clipping. Logged gradient norms are pre-clipping norms.

## Start windows and penetration

`StartStateSampler` filters TRAIN provenance and prohibited metadata before lazy
window loading. It samples uniformly among available penetration buckets and
then uniformly among windows within the selected bucket. World identifiers must
end in `p02`, `p05`, `p10`, `p20`, or `p40`, including suffixes without separators.
`split_start_states` reserves a deterministic stratified development VAL subset
from eligible TRAIN windows, disjoint by window index. Each retained penetration
needs at least two windows. Excluded penetration buckets enter neither subset.

Optional re-labelling randomly assigns present slots to CAV/HDV at load time to
reach the rounded target CAV count. It defaults off, supports only
`0.05/0.1/0.2/0.4`, never changes the stored window, and leaves physics/history
observations intact. Short motion history is also passed as driver history when
an explicit long driver history is absent.

The CLI uses the inherited R5b original data specification and
`TrajectoryDataset.from_real_packs` with an explicit TRAIN window list, verified
identities, zero start shifting, and only a minimal recorded horizon for t0
loading. Switching datasets and external evaluation sets are not assembled.
Checkpoint/config/data/output paths, including resolved symlinks, must stay in
this worktree. Restricted tokens are refused before access.

## Configuration and CLI

`configs/c4_policy.toml` documents all defaults:

- `run`: `output_dir`, `device`, `updates=3000`, `seed`, `threads`,
  `validation_batches`.
- `world_model.checkpoints`: list of frozen WM checkpoint paths; supply at least
  one. The command-line `--wm-ckpt path ...` overrides the list.
- `trainer`: `batch_size=16`, `horizon=60`, `gamma=.99`, `lambda_=.95`,
  `actor_lr=5e-5` (recommended `3e-5..1e-4`), `critic_lr=1e-4`,
  `entropy_weight=3e-4`, `uncertainty_weight=.1`, `ttc_budget=.01`, `dual_lr=.01`,
  `lagrange_init=0`, `grad_clip=100`, `target_tau=.02`, `return_ema=.99`,
  `checkpoint_every=100`, `validation_every=100`. Zero disables periodic
  checkpointing/validation respectively.
- `data`: `wm_config`, `val_fraction=.1`, `excluded_penetrations`,
  `relabel_cavs=false`, `target_penetration=.2`.
- `reward`: the fields of `contract.RewardWeights`: `delay`, `throughput`,
  `smooth`, `hard_brake`, `fairness`, `ttc_limit_s`.

```powershell
$env:TEMP='E:/agent-work/tmp'; $env:TMP='E:/agent-work/tmp'
$env:PYTHONPATH='src'; $env:PYTHONDONTWRITEBYTECODE='1'
python -m wms.control.train_cli --config configs/c4_policy.toml --wm-ckpt runs/active/local_r5b/wm_s_best.pt --smoke
```

The CLI lazily imports `obs`, `reward`, `policy`, `safety`, and `bridge`; it supports
`--device cuda`, `--max-updates N`, `--init-policy`, `--resume`, and
`--output-dir`. `--smoke` always forces CPU, batch 2, H 4, and two updates.
When resuming a smoke, two additional updates are run. Execution in this session
used CPU only. No GPU or SUMO was run.

Artifacts are `updates.jsonl`, `validation.jsonl`, `policy_NNNNNN.pt`,
`policy_latest.pt`, `policy_best.pt`, and `RUN_SUMMARY.json`. Per-update logs include
returns, all required reward components (contract `fairness_delay` is logged as
`fairness`), TTC cost, multiplier, entropy, CAV action statistics, gradient norms,
ensemble disagreement, return scale, and wall time. Best selection uses separate
VAL imagined returns. Resume saves both optimisers, policy/group/online/target
critic, dual, percentile estimates, update count, best score, Python/numpy/torch
RNGs, and training/validation sampler states. Checkpoints use plain types/tensors
and `weights_only=True` loading; the frozen ensemble fingerprint must match.

## Measured verification (CPU, 2026-10-12)

- Focused unit tests: **23 passed** in **38.81 s**. Tests cover hand-computed
  lambda-returns, dual-ascent sign/projection, physics reward summation, held
  actions, actor/group and HDV credit, differentiable terminal bootstrap,
  injected bridge cadence, disagreement penalties, detached critic fitting,
  target EMA, exact next-update resume, penetration mixing/re-labelling,
  source guards, VAL separation, and fake-backed CLI artifacts/resume.
- File-free fake smoke: two updates, batch 2, H 4, singleton WM, **7.572446 s**
  measured inside the smoke. Returns were **41.908443** and **41.645836**;
  actor gradient norms **0.408312 / 0.408869**, critic gradient norms
  **0.341415 / 0.338477**. Smoke reported `pass: true`.
- Ruff and `git diff --check`: passed. CLI `--help`: passed.
- Full guarded unit sweep at the earlier implementation checkpoint:
  **640 passed, 18 skipped, 1 failed**, **338.64 s**. Seventeen skips enforce the
  prohibited-path/workspace rule on existing synthetic fixtures; one is missing
  `tomllib` in this Python environment. The failure is the unchanged existing
  `tests/unit/data/test_drive_registry.py::test_committed_registry_matches_the_generator_output`
  (committed registry differs from regeneration). The later focused run includes
  two additional analytic-bootstrap/bridge tests and passes all 23 C4 tests.
  Native Git Bash was placed first on the test process PATH, and a Git ceiling
  prevented scratch directories from inheriting this worktree as their repo.

Reproduce the file-free fake smoke with:

```powershell
python tests/unit/test_c4_dreamer.py
```

## Integration limits and remaining evidence

The real R5b checkpoint and real TRAIN windows requested in the brief are outside
this workspace. The workspace-only read policy prevented opening them. Parallel
`obs/reward/policy/safety/bridge` files were also absent in this worktree during
verification. Accordingly all measured training/CLI smokes used injected local
fakes; **no real R5b TRAIN smoke is claimed**. The smallest integration action is
to make those modules and authorised checkpoint/TRAIN inputs available locally,
then run the CLI command above and resolve the pre-existing registry test failure
in its owning workstream.

This implementation has no observed-arrival entry process: it imagines from the
loaded slots and lets the WM handle exits. VAL separation is by windows, not a
claim of independent worlds or nonoverlapping time ranges. Re-labelling changes
controller ownership, not recorded driver-history evidence. Lane argmax and
saturated safety bounds remain nondifferentiable. Thirty-/sixty-second real-WM
memory, stability, throughput, and behavioural validity have not been measured;
these tiny CPU fakes establish execution and gradient contracts only.
