# V6: policy imagination credit (2026-10-15)

`trainer.imagination_grad_decay` is optional (`None` when omitted). An explicit value
in `(0, 1]` temporarily overrides each frozen WM's `grad_decay_per_step` during
imagination. The original value is restored in `finally`, including failed rollouts.
Training-time WM damping and checkpoint specs are not modified. Non-default overrides
participate in the trainer's model identity digest; the default digest is unchanged.
Trainer checkpoints also store both new config fields.

`configs/c4_policy_v6.toml` extends V5 with `imagination_grad_decay = 0.98`,
`credit_profile_every = 25`, and its own output directory. Across 60 physics steps,
the carry-path damping is approximately `0.98^60 = 0.30` instead of `0.9^60 = 0.0018`.
These factors describe the carried-state path, not the total actor gradient.

## Diagnostics

Every Nth update, `updates.jsonl` includes `credit_profile`. Zero (the default)
disables the diagnostic entirely. `norms[k]` is the L2 norm of the derivative of
the reward in the **kth one-based control window** with respect to the original
step-0 policy acceleration. Each window sums its two physics-step rewards; it is
not a cumulative prefix or a critic-bootstrapped return. The raw action precedes
safety and execution caps, so a saturated action can correctly have zero credit.

Windows are 1, 5, 10, 20, 30 where available, plus the terminal window (useful for
tiny runs). The diagnostic selects start row 0, member 0, future 0, and masks to
acting CAVs. Metadata records this scope and the number of active actions. It
reuses the training graph, retains it for the actor backward, and does not accumulate
parameter gradients. Only the selected reward row contributes to the norm, though
autograd still traverses batched operations. Disconnected actions have zero credit.

Existing `actor_grad_norm` is the **pre-clip** norm returned by `clip_grad_norm_`.
The existing `trainer.grad_clip` and nonfinite-gradient check remain in effect;
there is no additional clamp. `--smoke` caps CPU threads at two and, when profiling
is enabled, changes the interval to one so two-update runs exercise the diagnostic.
This also applies to resumed smoke runs.

## Verification

CPU-only local environment: `_out/v6-venv` (PyTorch 2.14.1+cpu, pytest 9.1.1).
Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`.

```text
_out/v6-venv/Scripts/python.exe -m pytest
  tests/unit/test_c4_v6imag.py tests/unit/test_c4_dreamer.py
  tests/unit/test_c4_policy2.py tests/unit/test_c4_policyv5.py
  tests/unit/test_c4_srcgrad.py tests/unit/test_c4_cfmem.py
  tests/unit/test_c4_futures.py tests/unit/test_c4_srcwait_imagination.py
  tests/unit/test_c4_accelcap.py tests/unit/test_c4_core.py
  -k "not test_start_windows_on_v5_smoke_conversions and not test_env_executes_lane_commands_in_sumo"
  -q --basetemp=_out/v6-verified-suites
151 passed, 2 deselected

_out/v6-venv/Scripts/ruff.exe check src/wms/control/dreamer.py
  src/wms/control/train_cli.py tests/unit/test_c4_v6imag.py --no-cache
All checks passed!

git diff --check
passed
```

The initial broad selection accidentally included the optional SUMO integration
test, which failed in setup on missing `onnxruntime`, before simulator launch.
It was explicitly excluded on the passing run; no SUMO execution was performed.
The other deselection requires data outside this worktree.

A separate execution comparison against pre-V6 commit `c7aca52` passed: two default
OFF updates were bit-identical for metrics (excluding elapsed time), all four
modules, both optimizer states, RNG states, and model digests. Focused tests also
cover stronger early-action credit at decay 1.0 versus 0.9, exception restoration,
profile values/cadence/JSONL, unchanged diagnostic-on updates, clipping, and resume.

Synthetic integration used the V6 trainer config with batch 2, horizon 4, entry
padding capped at 8, and profiling every update, on a small randomly initialized
real WM with real policy/bridge, entry queue, source-wait gradients, backlog critic,
and counterfactual rows. The model's original 0.9 decay was restored after each update:

```json
{"update":1,"returns":-4.10565710067749,"actor_grad_norm":0.1296059787273407,"credit_profile":{"norms":{"1":0.09314935654401779,"4":0.0048115868121385574},"active_actions":5}}
{"update":2,"returns":-3.463257312774658,"actor_grad_norm":0.11380411684513092,"credit_profile":{"norms":{"1":0.07854137569665909,"4":0.008392426185309887},"active_actions":5}}
```

**Outstanding acceptance check:** the requested trained-checkpoint/data smoke was
not run. Its checkpoint/physics and data/run-smoke inputs are outside the provided
workspace, while the execution policy explicitly forbids outside reads. The
synthetic integration above is not a replacement for that check. Stage those inputs
inside this worktree, point the existing data environment variables at the local
copies, then run `wms.control.train_cli --config configs/c4_policy_v6.toml --smoke`
with the local checkpoint and inspect both finite updates and their credit profiles.
