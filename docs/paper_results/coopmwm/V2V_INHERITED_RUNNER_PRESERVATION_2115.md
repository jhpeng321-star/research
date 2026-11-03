# Preserve inherited runner wiring before reviewed intent merge

The resumed worktree already contained the following uncommitted changes, which
are retained as a separate step rather than overwritten by the intent merge:

- `v2v_runner.admit`: separately authorised one-update real-SUMO smoke stage;
  all-arm smoke/rehearsal requirements continue to apply to normal training.
- `v2v_runner.run_arm`: preserve the original minimum pacing anchor when
  continuing an already-completed smoke update.
- `replay_bank._action_error`: attach causal V2V partner features for widened
  policies before measuring bank action error.
- Existing `test_v2v_live_cpu.py` is retained with these source changes.

Verification on 2026-11-03: `rl_driving` Python, pytest with bytecode/cache disabled,
`test_v2v_runner_cpu.py test_v2v_live_cpu.py test_replay_bank_v2v_cpu.py`:
**23 passed**, 50.63 s. These are CPU checks, not a V2V launch or physics gate.
No source was deployed and no simulation was dispatched by this step.

The final intent merge must preserve these semantics and requalify the merged
source on Linux. V2V remains held until the independent exact-commit audit and
the registered real-bank fix gate pass.
