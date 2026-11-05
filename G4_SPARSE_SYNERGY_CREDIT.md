# G4 — sparse measured macro synergy credit

Implementation: `tools/coopmwm/sparse_synergy_credit.py`, with a narrow
`ArmRuntime.train(..., synergy=envelope)` integration in `track2_backend.py`.
This is an explicit, default-off successor-phase recipe. Its configuration is
common to B and R. The existing u24 synergy criterion is not redefined.

## Fixed prospective registration

Use `registration()` rather than copying/tuning individual fields:

```python
from tools.coopmwm import sparse_synergy_credit as g4

cfg['g4_synergy_credit'] = dict(
    enabled=True, registration=g4.registration(),
    phase_id='prospectively-bound-successor-id', first_update=first_update)
```

The registration samples accepted stochastic joint macro starts by an
outcome-blind 1-in-16 hash of training seed, world seed, start time and proposer.
It excludes practice episodes, replayed/deterministic decisions, rejected
proposals, singleton moves and `gap_open` (one actuator despite two context
members). It uses the original recorded conditional slate, mask and action.
No reward, safety outcome, gain or diagnostic result controls selection.

At each update, B/R sample counts are matched **by actor arity**, with at most
8 starts per arm over the entire successor phase. Candidate scarcity gives
fewer paired samples; there is no arm-specific top-up. Two prospectively fixed
verification seeds are used. Pair-first arity ordering is 2, then 3, then 4.

Every start measures group, common base, and **every** member solo for each seed.
All branches use the same pre-action state, replayed prefix, CRN, policy bindings
and window `[macro_start, 1200]`. Charging includes the complete prefix: **1200
sim-s per branch**, even for late starts. No horizon is cropped to the macro hold.
For two actors the full quote is `8 * 2 arms * 2 seeds * 4 branches * 1200 =
153600 sim-s`; the four-actor worst case is **230400 sim-s**, the hard cumulative
G4 cap. This is a reservation within the broader run budget, not extra allowance.
Failed/unused reservations are not recycled or retried.

For each verification seed, with delay costs lower-is-better:

```text
total benefit = J_base - J_group
best solo benefit = J_base - min_i J_solo_i
S = min_i J_solo_i - J_group
macro advantage addition = clip(mean_seed(S) / 10000, -1, +1) * 1.0
```

The normalization reuses `accel_learner.UNIT`; the fixed weight, symmetric bound,
sampling rule and provenance are in the registration. The best solo is taken
**inside each seed**, before averaging. Solos have no zero floor. Negative and
zero synergy remain valid labels. The unmodified local-return trigger advantage
is retained, with this term added only to the initiating proposer's action.
Critic returns, continuous residual targets and F4 remain on their existing path.
Unsampled starts get exactly zero additional credit. No WM substitute exists.

## Coordinator / authorized worker API

1. Collect complete ordinary TRAIN episodes for both arms. Collection provenance
   must mark these tapes `g4_source='ordinary_training'`; their world seeds must
   be in the same update's registered `worlds_by_update`. Practice rows are
   excluded. Diagnostic tapes are not valid credit sources.
2. Use `opportunities(episodes)` to identify accepted joint boundaries. Capture
   the state **before that decision's physical command**, not the cached
   post-action `t0` used by historical whole-policy diagnostics. Supply
   `captures[arm][opportunity_id]` with `stage='pre_action'`, `time_s`, and the
   immutable SHA256 fields in `CAPTURE_KEYS`: snapshot, prefix, CRN, current
   policy and common base policy. Captures are worker-produced metadata; this
   module never opens snapshot payloads or hidden vehicle parameters.
3. Initialize the phase once with `state = initial_state(cfg)`; call
   `plan, next_state = plan_update(cfg, u, {'B': B, 'R': R}, captures, state)`.
   Persist the returned state rather than reinitializing it each update. Plans
   bind the phase, update, exact trigger record, captures, jobs and quote by hash.
4. `execute_plan(plan, reserve=..., run_branch=...)` is the explicit executor seam.
   `reserve(plan)` must atomically compare/persist the prospective state, reject
   duplicate plan IDs and precharge **all** paired jobs in the existing run
   ledger before the first dispatch. The broader budget's remaining allocation
   must also allow the quote. An exception aborts without a partial label/retry.
   The G4 module has no simulator import or implicit dispatch.
5. The later authorized SUMO adapter implements each fixed-macro job on the
   existing replay/safety/command-ownership path. Group retains all original
   actors; solo retains exactly its named actor; base retains none. Use
   `command_offsets(job, elapsed_s, observable_state=...)` after evaluating the
   **original full move** for release/role semantics, then mask offsets. Do not
   rebuild a singleton wall, remove vehicles, or run a whole-policy ablation.
   All other commands and subsequent decisions follow the same bound common
   base. Prefix/CRN replay, physical feasibility and safety checks belong to the
   existing worker, and must complete before it issues a graded receipt.
6. `measured_envelope(plan, results)` can also consume independently executed
   receipts. It requires the exact group/base/all-solo/two-seed job set, measured
   SUMO source, matched window/captures/CRN/masks, finite delay, full replay cost,
   and safety counters for every branch. It rejects partial, duplicate, stale,
   ungraded and WM receipts. Exact G1 reuse is possible only if **all** these
   immutable job bindings match; native-ancestor verification is not sufficient.
7. After the existing safety hook, call each runtime's
   `train(episodes, rewards=existing_reward_envelope, synergy=envelope)`. Both arms
   consume the same paired envelope, each applying only its own proposer rows.
   Enabled updates with zero sampled starts still pass an empty measured plan.
   Validation precedes any optimizer update. Return field `g4_synergy_credit`
   contains registration, plan SHA, measured/unsampled counts, per-seed total,
   best-solo and synergy values, normalized advantages and cumulative reservation.

The launcher/collector must bind the successor phase, worker capture adapter,
ledger reservation and executor explicitly. This implementation does not modify
the active launch profile or run counterfactual physics. These external adapters
are injected production interfaces; the CPU tests use clearly marked fakes.

## CPU checks

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest -q tools/coopmwm/test_sparse_synergy_credit_cpu.py tools/coopmwm/test_track2_backend_cpu.py
```

Tests cover complete branch enumeration, full-prefix cap, signed seedwise
best-solo arithmetic, bounded normalization, prospective sampling, matched B/R
effort, stale/malformed/partial/WM rejection, reservation ordering/failure,
original-move solo masking/release, and the real trigger PPO edge with an
untouched frozen base and unchanged critic returns.
