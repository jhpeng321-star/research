# G4 — sparse measured macro synergy credit

Implementation: `tools/coopmwm/sparse_synergy_credit.py` (contracts),
`g4_runtime.py` (production capture/coordinator/worker), with additive hooks in
`accel_plant.py`, `launch_training.py`, `launch_production.py` and `track2_backend.py`.
This is an explicit, default-off successor-phase recipe. Its configuration is
common to B and R. The existing u24 synergy criterion is not redefined.

## Fixed prospective registration

Use `registration()` rather than copying/tuning individual fields:

```python
from tools.coopmwm import sparse_synergy_credit as g4

cfg['g4_synergy_credit'] = dict(
    enabled=True, registration=g4.registration(),
    phase_id='prospectively-bound-successor-id', first_update=first_update)
cfg['g4_budget'] = dict(
    phase_id='prospectively-bound-successor-id', allocation_sim_s=230400,
    debits={}, provenance='owner-bound successor allocation from available slack')
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

The production translator validates the canonical parent quote first, then
derives the fixed G4 phase from available slack and any **explicitly approved**
`g4_budget.debits` (phase-to-sim-s mapping). It never raises the total or reduces
protected prior spend. An insufficient allocation, already-spent debit, or
remaining owner-budget shortfall refuses the entire paired transaction before
any branch. The full successor configuration, including these fields, is bound
by the existing `OWNER_APPROVAL.json` configuration hash. Reallocated workloads
must fit their remaining allocations; there is no automatic workload reduction.
Smoke excludes G4; updates before `first_update` retain the original learner path.

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

## Production orchestration and recovery

1. The existing ordinary-training `accel_plant.episode` installs `BoundaryCapture`
   only for enabled B/R jobs. Before sampling a decision, it captures the native
   prefix fingerprint/tape, generator state, AR memory, previous issued frame,
   causal intent boards, local report history and immutable policy weights.
   Only outcome-blind sampled, accepted joint boundaries publish immutable
   `jobs/<episode>/g4_captures/<opportunity>.pt` files. Completed ordinary episodes
   carry `g4_source` and the hash index. Practice is excluded.
2. The detached launcher waits for A's existing reference, then collects B. B
   publishes `g4/B_INPUT_u.json` after its safety/reward hooks and before training.
   R uses that receipt's **realized** practice count, collects, and publishes its
   own input. B waits for R's input rather than R waiting for B's optimizer.
   Thus both arm inputs are fixed before either optimizer runs.
3. `Coordinator(cfg).measure(u, episodes_by_arm)` acquires the same
   `ACCOUNTING.claim` lock as native training/diagnostic reservations. It derives
   cumulative G4 state from prior journaled plans, validates the complete paired
   quote against phase and remaining owner limits, then fsyncs and atomically
   replaces `CHARGES.jsonl` with **all branch charges and the full paired plan**.
   The plan and reservation state cannot be torn across independent files.
   Even zero-sample updates have a zero-cost journal/completion marker.
4. The production `run_branch(job, cfg)` adapter uses `accel_plant.episode`, the
   original V2V controller and its requests/safety/CommandOwnership writer.
   It replays all prefix commands and checks every fingerprint through t0.
   At t0 it restores causal boards/AR memory; seed 1 restores the captured
   generator exactly, seed 2 derives a fixed suffix seed from capture CRN hash
   and the registered seed ID. Every group/base/solo mask uses the same stream
   for that seed. Worlds and prefix are never reseeded.
5. The suffix runs only the bound frozen MAPPO base plus the selected **original
   full macro**, masked to the retained actors. There are no future triggers,
   guide proposals, learned residuals, vehicle removal or singleton-wall
   reconstruction. Each actor's wall release uses its own sensing/t-1 V2V report
   view via the production locality builder. Missing partner visibility keeps
   the original commitment until observable release or expiry. The native
   feasibility and single-writer safety chain remains authoritative.
6. All branches run through 1200, including the charged prefix. Graded receipts
   contain full-window source-inclusive delay and measured safety counters;
   `measured_envelope` validates the complete bundle. The same immutable
   `g4/ENVELOPE_u.json` is delivered to both `train_update(..., synergy=...)`
   calls after their existing safety hooks. Empty samples give an empty envelope,
   never a WM substitute. PPO uses the actual full-rollout temporal targets.
7. Restart reconstructs state from the existing ledger. A complete envelope can
   be adopted only with identical collected inputs and plan. A reserved attempt
   without its envelope **refuses redispatch**, including worker/startup failures
   or interruption between reservation and publication. Its charges and starts
   remain spent in later updates. Branch artifacts use the native
   `jobs/g4_<hash>/RESULT.json` layout, preserving ledger reconciliation.

These adapters are installed and default off. The active launch profile is
unchanged. This fix round executes only toy CPU physics; deployment still uses
the existing owner-approved production entry point.

## CPU checks

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest -q tools/coopmwm/test_g4_runtime_cpu.py tools/coopmwm/test_sparse_synergy_credit_cpu.py tools/coopmwm/test_track2_backend_cpu.py
```

Tests cover complete branch enumeration, full-prefix cap, signed seedwise
best-solo arithmetic, bounded normalization, prospective sampling, matched B/R
effort, stale/malformed/partial/WM rejection, reservation ordering/failure,
original-move solo masking/release, and the real trigger PPO edge with an
untouched frozen base and unchanged critic returns.
The orchestration suite enters through `launch_training.update` for both arms,
uses real collection/capture/worker adapters and real PPO/temporal targets, and
replaces only simulator physics plus unrelated admission/bank/safety fixtures.
It covers measured and no-sample paths, reservation/worker failures, restart and
next-update accounting, changed capture bytes, actor-local release, atomic write
interruption, remaining-owner budget and already-spent phase debits.
