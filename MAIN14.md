# MAIN14 — cooperation-ON loop, 2026-10-26

## Integration

Branch `c4/main14`, worktree `E:/agent-work/c4-main14`, starts at `c4/loop14c` **77d9e574**.

- Merged **ebb6e880** (`c4/coopw`): measured cooperation weights, two uncertainty futures,
  correlated group exploration, observable multi-CAV / side-by-side / share / demand start tilt,
  and WM-sha-bound horizon gates.
- Merged **cf456107** (`c4/sumocredit`): calibrated probability >= 0.8, K=8 learned-policy
  plan check, reference and sample-versus-mean tables, real-SUMO credit scaling.
- Merged **20942ccc** (`c4/gpubloop`), including its resource/probe support commits:
  frozen 16-start update-10 gate. Both CLI/orchestrator conflicts retain real-credit and
  gate-panel arguments.
- Cherry-picked **edc64d78** as **94a9eadf**: the executor's collect fixes became available
  early (01:48). Remote disk guards probe an existing ancestor; rsync creates destination
  parents for pushes and pulls. The extra gpuB operation scripts are not part of this arm's launch.

The interrupted `c4/coopw` work was inspected read-only: modified measurement profiling,
untracked `COOPW.md`, and `_scratch/`. The document still contains `{UNC}`, `{COST}`,
`{TESTS}` placeholders. These incomplete/unverified leftovers are excluded; that worktree
was not modified. This integration uses the committed measured weights and horizon receipt.

Integration additions: fixed TRAIN/VAL split seed across cycles (optimisation seeds still
advance), explicit current-WM horizon receipt override, bounded cache retirement, hashed
server-side base selection, and a detached fail-closed launcher.

## Registered arm

**C4-LOOP14-COOPON-20261026**, TRAIN; registration template
`configs/c4_main14_registration.json`; actual filled registration
`tools/main14/REGISTERED.json` and gpuR `main14_REGISTRATION.json`.

- Reserved seeds **968100–968227**, excluding **968100–968102**, consumed by the prior lane
  parity probe. Collection begins at 968103. No evaluation seeds.
- Four cycles, **150 policy updates/cycle**, 40 WM steps/cycle.
- **13 worlds/cycle**, two 300 s POLICY/B branch points, 10 s K=8 calibration, 30 s credit
  (group <=3). Charged upper bound **29,133 sim-s/cycle**, **116,532 total**; hard caps
  30,000 and 120,000. Reservations/charges precede SUMO, including any failed attempt.
- cpuX collection: **<=16 workers**, one numerical thread each. Ungated POLICY/B are retained;
  calibration and credit extras fit both >=0.8 deployment tables and credit multipliers.
  Long gated evaluation arms are excluded from this collection budget, as in SUMOCREDIT's
  13-world example; `policy_gate` supports the registered K=8 test-time check.
- gpuR 24 GB: batch 1, two futures, checkpointed imagination, CUDA memory fraction **0.88**,
  expandable segments, policy 2 threads, WM 4 threads, acceptance 4 CPU workers. Readiness FD
  probes use the same weights/tolerances on CPU.
- Horizon **10 -> 30 -> 60 s** at cooperation curriculum positions **0/150/300**, medium/high
  share only. Each accepted WM needs its own paired-difference receipt against the fixed measured
  13E-b2 threshold (0.8684210526, the measured 10 s bootstrap lower bound). Failed/missing
  evidence caps imagination at 10 s; no hand-set driver or behavioural threshold is added.

Config: `configs/c4_loop14_main14_gpur.json`; policy: `configs/c4_policy14_main14.toml`.
Actual native-return and low-penetration pack paths are corrected for gpuR's current layout.

## Unattended launch

`scripts/c4/main14_autolaunch.py` lives on gpuR and needs no laptop connection.
It is started with **setsid nohup**, stdin `/dev/null`, log
`/root/autodl-tmp/c4/v5/main14_autolaunch.log`; flock prevents duplicate waiters.
All PIDs are read by exact number/incarnation. No process is signalled.

1. Verify the committed source manifest, registered config and registration hashes.
2. Wait for the exact incarnation of **636520** to exit (zombie/reused PID counts as exited),
   and `runs/c4_wm_v10_round14_b/RUN_SUMMARY.json` to exist. Refuse while any GPU compute job
   remains. Protected training and relay processes are never touched.
3. Query cpuX's `~/actqual/runs/r14b/QUAL_r14b.json` (or explicit qualification JSON alternatives
   under `~/autopilot_r10/runs/r14b`). Fidelity alone is not a verdict. Cache/hash a complete
   snapshot. If no complete verdict was produced by **13:00 UTC+8**, expire without launch.
4. Choose 14b iff it introduces **no new FAIL** in aggregate/per-set sign, late-sign or
   magnitude items versus `QUAL_r13e.json`, and its **strict all-pair TTC ratio**
   (`fid.all_pairs.ratio`) is <= 13E's. Otherwise choose **13E b2 FINAL**,
   sha256 **952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8**.
   Verify scored checkpoint hashes match the actual gpuR finals. Missing measurements wait;
   they never silently authorise a model.
5. On gpuR, run the integrated unit suites and a **2-update CPU mechanics smoke using the
   selected real WM** (synthetic starts, zero new SUMO). Both must pass. Freeze/validate the
   16-start gate panel; recheck GPU/source integrity; initialise and sequence all loop phases.
6. On test/smoke/phase failure, log and write `main14_arm/WAIT_OWNER.json`; no automatic retry.
   Existing WM acceptance, incident, budget, consecutive-rejection and update-10 stop rules apply.

The disk floor is **1.5 GiB**. Only this arm's completed-cycle converted pair cache is retired,
after every file hashes identically to its retained cpuX original. Archive receipts preserve
all paths/hashes; current-policy return files, checkpoints, logs, and raw cpuX collection remain.
Code is an allowlisted, LF Git-object export; SUMO assignment XML is neither opened nor copied
by deployment. cpuX links the already-tested local fixture directory for simulator execution.

## Verification and operational receipt

The first integrated local run passed **333 tests** (pytest printed its complete success
summary; the harness timed out during process teardown). The focused additions and the
gpuR deployment checks are recorded in the final operational receipt below.
Ruff passed for the changed integration Python files. The launch repeats tests/smoke after
the real prerequisites are satisfied; pre-arm checks do not bypass that gate.

Deployment and arm evidence: `tools/main14/deployment_receipt.json`,
`tools/main14/REGISTERED.json`, gpuR `main14_arm/{ARMED,WAITER_INSTALLED}.json`.
The actual selected base is recorded later in `BASE_SELECTION.json` / `LAUNCHED.json`.

## Monitor / stop

On gpuR, under `/root/autodl-tmp/c4/v5`:

```bash
tail -f main14_autolaunch.log
cat main14_arm/waiter.pid main14_arm/WAITER_INSTALLED.json
cat main14_arm/WAIT_OWNER.json              # exists only on a failed launch/phase
/root/miniconda3/bin/python repo_main14/scripts/c4/loop14.py status --config main14_launch_config.json
touch main14_arm/STOP_ARM                   # disarm waiter, or stop driver between commands/phases
touch main14_loop/STOP_DRIVER               # graceful stop after current phase, once loop exists
touch main14_loop/policy/c00/STOP            # optional current-policy stop, at next update boundary
```

Use the actual current cycle's policy directory for the final command. No kill-by-name or
foreign PID signal is needed. Owner review is required before any retry/re-arm.
