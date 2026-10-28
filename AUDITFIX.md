# S-AUDITFIX — mechanical training-audit repairs

Branch `c4/auditfix`, based on `c4/polaudit-t` 3357a14b. CPU-only engineering
qualification; no policy restart or WM adoption performed.

## Fix table

| Finding | Change | Flag / activation | CPU test |
| --- | --- | --- | --- |
| F1 full resume replaces TOML trainer/reward/lr | New-phase full resume keeps resolved TOML dataclasses and optimizer rates, preserves Adam moments/counters, logs every difference, rejects every unapproved difference before loading weights. | `--new-phase --resume C1.pt`, repeat `--resume-allow SECTION.KEY`; default OFF | `tests/unit/test_auditfix_resume.py` (2 pass) |
| F2 zero-weight individual rows | Equivalent six-row config preset: individual=coop=0, group=18.8, cover_group_members=false; registered all/team terms and weight meanings unchanged. | Select `configs/c4_policy14_c4prep_sixrow.toml`; opt-in | `tests/unit/test_auditfix_sixrow.py`: synthetic fixed-seed loss/actor+group gradient parity, mean and q10/K8 |
| F8 different WM cannot use inherited WM3 authority | New measured-authority schema binds exact WM/config, ACCEPT, registered threshold, measured gate and same-source pairs300 CCT-1; builder writes NEW authority/BOUNDARY files. Actual loaded config hash checked too. | Supply new `--horizon-gate` and `--looplong-boundary` files; opt-in | `tests/unit/test_auditfix_authority.py` (11 pass): accepts new WM, rejects stale/mutated bindings, failed short gate, REJECT, wrong panel/threshold/incomplete CCT; even H_C=300 leaves dense cap90s |
| F6 direct real-pair gradient unreported | Every update with direct credit logs `direct_real_credit_grad_norm`, the L2 gradient of the actual scaled direct term over trainable actor/group parameters, before clipping. | Diagnostic whenever direct credit is active; no objective/coefficient change | `tests/unit/test_auditfix_direct_grad.py`: analytic secant norm, untouched .grad/RNG, identical two-step Adam result |
| F10 `--real-credit` silently no-op | Fail startup if supplied multipliers leave every configured weight knot unchanged after the existing guards, or counterfactuals are disabled; report guarded knots for useful files. | Explicit `--real-credit`; omit it for the audited unchanged-credit run | `tests/unit/test_auditfix_real_credit.py`: candidate-A no-op refusal and useful multiplier semantics |
| F11 set-but-inactive config | Startup stderr warnings identify configured keys bypassed by the current CLI/native/long-credit/focus mode; summary stores `inactive_config`. | Always diagnostic; mode-aware | `tests/unit/test_auditfix_warnings.py`: all audited sections and active-mode negative check |
| F12 resumed C1 can win final selection silently | Final summary and native-gate receipt contain `phase_selection` with `resumed_baseline_selected`, `selected_new_phase_update`, selected update/source and phase-update count; baseline selection emits stderr warning. | Always diagnostic; registered selector unchanged | `tests/unit/test_auditfix_selection.py`: C1 wins explicitly, new-update case unflagged |
| F9 q10 docstring says per-start | Corrected to per-(suffix time,start); metrics are t0-only. No computation change. | Documentation only | Original `tests/audit/test_polaudit_t.py::test_q10_selects_the_bad_tail_and_only_those_futures_get_gradient` |
| F13 CAV report command/intent/age zero | Documented residual actor's explicit zero-command/intent and default zero-age convention; checked deployment `PolicyExecutor` feeds the same values even with nonzero command memory. | Documentation/qualification only | `tests/unit/test_auditfix_observations.py`: real bridge/executor report parity and ordinary builder explicit-value support |

## CAV reports: what deployment actually feeds

In native-residual **WM imagination**, `bridge.make_cav_controller` zeroes report
`cmd_accel_mps2` and `lane_intent` even if recorded/external command memory exists.
It calls `build_observations` without `report_age_s`, so reports have age0.
These are absent prospective-command/intent features, not claims that native
actions or packet latency are physically zero.

The current deployment **actor interface**, `policy.PolicyExecutor.__call__`,
also overwrites memory-augmented command/intent with zeros for residual policies
before calling the same builder without report ages. Thus all three actor
features match imagination. Its `dwell_age` parameter is **lane dwell age**, not
CAV report age. Ordinary absolute policies/MAPPO may use previous command/intent
memory; the builder supports explicit nonzero report ages, but the residual
training/deployment callers do not supply them. The CPU regression uses the
actual bridge and executor and nonzero memory to verify this convention.

`deploy.py` may augment observer state with memory for other controllers; the
residual executor still suppresses those two actor features. This documents the
actor-input contract only. It does not qualify the legacy deploy/collector's
residual execution path; the independent deployment audit and S-EXECPARITY own
those execution/observer repairs.

## q10: registered intent and actual reduction

**Documentation fix, no code/behavior change.** `tail_credit` ranks K8 joint,
signed, CRN-paired row credits independently at every `(suffix time, start)`;
the actor loss averages those quantiles. It does not select one bad future pair
at t0 and freeze it for all subsequent suffix returns. At K8/q0.10, the two
lowest futures receive0.3/0.7 interpolation weight. Reported mean/pessimistic
credit metrics use t0, not the temporal average.

REGISTRATION.md S-ANTIEXPLOIT-1 (12:00, lines3512–3515) registers lower paired
reward-advantage q0.10/K8, linear interpolation, and no outcome-selected q;
it does **not** prescribe a single t0 ranking reused across suffix times.
The existing per-suffix reduction applies that registered pessimistic paired
advantage intent to the existing suffix-return loss. Calling it simply
"per start" was incomplete. A t0-only/reused-ranking objective would be a
scientific behavior change requiring a separate decision; none is made here.

## Final-selection baseline flag

The registered earliest-minimum selector is preserved. If C1/update180 wins,
`RUN_SUMMARY.json` and `native_gate_result.json` now explicitly say
`phase_selection.resumed_baseline_selected=true`,
`selected_new_phase_update=false`, `source=resumed_phase_start`, and zero
`selected_phase_updates`. A stderr warning says that no newly trained phase
checkpoint was selected. A completed optimizer phase is not thereby evidence of
a newly selected candidate. Operational recovery uses the original phase start.

## Startup inactive-setting warnings

Warnings name keys, not just whole sections: boundary auto-horizon replaces the
TOML stage/pad/gate schedule (last-knot spacing is still read); disabled uncertainty
fields are distinct from trainer K; native-init overrides lane/actor initialization,
lane credit and entropy knobs; long credit bypasses critic warmup/value/real-return
mix and paircrit's value updates; focus replaces start_tilt; validation_uniform
and run validation_batches have no validate call; CLI seed/updates win. Paircrit
artifacts can still be loaded during setup, which the warning explicitly notes.
This is reporting only, without altering mode decisions or sampling.

## `--real-credit`

This flag is the measured multiplier file, distinct from direct paired secant
credit. The audited individual0.401/group1/coop1/team1 file is wholly erased by
the registered floor and now raises a clear error after resume, before warmup or
updates. **Remove `--real-credit`** to request the unchanged registered objective.
Useful files still follow the existing floor/coop cap and report `guarded_knots`
in the summary. No floor, coefficient or weight meaning was changed.

## Direct real-pair gradient telemetry

`direct_real_credit_grad_norm` is separate from the imagination-only diagnostic
gradient shares and is emitted on **every** active direct-credit update, independent
of `grad_audit_every`. Its units include the current `return_scale` and any
coefficient already applied by the direct-loss provider. Compare against the
existing pre-clip `actor_grad_norm` with care: component vectors can cancel, so
norm ratios are not additive gradient shares. `autograd.grad(retain_graph=True)`
does not accumulate `.grad` or make another optimizer step. Direct credit's
ON/OFF/coefficient decision remains with S-EXPLOIT/S-LOOPPATCH.

## New-WM measured authority: exact commands

Keep the original WM3 constant and inherited-authority artifacts intact. The new
schema is `c4launch_measured_horizon/1`; both `HorizonConfig.bind` and
`looplong.read_boundary` verify it. The ordinary `coopw_horizon_gate/1` path is
also retained. Use only a completed, canonical ACCEPT for the exact new WM/config.

From the integrated runtime root (with `PYTHONPATH=src:.`), set `WM`, `WM_CONFIG`,
`PAIRS`, `ACCEPT`, `OLD_BOUNDARY`, `OUT` to absolute paths for the selected WM,
its accepted config, the unchanged registered pairs300 panel, its ACCEPT.json,
the previous90s BOUNDARY.json and a new existing task output directory. These
commands are preparation only and were not run on production inputs here:

```sh
python tools/coopw/horizon_gate.py --wm-ckpt "$WM" --wm-config "$WM_CONFIG" --pairs "$PAIRS" --threshold configs/coopw_horizon_gate_13E_b2.json --groups 96 --panel-seed 20261026 --horizons 10 30 60 90 120 180 300 --samples 1 --entrant-control --deterministic-crn --out "$OUT/HORIZON_GATE.json"
python -m tools.gated300.cct1 "$OUT/HORIZON_GATE.json" --panel pairs300 --out "$OUT/CCT1.json"
python -m wms.control.auditfix_authority --boundary-template "$OLD_BOUNDARY" --wm-ckpt "$WM" --wm-config "$WM_CONFIG" --gate "$OUT/HORIZON_GATE.json" --cct1 "$OUT/CCT1.json" --acceptance "$ACCEPT" --threshold configs/coopw_horizon_gate_13E_b2.json --authority-out "$OUT/HORIZON_AUTHORITY.json" --out "$OUT/BOUNDARY.json"
```

The first command retains the canonical gate's K1 measurement recipe, registered
96-group panel/seed/0.868421052631579 threshold, and explicitly measures the
restart's entrant-ON/CRN path; it does not relabel an old entrant-OFF result.
CCT-1 consumes those exact same rows without a new rollout. Its calculator is
owned by `c4/gated300` and must be merged for the second command. This branch
does not duplicate or modify that computation. Report calibration/validation
CCT-1 separately with their true panel labels; they cannot authorize a policy.

Append `--horizon-gate "$OUT/HORIZON_AUTHORITY.json" --looplong-boundary
"$OUT/BOUNDARY.json"` to the prepared policy command, with `--wm-ckpt "$WM"
--wm-config "$WM_CONFIG"`, `--resume C1.pt --new-phase` and the reviewed exact
resume allowances. New dense authority requires measured short10/30/60 and90
PASS under the unchanged legacy gate. It records a WM/config/source-hash-bound
CCT-1 even when H_C=0; CCT-1 failure does not revoke separately valid legacy90s.
It never authorizes raw credit above90s. Only S-GATED300 may interpret H_C and
enable its separately gated credit. Template pair/return paths, flags and
curriculum inputs are reused; this builder changes no horizon or sampler.

All nested artifact hashes are verified on every authority read. Acceptance and
WM-config binding are checked on initial launch as well as swaps. The CLI supplies
actual loaded config hashes to `HorizonConfig.bind`, including when no boundary
argument was supplied, so a valid authority for a different inference config
cannot grant the loaded trainer a horizon. Outputs use
exclusive creation, so a previous boundary/authority is never overwritten.
CCT-1 reported sigma/threshold/sign/coverage/missed-harm metrics, per-horizon
verdicts, contiguous H_C and world-bootstrap metadata are checked against the
registered definition; incomplete headers alone cannot serve as authority.

## Six-row preset

Resolve `configs/c4_policy14_c4prep_sixrow.toml` with the same input/anchor
preparation as its inherited candidate-A base. It pins the live registered
candidate-A weights at both knots `[0,1]`, as in the audited resolved TOML.
Alternatively copy its `[counterfactual]` overrides into that resolved restart
TOML. This removes exactly
the three dead individual rows for the registered three-member group (9 to 6),
without changing the objective. The formula remains
`D_all + 49.7*D_G - 10.3*sum_i D_(G\i)`. Other group sizes retain the same algebra;
six rows is the three-member case, not a forced group size or sampling change.
Do not apply nontrivial real-credit multipliers to this rewritten parameterization
and claim equivalence to multipliers applied to the original weights.

## New-phase resume

Use a new output directory. Set `[trainer] checkpoint_every = 1` (or 2–5) in the
restart TOML. `--new-phase` makes that value effective; the CPU test executes an
update and verifies the numbered checkpoint. Historical resume remains available,
but now prints its configuration differences rather than hiding them.

First run the intended policy command with `--resume C1.pt --new-phase` and no
allowances. It refuses before actor loading if anything differs, reporting exact
keys and checkpoint/requested values. Review those differences, then append one
`--resume-allow KEY` per approved change. Examples: `trainer.checkpoint_every`,
`trainer.actor_lr`, `reward.delay`, `optimizer.actor_optimizer.lr`. Optimizer lr
approval is separate because the saved optimizer can disagree even with its own
saved trainer config. Wildcards and unused/typo allowances are rejected.

The run summary records `resume_config`, `effective_config_before_updates` and
`effective_reward_before_updates`. New-phase asserts effective trainer/reward
equality with the resolved TOML immediately before updates; incompatible runtime
calibration/recovery must be resolved explicitly. It is not operational recovery.

## Verification

Local Python 3.13 / torch 2.15.0+cpu, isolated environment
`E:/Temp/opencode/auditfix-venv`; `PYTHONUTF8=1`, `PYTHONPATH=src;.`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`C4_ALLOWED_ROOTS=E:/agent-work/c4-auditfix;E:/Temp/opencode`,
`TEMP=TMP=E:/Temp/opencode`. Use the equivalent colon-separated path lists on
Linux. All verification is CPU-only, synthetic/registered config inputs; zero
SUMO, GPU jobs, remote operations or new scientific evidence.

- **24 new CPU tests pass** across the eight `test_auditfix_*.py` files.
- New tests + original training audit + anti-exploit regression: **51 passed**
  (17.30s). The original audit remains an unchanged historical-behavior record;
  in particular its legacy-resume reproduction still passes with new-phase OFF.
  Its existing tensor-to-float diagnostic emits one harmless PyTorch warning.
- Broader regression command below: **161 passed, 4 skipped** (160.93s).
  Two skips require CUDA; two require absent local `tools/loop14_checks/accept.py`
  simulation evidence. No skipped check is represented as qualified.
- After the final loaded-config binding change: authority + COOPW + C4PREP
  suites **50 passed** (8.96s), then the 51-test audit/new-fix suite passed again.
- Fatal-error Ruff (`E9,F63,F7,F82`) passes on every changed Python file;
  staged `git diff --check` passes.

```sh
python -m pytest tests/unit/test_longmem.py tests/unit/test_c4launch.py tests/unit/test_c4prep.py tests/unit/test_looplong.py tests/unit/test_policy14_recovery.py tests/unit/test_c4prep2_credit.py tests/unit/test_coopw.py tests/unit/test_c4_dreamer.py -q
python -m pytest tests/unit/test_auditfix_resume.py tests/unit/test_auditfix_sixrow.py tests/unit/test_auditfix_authority.py tests/unit/test_auditfix_direct_grad.py tests/unit/test_auditfix_real_credit.py tests/unit/test_auditfix_warnings.py tests/unit/test_auditfix_selection.py tests/unit/test_auditfix_observations.py tests/audit/test_polaudit_t.py tests/unit/test_antiexploit.py -q
```

The two existing C4PREP CLI test doubles now retain the supplied reward dataclass
so they model the trainer interface used by the new effective-reward telemetry;
their scratch/normal-path assertions remain intact.

## Integration surface

Behavior-changing activation is explicit: `--new-phase`, selecting the six-row
preset, or supplying the new measured authority/boundary. Remaining changes are
diagnostics/documentation and refusal of an explicitly requested no-op file.
Core commits: resume `25ad8053`, preset `24928e9d`, measured authority `72db105e`
(hardened `65322776`, loaded-config check `93c8b120`), direct-gradient telemetry
`a1c2663c`, real-credit refusal `1adb244d`, startup warnings `5ccc73e9`, baseline
selection flag `42ad61b7`, q10 docs `44a5988e`, report-input parity `e64fdcef`.

Additive helpers live in `phase_resume.py`, `auditfix_authority.py` and
`auditfix_diagnostics.py`. Shared-file merge points are CLI argument/setup/summary
hooks, Dreamer checkpoint restore and the direct-loss telemetry hook,
HorizonConfig binding, and boundary verification. No S-GATED300/S-EXPLOIT/
S-LOOPPATCH/S-POLSPEED implementation is imported or modified by this branch;
their selection horizon, curriculum, credit coefficient/ON-OFF, execution and
performance decisions remain integration-owned.
