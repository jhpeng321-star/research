# S-AUDITFIX — mechanical training-audit repairs

Branch `c4/auditfix`, based on `c4/polaudit-t` 3357a14b. CPU-only engineering
qualification; no policy restart or WM adoption performed.

## Fix table

| Finding | Change | Flag / activation | CPU test |
| --- | --- | --- | --- |
| F1 full resume replaces TOML trainer/reward/lr | New-phase full resume keeps resolved TOML dataclasses and optimizer rates, preserves Adam moments/counters, logs every difference, rejects every unapproved difference before loading weights. | `--new-phase --resume C1.pt`, repeat `--resume-allow SECTION.KEY`; default OFF | `tests/unit/test_auditfix_resume.py` (2 pass) |
| F2 zero-weight individual rows | Equivalent six-row config preset: individual=coop=0, group=18.8, cover_group_members=false; registered all/team terms and weight meanings unchanged. | Select `configs/c4_policy14_c4prep_sixrow.toml`; opt-in | `tests/unit/test_auditfix_sixrow.py`: fixed-seed loss/full actor gradient parity, mean and q10/K8 |
| F8 different WM cannot use inherited WM3 authority | New measured-authority schema binds exact WM/config, ACCEPT, registered threshold, measured gate and same-source pairs300 CCT-1; builder writes NEW authority/BOUNDARY files. Actual loaded config hash checked too. | Supply new `--horizon-gate` and `--looplong-boundary` files; opt-in | `tests/unit/test_auditfix_authority.py` (8 pass): accepts new WM, rejects stale/mutated bindings, failed short gate, REJECT, wrong panel/threshold; dense cap remains90s |
| F6 direct real-pair gradient unreported | Every update with direct credit logs `direct_real_credit_grad_norm`, the L2 gradient of the actual scaled direct term over trainable actor/group parameters, before clipping. | Diagnostic whenever direct credit is active; no objective/coefficient change | `tests/unit/test_auditfix_direct_grad.py`: analytic secant norm, untouched .grad/RNG, identical two-step Adam result |
| F10 `--real-credit` silently no-op | Fail startup if supplied multipliers leave every configured weight knot unchanged after the existing guards, or counterfactuals are disabled; report guarded knots for useful files. | Explicit `--real-credit`; omit it for the audited unchanged-credit run | `tests/unit/test_auditfix_real_credit.py`: candidate-A no-op refusal and useful multiplier semantics |
| F11 set-but-inactive config | Startup stderr warnings identify configured keys bypassed by the current CLI/native/long-credit/focus mode; summary stores `inactive_config`. | Always diagnostic; mode-aware | `tests/unit/test_auditfix_warnings.py`: all audited sections and active-mode negative check |
| F12 resumed C1 can win final selection silently | Final summary and native-gate receipt contain `phase_selection` with `resumed_baseline_selected`, `selected_new_phase_update`, selected update/source and phase-update count; baseline selection emits stderr warning. | Always diagnostic; registered selector unchanged | `tests/unit/test_auditfix_selection.py`: C1 wins explicitly, new-update case unflagged |

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
WM-config binding are checked on initial launch as well as swaps. Outputs use
exclusive creation, so a previous boundary/authority is never overwritten.

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
`E:/Temp/opencode/auditfix-venv`; `PYTHONUTF8=1`, `PYTHONPATH=src;.`.
`python -m pytest tests/unit/test_auditfix_resume.py -q`: **2 passed**.
