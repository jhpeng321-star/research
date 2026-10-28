# S-AUDITFIX — mechanical training-audit repairs

Branch `c4/auditfix`, based on `c4/polaudit-t` 3357a14b. CPU-only engineering
qualification; no policy restart or WM adoption performed.

## Fix table

| Finding | Change | Flag / activation | CPU test |
| --- | --- | --- | --- |
| F1 full resume replaces TOML trainer/reward/lr | New-phase full resume keeps resolved TOML dataclasses and optimizer rates, preserves Adam moments/counters, logs every difference, rejects every unapproved difference before loading weights. | `--new-phase --resume C1.pt`, repeat `--resume-allow SECTION.KEY`; default OFF | `tests/unit/test_auditfix_resume.py` (2 pass) |
| F2 zero-weight individual rows | Equivalent six-row config preset: individual=coop=0, group=18.8, cover_group_members=false; registered all/team terms and weight meanings unchanged. | Select `configs/c4_policy14_c4prep_sixrow.toml`; opt-in | `tests/unit/test_auditfix_sixrow.py`: fixed-seed loss/full actor gradient parity, mean and q10/K8 |
| F8 different WM cannot use inherited WM3 authority | New measured-authority schema binds exact WM/config, ACCEPT, registered threshold, measured gate and same-source pairs300 CCT-1; builder writes NEW authority/BOUNDARY files. Actual loaded config hash checked too. | Supply new `--horizon-gate` and `--looplong-boundary` files; opt-in | `tests/unit/test_auditfix_authority.py` (8 pass): accepts new WM, rejects stale/mutated bindings, failed short gate, REJECT, wrong panel/threshold; dense cap remains90s |

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
