# S-AUDITFIX — mechanical training-audit repairs

Branch `c4/auditfix`, based on `c4/polaudit-t` 3357a14b. CPU-only engineering
qualification; no policy restart or WM adoption performed.

## Fix table

| Finding | Change | Flag / activation | CPU test |
| --- | --- | --- | --- |
| F1 full resume replaces TOML trainer/reward/lr | New-phase full resume keeps resolved TOML dataclasses and optimizer rates, preserves Adam moments/counters, logs every difference, rejects every unapproved difference before loading weights. | `--new-phase --resume C1.pt`, repeat `--resume-allow SECTION.KEY`; default OFF | `tests/unit/test_auditfix_resume.py` (2 pass) |
| F2 zero-weight individual rows | Equivalent six-row config preset: individual=coop=0, group=18.8, cover_group_members=false; registered all/team terms and weight meanings unchanged. | Select `configs/c4_policy14_c4prep_sixrow.toml`; opt-in | `tests/unit/test_auditfix_sixrow.py`: fixed-seed loss/full actor gradient parity, mean and q10/K8 |

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
