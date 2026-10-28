# S-AUDITFIX — mechanical training-audit repairs

Branch `c4/auditfix`, based on `c4/polaudit-t` 3357a14b. CPU-only engineering
qualification; no policy restart or WM adoption performed.

## Fix table

| Finding | Change | Flag / activation | CPU test |
| --- | --- | --- | --- |
| F1 full resume replaces TOML trainer/reward/lr | New-phase full resume keeps resolved TOML dataclasses and optimizer rates, preserves Adam moments/counters, logs every difference, rejects every unapproved difference before loading weights. | `--new-phase --resume C1.pt`, repeat `--resume-allow SECTION.KEY`; default OFF | `tests/unit/test_auditfix_resume.py` (2 pass) |

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
