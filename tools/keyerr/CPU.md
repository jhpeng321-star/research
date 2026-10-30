# S-KEYERR CPU verification — 2026-10-30

Base: `95850e8d`; reproduction: `f400e4d2`; cap fix: `c17e1008`;
same-phase continuation + launcher: `e7efcb7c`. Windows, Python 3.13,
PyTorch `2.14.1+cpu`; local CPU only. No remote or simulator execution.

## Reproduction and root condition

Before the fix, `test_keyerr.py` had two failures at exactly
`continuous.project_native: state[EXEC_CAP_KEY]`: ordinary imagine and
`memory_imagine -> Replay.forward -> imagine -> model.rollout -> step`.
The synthetic first-passage hybrid WM lets both CAVs exit while HDVs remain.
`cav_exec_cap='idm'` stays enabled; `present & is_cav` becomes batch-wide false.
`step` suppressed `_exec_cap`, although its controller lifecycle contract still
invoked the callback. `_exec_cap` itself never returns None in enabled modes.
An entrant CAV counts in `cav`; absence of a leader alone does not suppress it.
The supplied traceback is the initial full no-grad forward, not a shortened VJP.
The evidence contains no failing physics-step/state dump; no exact tick is claimed.

## Final results: 482 passed, 5 CUDA-only skipped

Run from the worktree with `PYTHONPATH=src;.` on Windows (`src:.` on Linux),
`CUDA_VISIBLE_DEVICES=''`, `PYTHONUTF8=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `C4_ALLOWED_ROOTS=E:/Temp/opencode`.
Use `C:/Users/jpeng/Miniconda3/python.exe -m pytest -q -p no:cacheprovider`
and a new `--basetemp E:/Temp/opencode/keyerr-<unique>` for each command.
All filenames below are under `tests/unit/`:

* 146 passed, 2 CUDA-only skipped: `test_longgrad.py test_longskip.py
  test_gated300.py test_longmem300_budget.py test_longmem.py
  test_auditfix_resume.py test_keyerr.py test_keyerr_continuation.py`.
* 141 passed: `test_canda_integration.py test_zerostart.py
  test_policy14_integration.py test_policy14_recovery.py test_execparity.py
  test_execparity2.py test_execparity3.py test_execparity_continuous.py
  test_execparity_alwayscmd.py test_execparity_alwayscmd_replay.py
  test_execcap.py test_c4_cav_exit_callback.py`.
* 120 passed, 3 CUDA-only skipped: `test_nanfix.py test_gated300_cct1.py
  test_gated300_cct2.py test_exploit.py test_antiexploit.py
  test_polspeed2_report.py test_polspeed2_parallel.py`.
* 75 passed: `test_auditfix_warnings.py test_auditfix_sixrow.py
  test_auditfix_selection.py test_auditfix_real_credit.py
  test_auditfix_observations.py test_auditfix_direct_grad.py
  test_auditfix_authority.py test_looplong.py`.

No separately named DEADZONE2 suite exists in this base; execution, dead-zone,
shared-law/leader, zero-start and NANFIX regressions above cover that path.
One first long-suite command timed out; the completed rerun took 486.58 s.
Two old resume tests needed the allowed temp root; they then passed (3.88 s).
One old LOOPLONG text test needed UTF-8 mode instead of Windows GBK; it then passed.
No implementation changes were made for those environment failures.

## Exactness and continuation

The eight cap regressions cover no-grad forward, segment-2 backward, exits,
empty starts, entrant-only CAVs, re-entry and disabled-cap behavior. Caps equal
the real inferred cap, not an invented infinity. Nonempty forward tensors and
actor gradients match bitwise; all finite-gradient assertions pass.
Additionally loaded the actual pre-fix `WmSModel.step` AST from
`git show f400e4d2:src/wms/model/wms.py`, bound it to an identical real-WM trainer,
and compared full memory forward tensors, actor gradients and RNG with the
fixed implementation on a previously working nonempty-CAV trajectory: bitwise equal.
Twelve continuation tests cover origin 182 -> gate 192, remaining budget 143,
original panel baseline, input mismatches, saved curriculum/RNG, CLI wiring,
and bitwise trained-head/Adam preservation with the reset-on-new-phase flag true.
The actual supplied summary matches the resolved combined config and reward
exactly; it records completed updates 183..189 and phase origin 182.

Ruff passed on changed Python files and LONGGRAD/LONGSKIP execution modules/tests;
`bash -n tools/candA/stepK.sh` and `git diff --check` passed.
