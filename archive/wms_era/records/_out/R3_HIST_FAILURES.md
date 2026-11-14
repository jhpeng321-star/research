# R3-HIST verification attempts

Stable engineering task ID: **WM-S-R3-HIST-20261009**. Code and synthetic CPU
tests only; zero simulation, training runs, GPU use or scientific evaluation.

1. Default `python` lacked torch; `conda run -n rl_driving` reported an
   unregistered environment. Two workspace-local CPU dependency installations
   failed with `OSError: [Errno 28] No space left on device`. No dependency
   installation completed. The direct repository-documented `rl_driving`
   interpreter subsequently worked (`torch 2.12.1+cpu`). Only this task's pip
   download cache was purged, after enumeration found no links/junctions.
2. `_out/pytest_r3_hist_target01`, `_out/R3_HIST_TARGET01.xml`: **53 passed,
   1 failed**. The residual-mode GRU gradient was zero because the existing
   residual conditioning adapter is zero-initialised. The synthetic fixture now
   uses a tiny nonzero adapter to exercise upstream gradient flow without
   running an optimizer update. Default model initialization is preserved.
3. `_out/pytest_r3_hist_full01`, `_out/R3_HIST_FULL01.xml`: broad CPU verification
   reached the final tests but the E: volume filled. Pytest's final JUnit write
   failed with errno 28; the XML is incomplete, not a passing receipt. The
   attempt's generated synthetic fixture directory contained 632 files,
   155,148,781 logical bytes, and zero symlinks/junctions. Only that directory
   was removed after enumeration. The rerun uses a new basetemp and
   `-o tmp_path_retention_policy=failed` to clean successful fixtures immediately.

All attempts used workspace-local TMP/TEMP and outputs. No experiment was
allocated and no scientific conclusion changed. Final results are in R3_HIST.md.
