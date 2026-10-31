# Collector startup repair — no scientific work affected

First launch8c24d646 failed because optional psutil is not installed in the
existing Miniconda environment. Parent created one local child before importing
that dependency; child38544 wrote WATCH_ONCE/STOP and exited before collection,
scoring or commits. Exact owned PID38544 absent in independent Get-Process audit.
Original ignored runs/rbint_noon/{DISPATCH_ONCE,WATCH_ONCE,STOP}.json retained.
No inference, simulator, training, result overwrite or remote worker action.

Replaced psutil with standard-library Win32 GetSystemTimes/GlobalMemoryStatusEx
and GetProcessTimes on the freshly created Popen handle (Linux /proc fallback).
No installation or new environment. New --state-root admits an explicit diagnosed
analysis-only recovery while preserving old exclusive claims and terminal STOP.
Fresh namespace runs/rbint_noon_v2 requires the original STOP receipt before start;
no automatic restart or scientific retry. Existing evaluator jobs remain unchanged.
