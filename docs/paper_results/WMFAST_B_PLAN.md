# S-WMFAST-B implementation checkpoint

Base: `43942f75`, branch `c4/wmfastB`; owner request 2026-10-31.
Keep the registered long horizon and estimator. Stack variant rows, never starts:
all stochastic plans retain the original start geometry and replay recorded draws.
Use side-specific linear residual heads with original per-side GEMM geometry.
Unstack full outputs before gated300 prefix/reductions and the unchanged ES gates.
Execution-only environment switch: `WMS_SPEED_ES_STACK_SIDES=0` (default), 2 or 4.
Qualify CPU complete-output/gradient/RNG parity, worker transport, and bounded
real-WM probes; measure component timing and memory separately from projections.
Local GPU check: nvidia-smi reports driver not loaded; remote access still to check.
No optimizer/training run, physics/core edits, pilot changes, or push authorized.
