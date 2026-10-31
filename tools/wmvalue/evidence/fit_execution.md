# Frozen CPU fit receipt (2026-10-31 resume)

- Code/reconstruction commit6f334469 precedes real fitting; no hyperparameter search.
- Final command-filtered v3 exports: WMDATA160/3360,29.10s;
  WMDATA2300/6300,80.10s; all460 FIT episodes and9660 states.
- Source command masks are explicit nullable issued values, filtered by known
  present CAV identities at each decision. V3 also binds objective file hashes.
- V3 array hashes equal preserved v2 local copies exactly; fit_causal verifies
  these hashes against v3 receipts and the final exporter source hashes.
- 2 new prepared-FIT guard tests PASS (reserved seed, missing cells, terminal
  constraint); previous21 tests PASS; ruff PASS. No reserved outcomes used.
- Frozen fit executed once on existing local rl_driving Python/sklearn1.7.2,
  threadpool limit1:9.89s for V and D, then FIT-only E arm/time means.
- Both regressors use HGB150/15/.05/L2=10,seed20261031. Equal world mass, then
  equal arms and21 time states.111 observable features; D adds6 issued commands.
- Model preserved at _out/wmvalue_frozen_value6f334469.joblib; SHA256
  ef3e7bced5737c921a65596c566c865152990874be1d9901b8bbb6fb53735d8b.
- Full receipt: frozen_fit.json. Clock max discrepancy7.869271939853206e-06veh-s;
  source discrepancy0. No training-set accuracy used to select anything.
- Remote final exports remain under each corpus root:
  wmvalue_causal_fit6f334469_v3.{npz,json}; older packages/results are preserved.
- No A-E held-back test, WM inference, SUMO or policy training occurred. There is
  no outstanding WMVALUE process; all foreground export commands exited0.
