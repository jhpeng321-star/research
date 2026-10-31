# rbh-setdata — FIT-world per-driver histories + physics-likelihood scaffold

Deliverable of helper **rbh-setdata** for the WM-PHYS rebuild (rb_common.md).
Everything here is offline numpy/CPU; no SUMO runs, no GPU, no training, no
sealed data touched (native-arm FIT tapes only, sha256 manifest included).

## What is here

| file | what |
|---|---|
| `fit969400-431_hdv_hist_60s.npz` | per-HDV observable 60 s histories for FIT worlds 969400–969431 (native arm, window [240, 300] s) — schema `setdata_hdv_histories/1`, see SCHEMA.md |
| `fit969400-431_hdv_hist_60s_manifest.json` | per-world tape provenance (source path, sha256) + counts |
| `fit969400-431_per_driver_params.csv` | population row + per-driver MLE fits of SUMO IDM {v0, T, a, b, s0} + sigma, on the recorded motion |
| `fit969400-431_per_driver_params_summary.json` | the same summary as JSON |
| `extract_hdv_histories.py` | the extractor (runs on any `c4_sumo_episode/1` tape dir) |
| `fit_likelihood.py` | per-driver / population likelihood + MLE; **accepts any injected physics model fn** |
| `ref_models.py` | SUMO 1.27.1 IDM step (cited) + LC2013 scaffold scorer |
| `SCHEMA.md` | npz schema |
| `tests/test_rbsetdata.py` | CPU pytest suite (6 tests, synthetic tapes) |

Code of record lives in worktree `E:/agent-work/c4-rbh-setdata`, branch
`c4/rbh-setdata` (`tools/rbsetdata/`); copies here for the rebuild agents.

## How to fit your own physics model

The recorded motion is fitted with a Gaussian velocity-step likelihood: for
every tick pair (t, t+1) with a leader, `v_pred = fn(v, gap, lead_v, dt,
**params)` and `v[t+1] ~ N(v_pred, sigma²)`; sigma is maximised out in closed
form (the NLL is monotone in RMSE).  `--cf-model module:fn` injects any model
with that signature (default `ref_models.idm_default_fn`, the SUMO 1.27.1 IDM
step, MSCFModel_IDM.cpp:225-269 semantics).  Fitting = deterministic
coordinate descent (grid scan + golden refinement per coordinate), CPU.

    python tools/rbsetdata/fit_likelihood.py \
        --hist fit969400-431_hdv_hist_60s.npz \
        --out my_params.csv --cf-model your_module:your_fn

## Headline results (2026-10-31, world 969400–969431 HDV, 60 s windows)

* Population IDM (226 431 tick pairs): `v0=29.87, T=0.753, a=0.701, b=1.687,
  s0=1.91`, sigma = **0.178 m/s** per 0.5 s tick.
* Per-driver: 2 517 of 2 797 segments fitted (≥30 CF ticks); medians
  `v0=30.2, T=1.04, a=0.86, b=1.62, s0=2.2`; median sigma = 0.017 m/s —
  the recorded HDV motion is essentially IDM-exact between lane changes.
* Caveats: `v0` sits at the 36 bound for speed-limited drivers (weakly
  identified when the leader caps speed); the LC scaffold's population fit
  collapses to the no-change solution (bias at −4, CE 0.017 over 199 393
  decisions, 652 events) — use the per-driver LC rows or plug the
  authoritative LC2013 scorer from rb-shared/lcspec via the same harness.
