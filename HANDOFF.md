# HANDOFF — rbh-setdata (branch c4/rbh-setdata)

Updated: 2026-11-01 ~00:45. Status: DONE (deliverable shipped to
`E:/agent-work/rb-shared/setdata/`, see DONE.md there).

## What exists (new files only, no shared file touched)

* `tools/rbsetdata/extract_hdv_histories.py` — per-driver observable 60 s
  histories from any `c4_sumo_episode/1` episode tape (the S-WMDATA FIT block
  969400–969431 native arm on cpuQ
  `/home/ubuntu/wmdata20261031/jobs/s<seed>_native/deploy/episode/`).
  Output schema `setdata_hdv_histories/1` (`tools/rbsetdata/SCHEMA.md`):
  per-segment kinematics, leader/follower + adjacent-lane gaps (net bumper),
  lane-change events with decision-time features, junction-exit/merge events
  (the s≈0 ramp merge, incl. 2→2 ramp→acceleration-lane transits), flat npz +
  sha256 manifest.
* `tools/rbsetdata/fit_likelihood.py` — Gaussian velocity-step likelihood of
  recorded motion given ANY injected physics fn (`--cf-model module:fn`,
  contract `fn(v, gap, lead_v, dt, **params) -> v_next`); closed-form sigma;
  deterministic coordinate-descent MLE (grid + golden per coordinate) for
  per-driver or population parameters; LC scaffold scorer + multinomial CE.
* `tools/rbsetdata/ref_models.py` — SUMO 1.27.1 IDM step (numpy, elementwise;
  MSCFModel_IDM.cpp `_v` semantics, NUMERICAL_EPS=0.001, delta=4,
  stepping=0.25 → 2 sub-iterations at TS=0.5) + LC2013-style scaffold logistic
  (weights are fitted, not hand-set).  Supersede with rb-shared/idmspec and
  rb-shared/lcspec authoritative ports when they land; the harness takes any fn.
* `tools/rbsetdata/tests/test_rbsetdata.py` — 6 CPU tests (synthetic tape
  round-trip, neighbor/gap correctness, IDM behaviour + vector/scalar
  equality, parameter recovery on simulated IDM tapes, CLI end-to-end).

## How to run

    python -m pytest tools/rbsetdata/tests -q          # 6 passed, <5 s, CPU
    python tools/rbsetdata/extract_hdv_histories.py --episodes <tape dirs> --out hist.npz
    python tools/rbsetdata/fit_likelihood.py --hist hist.npz --out params.csv

Data product (already generated): `E:/agent-work/rb-shared/setdata/`
`fit969400-431_hdv_hist_60s.npz` + manifest + per-driver params CSV/JSON
(32 worlds, 2 831 HDV segments, 256 527 ticks, 652 LC, 1 094 junction exits).
Tapes pulled to `E:/Temp/opencode/fit_tapes/` (outside the repo).

## Known gaps / notes for S-RB-INT

* IDM likelihood excludes junction ticks and leaderless ticks; `v0` is
  weakly identified for leader-capped drivers (bound hits at 36 in ~10 % of
  drivers) — trust T/a/b/s0 more, or refit with the 300 s post-t0 window too.
* LC part is a scaffold (documented logistic over LC2013 incentive terms);
  its population fit collapses to the no-change solution under class
  imbalance (bias at bound).  Per-driver rows only where ≥8 decisions.
* `setdata_hdv_histories/1` keeps `lane_carry`-based neighbours through
  junctions (carried s); segments split on present-gaps (dropped ticks are
  indistinguishable from exits in the tape).
