# HANDOFF — rbh-setdata (branch c4/rbh-setdata)

Updated: 2026-11-01 ~00:45. Status: DONE (deliverable shipped to
`E:/agent-work/rb-shared/setdata/`, see DONE.md there).

RESUME VERIFICATION 2026-11-02 (fresh session, no rework): 6/6 CPU tests
pass (`python -m pytest tools/rbsetdata/tests -q`, 11.8 s); rb-shared copies
byte-identical to commit cdcc7c1a (sha256 match on all 5 code/doc files +
tests); npz loads clean (58 arrays, 32 worlds, 2 831 segments, 256 527
ticks, schema `setdata_hdv_histories/1`); manifest sha256s present;
per-driver CSV 2 798 rows (population + per-driver modes); summary JSON
headline matches DONE.md (v0=29.87, T=0.753, a=0.701, b=1.687, s0=1.91,
sigma=0.178 m/s, n=226 431 cf tick pairs). No changes to deliverables.

RESUME VERIFICATION 2026-11-04b (second fresh session today, no rework):
6/6 CPU tests pass (9.7 s); all 5 rb-shared code/doc files sha256-identical
to worktree; npz + manifest + CSV/JSON present, timestamps unchanged from
2026-11-01. No changes to deliverables; task remains DONE.

RESUME VERIFICATION 2026-11-03 (fresh session, no rework): 6/6 CPU tests
pass (30.7 s); all 6 rb-shared files sha256-identical to worktree commit;
npz re-checked: schema `setdata_hdv_histories/1`, 32 worlds, 2 831 segments,
256 527 ticks, 652 LC events, 1 094 junction exits (104 ramp merges) — all
match DONE.md. No changes to deliverables; task remains DONE.

RESUME VERIFICATION 2026-11-04 (fresh session, no rework): 6/6 CPU tests
pass (3.6 s); all 6 rb-shared code/doc files sha256-identical to commit
cdcc7c1a; npz re-checked: schema `setdata_hdv_histories/1`, 32 worlds,
2 831 segments, 256 527 ticks, 652 LC events, 104 ramp merges — all match
DONE.md. No changes to deliverables; task remains DONE.

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
