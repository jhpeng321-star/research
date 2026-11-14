# Stage 2 generation results — final population for the frozen escalation policy

Date: 2026-09-05/06. Status: **descriptive report on the stage2_generation
population; no fitting.** Seeds 8001-8240 (p = 0.20) and 8241-8360
(p = 0.30), registered under `stage2_generation` and excluded from
prospective evaluation (`idm/contract.py`). Hidden per-seed mainline demand
M ~ uniform [2200, 2800] veh/h drawn from the seed number; plateau-decline
profile (plateau 0-1200 s, three 100 s steps to 1500 by 1500, tail to 1800,
ramp 400, horizon natural clearance or 2400 s). Seeds 4601-4660 and
5001-5060 untouched.

## Arms and definitions

- **D**: frozen analytical damping, no wall — the prefix for everything.
- **Onset (t_on)**: first step in the D prefix with merge-zone health
  h < 0.15 for 10 consecutive seconds (search capped at t = 900). Seeds
  without an onset by t = 900 are recorded and their W arms skipped.
- **WON_H300 / WON_H450 / WON_H650**: damping + sustained wall engaged at
  t_on + 1, held 300/450/650 s, 5 s taper, tau restore (10 m/s, tau +3.0,
  200-400 m mainline band, 0.3 m/s^2 clip). Exact prefix identity through
  t_on required of every branch.
- **Delay**: branch incremental delay in TOTAL seconds over all vehicles,
  and s/veh (total / distinct vehicles seen in the branch).
- **Completions**: vehicles exiting the merge edge within the branch.
- **Targets per seed**: Delta_delay(WON_Hk − D) for each k; best_k per seed
  (by total s); duration-choice value = Delta_delay(best_k) −
  Delta_delay(H450).
- **Features** (repaired capture; no M, no absolute time): full RSU vector
  at t_on, t_on − 60, and fixed t = 150, including merge-edge and
  post-merge mean speeds at 10/30/60 s lags (the 1c-2 onset predictors),
  region counts/speeds/stds and densities, CAV counts in band and
  upstream, lane imbalance, 30 s ramp arrivals and inflow/discharge, and
  per-lane geometry with **unclipped** nearest-to-band-top.

## Methodology notes

- Deterministic launcher (interprocess lock); 8-way parallel; D repeated
  on every 20th seed.
- Bootstrap CI90 = 10k resamples over seeds (rng seed 7).
- Onset-prediction AUCs are direction-aware (max(AUC, 1−AUC)): features
  such as merge-edge mean speed fall before onset, so raw AUC < 0.5 means
  strong discrimination in the negative direction. The "within 60 s" pool
  is pre-onset snapshots only: every t_on − 60 row is positive by
  construction and is pooled against t = 150 rows whose onset lies more
  than 60 s away; the "within 100 s at t = 150" label uses t = 150 rows
  with onset after 150 only.
-summary-tables-

## Results

(filled from `runs/probe/stage2_generation/summary.json` — SHA-256 recorded
below — after compute completes.)

## Artifacts and commit

- `runs/probe/stage2_generation_p20_s8001_8240/` and
  `runs/probe/stage2_generation_p30_s8241_8360/` (manifest + per-seed JSON).
- `runs/probe/stage2_generation/summary.json` (+ `.sha256` sidecar).
- Code: `stage2_generation_contract.py`, `run_stage2_generation.py`,
  `analyze_stage2_generation.py`, registration in `idm/contract.py`.
- Commit: (hash at commit time).
