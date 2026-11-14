# Placement probe: is CAV placement the key?

Date: 2026-08-06
User steer: test whether *where* the CAVs are placed drives seed outcomes and
whether upstream placement helps. Two tests: (1) entry-mix analysis of the
existing uniform champion@p20 eval; (2) an upstream-only placement eval.

## Test 1 — entry-mix analysis (uniform selection, per seed)

Reproduced each seed's CAV draw from the two flows (mainline 2900 veh/h,
ramp 600 veh/h; ~725 mainline + 150 ramp departures) with the per-seed seeded
selector at p20. WIN (29 seeds) vs LOSS (15):

| | mainline CAVs | ramp CAVs | upstream frac |
|---|---|---|---|
| WIN | 146.9 | 30.9 | 0.826 |
| LOSS | 142.9 | 28.3 | 0.835 |

NULL: placement mix is statistically identical between winners and losers.
Uniform selection already puts ~83% of CAVs on the upstream mainline, so
there is almost no between-seed placement variance to explain the outcomes.

## Test 2 — upstream-only placement (eval, full-50)

Added `cav_placement="upstream"` (CAVs only among vehicles traversing
PRE_MERGE_EDGE; ramp/downstream never CAV). champion@p20, full 50 seeds:

| config | dom | win | TTC-worse | d_ttc | d_gap | d_loss |
|---|---|---|---|---|---|---|
| uniform (control) | 27 | 24 | 15 | −1.52 | −3.76 | −3.42 |
| upstream-only | 10 | 7 | 33 | +2.62 | −0.37 | −0.65 |

Upstream-only is MUCH worse: TTC-worse 33, d_ttc positive.

## Verdict: placement is falsified

- No within-seed placement correlation (Test 1) and the deterministic
  upstream placement is actively harmful (Test 2).
- Side-finding: the ramp CAVs that uniform placement includes (~17%) are the
  *smooth mergers*. Removing them leaves a 100%-human ramp inflow that makes
  the merge more chaotic — so ramp CAVs were helping, not wasted.
- The RSU-formation/placement premise is falsified at the deployment level.
  The seed-variance root cause stands: outcome is set by the baseline's
  intrinsic chaotic state (no-op TTC varies 3..21 with zero policy), and the
  champion is a disturbance-reducer that helps only where the baseline is
  disturbed (corr(no-op TTC, d_ttc) = −0.55). The actionable lever is the
  benefit gate (meter only when the corridor is disturbed), not placement.

## Artifacts

- Eval: `runs/eval/champion_iter90_at_p20_upstream_15_dt05_holdout101_150`
- Filter: `env/core.py` `_placement_pool` + `cav_placement` config
- Driver: `_eval_champion_iter90_at_p20_upstream_15_dt05_holdout101_150.py`
