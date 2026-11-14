# Preliminary tau-free counterfactual gate results (N=30)

Generated 2026-08-25 for the hyper-demand highway-merge scenario.

## Experimental design

- Ten SUMO seeds: 101--110.
- Three predeclared decision steps per seed: 120, 180, and 240.
- Three deterministic branches per case: off, tau-free soft wall, and strong wall.
- Treatment duration: 120 steps; outcome horizon: 300 steps.
- CAV penetration: 20% configured, with realized placement stored at each decision.
- Congestion label: queue speed below 10 m/s, at least 10 queued vehicles,
  pre-merge mean speed at most 10 m/s, sustained for 30 steps.
- Safety limits use the frozen no-control p95 calibration.

All 30 cases were eligible. Every case contains all three arms, no arm
terminated early, and no collision or teleport was observed.

## Readiness result

The frozen readiness audit returns `ready` for prototype gate training:

- Off labels: 21/30.
- Tau-free soft labels: 9/30.
- Strong labels: 0/30 because the strong wall violated the calibrated safety
  envelope whenever its extra traffic benefit would otherwise be attractive.
- Tau-free soft was safety-admissible in 25/30 cases.
- Tau-free soft was traffic-beneficial in 11/30 cases.
- Tau-free soft was both traffic-beneficial and safety-admissible in 9/30 cases.

The nine positive activation labels span six independent seeds. By decision
step, the label balance is 3 soft / 7 off at step 120, 2 / 8 at step 180, and
4 / 6 at step 240.

## Preliminary effects

Across the nine cases where the safety-first oracle activates tau-free soft,
the mean effect relative to off is:

- 720.3 veh-s lower queue area.
- 457.7 s lower incremental delay.
- 6.0 additional merge discharges.
- 5.44 fewer hard-brake vehicle-steps.
- 4.78 fewer low-TTC exposure steps.

Across all 30 cases, always-soft is not a good policy: it slightly reduces
mean queue area relative to always-off (11,802.7 vs. 11,895.0 veh-s) but has
higher mean delay (17,151.9 vs. 17,061.0 s) and slightly lower discharge
(319.1 vs. 319.6). This supports the need for a selective activation gate.

The safety-first counterfactual oracle activates in 9/30 cases. Relative to
always-off, it avoids 3 of the 8 sustained-congestion cases, reduces mean queue
area by 216.1 veh-s, reduces mean delay by 137.3 s, raises mean discharge from
319.6 to 321.4, reduces mean hard-brake exposure from 42.13 to 40.50
vehicle-steps, and reduces mean low-TTC exposure from 10.10 to 8.67 steps.
Its mean treatment duty is 0.12 rather than the always-soft duty of 0.40.

## Defensible claim today

The paired replay and labeling pipeline works, the tau-free wall produces
safety-admissible traffic benefit in a nontrivial subset of states, and the
30-case dataset has sufficient action/off balance to train a preliminary
placement-aware treatment-effect gate.

This is preliminary simulator evidence, not yet proof of predictor
generalization. Predictor evaluation must split by seed, not randomly by
decision window, because the three cases from each seed are correlated.

## Artifacts

- Dataset: `runs/probe/counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.jsonl`
- Generation summary: `runs/probe/counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.summary.json`
- Oracle report: `runs/probe/counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.oracle.json`
- Readiness report: `runs/probe/counterfactual_tau0_hyper_n30_s101_110_steps120_180_240.readiness.json`
