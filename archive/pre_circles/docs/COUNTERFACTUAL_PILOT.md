# Counterfactual treatment-effect pilot

This implementation turns the treatment-gate premise into a falsifiable SUMO
experiment. It does **not** train a model yet. Its purpose is to establish
whether action-conditioned headroom exists before model complexity is added.

## What is implemented

For each seed, the generator:

1. replays the no-wall episode to a causal warning landmark;
2. replays three branches from time zero with the same SUMO and CAV seed;
3. verifies that off, soft, and strong have identical pre-treatment hashes;
4. applies the selected wall for a fixed treatment window, then releases it;
5. derives future labels from traffic outcomes rather than from health;
6. writes one JSONL causal unit containing decision-time features and all
   three potential outcomes.

Decision features include the complete 16-dimensional supervisor state,
health at 5/10/20/30-step lags, path-shape summaries, queue/speed state, inflow,
and CAV coverage. The paired generator also records queue, pre-merge speed,
merge speed, density, inflow, and CAV coverage at 5/10/20/30/60-step lags plus
rolling 60-step changes and ranges. Health is one feature, not the target.

Outcome labels include sustained congestion, queue area, active-vehicle
incremental time loss, merge discharge, hard braking, longitudinal low-TTC
exposure, SUMO collisions/teleports, and treatment duty. The queue/event
thresholds in the CLI are **provisional** until calibrated and frozen on
no-control training seeds.

## Run the paired pilot

The default scenario is the workspace's hyper demand (`osm.hyper.sumocfg`).

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;D:\research-dense"

python experiments/counterfactual/generate_counterfactual_pilot.py --seeds 101 102 103
python experiments/counterfactual/analyze_counterfactual_oracle.py
```

Calibrate the independent event and no-control safety distributions before a
paper-scale pilot:

```powershell
python experiments/counterfactual/calibrate_counterfactual_outcomes.py --seeds 101 102 103 104 105 106 107 108 109 110
```

To sample several predeclared causal decision times rather than only the first
health warning:

```powershell
python experiments/counterfactual/generate_counterfactual_pilot.py --seeds 101 102 103 --decision-steps 60 120 180 240
```

The generation output is
`runs/probe/counterfactual_pilot.jsonl`; the aggregate oracle report is
`runs/probe/counterfactual_oracle_report.json`.

`--min-inband-scaled 0.2` is an optional sensitivity experiment requiring one
CAV already in the wall band. It is not the default because availability is a
legal decision-time predictor: warning states with no useful actuator coverage
should teach the gate to remain off.

## Oracle interpretation

The oracle sees all three future branches and chooses the best arm. It is an
upper bound, not deployable. It first rejects collision, teleport, hard-brake,
or low-TTC excess relative to off, then minimizes sustained congestion, queue
area, and delay while preferring higher discharge and lower duty.

The analysis classifies cases as rescue, mitigation, preventive benefit,
unnecessary/tied, harmful-only, or safety-rejected. A treatment-effect model is
worth building only if the oracle selects both off and active treatments across
seeds and materially beats always-off and always-on baselines.

## First full-horizon integration case

Seed 101 was run on hyper demand with the contract's 120-step treatment and
300-step outcome horizon. At the first warning landmark:

- health path (oldest to current): approximately `0.855, 0.014, 0, 0, 0`;
- pre-merge queue: `0` vehicles;
- pre-merge mean speed: `26.14 m/s`;
- wall-band CAV count: `0`, despite 13 active CAVs and two covered pre-merge
  lanes.

The health-only landmark therefore fired before traffic outcomes justified
control. Off avoided the provisional sustained-congestion event; soft and
strong induced it and increased queue area, delay, and hard-braking exposure.
The oracle selected off.

This single case is not a performance claim. It is evidence that a low health
value is not equivalent to positive treatment effect—the exact distinction the
new gate is meant to learn. It also shows why the next scientific step is
decision-landmark/outcome calibration and a multi-seed headroom pilot, not
training a neural network immediately.

## Current headroom screen (2026-08-25)

A provisional hyper screen at decision step 180 produced the first actionability
contrast under one common outcome definition:

- seed 101: neither treatment dominantly improved traffic; both reduced queue
  area only by worsening delay and discharge, so the least-intensive benefit
  action was **off**;
- seed 102: **soft** prevented the provisional severe event, reduced queue area
  by 4,305 vehicle-seconds, reduced delay by 2,494 seconds, discharged 14 more
  vehicles, and reduced low-TTC exposure by 8 steps;
- seed 102 soft also added 44 hard-brake vehicle-steps. With the current strict
  no-safety-excess rule, the deployable oracle therefore remained **off**.

This is evidence of traffic-treatment heterogeneity, but not yet safe policy
headroom. The gate must not be trained as deployable until a safety budget is
frozen from no-control calibration seeds or the wall mechanism is redesigned to
remove the network-level braking cost.

## Ten-seed calibration and readiness decision (2026-08-25)

Hyper no-control seeds 101–110 produced 40 complete calibration windows. The
highest-scoring definition satisfying the predeclared sample-count requirements
is:

- slow-vehicle threshold: `10 m/s`;
- at least `10` slow pre-merge vehicles;
- pre-merge mean speed at most `10 m/s`;
- persistence: `30` controller steps.

It labels 8 windows positive and 32 negative, with positive windows across six
seeds. Positive windows have a 1.51-standard-deviation higher future delay rate
and a 0.43-standard-deviation lower discharge rate. Treat this as the Phase-1
freeze candidate; it still needs held-out seed validation.

The no-control `p95` safety envelope is 0.201 hard-brake vehicle-steps/s and
4.67% low-TTC exposure, with zero collision and teleport windows. Seed-102 soft
has a hard-brake rate of 0.243, so it fails this envelope despite reducing TTC,
queue, delay, and the severe event. The automated readiness audit therefore
selects off for both current paired cases and blocks deployable gate training.

Artifacts:

- `runs/probe/counterfactual_outcome_calibration_hyper_n10.json`
- `runs/probe/counterfactual_readiness_hyper_n10cal_n2paired.json`

## Acceptance sequence

1. Calibrate the independent congestion label on no-control training seeds.
2. Freeze eligibility and outcome thresholds without looking at treatment
   comparisons.
3. Run a small hyper/dense paired pilot and inspect oracle action diversity.
4. Stop or redesign the treatments if the oracle almost always chooses off.
5. Only with measurable headroom, train simple regularized action-benefit
   models before considering a temporal neural model.
