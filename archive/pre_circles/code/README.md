# Traffic Shepherd: IDM damping and counterfactual escalation

This repository studies mixed-traffic control at a freeway merge using SUMO,
TraCI, and Ray RLlib. The current IEEE IV project builds a bounded,
decentralized IDM damping controller and asks when it should escalate to a
temporary coordinated moving wall.

Working paper title:

> Counterfactual Escalation from Decentralized CAV Damping to Coordinated
> Moving-Wall Control

The predictor is **not yet verified**. The current work is the IDM damping
foundation and its training-readiness/acceptance gates. Historical Krauss,
health-gate, and early wall experiments are retained as evidence, not as the
active pipeline. See [docs/INDEX.md](docs/INDEX.md) for the authoritative
document map and [docs/LEGACY_RESEARCH_RECORD.md](docs/LEGACY_RESEARCH_RECORD.md)
for the consolidated historical record.

## Setup

```powershell
conda activate rl_driving
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense
python -m pytest -q
```

The shared network and demand files live in
`D:\research\highway_merge_shepherd` and are resolved through
`config/defaults.py`. Do not copy those assets into this repository.

## Active structure

```text
config/          experiment defaults and constants
env/             SUMO environment, observations, rewards, IDM residual, wall lifecycle
policies/        analytical damping, escalation, wall, and evaluation policies
counterfactual/  paired-replay contracts, outcomes, feasibility, models, readiness
idm/             frozen IDM provenance/seed contract and pilot metrics
experiments/     maintained IDM and counterfactual CLI entry points
scenarios/       repository-owned IDM transition scenarios
tools/           maintained audits, builders, diagnostics, and smoke checks
tests/           unit and contract tests
docs/            current protocols plus indexed historical evidence
paper/           current IEEE IV manuscript source
output/          shareable generated reports and presentations
runs/            ignored training/evaluation artifacts; see docs/ARTIFACT_MANIFEST.md

train.py         RLlib training entry point
evaluate_policy.py / evaluate_sumo_baseline.py
                 general evaluation entry points
experiments/idm/
                 IDM pilots, timing scans, and checkpoint evaluation
experiments/counterfactual/
                 treatment, paired-data, predictor, and audit entry points
```

## Scientific contract

The active control architecture is:

1. Frozen analytical IDM damping with a bounded learned residual.
2. Deterministic safety, jerk, follower, and absorber-selection projection.
3. A two-stage escalation decision: predecision feasibility, then conditional
   benefit/risk prediction.
4. A temporary assemble-hold-release moving wall layered over identical damping.
5. Smooth return to the same damping controller after wall release.

Before predictor-scale data or paper claims, the project requires replay and
predecision identity, adequate treatment dose and lane coverage, safety,
beneficial/nonbeneficial outcome diversity, and prospective independent-seed
evidence. The controlling plan is
[docs/IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md](docs/IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md).

## Verification

```powershell
python -m pytest -q
```

Cleanup baseline on 2026-08-27: **531 passed**. Any structural change must
preserve this result before new experiments are started.

## Workspace boundary

This is the dense/hyper and IDM/counterfactual workspace. Medium-density talk
optimization remains in `D:\research_v2` and must not be resumed here.
