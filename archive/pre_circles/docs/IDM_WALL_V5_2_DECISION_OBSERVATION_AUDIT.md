# IDM wall v5.2 decision-observation audit

Status: **executed on existing development artifacts; no model fitted and no
new SUMO branches run**.

Date: 2026-08-30

## Purpose

This audit asks whether the frozen decision-state observation is ready for the
proposed feasibility and joint safety-benefit gates. It checks timestamp
identity, feature completeness and variation, target diversity by seed, and
coverage of the downstream region implicated by the v5.2 safety diagnosis.

It does not fit a classifier, select features, tune a rule, or estimate
prospective performance.

## Cohort recovered

- 48 declared decisions from 12 development seeds.
- 12 treatment-blind supervisor admissions across nine seeds.
- 10 compact-dose-qualified cases across seven seeds.
- Two failed-dose cases across two seeds.
- Five safe and five unsafe completed cases, each spanning five seeds.
- Three joint safe-beneficial cases across three seeds.
- Seven joint-negative completed cases across seven seeds.
- One feature contract across all 48 rows and zero prefix-identity failures.
- The corrected target builder reproduces every current joint label with zero
  errors; it now distinguishes legacy onset counts from the pilot's frozen
  hard-brake vehicle-step exposure rather than conflating the two metrics.

Thus Gate B has the minimum outcome diversity needed for a diagnostic screen.
Gate F does not: its failed-dose class spans only two seeds, below the frozen
minimum of three and far below what is needed for calibration.

## What the current observation contains

The stored artifact exposes 252 numeric decision-time fields, including:

- local and lane-level conditions on the upstream pre-merge edge;
- 5/10/20/30/60-second traffic lags and 60-second trends;
- queue, merge speed, speed variance, hard-brake, and low-TTC history;
- CAV placement and lane-coverage information;
- preflight member geometry, predicted span, gaps, TTC, and projected
  acceleration authority.

The causal timestamp contract is sound: features come from the control prefix
through the declared decision, and every paired prefix identity check passes.

## Decisive observation gap

There is no explicit state for post-merge edge `1487606178`, even though the
release-safety diagnosis places +231 aggregate excess hard-brake onsets there.

The existing field `local_downstream_minus_upstream_speed_mps` does not measure
the post-merge edge. Source inspection shows that both terms are calculated on
`PRE_MERGE_EDGE`: vehicles within 0--200 m of the merge are compared with
vehicles 200--400 m farther upstream. The name is spatially relative within the
approach and must not be interpreted as downstream bottleneck occupancy.

This omission matters because the unsafe branch divergence appears mostly more
than 60 seconds after release. A scalar instantaneous merge speed cannot fully
represent whether downstream storage, discharge, or a residual wave can absorb
the wall-induced inflow change.

## Feature variation

Fifty-three of 252 stored features are constant across the 12 admitted cases.
Seven belong to the frozen primary list:

- `health` and `health_change_30`;
- decision hard-brake and low-TTC counts;
- strong-band lane-coverage fraction;
- preflight member count and required-lane count.

These fields are not invalid globally, but they supply no discrimination in
this p30/m2700 development cohort. They must not be allowed to inflate the
reported input dimension or justify a neural network.

Descriptive safe-versus-unsafe differences occur in CAV fraction, active CAV
count, density history, and pre-merge speed-variance trend. With only five
cases per safety class, these differences are hypothesis-generating and may
not be used to select a final rule or claim predictive signal.

## Required observation v2 before fitting

The smallest justified extension is RSU aggregation over a frozen post-merge
region, not unrestricted vehicle trajectories. Add only decision-time and
strictly predecision histories for:

- post-merge vehicle count and occupancy/density proxy;
- harmonic/mean speed, minimum speed, and speed variance by lane and aggregate;
- post-merge hard-brake and low-TTC counts;
- merge-to-post-merge discharge and its 20/60-second trend;
- merge-versus-post-merge speed and occupancy gradients;
- an estimated downstream storage/recovery margin;
- 5/10/20/30/60-second lags for the compact aggregate family.

Gate F should continue to use only geometry, per-lane candidate availability,
gaps/TTC, projected authority, and dose-reliability history. Gate B may use the
upstream, merge, and post-merge traffic state. This preserves the scientific
separation between “can the wall be delivered?” and “will it be safe and
helpful?”

The v2 contract must be replayed on the already opened seeds first to prove
exact feature timestamps and branch identity. Any future data collection must
then be prospectively partitioned before outcomes are opened.

## Decision

Do not fit the current 252-field table. The next approved implementation slice
should:

1. freeze a compact observation-v2 schema with explicit post-merge aggregates;
2. add source-level and replay-identity tests;
3. retain a deterministic physical feasibility screen;
4. collect enough treatment-blind admitted cases for at least three failed-dose
   seeds before even a diagnostic Gate-F model;
5. run simple grouped, cross-fitted baselines before considering a neural
   network.

## Reproducibility

- Analyzer: `tools/audit_idm_wall_v5_2_decision_observations.py`
- Tests: `tests/test_audit_idm_wall_v5_2_decision_observations.py`
- Machine-readable result:
  `runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312/decision_observation_audit.json`
