# IDM adaptive compact wall v5.2 prospective protocol

Status: **executed at 30%; safety gate failed; predictor not authorized**.

Date: 2026-08-27

## Scientific decision

Primary estimand: whether an observable, physically feasible temporary wall is
expected to be safety-admissible, service-noninferior, and meaningfully
beneficial relative to identical decentralized IDM damping.

Collapse prevention is a prespecified secondary outcome. It is not required to
pass this safety/service pilot and cannot be a primary paper claim unless fresh
evidence later shows prevention diversity.

## Single treatment change from v5.1

Add both activation requirements:

```text
minimum selected-member front gap >= 20 m
minimum selected-member rear gap  >= 20 m
```

The 20 m value is development-derived. In excluded seeds 4101--4106, the three
unsafe completed walls had minimum gaps of 6.4, 7.2 and 18.0 m; all seven safe
completed walls were at or above 25.6 m. The threshold is the smallest round
value above the largest observed unsafe gap. This separation is hypothesis
generation, not validation.

No other treatment parameter changes:

- 20 s requested pulse;
- 200--1000 m recruitment band;
- full occupied-lane coverage;
- trajectory-coherent compact moving wall;
- 50 m maximum span;
- 10 s assembly timeout and 2 m/s convergence tolerance;
- adaptive servo;
- 20% maximum predecision reference-speed reduction;
- 1.0 m/s2 assembly/holding servo deceleration envelope;
- front/rear TTC projection and existing runtime aborts;
- release to identical analytical IDM damping.

Assembly failures and incomplete lane coverage remain treatment-dose failures.
They are not removed from the feasible-treatment denominator.

## Causal arms

Control:

> Frozen analytical selected-absorber IDM damping with the `disturbance_p30`
> reference and zero learned residual.

Treatment:

> The identical damping controller plus the v5.2 temporary wall.

Every branch uses the same 960-controller-step environment horizon. Outcome
scoring uses 260 postdecision steps. Changing decision time must change replay
horizon so `decision_step + replay_horizon = 960` in every branch.

## Prospective cohort

- demand: 2700 veh/h mainline plus 400 veh/h ramp;
- penetration: 30%, the lowest of 10%, 15%, 20%, 25%, or 30% that passed the
  treatment-blind availability rule in
  `docs/IDM_COMPACT_WALL_FEASIBILITY_SCAN_PROTOCOL.md`;
- HDV/CAV car following: frozen IDM contract;
- untouched wall-development seeds: 4301--4312;
- predeclared decision steps: 620, 660, 680 and 700;
- exact damping and engaged-treatment repeats: first two seeds with an admitted
  wall opportunity, selected by seed order without inspecting outcomes;
- development seeds 4101--4106 remain excluded from all prospective summaries.

A workspace audit on 2026-08-27 found no existing non-document run artifacts
for seeds 4301--4312. The manifest must repeat this audit and fail closed if an
existing assigned artifact is discovered outside the designated output.

## Required identity and dose checks

1. Damping-versus-damping exact replay passes on at least two seeds.
2. Engaged treatment-versus-treatment exact replay passes on at least two
   seeds.
3. Every treatment branch is physically identical to control through the
   declared decision.
4. Only latched wall members receive wall overrides.
5. Every out-of-band CAV retains the direct analytical damping command.
6. Release resumes identical damping.
7. Requested/realized duration, lane coverage, compactness, member changes,
   aborts, command deltas and safety projections are recorded.

## Frozen pilot criteria

Among activation-gap-admitted, formation-feasible cases:

- at least 80% receive at least 80% of requested duration;
- at least 80% lane coverage;
- compact cases complete assembly, spend at least 50% of override time in
  HOLDING, and remain within 50 m for at least 80% of geometry samples;
- at least 90% of completed treatments are safety-admissible;
- no excess collision or teleport;
- beneficial completed treatments span at least three independent seeds;
- nonbeneficial completed treatments span at least three independent seeds;
- completed treatments span at least six independent seeds;
- effects exceed exact-repeat/numerical noise.

Safety/service classification retains the existing frozen margins:

- hard-brake excess budget: max(2 vehicle-steps, 10% of damping exposure);
- low-TTC excess budget: 2 steps;
- maximum discharge loss: 2 vehicles;
- maximum delay loss: max(300 s aggregate, 5% of damping delay);
- meaningful benefit: at least 3 additional discharged vehicles or
  max(200 s aggregate, 2% of damping delay) benefit.

Collapse status, collapse onset, maximum sustained duration and prevention are
reported for every paired case but are not required for the primary
safety/service gate.

## Go/no-go decisions

Proceed to predictor data only if every identity check passes and the complete
safety/service pilot gate passes.

If dose fails, diagnose assembly, membership and frozen-lane coverage without
adding seeds. If safety remains below 90%, do not train a safety predictor on
this small pilot; diagnose the rejected/unsafe mechanism first. If positive or
negative completed outcomes span fewer than three seeds, do not generate a
paper-scale dataset. If the safety/service gate passes, freeze v5.2 source and
manifest hashes before any predictor-scale generation.

## Execution order

The treatment-blind penetration scan completed on 2026-08-30. The lowest
passing level was 30%: 12 admitted seed-window cases across 9 seeds and all
four decision steps. The analysis artifact SHA-256 is
`3b0baa9ad6b0ecaed0e1cfecd99b1c0c503d02b680a91402a0b3b39ea7475ad4`.
No treatment outcome influenced this selection. The paired pilot must consume
that exact feasibility analysis and its p30 case records.

## Commands

Focused and full tests:

```powershell
python -m pytest -q `
  tests/test_counterfactual_escalation.py `
  tests/test_counterfactual_twin_replay.py `
  tests/test_idm_wall_pilot.py
python -m pytest -q
```

Treatment-blind feasibility scan (completed):

```powershell
python experiments/idm/scan_idm_compact_wall_feasibility.py `
  --workers 4 `
  --output runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312
```

Corrected paired pilot:

```powershell
python `
  experiments/idm/run_idm_wall_multiwindow_pilot.py `
  --seeds 4301 4302 4303 4304 4305 4306 4307 4308 4309 4310 4311 4312 `
  --workers 2 `
  --feasibility-scan runs/probe/idm_compact_wall_feasibility_p10_p30_s4301_4312 `
  --output runs/eval/idm_wall_v5_2_prospective_p30_s4301_4312
```

The runner enforces the versioned v5.2 profile, the frozen analytical damping
arm, the treatment-blind p30 selection, and the `wall_development` seed role.
Estimated compute is about 20--45 minutes on two local workers for 12 admitted
wall branches plus the exact-repeat cases.

## Executed result

The paired pilot completed on 2026-08-30. Replay identity, causal-control
identity, dose, completed-seed diversity, beneficial/nonbeneficial diversity,
and zero excess collision/teleport checks passed. Safety admissibility was
5/10 completed cases (50%) against the frozen 90% requirement, so the overall
treatment gate failed and predictor training remains prohibited. See
`docs/IDM_WALL_V5_2_P30_PILOT_RESULTS.md`.
