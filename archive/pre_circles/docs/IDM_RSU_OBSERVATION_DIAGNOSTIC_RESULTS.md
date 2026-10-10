# IDM Live-Classic-RSU Observation Diagnostic

Status: complete development-only diagnostic; physical replay integrity passed,
but live existing RSU fields did not improve P30 action-benefit observability.
No MAPPO, predictor-scale data, treatment redesign, corridor-profile expansion,
or new seed was authorized.

Seeds 4121--4132 and every event in this report were already inspected and
remain permanently excluded from prospective claims.

## Scientific question and frozen scope

The diagnostic asks whether replacing artificial free-flow padding with the
same-step values of the existing eight classic RSU fields improves prediction
of helpful versus harmful P30 activations. It changes observation only.

Unchanged items include demand, seed, risk recurrence, decision step, candidate
CAV, P30 terminal reference, admission, safety projection, option duration,
release, 60-second outcome, labels and paired branches. No treatment branch was
regenerated. Replay stops at each frozen decision point.

## Reproduced cutoff audit

The source artifacts contain 30 P30 decisions at 2000 veh/h mainline plus the
frozen 400 veh/h ramp:

- recurrence 1: 12/12 candidates beyond 500 m;
- recurrence 2: 9/10 beyond 500 m;
- recurrence 3: 8/8 beyond 500 m;
- total: 29/30 beyond 500 m and exactly 29/30 classic blocks equal to
  `[0, 1, 0, 1, 0, 1, 0, 1]`;
- all four qualified harmful and all seven qualified helpful cases were padded;
- candidate absolute distance to the merge ranged from 499.514 to 1602.548 m.

The full per-case audit in
`runs/eval/idm_p30_rsu_live_m2000_observability_analysis_v1/summary.json`
records demand, seed, recurrence, decision step, vehicle, absolute merge
distance, disturbance-relative runway, live/padded state, qualification and
outcome.

The cutoff uses the candidate's absolute distance to the merge. Treatment
feasibility instead uses `candidate distance - disturbance distance`, which is
the available damping runway. These quantities must not be interchanged.

Outside 500 m, only the classic eight pre-merge, merge, ramp, ramp-queue and
merge-outflow values were padded. The ego block, four delay/service zone slots,
neighbor/trust set, wave signal, absorber gate, controller/pattern context and
separate action mask remained live. Every prior observation-sufficiency tool
treated the padding as ordinary numeric traffic state; none carried an
availability flag. The previous chance result therefore was not evidence
against live corridor RSU state.

## Versioned observation contract

The legacy `classic_rsu_configured_range_v1` contract remains the default with
the existing 500 m configured range. The new reversible diagnostic is named
`classic_rsu_live_2000m_diagnostic_v1` and requires exactly 2000 m.

The diagnostic preserves the 134-field order and every normalization. It adds
artifact metadata rather than actor features:

- observation schema/version;
- communication range;
- RSU availability;
- live/padded status for each classic zone block and shared delay block;
- candidate absolute distance to merge;
- same-step SUMO timestamp and age zero for available data.

The 2000 m contract is rejected unless the runner is in `--discover-only` mode
with a frozen P30 decision source. It cannot be used accidentally to launch a
treatment or training run.

Implementation files changed for this slice:

- `env/features.py`: version constants, range validation and causal RSU
  availability/timestamp metadata;
- `env/core.py`: passes the versioned contract to the unchanged assembler;
- `config/defaults.py`: records the legacy configured-range default;
- `tools/run_idm_decentralized_authority_verification.py`: observation-only
  source lock, separate physical/treatment/observation fingerprints and
  discover-only CLI guard;
- `tools/analyze_idm_rsu_observation_diagnostic.py`: matched cutoff audit,
  nested grouped models, calibration, coverage and clustered intervals;
- focused feature, runner and diagnostic tests;
- the results and two unimplemented future-design documents indexed here.

## Physical identity result

All 30 available frozen recurrences passed all identity checks:

- same demand and seed;
- same recurrence and decision step;
- same target vehicle;
- identical observation-independent physical-state fingerprint;
- identical P30 treatment and preview fingerprint;
- exact equality of all 126 non-classic-RSU physical observation fields.

Replay recovered 12 first, 10 second and 8 third recurrences. The other source
recurrences were already absent and remained absent. All 30 diagnostic classic
RSU blocks were available/live with same-step age zero. No treatment command or
postdecision state was generated.

## Grouped ablation contract

The primary sample is the same 11 fully qualified nonneutral P30 outcomes:
seven helpful and four harmful cases across seven development seeds. Inputs are
strictly predecision.

Compared feature families are:

1. stored 134 fields with legacy padding;
2. the same 134 fields with the existing classic RSU block live;
3. live 134 fields plus runway, predicted dose, terminal rear headroom,
   target-speed difference and preview projection margin.

The primary model is an L2-regularized logistic model. A deterministic
gradient-boosted-stump model is a nonlinear sensitivity. Outer evaluation is
leave-one-seed-out. Every operational threshold is chosen by an inner
leave-one-seed-out loop using only the outer training data. Confidence
intervals are 95% seed-clustered percentile intervals over fixed outer-fold
predictions; they are descriptive because the sample is very small.

## Results

For the primary regularized logistic model:

- padded 134: AUROC 0.214, balanced accuracy 0.321, Brier 0.619, harmful recall
  0.500, helpful recall 0.143 and coverage 0.273;
- live classic RSU 134: AUROC 0.214, balanced accuracy 0.446, Brier 0.631,
  harmful recall 0.750, helpful recall 0.143 and coverage 0.182;
- live plus authority 139: AUROC 0.214, balanced accuracy 0.518, Brier 0.629,
  harmful recall 0.750, helpful recall 0.286 and coverage 0.273.

The live-classic AUROC interval is [0.000, 0.600], balanced-accuracy interval
is [0.181, 0.733], Brier interval is [0.441, 0.865], harmful-recall interval is
[0.250, 1.000], and helpful-recall interval is [0.000, 0.500]. Live-minus-
padded AUROC is exactly 0.000 with paired fixed-prediction bootstrap interval
[0.000, 0.000], because live and padded variants produced the same out-of-fold
ranking in every resampled seed group.

Gradient-boosted stumps also failed: padded and live AUROC were approximately
0.18, while live plus authority reached only 0.25. The previous fixed-threshold
weighted-logistic analyzer returned AUROC 0.214 and balanced accuracy 0.286 for
all three families.

Baselines on the same 11 cases were:

- always activate: 100% coverage, four harmful and seven helpful activations;
- never activate: zero coverage and zero retained helpful activations;
- frozen predicted-dose-at-least-20-second rule: 90.9% coverage, three harmful
  and seven helpful activations.

The learned live-RSU threshold reduced harmful activations from four to one,
but did so by activating only two cases and retaining only one of seven helpful
cases. It therefore failed the nontrivial-coverage requirement.

## Frozen pass/fail decision

Integrity gates passed. Scientific observability gates failed:

- live-minus-padded AUROC at least +0.10: fail (0.000);
- grouped live AUROC at least 0.65: fail (0.214);
- balanced accuracy at least 0.60: fail (0.446);
- reduced harmful activations: pass, but only through near-abstention;
- at least 30% coverage and 50% helpful recall: fail;
- improvement robust to removing any seed or recurrence: fail; every removal
  delta was zero.

Conclusion: the existing live classic RSU fields do not materially improve
observability in this matched development diagnostic. This does not prove that
corridor information is useless. It rejects only these eight instantaneous,
coarse classic fields as a sufficient fix for the current P30 treatment.

## Failure diagnosis

The live fields vary, but helpful and harmful ranges overlap strongly. Ramp
queue is zero in every nonneutral case. The best individual post-hoc field is
merge outflow at AUROC 0.643; merge count reaches 0.607. Neither survives as a
grouped multivariable improvement. No such univariate value is a valid selected
predictor.

The more important limitation is the action-outcome contract. P30 is a sparse,
open-loop, fixed-duration action whose effect depends on shock interception,
rear response, safety projection, hold and release over a 60-second horizon.
One instantaneous whole-zone snapshot can describe loading yet still be
insufficient to identify the delayed causal response. The sample is only 11
nonneutral cases, and its confidence intervals are correspondingly broad.

This evidence favors treatment redesign and phase-resolved response logging
over a larger classifier. `corridor_profile_v1` is not promoted from this
failed screen. A spatial/temporal profile may be reconsidered only after a
rear-aware feedback treatment demonstrates repeatable action headroom.

## Exact commands

The replay command was run for `n=1`, `2` and `3`, substituting the matching
source and output directory:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
$py = "C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"

foreach ($n in 1,2,3) {
  & $py tools/run_idm_decentralized_authority_verification.py `
    --output "runs/eval/idm_p30_rsu_live_m2000_l${n}_development_v1" `
    --seeds 4121 4122 4123 4124 4125 4126 4127 4128 4129 4130 4131 4132 `
    --demand 2000 --workers 4 `
    --action-set activation_speed_ceiling_v1 --discover-only `
    --option-steps 20 --safety-horizon-steps 60 --cav-penetration 0.10 `
    --decision-index $n --maximum-landmarks-per-cell 3 `
    --reward-refs runs/eval/idm_decentralized_lower_reward_refs_v1.json `
    --candidate-ranking runway --candidate-count 1 --landmark risk_onset `
    --decision-source "runs/eval/idm_p30_p50_m2000_l${n}_p30_development_v1" `
    --rsu-observation-contract classic_rsu_live_2000m_diagnostic_v1
}
```

The ablation command was:

```powershell
& $py tools/analyze_idm_rsu_observation_diagnostic.py `
  --p30 `
    runs/eval/idm_p30_p50_m2000_l1_p30_development_v1 `
    runs/eval/idm_p30_p50_m2000_l2_p30_development_v1 `
    runs/eval/idm_p30_p50_m2000_l3_p30_development_v1 `
  --live `
    runs/eval/idm_p30_rsu_live_m2000_l1_development_v1 `
    runs/eval/idm_p30_rsu_live_m2000_l2_development_v1 `
    runs/eval/idm_p30_rsu_live_m2000_l3_development_v1 `
  --output runs/eval/idm_p30_rsu_live_m2000_observability_analysis_v1 `
  --bootstrap-replicates 2000
```

## Claim boundary and next step

There is still no verified trained IDM damping controller or working benefit
predictor. The correct next scientific decision is whether to approve the
rear-aware feedback treatment in
`IDM_REAR_AWARE_FEEDBACK_TREATMENT_PROPOSAL.md`. Observation/critic repairs in
`IDM_ACTOR_CRITIC_OBSERVATION_REPAIR_PLAN.md` are future prerequisites if and
only if a treatment passes physical headroom gates. MAPPO remains unauthorized.

## Software verification

- Focused RSU/observation/fingerprint/MAPPO-audit slice: 50 passed.
- Complete repository suite: 607 passed in 43.75 seconds.

Exact full-suite command:

```powershell
C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe -m pytest -q
```
