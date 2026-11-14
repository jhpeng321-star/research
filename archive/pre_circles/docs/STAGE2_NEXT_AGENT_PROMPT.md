# Paste-ready prompt for the execution agent

> **Stopped by owner; do not execute this prompt unchanged.** The subsequent
> [paper completion review](PAPER_COMPLETION_PLAN_2026-09-05.md) records the
> 1080/2400 horizon mismatch, damping command-restoration discrepancy, and
> controlling CAV-reported observation requirement. The replacement is a
> proposed plan, not a launched or approved experiment contract. The text
> below is preserved as the prior execution specification.

The historical prompt follows for provenance. Do not copy it into an execution task.

---

# Task: Stage 2 generation, then freeze, then prospective evaluation

You are executing a predefined plan in `D:\research-dense`. Follow
`AGENTS.md` and `docs/HANDOFF_2026-09-05.md`. Do not redesign the
experiment. Where this prompt is specific, do exactly that. If a stated
assumption about the code is false, **stop and report**.

## Setup

```powershell
conda activate rl_driving
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense
```

## Hard constraints

- IDM only.
- Do NOT open, read, or run seeds 4601-4660 or 5001-5060 except in
  Step 4 below, and only with a frozen hashed bundle.
- Do NOT start RL, a new v5.x wall, or decentralized-damping training.
- Do NOT modify frozen v5.x contracts. New code in new files.
- Preserve the dirty worktree. Review existing uncommitted Stage 2 files
  before rewriting them.
- `python -m pytest -q` before any SUMO compute. Report the count.
- Artifacts under `runs/probe/`. `runs/` is gitignored.
- Report delay in s/veh AND total s. Completions = merge-edge exits
  within the branch.
- Features must never include hidden demand M, absolute time, seed id,
  or any treatment-arm / outcome field.
- Commit only when the user asks, or after a completed results document
  if the existing project convention has been to commit each stage
  (this repo has been committing stage results; ask if unsure).

## Step 0 — review the uncommitted Stage 2 machinery

These files already exist in the worktree (may be uncommitted):

- `experiments/idm/stage2_generation_contract.py`
- `experiments/idm/run_stage2_generation.py`
- `experiments/idm/analyze_stage2_generation.py`
- `tests/test_stage2_generation_contract.py`
- `scenarios/idm_stage2/` (partial)
- `idm/contract.py` Stage 2 seed block 8001-8360

Review them against the contract in `docs/HANDOFF_2026-09-05.md` §3 and
§8. Fix only defects (leakage, wrong onset definition, clipped
geometry, missing integrity fields). Add focused tests if missing.
Run `python -m pytest -q`. Do not launch SUMO until tests pass.

Smoke: seeds 8001-8002, `--strate p20 --workers 1`, all arms. Verify
prefix identity through t_on, first divergence at t_on+1 (or later if
no in-band CAV — record the distribution), D-repeat if the seed is a
repeat seed, repaired CAV counts vary, no feature is the hidden demand.
Stop if identity fails or a branch exceeds ~500 s.

## Step 1 — Stage 2 generation (compute)

Population and arms are already in `stage2_generation_contract.py`:

- Plateau-decline; M ~ Uniform{2200..2800} seeded from the seed number
  (hidden); ramp 400; horizon natural clearance or 2400 s.
- p20: 8001-8240. p30: 8241-8360. Run p20 first, then p30. Do not
  overlap the two strata.
- Arms: D; WON_H300 / WON_H450 / WON_H650 engaged at onset
  (first D-prefix step with h < 0.15 for 10 s, search cap t = 900).
- Features: repaired RSU vector at t_on, t_on-60, and t = 150.
- 8 workers. Deterministic launcher already imported in the runner.

```powershell
python -m experiments.idm.run_stage2_generation --strate p20 --workers 8
python -m experiments.idm.run_stage2_generation --strate p30 --workers 8
```

When p20 finishes, run the analyzer and write
`docs/STAGE2_GENERATION_RESULTS.md` covering the six report items in
`analyze_stage2_generation.py`'s docstring (inventory/integrity;
per-bin beneficial fraction and median Δdelay; value of choosing
duration; p20 vs p30 when p30 exists; onset-prediction AUC check;
surprises). Report p20 to the owner immediately; do not wait for p30
to start the Step 2 design, but do not fit a frozen bundle until p30
is in or the owner says p20-only is enough.

## Step 2 — fit and freeze (artifact-only; after generation)

New files under `experiments/idm/` and `docs/STAGE2_FROZEN_GATE.md`.
Data: Stage 2 seeds only. Group = seed. Forbidden features as above.

Fit three components, each with grouped 5-fold CV, 3 fold seeds:

1. **Onset detector.** Logistic/ridge on the t=150 (and t_on-60) vector
   for "onset within 100 s" / "within 60 s". Report OOF AUC vs the
   simple rule "h < 0.15 for 10 s" (the treatment trigger). The frozen
   trigger may remain the h-rule if the learned detector does not
   improve lead time in a paired check; say which and why.
2. **Whether (escalate).** Fair single-feature threshold L0′ (choose
   feature + k in-fold to maximize policy value at λ=0 and λ=2) versus
   ridge on loading features (counts, densities, upstream_inflow_30s,
   merge/post-merge speeds). Headline: does the ridge beat L0′ at λ=0
   with CI90 clear of zero? If no, the frozen whether-gate **is** L0′.
   Also report a 10-bin reliability table for P(Δdelay < 0).
3. **Duration.** Among seeds the whether-gate would engage, predict
   argmax_k Δdelay(WON_Hk − D) for k ∈ {300,450,650}. Compare to
   always-H450 and to the per-seed best_k oracle. Report OOF value of
   choosing duration.

Policy value: engage when predicted Δdelay − λ·P(prevention) < 0, then
apply the duration head. λ ∈ {0, 1, 2, 4} s/veh per prevented collapse.
Report vs never / always-H450 / L0′ / per-seed oracle, seed-bootstrap
CI90.

Freeze one bundle (joblib + JSON manifest + SHA-256). Record exact
feature names, thresholds, duration map, and code identity. This is the
only object that may later be evaluated on untouched seeds.

Write `docs/STAGE2_FROZEN_GATE.md`. Claims not made: no prospective
number, no generalization.

## Step 3 — prospective closed-loop (compute; after freeze)

Assign a **new unused** seed block (e.g. 8401-8460) in `idm/contract.py`
as `stage2_prospective` and add it to the exclusion set **before**
running. Do **not** use 5001-5060 here.

Profile and wall as Stage 2. Per seed, one D branch and one **closed-loop
policy** branch that: detects onset on its own live state (or the frozen
detector); applies Gate F (CAV supply); applies the frozen whether-gate;
if escalate, holds the frozen duration; 5 s taper; no re-engagement in
v1. Also run always-H450 (engage at onset, hold 450), the legacy h-rule
(arm h<0.25, release h>0.45, re-engage allowed — see
`docs/SUSTAINED_WALL_V1_STAGE1B_PREP3_RESULTS.md` for the exact
semantics already implemented), and never (D only).

Report paired vs D: Δdelay (s/veh and total), incidence, duration,
completions, decel-cap and low-TTC by location, wall-on time, engagement
count. Seed-bootstrap CI90. Write
`docs/STAGE2_PROSPECTIVE_RESULTS.md`.

## Step 4 — reserved paper block (ONLY after the owner says go)

Seeds 5001-5060, **once**, same frozen bundle, same arms as Step 3.
Do not look at outcomes until the run is complete. Do not retune.
Write `docs/STAGE2_FINAL_RESERVED_RESULTS.md`.

## Report back after each step

1. pytest count; files created or changed.
2. Integrity: branches, identity failures, collisions/teleports,
   no-onset count.
3. The descriptive / OOF / prospective tables specified above.
4. Anything you decided yourself or any assumption that was false.

Do not invent Stage 3 experiments (ALINEA, ramped demand, p10, Krauss)
unless the owner asks. If Stage 2 p20's boundary or duration curve
contradicts 1c-2, stop and report before fitting.
---
