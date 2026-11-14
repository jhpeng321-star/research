# AGENTS.md — research-dense

**Latest owner clarification, 2026-09-06: prioritize an existing trained
controller; escalation is the intended contribution.** Start with
`docs/PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md`. A coordinating task owns the
bounded transfer check in `runs/probe/published_idm_fit_check_2026-09-06/`;
inspect its manifest/process state before doing anything and do not duplicate
its corridor or penetration runs. The released RL candidate is not yet linked
to an exact paper checkpoint or accepted in this setting. Do not claim otherwise.

The v1 campaign is checkpoint-paused at 36/8/8; do not restart either
coordinator. Native-selectable v2 completed its bounded learning/physical check
with NO-GO. The separate 480-second v2.1 recovery prototype removed the rebound
but worsened pulse effort and also stopped with NO-GO, with no training.
Read `docs/NATIVE_SELECTABLE_DAMPING_V2_RESULTS.md` and
`docs/NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md`, including the correction
to the nondiscriminative `getSpeedWithoutTraCI` handback statistic and the later
passing paired unfinished-command release/re-engagement check. Preserve all
sources, checkpoints and negative results. No bespoke successor campaign or
parameter search is authorized. Acceptance stays sealed; wall training paused.

Dense/hyper Traffic Shepherd workspace (SUMO + TraCI + Ray RLlib). The active
IEEE IV project is IDM-native decentralized damping followed by conditional
escalation to coordinated moving-wall control.

## Workspace boundaries

- `D:\research-dense\` — this codebase and its new dense/hyper experiments.
- `D:\research\highway_merge_shepherd\` — shared SUMO network and demand assets.
- `D:\research_v2\` — earlier medium-density work; reference only. **Do not
  resume medium optimization here.**
- Preserve the dirty worktree and all user changes. Never replace current IDM
  work with the older Krauss pipeline.

## Setup

```powershell
conda activate rl_driving
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "D:\research-dense;$env:PYTHONPATH"
Set-Location D:\research-dense
```

## Current scientific order

1. Audit provenance, observations and execution semantics of an existing trained
   damping candidate; validate its adapter with native IDM background traffic.
2. Require physical benefit, service/source protection and safety in bounded
   development checks. If learning is later necessary, first pass action-credit,
   reward-alignment and learning-signal gates under a separate frozen budget.
3. Prospectively evaluate the fixed candidate and adapter on untouched seeds.
4. Freeze that checkpoint as the counterfactual control arm.
5. Verify damping-preserving moving-wall dose, safety, and outcome headroom.
6. Only then train and prospectively validate the two-stage escalation predictor.

Do not generate predictor-scale data or claim prediction, calibration, collapse
prevention, or IDM generalization before its frozen gate passes. Record all
development seeds and exclude them from final prospective evaluation.

Authoritative navigation is in `docs/INDEX.md`. The owner clarified on
2026-09-05 that **standalone trained damping is required before escalation**.
The 2026-09-06 reuse priority above supersedes the bespoke implementation route
in `docs/TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md`. Historical implementation and
the bounded PyTorch PPO check are in `docs/NATIVE_DAMPING_V1_IMPLEMENTATION.md`
and `docs/NATIVE_DAMPING_V1_LEARNING_CHECK.md`; do not confuse that learner
with the historical RLlib per-vehicle terminal path.
For the stopped registered full campaign, see
`docs/NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md` and the single directory
`runs/rllib/native_damping_campaign_v1/`; inspect its current stage before
launching or resuming anything.
Historical scheduling was parallel across the three independent optimizer seeds;
read `docs/NATIVE_DAMPING_PARALLEL_EXECUTION_2026-09-05.md` and inspect
`runs/rllib/native_damping_campaign_v1/parallel_execution/` before acting.
The previous serial coordinator is stopped; do not restart it or duplicate
the isolated worker runs. Do not edit the active parallel wrapper or frozen
scientific sources while those processes are running. Reserved acceptance seeds must
not be generated during training/development. The older
`docs/HANDOFF_2026-09-05.md` and `docs/STAGE2_NEXT_AGENT_PROMPT.md` are
stopped historical directions, not prompts to resume. The v5.x master plan
(`docs/IDM_DECENTRALIZED_ESCALATION_MASTER_PLAN.md`) is historical: do
not resume the compact-wall treadmill. Negative analytical/P30 results do
not establish that trained damping is finished or authorize substituting an
analytical controller for the owner's required learned checkpoint. A new
learning contract must address the documented authority, observation and
credit problems; do not restart a rejected training run unchanged.

## Artifacts

- New checkpoints: `runs/rllib/`
- Evaluation/pilots: `runs/eval/`
- Treatment and predictor probes: `runs/probe/`
- Shareable generated files: `output/`

Do not put transient logs, virtual environments, caches, or rendered QA files
at repository root.

## Verify

```powershell
python -m pytest -q
```
