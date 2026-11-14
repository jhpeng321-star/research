# Observation-to-outcome execution — 2026-09-06

The owner adopted the [history-informed route](../output/history_route_audit/DECISION.md)
as the active goal. The target is effective CAV-observable escalation over the
frozen CIRCLES controller. Efficacy is an empirical requirement, not a promised
outcome. Damping learning and gate training remain paused.

Completed: all five registered replays passed exact original-outcome and
independent diagnostic reconstruction checks. The block consumed 12,000 simulated
seconds with zero retries. Comparisons are in
`runs/active/observer_replay_v1/comparisons/`; causal report summaries and event
attribution are in `output/observer_replay_v1/`. These are explanatory replays,
so the original physical effects and negative acceptance decisions are unchanged.

The acquired report evidence identified a specific native-handback sensing gap.
The separately registered [report-aware dissolution check](RSU_DISSOLUTION_CHECK.md)
tests that one hypothesis, using two treatment arms and reusing verified damping.
It is not a broad parameter search or a qualified gate.

## First budget

Five existing full-horizon arms, 12,000 simulated seconds maximum, zero automatic
retries, no new seeds, no controller/profile changes. Seed 4107 at M2700/p20:
damping, wall at 600 s, assembly-only at 600 s. Seed 4108 at M2700/p20: damping and
wall at 600 s. These are two explanatory cases. Original results stay immutable.

The first damping replay must pass an exact original-versus-observer comparison
before the other four can launch. Independent remaining arms may run in parallel.
Failed launches consume their allocation; a failure or mismatch stops further
launches. No additional physical smoke runs are allocated: observer behavior and
information boundaries are tested with synthetic APIs before this block.

Supported commands:

```powershell
python -m traffic_shepherd.escalation_study register-observer-replays --output runs/active/observer_replay_v1
python -m traffic_shepherd.escalation_study run-observer-replay --registration runs/active/observer_replay_v1/registration.json --job-id s4107_m2700_p20_damping
python -m traffic_shepherd.escalation_study analyze-observer-replays --registration runs/active/observer_replay_v1/registration.json
```

Registration freezes source/model/scenario identities and all original job and
communication specifications. Existing policy history remains 20 s. New offline
regional report reconstruction uses a separately declared 60 s history; it cannot
alter the controller or the saved decision evidence.

## Information and analysis

Record all-vehicle physical interactions at 0.5 s and join event contributors to
the anonymously reported local observations. Full-state identities are an offline
audit crosswalk, never online features. A detection is credited only if the actual
report contains matching geometry and it was received by the relevant cutoff.

Keep pre-engagement, during-assembly/HOLD, and post-release information separate.
Trace major paired TTC/braking/service differences beyond the already answered
phase totals. Examine the mechanism separating continued HOLD from assembly-only
release, and the contrasting seed. A retrospective explanation is not a validated
predictor and two seeds cannot establish observation sufficiency.

## Next decision

Only a timely, actionable report-based mechanism justifies one focused revision
with its own registration and budget. Neither another duration sweep nor a model
ladder follows automatically. Retain all physical costs and negative outcomes.
If the current maneuver imposes a physical cost that cannot be addressed by the
identified information, document that limitation and formulate an explicit
actuator change before further simulation. Acceptance data remains sealed.
