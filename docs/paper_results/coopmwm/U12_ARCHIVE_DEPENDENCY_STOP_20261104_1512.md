# u12 diagnostic dependency stop — 2026-11-04 15:12 UTC+8

This supersedes the earlier A18/B11/R11 liveness observation.

- r3 B failed at 15:00:38 while loading the first isolated-synergy snapshot:
  `_decision_snapshot -> _ls_snapshot -> envelope['episode_path']/RESULT.json`.
  Missing exact input:
  `/home/ubuntu/coopmwm20261102_warm1/jobs/lscollect1_B3_986052/RESULT.json`.
- B12 checkpoint exists and all 24 ordinary update jobs completed without
  episode failure. No B_UPDATE_012 receipt is sealed yet. B12 composite safety
  passes: 24 pairs, collisions/teleports zero, critical-total B-A +3.5833
  veh-events, 95% CI [-14.75, 23.1875].
- LESSON_CHECK_u012_B contains 12 snapshot_error rows, zero graded entries.
  This is missing diagnostic data, not evidence of a policy loss or cooperation.
- R then exited on the durable guided engineering guard; r3 supervisor exited
  on its failed B child. Their three PIDs are absent. A19, monitor4 and both
  movers remain live. No completed physics has been repeated or checkpoint
  adoption attempted here.
- Likely source is the MAIN14:35 SHA-gated warm1/jobs move to cpuS. Trace the
  exact manifest/pointer before restoring; protect all transitive episode_path
  RESULT dependencies. Recovery must adopt the already-written B12 checkpoint
  and never perform its completed PPO update again.
- MAIN/operator notified through `_sol/COOPMWM_U12_ARCHIVE_STOP_1512_MAIN.md`.
  Evidence: `U12_CONTINUATION_1793776309256241300.json` and
  `RESOURCE_READINESS_1130_1793776273094578100.json`. Zero new sim-s.
