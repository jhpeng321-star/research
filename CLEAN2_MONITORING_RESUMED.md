# Clean2 monitoring continuity — 2026-11-05 22:14:21 +08

Recovered the preserved launch and existing monitoring. The read-only audit in
`CLEAN2_RESUME_AUDIT.json` succeeded on cpuV at 22:12:32.980 +08.

- Runtime remains `285663a103d18fd33e9137db14fba33b62423a23`; all 538
  deployed source hashes, CONFIG/approval/manifests and final qualification
  receipt bindings passed independent verification.
- End-to-end smoke: PASS, 14,400 sim-s. Single full-window gate: PASS,
  26/26 branches, zero collisions and teleports. Actual training launch was
  19:09:40.913 +08; the local follower registered it after reconnection at 22:06.
- Original owners A301793/B301794/R301795/P3monitor301796, start_ticks6682640,
  and cpuV chain296173/start6536748 are alive. Chain is MONITOR/RUNNING.
- Combined sealed monitors u1–u9 all say CONTINUE with no DIAGNOSE and zero
  collisions/teleports. Existing local follower9504 (created18:49:32) remains
  the sole local follower and has appended updates1–9, including u2/u4 probes
  and u6 replicated results, to `CLEAN1_STATUS.md` with CLEAN2 prefixes.
- u9 B/R bank CE is 0.73211/1.0141, layer KL0.0089448/0.0038716,
  selected-move infeasible-rate0/0, safety penalties-100.98/-138.08.
  R's critical-event excess21.5 CI95[0.59375,45.939] is still reported under
  its registered report-only rule; any collision/teleport remains a pause.
- u6 N5 test:10/10cells,64full-window branches,47,920sim-s. B total benefit
  vs common base is +726.399612veh-s CI95[27.701201,2013.195193]; B
  group-over-best-single synergy is -0.034074 CI95[-0.075602,-0.000048185].
  R's corresponding means are -59.383381 and -59.392844. This descriptive
  result does not demonstrate positive synergy. The primary u24 criterion
  remains group-over-best-single synergy; u12/u24 checks stay scheduled.
- Clean2 charges:1,144,520sim-s/1,040rows; global2,393,960/5,000,000.
  All effective phase caps passed. Clean2 disjoint cap3,184,160 and remaining
  approved plan_fix reservation remain bound by the original accounting.
  Clean1 durable STOP and ledger prefix are unchanged.

Verification used `ops/clean2_resume_audit_remote.py` over accepted F6 transport,
including existing exact smoke/fix/reporting/gate validators and clean1-prefix
accounting checks. The successful audit is committed as `e07fc449`.
No runtime source, controller, experiment configuration or monitor owner changed.
No additional physics was dispatched. Existing cpuV chain/P3 and local follower
continue monitoring; the cpuV chain is independent of laptop connectivity.

The matching timestamped continuation entry is appended to
`E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`.
