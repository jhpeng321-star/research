# S-C1CLOSED wall-input recovery (no wall forecast)

- The historical wall cache survives with matching original SHA256 b269964e.
  Its 67 starts have only 10 half-second history rows, insufficient for WM2's
  registered genuine 60 s history. It contains wall members and a timeline
  summary, not a snapshot of the persistent wall executor.
- A second named cache, `wallc1_states_h60_real/wall_same.json`, **also survives**:
  SHA256 dc3be57206c359a4b26361956b815e5c6107f9e592195e0be00430d188efc01b;
  120 starts, genuine 121-tick / 60 s histories, old permitted worlds 8xxx.
  This corrects any inference that all full-history wall inputs were deleted.
- None of these 120 starts has a saved wall/controller executor state or a
  nonempty `cav_controllers` map. Membership and requested speed/hold survive,
  but the timeline also contains future release/abort/owner-end outcomes.
  Those future fields must not initialize or drive a closed-loop executor.
- All cached futures end at 120 s. This cache cannot supply 180/300 s labels.
  Raw episode/run paths were not opened, nor were any paths reconstructed from
  the timeline followed. No new wall panel was registered or inferred.
- Current blocker is **causal wall executor restoration and long-horizon
  recorded outcomes**, not absence of all trajectories/histories. A registered
  follow-up may establish a pre-t0-only restoration from public causal events,
  then freeze exact eligible starts before any wall inference. An ex-post
  membership/release replay is not full wall-v2.1 feedback.
- Receipts: `tools/c1closed/evidence/WALL_INPUT_SCHEMA.json` and
  `WALL_H60_AUDIT.json`; audit code `wall_inputs.py`, `wall_h60.py`. Zero SUMO,
  training, forecast, or score computation in this recovery.
