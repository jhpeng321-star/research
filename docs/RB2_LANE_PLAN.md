# Round-2 lane repair contract

- Base FULL-v4 `22cc32f6`; no new behavioral thresholds, neural corrections,
  runtime emission noise, SUMO runs, or population refitting.
- Diagnose only FIT969400–969431, native first; use all five FIT arms as a
  disclosed intervention diagnostic. Reserved969432–969439 excluded.
- Consume S-RB-FIT `lane_raw_c5602102891e` unchanged, including population
  weights, five-coordinate joint profiles, inverse temperature and causal
  observation likelihood. One draw per driver/future, fixed across actions;
  entrants use population. No averaging discrete profiles into a driver.
- Saved native tape driver IDs cannot be transferred to other observable
  exports. Recondition LC from causal240–300 motion in the caller's ID space;
  parameter inference sees only pre-start observations. Reconstruct wishes
  for the sampled profile from that same prehistory.
- Source audit: fixed-point accumulation/thresholds; anticipation and lookahead;
  motivation-specific decay (there is no generic `changeWish` in this tag);
  gap vetoes; sublane-off action cadence; reset on actual lane change;
  serial negotiation limits. Preserve previous immutable diagnostics.
- Report exact positive-step matches and compressed runs separately. A
  teacher-forced proposal count is not a free-running event count. Counts and
  vehicle-time rates must share the recorded target cohort/window.
- Publish tested code plus reproducible FIT receipts for INT. No qualification
  claim follows from a conditional lane diagnostic.
