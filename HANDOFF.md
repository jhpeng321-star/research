# S-RB-FIT — owner01:28 continuation active

Resume2026-11-02: old watcher exited. Both previously SSH-blocked latest-source
fits are now COMPLETE on all32FIT worlds, with owned PIDs exited. Recovery receipt:
`artifacts/rbfit/receipts/RESUME_20261102.md`; independent verification/publication
in progress. Existing IDM delivery remains ready.

<!-- LC-CONTINUATION:BEGIN -->
Latest verified `lane_source_5a5b15ef118c` (2026-11-02T09:02:05+08:00); LC SHAe61f897c5e7a, snapshotv2;3130driver posteriors.
Raw physical expected changes5388.3 vs626recorded; eventBrier0.007397. Likelihood-emission Brier0.002341 is a separate diagnostic. **NOT ADOPTED**; no free-running/action-effect qualification.
Boundary-aware LC sampler required; weights already conditioned. Inverse temperature=not applicable; IDM preferred delivery remains `shrinkage_v1`. Live receipts:LC_WATCH_0128_STATE.json.
<!-- LC-CONTINUATION:END -->


Branch `c4/rb-fit`, basef0db3c82. Prior work preserved; owner instructions in
REGISTRATION5683/5701/5734/5736. Continue substantive LC source refits through09:00;
INT freezes09:30. Detailed previous handoff:docs/RBFIT_HANDOFF_ARCHIVE_0205.md.

## Ready for INT / UNC
- Preferred IDM:`artifacts/rbfit/shrinkage_v1`; already-materialized joint
  population/driver posteriors.3,130drivers/174no-evidence population fallbacks.
- Existing schemas `wm_phys.idm_population.v1` / `wm_phys.idm_posteriors.v1`;
  parameter_order `[v0,time_headway,accel,decel,min_gap]`, units m/s,s,m/s²,m/s²,m.
  Population support[C,5]+weights[C]; driver support_indices+weights select
  that same support. Moments are summaries, not new log-normal draws.
- Identity=world:observable numeric VID, causal60s ending300. Unknown/missing
  history→population. Do not mistake packed slots for tape IDs. `condition_motion`
  supports caller-supplied causal motion without saved-ID assumptions.
- Learn prior_count19.35568 from prefix CRPS; ALREADY APPLIED, do not shrink twice.
  UNCf8b6faea `FittedHiddenState.from_directory` verified on all exports and16
  paired/replayed futures. Draw once/cache per identity, common across actions;
  future entrants use the same learned population. v0/actual speed limit=speedFactor.
- Pooled point:`pooled_full_v5/idm_pooled.json`, v0/tau/a/b/s0=
  29.839994/.770429/.795080/1.772102/1.933667; delta4. Shared model files untouched.

## LC continuation (02:02 dispatch snapshot)
- Published `artifacts/rbfit/lane_directional_v3`: all3,130sequence posteriors
  independently verified,1,630zero-information histories=population. Later-FIT
  emission eventBrier.002359 vs population.002463/source-rate.002412. Raw physics
  still7811expected changes vs626real; **NOT ADOPTED**. Old search permits
  cooperative>1; boundary v4 corrects that domain. Prior v1/v2 outputs preserved.
- Boundary v4:209source-domain profiles,33cached/176new; full job579949cpuQ.
  Raw-physical LS next uses that completed cache rather than rerunning LC.
  One-world v5 smoke independently verified:99expected vs30real changes
  (v4raw301/default533), still no free-running qualification.
- Latest LANE0a1e7e93 frozen at `source_versions/ec0f2a74974b`; physical source
  SHAc1aa89fafb69, geometry59017d3ec30a. New snapshotv2 restores PRE-change
  upstream origins. Full32FIT-world/209profile job617309cpuQ,2nice19workers.
  No old-source decisions reused across physics/snapshot changes.
- LC schemas `wm_phys.lc2013_{population,posteriors}.v1`; five-key order
  `[lcStrategic,lcCooperative,lcSpeedGain,lcKeepRight,lcAssertive]`; joint support
  and driver weights, conditioning_end_s300. Weights already conditioned.
  Emission probabilities are discrepancy only, NEVER runtime event noise.
- Boundary-aware reader: first4LC coordinates≥0, assertive>0; cooperative[0,1].
  FIT `draw_lane_parameters` supports zero behavioral settings. NEW explicit
  `lc_fit_for_unc.load_boundary_fitted_state` works with actual UNC planning;
  5tests/8paired-replayed futures PASS, IDM unchanged. LC remains experimental.

## Evidence / limits
- Native FIT969400–969431 only, six explicit observable files. No settings labels.
  CF mainline only; acceleration derived from speeds, stable lanes/leaders.
- Conditional later-FIT pooled acceleration RMSE.299 vs initial.902m/s²;10s
  position MAE5.58 vs constant-speed13.81m. Shrunk physical acceleration CRPS
  .14657 vs raw.16351/population.16560;95%coverage90.48%, not full calibration.
- SUMOCORE six-FIT-state diagnosis: inferred-minus-default600s MAPPO/PI effects
  +4171/+3909veh-s, almost all road. Incumbent/entrant settings changed together;
  culpable parameter unidentified. Shrinkage is not demonstrated sign repair.
- Prior22CPU tests passed; new11numerical/runtime+7continuation+5UNC bridge PASS.
  Initial coordinate overlay missing secure_gap dependency repaired pre-dispatch;
  original failure/success receipts retained. No GPU/SUMO/neural training.
- Report≤40lines:`docs/paper_results/RBFIT.md`; reproducer/JSON APIs:
  `tools/rbfit/README.md`. Coordination:`rb-shared/fit/CONTINUATION_0128.md`.
