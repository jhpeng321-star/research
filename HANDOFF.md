# S-RB-FIT — recovered final delivery, 2026-11-02

Resume complete: both previously SSH-blocked fits independently verified and
published; current-source physical LS fit reused the completed decision cache.
All three owned job PIDs exited. Old watcher exited; no repeated LC inference.
Recovery receipt:`artifacts/rbfit/receipts/RESUME_20261102.md`.

<!-- LC-CONTINUATION:BEGIN -->
Latest verified `lane_raw_5a5b15ef118c` (2026-11-02T09:04:39+08:00); LC SHAe61f897c5e7a, snapshotv2;3130driver posteriors.
Raw physical expected changes435.7 vs626recorded; eventBrier0.002308. Likelihood-emission Brier0.365999 is a separate diagnostic. **NOT ADOPTED**; no free-running/action-effect qualification.
Boundary-aware LC sampler required; weights already conditioned. Inverse temperature=0.2656413574761416; IDM preferred delivery remains `shrinkage_v1`. Live receipts:LC_WATCH_0128_STATE.json.
<!-- LC-CONTINUATION:END -->

**Exact directed event fidelity:**60.983 expected matches/626 recorded events;
ratio-of-expectations precision13.998%, recall9.742%. Same-cache source defaults:
263matches/12383positive steps, precision2.124%, recall42.013%. Lower Brier does
not establish correct timing/position or autonomous lane behavior. Learned
population puts94.549% mass atlcKeepRight0; this is inferred, not hand-disabled.

Branch `c4/rb-fit`, basef0db3c82. Prior work preserved; owner instructions in
REGISTRATION5683/5701/5734/5736. Original continuation window has ended; recovery
preserves that history. Previous handoff:docs/RBFIT_HANDOFF_ARCHIVE_0205.md.

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

## Latest LC export / integration
- `artifacts/rbfit/lane_raw_5a5b15ef118c`:209source-domain profiles,3,130causal
  joint posteriors;1,599zero-information histories=population. Fit240..300s,
  later conditional validation300..360s on all32native FIT worlds.
- Frozen LANE80541278, LC SHAe61f897c5e7a, geometry59017d3ec30a; snapshotv2
  restores PRE-change lane origins. Source bundle:`source_versions/5a5b15ef118c`.
- Recovered likelihood exports:`lane_source_da190640f0ca` and
  `lane_source_5a5b15ef118c`. Latest likelihood fit emission Brier.002341 hides
  5388raw expected proposals; raw LS is the candidate for physical testing.
- Raw LS causal inverse-temperature.265641 is already applied to driver weights.
  Physical population mean/SD and across-driver posterior-mean SD in VERIFIED.json;
  weak driver separation is retained as uncertainty, not invented labels.
- LC schemas `wm_phys.lc2013_{population,posteriors}.v1`; five-key order
  `[lcStrategic,lcCooperative,lcSpeedGain,lcKeepRight,lcAssertive]`; joint support
  and driver weights, conditioning_end_s300. Weights already conditioned.
  Emission probabilities are discrepancy only, NEVER runtime event noise.
- Boundary-aware reader: first4LC coordinates≥0, assertive>0; cooperative[0,1].
  FIT `draw_lane_parameters` supports zero behavioral settings. NEW explicit
  `lc_fit_for_unc.load_boundary_fitted_state` works with actual UNC planning;
  Current UNC also supports source-zero LC directly. Latest export:all3,130driver
  draws+16paired/replayed actual UNC futures PASS, CAV/padding/RNG preserved.
  Receipt:`receipts/LC_DELIVERY_20261102.json`. IDM unchanged; LC experimental.

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
  Current-source11numerical tests rerun PASS1.68s; independent posterior error
  ≤2.06e-15. Cache-only raw fit PID3951028 and recovered977061/1076804 exited.
  Initial coordinate overlay missing secure_gap dependency repaired pre-dispatch;
  original failure/success receipts retained. No GPU/SUMO/neural training.
- Report≤40lines:`docs/paper_results/RBFIT.md`; reproducer/JSON APIs:
  `tools/rbfit/README.md`. Coordination:`rb-shared/fit/CONTINUATION_0128.md`.
