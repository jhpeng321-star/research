# ~09:30 engineering freeze — FULL-v4 complete, no switching adoption

- Candidate`WM-PHYS-FULL-v4-gpu`,source22cc32f65a80af482a1a277c5f22377fa727d0a3.
  Recipe`configs/wm_phys.toml`;immutable gpuA runtime
  `/root/rbeval20261031/phys_code_22cc32f65a80`.
- Immutable freeze receipt`noon/20261101T093000+0800/FREEZE.json` was written
  at09:39:29 after the09:30 collection/scoring transaction finished; engineering
  decision was registered during that transaction. No model source was changed.
- Source-faithful SUMO-IDM/LC2013 highway subset/merge+move→LC→emit; fitted
  causal-motion IDM posterior with already-applied strength19.35568 and learned
  Gamma-Poisson arrivals; physics-native CAV/background; learned correctionsOFF.
- A/B/C selectors remain in the frozen recipe. Existing A2/B2/C2 evaluations
  use earlier4015b641 and remain separately source-labeled,not settings-only
  ablations. No finished inference is repeated merely to fill the time window.
- FittedLC staysOFF: FIT precision/recall/source compatibility and autonomous
  qualification unresolved. LANE09:00 observer closed with0new verified FIT
  summaries. Optional-source style completion does not change active defaults.
- FULL-v4**160/160 tasks,20complete K8 worlds,800arm-futures COMPLETE**.
  Local corrected distribution/paired/HDV60/credibility reports in09:30receipt.
  Independent read-only remote audit:56/56/48 COMPLETE,noSTOP; all3workers and
  monitor exited,MONITOR_COMPLETE; measured peak ownGPU3508MiB.
- Remote frozen original score is retained; current local corrected scorer
  and full-world uncertainty interpretation govern the morning report.
- A2/B2 each14complete worlds/124tasks; C2sixcomplete worlds/53tasks in this
  snapshot. Their original owners continue through11:30; collect more worlds.
- Reserved8 result remains point switchingFAIL:delay CRPS−393/−3040/−11647,
  HDV60−15.26m,but300srate-distance+5.625pp. Primary full-world CIs unavailable
  because singleton cells remain; probabilistic controller confidence unqualified.
- Model gate287CPU PASS/4skips+8recorded FIT smokes retained; current touched
  analysis/orchestration53CPU PASS3.14s. No newSUMO/training/cpuQ/peer signals.
- Collector12952(v7) acknowledged owner pause AFTER the freeze publication.
  Commit this documentation then resume; analysis/runtime source hashes unchanged.
  Final collection11:35; report by11:45. No automatic retries or duplicate jobs.
