# S-C1FLOOR step 1 — source contracts (2026-10-31)

- Owner authority read verbatim: active C4 registration line 5567, 30,000 actual
  simulation seconds including prefixes, gpuR CPUs subordinate to SUMOCF/COOPMEASURE.
- Requested `docs/paper_results/C1CLOSED_DESIGN.md` does not exist. The live
  registered file is `E:/agent-work/c4-c1closed/C1CLOSED_DESIGN.md` (118 lines read).
  Primary starts are WMDATA 969400–969439, t0=300; horizons
  10/30/60/90/120/180/300; original-ID HDVs, median within start then across worlds,
  whole-world bootstrap 2,000 draws, seed 0, percentile 95% intervals.
- Reuse `tools/c1closed/labels.py` for recorded trajectory endpoints and left-boundary
  road/source delay components. Preserve source bytes and hash when staging.
  CSV analysis/aggregate-field implementation is an upstream dependency: no
  finalized horizon CSV or separate queue/segment-field scorer is present yet.
- Read S-COOPMEAS2 `tools/evalprep/coopmeas2_runner.py` and
  `traffic_shepherd/c4/branching.py`: precision/RNG save, Python snapshot,
  fresh SUMO for every restore, subscription/speedMode repair, loaderTime guard.
- Read WMDATA wrappers and the live gpuR WMDATA2 worker/observer/zero wrapper.
  Frozen E3 source manifest SHA b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193.
  The observer records genuine t300 IDs/history and observable trajectory tapes;
  native requires no policy execution or controller memory. Native is an eligible
  C1CLOSED arm and permits a clean RNG-only branch without policy confounding.
- No SUMO launched; no budget charge. Worktree clean initially, branch c4/c1floor.
  No simulator assignment/truth inputs or prohibited seed data opened.
