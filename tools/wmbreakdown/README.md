# S-WMBREAKDOWN analysis contract (2026-10-31)

Analysis only: no simulation, training, or inference. Only explicit registered
seeds, observable converted arrays, recorded `traj_base.npz` (t/s/v/lane only),
clock JSONs, and canonical already-saved prediction JSONs are read. Remote
analysis is single-process, low priority, read-only; no process is signalled.

1. Extract the exact `breakdown` function from c4-entrant/tools/lrdiag/replay.py
   through Python AST, hash it, and execute that function alone (no model imports).
   Use full-traffic upstream/merge/ramp 10 s pooled vehicle-time mean speeds,
   60 s recorded prehistory, valid mapped lanes. Evaluate 120/300/600 s separately;
   dominant step selection is horizon-dependent, not a nested event process.
   No hand-chosen speed threshold. Pre-existing steps remain separate; final
   30 s cannot supply new onsets. BIC detection is a speed-loss proxy, not proof
   of a metastable phase transition.
2. Use the views' exact t0 state plus original post-t0 tape, excluding the endpoint,
   for 600 s (views only span 300 s). Match the raw trajectory and clock SHA256
   against view episode metadata. Compare post-t0 converted/raw bin means.
3. WMDATA 969400–969431 = development; 969432–969439 = reserved evaluation,
   reported separately, no label use in any baseline. WMDATA2 969550–969599 is
   a separate descriptive replication. These seeds had all been used by the
   earlier WMCALIB calibration analysis: the reserved split is clean for this
   analysis's baselines, not a historically untouched qualification panel.
4. Per-arm and demand×penetration empirical occurrence proportions, Wilson CIs,
   detected-onset summaries and paired shifts vs native/CTRL0 are descriptive
   across worlds, not repeated trials conditional on an identical observation.
   Compare 600 s positive delay gain (reference minus arm) with avoidance,
   newly induced events, both-event timing shifts and no-event pairs. Exclude
   pre-existing-step pairs from that classification; retain counts explicitly.
5. Canonical 320 K8 files are selected and SHA-verified by WMCALIB's manifest;
   no alternate-device duplicate pooling. Audit saved fields and scorer source.
   Never infer breakdown from J. No native WM arm exists in that contract.
6. Score absolute team delay with `costs.real_J` against recorded cumulative
   `team_delay_vs`; also report inherited training `J` vs objective_h.json and
   paired arm−CTRL0 distributions. Draws are coupled within each future.
   CRPS(F,y)=mean|X−y|−0.5 mean|X−X'| (empirical predictive distribution);
   fair-ensemble CRPS also retained for K=8 finite-sample sensitivity. Baselines:
   pooled empirical climatology, same-arm empirical climatology, same-arm point
   mean. Dev baselines leave the entire target world out; reserved baselines use
   only the 32 dev worlds. Also score same-cell/same-arm empirical climatology.
7. Report nine-bin rank PIT (r=#draws<real; ties averaged), min–max coverage
   (exchangeability expectation 7/9), central quantile coverage, within-K SD,
   cross-world real SD, mean error and CRPS. Across-world SD is not conditional
   noise. World-bootstrap paired CRPS differences account for arm dependence.

Reproduce: `python tools/wmbreakdown/collect.py`, then
`python tools/wmbreakdown/analyze.py`. Requires existing Python with numpy and
matplotlib; SSH uses existing host interpreters. Evidence is task-owned under
`docs/paper_results/wmbreakdown/`; no raw truth or controller binaries are copied.
