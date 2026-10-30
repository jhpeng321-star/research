# S-WMREQUAL2 — MAIN 2026-10-30 22:55

Base `de5b627a`, branch `c4/wmrequal2`; existing validation rows first.
No training, SUMO, push, deletions, or peer process operations. Read only named
permitted inputs; no excluded paths, driver truth, or vehicle assignment files.

1. Bind the 36-world/three-arm validation runtime and identify bridge ownership
   per original/entrant cohort. Commit the audit and provenance.
2. Reconstruct all K8 contrasts from the 36 retained validation batches, reproduce
   the prior summary, and compute per-arm and unaffected-subset statistics at
   60/90/120/180/300 s. Retain additive common-median decomposition, gain/harm
   counts, zero-real denominators, world bootstrap CIs and qualification limits.
3. Assess ownership repair applicability before compute: only affected arms need
   rescoring; prefer CPU, otherwise gpuR/gpuA only within owner's resource limits.
   A different timeline/ownership mismatch must not be mislabeled as the audited
   policy_ramp_merge bug or as measured learned-model error.
4. Deliver `docs/paper_results/WMREQUAL2.md` in <=60 lines with audited evidence
   SHA; commit the report. Its delivery SHA is obtainable from path-specific log.

Named external input: `E:/agent-work/c4-distill/docs/paper_results/SETUPAUDIT2.md`.
Remote provenance inputs are exact paths already named by the retained scorer:
`/root/autodl-tmp/c4/wmrequal20261030/{SOURCE.json,src/wms/...}` and
`/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json`.
