# RBEVAL bootstrap uncertainty correction — 2026-11-01 resume

The original frozen WM2 panel completed all160 tasks and its original report is
preserved at `/root/rbeval20261031/reports/wm2_complete_1793478613086371110`.
`RBEVAL_WM2_COMPLETE_VERIFIED.json` verifies integrity, input identity and detector
decisions. Integrity verification is not validation of the original statistical CIs.

## Retracted uncertainty

- Earlier partial-progress bootstrap CIs included unscored same-cell worlds in
  resampling. They can spuriously vary the contribution of a scored singleton.
- The original full-panel bootstrap could report nonzero standard deviations
  from reduction roundoff on constant draws, producing near-zero-width CIs.
  Its bootstrap CIs must not be used as evidence of superiority/sign/ranking.
- Singleton strata cannot acquire missing world uncertainty from finite-K noise.
  Wilson intervals and factual/predicted point estimates are unaffected.

## Corrected implementation

`score.py` revision `scored-only-singleton-roundoff-v2`:

1. Bootstrap strata include only complete scored worlds. The full development
   pool still supplies climatology; evaluation labels never enter that pool.
2. At least one contributing stratum must contain two finite scored worlds.
3. Bootstrap draws must vary beyond floating-point reduction noise. Constant
   or singleton-only fixed strata have unavailable CIs, never tiny CIs or PASS.
4. Preserve10,000 draws, seed20261031, fixed demand×penetration×split composition,
   whole-world pairing, finite-K event resampling and the joint max-stat family.

Full20-world bootstrap weights are exactly unchanged; this fixes estimability
and roundoff checks. Regression tests cover two singleton cells, the complete
development12 singleton composition at realistic delay magnitudes, K-only noise,
genuine two-world strata and chunked-family equivalence.

New scoring is CPU-only from the same saved forecasts, into a NEW directory.
Do not edit a live/frozen candidate scorer or runtime. Any original frozen result
must keep its identity and be accompanied by this retraction; candidates can be
rescored after completion using this source in a separate directory. No inference,
new model selection, detector change or evaluation-panel change is authorized here.

## Corrected complete baseline published

- Source63cf5d83, new local output
  `tools/rbeval/evidence/reports/wm2_corrected_63cf5d83/`; no GPU or inference.
- `RBEVAL_WM2_CORRECTED_VERIFIED.json` binds the corrected scorer and all report
  hashes to the same160 original task/identity hashes and20 complete K8 worlds.
- 70,488 CSV rows checked: point data, Wilson intervals, randomized ranks and
  detector outcomes equivalent; maximum cross-platform numerical delta
  1.0913936421275139e-11 (comparison tolerance abs1e-8/rel1e-12).
- Development12 singleton CIs unavailable; corrected gate remains NOT_QUALIFIED.
  Original15-file report and original integrity manifest retained separately.
- The remote original-report standby remains bounded through11:30+08; existing
  candidate owners supervise their panels. No duplicate watcher was started.
