# Concurrent overlap and logging integration

- Immutable source `/home/ubuntu/coopmwm20261102_overlapcpu2`: 195 CPU tests
  PASS, zero skips, 31.309 s, zero SUMO. Prior overlapcpu1 failure retained:
  two independent-oracle tests suppressed scripted RNG calls after the
  counterfactual group geometry diverged. The oracle now consumes the original
  scheduled calls regardless of counterfactual eligibility; production unchanged.
- MAIN0249 recheck rule: ALL active plans interact. One additional K8 batch at
  the new plan's H_final compares keep-all+new, keep-old/drop-new, and one option
  ending each old plan. One shared bootstrap selects minimum whole-road delay.
  Old plans truncate only when the new plan is actually executed after beta.
  Log old plan start/executed duration/predicted truncation gain. R keeps all.
- Per-action remaining commitments permit old-plan truncation counterfactuals.
  Source-bound GPU gate additionally checks background expiry and explicit empty
  background overrides at K4/H300 and K8/H420, full-versus-fast exact parity.
- Proposer RecordingEvaluator logs full screened pools; no proposer seeds active.
  Existing search remains byte-identical without optional seeds (CPU regression).
- Exact grader follow-up (marginal other-plan replay/frozen temperature/buffer)
  and runtime wiring remain pending. This CPU receipt does NOT authorize B/R.
