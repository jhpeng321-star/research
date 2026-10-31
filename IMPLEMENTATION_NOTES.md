# S-C1PIDIAG implementation receipts

- Baseline CPU smoke: all four strict checkpoint loads and 120 s forecasts pass.
  First PI 13E 60 s median 18.506561 m vs archived refresh 18.506622 m.
- Initial 6-cell component preflight caught four adapter errors before dispatch:
  disabling `lane_learnable` requires disabling its dependent learned-lane
  secure-gap switch; deleting the pivot net breaks courtesy intent features.
  No main grid was launched on this implementation. All failures remain in
  remote `units/` and `evidence/`; source snapshots are retained by hash.
- v2 MOBIL intervention retains the frozen pivot net for courtesy features but
  bypasses its application to lane decisions, replaces learned lane heads with
  original deterministic MOBIL logits, and disables `lane_change_secure_gap`
  together with `lane_learnable`. Longitudinal physics and courtesy features
  remain; this is a lane-pipeline intervention, not a trained replacement model.
- v2 uses distinct write-once `units_v2/` and `evidence_v2/`. All tests/ablations
  are CPU-only on gpuA at nice19; no peer processes are signalled.
- Local pack-index recovery: 0/70 fixed starts admit 60 s of pre-t0 pack history;
  0/140 named original shard/run directories exist. Input audit finds no source
  queue or demand channels, so autonomous-entry+mix cells are unavailable.
  This is an availability result for these bindings, not a global deletion claim.
- Pack-clock byte check independently aligns all 73 modelled PI snapshot
  kinematics to pack offset 1, and all 94 wall snapshot kinematics to offset 10;
  offset 0 is empty in both examples. The nominal-origin conclusion is not an
  assumption that the pack retained a 300 s warm-up. Clock-check v1/v2 retained.
- Source inspection confirms broad A0 removes all learned longitudinal
  corrections, including courtesy/yield/adaptation, not only the residual net.
  Separately registered built-in single-switch tests isolate those mechanisms.
- The first queued auxiliary watcher (PID 216929, matching recorded start ticks,
  no children, no auxiliary DISPATCH) was stopped by its exact PID before any
  auxiliary inference. Builder review added causal backward velocity differences
  for observed acceleration and full-state junction carry (matching scorer
  observability). Source snapshot and stop receipt retained; AUXLAUNCH3 is the
  corrected queued run. No peer PID was signalled.
- All 1,930 v2 teacher-forced cells completed without error. Conditional feedback
  initially failed 195/350 cells because the frozen runtime's default PI source
  lookup has no simulation-layer source tree. The 155 successful wall forecasts
  and all 195 failed attempts remain intact. Recovery explicitly binds surviving
  `speedhead20261030/e3final/runtime` controller classes, the same dependency API
  used by C1CLOSED, and reruns only the 350 feedback cells in a separate folder.
  Auxiliary gating then checks the successful TF + recovered-feedback receipts.
  Its prior assertion failure produced zero auxiliary forecasts. Exact-PID-only
  stop/requeue of the waiting extra-switch parent preserves the 24-worker cap.
- MOBIL is an explicitly bundled lane-pipeline intervention: removing lane-v5
  also removes its merge-probability cues from residual conditioning. The pivot
  net remains for courtesy-intent features. No path-specific "lanes alone"
  causal claim is made. Single built-in longitudinal switches remain separate.
- Current original-panel snapshot cells give every model a one-tick history;
  archived refresh's adapter did so only for long-history models. Therefore the
  legacy source control preserves the *archived replay contract*, and comparison
  to the current panel measures composite source/protocol sensitivity, not an
  isolated torch-version effect. Missingness agrees in all 420 archived checks;
  numerical parity does not. Current diagnostics must not be presented as an
  exact reproduction of the full historical published panel.
- The full-history late-start feedback supplement uses the same external code
  with a freshly reset PI previous-command state. PIExternalU.reset initializes
  that state from measured ego speed; normalized tapes do not bind the prior
  commanded speed / writer acknowledgement. Therefore this supplement is a
  conditional-reset stress test, **not a validated continuation of the recorded
  PI executor**. Its large 120 s errors are preserved but must not be attributed
  solely to WM dynamics. The full-history TF contrasts remain valid because
  they use the recorded CAV future; a fair continuation must bind causal executor
  state as well as observed driver history. The original-clock feedback checks
  retain the explicitly scoped refresh common-plant/reset contract.
