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
