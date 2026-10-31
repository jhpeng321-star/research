# S-WMVALUE — INCOMPLETE (2026-10-31)

**The decisive offline test is blocked before real-data fitting; no PASS/FAIL claim.**
The OWNER's WM-centric route remains the target. Design was committed as
`12685912` before fitting; guarded CPU components/tests as `44017297`.
Shared REGISTRATION.md contains the pre-registration and verdict entries.

## Registered comparison (remaining team delay, veh-s, endpoint absolute t900)

| Method | Sign accuracy | Spearman | MAE (veh-s) | World-cluster CI |
| --- | --- | --- | --- | --- |
| A: WM2 b8797242 + mix to endpoint | NA | NA | NA | NA |
| B: WM 60 s + real-trained V | NA | NA | NA | NA |
| C: WM 90 s + real-trained V | NA | NA | NA | NA |
| D: no-WM V + current command summary | NA | NA | NA | NA |
| E: FIT arm/time means | NA | NA | NA | NA |

NA means **not evaluated**, not zero. No reserved-world outcomes were opened.
Registered gate: B or C must beat each A,D,E on sign AND MAE, with positive
paired-difference 95% lower bounds (10,000 whole-world bootstrap draws).

## Data gate evidence

Audited 160 WMDATA FIT episodes (969400..969431, five arms) and 300 WMDATA2
FIT episodes (969550..969599, six arms). All 460 raw archives exist, cover
absolute t0..900 at 0.5 s, and have objective_600.json. Trimmed views stop at600.
**0/460** carry entry_backlog, request_counter, or sensor_mask in episode.npz;
**0/460** have observations.npz or observation_contract.json sidecars.
traj_base.npz has only a/lane/meta/s/t/v/vid, with no source queue telemetry.
The frozen observer serializes queue history only through t300 in start.pt;
subsequent real objective clocks are only at control120/300/600 (not every t0).
Thus raw road tapes alone do not supply the required per-time observed source
state or the complete remaining team-delay labels on the registered grid.
An existing CAV/RSU projection helper exists; missing raw sensor masks alone
does NOT prove vehicle visibility cannot be reconstructed offline.

## Delivery and recommendation

`tools/wmvalue`: fixed HGB V, stronger command-only D, FIT arm means, K8
J_H+V composition, all400 pairs, sign/Spearman/MAE, paired world CIs and gate.
**14 CPU tests pass; ruff passes.** Tests include leakage/missing-data guards,
absolute900 clock, unobserved-truth invariance, and failure to beat D despite
beating A/E. Synthetic unit-test fits are not recorded-data scientific fits.
`tools/wmvalue/evidence/status.json` binds design and full schema receipts.
The frozen WM runtime adapter and real-to-observable exporter remain unbuilt.

**Recommendation:** recover existing per-time source/request telemetry if it
exists elsewhere, or pre-register an audited causal observation reconstruction
amendment before fitting (no new SUMO needed to investigate). The existing
prepare.py generates departure schedules from seed/demand; that is not silently
accepted here as recorded source sensing. Bind the existing CAV/RSU projection,
prove prefix-only features, reconcile source/road integrals to real clocks, then
run the frozen A–E test. Keep reserved worlds closed until the FIT gate passes.
No policy-training recommendation or GPU-hour estimate is justified by this
INCOMPLETE result. Both checked GPUs were busy; no inference was launched.
No SUMO, real-data fitting, reserved/evaluation outcomes, policy run or push.
