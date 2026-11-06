# S-LAUNCH W4 recovery — 2026-11-06

## Adopted state (cpuV audit 2026-11-06 11:38:45 +08)

- Existing branch `c4/clean3-launch`, initial HEAD `8e074be3`; preserve the pre-existing untracked `ops/clean3_inventory_admission.py`.
- Read full `tracks_spec.md`, REGISTRATION lines 7488–7787, git status/history and current chain authority before action. Earlier setup, integration, Linux qualification and failed-root evidence are already complete.
- Sole r12 chain is **alive**: PID `560611`, start ticks `12554274`, exact source `3e5bc9f9a41d7e34cf605ad257f340daba7525d0`, control `/home/ubuntu/cpuv_setup_20261105/clean3_chain_3e5bc9f9`.
- Chain actually started **11:28:14 +08**; the authority's `created` is an earlier preparation time. State at the audit: `PREPARE/RUNNING`; root `/home/ubuntu/coopmwm20261106_clean3_r6` has an empty ledger, no final CONFIG, smoke/gate/launch receipt, STOP or FAULT yet.
- Independently SHA-verified all **735** deployed source files. Exact-source Linux qualification return code 0, 209.2664 seconds; no repeated qualification.
- G1 completed `NO-GO`: zero executable joint labels, jobs and charged simulation. Existing registered rule continues regardless. Five verified B wall examples admitted (three distinct moves), native source SHA `7a225dbd59436392c29fd3cad6e9df3c2deb01af976289bc787e9865c0f1471b`.

## Current budget authority

REGISTRATION owner 03:00 `approve` raised clean3-scope cap to **7,300,000**, released unspent plan_fix reservation; separate 250,000 coop_probe yields overall 7,550,000. REGISTRATION 11:13 prospectively accounts completed G1 at actual spend, once. Independent preserved-root ledger audit confirms 22,800 prior failed-clean3 sim-s.

Current bound: `3,098,640 + 22,800 + 500,000 + (3,680,400 - 230,400 + 0) = 7,071,440 <= 7,300,000`; spare 228,560. All other full u1–u24 phase allocations and >=15% margins remain required. No mining/probe reservation reassigned. Final config/PHASE_COVERAGE and exact smoke/fix/lever/reporting/gate bindings still must pass before training launch.

## Ownership / next action

Existing Claude launch operator retains execution/fix/dispatch ownership per current registered handoff. W4 independently reads public receipts over shared accepted F6 transport; do not launch a second chain or replay any finished step. Monitor r12 through smoke, gate and phase-coverage admission; register actual LAUNCHED identity when available. A retains full safety rule; B/R composite reported, any collision/teleport pauses; u24 group-over-best-single criterion unchanged.

Audit `CLEAN3_RESUME_W4_READONLY_01.json`; operator `ops/clean3_resume_w4_remote.py`. This continuation adds no physics, process signal or dispatch.
