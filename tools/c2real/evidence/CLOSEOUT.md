# S-C2REAL-1 close-out — 2026-10-27

Status: analysis complete on all existing main-loop paired data; primary 60 s WM fidelity remains unidentified by collected predictions. Append-only close-out recorded in `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md`. Main coordinator owns master-ledger integration.

## Recovery and scope

The prior analysis had finished (own PID 116550, start_ticks 847075860). Reused it without a duplicate dispatch. Zero new SUMO, training, GPU work, evaluation seeds or simulation charges. GPUA CPU numerical analysis and verification: nice 19, one numerical thread, CUDA disabled. No signals sent to any process.

Coverage: cycles 0/1/2 have 4/13/13 collected worlds, two starts per world, 60 unique pairs and 120 tapes. Nine registered cycle-0 worlds are uncollected; c00 conversion retries duplicate both arms. Thirteen failed cycle-1 attempts contain no paired outcomes. Coverage receipt binds all examined journals, conversion receipts/manifests and credit/gate records.

## Verdicts

- **Cycle 2 better than cycle 1 in real SUMO: inconclusive.** Policy-minus-native J300 difference −315.543 [−1320.852,+718.175] veh-s/start. Independently resampled disjoint cycle worlds; within-start CRN policy/native pairing retained.
- **WM over-promises: supported for cycle 2 at 30 s, unidentified at 60 s.** Collected 60 s prediction bias/CI unavailable. Equal-world same-start 30 s bias: c1 +0.699 [−4.239,+6.267], c2 +4.635 [+0.739,+8.506] veh-s. Increased optimism +3.936 [−2.930,+10.375] is inconclusive. Eligibility-selected 300 s diagnostics do not substitute for 60 s fidelity. No exploitation conclusion.
- **Metering increased: inconclusive.** Cycle-2-minus-cycle-1 policy/native rate-effect +0.186919 [−0.320855,+0.682711] events/CAV-hour. Frozen COOPMEAS 9119cd77, unpaired world bootstrap across cycles.

All intervals are 5,000-draw whole-world percentile 95% CIs, seed 20261027. CSV contains per-start, per-world, per-cycle, pooled, CAV-share/demand, WM bias/correlation and behaviour counts/exposures/rates/CIs. Original guard scales sum two starts; primary report averages them.

## Verification and delivery

`python tools/c2real/operate.py verify`: PASS on gpuA, 772 analysis hashes plus 95 coverage hashes; accounting/pairing/clocks, all three guards, real and behaviour world-bootstrap CIs, same-start prediction residuals/correlations, report word count and rounded real CIs. `delivery`: PASS, both main and branch MD/CSV copies match numerical receipt hashes. Report is 313 whitespace-delimited words.

- Report SHA256: `dfd7eaeff024b42c7e74727930db8259ac1a9576bdbf7e50be9a1ac971167aea`
- CSV SHA256: `6bdfe731cc49afc57268722cd8440729d5a87f01a31578897793820209a65a5b`
- Deliverables: `E:/research-dense/docs/paper_results/C2REAL.md` and `C2REAL.csv`, plus tracked copies under `docs/paper_results/` on `c4/restart`.
- Recovery/coverage commits: `82591868`, `1d8d2e3b`. Final report committed separately; no push.
