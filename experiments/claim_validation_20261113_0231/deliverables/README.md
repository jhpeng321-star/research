# Deliverables: claim validation 20261113_0231 (Amendment 72) - for the professor

Assembled 2026-11-13 from existing records only (zero simulation, no new statistics, no existing file modified).
`REG` = `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md` (line numbers as of this assembly).
Both registered primaries are NOT SUPPORTED (E1 admitted count 20 vs 20, p 1.0; E2 F - H20 +686.9 [-205.5, +1,656.5]
veh-s, p 0.142; Holm-adjusted 1.0 and 0.284). Post hoc items are reported separately in CLAIM_EVIDENCE_REPORT.md section 2.

## Inventory of the requested deliverables

| Requested | Status | Where |
|---|---|---|
| AUDIT_AND_REUSE.md | COMPLETE (follow-up) | `AUDIT_AND_REUSE.md` (audit, reuse vs new, seed disjointness, unverified items, conflicts C1-C7) |
| PROTOCOL.json / PROTOCOL.md | EXISTS (frozen) | `../PROTOCOL.json` 537bc8fa..., `../PROTOCOL.md` 6c579017... (match the shas registered in A72) |
| MANIFESTS/ | COMPLETE | `MANIFESTS/SHA256SUMS.txt` (every cited file) |
| RAW_RESULTS/ | COMPLETE as an index | `RAW_RESULTS/INDEX.md` (files left in place); FROZEN_BANKS + ADMISSION extracted from tgz |
| E1_SUMMARY.csv | COMPLETE | `E1_SUMMARY.csv` (built by `scripts/build_summaries.py`) |
| E2_SUMMARY.csv | COMPLETE except replay | `E2_SUMMARY.csv`; secondaries + Holm filled from the finished `../e2_secondary/E2_SECONDARY_REPORT.md` |
| BUDGET.csv | COMPLETE (some charges "not on laptop") | `BUDGET.csv` |
| figures | COMPLETE | `figures/fig_e1_admitted_counts.png`, `fig_e1_bank_contrast.png`, `fig_e2_arm_differences.png`; script `figures/make_figures.py` |
| CLAIM_EVIDENCE_REPORT.md | COMPLETE | `CLAIM_EVIDENCE_REPORT.md` |

## PENDING / not run, and why

- E2 same-state timing replay (registered secondary): NOT RUN, 0 sim-s; prepared only (dry run). Needs an allocation.
- WM-ACT sensitivity: verification running on Box 3 since 12:42:07; no result in the record; confirmation not run.
- E1-FIX (Amendment 74): GPU scoring running on gpuR since 13:21 (`REG:11682`); no result, no SUMO.
- A75: intent only.
- Box-side charges for A72 stages and the A69 Box 4 stages are only in REGISTRATION.md, not as CHARGES files on the laptop.

## Record flags (no number conflicts found)

1. `e1_conf/E1_ANALYSIS.json` marks `by_budget.6.primary = true` but its `role` says secondary (delay); the registered
   E1 primary is the admitted count in FROZEN_BANKS.json.
2. `BUILD_NOTES.md` says never-activated descriptors rank last; the scorer as run scored them 0.0 (the disclosed 06:27
   ranking artefact).
3. E1 confirmation stage charge is not stated in the record; 1,267,200 is derived (Box 4 cumulative 3,004,800 - 1,737,600)
   and agrees with 11 banks x 96 x 1,200.
4. A72 mechanics cap 9,600 vs 19,200 identity charged on two boxes (addendum P1 allows "up to 19,200 ... inside the total cap").
5. The record does not say whether WM-ACT budgets (748,800 + <= 86,400 + <= 691,200) count against the A72 cap of
   4,387,200 (4,137,600 already charged).
6. `REG:11665`: the 13:02 A73 admit allocation note has the wrong clock (box log authoritative), as recorded.
7. `REG:11673` clock correction: header times 13:20-14:50 on 11-13 were estimates ahead of real time (all written before 13:17).
8. E2_SECONDARY_REPORT notes BUILD_NOTES `--block e2r` should be `--block dev` for the replay.

## Rebuild

    C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I scripts/build_summaries.py
    C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I figures/make_figures.py
