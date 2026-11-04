# Public accounting reconciliation — Nov 4, 14:38 UTC+8

The read-only progress reporter previously used the shared bank binding alone.
That binding includes both registered bank transfers, but the practice reference
transfer is registered in the V2V runtime: 345,600 sim-s from `paired_branch` to
`v2v_train` (amendment `85c5a9bbd01d3d527180e9ff5bd4a0c03470ea65f46e90ab27aa585a95c2a8dc`).
Using only the bank binding understated training availability and overstated
paired-branch availability by that amount. The total charged was unaffected.

`tools/coopmwm/public_budget.py` now reconciles the shared budget with that exact
registered runtime-only transfer. It checks the registration and endpoint values,
refuses unknown/conflicting transfers, avoids applying an already shared transfer
twice, and retains newer shared bank funding. It does not mutate either input.
The local `_sol/coopmwm_public_progress.py` observer injects this pure function
into its read-only probe; no file or code was deployed to a live source root.

Verification:
- `python -m unittest tools.coopmwm.test_public_budget_cpu -v`: **5/5 PASS**.
- Real cpuQ observer PASS: `PUBLIC_PROGRESS_1793774287773234600.json`.
  Registered transfer reconciled, frozen-runtime phase delta empty, all balances
  nonnegative, total charged **14,315,120 / 17,471,840 sim-s** including retained
  811,640 exactly once. No new SUMO, allocation, runtime, learner or ledger change.
- Archive extension recovered naturally and certified **4,342,622,308 bytes**;
  590,639,549 further eligible bytes at its latest scan. cpuQ free 5,278,867,456
  bytes (~4.92 GiB), still below the 6 GiB launch3 floor. MAIN14:35's separately
  owned SHA-gated cpuS moves are registered and running; no duplicate mover here.

B11 learning mechanics and composite safety pass: four full passes, KL0.043543,
8 cooperative examples, action MAE1.197105 -> 1.180883 m/s²; pass-1 loss-change
CI[-0.003953,0.002547], no significant harm. Critical-total B-A+18.1875
CI[-9.5625,45.7297], collisions/teleports zero on 24 pairs. Intent loss rose
slightly. R10 also passes. The completed u6 isolated check remains the current
cooperation evidence; it does not establish positive learned cooperation.
