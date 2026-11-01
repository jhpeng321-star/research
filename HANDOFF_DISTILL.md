# S-DISTILL / MAIN + S-WMTEACH handoff

- Fresh shared live panel: `docs/paper_results/distill_live/PANEL.json`.
  SHA256(canonical JSON) `72776ff122956f021d1caf2db9622674b0cde2a74af8bd9f20c3de5ba0a383a6`.
  DEV984000..984023, two repeats/all12cells; high-share/high-demand first.
  Three-host safe-name/current-registration exposure audits PASS before dispatch.
- Original registered COOPPLAN720decisions are DEV never-fit/tuned; S-DISTILL
  preserved that role and used them only for frozen policy audit. Re-role needs MAIN.
- Student frozen from EXAM300states/60TRAINworlds: MLP/confidence0.9 selected
  by world-grouped CV; mean argmax plus CV confidence, not previous radius gate.
  Honest nested-CV saving−16.497[−49.493,+0.003]veh-s/state,2moves; no offline success.
- Portable policy `docs/paper_results/distill_live/POLICY.json` needs only numpy;
  `tools.distill.portable.Student.step(ObservableState)` uses same COOPPLAN options.
  All1020existing feature contexts match original sklearn predictions/actions.
- Live cap199200<=200000,24paired full1200s episodes, extra1200mechanics,
  complete H120 C_G reference/joint/two-solo reruns for EVERY executed group.
  Per-world credit5850sim-s means explicit budget pruning when next full quartet
  cannot fit; proposal/pruning logs preserved. No credit label chooses an action.
- gpuA immutable root `/root/autodl-tmp/distill20261101_v1`; four CPU worldworkers,
  at most four isolated creditworkers; nice19/load16/mem8GiB/disk2GiB gates.
- Status/collect: `py -3.13 -B -m tools.distill.ops status` / `collect`.
  Never relaunch/retry an existing dispatch or overwrite sources/policy/receipts.
- All training uses existing labels, zero newSUMO. Report `docs/paper_results/DISTILL.md`.
