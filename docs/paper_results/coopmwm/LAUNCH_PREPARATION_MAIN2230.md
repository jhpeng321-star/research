# MAIN22:30 launch preparation (no activation)

Full brief read through986. All audit-fix files remain owned by the final
integrator and bank-perseed builder. R's lag-1 cap stays as explicitly decided
by MAIN. No training, builder handover, gate or GPU release was dispatched.

## Source and qualification

- Final branch is currently `c4/launch-final` at `f6288c07`, with active dirty
  integration work. `LAUNCH_FINAL_READY.json` is absent. Preserve the WIP and
  merge that single branch only after READY; do not reconstruct its changes.
- Our new commits touch qualification/reporting only. Preserve `d00ee69f`
  (final dependency overlay) and `fddd2a8b` (legacy-hold status reporting) during
  the future merge. No new learner changes were made by this continuation.
- Existing I1 qualification is complete and must not be repeated. For the final
  committed integration, inspect whether I2 already has STAGED/DISPATCH/RESULT
  receipts before using it. Use another generation if it belongs to a different
  pin. Run the existing `intent_linux_ops` with `--final-release` and the full
  merged HEAD so per-seed/all8 code and both suites are explicitly staged.
- The available independent audit still has blockers against `f3caf85f`.
  Qualification may precede the re-audit; the real-bank gate and launch may not.

## Builder stop plan

- Identity-checked old arm/grader owners are absent: B411152, R411153,
  B-grader3305068 and R-grader3305069. Their resource hold is intentional and
  remains active; do not restart them to supply the new bank builder.
- Fresh cpuQ process inspection found no running `tools.coopmwm.bank_builder`
  or `tools.coopmwm.replay_start` command. Re-check immediately before takeover.
  The original collector3101232 and broker3105652 remain live and are retained.
- If a builder is present at takeover, first match its boot ID, start ticks,
  argv and control-root owner receipt. Write a registered, additive STOP.json
  in THAT control root; let its in-flight complete branch set finish. Confirm
  exit and lock release before starting the replacement. Do not use the old
  `bank_builder_ops quiesce` action: it targets the historical0406 owner.
- Use a NEW qualified builder control/binding with
  `verify_mode='all8_both_seeds'`, shared accounting and the final source hashes.
  Preserve completed first-seed lists and their candidate specs/tapes. The
  per-seed reader must consume each winning example's own plan and seed tape.
  Do not delete, replay or silently reinterpret old failed claims.

## R-count preflight after final qualification

Run from the **qualified final source**, in a fresh process (no installed old
bank adapter), with its qualified PYTHONPATH. The new preview CLI was inspected
on the builder branch; verify the final merged help before use:

```text
nice -n 19 /home/ubuntu/miniconda3/bin/python -B -m tools.coopmwm.bank_statistics preview --root /home/ubuntu/coopmwm20261102_warm1 --arm R --rule per_seed_v1 --out <new-control>/R_PERSEED_PREFLIGHT.json
```

Run B separately with the same rule plus
`--source /home/ubuntu/coopmwm20261102_lscollect1`. Outputs are immutable, so use
new paths and adopt existing results on reconnect. Report `examples_admitted`,
`admitted_by_seed`, `cooperative_examples`, `data_missing_examples`, policy
versions and tape identity evidence for BOTH arms. A missing-tape error is a
blocker to diagnose, not a zero-entry result. Keep the registered R fairness cap.

## Resources and budget

- cpuQ at the latest resource check: disk6,085,410,816bytes (356,040,128bytes
  short of6GiB), MemAvailable22,712,332,288bytes. MAIN's cleanup agent owns
  further storage work; no parallel cleanup was performed.
- Global charged10,202,670 / registered17,471,840sim-s. No V2V phase is spent.
  Latest phase snapshot: `bank_verification_ext`137,180remaining;
  `sumo_verification`1,045,800remaining; restricted `sumo_verification_k8`
  1,500,000remaining; training201,600remaining (legacy7updates).
- Re-quote the ACTUAL bank slate with the final all8 code, including complete
  seed-2 lists, extended windows and per-seed solos. Register any allowed phase
  transfer before builder use; do not treat other phases as implicitly fungible.
  The total ceiling and every existing charge remain intact.
- Launch still requires fresh exact-commit clean audit, Linux PASS, own-seed
  offline/exact cooperation gate PASS, registered budgets/runtime, disk+memory,
  and reviewed cooperation search `--alternate` on both GPU hosts. No fallback.
