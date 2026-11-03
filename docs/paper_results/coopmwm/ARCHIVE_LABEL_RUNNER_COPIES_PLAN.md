# MAIN14:50 candidate (a) — completed label-runner copies

Disk reclaimed by the interim-tape archive allowed B-u12 to finish and the
existing B-u13/R-u12 workers to start, but their new outputs again consume the
margin. Qualify a further narrow subset of approved `label_attempts` artifacts:

- Only TRAIN986000..986071 labels with successful final `accepted`/`rejected`
  verification receipts in SUMOENV1/SUMOPATH1. Failed/unresolved labels excluded.
- Archive each completed verifier's `SNAPSHOT.pt` input **copy** and its
  `<branch>.result.json` runner-return **copies**. These are consumed only inside
  `SubprocessBranchRunner.prepare/run` during the completed decision.
- Preserve every original collection snapshot, pending envelope, episode RESULT,
  label/verification/confirmation receipt, and every canonical branch
  `RESULT.json(.gz)` and `observable.jsonl.gz` on cpuQ. Bank reconstruction uses
  the collection inputs (`replay_bank._ls_snapshot/_decision_snapshot`); imitation
  reads the chosen canonical branch RESULT and, for V2V, its observable trace.
- Inventory: **878 files /1,220,993,026 bytes**. Target
  gpuR:`/root/cpuq_archive_20261103/ls_label_runner_copies`.
- Same SHA-gated maintenance core, compressed rsync, independent remote per-file
  verification, privileged FD/config audit and source rehash before any unlink.
  A detached local SSH relay keeps the existing forwarded agent available; no
  account/key/system changes. One-use claim and PID/create-time receipt prevent
  duplicate relays; remote owner has PID/boot/start identity.
- **Nine CPU safety tests pass** (0.048s), including successful-final selection,
  failed/eval/path-traversal exclusions and canonical-result preservation. The
  staged immutable sources have SHA receipts. No SUMO/inference/reservation.

This remains below the requested10GiB free target. Live banks/checkpoints/owners,
other source roots, and unsearched pending envelopes are not archive candidates.
