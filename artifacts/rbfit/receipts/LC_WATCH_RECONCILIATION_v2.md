# Owned watcher reconciliation (~02:20, 2026-11-01)

- First watcher34656 successfully verified/downloaded/committed boundary v4
  at6588e69b. Artifact paths and hashes were correct; a loop-variable shadow
  mislabeled the notification/report/commit message as lane_events.json.
  Corrected display metadata in a NEW commit; no history rewrite or refit.
- Windows GBK subprocess capture also logged a Unicode decoding error; explicit
  UTF-8 capture added. No physical parameter/artifact contents changed.
- Verified local PID34656 via Win32_Process creation02:10:27 and exact own
  watch_lc.py command, then stopped ONLY that watcher PID. No CPU worker or peer
  process was signalled. Original watcher receipt/log/state preserved.
- Seven watcher tests PASS: successful publication identity, UTF-8 capture,
  OS singleton/release, duplicate raw/watcher guards, preservation of staged
  work and edited handoffs. Publication defers while an unfamiliar index is busy.
- Resume uses the existing state/claims and a NEW session receipt. Terminal
  remote jobs are never retried; load admission before a claim remains deferrable.
- Normal window09:00, new starts through08:40, existing verified work may drain
  to09:25 ahead of the owner's09:30 freeze. Future source fits may use4CPU workers
  only below load16/32 CPUs; the currently running2-worker617309 is unchanged.
