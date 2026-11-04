# Resume adoption — November 4, 09:50 +08

The complete 1,125-line brief was read before live operations. Git history and
the existing handoff/results/design were checked; existing changes are retained.
Later addenda govern: the registered cap is 17,471,840 sim-s, and the active
experiment is V2V cooperative-only B-REPLAY. Historical warm-memory qualification,
restarts and permanent legacy stop are complete and are not dispatched again.

## Direct observations

- At unix 1793756975.78, the current root had **A8/B3/R3**. A's owned runner
  2049830 was live. B/R remain at the recorded mechanics pause, with release,
  statistical pass-1 refinement, kl_targeted continuation and practice-starts
  assigned to MAIN's separate operator.
- The deployed runtime SHA remains
  `4a8890cd1a7e99c3c11944b0a5bb4dd084dc7308bc0c0eb0d3adc934785f58df`.
  Monitor3 is live and bound to that exact runtime. The operator's branch has
  newer commits; neither an unfinished release nor a competing monitor is adopted.
- cpuQ free space was 6,341,668,864 bytes (5.91 GiB), below the 6 GiB launch
  preflight by 100,782,080 bytes. MemAvailable was 16,213,340 kB. The registered
  resource guards are preserved.
- Both archive workers remain live. The V2V extension still has 1,035,251,575
  SHA-certified bytes moved and zero currently eligible files. Its latest scan
  succeeded after a recorded transport timeout; sources were retained on failure.
  A-ahead RESULT payloads are future same-update B/R safety inputs and stay protected.
- Existing B/R collection drain processes 900006/900007 and backfill1132856
  remain live. Backfill stays disk-paused. Current-V2V collection wiring remains
  the previously reported blocker; restarting an exhausted old queue is not a fix.

## Evidence and next boundary

`RESUME_LIVE_0815_1793756977072266000.json` and
`V2V_ARCHIVE_SCOPE_0712_V2_STATUS_1793757003788035100.json` are direct public
receipt/process probes. No new SUMO, inference, charges, process signals,
deletion, safety release or duplicate dispatch was performed by this adoption.

Next: notify the existing recovery owner about the resource shortfall and retained
monitor; adopt its qualified release only after it publishes source/runtime and
Linux receipts. Verify B/R u4 progress and the actual B-u6 lesson/isolated synergy.
