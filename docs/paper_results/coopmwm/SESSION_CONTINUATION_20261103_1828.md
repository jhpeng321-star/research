# S-COOPMWM live continuation — 2026-11-03 18:28 UTC+8

## Authority and ownership
- Read the full brief through line872 / MAIN18:00b. V2V is FIX-FIRST and HELD;
  freeing disk must not restart the superseded resource watcher. MAIN's hold of
  PID4125157 is adopted; this session has not relaunched it.
- Current global registered cap17,471,840sim-s (67320dae); all3 common-u0 V2V
  arms remain funded. Planned-command intent/fix-gate budget must be re-quoted
  within the17.5M ceiling before the new launcher runs.
- The concurrently thawed/resumed runtime owner is implementing live adapters,
  rolling storage and the qualified launch path. Its work is adopted, not duplicated.
  Full-pass implementation ee6a803f is retained; df=1 bank admission is superseded
  by MAIN17:55's per-vehicle paired-bootstrap rule and missing second-seed solos.

## Completed and retained
- Fixed-u8 public result `INTERIM_U8_RESULT.json` SHA
  f4f357e377fe5c5385460af42ef3fb717dcb2760f9233f91e7d9e95f21c74ef2.
  Full-panel J900 B-A/B-MAPPO/B-R CIs all include zero; B-native saves2.408%,
  Holm p0.004400. Focus comparisons remain unresolved. No collision/teleport
  differences; no training/checkpoint change conditioned on the interim.
- F5c9 archive is complete:7,391files /5,048,777,074bytes, all destination/source
  SHA-verified. Partial-resume receipts preserve V1/V2 faults and V3 completion.
  The independent529,080,134-byte interim-tape move is adopted and not repeated.
- Durable rolling mover4116186 is adopted; source and per-batch proofs retained.
  Latest maintenance rules:15min cadence,age>30min,target8GiB,gpuR floor4GiB.
- Scenario cleanup is complete under the reference-audited MAIN17:52 scope.
  This session's earlier delete invocation stopped locally on the source-identity
  guard before any RPC; the runtime owner completed/recovered the authoritative
  deletion receipt. No second deletion was dispatched here.

## Fresh observations
- At18:27:58 cpuQfree6,875,627,520bytes (~6.40GiB), original B-u18/R-u17.
  This exceeds the old6GiB resource threshold but NOT the new10GiB cleanup target;
  V2V must remain held until the intent/fix gate passes regardless of free disk.
- Current-policy collection is live: accepted B/R-bank decisions at17:11:38 /
  17:18:45 use policy14. Four active B searches use policy13/14; queued B
  snapshots15/16 and R centre-only snapshots15 are newer. All4 GPU workers
  were alive/8 gates/no failures at17:40:37–52,utilisation96%/93%.
  Old LS labels are intentionally drain-only; their last timestamp is not the
  current-bank progress clock. Do not kill busy searches to refresh that directory.
- `CURRENT_COLLECTION_FLOW_AUDIT_1793698612756375100.json` corrects the earlier
  reporting-only group-size field: actual groups2..6,not8 descriptor keys.

## Next actions, latest order
1. Continue authorised reference-checked storage recovery to>=10GiB; keep every
   bank/grade/analysis/checkpoint dependency. e3final is LIVE and must stay.
2. Register/run missing second-seed solo branches for cooperative candidates,
   especially986066/986048/986003; preserve old receipts and report BANK_STAT_V2.
   Registration for10 candidates has now landed as3c27d5a3: ADOPT it rather than
   registering a second batch; inspect its durable dispatch before any run.
3. Adopt reviewed READY branches (INTENT_INT_READY, BANKFILT_READY,
   GATESUMO_READY, LAUNCH3_READY), qualify Linux and run the real-bank/exact gate.
4. Report COOPFIX_GATE.json. Launch only the new intent/fix-bound launcher on PASS;
   obey23:00 fallback registration and never mix versions inside an arm.
5. Keep original B/R running until the fixed version is ready, then pause only at
   the registered update boundary; retain checkpoints and all existing owners.

No forbidden data path, driver/vType truth content, new environment, push, reset,
name-based process kill or unapproved source deletion was introduced by this session.
