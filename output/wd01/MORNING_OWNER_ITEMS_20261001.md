# Morning owner items, 2026-10-01 (world model session)

1. **Part B v0.4a (WD-01-PB-V04A): verdict FAIL for every variant.** This is
   the ONE CAL read; the report is `REPORT_V04A.md`, merged into zcode/wd01.
   - Level accuracy A2 PASSES at τ 30/60/120. Headline strict MAE vs ridge:
     244.7 vs 286.9, 501.3 vs 584.2, 1039.2 vs 1192.5.
   - Contrast sign accuracy is high: damping−native 0.87–1.00 on
     |dY| > median pairs.
   - A3(ii) fails at every gating τ: the contrast VALUE distribution (dV
     CRPS) is worse than the zero-contrast and TRAIN-cell-mean baselines.
   - Strict A1 coverage is 0.880, outside the 0.75–0.85 band.
   - Same picture as v0.3.2: sign skill, no value skill. Owner: accept the
     FAIL. WM-N1 (Idea A) targets exactly this gap.
2. **Wall law v2.2: APPROVED and FROZEN** (owner ack via main, 2026-10-01). Kept for the record:
   - `REGISTRATION_WALL_V22.md`, LF-sha256
     `e7fd353616eea9ef2ab0440c751c8c9211f11a3f5de97f1b911b7a528f6f01e7`
     (zcode/wallv22 8da4eff). The owner spec is verbatim in §1.
   - Proposer D_min = 70·v_wall (10 s assembly + 60 s minimal hold); 109/131
     J4 requests would have been feasible (83 %).
   - Hold cap = min(H, (2644.2 − s_lead)/v_wall).
   - Patches, the smoke fixture (3 × 300 sim-s) and the J4B draft
     (105,600 sim-s) have gone to the recon.
   - No v2.2 run happens before the ack, including the smoke.
3. **J-PULSE: APPROVED WITH the 240 s option and FROZEN.**
   - LF-sha256 `a4d1c5f9…3621ea7` (zcode/wd01 36766cd).
   - 44 worlds × 6 arms = 158,400 sim-s. The spec has gone to the recon.
4. **FEAT1 size and time are larger than planned.**
   - The P9 estimate for v1 + Idea A channels is ~38 GB (not ~13 GB) and
     ~5 h at 6 workers.
   - The output goes to D:/rd-staging/wmn1_feat (201 GB free).
   - Starting ~05:00 means it finishes around 10:00, so the corpus may be
     ready AFTER 08:00. Member-1 training can start as soon as the pack
     verifies.
   - Fixes are in progress for J4 member_left_network runs that crashed the
     v1 featurizer (55 TRAIN runs; task f87dc1a0).
5. **Action representation (a-priori, schedules only).**
   - Of 4,952 TRAIN pairs, 96.1 % are whole-window. Only 5 are ≤ 60 s, 189
     are 60–240 s, and just 2 differ only in the first 60 s.
   - So short-commitment effects are barely represented until J-PULSE
     returns. This is disclosed in v0.4b and in the WM-N1 eval.
6. **Part B v0.4b registration: ACK NEEDED** (unless main acked it).
   - `REGISTRATION_PARTB_V04B.md`, LF-sha256 `e00cd98e…5ad9d` (zcode/pbv04b2
     7203ad0).
   - Owner questions are in §11. No fit happens before the ack.
7. **Gate G1 PASSES on real J4 TRAIN windows:** max position/speed error
   0.0 vs the Part A v2g rollout (zcode/revA3g 31232fd).
8. **J4B registration: ACK NEEDED** (unless main acked it).
   - `walls_design/merge_release/REGISTRATION_J4B.md`, LF `2463f5be…80b95`
     (zcode/wd01 eb3661f).
   - The 44 J4 worlds under v2.2, 105,600 sim-s. The launch is gated on the
     SMOKE_V22 PASS and the owner's AUTHORIZATION.
9. **FEAT1 first attempt aborted, then relaunched** (disclosure).
   - The shard keys collided: an audit found 980 of 2,476 runs colliding.
     The keys were policy-based (fixed1/fixed2, switching1–3), plus WM-72
     cross-batch overlaps (chunk2/3 native/pi vs truearms).
   - Nothing was overwritten. The partial tree `D:/rd-staging/wmn1_feat/shards`
     is kept.
   - The fix is zcode/feat1fix bf5d81a (0 collisions; 161 tests). It was
     relaunched into shards_v2/pack_v2 (GLM 4a9ca034), ETA ~10:45.
   - 13 of 2,476 runs are unreadable and will be listed with reasons.
   - Minor: the fix task's micro-smoke used w8410m2700p02t480, which turned
     out to be a chunk-1 CALIBRATION world rather than TRAIN. Its physics
     rows were featurized into an E: tmp dir; CAL runs are in the FEAT1
     scope anyway. No outcome analysis was done. Disclosed.
10. **FEAT1 throughput is lower than estimated.**
    - 72 shards done after ~15 min, about 290/h at 6 workers. So the 2,463
      runs take ~8.5 h (ETA ~14:00), not 5 h.
    - Options:
      - allow > 6 workers if RAM/commit allow (the owner cap is 6);
      - start member-1 training on a partial pack. That needs a pack of
        completed shards; possible, but disclose it.
11. **v2.2 smoke FAILED on run 1: OWNER DECISION NEEDED.**
    - The run: the wall assembled upstream and held 84 s at 15.9 m/s. It
      then aborted with `non_member_rear_end_risk:ramp.29` when the lead was
      92 m before the merge start, ~6 s before the scheduled merge_release.
    - WM recommendation: keep the law as frozen, and amend the smoke fixture
      only:
      - a registered safety abort is OK;
      - the smoke fails only on member_left_network, a missing cap/receipt,
        or a release past the merge start;
      - run all 3 runs with ≥ 1 merge_release required.
    - Alternatives:
      - exempt on-ramp vehicles from the rear-end rule near the merge
        (v2.2.1 law registration);
      - release earlier.
    - Also a logging fix: `a_priori_effective_hold_s` is computed at release,
      not at assembly.
12. **v2.2 re-smoke R2: DRAFT READY** (pending the item-11 decision).
    - Branch zcode/smoke22r2 69f3b43. `SMOKE_V22_R2.json` LF `0418480c…f112`,
      `.md` LF `75af6871…a278`.
    - Pass rule as recommended. 3 TRAIN worlds chosen far upstream (e.g.
      w8661 W-mid-early, w8697 W-mid-early); new runs root
      `campaign_walls_v22_smoke_r2`.
    - Behaviour-neutral hold_cap logging patch; R1 diagnosis in
      SMOKE_V22_R1_DIAGNOSIS.md.
    - On the owner's pick: freeze, then send to the recon.
13. **FEAT1: 12 workers give ~335 runs/h**, the same as 6 workers, so the
    bottleneck is not CPU. The ETA is ~13:15.
    - Options: (A) a partial TRAIN pack for member-1 training now, disclosed;
      (B) wait.
    - Asked of main (07:15).
    - **Main's framing, for the 08:00 brief:**
      - A = a disclosed PREVIEW member-1 on the partial corpus. It is NOT
        part of the final ensemble; the final members train on the full pack.
      - B = wait (~13:15).
      - A GLM profiling task is running (read-only) and may prepare a
        byte-identical speed-up for the remaining runs.
14. **Two-GPU-box plan: OWNER PICK** (a / b / c; estimates were sent to main at
    07:40; configs in bundle/box_configs/2box from task 1ca1766d).
    - WM recommends (b), the lean 9: Idea A m1–5 on A; (i) → (ii) → (iii) →
      N m1 on B.
    - Upgrade to (c), co-run, only if the timing gate shows ≥ 1.3×.
15. **Update 08:41.**
    - FEAT1 is at ~600 runs/h with 12 workers; ETA ~10:45, so the A/B preview
      is likely moot.
    - The profiler found a byte-identical 2.4–3.3× fix (zcode/feat1prof
      b1d9598). It was NOT applied to the live run; it is kept for
      J-PULSE/J4B/chunk-5.
    - The 2-box configs are merged (e5bead4); WM recommends (b).
16. **Update 10:22.**
    - **v0.4b code:** READY and merged (d0875d7; 73 tests; TRAIN-only
      dry-run). It is waiting for the owner's ack of e00cd98e; the exact
      post-ack steps are in output/wd01/partB_v04b/README.md.
    - **FEAT1 resume:** live at 10 workers with the byte-identical
      --idea-fast flag.
    - **09:28 crash root cause:** the parent could not unpickle a worker's
      wall-token exception (missing sys.path), which broke the pool. Now fixed,
      with per-run [error] logging; any failing run will be listed.
    - **Variant c box-B:** fixed to run (i) alone first (25771cd).
17. **Resolved, 13:50 (owner: "fix the test, approve J4b").**
    - SMOKE_V22_R2 is FROZEN (LF c5df40b0…, 0006a56). It is a post-hoc,
      disclosed amendment carrying the owner's exact fail list; the recon
      re-smokes now.
    - J4B is FROZEN (811f718).
    - GPU plan = variant (c) on 2× AutoDL vGPU-48GB Blackwell. The torch
      ≥ 2.7/cu128 fix is in progress.
