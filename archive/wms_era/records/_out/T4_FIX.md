COMPLETED — WM-S trainer fix verified; documented real-data CPU dry run PASSED; full suite has only the 4 known physics_torch failures.
Code: branch wms/p2, HEAD 588093d (fix commit 1c46c3f in history).

## Kept work committed this session
- _out/T4.md prior status preserved (commit 588093d; only _out/T4.md changed, _ext$n junction left untracked).

## Fix verification (1c46c3f)
- test_cav_mask_follows_identity_when_a_slot_is_reused passes: per-tick CAV masks feed rollout and loss; a reused HDV slot is neither CAV-teacher-forced nor counted as HDV at t0.
- Targeted: python -m pytest output/wd01/wm_n1/wms/tests/test_wms.py -q -> 11 passed.

## Real-data CPU dry run (READ-ONLY TRAIN)
Command (cwd output/wd01/wm_n1; OMP_NUM_THREADS=MKL_NUM_THREADS=4):
  WM_N1_TRAIN_OUTCOMES=D:/rd-staging/wmn1_feat/labels_cov_20261004T072826Z
  WM_N1_TRAIN_IDENTITIES=E:/tmp/wmn1_train_runs_20261002/sidecars3_20261003T0005CN/identities
  python -m wms.cli dry-run --pack D:/rd-staging/wmn1_feat/pack_v3 \
    --pack E:/rd-wt/dspeed-data/_out/pack_dspeed --pack E:/rd-wt/damping-snap/_out/pack_damping \
    --identities <3 identity dirs> --overlay <2 overlays> \
    --out E:/rd-wt/ws-p2/_out/wms_dry3 --threads 4 --sumo-fidelity
Result: status passed; 2 optimizer steps; batch 4; 10 s unroll (20 steps); threads 4; num-workers 0.
  update1 loss 8.7188 (pos 8.0051, spd 0.4307, lane 0.2830); update2 loss 8.4742 (pos 7.7745, spd 0.4178, lane 0.2819)
  both finite=true; valid_targets=5981 (>0); valid_hdv_slots_t0=313; g(0)=0.0; gate monotone; gate_last=0.1275
  data: mode a11-index-window, split TRAIN, n_records 3060, horizon_steps 240
  receipt: _out/wms_dry3/TRAIN_RECEIPT.json ; stdout _out/wms_dry3.stdout.log
Memory: pack path is mmap/one-window-at-a-time (from_real_packs dataset build ~0.33 GB working set; no whole-pack/full-run load).
  A single 20-step BPTT backward peaks ~4-10 GB private because HdvResidualNet builds [B,K,K,64] pairwise attention bias at K=192
  (model compute, not data loading); observed process peak ~8.4 GB working set / ~9.9 GB commit, nowhere near the ~50 GB crash.

## Full suite
python -m pytest output/wd01/wm_n1/wms -q (threads 4): 4 failed, 11 passed.
  All 4 failures are the pre-existing test_physics_torch.py tests (numpy-parity, deterministic entry, mandatory-merge x2);
  physics_torch is off the WM-S trainer path and was NOT edited (fixes belong to branch wms/torchphys).

## Archive
git archive --format=tar --output=_out/code_wms_588093d.tar HEAD output/wd01/wm_n1
SHA-256: 5f9af149a84c19a9446001cc3c98e60da280e2bed33e1adc9aa22af0b671d0f4
size 9,185,280 bytes; 694 entries; contains output/wd01/wm_n1/wms.

## Prior notes (superseded above where noted)
PARTIAL — trainer fixes committed; required full suite and documented real-data dry run remain blocked.
Code commit: 1c46c3f (Fix WM-S per-tick CAV identity masks).
BLOCKER 3: FIXED. t0 CAV status is now masked by same-identity alive-at-tick status; rollout and loss use each tick's CAV mask.
Synthetic result: the old cumulative rollout_mask made the broadcast loss-neutral for both a departed CAV and a reused slot (all loss mask entries false); stale CAV teacher forcing/status still needed fixing.
Regression: test_cav_mask_follows_identity_when_a_slot_is_reused passes; reused HDV receives neither CAV teacher forcing nor HDV loss as a t0 identity.
REVIEW #11 (MAJOR, zero optimizer steps): FIXED with drop_last=False and explicit per-stage zero-step refusal; small-dataset test passes.
REVIEW #10 (MAJOR, driver/history.py): NOT ON WM-S trainer path; WM-S uses _prior_history, which clamps history to the containing run start.
CPU: 4 threads; targeted WM-S tests pass (11 passed); ruff passes.
Required suite: python -m pytest output/wd01/wm_n1/wms -q — 4 failed, 11 passed; pre-existing physics_torch failures (not changed; outside p2 trainer path).
Documented 2-step real-data dry run: NOW RUN AND PASSED (see above; the earlier "not run" blocker was a missing-sidecar/junction issue resolved by the _ext junctions and the documented env vars).
Archive command (this session): git archive --format=tar --output=_out/code_wms_588093d.tar HEAD output/wd01/wm_n1
Archive SHA-256: 5f9af149a84c19a9446001cc3c98e60da280e2bed33e1adc9aa22af0b671d0f4
Next: GPU timing/memory still unmeasured; repair the four pre-existing physics_torch test failures on branch wms/torchphys; lane/physics remain uncalibrated vs SUMO (T2/T3 scope).
