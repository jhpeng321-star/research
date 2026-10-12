# WM-S-R5B-20261012 — round 5b (registration + launch record)

- Owner request 2026-10-12 ~13:35: build the stabilising fixes now, stop the current R5 run and launch the new one when ready;
  owner chose "add damping, then launch" (~13:55).
- Why: R5 (WM-S-R5-20261012, configs/r5_f23_half.toml) drifted upward in its 60 s stage (train position median 97 -> 165,
  val 60 s 125 -> 160; R4 flat ~80-86 / 115-125). New per-step logging on the box smoke showed exploding BPTT gradients:
  pre-clip global grad norm 6e2-4e3 (60 s, truncation 120) and 1e8-4e8 (120 s, truncation 240) against clip 1.0, so every
  update was a unit step in a direction dominated by the exploding chains.
- R5 stopped by the coordinator (owner request) at step 3503/4065 (62 steps into the 120 s stage), PID 4648, 14:06 CST.
  Kept on the box: wm_s_stage10/30/60.pt, wm_s_best.pt (12:25), train/val logs. Not fetched yet.
- R5b = R5's 30 s-stage checkpoint (wm_s_stage30.pt, sha256 1738db79..., val 30 s 40.3 vs R4 ~47-52) continued through
  the 60 s and 120 s stages only, config `configs/r5b.toml` (code df5b307): same FIX 1 + FIX 2 (truncation 120/240) + FIX 3,
  plus (a) batch 32, (b) lr 3e-4 (60 s) / 1e-4 (120 s), (d) grad_norm / grad_clipped / lr logged per step,
  (e) NEW forward-exact backward-only gradient damping through the carried rollout state, grad_decay_per_step = 0.90.
  Damping strength chosen on TRAINING data box smokes (4+3 steps): pre-clip norm 10-27 / 35-111 at 0.90 vs 20-46 /
  2.3e3-5.1e3 at 0.93. Steps 940 (60 s) + 625 (120 s), as R5-half. Full unit suite 632 passed; prepush PASS.
- Box env: /root/autodl-tmp/wms_train_r5b/r5benv.sh (R5 env + WMS_DATA_CKPT_R5_STAGE30); code archive sha256 b4a10c5c...
  (+ configs/r5b.toml updated to 0.90, sha256 cbe7526f...). Failed/old smokes kept (smoke, smoke_g0.93, smoke_g0.90).
- LAUNCHED once: PID 10363, 2026-10-12 14:07:20 CST, out /root/autodl-tmp/wms_train_r5b/r5b_main_a1,
  log /root/autodl-tmp/wms_train_r5b/launch_r5b_main_a1.log. At 14:09: step 11, ~8 s/step, grad norm 8-11, GPU 4.5 GB.
- Projected: 940 x ~8 s + 625 x ~16 s + validation ~= 5-5.5 h -> finish ~19:15-19:45 CST; scoring ~1 h after fetch.
- Evaluation: same as R5 (score_round5.py + diagnose_round5.py; comparators C1, R4) — also score R5's own best/stage-60
  checkpoints for the R5 vs R5b comparison. Interpretation caveat: R5b changes several things at once relative to R5
  (batch, lr, damping) on top of R5's fixes; attribution within R5b by the logged gradient norms and diagnostics only.
