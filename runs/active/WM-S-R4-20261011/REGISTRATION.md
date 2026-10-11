# WM-S-R4-20261011 — round 4 (registration, written before launch)

- Owner approval: 2026-10-11 ~01:50 local ("I turned on gpuA, start the training").
- Code: commit 711b078 on main (github.com/jhpeng321-star/research), config `configs/r4.toml`.
- Recipe: R3 data/recipe (C1 architecture + curriculum 10/30/60/120 s, 2500/2500/1875/1250 steps, C1 packs + pack_v3,
  60 s driver history, soft residual bound 4 m/s^2, residual-size penalty 0.01, lane-rate loss, best-long-horizon
  checkpoint) with braking losses OFF, merge_cal OFF, lane_rate_target 0.00124327163 (TRUE rate, junction toggles
  excluded), and the NEW saturating residual gate g(k) = sigmoid(a)(1 - exp(-k/softplus(b))), init g_max 0.30, tau 50 steps.
- Why: docs/WM_CHANGE_LOG.md rows R3, R3-DIAG, LANEDWELL, LANECAL and "Round 4 plan".
- Budget: ONE gpuA run (RTX 4090 D) <= 12 h wall clock incl. a box smoke (<= 15 min); then shut the box down.
  No other GPU runs, no paired controls (owner protocol: one GPU, one main model per round).
- Comparator: C1 (data_cache/ckpt_c1fast/wm_s.pt), the current best. No black-box retraining.
- Evaluation (after training, CPU, clean scorer with pairing key world+key+arm+t0): wall_same 67 + wall_pilot 53 h60 states,
  plus the real-60 s-history set; horizons 10/30/60/120 s; blow-ups; fingerprints: learned g(k) curve, late |g*da|,
  mean signed speed error 0-60 s, lane-change rate per HDV per 10 s (TRUE definition) vs recorded, wrong-lane share,
  best vs last checkpoint, per-stage checkpoints.
- Judging (owner, 2026-10-11): no fixed pass/fail rule — "look at the outcome first; neither is likely the final model;
  determine what is going on and continue fixing".
- Data/safety: no sealed/CAL/HOLDOUT/chunk-5/INC1/J4B data; the switching pack is not used unless R3's receipt shows
  it was (then identical use); never launch in retry loops; LF scripts only; outputs under /root/autodl-tmp/wms_train_r4/.

## Finish (2026-10-11)
- Training finished ~09:00 local: 8125/8125 steps, loss finite throughout (PID 2927, started ~01:58).
- Results fetched to runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/ (tar tmp/r4_box/r4_main_a1_results.tar.gz); sha256 equal to the box: wm_s.pt 3d73f4ac..., wm_s_best.pt 6e935700..., TRAIN_RECEIPT.json cbb0c9ec....
- gpuA shut down after the fetch (ssh no longer serves the box). Next: CPU scoring vs C1.
