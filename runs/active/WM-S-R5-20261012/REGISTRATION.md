# WM-S-R5-20261012 — round 5 (registration, written before launch)

- Owner approval: 2026-10-12 ~03:45 local ("go" — merge, register, start the GPU box, preflight, launch).
- Code: commit 4dc08f6 on main (github.com/jhpeng321-star/research; branch wms/r5-prep fast-forwarded), config
  `configs/r5_f23.toml` (extends r5_f2 -> r5_base -> r4). Full unit suite on main: 629 passed, 1 skipped; prepush PASS.
- Recipe = R4 recipe (data, curriculum 10/30/60/120 s with 2500/2500/1875/1250 steps, batch 8, lr 1e-3, saturating
  residual gate, soft residual bound, residual-size penalty, best-long-horizon selection) initialised from the R4
  checkpoint (103 tensors loaded, 2 new = lane-gate parameters), PLUS three changes in ONE run (owner decision):
  - FIX 1 learned driver settings in the physics: rollout_mode = hybrid; driver_grad_across_chunks = true (encode once
    at t0, leaf-tensor gradients from every chunk). Settings learned from prediction error only — no driver-type list,
    no labels, no true parameters (owner rule 2026-10-11).
  - FIX 2 long-horizon learning: truncation_steps_per_stage = [20, 60, 120, 240] (the 120 s stage back-propagates
    through the whole 120 s), per-step activation checkpointing; grad clipping 1.0 unchanged.
  - FIX 3 lanes: lane_logit_cap 1.5; separate saturating lane gate (init g_max 0.16, tau 30 steps); flip-rate loss
    weight 1.0, allowance 0.00124327163 per HDV-step (TRUE rate) replacing the lane-rate loss (weight 0).
    Flip-rate weight is fixed at 1.0 for the whole run (owner-side decision 2026-10-12; no mid-run change).
- Why: docs/WM_CHANGE_LOG.md rows R4, R4-DIAG-ERR, R4-DIAG-LANE, R4-DIAG-HIST, R4-DIAG-ORACLE, R4-CHK-TYPEID, R4-CHK-FADE;
  preparation runs/active/WM-S-R5-PREP-20261012/PREP_REPORT.md (+ CODE_REVIEW_SONNET.md).
- Budget: ONE gpuA run (RTX 4090 D, 49 GB) <= 14 h wall clock incl. preflight + a short GPU smoke (<= 20 min), then shut
  the box down. Expected ~9-11 h (R4 took 6.1 h of pure training; FIX 1/2 add encoder + checkpoint recompute).
  GPU memory must be measured in the box smoke (R4 already peaked at 21.9 GB at batch 8; the CPU-RSS estimate in the
  prep report is not a GPU measurement). If the smoke exceeds ~44 GB, stop and report (do not change the recipe silently).
- Comparators: C1 (data_cache/ckpt_c1fast/wm_s.pt) and R4 (runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt).
  Reference ceilings only (given types, NOT the method): physics + true types 4.31/48.65 m, + identified types 5.12/46.54 m
  at 60/120 s on wall_same.
- Evaluation (CPU, clean scorer, pairing world+key+source_arm+t0): wall_same 67, wall_pilot 53, real-history 120;
  horizons 10/30/60/120 s; blow-ups; paired world-bootstrap CIs vs C1 and R4; then the pre-registered attribution in
  runs/active/WM-S-R5-PREP-20261012/DIAGNOSIS_PLAN.md (error split wrong-lane vs right-lane, survivor speed drift,
  lane rates vs recorded, inference-time switch-offs, recorded-lane teacher forcing). Added here: compare the encoder's
  learned settings directly with the true driver types (evaluation-only; per-type distributions / separability) as C2
  evidence.
- Judging: diagnostic framing as in round 4 (no fixed pass/fail); hoped-for: better than C1 and R4 at 120 s, no blow-ups,
  learned settings that track the true types.
- Data/safety: no sealed/CAL/HOLDOUT/chunk-5/INC1/J4B data; same packs/lists as R4 (911 train / 151 val windows); never
  launch in retry loops; LF scripts only; outputs under /root/autodl-tmp/wms_train_r5/.
