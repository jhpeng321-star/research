# R9_STUCK_DIAG - why round 9 leaves ramp cars stuck and blows up (S-DIAG9, 2026-10-23, inference only)

All numbers: gpuR CPU, 72-state `wall_same` set, entries off, CAVs "teacher-forced" by the harness, r9 = `wm_s_stage300.pt`
(md5 9040d65e, the file cpuN scored), round 8 = `wm_s_stage300.pt`. DIAGNOSTIC variants are inference-time switches; no weights, configs or
model code were touched. Scripts: `tools/diag_r9stuck_trace.py` (rollout + traces + switches), `tools/diag_r9stuck_analyze.py`,
`tools/diag_r9stuck_check_base.py` (verbatim copy of cpuN `v8wm3_check_cpun_k.py`). Raw pickles on gpuR `/root/autodl-tmp/c4/v5/diag_r9stuck/`
(`all_r9.pkl` 72 states k=1 with traces, `abl_k2.pkl` 24 states k=2, `tf.pkl`, `lanes*.pkl`).
Reproduction (k=1, 72 states): r9 stuck 114/368 (SUMO 4), blow-ups 47/72, HDV ds120 493 m; round 8: 6, 5/72, 41 m (cpuN k=5: 611, 49/72, 462 m vs 55, 5/72, 38 m).

## Verdict: two causes; the leading hypothesis is right about the stuck mechanism but is NOT the main cause of the blow-ups

1. **Contract/harness bug (all of the blow-up, ~2/3 of the stuck cars): the first_passage step ignores the recorded CAV teacher.**
   `WmSModel.step` passes `cav_teacher if self.merge_pivot is None else None` and only runs `_apply_cav_overrides` when a `cav_controller`
   exists (`wms.py` ~1480-1485). `WmsPredictor.predict_states` (replay of a recorded arm) passes `cav_teacher=teacher` and `cav_controller=None`
   (entries off), so in r9 the CAVs (the wall arm: 4-37 slow blockers) are NOT replayed; they follow their native hold-speed prediction.
   Probe (6 states): CAV median |ds| at 30/60/120 s is 0 m for round 8 and 19-409 m for r9. The mainline then free-flows: mean HDV speed at 120 s
   WM/SUMO 25.5/15.3 m/s (r9) vs 17.0/14.5 (r8); lane-0 cars, which never touch the ramp, are ~500 m off (r8: 33 m). So the r9-vs-r8 table is confounded.
   Feeding the SAME recorded teacher through the supported `cav_controller` hook (switch `tf=1`, 24 states, k=1): HDV ds120 488 -> 47.6 m,
   blow-ups 17/24 -> 3/24 (round 8: 3/24, 41.7 m), stuck 45 -> 17 (round 8: 5).
2. **Boundary design (the remaining stuck cars): pure gap acceptance with a follower closing-speed term and no follower-yield channel.**
   Stuck cars waiting at the lane end (no TF; 5103 rows, 92 cars): ego v median 0.0 m/s, follower v_f median 22.2 m/s, gap_f 15.7 m, required d_f median
   262 m (p90 419), the closing term `relu(v_f^2 - v^2)/(2 b_f)` is 95 % of d_f, the follower is the binding side in 98 % of rows, c median -2.0, c >= 0 in
   0.1 % of rows, nearest-follower acceleration median -0.1 m/s^2 (no yield). Hybrid noise cannot overcome c < -2 (needs tau*z + w*eps >= 2, logistic tail ~1e-3).
   With the CAV teacher the effect shrinks but remains (11 waiting cars: v_f 7.0, gap_f 5.7, d_f 30 m, c -1.8).
   The boundary itself is calibrated on recorded states: at SUMO's own merge instants (n=364) c under r9 parameters has median 0.0 and 78 % >= -0.5
   (SUMO merges at v_f median 1.2 m/s, gap_f 4.1 m, ego 1.2 m/s, d_f 3.1 m). The rollout STATE differs: lane-1 followers of the same cars are as slow as
   SUMO's for 0-5 s (6.9 vs 6.5 m/s), then diverge (15-30 s: 7.8 vs 3.5; 60-120 s: 25.3 vs 13.9 m/s, gap_f 22 vs 55 m).
   Correction to the hypothesis: SUMO followers do not visibly yield either (follower acceleration median 0.0, p10 -0.6, 3761 SUMO waiting rows); they are
   already slow/queued, i.e. the TF-fitted boundary only fires in jammed lane-1 states and the rollout jam dissolves (bistable: no jam -> no merge -> no jam).
   Round 8 avoids this: its law anticipates follower braking (`v_f + a_f*tau_a`, `gap + (v-v_f)*tau_a`), divides the requirement by a heterogeneous alpha,
   adds a learned follower yield (`core_yield`) and `merge_end_wait`; `merge_core` is not executed at all when `merge_pivot` is set.

## Ablations (DIAGNOSTIC, inference-only). 24 states x k=2, 274 ramp HDVs, SUMO twins stuck: 4

| variant | stuck in lane 2 | med dep err WM-SUMO s | HDV ds120 m | ramp ds60 m | blow-ups |
|---|---|---|---|---|---|
| (a) r9 as is | 80 | -2.0 | 440 | 72.8 | 17/24 |
| (b) fitted-init pivot values (training undone) | 84 | -3.8 | 347 | 74.4 | 16/24 |
| (c) ramp_lane_end_stop off | 88 | -1.5 | 413 | 88.3 | 17/24 |
| (d) ramp_risk_retain off | 87 | -2.0 | 421 | 77.3 | 18/24 |
| (c+d) both off | 100 | -5.2 | 438 | 69.2 | 17/24 |
| (e) round-8 law + physics with r9 weights (`merge_pivot` removed, so the CAV teacher is also restored) | 14 | -9.8 | 45.6 | 174 | 4/24 |
| round 8 | 8 | -11.0 | 45.6 | 119 | 5/24 |
| (f) b_f = b_l = 2.08 | 36 | -6.8 | 407 | 168 | 17/24 |
| (g) w = 1.0 | 11 | -17.0 | 371 | 226 | 17/24 |
| (f+g) | 6 | -16.8 | 354 | 244 | 15/24 |
| b_f = 1000 (closing term removed) | 19 | -12.5 | 383 | 185 | 18/24 |
| r9 + round-8 merge_core follower yield re-enabled | 84 | -1.5 | 393 | 66.9 | 17/24 |

CAV-teacher replay (switch `tf=1`, same 24 states, k=1, one run):

| variant | stuck | HDV ds60 / ds120 m | blow-ups | mean HDV speed WM/SUMO at 60 s, 120 s |
|---|---|---|---|---|
| r9 (no teacher) | 45 | 73 / 488 | 17/24 | 24.2/21.0, 25.5/15.3 |
| r9 + teacher | 17 | 36 / 47.6 | 3/24 | 20.5/20.7, 17.7/15.2 |
| r9 + teacher + (f+g) | 5 | 31 / 41.7 | 3/24 | 21.1/21.0, 17.3/14.5 |
| round 8 | 5 | 35 / 41.7 | 3/24 | 20.8/20.8, 17.0/14.5 |

Reading: training is not responsible (b); the physics flags are not (c, d; cars stop 2-3 m before the end, pinned share of waiting rows 13 %); the frozen yield head does
not help (it only acts near margin 0); (f), (g) and removing the closing term cut the stuck count but leave the blow-up (the blow-up is not the stuck cars); restoring the CAV
teacher removes the blow-up and most of the stuck excess; the rest is the boundary. `merge_end_wait` is irrelevant (round-8 law with it off: 3 stuck, 41.7 m, 24 states k=1).
Departure-time medians are survivor-biased (stuck cars drop out) and must not be used to rank. With the teacher the 60 s merge count is 39 vs SUMO 41 (round 8: 50).

## Per-car evidence (no teacher; 114 stuck car-samples of 368, 72 states k=1)
- 78 wait with a fast follower (median v_f > 10 m/s while eligible), 21 with a slow follower (c still < 0), 15 mostly physically vetoed (lane-1 body overlaps the ego, c = -inf);
  none is still moving. Only 5 ever touch x = 75.44; 22 never satisfy "within 10 m of the end and v < 2" (queued behind another stuck ramp car).
- Max c over each stuck car's life: median -0.73 (p90 -0.14); only 25 % ever reach -0.5, so the persistent latent (tau = 0.28) rarely licenses a crossing.
- Merges that do happen in the WM: gap_f median 14.8 m, v_f 5.6 m/s (SUMO 4.1 m, 1.2 m/s): polite merges that cause no merge-induced jam.

## Fix classification (nothing implemented)
(i) Bug / contract fix inside owner decisions
- Replay the recorded CAV arm for first_passage models: pass a `cav_controller` returning `teacher[:, i+1]` from the predictor (what `tf=1` does in
  `tools/diag_r9stuck_trace.py`), or make `step` honour a supplied `cav_teacher` (deployment keeps the causal no-teacher path). Until then every first_passage eval with the
  recorded CAV arm is invalid for HDV/blow-up metrics; re-score r9 with the teacher before judging it. Training uses the causal replay of the recorded CAV arm and is not
  affected by this, but the retired on-policy term would have exposed it.
(ii) Model/objective changes needing an owner decision
- A follower-yield or anticipation channel in the pivot (anticipated `v_f + a_f*tau_a`, a mainline yield tied to c, or a yield learned on-policy); the frozen round-8 yield head is not enough.
- Boundary parameterization: relative closing speed instead of `v_f^2 - v^2`, a cap on the closing term, wider heterogeneity (w = 1.0 or larger tau), or a waiting-time/impatience term for lane-end waiters.
- Restore an on-policy/closed-loop term: a TF-only 90-step fit cannot see the rollout shift of lane-1 followers (recorded states are calibrated, c median 0.0).
- Ramp-car approach: cars decelerate to v = 0 near the end, which maximizes d_f.

Caveats: one harness (entries off); ablations 24 states and k <= 2, TF run k = 1 (one sample); no claims about SUMO-internal LC2013 cooperation; gap/c use the model's post-move geometry.
No evaluation seeds 964000-964199, sealed/holdout paths or hetdata_truth were used.
