# WM-S change log — what we tried, what happened, what not to repeat

Single place to check **before** proposing any model/training change, so we never
back-track. One row per change. Update it whenever a run is scored or a decision is
made. Status: **KEPT** (in current best or next round) · **REJECTED** (do not repeat
as-is) · **DEFERRED** (planned, not yet tested) · **PENDING** (being tested) ·
**VETOED** (owner said no).

Current best model: **C1FAST structured** (`runs/active/WM-S-C1FAST-20261006/box/full_struct/wm_s.pt`).
Unseen-wall HDV median |ds| at 60 s on `wall_same`: 8.71 m (full 67 states); 12.84 m on
the 20-state diagnostic subset.

Owner rules (2026-10-09): compare new models only with the previous best (no
retrained black boxes); judge on unseen controllers (walls, PI) at 60/120 s plus
blow-up rate; prefer one change per run, or pair runs that isolate the risky change.

## Changes

| # | change | where tried | result | status | evidence |
|---|---|---|---|---|---|
| 1 | Physics core IDM/MOBIL + gated residual (structured WM-S) | C1FAST | beats both black boxes on unseen PI and unseen walls (8.71 vs 25.41 / 40.07 m at 60 s) | KEPT | `E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md` |
| 2 | Random start-time shift `start_shift_prob 0.75` | C1FAST (on), MAIN2 (off) | C1 trained with it; MAIN2 turned it off (bundled, not isolated) | KEPT (C1 value) | C1 vs MAIN2 TRAIN_RECEIPT diff |
| 3 | **Hard residual clamp** `--residual-cap-mps2 1.5` (and asym +1.5/-3.0 in brake run) | MAIN2 brake + no-brake | **main cause of the MAIN2 regression**: zero gradient when saturated → raw residual grew to mean 6.7 m/s² (p95 26) vs 1.8 in C1; cap saturated on 59% of HDV-steps; adding only this cap to C1 doubles its error (12.8 → 25.2 m) | **REJECTED — never use a hard clamp in training** | `E:/rd-wt/ws-bisect-settings/_out/bisect_settings/`, `E:/rd-wt/ws-bisect-residual/_out/BISECT_RESIDUAL.md` |
| 4 | Braking-fix losses (asym speed 2, decel weight 5 at a<-3, propagation 1/3 s) | MAIN2 brake (with the hard cap) | worse than no-brake at 10–60 s, more blow-ups; but confounded by the hard cap, so **not a fair test** | PENDING (round 3, no cap) | `E:/rd-wt/ws-main2score/_out/MAIN2_SCORE_RESULT.md` |
| 5 | MAIN2 bundle (switching 50% + event oversample 2, entries-v2, projection, residual cap, lane-logit cap 3, traffic/stability/conservation losses, merge_cal, start-shift 0) | MAIN2 | both models ~5× worse than C1 at 60 s (42.8 / 49.7 vs 8.7 m); regression appears only in the 60/120 s stages | REJECTED as a bundle | `E:/rd-wt/ws-bisect-stages/_out/BISECT_STAGES.md` |
| 6 | Late-stage lane-change drift (pred 9.3% vs recorded 1.23% per HDV-step) | MAIN2 | appears only in 60/120 s stages, not CAV-related, not from the residual, not from switching data (0.09% at stage 10 with switching present) — likely a long-horizon loss term | problem to guard against | `E:/rd-wt/ws-bisect-residual/_out/BISECT_LANE_STAGES.md` |
| 7 | Traffic / stability / conservation loss terms | MAIN2 | traffic share 0.01–2% of loss, stability and conservation 0%; prime suspects for the lane drift (unproven) | REJECTED for now (not in round 3) | MAIN2 val_log loss_share |
| 8 | Switching data (per-CAV controller switching) | MAIN2 at 50% + oversample 2 | cleared of the lane-drift blame; own effect not isolated | DEFERRED (owner's main data goal; next clean round) | BISECT_LANE_STAGES.md |
| 9 | Gap projection (inference setting) | scored on C1 | slightly helps C1 at scoring time (12.8 → 10.1 m) | KEPT as scoring option (no retrain) | bisect_settings `c_projection_on` |
| 10 | Merge calibration (inference setting) | scored on C1 / MAIN2 | slightly helps C1 (12.8 → 11.1 m); not the MAIN2 culprit | KEPT as scoring option | bisect_settings `c_merge_cal_on`, `n_merge_cal_off` |
| 11 | Lane-logit cap 3.0 | scored on C1 / MAIN2 | harmless (C1 13.4 m) | neutral | bisect_settings |
| 12 | pack_v3 data (was silently 0 windows: pack name vs folder basename mismatch) | fixed 2026-10-09 | loader fixed (fail-closed); +294 train / 47 val windows; new lists keep eval worlds out of training | KEPT (round 3) | `E:/rd-wt/ws-packv3fix/_out/PACKV3_FIX.md` |
| 13 | Eval-world disjoint training lists (lpwo_v3) | fixed 2026-10-09 | all earlier models trained on the eval worlds' other arms (controller still unseen) | KEPT | PACKV3_FIX.md §3 |
| 14 | Vehicle entries in eval/training | small C1 on/off check, 20-state subset | entries ON does not help C1: 60 s 11.96 → 12.49 m, 120 s 111.8 → 169.6 m (C1 never trained with entries; 70–110 entrants per state by 120 s) | REJECTED for round 3 (revisit only in a model trained with entries) | `E:/rd-wt/ws-bisect-settings/_out/ENTRIES_SMALL.md` |
| 15 | Hybrid rollout / driver_params mode | OLDHIST | driver_params mode drops the residual — do not score in that mode | REJECTED (driver_params scoring) | OLDHIST_RESULT.md |
| 16 | Self-recovery / pushforward training | proposed 10-08 | owner vetoed | VETOED | owner 2026-10-08 |
| 17 | Learn a direct 120 s predictor and converge rollouts to it | discussed 10-08 | discussion only | DEFERRED (idea) | owner 2026-10-08 |
| 18 | Retraining black-box baselines | MAIN2 bb_traj/bb_id | bb_traj crashed (stability term needs residual head); bb_id stopped by owner | REJECTED (use old C1 baselines) | `runs/active/WM-S-MAIN2-20261009/REGISTRATION.md` |

## Round 3 (being built 2026-10-09; branches `wms/r3-*`)

Base = C1 recipe (no hard cap, start shift 0.75, no traffic/stability/conservation) + pack_v3 (#12, #13).

| fix | flag | role | separable? |
|---|---|---|---|
| R3-1 smooth residual bound + residual size penalty | `--residual-soft-bound-mps2`, `--residual-mag-weight` | guard against #3 | fingerprint: logged residual size |
| R3-3 lane-change-rate loss | `--lane-rate-weight/--lane-rate-target` | guard against #6 | fingerprint: logged lane rate (can't tell if drift would have happened anyway) |
| R3-4 keep best long-horizon checkpoint | `--select-by-long-horizon` | guard | exactly (best vs last checkpoint) |
| R3-5 30–60 s per-driver history | `--history-s`, `--history-stride-s` | long-horizon accuracy, C2 | fingerprint only (score with history cut to 4.5 s); owner chose one GPU / one model per round |
| braking losses, no cap (#4) | braking env flags | missed jams | confounded with base guards only |
| R3-2 jam-window oversampling | `--jam-oversample` | missed jams | **overlaps braking losses, history and pack_v3** → DEFERRED |
| switching data 30%, no oversample (#8) | `--switch-mix-ratio 0.3` | owner's main data | **overlaps pack_v3 as a data change** → DEFERRED to its own round |

All round-3 features are flags defaulting OFF; deferring = leaving the flag off.

## Do-not-repeat list

- Hard clamps on the learned residual during training (#3).
- Bundling many recipe changes into one run without a paired control (#5).
- Scoring with `driver_params` rollout mode (#15).
- Assuming an OpenCode agent is still running without `task_status` (two agents silently timed out on 2026-10-09).
- Self-recovery/pushforward training (#16, vetoed).

Owner decision 2026-10-09: from round 3 on, ONE GPU and ONE main model per round (R3 = C1 recipe + pack_v3 + R3-1/3/4 guards + history + braking losses, no cap); no paired control runs; attribution by fingerprints.

## Diagnostics (no model change)

| date | finding | evidence | implication |
|---|---|---|---|
| 2026-10-10 | C1 transformer attention: leader gets only 6.4% of weight (below a uniform read); 60% goes to adjacent-lane cars (nearest ahead/behind ~20% each, MOBIL-like); 42% to cars behind; 38% to cars >50 m; effective ~4.7 of 13.5 candidates. Heads specialise (lead-car head up to 34% on leader when following; one rear-looking head; adjacent-lane heads). Lane-changers on the main road do NOT shift attention toward the target lane. CAVs get weight in proportion to their share (no CAV flag). | `E:/rd-wt/ws-attn/_out/ATTENTION_CHECK.md` (wms/attn-check 0b1c4d9) | Plausible since IDM already covers the leader, but heavy weight behind/far is a candidate weakness; a role-structured neighbour network (leader, leader+1, adjacent leader/follower) is a DEFERRED candidate fix for a future round |
| 2026-10-10 | Scoring harness reconciled (wms/r3-score2 1395183): old C1 fingerprints over the first 60 s match the bisect values (|g*da| 1.76 m/s^2, lane-change 0.98% vs recorded 1.18%), but over the full 120 s C1 ALSO drifts: |g*da| 7.46 (p95 34.2), predicted lane-change rate 8.49% vs recorded 1.55%, speed bias -1.14 m/s. | `E:/rd-wt/ws-r3-score2/_out/R3_SCORE.md` | Late-horizon lane drift and residual growth are not unique to MAIN2; the R3 soft bound, residual penalty and lane-rate loss target exactly this. Compare R3 vs C1 on both 60 s and 120 s fingerprint windows. |
| R3 | Round 3 (WM-S-R3-20261010): C1 recipe + pack_v3 + soft residual bound 4 + residual penalty 0.01 + lane-rate loss + best-long-horizon ckpt + 60 s history + braking losses (no cap), merge_cal ON | gpuA 2026-10-10 00:57-08:56 | QUICK wall_same (20 states, standard 5 s-history states) vs C1: 10 s 1.89 vs 2.29 m; 30 s 5.51 vs 5.08; 60 s 14.80 vs 11.96 (paired +1.71 [0.55, 5.44], worse); 120 s 110.8 vs 111.8 (+10.4 [-23.9, 48.3], n.s.); blow-ups 1/20 vs 0/20. Best ckpt similar (60 s 15.5). Fingerprints: residual tamed over 120 s (mean 1.64 vs 7.46 m/s^2; p95 6.5 vs 34.2); lane-change rate over 120 s 4.5% vs C1 8.5% (recorded 1.6%) but first 60 s 2.1% vs 0.98% (recorded 1.2%); speed bias -1.17 m/s (first 60 s) vs -0.31 => R3 is TOO SLOW (braking losses over-correct). | PENDING (real-60 s-history set + full sets running); current best stays C1 | `E:/rd-wt/ws-r3-score2/_out/r3_score/r3_quick_h60/ROUND_SCORE.md` |
| R3 (final) | Round 3 full scoring | 2026-10-10 | FULL standard sets (5 s history): wall_same 67 states 60 s 13.39 vs C1 8.71 m (paired +1.53 [-0.97, 4.43]); 120 s 131.9 vs 99.8 (+39.3 [-1.7, 64.5]); wall_pilot 60 s +1.53 [-0.97, 6.73], 120 s +35.6 [-1.7, 74.0]; 10 s slightly better (-0.14); blow-ups 2/66 vs 0/66. Best ckpt: 60 s +1.70 [-0.33, 3.88], 120 s +23.9, 0 blow-ups. REAL-60 s-history quick set (20): 60 s 28.3 vs 16.8 (+4.42 [0.17, 16.68], worse), 120 s +4.9 n.s., blow-ups 3/19 vs 0/19 => the history encoder did not help. | REJECTED as new best: C1 stays current best. Kept learnings: soft bound + residual penalty tame the 120 s correction (1.6 vs 7.5 m/s^2) and halve 120 s lane drift. Suspected harms: braking losses (over-braking, speed bias -1.2 m/s), 60 s history encoder (worse on real-history states), lane changes too frequent early (2.1% vs 1.2%). Next round candidate (one model): C1 recipe + pack_v3 + soft bound + residual penalty + lane-rate (stronger) + best-ckpt; NO braking losses, NO history. | `E:/rd-wt/ws-r3-score2/_out/r3_score/{r3_full_h60,r3_quick_h60real}/ROUND_SCORE.md` |
| R3-DIAG | Round-3 diagnostics (analysis only; no training/simulation): saved-checkpoint inference of R3 last vs C1 on the stored eval states | 2026-10-10 | Step 0 reproduces the harness bit-for-bit (C1 60 s 11.9576). D1 history (real-60 s-history quick 20): R3 full 28.31 / cut-to-4.5 s 27.16 / empty 39.1 / zero-embedding 34.2 vs C1 16.80 m at 60 s; cut-minus-full 0.00 [-2.35, 0.00] => history encoder is NOT the cause (R3 worse than C1 with or without it). D2 full wall_same 67: wrong-lane share at 60 s R3 7.9% vs C1 13.8%; removing wrong-lane HDVs leaves +3.88 of the +4.68 m gap; pooled signed speed error 0-60 s R3 -1.08 vs C1 -0.34 m/s; free-flow HDVs worse for R3 (median 11.9 vs 8.8 m). D3 merge calibration toggle moves R3 <= 1 m (n.s.). D4: hard-brake recall R3 78.8% vs C1 85.0% (no gain from braking losses); both over-predict hard braking (3.45% / 2.55% of steps vs 0.74% recorded); lane changes per vehicle per 10 s, 0-60 s: C1 0.159, R3 0.352, recorded 0.024; 60-120 s: C1 4.50, R3 1.22, recorded 0.037. | Ranked causes: (1) braking losses -> too slow (best supported, not ablated); (2) lane timing (secondary); history encoder and merge calibration not causes. Harness defects found: legacy paired_bootstrap keyed on (world, t0) so the full-set pairs collapsed to 29 of 67 (clean src/wms/eval/report.py already keys world+key+arm+t0; corrected 60 s R3-C1 +1.53 [-0.02, 4.43]); the "recorded lane-change rate" 1.2%/step counts junction-lane (-1) toggles, true ~0.12%/step, so the lane_rate_target default 0.0123 (src/wms/config/train.py) is ~10x too high — VERIFY before round 4. | `runs/active/WM-S-R3-DIAG-20261010/DIAG_REPORT.md` |
| LANEDWELL | Lane-change commitment at inference (dwell D 3/6/10 s, hysteresis m 0.5/1.0, D6+m0.5) on C1 and R3; analysis only | 2026-10-10 | Selected on seen_controls (40 states) by a pre-registered rule: m1.0. Unseen walls m1.0 vs base: 60 s paired 0.00 to -0.04 m, 120 s +1.7 to +15.8 m (worse, n.s.), C1 blow-ups 0 -> 1. Lane flips fall 2-4x but stay up to 40x recorded; wrong-lane share not better. 63-73% of late lane changes are not near-ties; MOBIL-only logits on predicted states already flip 3-9x too often. TRUE recorded lane-change rate: seen_controls 0.349%/HDV-step (wall_same 0.141%); trainer default lane_rate_target 0.0123 is the junction-toggle count (1.50%) -> ~3.5x too high. | REJECTED (do not add dwell/hysteresis as an inference rule). Next: calibrate lane_rate_target to the training arm mix (~0.0035) and consider a bounded lane delta. | `runs/active/WM-S-LANEDWELL-20261010/LANEDWELL_REPORT.md` |
| LANECAL | Lane-label audit + change-logit offset + residual-gate cap at inference (C1, R3; analysis only, pre-registered selection on seen_controls 40, one confirmation on the unseen walls) | 2026-10-11 | Labels: C1/R3 lane CE targets contain NO junction artefacts (junction steps masked); loader-style change labels are ~3-4x the TRUE rate; TRUE training rate 0.142% (C1 list) / 0.124% (R3 list), so lane_rate_target 0.0123 is ~10x too high. R3 had lane_head off (plain CE + lane-rate loss): the class-balanced CE was never used by C1 or R3. Change-logit offsets (C1 F1.5, R3 K1) bring lane flips to/below the recorded rate (C1 60-120 s 4.50 -> 0.035) but 120 s error does NOT improve (paired +25.5 / +27.7 m C1, +3.1 / +1.75 m R3; CIs include 0). Gate cap: C1 capped at g(60 steps) => 120 s paired -24.2 [-55.7, -10.4] (wall_same) / -32.3 [-58.2, -17.4] (wall_pilot), 70% of states better, 0 blow-ups; but across-state 60 s median 8.71 -> 11.08 and 120 s tail mean not better (+39 m). R3 capped at g(120): -9.3 / -4.3 m n.s., blow-ups 2 -> 1. g(k): C1 0.155/0.195/0.287/0.428/0.836, R3 0.136/0.180/0.328/0.548/0.936 at k=10/20/60/120/240. | Lane prior is NOT the main driver of 120 s error (lane-flip fixes alone don't help). The ever-growing residual gate IS a lever for 120 s (C1). Next: training-side gate with a learned ceiling / saturating schedule (not tested in training), corrected lane_rate_target (~0.0013). | `runs/active/WM-S-LANECAL-20261010/LANECAL_REPORT.md` |

## Round 4 plan (owner-agreed direction 2026-10-11; not started, no training yet)
Recipe = C1 architecture/curriculum + pack_v3 + 60 s driver history (neutral in R3-DIAG) + R3 guards (soft residual bound 4 m/s^2, residual-size penalty 0.01, best-long-horizon checkpoint) + merge_cal OFF, MINUS braking losses (asym speed, decel weight, propagation), with lane_rate_target corrected 0.0123 -> TRUE training rate (~0.0013), and NEW saturating residual gate g(k) = g_max*(1 - exp(-k/tau)) (g_max, tau learned; init g_max ~0.3, tau ~20-30 s). Attribution by fingerprints: gate -> learned g(k) curve + late |g*da|; braking removal -> speed bias; lane target -> lane-change rate (TRUE definition).

## Future directions (owner: keep as possible ways to move in)
- Separate gates for acceleration and lane deltas (LANECAL cap scaled both).
- Non-monotone / free-shape gate (can decrease); confidence-based gate driven by the predicted log-sigma.
- Longer-horizon training: more 60/120 s curriculum steps, longer BPTT window (no pushforward/self-recovery: retired).
- Milder braking terms, tested alone.
- History encoder as a separate C2 track (train/judge on real-history states).
- Vehicle entries in training and evaluation.
- Role-structured neighbour network (leader gets only 6% attention in C1).
- Lane change as a hazard/event model; per-driver MOBIL threshold/politeness (C2).
