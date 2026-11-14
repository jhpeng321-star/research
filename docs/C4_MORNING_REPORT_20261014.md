# C4 morning report — 2026-10-14

## Status at 07:35
| Machine | Job | State |
|---|---|---|
| gpuW | **World model v4** (owner-approved recipe), from scratch, started 03:59 | step 1,459, in the 30 s stage, healthy; expected end ~12:30 (max_hours 8.5) |
| cpuT | MAPPO fine-tune at accel cap 2.6 (from it81) | iteration 89, 0 collisions; reward flat ~-1.21/step (noisy) |
| cpuG | Data box | idle, ON (owner); GENRAND data copied to gpuW |
| gpuB | — | off |

## Overnight results
1. **Root cause of the ~9x over-credit**: imagined CAVs executed commands without SUMO's speedMode-27 IDM cap (SUMO: realized ≈ min(command, own IDM accel); MAPPO's +accel requests are almost entirely overridden — it controls by braking). Fix `cav_exec_cap="idm"` (imagination-time, physics): CAV pos. error at 30 s 53 -> 20 m; closed-loop excess +51/+104/+248 -> +4.5/+29/+149 veh-s at 30/60/120 s. Residual = other WM error. *Decision*: use CAVs' known vehicle-type settings in the cap (gate 1: 0.31 -> 0.08 m/s^2 accel error) or keep WM-inferred.
2. **Lane changes**: rollouts used argmax -> zero discretionary changes (truth 8.3/min, model 2.2 = forced merges only); hazard over-confident and mis-ranked. With true lanes, 120 s error 55 -> 22 m. Fixed (sampled decisions, decoupled training, v2 cues, hazard-owned merges, stopped-merger gaps) — in v4.
3. **Training data contained designed controllers** (CIRCLES damping in 19 % of "random" CAV time; scripted gap-opening/merge-aimed segments) and command=realized labels. Strict hygiene leaves the old data with 0.04 % undirected CAV actions -> **new undirected data collected**: C4-GENRAND-20261014 (seeds 961000-961515; 480 plain episodes + 36 x 14 branch points x 5 arms = 2,520 paired branches; 0 collisions/teleports; target-speed mode with CAV speeds within 2-6 % of native).
4. **Imagined entrance queues ~9x too long** (entry process required gaps from an outcome distribution, no speed reduction, entrant 5.1 m off). SUMO-like insertion admits 99-100 % of SUMO's real insertions (v2: 19-54 %). In v4 (`entry_insert_rule="sumo_like"`). Note: the policy's srcgrad surrogate still models the old blocking.
5. **v3 error breakdown**: 10 s braking dip = conservative driver-setting drift (T 1.38 vs SUMO 1.14) + long-horizon residual braking leaking into the first seconds; 60 s gap opened in the 30 s stage; V3best == V3last (selection always took the final stage). Fixes in v4: all-horizon validation, multi-horizon selection, 30 % short batches in long stages, 30 s stage 1,500 steps. Driver-setting prior skipped (owner).
6. **Decision check (C4/V2/V3)**: all equal; 120 s decisions not better despite halved car error; only 60 s group/single-CAV +0.06. Over-credit 9-10x everywhere (explained by #1).
7. **Wall benefit (9 clean matched pairs, directional)**: all WMs under-credit the 120 s wall benefit (<= ~half); C1/R4 best; C4/V2/V3 predict net harm on average. More pairs need SUMO (~40 min) — needs registration + OK.
8. **Disclosure**: the wall-check agent's divergence scan read basic position data of 20 CALIBRATION-split runs (printed divergence magnitudes only; not used). Nothing sealed/holdout-named opened.

## v4 recipe (configs/c4_wm_v4.toml, c4/int 2ca1f8a)
v3 + lane fixes + hygiene strict + junction ffill + GENRAND plain (mix 0.30) + paired branches with **loss.paired_diff** (weight 0.5, 20 % paired batches) + SUMO-like entries + multi-horizon schedule. Smoke and preflight passed.

**Early v4 validation** (note: v4's validation set now includes GENRAND windows, so numbers are NOT directly comparable with earlier runs; the unseen-wall scorer is the fair comparison after training): 10 s 6.32 at the end of stage 1; 30 s 45.4 at step 1400. Paired-difference share 8.6 % of the loss on paired batches; lane-change rate 0.002-0.003 per step (sampled).

## After v4 finishes (~12:30), automatically
Unseen-wall scores (with physics.json), decision check with `cav_exec_cap=idm`, lane calibration tool, v3-style error breakdown, wall benefit re-check on the 9 pairs. Then discussion before the final policy.

## Decisions pending
1. CAV settings in the execution cap: WM-inferred vs known vehicle type.
2. Policy entry-queue gradient: update srcgrad surrogate for the SUMO-like entry rule, or switch it off.
3. More wall benefit pairs (SUMO, ~40 min, registration).
4. MAPPO: keep the 2.6 fine-tune running until Thursday's evaluation?
