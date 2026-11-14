# C4 morning report — 2026-10-13 (overnight run log; updated through the night)

## What is running
| Box | Job | Started | Status |
|---|---|---|---|
| gpuW (AutoDL RTX 4080 SUPER 32 GB, day rent) | C4 world model from scratch (`configs/c4_wm.toml`, run `c4_wm_main`) | 02:20 | see latest status below |
| gpuW | C4 policy phase 1, dreaming in frozen R5b (`configs/c4_policy_gpuw.toml`, run `policy_p1`) | 04:25 | phase 2 switches to the C4 WM when it finishes |
| cpuT (Tencent 48-core spot, Ubuntu) | universal MAPPO (`configs/c4_mappo.toml`, run `mappo_main`) | 04:25 (fresh restart) | see latest status below |

Local copies (pulled every 10 min): `runs/active/WM-S-C4-20261013/fetched/`.

## Decisions I made overnight (owner gave me authority to keep training going)
1. **gpuW data upload** broke once (10 GB pack); resumed in 6 parallel parts, reassembled, sha256 verified against the bundle.
2. **World-model batch 8 -> 16, steps 2500/2500/1500 -> 1500/1500/600**: the 4080S vGPU is CPU-bound (1.7 s/step, GPU 10-40 % busy at batch 8); batch 16 costs the same wall time per step. Estimated ~5 h instead of ~9 h.
3. **Policy not co-run until fixed**: with the cooperation rollouts the policy needed >17-29 GB and OOMed next to the WM. A Claude agent found 84 % of the memory was the world model's pairwise attention bias stored for every counterfactual row; fix = exact activation checkpointing of pure per-step calls (gradients identical, tested at rtol 1e-5), ~40 % slower, ~8 GB for batch 16 / 20 s / 6 cooperation starts. Policy runs with a CUDA memory cap (`C4_GPU_MEM_FRACTION=0.45`) so it can never starve the WM run.
4. **MAPPO fairness fix + fresh restart**: the MAPPO config still had the old (uncalibrated) lane-gap rule while our policy uses the SUMO-calibrated one. Fixed (identical rules) and restarted MAPPO from scratch at 04:25 (the first 18 iterations with the old rule are kept as `mappo_oldlanes_18it`, not used).
5. **MAPPO update threads 16 -> 44** (update was 10 min vs 1 min SUMO collection; workers idle during the update). Also raised the open-file limit (46 workers hit the 1024 default at launch).
7. **Safety-constraint budget fixed (both methods), both restarted fresh at ~05:20**: the TTC budget was 0.01 per
   control step, but recorded TRAIN traffic itself has 3.64 (HDV-HDV close calls the CAVs can't remove), so the
   Lagrange multiplier grew without bound (6.5 after 160 policy updates) and would have dominated training. Budget
   = 3.6 for the C4 policy and MAPPO alike ("no more close calls than recorded traffic"). Earlier runs kept as
   `policy_p1_badbudget`, `mappo_badbudget_*` (not used).
8. **Policy phase 2 (08:05)**: switched the dream world to an ensemble of the new C4 world model (best checkpoint) +
   R5b, so the ensemble-disagreement penalty discourages exploiting either model; initialised from phase-1 best;
   entropy bonus 3e-4 -> 1e-3 because exploration collapsed in phase 1.
9. **Dreamer loop iteration 1 started (08:40-09:00)** on gpuW's spare cores: 120 SUMO episodes with the phase-1
   policy (penetrations 5/10/20/40 %, demands 2400/2600/2700, seeds 4107-4116, 10 min each; 0 collisions, 0 failures),
   converted to 5,280 training windows, and a WM refresh (init = C4 WM best; 30 % collected windows; 200/400/150 steps,
   ~1.3 h) is running. The policy switches to the refreshed WM when it finishes.
10. **Thread oversubscription fixed**: each collector worker's torch started 127 threads on the 16-core-quota box
   (load 40, >18 min/episode, also slowing the policy). Set single-threaded BLAS/OpenMP per worker (code fix in
   `traffic_shepherd/c4/collect.py`); episodes now ~25 s.
11. **Dreamer loop iteration 1 closed (10:37)**: WM refresh finished (`loop_it1/wm_refresh`); policy phase 3 now dreams in
   an ensemble of the refreshed WM + the original C4 WM (both learned lanes; disagreement 0.005 vs 0.04 for C4+R5b),
   initialised from phase-2 best, entropy bonus 3e-3 (entropy had kept collapsing to 0.07 at 1e-3; back to ~0.38).
12. **Entry-queue reward fix deployed (~12:50)** (owner-approved): reward counts vehicles waiting to enter
   (1 veh-s of entry waiting = 1 veh-s of on-road delay); SUMO queue parity exact vs the episode summaries;
   imagination uses the WM entry process with a backlog. MAPPO fine-tunes from iteration 44 (`mappo_ft`,
   `--init-from`); policy phase 4 (`policy_p4`): full cooperation weights from update 0, 30 s imagination, entries on,
   ensemble refreshed + C4 WM, ~40 s/update on gpuW.
6. **Spot safety on cpuT**: checkpoint every iteration (atomic, last 3 kept), spot-termination watcher (Tencent metadata, 5 s), graceful-stop + resume verified on the box twice.

## Results so far
### World model (finished 07:53, ~5.5 h on gpuW)
- From scratch, learned HDV lane changes, no designed controllers in its data. Checkpoints: `wm_s_best.pt`
  (sha256 1dca2818...), `wm_s_stage10/30/60.pt`; local copies in `fetched/gpuW/c4_wm_main/`.
- Training losses at the end of the 10 s stage matched C1/R4 (position 5.1 vs 5.3/4.8).
- Stage-30 checkpoint on wall_same (unseen walls, median HDV |ds| m at 10/30/60/120 s): 1.83 / 5.65 / 23.45 / 138.9
  (C1 2.04/5.07/8.71/99.8; R5b best 1.72/4.47/12.48/116.9).
- **Final C4 world model** (median HDV |ds| m; scorer labels it "R5-last/best"; `score/SCORE_final.md`):

  | set | model | 10 s | 30 s | 60 s | 120 s | blow-ups |
  |---|---|---|---|---|---|---|
  | wall_same (67) | C1 | 2.04 | 5.07 | 8.71 | 99.84 | 0/66 |
  | | R4 | 2.17 | 4.88 | 9.96 | 110.69 | 0/66 |
  | | R5b best | 1.72 | 4.47 | 12.48 | 116.86 | 1/66 |
  | | **C4 WM** | **1.68** | 6.03 | 12.38 | **93.64** | 1/66 |
  | wall_pilot (53) | C1 | 2.06 | 5.06 | 8.34 | 96.76 | 0/52 |
  | | R5b best | 1.72 | 3.47 | 9.08 | 112.63 | 1/52 |
  | | **C4 WM** | **1.68** | 5.92 | 11.07 | **81.84** | 1/52 |

  Best model so far at 10 s and at 120 s on both unseen-wall sets (learned lanes instead of MOBIL; trained from
  scratch, no designed controllers). Weaker than R5b at 30 s (6.0 vs 4.5) - fewer steps from scratch; the Dreamer
  loop's WM refresh is the place to recover it.

- **Refreshed WM after Dreamer-loop iteration 1** (trained 30 % on policy-driven SUMO episodes): on the unseen walls
  wall_same 1.70 / 6.13 / 13.55 / 108.0, wall_pilot 1.70 / 5.88 / 11.29 / 88.2 - a small loss at 60/120 s vs the C4 WM
  (the data now leans to policy-driven traffic, which is where it must be accurate for the policy); the policy dreams in
  BOTH (ensemble), so the original model's strengths are kept. (`score_refresh1/ROUND5_SCORE.md`)

### Policy
- Phase 1 (R5b dream world, 750 updates): benefit vs native in imagination went from -2.5 to ~0 (held-out
  validation +0.07 at update 400); imagined delay 37 -> 28; no cooperation gain yet (curriculum ramps it later);
  exploration (entropy) collapsed 1.43 -> 0.29.
- Phase 2 started 08:05: dreams in an ENSEMBLE of the new C4 world model + R5b (disagreement penalty active),
  initialised from phase-1 best, entropy bonus raised to 1e-3. ~21 s/update.

### MAPPO
- Fresh run since 05:20 (calibrated budget); see the learning curve in `fetched/cpuT_mappo_main/log.jsonl`.

### First SUMO head-to-head (11:00, gpuW, 24 paired 10-min episodes, seeds 4117-4120, 5/20/40 %, 2400/2700 veh/h)
| controller | total wait (veh-s) | on-road delay | entry waiting | braking | TTC<3 s | collisions |
|---|---|---|---|---|---|---|
| native | **28,766** | 23,332 | **5,434** | 511 | 475 | 0 |
| CIRCLES damping | 32,484 | 23,501 | 8,983 | 557 | 656 | 0 |
| PI | 29,842 | 21,837 | 8,006 | 488 | 478 | 0 |
| C4 policy (phase-1 best) | 30,357 | **20,138** | 10,219 | 404 | **443** | 0 |
| MAPPO (it 35) | 30,889 | 20,613 | 10,276 | **380** | 453 | 0 |

Both learned controllers smooth the road (on-road delay -13 %, braking -23 %) but hold vehicles back at the entrances
(entry waiting ~2x), so total wait is worse than native. Cause: the shared team reward ignored vehicles waiting to
enter. **Owner approved the fix (12:00)**: add the entry queue to the reward for both methods (SUMO: per-step pending
vehicles; imagination: WM entry process with a backlog), then fine-tune MAPPO from its checkpoint and continue the
policy. (Phase-2 policy was not measured: same checkpoint filename as phase 1 - will be included next time.)

## Needs your decision in the morning
- Evaluation seed block for the final comparison (SUMO seeds the evaluation may use; exposed 4107-4148 are used for training/collection).
- Whether to extend the gpuW rental (WM + policy + Dreamer loop need the GPU through Thursday).
