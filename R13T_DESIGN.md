# R13 temporal-emission variants: F / M / S / T (prepared, NOT registered, NOT launched)

Branch `c4/v10r13t` from `079e39ff`. BRAKE_DIAG: the learned temporal emission (`temporal` row 0) adds zero-mean, heavy-tailed noise in closed loop. S-FACTORIAL: zeroing it fixes the hard brakes but loses a needed mean (delay and blow-ups); its state-conditional mean keeps the delays and fixes the hard brakes. All work was on local CPU.

## 1. Switches (all default = round 12, omitted from the flat config at default; F/M/S mutually exclusive)

- **F** `model.temporal_accel_off`: emission = exact 0 (the BRAKE_DIAG `zerotemp` tensor).
- **M** `model.temporal_accel_lane_bias` + `temporal_lane_bias_init`: emission = learned per-lane constant (pre-gate).
- **S** `model.temporal_accel_state_table` + `temporal_state_table_init`: the S-FACTORIAL T-mean arm.
  - Emission = deterministic lookup of the current state + k in five mean tables.
  - Features (lane, v, bumper gap, closing, s, k), bins and backoff (≥ 30 rows) are taken exactly from `factworker.py`/`fit_table.py`; a test checks them against an inlined numpy copy.
  - The tables are parameters, trained at the `temporal` lr scale (×5, near-fixed); the counts are buffers.
- **T** `model.temporal_accel_reinit` (zeroes row 0 after loading, live gradient) + `loss.accel_tail` (§2).

## 2. Tail-matching loss (`losses/accel_tail.py`)

Rows: all rollout rows of present lane-1 HDVs, not only onsets.
- **Cells.** Brake-persistence convention (gap × closing × leader-decel), now shared via `brake_persist.leader_cells`; a test confirms the old ids.
- **Score.** Predicted per-cell quantiles q̂(u) at u = 0.01…0.99 of a = Δv/dt and |jerk|, scored with pinball against the recorded TRAIN cell quantiles (constant at a perfect match subtracted).
  - dE ρ_u/dq = F_rec(q) − u: minimal at the recorded quantile, bounded gradient, CRPS decomposition.
  - The 4 %-vs-0.18 % excess lies above the 1 % level.
  - Support rule: n·min(u, 1−u) ≥ 1.
- **Table.** `tools/r13t_tail_table.py`; the loss refuses non-TRAIN tables. Only val-role episodes of the r9 TRAIN pool are local (192 episodes, 10.5 M rows), so the table was built provisionally (uncommitted) and used only for measurements.

## 3. Rules and numbers (local stand-in init = round-9 final + round 12's init transforms)

- **T weights:** w = (G_brake/2)/G_term, gradient-norm parity with brake_persist(+quantile); 4 batches × 20 s. Result: **accel 10.33, jerk 38.74** (per batch 5.6–17.7 / 20–170). Re-measure at the round-12 final.
- **absolute_weight:** 9.067 × G_contrast/G_abs at the zero-init head (R11 TRAIN cache) = **4.525** (seeds 4.13–4.78).
- **M init:** gate-weighted mean emission per lane over 60 s free rollouts (54k rows) = **[2.403, 0.7526, 0.0095]**. Pre-gate mean [1.72, −0.79, −1.86], SD 8–14; the k-profile is lost.
- **S table** (`docs/r13s_state_table.npz/.json`): fitted on the same rollouts. R² in-sample 0.31, held-out-by-window 0.22 (S-FACTORIAL 0.27); 350 level-0 cells. Re-fit on TRAIN-split windows at launch.

## 4. Configs and gates

| config | content | init |
|---|---|---|
| round13f | r12 + repair A + F | round 12's (round-9 final + its transforms) → round 12 = paired control |
| round13m | 13F with M | same |
| round13s | 13F with S | same |
| round13t | r12 + T | round-12 final |
| round13at | 13A + T | round-12 final |

- Gates `scripts/round13{f,m,s}_gate.sh`: the round-12 gate up to names, prior default off as round 12 was launched. The verdict `wms.train.round13f_gate` picks its mode from the config: new tensors admitted, init values/sha checked, mode checked in every reload. No 13T/13AT gate yet.
- **13A defect fixed here:** 13A re-applied round 12's init transforms to round 12's own final weights (zeroing the driver rows and the residual read-out). Now `checkpoint`/`false`.

## 5. Tests

- New tests, all passing:
  - test_r13f_temporal_off 8, test_r13m_lane_bias 6, test_r13s_state_table 7, test_r13t_tail 13;
  - launcher tests 23 each for 13f, 13m and 13s;
  - the R13 + round-12 related set: 250 passed.
- Bit identity (`tools/r13_bit_identity.py`, separate processes, old vs new src):
  - round 12: 345/345 identical;
  - 13F vs 079e39ff + `diag_zero_temporal`: 336/336, including gradients;
  - M with b = 0 and S with a zero table equal 13F on all 336 shared tensors.
- Full suite: 4236 passed, 38 failed, 186 skipped, 85 errors.
  - Almost all failures/errors are files missing from the fresh worktree: untracked `data/r9_train/wm_s_stage300.pt`, `output/*`.
  - Also failing: the ledger tests, the round11_gate import-layer test (fails on base too), and prepush not-a-repo.
  - The failed tests also fail or error on the base src; none of them involve changed files.

## 6. Launch condition

Launch only after the round-12 results, with registration + ack first. Run one fix against the current best: F, M and S are round 12's paired arms; T applies if round 12 becomes best; AT only by owner/coordinator choice.

Recommended order: **S** (the factorial winner), then M. F is risky (factorial: blow-ups 12.5 → 40 %).

On gpuR beforehand:
- re-fit the S table and the M init on TRAIN-split windows;
- for T: build the TRAIN tail table and re-measure the weights at the round-12 final.

## 7. Closed-loop statistics for the temporal path?

The round-12 terms do not constrain the per-row acceleration distribution: position integrates a zero-mean pulse out, and the braking terms are onset-only. `accel_tail` (T) is such a closed-loop statistic. S removes the noise structurally, by having no history input. Nothing more was built.

## 8. Risks

- **S/M:** a frozen mean also carries the TTC/overlap excess (factorial: mean vs zero gives TTC +5.5, overlap +3.8). If training improves the core, the needed mean should shrink, but the tables barely move (×5 lr). The S table is crude (R² ≈ 0.2–0.3) and fitted on few rows.
- **T:** tail vs motion trade-off; sparse, noisy gradient (weights vary ×3 per batch); state-dependent cells.
- **F:** loss of the needed mean (delay, blow-ups).
