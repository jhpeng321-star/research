# EXIT_FIX — A11 rollout road departures (TRAIN-only, no SUMO)

Owner-approved exploratory fix. Model member_1.pt sha256 41221852…f21e8; climatology TRAIN_CLIMATOLOGY.json (TRAIN). Diag: `_out/exit_diag.py` → `_out/exit_diag.txt|json`.

## Root cause
The rollout world `s_m` is the **model `s_down`** coordinate (`features.geometry.s_down_of` = physical s − lane offset), but the exit test in `rollout2/step.py` compared it to `params.llen` = the **physical** lane end. For main lanes the offset is ~2645 m, so `s_down > 3199` never fired inside a 120 s window → no CAV ever departed. Recorded `cav_plan_ch` marks a slot dead once the vehicle is absent from the physics row, i.e. when its physical s passes `llen`; the correct model-space exit is `s_down > llen[lane] − off[lane]` (lane0/1 ≈ 553.3 m, lane2 = 75.44 m; verified on 16 exiting t0-CAVs, last physical s 3185–3194 < 3199).

New-CAV gap: training’s plan uses a monotone alive-since-t0 mask (`cumprod(target_mask)`), so recorded summaries do **not** count entrants; the rollout sets `arrivals_enabled=False` and keeps t0 slots. No entrant mismatch — departures were the only structural gap.

## Fix (flag-gated, default OFF = byte-identical)
- `rollout2/state.py`: `Rollout2Config.physical_exits: bool = False`.
- `rollout2/step.py`: when on, drop `s_new > llen[lane] − off[lane]` (else the legacy `s_new > llen`).
- `closedloop/a11_predictor.py`: `batched_rollout(..., physical_exits=False)`; `A11Predictor(..., physical_exits=False)` forwards it.
- Tests: `rollout2/tests/test_exits.py` (7 synthetic: threshold derivation, drop/keep, legacy still fires at raw length, flag-off bit-equality, plan-count drop, predictor forwarding).

## Re-run (8 TRAIN chunk-4 states, per-bin `_out/exit_diag.json`)
`n_cav` w8676 (bin0…bin11): rec `[22.2…8.6]`, OLD `[25.0…21.9]` (constant), NEW `[22.4…2.2]` → exits now reproduce the recorded drop; NEW over-drops late bins.
Head dV at tau=300 s (treatment−native, +ve worse):

| state | scorer(rec) | OLD | NEW | TRUE |
|---|---|---|---|---|
| w8676 360 | 1355 | 3732 | 61 | 1123 |
| w8626 300 | −519 | 501 | 102 | 39 |
| w8629 420 | −155 | 180 | −41 | −55 |
| w8678 300 | 1308 | 10577 | 452 | 1324 |
| w8677 360 | 680 | 5186 | 211 | 2382 |
| w8680 300 | 995 | 6947 | −31 | −91 |
| w8679 540 | 1284 | 4026 | 118 | 1622 |
| w8683 540 | 270 | 1922 | 159 | −1785 |

## Conclusion
Exits remove the structural OOD inflation: OLD’s near-constant `+3700…+10577` collapses to `−41…+452`, and NEW beats OLD vs TRUE on all 8 states (scorer/recorded plan still closest on the positive-dV majority). What remains is **speed/dynamics**, not exits: recorded `mean_v` 15.8–25.5 and `min_v` ≈0–5 (stopped CAVs) vs NEW `mean_v` 22–27, `min_v` 8–21, so the rollout over-predicts motion → more CAVs cross the exit (late-bin `n_cav` under-shoot) and mean plan speed is high. Next: reconcile the rollout’s `s_down`-vs-physical coordinate in segment limits/clamps and congestion, or recalibrate the plan head on departure-aware rollout summaries.
