# S-C1PIDIAG — why the newer checkpoints lose the original C1 comparison

**Verdict:** the original-start 60 s regression is real, including teacher forcing; no tested inference switch restores it. The largest documented comparison mismatch is training coverage, with an additional measurable WM2 history penalty. This is not evidence of a universal structured-architecture decline or a universal history fix.
**Coverage attribution:** old C1 and frozen BBs trained under other controllers on **40/40 PI and 13/13 wall worlds** here; 13E/WM2 have **0/40 and 0/13 overlap in audited corpora**. C1FAST used 560 trajectory + 174 damping + 57 native windows; newer corpora use native/random/pairs, with WM2 adding low-penetration/own-policy data. Different weights, physics and corpora remain causally confounded; copied initialization ancestry is still incomplete. [Coverage evidence](c1pidiag/COVERAGE.json).

## Fixed original starts: teacher-forced attribution
Preregistered first sorted 40 PI `u_same` starts/40 worlds + 30 `wall_same` starts/13 worlds; checkpoints C1 `87ed02e0`, 13E `952ec059`, WM2 `b8797242`, BB-traj `afaaa32c`. PI has snapshot history, wall 5 s. Teacher forcing supplies CAVs only; HDVs free-run.
All errors are metres, median of start-level HDV medians. Absolute columns use each model's surviving IDs; **Δ = WM−BB-traj on common surviving HDV IDs within start**, then median across starts. Brackets: descriptive 95% whole-world bootstrap, 2,000 draws, seed 0; not multiplicity-adjusted. Thus Δ need not equal subtraction of displayed absolute medians.
| Cohort / model or switch | Absolute 60 / 120 s | Paired Δ60 [95% CI] | Paired Δ120 [95% CI] |
|---|---:|---:|---:|
| PI / old C1 | 15.41 / 55.97 | −4.84 [−9.32,−2.32] | −45.25 [−78.65,−22.78] |
| PI / BB-traj | 22.40 / 131.62 | 0 | 0 |
| PI / 13E | 44.88 / 72.61 | +24.49 [11.13,33.05] | −36.27 [−71.98,−5.86] |
| PI / WM2 | 42.73 / 45.18 | +17.85 [3.24,31.27] | −47.82 [−95.20,−27.75] |
| PI / 13E residual-net + temporal off | 39.16 / 74.48 | +18.68 [7.43,29.13] | −32.95 [−82.69,−3.55] |
| PI / WM2 residual-net + temporal off | 45.54 / 37.58 | +21.16 [7.31,30.87] | −46.19 [−109.09,−25.39] |
| Wall / old C1 | 14.51 / 134.21 | −8.47 [−20.56,−2.57] | +22.19 [−40.57,85.94] |
| Wall / BB-traj | 27.74 / 122.26 | 0 | 0 |
| Wall / 13E | 37.52 / 97.48 | +6.86 [−6.54,19.26] | +24.15 [−71.03,64.27] |
| Wall / WM2 | 27.72 / 90.60 | −3.07 [−9.94,14.72] | +14.95 [−76.81,169.55] |
| Wall / 13E MOBIL pipeline | 25.18 / 129.66 | +0.09 [−4.90,8.23] | +18.36 [−58.70,97.75] |
| Wall / WM2 learned-yield off | 20.98 / 90.62 | −3.07 [−10.02,2.10] | +0.07 [−75.71,55.05] |
**Components:** population defaults, all-corrections-off, courtesy-off and ramp-adaptation-off do not restore the original PI advantage. Residual-net-off is not a robust 60 s PI improvement; for 13E it costs **+1.30 [0.18,6.86] m at 120 s** versus its baseline. MOBIL is a coupled lane/intent-pipeline intervention, not a pure lane-only effect; WM2 wall error increases **+4.90 [0.33,10.56] m at 60 s** under it. No general component rollback is recommended.

## Genuine-history test: separate fixed clock t0′ = t0 + 60 s
Original recordings cannot supply genuine pre-original-t0 60 s history. At the same 70 episodes' prospectively shifted starts, rebuild [t0′−60,t0′] from same-arm recorded observations, including causal acceleration/junction carry; never give original-t0 forecasts their future. The table uses full60/inferred/original components. These later starts must not replace the original panel.
| Shifted cohort / model, TF | Absolute 60 / 120 s | Paired Δ60 [95% CI] | Paired Δ120 [95% CI] |
|---|---:|---:|---:|
| PI / old C1 | 49.45 / 43.10 | −44.76 [−60.94,−18.40] | −150.52 [−271.96,−50.32] |
| PI / BB-traj | 85.90 / 242.60 | 0 | 0 |
| PI / 13E | 60.52 / 76.20 | −28.59 [−55.67,−15.11] | −142.15 [−231.13,7.25] |
| PI / WM2 | 43.68 / 56.61 | −45.05 [−52.73,−25.25] | −144.74 [−250.11,−36.22] |
| Wall / old C1 | 53.72 / unavailable | −10.65 [−52.15,59.59] | unavailable |
| Wall / BB-traj | 67.45 / unavailable | 0 | unavailable |
| Wall / 13E | 81.22 / unavailable | +14.71 [−47.76,141.15] | unavailable |
| Wall / WM2 | 111.38 / unavailable | +16.45 [−56.09,199.28] | unavailable |
**History attribution within identical shifted starts:** WM2 PI full60−snapshot **−13.47 [−26.92,−1.49] m** at 60 s; 13E PI **+6.55 [−1.24,17.06]**. Wall effects are mixed: 13E **−17.79 [−29.65,−0.03]**, WM2 **+3.57 [−18.10,30.38]**. Even shifted PI snapshots change relative model ranking, so the later-start improvement cannot all be credited to history. Population ablation also changes the encoder embedding; C1's population toggle is inert, not physical-style identification.

## Feedback, runtime and publication boundary
Original-start common-plant conditional PI feedback gives C1/BB/13E/WM2 **26.93/30.11/56.89/59.32 m at 60 s**; paired newer−BB **+15.55 [7.67,27.36] / +26.63 [10.48,34.58]**. Best component configurations were also feedback-scored; their complete 60/120 s results remain in the exports.
Late-start full-history **reset-controller** PI feedback is adverse: 13E **180.40/1044.92 m**, WM2 **205.23/579.25 m** at 60/120 s; paired Δ120 **+631.68 [86.92,1104.60] / +201.00 [141.73,623.94]**, only 10/12 surviving starts. PI's prior commanded-speed state is unbound: these are reset stress tests, **not validated continuation of the recorded PI executor**, so do not assign all this error to HDV dynamics or claim deployment rescue. [Supplement](c1pidiag/aux_feedback/SUMMARY.json).
Archived-source replay on the same CPU independently retains the PI 60 s regression: C1/BB/13E **15.39/22.45/44.64 m**, paired 13E−BB **+23.43 [9.29,32.74]**. Archived-result reproduction has median absolute endpoint discrepancy 0.000123 m but worst 343.44 m; source/protocol and numerical sensitivity are disclosed, not called exact parity. Current snapshot adapters also differ for short-history models.
**Fair protocol/fix:** always supply genuine observed 60 s to every model API, bind causal controller/executor state and one CAV plant, and retain common-ID denominators. History helps WM2 PI but is not a universal fix. Existing legacy starts cannot be retrospectively given their future; fully initialized autonomous continuation belongs to the distinct S-C1CLOSED panel.
**Paper B recommendation:** keep old C1 for the matched-corpus **historical 60 s teacher-forced** headline; retain adverse 13E refresh and describe newer full-history PI results as an honest conditional extension. Do not claim universal newer-model loss, universal newer superiority, a proven coverage-only cause, or a fair full-feedback win. No retraining is implied.
Scope: original 120 s denominators vary by model; shifted TF has 37 PI / 29 wall defined 60 s starts and 21–24 PI 120 s starts. Shifted wall has no 120 s labels; nominal endpoint tolerance ≤0.5 s. Presence/exits remain conditional on recordings. Entries are OFF; WM2 entrant mix is dormant and entries+mix unavailable without observed source queue/demand, not silently evaluated as production WM2+mix.
Evidence: [full matrix/CIs](c1pidiag/SUMMARY.json), [per-state](c1pidiag/PER_STATE.csv), [raw per-ID errors](c1pidiag/RAW_UNITS.jsonl.gz), [implementation caveats](../../IMPLEMENTATION_NOTES.md), [source hashes](c1pidiag/SOURCE_RECEIPT.json), [run audit](c1pidiag/FINAL_RUN_AUDIT.json). **4,730 canonical cells completed**, no unresolved scoring errors; failed preflights/195 adapter attempts preserved. CPU nice19, ≤24 one-thread workers; GPU rejected at 10,701 MiB free; zero SUMO/training/GPU inference; 11 registered dispatch identities inactive.
