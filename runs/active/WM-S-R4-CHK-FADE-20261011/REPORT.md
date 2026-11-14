# WM-S-R4-CHK-FADE-20261011 — inference-only fade schedule check

**Status: registered before this check's unseen-set scoring.** This is CPU inference only: no training, optimizer steps, SUMO, GPU, or checkpoint modification. Max 6 worker processes, one Torch/BLAS thread per worker; all outputs stay in this directory and all launched processes use `TEMP=TMP=E:/agent-work/tmp` and `PYTHONPATH=src`.

## Question and nearest evidence

Question: can multiplying the learned R4/C1 gate by an inference-time factor that fades to zero after a selected horizon retain short-horizon accuracy while reducing long-horizon error? Closest evidence: WM-S-LANECAL-20261010 found the C1 gate cap at g(60) lowered wall 120 s paired median error but raised the 60 s across-state median 8.71→11.08 m; R4-DIAG-ORACLE found residual-off physics has 15.68/53.12 m state-median |ds| at 60/120 s vs R4 9.96/110.69 and C1 8.71/99.84 on wall_same, while the residual reduces short horizon but over-slows long rollouts. This check isolates *fade timing and scope* without changing the trained weights. No evidence about longer-horizon training is directly measured here.

## Locked protocol

Models: R4 checkpoint `runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt`; C1 `data_cache/ckpt_c1fast/wm_s.pt`. Scoring uses clean `src/wms` scorer, CPU, entries OFF, recorded CAV teacher, 0.5 s rollout steps, 10/30/60/120 s horizons. Paired identities are `(world,key,source_arm,t0_s)`; bootstrap resamples whole worlds, 2,000 replicates, seed 0.

For each model, screen ONLY seen-controls states: the same fixed 40 of 217 states as LANECAL, selected by `random.Random(0).sample(sorted(idents), 40)` where an ident is `(world, round(t0_s,3), key, source_arm)`. Each model tests unmodified baseline and the full grid: `k_start ∈ {60,120,160}` steps; `k_len ∈ {0,40}` steps; scope in `{accel residual only, lane delta only, both}`. The factor is 1 through `k_start`; thereafter a zero-length schedule sets it to 0, while a 40-step schedule decreases linearly to 0 over 40 physics steps. Multiply the relevant residual component(s) by this factor. Also screen R4 lane-delta-only `g(k)` capped at `g(20)` as the R4-DIAG-LANE reference. Select one schedule/model by lowest seen-controls across-state median state-median HDV |ds| at 120 s, eligible only if its 60 s median is no more than baseline +0.5 m and its blow-up count is no higher than baseline. Ties resolve by earlier start, then shorter fade, then scope order `accel`, `lane`, `both`. If no schedule qualifies, select baseline and report no eligible fade.

After saving the selections, score only each selected schedule plus that model's unmodified baseline once on `wall_same` (67), `wall_pilot` (53), and `wall_same_real` from `data_cache/wallc1_states_h60_real` (the explicitly specified real-history set; report its actual roster size). No wall/real-history result may influence selection. Report median |ds| at 10/30/60/120 s, paired medians and world-bootstrap 95% CIs against unmodified same-model baseline and C1, blow-ups, wrong-lane share, TRUE lane changes per HDV per 10 s (the LANEDWELL `lane_lib.py` definition: matched present HDVs, forward-fill lane -1 junction values, count transitions), and signed ds and speed error among 120 s survivors. Preserve denominators, missing/error states, and numeric precision in `results.json`.

Baseline integrity gate: before screening, the R4 and C1 cached full `wall_same` summaries must reproduce C1 60 s 8.71 m and 120 s 99.84 m and R4 60 s 9.96 m and 120 s 110.69 m (within 0.01 m). If not, stop before screening/selection.

## Progress and results

Seen-controls selection is complete under the locked rule and has been persisted below. No results from this check's unseen-wall or real-history scoring have been read. The preregistration, screening rule, checkpoints, sources, scopes, paired statistic, and stopping gate above are fixed.

Execution receipt: first seen-controls attempt stopped on a script `AttributeError` before a state rollout (`torch.MethodType`); no seen result or unseen result was produced. Corrected to `types.MethodType`; retry remains within the registered six-worker CPU budget. Baseline cache gate passed (C1 wall_same 8.709604/99.840323 m; R4 9.962799/110.686817 m at 60/120 s).

Second screen receipt: R4 completed the grid but selector caught wrapper errors in both-scope variants (`gate(k)` receives a vector step index; the first wrapper treated it as a scalar). Those variants were invalid and selection stopped before C1 or any confirmation; all screen rollouts remain preserved in the result receipt. The wrapper is corrected to vectorize the fade gate, and combined scope now scales through the gate exactly once. Re-run uses the fixed registered rule; failed outputs are retained in `results.json` progress.

<!-- Results appended after selection and one-time confirmation. -->

## Restart execution note

The owner-cancelled attempt's oversized `results.json` was moved aside as `results_bloated_cancelled_2d35884b.json`; this restart does not open or load it and re-screens from the registered sources. The prior implementation retained full rollout frames for every state/variant in parent results. The restart worker contract now returns only per-state horizon summaries, wrong-lane and lane-change counts, survivor signed-error sums, and status; full trajectories are neither returned to nor serialized by the parent. Results persist aggregate summaries only, with per-variant memory checkpoints and a 3 GiB parent / 10 GiB process-tree stop threshold. Protocol text above remains locked.

## Seen-controls selection (persisted before confirmation)

The cached baseline integrity gate passed: C1 wall_same 8.709604/99.840322 m and R4 9.962799/110.686817 m at 60/120 s (tolerance 0.01 m). The fixed 40-state seen-controls screen selected **R4 `both_s120_l40`** (120 s state-median |ds| 165.772709 m; 60 s 17.416767 m; 6/39 blow-ups; eligible grid schedules 2) and **C1 `lane_s160_l40`** (120 s 179.775620 m; 60 s 25.423981 m; 13/39 blow-ups; eligible grid schedules 14). Baseline 120 s medians were R4 155.603474 m and C1 207.051811 m. These selections and full summaries were written to `results.json` and `selection.json`; no unseen-set outcomes have been used.

## One-time confirmation results

All confirmation outcomes below are post-selection. Exact per-horizon state medians, paired differences/CIs, denominators, wrong-lane counts/shares, lane-change counts/rates, survivor sums, and errors are in `results.json`. The real-history roster contains 120 states (34 worlds).

| Set (states) | Model / selected | Median |ds| at 10/30/60/120 s (m) | Paired 120 s vs same-model base (m, world-bootstrap 95% CI) | Paired 120 s vs C1 (m, 95% CI) | Blow-ups | Wrong-lane share at 120 s | TRUE lane changes / HDV / 10 s, 0–60 / 60–120 | 120 s survivor signed ds / speed error |
|---|---|---|---|---|---|---|---|---|
| wall_same (67) | R4 `both_s120_l40` | 2.173431 / 4.875946 / 9.962799 / 56.488068 | -10.696258 [-35.387867, 7.832938] | -23.808329 [-61.471643, 3.030325] | 0/66 | 0.247086 (212/858) | 0.259661 / 0.147348 | +165.916654 m / +2.724656 m/s |
| wall_same (67) | C1 `lane_s160_l40` | 2.037518 / 5.071869 / 8.709604 / 125.863274 | +13.874726 [3.627014, 30.509677] | — (C1 reference) | 0/66 | 0.449883 (386/858) | 0.158882 / 1.080332 | +29.398248 m / -6.234959 m/s |
| wall_pilot (53) | R4 `both_s120_l40` | 2.179359 / 4.759136 / 9.406921 / 56.488068 | -17.618256 [-35.089813, -0.066589] | -30.105782 [-63.903949, -5.269412] | 0/52 | 0.265248 (187/705) | 0.270002 / 0.156290 | +76.897385 m / +0.875249 m/s |
| wall_pilot (53) | C1 `lane_s160_l40` | 2.060821 / 5.061310 / 8.342747 / 110.713661 | +8.720276 [-26.143367, 22.695846] | — (C1 reference) | 0/52 | 0.422695 (298/705) | 0.158237 / 1.142419 | +0.442772 m / -6.740215 m/s |
| real history (120) | R4 `both_s120_l40` | 2.034440 / 5.619873 / 13.811486 / 40.158730 | -12.131847 [-28.136097, -2.017822] | -15.261086 [-36.413788, -5.437458] | 2/114 | 0.153543 (312/2032) | 0.440649 / 0.122142 | +40.726122 m / +1.636838 m/s |
| real history (120) | C1 `lane_s160_l40` | 2.037536 / 5.615501 / 13.043577 / 95.103566 | +3.926926 [-1.571655, 12.576964] | — (C1 reference) | 0/114 | 0.440453 (895/2032) | 0.215844 / 1.655618 | +18.152737 m / -3.236921 m/s |

Restart memory receipt: the live sampled peak among C1 screen variants was 2,167.8 MiB parent and 3,450.2 MiB parent plus workers; `results.json` is 176,523 bytes (<5 MB). The first R4 screen pass preceded live worker-tree sampling, so its stored memory rows reflect post-pool RSS rather than an in-pool peak; the C1 pass sampled while worker processes were live. No per-variant upward growth or configured cap was observed. The aggregate outputs contain no full per-HDV trajectories.

Lane-rate denominator note: the compact worker counted lane-change steps only where both matched adjacent HDV lane labels were available after forward-fill; the registered LANEDWELL denominator counts all adjacent matched HDV steps, including any remaining lane-unknown steps. Thus reported lane rates use the available-label denominator and are not strictly identical to the registered all-matched-step denominator. Per-state counts and denominators are preserved in `results.json`; no unseen rollout was repeated to reconstruct omitted time-series labels.

Scope-bounded close-out: requested artifacts and verification receipts are complete in this run directory. The project-wide experiment ledger, open-questions file, route, and generated summary were not changed because this task explicitly restricted writes to this run directory and `E:/agent-work/tmp`; owner-directed project-wide close-out remains pending.

Interpretation: inference-time fade improved R4 120 s paired error on wall_pilot and real history (both CIs below zero), while wall_same was inconclusive; C1's selected fade raised 120 s error on wall_same and was inconclusive on the other sets. This is evidence for a model-dependent inference-time correction, not evidence that longer-horizon training (option B) works. Longer-horizon training remains unmeasured; the R4 result gives a specific comparison worth testing against training if separately registered.
