# Stage 1b v1 gate fit results (development fit on the p20 stratum)

Date: 2026-09-03/04. Status: **development fit; descriptive; no frozen gate
bundle yet** (the frozen gate would be trained later on the deployable
profile). Data: seeds 7001-7240 only; one row per (seed, t_d), t_d in
{150, 250, 350}; group = seed for every split. Features: the D-prefix RSU
vector at t_d; hidden demand, absolute time and all outcome/W-arm quantities
excluded; t_d only at rung L5 as a categorical.

Recording-note constraint: the generation schema stored 120 s backward
differences (per the generation amendment); the 10/30/60 s trends were not
recorded, so L2's trend block is the 120 s difference features. Constant
recording-defect features (cav_count_band, cav_count_upstream,
ramp_queue_length — see the Part B results) are dropped by a
zero-variance guard; the per-lane geometry and vehicle-level features vary.

## Ladder table (OOF, 5-fold grouped by seed, 3 fold seeds; mean of metrics;
(rh) = mean±range across the 3 repeats)

| rung | model | R² | Spearman(Δdelay) | AUC P(Δ<0) | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| L0 (threshold k per fold) | rule | (n/a — rule engages nothing OOF; AUC 0.686, Brier 0.319) | | | |
| L1 loading | ridge | 0.168 | 0.417 | 0.760 | 0.214 |
| L1 loading | histgbm | 0.120 | 0.339 | 0.702 | 0.232 |
| L2 +120 s diffs | ridge | 0.168 | 0.413 | 0.757 | 0.213 |
| L2 +120 s diffs | histgbm | 0.092 | 0.299 | 0.679 | 0.249 |
| L3 +CAV geometry | ridge | 0.168 | 0.413 | 0.757 | 0.213 |
| L3 +CAV geometry | histgbm | 0.092 | 0.299 | 0.679 | 0.249 |
| L4 +30 s flow/history | ridge | 0.165 | 0.409 | 0.750 | 0.213 |
| L4 +30 s flow/history | histgbm | 0.069 | 0.273 | 0.673 | 0.252 |
| L5 +t_d | ridge | 0.165 | 0.408 | 0.749 | 0.214 |
| L5 +t_d | histgbm | 0.072 | 0.275 | 0.668 | 0.254 |

The loading level (L1) carries essentially all of the signal; every added
rung is neutral for ridge and mildly degrading for the histogram gradient
booster (its weakest performance is at L4-L5; the 120 s diff features and
t_d categorical do not help it).

## Policy values (OOF; engage if pred − λ·P(benefit) < 0; λ ∈ {0,1,2,4}
s/veh per prevented collapse; value per seed = −Δdelay + λ·prevented when
engaged, else 0; baselines never 0.000, always-W250 0.705 (λ0) / 0.830 (λ2),
oracle 4.674 (λ0) / 4.752 (λ2))

| rung | model | value λ=0 | vs L0 (CI90) | value λ=2 | vs L0 (CI90) |
| --- | --- | ---: | ---: | ---: | ---: |
| L0 | rule | 0.000 | (baseline) | 1.507 | (baseline) |
| L1 | ridge | 1.608 | +1.61 [+1.26, +1.96] | 1.647 (1.65 in value_vs_csv) | +0.14 [−0.03, +0.30] |
| L1 | histgbm | 1.368 | +1.37 [1.03, 1.70] | 1.419 | −0.09 [−0.28, +0.10] |
| L2 | ridge | 1.582 | +1.58 [1.24, 1.92] | 1.710 | +0.17 [+0.00, +0.35] |
| L2 | histgbm | 1.178 | +1.18 [0.84, 1.52] | 1.211 | −0.30 [−0.48, −0.12] |
| L4 | ridge | 1.520 | +1.52 [1.17, 1.87] | — | +0.12 [−0.05, +0.30] |
| L4 | histgbm | 1.136 | +1.14 [0.80, 1.47] | — | −0.39 [−0.59, −0.20] |
| L5 | ridge | 1.513 | +1.51 [1.16, 1.87] | — | +0.12 [−0.05, +0.30] |
| L5 | histgbm | 1.093 | +1.09 [0.76, 1.43] | — | −0.49 [−0.69, −0.29] |

**Headline, answered explicitly with CIs. At λ = 0: every rung L1-L5 beats
L0 on OOF policy value — L1 ridge +1.61 s/veh (CI90 [1.26, 1.96]) (and the
best rung is L1 itself; the ladder is flat beyond it in policy-value terms).
At λ = 2: NO rung robustly beats L0** — L1 ridge +0.14 [−0.03, +0.30]
(includes 0), L1 histgbm −0.09 [−0.28, +0.10], and the higher rungs are
significantly negative for the booster (−0.30 to −0.49, CIs excluding 0).
Interpretation: the delay-value signal favours the learned loading gate
decisively at λ = 0; once prevention is priced (λ = 2), the loading
threshold baseline is not clearly beaten, i.e. the publishable statement is
"the gate is close to a loading threshold at λ = 2; the learned benefit sits
in the delay channel". Also note L0's OOF policy value is 0.000 at λ = 0
(the fold-internal k never engages out-of-fold on average while
always-W250 already yields +0.70): the "k chosen by training value" rule
generalizes poorly.

Oracle capture: L1 ridge 0.344 of the per-seed oracle at λ = 0 (0.347 at
λ = 2), vs always-W250's 0.151. L1's OOF value also beats always-W250
(+0.90, CI in the summary).

## Timing selection

Evaluated (Part 1 timing-matters > 20% in bins 2400/2500/2600). Choosing t_d
per seed by predicted value (L1 ridge) gives λ = 2 value 0.772 vs fixed
W250's 0.830: mean diff −0.058 s/veh, CI90 [−0.487, +0.366] — **timing
selection at the seed level is NOT supported by the fit** (no evidence it
beats always-W250; the boundary curve still shows a bin-level late-engagement
preference).

## Calibration summary

P(benefit) classifier (L1 ridge, OOF): AUC 0.760, Brier 0.214, 10-bin
reliability stored in `gate_fit_summary.json` (per-rung tables). Brier
hovering ≈ the AUC level suggests moderate calibration; no sharpening was
applied.

## Artifacts

- `runs/probe/stage1b_generation/gate_fit_summary.json` — all tables above
  including reliability tables, per-seed oracle vectors and the complete
  ladder metric arrays (SHA-256 at commit time).
- `runs/probe/stage1b_generation/gate_predictions...` (no — the raw OOF
  arrays are inside the summary's ladder entries).
- Code: `experiments/idm/fit_stage1b_v1_gate.py`.

## Claims not made

- No frozen gate bundle; no prospective fit; the fit is text on development
  seeds only and must be retrained on the deployable profile before use.
- The 10/30/60 s trend block is approximated by the recorded 120 s
  differences (recording limitation).
- L0 as implemented ("k per fold by training policy value") is one specific
  threshold-baseline variant; other k-selection rules could shift the
  λ = 2 comparison slightly, but the λ = 0 dominance of L1 is robust.
